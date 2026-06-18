"""
Screen generated SMILES through the existing EGFR scoring pipeline.

Steps (no docking — that comes later):
    1. Load pretrained generator checkpoint
    2. Generate N molecules
    3. Report validity / uniqueness / novelty
    4. For each valid unique novel molecule:
        a. Backbone activity prediction (Model 1 — general EGFR)
        b. Covalent warhead flag
        c. ADMET filter
    5. Report and save ranked results

Run:
    PYTHONPATH=. .venv/Scripts/python.exe scripts/screen_generated.py

Prerequisites:
    scripts/pretrain_generator.py     (models/generator/pretrained_gru.pt)
    scripts/train_models.py           (models/qsar/general/)

Output:
    models/generator/screen_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from rdkit import Chem

from src.admet.filters import evaluate_admet
from src.features.covalent import covalent_confidence, detect_warheads
from src.generation.sampler import evaluate_metrics, load_checkpoint, sample_smiles
from src.scoring.applicability_domain import ApplicabilityDomain
from src.utils.logging import get_logger

logger = get_logger(__name__)

_GEN_DIR = PROJECT_ROOT / "models" / "generator"
_MODEL_DIR = PROJECT_ROOT / "models" / "qsar" / "general"
_EGFR_CSV = PROJECT_ROOT / "data" / "interim" / "egfr_cleaned.csv"
_ERBB2_CSV = PROJECT_ROOT / "data" / "interim" / "erbb2_cleaned.csv"


def load_train_smiles() -> set[str]:
    """Load canonical training SMILES for novelty calculation."""
    frames = []
    for csv in (_EGFR_CSV, _ERBB2_CSV):
        if csv.exists():
            frames.append(pd.read_csv(csv, usecols=["canonical_smiles"]))
    if not frames:
        return set()
    raw = pd.concat(frames)["canonical_smiles"].dropna().tolist()
    result: set[str] = set()
    for smi in raw:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            result.add(Chem.MolToSmiles(mol))
    return result


def predict_backbone(smiles_list: list[str]) -> list[float | None]:
    """Run backbone EGFR model (Model 1) predictions. Returns None on failure."""
    try:
        import numpy as np

        from src.features.descriptors import DESCRIPTOR_NAMES, compute_descriptors
        from src.features.fingerprints import morgan_fingerprint
        from src.models.qsar import QSARTrainer
        from src.utils.config import load_model_config

        cfg = load_model_config()
        trainer = QSARTrainer.load(_MODEL_DIR, cfg)
        preds: list[float | None] = []
        for smi in smiles_list:
            try:
                fp = morgan_fingerprint(smi, radius=2, n_bits=2048, use_chirality=True)
                if fp is None:
                    preds.append(None)
                    continue
                desc = compute_descriptors(smi)
                if desc is None:
                    preds.append(None)
                    continue
                desc_vec = np.array(
                    [desc[k] for k in DESCRIPTOR_NAMES], dtype=np.float32
                )
                feat = np.concatenate([fp.astype(np.float32), desc_vec])
                p = trainer.predict([feat])
                preds.append(round(float(p[0]), 3))
            except Exception:
                preds.append(None)
        return preds
    except Exception as exc:
        logger.warning(f"Backbone prediction failed: {exc}")
        return [None] * len(smiles_list)


def screen(
    smiles_list: list[str],
    train_smiles: set[str],
    ad: ApplicabilityDomain | None = None,
) -> list[dict]:
    """
    Run covalent + ADMET + backbone scoring on a list of SMILES.

    Returns a list of result dicts sorted by pred_pic50 descending.
    If ad is supplied, each row also contains domain / max_tanimoto / confidence_factor.
    """
    preds = predict_backbone(smiles_list)
    rows = []
    for smi, pred in zip(smiles_list, preds):
        warheads = detect_warheads(smi)
        confidence = covalent_confidence(smi)
        admet = evaluate_admet(smi)
        row: dict = {
            "smiles": smi,
            "pred_pic50": pred,
            "warheads": warheads,
            "docking_confidence": confidence,
            "admet_status": admet["admet_status"],
            "admet_qed": admet["qed"],
            "admet_sa": admet["sa_score"],
            "admet_flag_reasons": admet["flag_reasons"],
            "novel": smi not in train_smiles,
        }
        if ad is not None:
            ad_result = ad.predict(smi)
            row["domain"] = ad_result["domain"]
            row["max_tanimoto"] = ad_result["max_tanimoto"]
            row["confidence_factor"] = ad_result["confidence_factor"]
        rows.append(row)
    rows.sort(key=lambda r: r["pred_pic50"] or 0.0, reverse=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen generated SMILES")
    parser.add_argument("--n_generate", type=int, default=1000)
    # 0.8 is the validity/diversity sweet spot from the temperature sweep
    # (92.3% validity, scaffold diversity 0.50); see temperature_sweep.json.
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_len", type=int, default=120)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--top_n", type=int, default=20, help="Number of top hits to print"
    )
    args = parser.parse_args()

    logger.info("=== screen_generated: generate + score pipeline ===")

    # Prefer the EGFR fine-tuned checkpoint; fall back to legacy single-corpus one.
    ckpt_path = _GEN_DIR / "egfr_finetuned_gru.pt"
    if not ckpt_path.exists():
        ckpt_path = _GEN_DIR / "pretrained_gru.pt"
    tok_path = _GEN_DIR / "tokenizer.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No generator checkpoint found in {_GEN_DIR}. "
            "Run scripts/pretrain_generator.py then scripts/finetune_generator.py first."
        )

    # ── Generate ──────────────────────────────────────────────────────────────
    model, tokenizer = load_checkpoint(ckpt_path, tok_path, device_str=args.device)
    logger.info(
        f"Generating {args.n_generate} SMILES (temperature={args.temperature})..."
    )
    generated = sample_smiles(
        model,
        tokenizer,
        n=args.n_generate,
        max_len=args.max_len,
        temperature=args.temperature,
        device_str=args.device,
    )

    train_smiles = load_train_smiles()
    metrics = evaluate_metrics(generated, train_smiles=train_smiles)

    # ── Build applicability domain ──────────────────────────────────────────
    logger.info("Building applicability domain from training SMILES ...")
    ad = ApplicabilityDomain.from_config()
    ad.fit(list(train_smiles))

    print("\n=== Generation metrics ===")
    print(
        f"  Validity   : {metrics['validity']:.1%}  ({metrics['n_valid']}/{metrics['n_generated']})"
    )
    print(
        f"  Uniqueness : {metrics['uniqueness']:.1%}  ({metrics['n_unique']} unique valid)"
    )
    print(
        f"  Novelty    : {metrics['novelty']:.1%}  ({metrics['n_novel']} not in training)"
    )

    # ── Get valid unique novel SMILES for screening ────────────────────────────
    from rdkit import Chem as _Chem

    seen: set[str] = set()
    candidates: list[str] = []
    for smi in generated:
        mol = _Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        can = _Chem.MolToSmiles(mol)
        if can not in seen and can not in train_smiles:
            seen.add(can)
            candidates.append(can)

    logger.info(f"Screening {len(candidates)} valid unique novel molecules...")
    if not candidates:
        logger.warning(
            "No novel valid molecules to screen — try lower temperature or more epochs."
        )
        return

    # ── Score ─────────────────────────────────────────────────────────────────
    rows = screen(candidates, train_smiles, ad=ad)

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n=== Top {min(args.top_n, len(rows))} hits (valid, unique, novel) ===")
    print(
        f"{'#':>3}  {'pred_pIC50':>10}  {'ADMET':>5}  {'Cov':>3}  {'Domain':>12}  {'Sim':>5}  SMILES"
    )
    print("-" * 86)
    for i, r in enumerate(rows[: args.top_n], 1):
        pic50 = f"{r['pred_pic50']:.3f}" if r["pred_pic50"] else "  N/A"
        cov = "Y" if r["warheads"] else "N"
        domain = r.get("domain", "n/a")
        sim = (
            f"{r['max_tanimoto']:.3f}" if r.get("max_tanimoto") is not None else "  N/A"
        )
        print(
            f"{i:>3}  {pic50:>10}  {r['admet_status']:>5}  {cov:>3}  {domain:>12}  {sim:>5}  {r['smiles'][:42]}"
        )

    n_pass = sum(1 for r in rows if r["admet_status"] == "pass")
    n_cov = sum(1 for r in rows if r["warheads"])
    n_in = sum(1 for r in rows if r.get("domain") == "in_domain")
    n_bord = sum(1 for r in rows if r.get("domain") == "borderline")
    n_out = sum(1 for r in rows if r.get("domain") == "out_of_domain")
    print(f"\n  Screened: {len(rows)}  ADMET pass: {n_pass}  Covalent: {n_cov}")
    print(f"  Domain — in_domain: {n_in}  borderline: {n_bord}  out_of_domain: {n_out}")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "generation_metrics": metrics,
        "n_screened": len(rows),
        "n_admet_pass": n_pass,
        "n_covalent": n_cov,
        "applicability_domain": {
            "n_train": ad.n_train,
            "in_domain_threshold": ad.in_domain_threshold,
            "borderline_threshold": ad.borderline_threshold,
            "n_in_domain": n_in,
            "n_borderline": n_bord,
            "n_out_of_domain": n_out,
        },
        "top_hits": rows[:50],
        "note": (
            "EXPLORATORY. Generated by two-stage GRU (drug-like base + EGFR fine-tune). "
            "Backbone pIC50 is an in-sample estimate (all-EGFR model). "
            "ADMET is approximate. Applicability domain via max Tanimoto to EGFR training set. "
            "No docking performed at this stage."
        ),
    }
    out_path = _GEN_DIR / "screen_results.json"
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
