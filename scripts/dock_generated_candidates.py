"""
Phase 23 (part 3) — Post-hoc docking of de-novo generated candidates.

Generates a fresh batch from the EGFR fine-tuned GRU, screens it through the
backbone + covalent + ADMET + applicability-domain pipeline, selects the top
~20 qualifying candidates (ADMET pass AND in-domain, non-covalent preferred),
and docks each into BOTH pockets (L858R = 2ITZ, WT = 2ITY_aligned) with Vina —
exactly the same box and engine settings as the known-library docking, so the
two sets are directly comparable in the final ranking.

Selection rule (in order):
  1. valid, unique, novel (not in EGFR/ErbB2 training set)
  2. ADMET status == "pass"
  3. AD domain == "in_domain"
  4. prefer non-covalent; fill remaining slots with covalent only if needed
  5. rank by backbone pred_pIC50, take top N (default 20)

All output is EXPLORATORY: generated SMILES are scored by an in-sample backbone,
and docking is rigid-receptor Vina across two crystal structures.

Run:
  PYTHONPATH=. .venv/Scripts/python.exe scripts/dock_generated_candidates.py
  PYTHONPATH=. .venv/Scripts/python.exe scripts/dock_generated_candidates.py --n_select 20 --n_generate 2000

Runtime: ~20-40 min (N x 2 pockets, exhaustiveness=8, all CPU cores).
Prerequisites: models/generator/egfr_finetuned_gru.pt, models/qsar/general/,
aligned 2ITY receptor (scripts/sanity_check_docking.py).

Output: models/generator/generated_docking_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))  # reuse screen helpers

# dock_library provides the shared box loader + receptor paths
from dock_library import EXHAUSTIVENESS, N_POSES, RECEPTORS, SEED, _load_box  # type: ignore
from rdkit import Chem

# screen() + load_train_smiles() live in scripts/screen_generated.py
from screen_generated import load_train_smiles, screen  # type: ignore

from src.docking.parse_results import best_affinity
from src.docking.prepare_ligands import smiles_to_pdbqt
from src.docking.vina_runner import run_vina
from src.generation.sampler import load_checkpoint, sample_smiles
from src.scoring.applicability_domain import ApplicabilityDomain
from src.utils.logging import get_logger

logger = get_logger(__name__)

_GEN_DIR = PROJECT_ROOT / "models" / "generator"
_RESULTS_OUT = _GEN_DIR / "generated_docking_results.json"


# ── Candidate selection ───────────────────────────────────────────────────────


def select_generated_candidates(
    n_select: int,
    n_generate: int,
    temperature: float,
    device: str,
) -> list[dict]:
    """
    Generate + screen, then select the top n_select qualifying candidates.

    Qualifying = valid, unique, novel, ADMET pass, in_domain.
    Non-covalent candidates are preferred; covalent ones only fill leftover slots.
    Returns screen-row dicts (smiles, pred_pic50, warheads, admet_qed, domain, ...).
    """
    ckpt = _GEN_DIR / "egfr_finetuned_gru.pt"
    tok = _GEN_DIR / "tokenizer.json"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Generator checkpoint not found: {ckpt}. Run finetune_generator.py first."
        )

    logger.info(f"Sampling {n_generate} molecules at temp={temperature} ...")
    model, tokenizer = load_checkpoint(ckpt, tok, device_str=device)
    raw = sample_smiles(
        model, tokenizer, n=n_generate, temperature=temperature, device_str=device
    )

    # valid + unique (canonical) + novel
    train_smiles = load_train_smiles()
    seen: set[str] = set()
    candidates: list[str] = []
    for smi in raw:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        can = Chem.MolToSmiles(mol)
        if can in seen or can in train_smiles:
            continue
        seen.add(can)
        candidates.append(can)
    logger.info(f"  {len(candidates)} valid/unique/novel candidates")

    ad = ApplicabilityDomain.from_config()
    ad.fit(list(train_smiles))

    rows = screen(candidates, train_smiles, ad=ad)

    qualifying = [
        r
        for r in rows
        if r.get("pred_pic50") is not None
        and r.get("admet_status") == "pass"
        and r.get("domain") == "in_domain"
    ]
    non_cov = [r for r in qualifying if not r["warheads"]]
    cov = [r for r in qualifying if r["warheads"]]
    non_cov.sort(key=lambda r: r["pred_pic50"], reverse=True)
    cov.sort(key=lambda r: r["pred_pic50"], reverse=True)

    selected = non_cov[:n_select]
    if len(selected) < n_select:
        need = n_select - len(selected)
        logger.info(
            f"  only {len(selected)} non-covalent qualifiers; "
            f"topping up with {min(need, len(cov))} covalent"
        )
        selected += cov[:need]

    logger.info(
        f"Selected {len(selected)} generated candidates "
        f"({sum(1 for r in selected if not r['warheads'])} non-covalent, "
        f"{sum(1 for r in selected if r['warheads'])} covalent)"
    )
    return selected


# ── Docking ────────────────────────────────────────────────────────────────────


def dock_generated(candidates: list[dict]) -> list[dict]:
    """Prepare ligand PDBQTs and dock each candidate into both pockets."""
    ligand_dir = PROJECT_ROOT / "data" / "docking" / "ligands" / "generated"
    dock_dir = PROJECT_ROOT / "data" / "docking" / "results" / "generated"
    ligand_dir.mkdir(parents=True, exist_ok=True)
    dock_dir.mkdir(parents=True, exist_ok=True)

    box = _load_box()
    logger.info(
        f"Box center=({box['center_x']:.3f},{box['center_y']:.3f},{box['center_z']:.3f}) "
        f"size={box['size_x']:.1f} A^3"
    )

    results: list[dict] = []
    total = len(candidates) * len(RECEPTORS)
    done = 0
    for idx, cand in enumerate(candidates, 1):
        cid = f"gen_{idx:03d}"
        smiles = cand["smiles"]
        lig = ligand_dir / f"{cid}.pdbqt"

        try:
            if not lig.exists():
                smiles_to_pdbqt(smiles, lig, name=cid, seed=SEED)
        except Exception as exc:
            logger.warning(f"Ligand prep failed for {cid} ({smiles[:50]}): {exc}")
            results.append(_row(cid, cand, None, None, "failed"))
            done += len(RECEPTORS)
            continue

        pocket_scores: dict[str, float | None] = {}
        for pocket, rec in RECEPTORS.items():
            done += 1
            logger.info(f"[{done}/{total}] Docking {cid} into {pocket} ...")
            try:
                out_pdbqt, _ = run_vina(
                    receptor=rec,
                    ligand=lig,
                    out_dir=dock_dir,
                    box=box,
                    n_poses=N_POSES,
                    exhaustiveness=EXHAUSTIVENESS,
                    seed=SEED,
                )
                pocket_scores[pocket] = best_affinity(out_pdbqt)
            except Exception as exc:
                logger.warning(f"  Vina failed: {cid}/{pocket}: {exc}")
                pocket_scores[pocket] = None

        l858r = pocket_scores.get("L858R")
        wt = pocket_scores.get("WT")
        if l858r is not None and wt is not None:
            status = "ok"
        elif l858r is not None or wt is not None:
            status = "partial"
        else:
            status = "failed"
        results.append(_row(cid, cand, l858r, wt, status))

    return results


def _row(
    cid: str, cand: dict, l858r: float | None, wt: float | None, status: str
) -> dict:
    delta = round(l858r - wt, 3) if (l858r is not None and wt is not None) else None
    return {
        "cid": cid,
        "smiles": cand["smiles"],
        "pred_pic50": cand.get("pred_pic50"),
        "warheads": cand.get("warheads", []),
        "docking_confidence": cand.get("docking_confidence", "standard"),
        "admet_status": cand.get("admet_status"),
        "admet_qed": cand.get("admet_qed"),
        "domain": cand.get("domain"),
        "max_tanimoto": cand.get("max_tanimoto"),
        "confidence_factor": cand.get("confidence_factor"),
        "l858r_score": l858r,
        "wt_score": wt,
        "selectivity_delta": delta,
        "docking_status": status,
    }


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Dock generated candidates")
    parser.add_argument("--n_select", type=int, default=20)
    parser.add_argument("--n_generate", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Phase 23 — post-hoc docking of generated candidates")
    logger.info("=" * 70)

    candidates = select_generated_candidates(
        n_select=args.n_select,
        n_generate=args.n_generate,
        temperature=args.temperature,
        device=args.device,
    )

    rows = dock_generated(candidates)

    n_ok = sum(1 for r in rows if r["docking_status"] == "ok")
    l858r_sel = sum(
        1
        for r in rows
        if r["selectivity_delta"] is not None and r["selectivity_delta"] < 0
    )
    summary = {
        "n_selected": len(rows),
        "n_ok": n_ok,
        "n_l858r_selective": l858r_sel,
        "temperature": args.temperature,
        "n_generate": args.n_generate,
        "compounds": rows,
        "note": (
            "EXPLORATORY. Generated de-novo candidates (EGFR fine-tuned GRU, temp 0.8) "
            "screened (ADMET pass + in_domain, non-covalent preferred) then docked "
            "rigid-receptor Vina (2ITZ L858R / 2ITY_aligned WT, exhaustiveness=8, seed=42). "
            "Backbone pIC50 is in-sample; docking is coarse. selectivity_delta = L858R - WT; "
            "negative = L858R-selective."
        ),
    }

    _RESULTS_OUT.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    logger.info(
        f"Done. {n_ok}/{len(rows)} docked OK, {l858r_sel} L858R-selective. "
        f"Saved: {_RESULTS_OUT}"
    )


if __name__ == "__main__":
    main()
