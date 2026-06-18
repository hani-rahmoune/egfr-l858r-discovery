"""
Pretrain the BASE character-level GRU SMILES generator on a large drug-like corpus.

The base model learns valid SMILES *grammar and scaffold diversity* from a broad
generic corpus (MOSES drug-like, see scripts/download_drug_corpus.py). The tiny
EGFR/ErbB2 set (~1347 molecules) is too small for this on its own — it only
reaches ~56% validity. A broad base lifts validity well past 90%; domain-specific
chemistry is then recovered by scripts/finetune_generator.py.

Key design: the tokenizer is fit on the UNION of the drug-like corpus AND the
EGFR/ErbB2 corpus, so the same vocabulary (and index layout) covers both the base
pretraining and the subsequent fine-tuning. The tokenizer is saved once here.

Run:
    # 1. build the corpus (once)
    PYTHONPATH=. .venv/Scripts/python.exe scripts/download_drug_corpus.py
    # 2. pretrain the base
    PYTHONPATH=. .venv/Scripts/python.exe scripts/pretrain_generator.py

Outputs (models/generator/):
    pretrained_base_gru.pt   — base checkpoint (by val loss)
    tokenizer.json           — shared vocabulary (union of base + EGFR)
    pretrain_base_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.generation.sampler import evaluate_metrics, load_checkpoint, sample_smiles
from src.generation.tokenizer import SMILESTokenizer
from src.generation.trainer import train_model
from src.utils.config import load_model_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_EGFR_CSV = PROJECT_ROOT / "data" / "interim" / "egfr_cleaned.csv"
_ERBB2_CSV = PROJECT_ROOT / "data" / "interim" / "erbb2_cleaned.csv"
_DRUG_LIKE = PROJECT_ROOT / "data" / "interim" / "drug_like_corpus.smi"
_SAVE_DIR = PROJECT_ROOT / "models" / "generator"
_BASE_CKPT = "pretrained_base_gru.pt"


def load_egfr_corpus() -> list[str]:
    """Load + canonicalise + dedup the EGFR/ErbB2 kinase corpus (~1347)."""
    from rdkit import Chem

    frames = []
    for csv in (_EGFR_CSV, _ERBB2_CSV):
        if csv.exists():
            frames.append(pd.read_csv(csv, usecols=["canonical_smiles"]))
            logger.info(f"Loaded {csv.name}")
        else:
            logger.warning(f"Not found: {csv}")
    if not frames:
        return []
    raw = pd.concat(frames, ignore_index=True)["canonical_smiles"].dropna().tolist()
    seen: set[str] = set()
    clean: list[str] = []
    for smi in raw:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        can = Chem.MolToSmiles(mol)
        if can not in seen:
            seen.add(can)
            clean.append(can)
    logger.info(f"EGFR/ErbB2 corpus: {len(clean)} unique SMILES")
    return clean


def load_drug_like_corpus(path: Path, max_corpus: int | None) -> list[str]:
    """Load the pre-built drug-like corpus (one canonical SMILES per line)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Drug-like corpus not found: {path}. "
            "Run scripts/download_drug_corpus.py first."
        )
    lines = [
        l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    logger.info(f"Drug-like corpus: {len(lines)} SMILES from {path.name}")
    if max_corpus and len(lines) > max_corpus:
        lines = lines[:max_corpus]  # file is already shuffled by download script
        logger.info(f"Capped drug-like corpus to {len(lines)} for CPU feasibility")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretrain BASE GRU on drug-like corpus"
    )
    parser.add_argument(
        "--corpus", type=Path, default=_DRUG_LIKE, help="Drug-like corpus .smi file"
    )
    parser.add_argument(
        "--max_corpus",
        type=int,
        default=None,
        help="Override config cap on corpus size",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--n_sample",
        type=int,
        default=1000,
        help="Samples generated after training for a base sanity check",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = load_model_config().get("generator", {}).get("pretrain", {})
    embed_dim = cfg.get("embed_dim", 128)
    hidden_dim = cfg.get("hidden_size", 512)
    num_layers = cfg.get("num_layers", 3)
    dropout = cfg.get("dropout", 0.1)
    batch_size = args.batch_size or cfg.get("batch_size", 256)
    lr = args.lr or cfg.get("lr", 1e-3)
    epochs = args.epochs or cfg.get("epochs", 8)
    max_len = cfg.get("max_len", 120)
    patience = cfg.get("patience", 3)
    val_frac = cfg.get("val_fraction", 0.05)
    seed = cfg.get("seed", 42)
    max_corpus = args.max_corpus or cfg.get("max_corpus", 80000)

    logger.info("=== pretrain_generator (BASE) ===")

    # ── Corpora ───────────────────────────────────────────────────────────────
    drug_like = load_drug_like_corpus(args.corpus, max_corpus)
    egfr = load_egfr_corpus()

    # ── Tokenizer: union vocab so finetune reuses the same indices ─────────────
    tokenizer = SMILESTokenizer().fit(drug_like + egfr)
    logger.info(f"Tokenizer (union vocab): vocab_size={tokenizer.vocab_size}")
    _SAVE_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(_SAVE_DIR / "tokenizer.json")

    # ── Train base on drug-like corpus only ────────────────────────────────────
    summary = train_model(
        smiles_list=drug_like,
        tokenizer=tokenizer,
        save_dir=_SAVE_DIR,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        patience=patience,
        max_len=max_len,
        val_fraction=val_frac,
        seed=seed,
        device_str=args.device,
        ckpt_name=_BASE_CKPT,
    )

    # ── Base sanity check: generation metrics (novelty vs drug-like corpus) ─────
    logger.info(
        f"Generating {args.n_sample} base samples (temperature={args.temperature})..."
    )
    model, tok = load_checkpoint(
        _SAVE_DIR / _BASE_CKPT,
        _SAVE_DIR / "tokenizer.json",
        device_str=args.device,
    )
    generated = sample_smiles(
        model,
        tok,
        n=args.n_sample,
        max_len=max_len,
        temperature=args.temperature,
        device_str=args.device,
    )
    metrics = evaluate_metrics(generated, train_smiles=set(drug_like))

    print("\n=== Base generation metrics (vs drug-like corpus) ===")
    print(
        f"  Validity          : {metrics['validity']:.1%}  ({metrics['n_valid']}/{metrics['n_generated']})"
    )
    print(
        f"  Uniqueness        : {metrics['uniqueness']:.1%}  ({metrics['n_unique']} unique valid)"
    )
    print(
        f"  Novelty           : {metrics['novelty']:.1%}  ({metrics['n_novel']} not in base corpus)"
    )
    print(
        f"  Scaffold diversity: {metrics['scaffold_diversity']:.3f}  ({metrics['n_scaffolds']} scaffolds)"
    )
    print(
        f"  Best val loss     : {summary['best_val_loss']} (epoch {summary['best_epoch']})"
    )

    result = {
        **summary,
        "stage": "base_pretrain",
        "drug_like_corpus_size": len(drug_like),
        "egfr_corpus_size": len(egfr),
        "vocab_size": tokenizer.vocab_size,
        "generation": {
            "n_sample": args.n_sample,
            "temperature": args.temperature,
            **metrics,
        },
        "note": (
            "Base GRU pretrained on MOSES drug-like corpus for SMILES grammar + diversity. "
            "Fine-tune on EGFR actives via scripts/finetune_generator.py before scoring."
        ),
    }
    out_path = _SAVE_DIR / "pretrain_base_results.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
