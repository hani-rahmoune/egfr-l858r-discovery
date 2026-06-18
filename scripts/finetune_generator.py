"""
Fine-tune the base GRU generator on the EGFR/ErbB2 actives.

Warm-starts from the drug-like base checkpoint (scripts/pretrain_generator.py) and
adapts it to EGFR kinase-inhibitor chemistry on the ~1347 actives, using the SAME
tokenizer (union vocab) saved by the base run. The base supplies valid SMILES
grammar; fine-tuning steers the distribution toward EGFR scaffolds without
collapsing back to the ~56% validity of training on the tiny corpus alone.

Run (after pretrain_generator.py):
    PYTHONPATH=. .venv/Scripts/python.exe scripts/finetune_generator.py

Outputs (models/generator/):
    egfr_finetuned_gru.pt    — fine-tuned checkpoint
    finetune_results.json    — training summary + generation metrics

After fine-tuning, generates 1000 samples and reports validity, uniqueness,
novelty, and Bemis-Murcko scaffold diversity (vs the EGFR corpus).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.sampler import evaluate_metrics, load_checkpoint, sample_smiles
from src.generation.tokenizer import SMILESTokenizer
from src.generation.trainer import finetune_model
from src.utils.config import load_model_config
from src.utils.logging import get_logger

# Reuse the EGFR corpus loader from the base script
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from pretrain_generator import load_egfr_corpus  # noqa: E402

logger = get_logger(__name__)

_SAVE_DIR = PROJECT_ROOT / "models" / "generator"
_BASE_CKPT = _SAVE_DIR / "pretrained_base_gru.pt"
_TOKENIZER = _SAVE_DIR / "tokenizer.json"
_FINETUNE_CKPT = "egfr_finetuned_gru.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune GRU on EGFR actives")
    parser.add_argument("--base_ckpt", type=Path, default=_BASE_CKPT)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n_sample", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if not args.base_ckpt.exists():
        raise FileNotFoundError(
            f"Base checkpoint not found: {args.base_ckpt}. "
            "Run scripts/pretrain_generator.py first."
        )
    if not _TOKENIZER.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {_TOKENIZER}. Run pretrain first."
        )

    # Architecture must match the base checkpoint
    pcfg = load_model_config().get("generator", {}).get("pretrain", {})
    fcfg = load_model_config().get("generator", {}).get("finetune", {})
    embed_dim = pcfg.get("embed_dim", 128)
    hidden_dim = pcfg.get("hidden_size", 512)
    num_layers = pcfg.get("num_layers", 3)
    dropout = pcfg.get("dropout", 0.1)
    max_len = pcfg.get("max_len", 120)
    batch_size = args.batch_size or fcfg.get("batch_size", 64)
    lr = args.lr or fcfg.get("lr", 5e-4)
    epochs = args.epochs or fcfg.get("epochs", 40)
    patience = fcfg.get("patience", 8)
    val_frac = fcfg.get("val_fraction", 0.10)
    seed = fcfg.get("seed", 42)

    logger.info("=== finetune_generator (EGFR) ===")

    tokenizer = SMILESTokenizer.load(_TOKENIZER)
    egfr = load_egfr_corpus()
    if not egfr:
        raise RuntimeError("EGFR corpus is empty — check data/interim/ CSVs.")
    logger.info(
        f"Fine-tuning on {len(egfr)} EGFR/ErbB2 actives, vocab={tokenizer.vocab_size}"
    )

    summary = finetune_model(
        smiles_list=egfr,
        tokenizer=tokenizer,
        save_dir=_SAVE_DIR,
        init_ckpt=args.base_ckpt,
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
        ckpt_name=_FINETUNE_CKPT,
    )

    # ── Generate + evaluate (novelty / scaffold-novelty vs EGFR corpus) ─────────
    logger.info(
        f"Generating {args.n_sample} samples (temperature={args.temperature})..."
    )
    model, tok = load_checkpoint(
        _SAVE_DIR / _FINETUNE_CKPT,
        _TOKENIZER,
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
    egfr_set = set(egfr)
    metrics = evaluate_metrics(generated, train_smiles=egfr_set)

    print("\n=== EGFR fine-tuned generation metrics ===")
    print(
        f"  Validity          : {metrics['validity']:.1%}  ({metrics['n_valid']}/{metrics['n_generated']})"
    )
    print(
        f"  Uniqueness        : {metrics['uniqueness']:.1%}  ({metrics['n_unique']} unique valid)"
    )
    print(
        f"  Novelty           : {metrics['novelty']:.1%}  ({metrics['n_novel']} not in EGFR corpus)"
    )
    print(
        f"  Scaffold diversity: {metrics['scaffold_diversity']:.3f}  ({metrics['n_scaffolds']} distinct scaffolds)"
    )
    print(
        f"  Scaffold novelty  : {metrics['scaffold_novelty']:.1%}  ({metrics['n_novel_scaffolds']} scaffolds not in EGFR corpus)"
    )
    print(
        f"  Best val loss     : {summary['best_val_loss']} (epoch {summary['best_epoch']})"
    )

    result = {
        **summary,
        "stage": "egfr_finetune",
        "base_ckpt": str(args.base_ckpt),
        "egfr_corpus_size": len(egfr),
        "vocab_size": tokenizer.vocab_size,
        "generation": {
            "n_sample": args.n_sample,
            "temperature": args.temperature,
            **metrics,
        },
        "note": (
            "GRU fine-tuned on EGFR/ErbB2 actives from a MOSES drug-like base. "
            "Validity/diversity reflect grammar learned from the base; novelty and "
            "scaffold_novelty are measured against the EGFR corpus. EXPLORATORY: "
            "downstream activity scoring uses the in-sample backbone."
        ),
    }
    out_path = _SAVE_DIR / "finetune_results.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
