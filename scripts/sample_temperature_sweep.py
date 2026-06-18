"""
Sweep sampling temperature on the EGFR fine-tuned GRU (no retraining).

Lower temperatures sharpen the next-token distribution, trading a little
diversity for substantially higher SMILES validity. This is the standard,
free lever to hit a validity target from an existing checkpoint — it does NOT
touch the (expensive, CPU-bound) base pretrain.

Run:
    PYTHONPATH=. .venv/Scripts/python.exe scripts/sample_temperature_sweep.py
    PYTHONPATH=. .venv/Scripts/python.exe scripts/sample_temperature_sweep.py --temps 1.0 0.9 0.8 0.7 --n 1000

Output:
    models/generator/temperature_sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.sampler import evaluate_metrics, load_checkpoint, sample_smiles
from src.utils.logging import get_logger

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from pretrain_generator import load_egfr_corpus  # noqa: E402

logger = get_logger(__name__)

_SAVE_DIR = PROJECT_ROOT / "models" / "generator"
_CKPT = _SAVE_DIR / "egfr_finetuned_gru.pt"
_TOK = _SAVE_DIR / "tokenizer.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Temperature sweep on fine-tuned GRU")
    parser.add_argument("--temps", type=float, nargs="+", default=[1.0, 0.9, 0.8, 0.7])
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--max_len", type=int, default=120)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if not _CKPT.exists():
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {_CKPT}")

    model, tok = load_checkpoint(_CKPT, _TOK, device_str=args.device)
    egfr_set = set(load_egfr_corpus())

    rows = []
    print(
        f"\n{'temp':>5}  {'valid':>7}  {'uniq':>7}  {'novel':>7}  {'scaf_div':>8}  {'scaf_nov':>8}"
    )
    print("-" * 56)
    for temp in args.temps:
        logger.info(f"Sampling {args.n} at temperature={temp} ...")
        gen = sample_smiles(
            model,
            tok,
            n=args.n,
            max_len=args.max_len,
            temperature=temp,
            device_str=args.device,
        )
        m = evaluate_metrics(gen, train_smiles=egfr_set)
        rows.append({"temperature": temp, **m})
        print(
            f"{temp:>5.2f}  {m['validity']:>6.1%}  {m['uniqueness']:>6.1%}  "
            f"{m['novelty']:>6.1%}  {m['scaffold_diversity']:>8.3f}  "
            f"{m['scaffold_novelty']:>7.1%}"
        )

    out = {
        "n_per_temp": args.n,
        "checkpoint": str(_CKPT),
        "sweep": rows,
        "note": (
            "Temperature sweep on the EGFR fine-tuned GRU. Lower temperature raises "
            "validity at some cost to diversity. No retraining performed."
        ),
    }
    (_SAVE_DIR / "temperature_sweep.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    logger.info(f"Saved {_SAVE_DIR / 'temperature_sweep.json'}")


if __name__ == "__main__":
    main()
