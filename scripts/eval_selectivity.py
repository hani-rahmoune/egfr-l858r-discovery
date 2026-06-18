"""
Model 4 — Derived selectivity evaluation (EXPLORATORY).

** All output from this script is labeled exploratory. **
** This is NOT a selectivity model. There are only 9 paired molecules. **

Sign convention (CLAUDE.md):
    selectivity_delta = pIC50_mutant - pIC50_wt
    Positive = mutant-selective (desired)

Mutant predictor : general EGFR backbone (Model 1), consistent with the
                   Model 3 LOOCV verdict (L858R calibration did not beat
                   the backbone; see models/qsar/l858r/loocv_results.json).
WT predictor     : WT-proxy model (Model 2).

CAVEATS (stated explicitly per design requirements):
  1. All 9 paired molecules appear in the training data for BOTH models.
     There is no holdout for the activity predictions — only the selectivity
     label is truly out-of-sample in the LOO loop.
  2. The backbone and WT-proxy share ~80% of their training data.  The
     derived delta (backbone_pred - wt_proxy_pred) has a large shared-noise
     component and may be near-zero even for selective compounds.
  3. Spearman r on n=9 is extremely unstable.  One rank swap changes r by
     approximately 0.17.  No result from this analysis is statistically
     significant without larger paired datasets.

Run:
    PYTHONPATH=. .venv/Scripts/python.exe scripts/eval_selectivity.py

Output:
    models/qsar/selectivity/selectivity_results.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.models.qsar import QSARTrainer
from src.models.selectivity import (
    EXPLORATORY_LABEL,
    evaluate_selectivity,
)
from src.utils.config import get_project_root, load_model_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

ROOT = get_project_root()
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models" / "qsar"
OUT_DIR = ROOT / "models" / "qsar" / "selectivity"


def _print_per_pair(results: dict) -> None:
    pairs = results["per_pair"]
    hdr = (
        f"  {'#':>2}  {'true_delta':>11}  {'derived_delta':>13}"
        f"  {'loomean_pred':>12}  {'pred_L858R':>10}  {'pred_WT':>9}"
    )
    sep = "  " + "-" * (len(hdr) - 2)
    print(sep)
    print(hdr)
    print(sep)
    for p in pairs:
        print(
            f"  {p['orig_idx']:>2}  {p['true_delta']:>+11.3f}"
            f"  {p['derived_delta']:>+13.3f}"
            f"  {p['loomean_pred']:>+12.3f}"
            f"  {p['pred_l858r']:>10.3f}"
            f"  {p['pred_wt']:>9.3f}"
        )
    print(sep)


def _print_results(results: dict) -> None:
    print()
    print(f"  {EXPLORATORY_LABEL}")
    print(
        f"  n_pairs = {results['n_pairs']}  (sign: delta = mutant_pIC50 - WT_pIC50, positive = mutant-selective)"
    )
    print()
    print("  CAVEATS:")
    print(
        "    - All 9 pairs are in-sample for both activity models (no activity holdout)"
    )
    print("    - Backbone (Model 1) and WT-proxy (Model 2) share ~80% training data")
    print(f"    - {results['stability_note']}")
    print()
    print("  Per-pair predictions:")
    _print_per_pair(results)
    print()

    hdr2 = f"  {'Method':<22}  {'Spearman r':>10}  {'p-value':>9}"
    sep2 = "  " + "-" * (len(hdr2) - 2)
    print(sep2)
    print(hdr2)
    print(sep2)
    methods = [
        ("Derived (back - WT)", results["spearman_derived"], results["pvalue_derived"]),
        ("LOO mean baseline", results["spearman_loomean"], results["pvalue_loomean"]),
        (
            "Constant mean (=0)",
            results["spearman_constant"],
            results["pvalue_constant"],
        ),
    ]
    for label, r, p in methods:
        p_str = f"{p:.3f}" if not np.isnan(p) else "  n/a"
        r_str = f"{r:.3f}" if not np.isnan(r) else "  n/a"
        print(f"  {label:<22}  {r_str:>10}  {p_str:>9}")
    print(sep2)
    print()
    print(
        "  NOTE: LOO mean r=-1.000 is an expected mathematical artifact, not a "
        "meaningful result.  When the LOO mean is computed by excluding each true "
        "delta in turn, the resulting predictions are in exact reverse rank order "
        "of the true deltas (removing the largest value lowers the mean; removing "
        "the smallest raises it).  This anti-correlation is structural, not "
        "informative."
    )
    print()
    print(f"  VERDICT: {results['verdict']}")
    print()


def main() -> None:
    cfg = load_model_config()

    # ── Load selectivity dataset ──────────────────────────────────────────────
    sel_path = PROCESSED / "egfr_selectivity_dataset.csv"
    if not sel_path.exists():
        logger.error(f"Selectivity dataset not found: {sel_path}")
        return

    sel_df = pd.read_csv(sel_path)
    logger.info(
        f"Loaded selectivity dataset: {len(sel_df)} pairs.  "
        f"Columns: {sel_df.columns.tolist()}"
    )

    # ── Verify sign convention ────────────────────────────────────────────────
    if "selectivity_delta" in sel_df.columns:
        computed = (sel_df["pic50_mutant"] - sel_df["pic50_wt"]).round(6)
        stored = sel_df["selectivity_delta"].round(6)
        if not (computed == stored).all():
            logger.error(
                "selectivity_delta in file does NOT equal pic50_mutant - pic50_wt.  "
                "Sign convention mismatch — check build_egfr_dataset.py."
            )
            return
        logger.info(
            "Sign convention confirmed: selectivity_delta = pic50_mutant - pic50_wt  "
            "(positive = mutant-selective, consistent with CLAUDE.md)"
        )
    else:
        sel_df["selectivity_delta"] = sel_df["pic50_mutant"] - sel_df["pic50_wt"]
        logger.info("Computed selectivity_delta = pic50_mutant - pic50_wt")

    # ── Load activity models ──────────────────────────────────────────────────
    backbone_dir = MODELS / "general"
    wt_proxy_dir = MODELS / "wt_proxy"

    for d in (backbone_dir, wt_proxy_dir):
        if not d.exists():
            logger.error(f"Model artifact not found: {d}.  Run train_models.py first.")
            return

    backbone_trainer = QSARTrainer.load(backbone_dir, cfg)
    wt_proxy_trainer = QSARTrainer.load(wt_proxy_dir, cfg)
    logger.info(f"Mutant predictor : Model 1 (backbone) — {backbone_trainer.best_name}")
    logger.info(
        f"WT predictor     : Model 2 (WT-proxy)  — {wt_proxy_trainer.best_name}"
    )
    logger.info(
        "NOTE: Both models were trained on overlapping datasets.  Derived delta"
        " = backbone_pred - wt_proxy_pred has a large shared-noise component."
    )

    # ── Run evaluation ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"{EXPLORATORY_LABEL}")
    logger.info("MODEL 4 — Derived selectivity, LOO evaluation")
    logger.info("=" * 60)

    results = evaluate_selectivity(sel_df, backbone_trainer, wt_proxy_trainer)

    # ── Save first, print second ──────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_results = {
        "n_pairs": results["n_pairs"],
        "spearman_derived": results["spearman_derived"],
        "pvalue_derived": results["pvalue_derived"],
        "spearman_loomean": results["spearman_loomean"],
        "pvalue_loomean": results["pvalue_loomean"],
        "spearman_constant": results["spearman_constant"],
        "pvalue_constant": (
            float("nan")
            if np.isnan(results["pvalue_constant"])
            else results["pvalue_constant"]
        ),
        "stability_note": results["stability_note"],
        "verdict": results["verdict"],
        "per_pair": results["per_pair"],
        "exploratory": True,
        "sign_convention": "selectivity_delta = pIC50_mutant - pIC50_wt; positive = mutant-selective",
        "mutant_model": f"Model 1 backbone ({backbone_trainer.best_name})",
        "wt_model": f"Model 2 WT-proxy ({wt_proxy_trainer.best_name})",
        "caveats": [
            "All 9 pairs are in training data for both activity models (no activity holdout)",
            "Backbone and WT-proxy share ~80% training data; derived delta has large shared-noise component",
            f"Spearman r on n={results['n_pairs']} is extremely unstable",
        ],
    }

    out_path = OUT_DIR / "selectivity_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            json_results,
            f,
            indent=2,
            default=lambda x: (
                float(x) if isinstance(x, (np.floating, np.integer)) else str(x)
            ),
        )
    logger.info(f"Saved selectivity results to {out_path}")

    _print_results(results)


if __name__ == "__main__":
    main()
