"""
Model 3 — L858R-adapted QSAR calibration (EXPLORATORY).

** All output from this script is labeled exploratory. **
** Do not report these predictions as validated model outputs. **

Fine-tunes the general EGFR backbone on the 22 L858R molecules via LOOCV.

Design
------
The general backbone (Model 1) was trained on all 1253 EGFR molecules,
including the 22 L858R records.  A naive comparison would leak those
22 molecules into the backbone.  This script prevents that:

  In each LOO fold i:
    1. Retrain backbone RF on all 1252 general EGFR molecules EXCLUDING
       the held-out L858R molecule.  This backbone has NEVER seen molecule i.
    2. Backbone predicts molecule i  → unbiased backbone prediction.
    3. Calibration training set: backbone's OOB predictions for the 21 other
       L858R molecules (OOB = out-of-bag, not in-sample — avoids overfitting
       bias in calibration training).
    4. Fit MeanShift and Ridge calibrators on (OOB backbone pred, y_true).
    5. Apply calibrators to predict molecule i.

Calibrators are fitted on backbone predictions only (no raw molecular features
are passed to the calibrators directly), so they are NOT standalone models.

Backbone n_estimators in the LOO loop is 100 (not 300 as in saved artifacts)
to keep runtime manageable; this trades ~1-2% accuracy for 3× speedup.

Run:
    PYTHONPATH=. .venv/Scripts/python.exe scripts/train_l858r_model.py

Output:
    models/qsar/l858r/loocv_results.json   — full per-seed and summary stats
    (no best_model.pkl — there is no single model artifact for Model 3)
"""

from __future__ import annotations

import json

import pandas as pd

from src.models.l858r_calibration import (
    EXPLORATORY_LABEL,
    interpret_result,
    run_loocv,
)
from src.models.qsar import get_feature_cols
from src.utils.config import get_project_root, load_model_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

ROOT = get_project_root()
PROCESSED = ROOT / "data" / "processed"
OUT_DIR = ROOT / "models" / "qsar" / "l858r"
SEEDS = [42, 7, 13, 99, 123]

# Backbone RF size in the LOO loop.
# 100 trees (not 300) trades a small accuracy loss for a 3× speedup.
# The saved artifacts use 300; this value only affects the LOO evaluation.
N_ESTIMATORS_LOO = 100


def _print_results(results: dict) -> None:
    summary = results["summary"]

    print()
    print(f"  {EXPLORATORY_LABEL}")
    print(f"  n_L858R = {results['n_l858r']}  seeds = {results['seeds']}")
    print()

    hdr = (
        f"  {'Method':<14}  {'Spearman r (mean+-std)':>22}"
        f"  {'RMSE (mean+-std)':>18}"
    )
    sep = "  " + "-" * (len(hdr) - 2)
    print(sep)
    print(hdr)
    print(sep)

    method_labels = {
        "backbone": "Backbone (base)",
        "shift": "Mean-shift cal",
        "ridge": "Ridge cal",
    }
    for method, label in method_labels.items():
        s = summary[method]
        spear_str = f"{s['spearman_mean']:.3f} ± {s['spearman_std']:.3f}"
        rmse_str = f"{s['rmse_mean']:.3f} ± {s['rmse_std']:.3f}"
        print(f"  {label:<14}  {spear_str:>22}  {rmse_str:>18}")

    print(sep)
    print()

    # Per-seed detail
    print("  Per-seed Spearman r:")
    hdr2 = f"  {'Seed':>6}  {'Backbone':>10}  {'Shift':>10}  {'Ridge':>10}"
    sep2 = "  " + "-" * (len(hdr2) - 2)
    print(sep2)
    print(hdr2)
    print(sep2)
    for seed in results["seeds"]:
        ps = results["per_seed"][seed]
        print(
            f"  {seed:>6}  {ps['backbone_spearman']:>10.3f}"
            f"  {ps['shift_spearman']:>10.3f}"
            f"  {ps['ridge_spearman']:>10.3f}"
        )
    print(sep2)
    print()


def main() -> None:
    cfg = load_model_config()

    parquet_path = PROCESSED / "features_egfr_general.parquet"
    if not parquet_path.exists():
        logger.error(
            f"features_egfr_general.parquet not found at {parquet_path}. "
            f"Run compute_features.py and assign_splits.py first."
        )
        return

    general_df = pd.read_parquet(parquet_path)

    if "mutation_flag" not in general_df.columns:
        logger.error(
            "mutation_flag column not found in features_egfr_general.parquet. "
            "The parquet was built before the relabeling step — re-run the full pipeline."
        )
        return

    n_l858r = (general_df["mutation_flag"] == "L858R").sum()
    logger.info(f"General parquet: {len(general_df)} rows, {n_l858r} flagged L858R")

    if n_l858r == 0:
        logger.error("No L858R molecules found in general parquet.")
        return

    feature_cols = get_feature_cols(general_df)
    logger.info(f"Feature columns: {len(feature_cols)}")

    logger.info("=" * 60)
    logger.info(f"{EXPLORATORY_LABEL}")
    logger.info("MODEL 3 — L858R calibration, LOOCV evaluation")
    logger.info("=" * 60)
    logger.info(
        f"Backbone n_estimators in LOO loop: {N_ESTIMATORS_LOO} "
        f"(saved artifacts use 300; see script header for rationale)"
    )

    results = run_loocv(
        general_df=general_df,
        feature_cols=feature_cols,
        seeds=SEEDS,
        n_estimators_loo=N_ESTIMATORS_LOO,
        ridge_alpha=1.0,
        model_cfg=cfg,
    )

    verdict = interpret_result(results["summary"])

    # ── Save results first (before printing, so data survives terminal errors) ─
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    json_results = {
        "summary": results["summary"],
        "per_seed": {str(s): v for s, v in results["per_seed"].items()},
        "n_l858r": int(results["n_l858r"]),
        "seeds": results["seeds"],
        "n_estimators_loo": N_ESTIMATORS_LOO,
        "exploratory": True,
        "verdict": verdict,
    }

    out_path = OUT_DIR / "loocv_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2)
    logger.info(f"Saved LOOCV results to {out_path}")

    _print_results(results)
    print(f"  VERDICT: {verdict}")
    print()
    logger.info(
        "NOTE: No best_model.pkl is saved for Model 3. "
        "There is no single deployable artifact — Model 3 is a "
        "calibration protocol, not a standalone model."
    )


if __name__ == "__main__":
    main()
