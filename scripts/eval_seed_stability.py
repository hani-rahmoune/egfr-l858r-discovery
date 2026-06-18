"""
5-seed scaffold-split stability evaluation for Model 1 and Model 2.

Reruns scaffold splitting and full model selection (RF / XGB / LGB) at each
seed.  Reports mean +/- std for test R2, RMSE, and Pearson r.

Seeds: [42, 7, 13, 99, 123]

Run:
    PYTHONPATH=. .venv/Scripts/python.exe scripts/eval_seed_stability.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.mlflow_utils import log_seed_summary, start_run
from src.models.qsar import QSARTrainer
from src.splitting.scaffold_split import scaffold_split
from src.utils.config import get_project_root, load_model_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

ROOT = get_project_root()
PROCESSED = ROOT / "data" / "processed"
SEEDS = [42, 7, 13, 99, 123]


def run_one_seed(parquet_path: Path, seed: int, cfg: dict) -> dict:
    """Run one scaffold split + model selection at given seed. Returns test metrics dict."""
    df = pd.read_parquet(parquet_path)

    split_cfg = cfg.get("scaffold_split", {})
    train_r = split_cfg.get("train_ratio", 0.70)
    val_r = split_cfg.get("val_ratio", 0.15)
    test_r = split_cfg.get("test_ratio", 0.15)

    working = df[["canonical_smiles"]].copy().reset_index(drop=False)
    working.columns = ["original_index", "canonical_smiles"]

    train_df, val_df, test_df = scaffold_split(
        working,
        smiles_col="canonical_smiles",
        train_ratio=train_r,
        val_ratio=val_r,
        test_ratio=test_r,
        seed=seed,
    )

    split_map: dict[int, str] = {}
    for sub_df, label in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        for orig_idx, split_label in zip(sub_df["original_index"], sub_df["split"]):
            split_map[orig_idx] = split_label

    df = df.copy()
    df["split"] = [split_map[i] for i in range(len(df))]

    # Write to a temp parquet so QSARTrainer.fit_from_parquet can read it
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_path = Path(f.name)
    try:
        df.to_parquet(tmp_path, index=False)
        trainer = QSARTrainer(cfg)
        trainer.fit_from_parquet(tmp_path, label=f"seed={seed}")
    finally:
        tmp_path.unlink(missing_ok=True)

    return {**trainer.test_metrics, "best_model": trainer.best_name, "seed": seed}


def evaluate_model(name: str, parquet_name: str, cfg: dict) -> list[dict]:
    """Returns per-seed metrics list so callers can log or compare downstream."""
    parquet_path = PROCESSED / f"{parquet_name}.parquet"
    if not parquet_path.exists():
        logger.error(f"{name}: {parquet_path} not found")
        return []

    logger.info("=" * 60)
    logger.info(f"{name} — 5-seed scaffold-split evaluation")
    logger.info("=" * 60)

    task = parquet_name.replace("features_", "")  # e.g. "egfr_general" or "wt_proxy"

    per_seed: list[dict] = []
    for seed in SEEDS:
        logger.info(f"  seed={seed} ...")
        m = run_one_seed(parquet_path, seed, cfg)
        per_seed.append(m)
        logger.info(
            f"    best={m['best_model']}  RMSE={m['rmse']:.3f}  R2={m['r2']:.3f}  pearson_r={m['pearson_r']:.3f}  n={m['n']}"
        )

    r2s = [m["r2"] for m in per_seed]
    rmses = [m["rmse"] for m in per_seed]
    pearsons = [m["pearson_r"] for m in per_seed]
    ns = [m["n"] for m in per_seed]

    print()
    print(f"{name}")
    hdr = f"  {'Seed':>5}  {'Best model':<16}  {'RMSE':>8}  {'R2':>7}  {'Pearson r':>10}  {'n':>5}"
    sep = "  " + "-" * (len(hdr) - 2)
    print(sep)
    print(hdr)
    print(sep)
    for seed, m in zip(SEEDS, per_seed):
        print(
            f"  {seed:>5}  {m['best_model']:<16}"
            f"  {m['rmse']:>8.3f}  {m['r2']:>7.3f}"
            f"  {m['pearson_r']:>10.3f}  {m['n']:>5}"
        )
    print(sep)
    print(
        f"  {'mean':>5}  {'':16}"
        f"  {np.mean(rmses):>8.3f}  {np.mean(r2s):>7.3f}"
        f"  {np.mean(pearsons):>10.3f}  {int(np.mean(ns)):>5}"
    )
    print(
        f"  {'std':>5}  {'':16}"
        f"  {np.std(rmses):>8.3f}  {np.std(r2s):>7.3f}"
        f"  {np.std(pearsons):>10.3f}"
    )
    print(sep)
    print()
    print(
        f"  Summary  RMSE={np.mean(rmses):.3f}+/-{np.std(rmses):.3f}  "
        f"R2={np.mean(r2s):.3f}+/-{np.std(r2s):.3f}  "
        f"Pearson_r={np.mean(pearsons):.3f}+/-{np.std(pearsons):.3f}"
    )
    print()

    # Log summary to MLflow (best-winning model name from seed 42 run as the model label)
    best_model_name = per_seed[0].get("best_model", "qsar")
    try:
        log_seed_summary(
            task=task,
            model=best_model_name,
            per_seed=[{k: v for k, v in m.items() if k != "best_model"} for m in per_seed],
            params={"seeds": str(SEEDS), "n_seeds": len(SEEDS), "fp_type": "morgan_ecfp4_desc"},
        )
    except Exception as exc:
        logger.warning(f"MLflow logging failed (non-fatal): {exc}")

    return per_seed


def main() -> None:
    cfg = load_model_config()

    evaluate_model(
        name="Model 1 — EGFR general backbone",
        parquet_name="features_egfr_general",
        cfg=cfg,
    )

    evaluate_model(
        name="Model 2 — WT-proxy",
        parquet_name="features_wt_proxy",
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
