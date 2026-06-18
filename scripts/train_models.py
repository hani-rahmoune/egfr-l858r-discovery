"""
Train Model 1 (EGFR general backbone) and Model 2 (WT-proxy).

Model 1 — EGFR general backbone
  Input:  data/processed/features_egfr_general.parquet
  Output: models/qsar/general/

Model 2 — WT-proxy comparator
  Input:  data/processed/features_wt_proxy.parquet
  Output: models/qsar/wt_proxy/
  NOTE: ~94% of this data is unspecified-mutation EGFR, NOT pure WT.
        Label as WT-proxy, never as wild-type-only.

Prerequisites:
  - scripts/compute_features.py must have run
  - scripts/assign_splits.py must have run (parquets need a 'split' column)

Run:
    PYTHONPATH=. .venv/Scripts/python.exe scripts/train_models.py
"""

from __future__ import annotations

from src.models.qsar import QSARTrainer
from src.utils.config import get_project_root, load_model_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

ROOT = get_project_root()
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models" / "qsar"


def train_and_save(label: str, parquet_name: str, out_name: str, cfg: dict) -> dict:
    parquet_path = PROCESSED / f"{parquet_name}.parquet"
    out_dir = MODELS / out_name

    if not parquet_path.exists():
        logger.error(
            f"{label}: {parquet_path} not found — run compute_features.py and assign_splits.py first"
        )
        return {}

    trainer = QSARTrainer(cfg)
    trainer.fit_from_parquet(parquet_path, label=label)
    trainer.save(out_dir)
    return trainer.test_metrics


def print_results_table(results: dict[str, dict]) -> None:
    header = f"{'Model':<28} {'Best':<14} {'Test RMSE':>10} {'Test MAE':>9} {'Test R2':>8} {'Pearson r':>10} {'n':>5}"
    sep = "-" * len(header)
    print()
    print(sep)
    print("SCAFFOLD-SPLIT TEST METRICS")
    print(sep)
    print(header)
    print(sep)

    for label, (best_name, metrics) in results.items():
        if not metrics:
            print(f"  {label:<26}  FAILED")
            continue
        print(
            f"  {label:<26}  {best_name:<14}"
            f"  {metrics['rmse']:>8.3f}"
            f"  {metrics['mae']:>8.3f}"
            f"  {metrics['r2']:>7.3f}"
            f"  {metrics['pearson_r']:>9.3f}"
            f"  {metrics['n']:>5}"
        )
    print(sep)


def main() -> None:
    cfg = load_model_config()

    # Model 1 — EGFR general backbone
    logger.info("=" * 60)
    logger.info("MODEL 1 — EGFR general backbone")
    logger.info("=" * 60)
    m1_metrics = train_and_save(
        label="EGFR general (Model 1)",
        parquet_name="features_egfr_general",
        out_name="general",
        cfg=cfg,
    )
    m1_trainer = QSARTrainer.load(MODELS / "general", cfg)

    # Model 2 — WT-proxy
    logger.info("=" * 60)
    logger.info("MODEL 2 — WT-proxy comparator")
    logger.info("=" * 60)
    m2_metrics = train_and_save(
        label="WT-proxy (Model 2)",
        parquet_name="features_wt_proxy",
        out_name="wt_proxy",
        cfg=cfg,
    )
    m2_trainer = QSARTrainer.load(MODELS / "wt_proxy", cfg)

    results = {
        "Model 1 — EGFR general": (m1_trainer.best_name, m1_metrics),
        "Model 2 — WT-proxy": (m2_trainer.best_name, m2_metrics),
    }
    print_results_table(results)

    print()
    print("Val RMSE by candidate (Model 1):")
    for name, m in m1_trainer.val_metrics.items():
        print(f"  {name:<16}  RMSE={m['rmse']:.3f}  R²={m['r2']:.3f}")

    print()
    print("Val RMSE by candidate (Model 2):")
    for name, m in m2_trainer.val_metrics.items():
        print(f"  {name:<16}  RMSE={m['rmse']:.3f}  R²={m['r2']:.3f}")

    print()
    print("Artifacts written to:")
    print(f"  {MODELS / 'general'}")
    print(f"  {MODELS / 'wt_proxy'}")


if __name__ == "__main__":
    main()
