"""
Assign Bemis-Murcko scaffold-based train/val/test splits to feature parquets.

Adds a 'split' column ('train'/'val'/'test') to:
  features_egfr_general.parquet
  features_wt_proxy.parquet
  features_erbb2.parquet

features_l858r.parquet is intentionally excluded: 19 records are used for
fine-tuning and LOOCV only, not for a split-based eval.

Run:
    python scripts/assign_splits.py [--dry-run]

    --dry-run  Print split sizes without writing files.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.splitting.scaffold_split import scaffold_split
from src.utils.config import get_seed, load_model_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

TARGETS = [
    ("EGFR general", "data.processed.features_mutant", "features_egfr_general"),
    ("WT-proxy", "data.processed.features_wt", "features_wt_proxy"),
    ("ERBB2", None, "features_erbb2"),
]

PROCESSED = None  # set at runtime


def _parquet_path(name: str):
    from src.utils.config import get_project_root

    return get_project_root() / "data" / "processed" / f"{name}.parquet"


def assign_split_to_parquet(
    label: str,
    name: str,
    cfg: dict,
    seed: int,
    dry_run: bool,
) -> None:
    path = _parquet_path(name)
    if not path.exists():
        logger.warning(f"{label}: {path} not found, skipping")
        return

    df = pd.read_parquet(path)

    if "canonical_smiles" not in df.columns:
        logger.error(f"{label}: no canonical_smiles column, cannot split")
        return

    if "split" in df.columns:
        logger.info(f"{label}: 'split' column already present — overwriting")

    split_cfg = cfg.get("scaffold_split", {})
    train_r = split_cfg.get("train_ratio", 0.70)
    val_r = split_cfg.get("val_ratio", 0.15)
    test_r = split_cfg.get("test_ratio", 0.15)

    # scaffold_split returns three dataframes, each with a 'split' column and
    # a 'scaffold' column.  We need to map the assignments back to the original
    # row order, so we work via the index.
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

    # Build a mapping from original_index -> split label
    split_map: dict[int, str] = {}
    for sub_df in (train_df, val_df, test_df):
        for orig_idx, split_label in zip(sub_df["original_index"], sub_df["split"]):
            split_map[orig_idx] = split_label

    df["split"] = [split_map[i] for i in range(len(df))]

    n_train = (df["split"] == "train").sum()
    n_val = (df["split"] == "val").sum()
    n_test = (df["split"] == "test").sum()
    n_total = len(df)

    logger.info(
        f"{label}  total={n_total}  "
        f"train={n_train} ({n_train/n_total:.1%})  "
        f"val={n_val} ({n_val/n_total:.1%})  "
        f"test={n_test} ({n_test/n_total:.1%})"
    )

    if dry_run:
        logger.info(f"{label}: dry-run, not writing")
        return

    df.to_parquet(path, index=False)
    logger.info(f"{label}: saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_model_config()
    seed = get_seed()

    logger.info(f"scaffold_split seed={seed}  dry_run={args.dry_run}")

    for label, _, name in TARGETS:
        assign_split_to_parquet(label, name, cfg, seed, dry_run=args.dry_run)

    if args.dry_run:
        logger.info("Dry-run complete. Re-run without --dry-run to write files.")
    else:
        logger.info("All splits assigned and saved.")


if __name__ == "__main__":
    main()
