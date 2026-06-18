"""
Builds three datasets from cleaned EGFR data:
  1. egfr_l858r_dataset.csv   — L858R labeled records only
  2. egfr_wt_dataset.csv      — WT labeled + unknown records (WT proxy)
  3. egfr_selectivity_dataset.csv — molecules with BOTH measurements

Scarce data strategy:
  - 19 L858R records is too few for a standalone model
  - WT model trained on ~1037 molecules (67 labeled + 970 unknown proxy)
  - L858R model uses transfer learning from WT model as starting point
  - ERBB2 data available as additional pretraining source

Run:
    python scripts/build_egfr_dataset.py
"""

from __future__ import annotations

import pandas as pd

from src.utils.config import get_project_root
from src.utils.logging import get_logger

logger = get_logger(__name__)

ROOT = get_project_root()
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


def build_mutant_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Extract L858R labeled records. Log a warning if count is below 50."""
    l858r = df[df["mutation_flag"] == "L858R"].copy()
    logger.info(f"L858R records: {len(l858r)}")
    if len(l858r) < 50:
        logger.warning(
            f"Only {len(l858r)} L858R records. "
            "Transfer learning from WT model is mandatory. "
            "Document this in LIMITATIONS.md."
        )
    l858r["dataset"] = "l858r"
    return l858r.reset_index(drop=True)


def build_wt_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    WT dataset = explicit WT + unknown records.
    Unknown records in ChEMBL EGFR assays are overwhelmingly WT
    since mutation-specific assays are always annotated.
    """
    wt = df[df["mutation_flag"].isin(["wild_type", "unknown"])].copy()
    n_explicit = (wt["mutation_flag"] == "wild_type").sum()
    n_proxy = (wt["mutation_flag"] == "unknown").sum()
    logger.info(
        f"WT dataset: {len(wt)} records ({n_explicit} explicit WT, {n_proxy} unknown proxy)"
    )
    wt["dataset"] = "wt"
    return wt.reset_index(drop=True)


def build_selectivity_dataset(
    df_mutant: pd.DataFrame,
    df_wt: pd.DataFrame,
) -> pd.DataFrame:
    """
    Molecules with BOTH L858R and WT measurements.
    selectivity_delta = pIC50_mutant - pIC50_wt
    Positive = more potent against L858R.
    With only 19 L858R records, this will be very small.
    """
    mut_agg = (
        df_mutant.groupby("canonical_smiles")["pic50"]
        .median()
        .rename("pic50_mutant")
        .reset_index()
    )
    wt_agg = (
        df_wt.groupby("canonical_smiles")["pic50"]
        .median()
        .rename("pic50_wt")
        .reset_index()
    )
    paired = pd.merge(mut_agg, wt_agg, on="canonical_smiles", how="inner")
    paired["selectivity_delta"] = paired["pic50_mutant"] - paired["pic50_wt"]

    logger.info(f"Paired selectivity dataset: {len(paired)} molecules")
    if len(paired) > 0:
        logger.info(
            f"  Mutant-selective (delta>0): {(paired['selectivity_delta'] > 0).sum()}"
        )
        logger.info(
            f"  WT-biased (delta<0):        {(paired['selectivity_delta'] < 0).sum()}"
        )
        logger.info(f"  Mean delta: {paired['selectivity_delta'].mean():.3f}")

    return paired.reset_index(drop=True)


def main() -> None:
    cleaned_path = ROOT / "data" / "interim" / "egfr_cleaned.csv"
    if not cleaned_path.exists():
        logger.error(
            "Cleaned data not found. Run scripts/clean_bioactivity_data.py first."
        )
        return

    df = pd.read_csv(cleaned_path)
    logger.info(f"Loaded cleaned EGFR: {len(df)} rows")

    df_mutant = build_mutant_dataset(df)
    df_wt = build_wt_dataset(df)
    df_sel = build_selectivity_dataset(df_mutant, df_wt)

    # Save
    df_mutant.to_csv(PROCESSED / "egfr_l858r_dataset.csv", index=False)
    df_wt.to_csv(PROCESSED / "egfr_wt_dataset.csv", index=False)
    df_sel.to_csv(PROCESSED / "egfr_selectivity_dataset.csv", index=False)

    logger.info("\n=== Dataset Summary ===")
    logger.info(
        f"L858R:      {len(df_mutant)} records -> data/processed/egfr_l858r_dataset.csv"
    )
    logger.info(
        f"WT:         {len(df_wt)} records -> data/processed/egfr_wt_dataset.csv"
    )
    logger.info(
        f"Paired:     {len(df_sel)} molecules -> data/processed/egfr_selectivity_dataset.csv"
    )
    logger.info(
        "\nStrategy: train WT model first, then fine-tune on L858R via transfer learning."
    )


if __name__ == "__main__":
    main()
