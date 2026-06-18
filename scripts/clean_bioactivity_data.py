"""
Reads raw ChEMBL data, applies the full cleaning pipeline,
and saves cleaned interim files ready for dataset construction.

Run:
    python scripts/clean_bioactivity_data.py
"""

from __future__ import annotations

import pandas as pd

from src.data.cleaning import run_full_cleaning_pipeline
from src.data.standardization import run_activity_standardization
from src.utils.config import get_project_root
from src.utils.logging import get_logger

logger = get_logger(__name__)

ROOT = get_project_root()


def main() -> None:
    # ── Load raw data ──────────────────────────────────────────────
    egfr_path = ROOT / "data" / "raw" / "chembl_egfr_bioactivity.csv"
    erbb2_path = ROOT / "data" / "raw" / "chembl_erbb2_bioactivity.csv"

    if not egfr_path.exists():
        logger.error("Raw EGFR file not found. Run the Colab notebook first.")
        return

    egfr_raw = pd.read_csv(egfr_path)
    logger.info(f"Loaded EGFR raw: {len(egfr_raw)} rows")

    erbb2_raw = pd.read_csv(erbb2_path) if erbb2_path.exists() else pd.DataFrame()
    if len(erbb2_raw) > 0:
        logger.info(f"Loaded ERBB2 raw: {len(erbb2_raw)} rows")

    # ── Clean EGFR ────────────────────────────────────────────────
    egfr_clean = run_full_cleaning_pipeline(egfr_raw, smiles_col="canonical_smiles")
    egfr_std = run_activity_standardization(
        egfr_clean,
        value_col="standard_value",
        unit_col="standard_units",
        type_col="standard_type",
        pchembl_col="pchembl_value",
    )

    # ── Clean ERBB2 ───────────────────────────────────────────────
    if len(erbb2_raw) > 0:
        erbb2_clean = run_full_cleaning_pipeline(
            erbb2_raw, smiles_col="canonical_smiles"
        )
        erbb2_std = run_activity_standardization(
            erbb2_clean,
            value_col="standard_value",
            unit_col="standard_units",
            type_col="standard_type",
            pchembl_col="pchembl_value",
        )
        erbb2_std["source"] = "erbb2"
        erbb2_out = ROOT / "data" / "interim" / "erbb2_cleaned.csv"
        erbb2_std.to_csv(erbb2_out, index=False)
        logger.info(f"Saved ERBB2 cleaned: {erbb2_out} ({len(erbb2_std)} rows)")

    # ── Relabel mislabelled L858R records ─────────────────────────
    # Assay CHEMBL4380726 is a KINOMEscan Kd assay on the EGFR L858R
    # construct.  Its title says "wild-type partial length EGFR L858R
    # mutant" where "wild-type" refers to the protein backbone (R669-
    # V1011), NOT the mutation status.  The Colab notebook therefore
    # assigned mutation_flag = "wild_type" in error.  Keyed on the
    # stable assay ID so this is idempotent; raw file is never touched.
    if "mutation_flag" in egfr_std.columns and "assay_chembl_id" in egfr_std.columns:
        logger.info("mutation_flag BEFORE relabel:")
        logger.info(egfr_std["mutation_flag"].value_counts(dropna=False).to_string())
        relabel_mask = egfr_std["assay_chembl_id"] == "CHEMBL4380726"
        egfr_std.loc[relabel_mask, "mutation_flag"] = "L858R"
        logger.info("mutation_flag AFTER  relabel:")
        logger.info(egfr_std["mutation_flag"].value_counts(dropna=False).to_string())
        logger.info("Records relabelled: %d (expected 5)", relabel_mask.sum())

    # ── Save EGFR cleaned ─────────────────────────────────────────
    interim_dir = ROOT / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    egfr_std["source"] = "egfr"
    out_path = interim_dir / "egfr_cleaned.csv"
    egfr_std.to_csv(out_path, index=False)

    # ── Summary ───────────────────────────────────────────────────
    logger.info("=== Cleaning Summary ===")
    logger.info(f"EGFR raw:     {len(egfr_raw)}")
    logger.info(f"EGFR cleaned: {len(egfr_std)}")
    logger.info(
        f"pIC50 range:  [{egfr_std['pic50'].min():.2f}, {egfr_std['pic50'].max():.2f}]"
    )

    if "mutation_flag" in egfr_std.columns:
        logger.info(
            f"Mutation flags:\n{egfr_std['mutation_flag'].value_counts().to_string()}"
        )


if __name__ == "__main__":
    main()
