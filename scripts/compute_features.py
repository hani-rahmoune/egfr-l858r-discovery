"""
Computes Morgan ECFP4 + RDKit descriptors for all datasets.

Datasets produced:
  features_egfr_general.parquet  — all 1253 EGFR records, backbone model
  features_wt_proxy.parquet      — 1021 records (64 explicit WT + 957 unspecified EGFR)
  features_l858r.parquet         — 19 L858R records, calibration/evaluation only
  features_erbb2.parquet         — 604 ERBB2 records, optional pretraining source

Note on L858R:
  19 records is not enough for a standalone model.
  Used for LOOCV evaluation and mutation-aware calibration only.

Note on WT/proxy:
  Called WT-proxy because it contains 64 explicit WT + 957 unspecified EGFR assays.
  Do not call it WT-only.

Run:
    python scripts/compute_features.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.descriptors import DESCRIPTOR_NAMES, compute_descriptor_matrix
from src.features.fingerprints import compute_fingerprint_matrix
from src.utils.config import get_project_root
from src.utils.logging import get_logger

logger = get_logger(__name__)

ROOT = get_project_root()
PROCESSED = ROOT / "data" / "processed"
INTERIM = ROOT / "data" / "interim"

# Feature column names, fixed order — models depend on this
FP_COLS = [f"fp_{i}" for i in range(2048)]
FEATURE_COLS = FP_COLS + DESCRIPTOR_NAMES  # 2048 + 11 = 2059 total


def build_feature_matrix(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    label: str = "",
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Combine Morgan ECFP4 (2048 bits) + RDKit descriptors (11) = 2059 features.
    Returns (X, y, valid_indices).
    Molecules that fail fingerprint or descriptor computation are excluded.
    """
    smiles_list = df[smiles_col].tolist()

    fp_matrix, fp_valid = compute_fingerprint_matrix(
        smiles_list, fp_type="morgan_ecfp4", radius=2, n_bits=2048
    )
    desc_matrix, desc_valid = compute_descriptor_matrix(smiles_list)

    # Keep only molecules that succeeded in both
    valid_set = set(fp_valid) & set(desc_valid)
    valid_indices = sorted(valid_set)

    fp_map = {orig: i for i, orig in enumerate(fp_valid)}
    desc_map = {orig: i for i, orig in enumerate(desc_valid)}

    fp_rows = np.array([fp_matrix[fp_map[i]] for i in valid_indices])
    desc_rows = np.array([desc_matrix[desc_map[i]] for i in valid_indices])

    X = np.concatenate([fp_rows, desc_rows], axis=1).astype(np.float32)
    y = df["pic50"].iloc[valid_indices].values.astype(np.float32)

    logger.info(
        f"{label}: {len(valid_indices)}/{len(df)} molecules, "
        f"shape {X.shape}, "
        f"{len(df) - len(valid_indices)} skipped"
    )
    return X, y, valid_indices


def save_features(
    X: np.ndarray,
    y: np.ndarray,
    valid_indices: list[int],
    df: pd.DataFrame,
    output_path,
    label: str,
) -> None:
    """Save feature matrix + targets + metadata as parquet."""
    feature_df = pd.DataFrame(X, columns=FEATURE_COLS)
    feature_df["pic50"] = y

    # Attach useful metadata columns if present
    for col in ["canonical_smiles", "mutation_flag", "scaffold", "split", "source"]:
        if col in df.columns:
            feature_df[col] = df[col].iloc[valid_indices].values

    feature_df.to_parquet(output_path, index=False)
    logger.info(
        f"Saved {label}: {output_path} ({len(feature_df)} rows, {X.shape[1]} features)"
    )


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # ── 1. General EGFR backbone (all 1253 records) ───────────────
    # This is the main training set for Model 1.
    # Contains all mutation types: WT, L858R, T790M, del19, unknown.
    egfr_cleaned_path = INTERIM / "egfr_cleaned.csv"
    if not egfr_cleaned_path.exists():
        logger.error("egfr_cleaned.csv not found. Run clean_bioactivity_data.py first.")
        return

    egfr_all = pd.read_csv(egfr_cleaned_path)
    X_gen, y_gen, idx_gen = build_feature_matrix(egfr_all, label="EGFR general")
    save_features(
        X_gen,
        y_gen,
        idx_gen,
        egfr_all,
        PROCESSED / "features_egfr_general.parquet",
        label="EGFR general",
    )

    # ── 2. WT-proxy dataset (64 explicit WT + 957 unspecified EGFR) ──
    # Used for Model 2. Called WT-proxy, not WT-only.
    # The 957 unspecified records are standard EGFR assays without mutation annotation,
    # which in ChEMBL practice correspond to non-mutant EGFR.
    wt_path = PROCESSED / "egfr_wt_dataset.csv"
    if wt_path.exists():
        wt_df = pd.read_csv(wt_path)
        X_wt, y_wt, idx_wt = build_feature_matrix(wt_df, label="WT-proxy")
        save_features(
            X_wt,
            y_wt,
            idx_wt,
            wt_df,
            PROCESSED / "features_wt_proxy.parquet",
            label="WT-proxy",
        )

    # ── 3. L858R calibration set (19 records) ────────────────────
    # Not enough for a standalone model.
    # Used for LOOCV evaluation and mutation-aware calibration only.
    mut_path = PROCESSED / "egfr_l858r_dataset.csv"
    if mut_path.exists():
        mut_df = pd.read_csv(mut_path)
        X_mut, y_mut, idx_mut = build_feature_matrix(mut_df, label="L858R")
        save_features(
            X_mut,
            y_mut,
            idx_mut,
            mut_df,
            PROCESSED / "features_l858r.parquet",
            label="L858R",
        )

    # ── 4. ERBB2 (optional pretraining source) ────────────────────
    erbb2_path = INTERIM / "erbb2_cleaned.csv"
    if erbb2_path.exists():
        erbb2_df = pd.read_csv(erbb2_path)
        X_er, y_er, idx_er = build_feature_matrix(erbb2_df, label="ERBB2")
        save_features(
            X_er,
            y_er,
            idx_er,
            erbb2_df,
            PROCESSED / "features_erbb2.parquet",
            label="ERBB2",
        )

    # ── Summary ───────────────────────────────────────────────────
    logger.info("\n=== Feature Matrix Summary ===")
    logger.info(f"EGFR general (backbone):  {X_gen.shape}")
    logger.info(f"WT-proxy (comparator):    {X_wt.shape}")
    logger.info(f"L858R (calibration only): {X_mut.shape}")
    logger.info(
        f"Feature dimension:        2048 Morgan ECFP4 + 11 descriptors = {X_gen.shape[1]}"
    )
    logger.info("\nModeling plan:")
    logger.info("  Model 1 — EGFR general backbone    trained on 1253 molecules")
    logger.info("  Model 2 — WT-proxy comparator       trained on ~1021 molecules")
    logger.info("  Model 3 — L858R-adapted             general model + LOOCV on 19")
    logger.info("  Model 4 — T790M (later)             200 resistance mutation records")


if __name__ == "__main__":
    main()
