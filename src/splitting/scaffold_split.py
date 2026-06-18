"""Bemis-Murcko scaffold splitting: no scaffold appears in more than one data split."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_bemis_murcko_scaffold(smiles: str) -> str | None:
    """
    Compute Bemis-Murcko scaffold for a molecule.
    Returns the scaffold SMILES, or None if computation fails.
    Aliphatic-only molecules return an empty string scaffold.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return None


def assign_scaffolds(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
) -> pd.DataFrame:
    """Add scaffold column. Molecules with no computable scaffold get 'no_scaffold'."""
    df = df.copy()
    df["scaffold"] = df[smiles_col].apply(
        lambda s: get_bemis_murcko_scaffold(s) or "no_scaffold"
    )
    logger.info(
        f"Found {df['scaffold'].nunique()} unique scaffolds for {len(df)} molecules"
    )
    return df


def scaffold_split(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by Bemis-Murcko scaffold so no scaffold appears in more than one split.
    This prevents data leakage from chemically similar molecules.

    Algorithm:
      1. Group molecules by scaffold
      2. Sort groups largest-first for balanced splits
      3. Assign groups greedily until ratio targets are met

    With scarce L858R data the test set may be small — this is expected and logged.
    """
    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    ), "Ratios must sum to 1.0"

    df = assign_scaffolds(df, smiles_col=smiles_col)

    # Build scaffold -> row index mapping
    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, scaffold in enumerate(df["scaffold"]):
        scaffold_to_indices[scaffold].append(idx)

    # Sort largest-first then shuffle: prevents all large scaffolds from landing in train
    scaffold_groups = sorted(scaffold_to_indices.values(), key=len, reverse=True)
    np.random.default_rng(seed).shuffle(scaffold_groups)

    n_total = len(df)
    train_cutoff = int(n_total * train_ratio)
    val_cutoff = int(n_total * (train_ratio + val_ratio))

    train_idx, val_idx, test_idx = [], [], []
    for group in scaffold_groups:
        if len(train_idx) < train_cutoff:
            train_idx.extend(group)
        elif len(train_idx) + len(val_idx) < val_cutoff:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    logger.info(
        f"Split sizes: train={len(train_df)} ({len(train_df)/n_total:.1%}), "
        f"val={len(val_df)} ({len(val_df)/n_total:.1%}), "
        f"test={len(test_df)} ({len(test_df)/n_total:.1%})"
    )

    if len(test_df) < 20:
        logger.warning(
            f"Test set only has {len(test_df)} molecules. "
            "Expected with scarce L858R data — document this in limitations."
        )

    # Verify no leakage
    t, v, s = (
        set(train_df["scaffold"]),
        set(val_df["scaffold"]),
        set(test_df["scaffold"]),
    )
    if t & v or t & s or v & s:
        logger.warning("Scaffold leakage detected between splits")
    else:
        logger.info("No scaffold leakage confirmed")

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
