from __future__ import annotations

import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from src.utils.logging import get_logger

logger = get_logger(__name__)


def is_valid_smiles(smiles: str) -> bool:
    if not isinstance(smiles, str) or not smiles.strip():
        return False
    return Chem.MolFromSmiles(smiles) is not None


def standardize_smiles(smiles: str) -> str | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        mol = rdMolStandardize.Normalizer().normalize(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        return canonical if canonical else None
    except Exception:
        return None


def remove_invalid_smiles(df: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
    n = len(df)
    df = df.dropna(subset=[smiles_col]).copy()
    df = df[df[smiles_col].str.strip() != ""]
    df = df[df[smiles_col].apply(is_valid_smiles)]
    logger.info(f"remove_invalid_smiles: {n} -> {len(df)} ({n - len(df)} removed)")
    return df.reset_index(drop=True)


def add_canonical_smiles(df: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
    df = df.copy()
    df["canonical_smiles"] = df[smiles_col].apply(standardize_smiles)
    n_failed = df["canonical_smiles"].isna().sum()
    if n_failed > 0:
        logger.info(f"add_canonical_smiles: {n_failed} failed, removing")
    return df.dropna(subset=["canonical_smiles"]).reset_index(drop=True)


def remove_mixtures(df: pd.DataFrame, smiles_col: str = "canonical_smiles") -> pd.DataFrame:
    df = df.copy()
    is_mix = df[smiles_col].str.contains(r"\.", regex=False, na=False)
    if is_mix.sum() > 0:
        logger.info(f"remove_mixtures: removing {is_mix.sum()}")
    return df[~is_mix].reset_index(drop=True)


def deduplicate_by_smiles(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    activity_col: str = "pic50",
    keep: str = "median",
) -> pd.DataFrame:
    n = len(df)
    if keep == "median" and activity_col in df.columns:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        agg = {c: "median" for c in numeric_cols if c != smiles_col}
        for c in df.columns:
            if c not in numeric_cols and c != smiles_col:
                agg[c] = "first"
        df = df.groupby(smiles_col, as_index=False).agg(agg)
    else:
        df = df.drop_duplicates(subset=[smiles_col], keep="first")
    logger.info(f"deduplicate: {n} -> {len(df)}")
    return df.reset_index(drop=True)


def filter_by_atom_types(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    allowed_atoms: set[str] | None = None,
) -> pd.DataFrame:
    if allowed_atoms is None:
        allowed_atoms = {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"}

    def _ok(smi: str) -> bool:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        return {a.GetSymbol() for a in mol.GetAtoms()}.issubset(allowed_atoms)

    n = len(df)
    df = df[df[smiles_col].apply(_ok)].reset_index(drop=True)
    if n - len(df) > 0:
        logger.info(f"filter_by_atom_types: removed {n - len(df)}")
    return df


def filter_by_heavy_atom_count(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    min_heavy: int = 7,
    max_heavy: int = 70,
) -> pd.DataFrame:
    def _count(smi: str) -> int:
        mol = Chem.MolFromSmiles(smi)
        return mol.GetNumHeavyAtoms() if mol else 0

    counts = df[smiles_col].apply(_count)
    mask = (counts >= min_heavy) & (counts <= max_heavy)
    if (~mask).sum() > 0:
        logger.info(f"filter_by_heavy_atom_count: removed {(~mask).sum()}")
    return df[mask].reset_index(drop=True)


def run_full_cleaning_pipeline(df: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
    logger.info(f"Cleaning pipeline start: {len(df)} records")
    df = remove_invalid_smiles(df, smiles_col=smiles_col)
    df = add_canonical_smiles(df, smiles_col=smiles_col)
    df = remove_mixtures(df)
    df = filter_by_atom_types(df)
    df = filter_by_heavy_atom_count(df)
    logger.info(f"Cleaning done: {len(df)} records")
    return df