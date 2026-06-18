"""
Regression tests for the CHEMBL4380726 L858R relabeling step.

The relabeling logic lives in scripts/clean_bioactivity_data.py.
Unit tests exercise the logic directly with synthetic data.
Integration tests verify the actual cleaned output file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.cleaning import filter_by_heavy_atom_count

# ── SMILES from the two oversized CHEMBL4380726 records (74 heavy atoms each)
_PROTAC_SMILES = {
    "CHEMBL4529558": (
        "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCN(C(=O)CCCCCCCCCC(=O)"
        "N[C@H](C(=O)N2C[C@H](O)C[C@H]2C(=O)NCc2ccc(-c3scnc3C)cc2)C(C)(C)C)CC1"
    ),
    "CHEMBL4578319": (
        "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCN(C(=O)CCCCCCCCCC(=O)"
        "N[C@H](C(=O)N2C[C@@H](O)C[C@@H]2C(=O)NCc2ccc(-c3scnc3C)cc2)C(C)(C)C)CC1"
    ),
}

_SURVIVOR_SMILES = {
    # CHEMBL939 — afatinib core, 31 heavy atoms, passes max_heavy=70
    "CHEMBL939": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
}


def _apply_relabel(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror the relabeling logic from clean_bioactivity_data.py."""
    df = df.copy()
    if "mutation_flag" in df.columns and "assay_chembl_id" in df.columns:
        mask = df["assay_chembl_id"] == "CHEMBL4380726"
        df.loc[mask, "mutation_flag"] = "L858R"
    return df


def _make_assay_df(
    flags: list[str], assay_ids: list[str] | None = None
) -> pd.DataFrame:
    n = len(flags)
    if assay_ids is None:
        assay_ids = ["CHEMBL4380726"] * n
    return pd.DataFrame(
        {
            "assay_chembl_id": assay_ids,
            "mutation_flag": flags,
            "pic50": [7.5] * n,
        }
    )


# ── Unit tests ────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestRelabelPattern:
    def test_wild_type_rows_become_l858r(self):
        df = _make_assay_df(["wild_type", "wild_type", "wild_type"])
        result = _apply_relabel(df)
        assert (result["mutation_flag"] == "L858R").all()

    def test_other_assays_are_untouched(self):
        df = _make_assay_df(
            ["wild_type", "wild_type", "unknown"],
            assay_ids=["CHEMBL4380726", "CHEMBL4380726", "CHEMBL9999999"],
        )
        result = _apply_relabel(df)
        assert result.loc[2, "mutation_flag"] == "unknown"

    def test_idempotent_applying_once_vs_twice(self):
        df = _make_assay_df(["wild_type", "wild_type"])
        once = _apply_relabel(df)
        twice = _apply_relabel(once)
        pd.testing.assert_frame_equal(once, twice)

    def test_idempotent_l858r_count_unchanged(self):
        df = _make_assay_df(["L858R", "L858R", "wild_type"])
        result = _apply_relabel(df)
        assert (result["mutation_flag"] == "L858R").sum() == 3

    def test_already_correct_labels_survive(self):
        # If the column already has L858R for CHEMBL4380726, nothing changes.
        df = _make_assay_df(["L858R", "L858R"])
        result = _apply_relabel(df)
        assert (result["mutation_flag"] == "L858R").all()


@pytest.mark.unit
class TestPROTACHeavyAtomFilter:
    """The two oversized CHEMBL4380726 records have 74 heavy atoms and are
    legitimately removed by the max_heavy=70 cutoff in the cleaning pipeline."""

    def test_chembl4529558_has_74_heavy_atoms(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles(_PROTAC_SMILES["CHEMBL4529558"])
        assert mol is not None
        assert mol.GetNumHeavyAtoms() == 74

    def test_chembl4578319_has_74_heavy_atoms(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles(_PROTAC_SMILES["CHEMBL4578319"])
        assert mol is not None
        assert mol.GetNumHeavyAtoms() == 74

    def test_both_filtered_by_max_heavy_70(self):
        df = pd.DataFrame({"canonical_smiles": list(_PROTAC_SMILES.values())})
        result = filter_by_heavy_atom_count(df, max_heavy=70)
        assert len(result) == 0

    def test_chembl939_survivor_passes_filter(self):
        df = pd.DataFrame({"canonical_smiles": list(_SURVIVOR_SMILES.values())})
        result = filter_by_heavy_atom_count(df, max_heavy=70)
        assert len(result) == 1

    def test_mixed_batch_removes_only_oversized(self):
        all_smiles = list(_PROTAC_SMILES.values()) + list(_SURVIVOR_SMILES.values())
        df = pd.DataFrame({"canonical_smiles": all_smiles})
        result = filter_by_heavy_atom_count(df, max_heavy=70)
        assert len(result) == len(_SURVIVOR_SMILES)


# ── Integration tests ─────────────────────────────────────────────────────────

_CLEANED_PATH = Path("data/interim/egfr_cleaned.csv")


@pytest.fixture(scope="module")
def cleaned_df():
    if not _CLEANED_PATH.exists():
        pytest.skip("egfr_cleaned.csv not found — run clean_bioactivity_data.py first")
    return pd.read_csv(_CLEANED_PATH)


@pytest.mark.integration
class TestCleanedDatasetCounts:
    def test_l858r_count_is_22(self, cleaned_df):
        assert (cleaned_df["mutation_flag"] == "L858R").sum() == 22

    def test_no_chembl4380726_survivor_is_wild_type(self, cleaned_df):
        assay_rows = cleaned_df[cleaned_df["assay_chembl_id"] == "CHEMBL4380726"]
        assert len(assay_rows) > 0, "expected CHEMBL4380726 survivors in cleaned data"
        assert (assay_rows["mutation_flag"] == "wild_type").sum() == 0

    def test_all_chembl4380726_survivors_are_l858r(self, cleaned_df):
        assay_rows = cleaned_df[cleaned_df["assay_chembl_id"] == "CHEMBL4380726"]
        assert (assay_rows["mutation_flag"] == "L858R").all()

    def test_oversized_protac_records_are_absent(self, cleaned_df):
        oversized_ids = {"CHEMBL4529558", "CHEMBL4578319"}
        assert cleaned_df["molecule_chembl_id"].isin(oversized_ids).sum() == 0

    def test_relabel_is_idempotent_on_cleaned_output(self, cleaned_df):
        once = _apply_relabel(cleaned_df)
        twice = _apply_relabel(once)
        pd.testing.assert_series_equal(
            once["mutation_flag"].reset_index(drop=True),
            twice["mutation_flag"].reset_index(drop=True),
        )
