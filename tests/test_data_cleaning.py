from __future__ import annotations

import pandas as pd
import pytest

from src.data.cleaning import (
    deduplicate_by_smiles,
    filter_by_heavy_atom_count,
    is_valid_smiles,
    remove_invalid_smiles,
    run_full_cleaning_pipeline,
    standardize_smiles,
)


@pytest.mark.unit
class TestIsValidSmiles:
    def test_valid(self):
        assert is_valid_smiles("CCO") is True

    def test_empty(self):
        assert is_valid_smiles("") is False

    def test_none(self):
        assert is_valid_smiles(None) is False

    def test_garbage(self):
        assert is_valid_smiles("not_smiles") is False

    def test_unbalanced(self):
        assert is_valid_smiles("C(C") is False


@pytest.mark.unit
class TestStandardizeSmiles:
    def test_returns_string(self):
        assert isinstance(standardize_smiles("CCO"), str)

    def test_invalid_none(self):
        assert standardize_smiles("bad") is None

    def test_none_none(self):
        assert standardize_smiles(None) is None

    def test_salt_removed(self):
        result = standardize_smiles("CC(=O)[O-].[Na+]")
        assert result is not None and "[Na+]" not in result

    def test_canonical_deterministic(self):
        assert standardize_smiles("c1ccccc1") == standardize_smiles("C1=CC=CC=C1")


@pytest.mark.unit
class TestRemoveInvalidSmiles:
    def test_keeps_valid(self):
        df = pd.DataFrame({"smiles": ["CCO", "c1ccccc1"]})
        assert len(remove_invalid_smiles(df)) == 2

    def test_removes_invalid(self):
        df = pd.DataFrame({"smiles": ["CCO", "bad", None, ""]})
        assert len(remove_invalid_smiles(df)) == 1

    def test_all_invalid_empty(self):
        df = pd.DataFrame({"smiles": ["bad", None, ""]})
        assert len(remove_invalid_smiles(df)) == 0


@pytest.mark.unit
class TestDeduplicateBySmiles:
    def test_removes_dupes(self):
        df = pd.DataFrame(
            {"canonical_smiles": ["CCO", "CCO", "c1ccccc1"], "pic50": [7.0, 7.5, 6.0]}
        )
        assert len(deduplicate_by_smiles(df)) == 2

    def test_median(self):
        df = pd.DataFrame({"canonical_smiles": ["CCO", "CCO"], "pic50": [7.0, 8.0]})
        assert deduplicate_by_smiles(df)["pic50"].iloc[0] == pytest.approx(7.5)


@pytest.mark.unit
class TestFilterByHeavyAtomCount:
    def test_removes_small(self):
        df = pd.DataFrame({"canonical_smiles": ["CO", "CCO"]})
        assert len(filter_by_heavy_atom_count(df, min_heavy=5)) == 0

    def test_keeps_drug_like(self):
        df = pd.DataFrame({"canonical_smiles": ["CC(=O)Nc1ccc(O)cc1"]})
        assert len(filter_by_heavy_atom_count(df, min_heavy=7)) == 1


@pytest.mark.unit
class TestFullPipeline:
    def test_runs(self, tiny_molecules_df):
        result = run_full_cleaning_pipeline(tiny_molecules_df)
        assert len(result) > 0
        assert "canonical_smiles" in result.columns

    def test_output_leq_input(self, tiny_molecules_df):
        assert len(run_full_cleaning_pipeline(tiny_molecules_df)) <= len(
            tiny_molecules_df
        )
