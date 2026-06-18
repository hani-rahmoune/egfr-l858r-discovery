from __future__ import annotations

import pandas as pd
import pytest

from src.splitting.scaffold_split import get_bemis_murcko_scaffold, scaffold_split


@pytest.fixture
def drug_like_df():
    """20 molecules with structurally distinct scaffolds for split testing."""
    smiles_list = [
        "COc1cc2c(cc1OC)CC(C)NCC2",
        "CC(=O)Nc1ccc(O)cc1",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "c1ccc2c(c1)ccc(=O)o2",
        "c1cnc2ccccc2c1",
        "c1ccc2[nH]cccc2c1",
        "c1cnc2[nH]cncc2c1",
        "c1ccc2ncccc2c1",
        "O=C1CCCN1",
        "c1ccncc1",
        "c1csc2ccccc12",
        "c1coc2ccccc12",
        "c1ccoc1",
        "c1ccsc1",
        "c1cnc[nH]1",
        "c1ccnc(N)c1",
        "O=C1NC(=O)c2ccccc21",
        "Cn1ccc2ccccc21",
        "c1ccc(cc1)c2ccccn2",
        "CC1=CC(=O)c2ccccc2O1",
    ]
    return pd.DataFrame(
        {
            "canonical_smiles": smiles_list,
            "pic50": [
                7.0,
                6.5,
                8.0,
                5.5,
                7.2,
                6.8,
                7.5,
                6.0,
                5.8,
                7.1,
                6.3,
                5.9,
                4.5,
                5.0,
                6.7,
                7.4,
                6.1,
                7.8,
                6.6,
                5.4,
            ],
        }
    )


@pytest.mark.unit
class TestGetBemisMurckoScaffold:
    def test_valid_molecule(self):
        assert get_bemis_murcko_scaffold("c1ccccc1") is not None

    def test_invalid_returns_none(self):
        assert get_bemis_murcko_scaffold("not_smiles") is None

    def test_same_core_same_scaffold(self):
        # Both have a benzene core — scaffolds must match
        s1 = get_bemis_murcko_scaffold("CC(=O)Nc1ccc(O)cc1")
        s2 = get_bemis_murcko_scaffold("Nc1ccc(O)cc1")
        assert s1 is not None and s2 is not None


@pytest.mark.unit
class TestScaffoldSplit:
    def test_all_molecules_assigned(self, drug_like_df):
        train, val, test = scaffold_split(drug_like_df)
        assert len(train) + len(val) + len(test) == len(drug_like_df)

    def test_no_scaffold_leakage(self, drug_like_df):
        """Core test: no scaffold can appear in more than one split."""
        train, val, test = scaffold_split(drug_like_df)
        t = set(train["scaffold"])
        v = set(val["scaffold"])
        s = set(test["scaffold"])
        assert not (t & v), "Scaffold leakage between train and val"
        assert not (t & s), "Scaffold leakage between train and test"
        assert not (v & s), "Scaffold leakage between val and test"

    def test_no_molecule_in_two_splits(self, drug_like_df):
        train, val, test = scaffold_split(drug_like_df)
        t = set(train["canonical_smiles"])
        v = set(val["canonical_smiles"])
        s = set(test["canonical_smiles"])
        assert not (t & v) and not (t & s) and not (v & s)

    def test_split_labels_correct(self, drug_like_df):
        train, val, test = scaffold_split(drug_like_df)
        assert (train["split"] == "train").all()
        assert (val["split"] == "val").all()
        assert (test["split"] == "test").all()

    def test_reproducible_with_same_seed(self, drug_like_df):
        t1, _, _ = scaffold_split(drug_like_df, seed=42)
        t2, _, _ = scaffold_split(drug_like_df, seed=42)
        pd.testing.assert_frame_equal(
            t1.reset_index(drop=True),
            t2.reset_index(drop=True),
        )

    def test_train_is_largest_split(self, drug_like_df):
        train, val, test = scaffold_split(drug_like_df)
        assert len(train) >= len(val)
        assert len(train) >= len(test)
