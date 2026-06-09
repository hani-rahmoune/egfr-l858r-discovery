from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

VALID_SMILES = [
    "CCO", "c1ccccc1", "CC(=O)Nc1ccc(O)cc1",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    "COc1cc2c(cc1OC)CC(C)NCC2",
    "Fc1ccc(cc1)C(=O)Nc2ccncc2",
    "c1ccc2c(c1)ccc(=O)o2",
    "CCCCc1cccc(c1)NC(=O)c2ccc(cc2)F",
    "CC(C)(C)c1ccc(cc1)C(=O)Nc2ccc(cc2)OC",
]


@pytest.fixture
def tiny_molecules_df():
    return pd.DataFrame({
        "smiles": VALID_SMILES,
        "molecule_chembl_id": [f"CHEMBL{i:06d}" for i in range(len(VALID_SMILES))],
        "activity_type": ["IC50"] * len(VALID_SMILES),
        "activity_value": [100, 500, 10, 5000, 1, 250, 50, 2000, 8, 400],
        "activity_units": ["nM"] * len(VALID_SMILES),
        "pchembl_value": [7.0, 6.3, 8.0, 5.3, 9.0, 6.6, 7.3, 5.7, 8.1, 6.4],
        "assay_description": [
            "EGFR L858R inhibition", "EGFR wild type inhibition",
            "EGFR L858R binding assay", "EGFR WT kinase assay",
            "EGFR L858R cellular IC50", "EGFR inhibition assay",
            "EGFR L858R biochemical", "EGFR wild-type kinase",
            "EGFR L858R assay", "EGFR wild type cellular",
        ],
        "mutation_flag": [
            "L858R", "wild_type", "L858R", "wild_type", "L858R",
            "unknown", "L858R", "wild_type", "L858R", "wild_type",
        ],
    })


@pytest.fixture
def tiny_cleaned_df(tiny_molecules_df):
    df = tiny_molecules_df.copy()
    df["canonical_smiles"] = df["smiles"]
    df["pic50"] = df["pchembl_value"]
    df["binary_label"] = (df["pic50"] >= 6.0).astype(int)
    return df