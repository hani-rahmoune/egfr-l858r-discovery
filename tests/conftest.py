from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# CI artifact guard (per-artifact)
# ---------------------------------------------------------------------------
# Some tests are marked @unit but assert that a real pipeline artifact exists.
# Two kinds of artifact differ in whether they reach CI:
#   * gitignored outputs (data/generated/*.csv, *.pkl) are ABSENT in CI
#   * tracked JSON results (models/**/*.json) are PRESENT in CI
# A single tracked sentinel cannot tell the two apart, so each test below is
# mapped to the specific file it reads.  A test is skipped only when ITS file
# is missing.  In CI the CSV-backed tests skip (csv gitignored) while the
# JSON-backed tests still run (json tracked); locally every file is present so
# nothing is skipped.
_ROOT = Path(__file__).resolve().parent.parent

# test name -> the artifact file it depends on (relative to repo root)
_TEST_ARTIFACTS: dict[str, str] = {
    # gitignored CSV: absent in CI, present locally after rank_candidates.py
    "test_final_ranking_loads": "data/generated/final_ranked_candidates.csv",
    "test_final_ranking_has_both_sources": "data/generated/final_ranked_candidates.csv",
    "test_lookup_ranking_known": "data/generated/final_ranked_candidates.csv",
    "test_lookup_ranking_generated": "data/generated/final_ranked_candidates.csv",
    "test_lookup_ranking_returns_smiles": "data/generated/final_ranked_candidates.csv",
    # tracked JSON: present in CI, kept under the guard for local robustness
    "test_qsar_metrics": "models/qsar/general/metadata.json",
    "test_fingerprint_ablation": "models/qsar/fingerprint_ablation_results.json",
    "test_model3_verdict": "models/qsar/l858r/loocv_results.json",
    "test_model4_verdict": "models/qsar/selectivity/selectivity_results.json",
    "test_docking_noise": "models/qsar/docking_noise_results.json",
    "test_sanity_check": "models/qsar/sanity_check_docking.json",
    "test_generated_docking": "models/generator/generated_docking_results.json",
    "test_rl_results": "models/generator/rl_results.json",
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        rel = _TEST_ARTIFACTS.get(item.name)
        if rel is not None and not (_ROOT / rel).exists():
            item.add_marker(
                pytest.mark.skip(
                    reason=f"artifact absent ({rel}), run the pipeline first"
                )
            )


VALID_SMILES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)Nc1ccc(O)cc1",
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
    return pd.DataFrame(
        {
            "smiles": VALID_SMILES,
            "molecule_chembl_id": [f"CHEMBL{i:06d}" for i in range(len(VALID_SMILES))],
            "activity_type": ["IC50"] * len(VALID_SMILES),
            "activity_value": [100, 500, 10, 5000, 1, 250, 50, 2000, 8, 400],
            "activity_units": ["nM"] * len(VALID_SMILES),
            "pchembl_value": [7.0, 6.3, 8.0, 5.3, 9.0, 6.6, 7.3, 5.7, 8.1, 6.4],
            "assay_description": [
                "EGFR L858R inhibition",
                "EGFR wild type inhibition",
                "EGFR L858R binding assay",
                "EGFR WT kinase assay",
                "EGFR L858R cellular IC50",
                "EGFR inhibition assay",
                "EGFR L858R biochemical",
                "EGFR wild-type kinase",
                "EGFR L858R assay",
                "EGFR wild type cellular",
            ],
            "mutation_flag": [
                "L858R",
                "wild_type",
                "L858R",
                "wild_type",
                "L858R",
                "unknown",
                "L858R",
                "wild_type",
                "L858R",
                "wild_type",
            ],
        }
    )


@pytest.fixture
def tiny_cleaned_df(tiny_molecules_df):
    df = tiny_molecules_df.copy()
    df["canonical_smiles"] = df["smiles"]
    df["pic50"] = df["pchembl_value"]
    df["binary_label"] = (df["pic50"] >= 6.0).astype(int)
    return df
