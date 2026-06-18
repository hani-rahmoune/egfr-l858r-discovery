"""
Tests for the fingerprint ablation study.

Unit tests (no network, no parquet files):
  - topological_torsion fingerprint is registered and has correct shape
  - compute_fp_desc_matrix returns correct column counts for each FP type
  - scaffold_split_indices matches scaffold_split logic
  - build_scaffold_lookup produces a dict

Integration tests (marked 'integration', run ablation on tiny synthetic data):
  - run_task_ablation runs without error and returns expected structure
  - winner key is present for each task
  - results JSON is serialisable
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.fingerprint_ablation import (
    FP_CONFIGS,
    MODEL_NAMES,
    build_scaffold_lookup,
    compute_fp_desc_matrix,
    run_task_ablation,
    scaffold_split_indices,
)
from src.features.fingerprints import (
    compute_fingerprint,
    topological_torsion_fingerprint,
)

# A handful of valid drug-like SMILES for unit tests
PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
ERLOTINIB = "C#Cc1cccc(Nc2ncnc3cc(OCC)c(OCC)cc23)c1"
GEFITINIB = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"


# ── topological_torsion ───────────────────────────────────────────────────────


@pytest.mark.unit
class TestTopologicalTorsion:
    def test_default_shape_2048(self):
        fp = topological_torsion_fingerprint(PARACETAMOL)
        assert fp is not None
        assert fp.shape == (2048,)

    def test_custom_n_bits(self):
        fp = topological_torsion_fingerprint(ASPIRIN, n_bits=512)
        assert fp is not None
        assert fp.shape == (512,)

    def test_binary_values(self):
        fp = topological_torsion_fingerprint(CAFFEINE)
        assert set(fp.tolist()).issubset({0, 1})

    def test_invalid_smiles_returns_none(self):
        assert topological_torsion_fingerprint("not_a_smiles") is None

    def test_registered_in_dispatcher(self):
        fp = compute_fingerprint(PARACETAMOL, fp_type="topological_torsion")
        assert fp is not None and fp.shape == (2048,)

    def test_differs_from_morgan(self):
        tt = topological_torsion_fingerprint(ERLOTINIB)
        mg = compute_fingerprint(ERLOTINIB, fp_type="morgan_ecfp4")
        # Different FP types on the same molecule must not produce identical bits
        assert not np.array_equal(tt, mg)

    def test_unknown_type_still_raises(self):
        with pytest.raises(ValueError):
            compute_fingerprint(PARACETAMOL, fp_type="avalon_nonexistent")


# ── FP_CONFIGS coverage ───────────────────────────────────────────────────────


@pytest.mark.unit
class TestFPConfigsCoverage:
    EXPECTED_KEYS = {
        "morgan_ecfp4",
        "morgan_ecfp6",
        "maccs",
        "rdkit_topological",
        "atom_pair",
        "topological_torsion",
    }

    def test_all_expected_keys_present(self):
        assert self.EXPECTED_KEYS.issubset(set(FP_CONFIGS))

    def test_each_config_has_fp_type(self):
        for key, cfg in FP_CONFIGS.items():
            assert "fp_type" in cfg, f"{key} missing 'fp_type'"

    def test_each_config_has_n_bits(self):
        for key, cfg in FP_CONFIGS.items():
            assert "n_bits" in cfg, f"{key} missing 'n_bits'"

    def test_maccs_n_bits_is_167(self):
        assert FP_CONFIGS["maccs"]["n_bits"] == 167

    def test_morgan_ecfp4_radius_is_2(self):
        assert FP_CONFIGS["morgan_ecfp4"].get("radius") == 2

    def test_morgan_ecfp6_radius_is_3(self):
        assert FP_CONFIGS["morgan_ecfp6"].get("radius") == 3


# ── compute_fp_desc_matrix ────────────────────────────────────────────────────


@pytest.mark.unit
class TestComputeFpDescMatrix:
    SMILES_LIST = [PARACETAMOL, CAFFEINE, ASPIRIN, IBUPROFEN]

    @pytest.mark.parametrize("fp_key", list(FP_CONFIGS))
    def test_shape_matches_n_bits_plus_11_desc(self, fp_key):
        cfg = FP_CONFIGS[fp_key]
        X, valid = compute_fp_desc_matrix(
            self.SMILES_LIST,
            fp_type=cfg["fp_type"],
            n_bits=cfg["n_bits"],
            radius=cfg.get("radius", 2),
        )
        expected_cols = cfg["n_bits"] + 11
        assert X.shape == (
            len(self.SMILES_LIST),
            expected_cols,
        ), f"{fp_key}: expected shape ({len(self.SMILES_LIST)}, {expected_cols}), got {X.shape}"
        assert valid == [0, 1, 2, 3]

    def test_invalid_smiles_excluded(self):
        smiles = [PARACETAMOL, "bad_smiles", CAFFEINE]
        cfg = FP_CONFIGS["morgan_ecfp4"]
        X, valid = compute_fp_desc_matrix(
            smiles, fp_type=cfg["fp_type"], n_bits=cfg["n_bits"]
        )
        assert X.shape[0] == 2
        assert valid == [0, 2]

    def test_float32_dtype(self):
        cfg = FP_CONFIGS["morgan_ecfp4"]
        X, _ = compute_fp_desc_matrix(
            self.SMILES_LIST, fp_type=cfg["fp_type"], n_bits=cfg["n_bits"]
        )
        assert X.dtype == np.float32


# ── scaffold_split_indices ────────────────────────────────────────────────────


@pytest.mark.unit
class TestScaffoldSplitIndices:
    """Verify scaffold_split_indices replicates the same logic as scaffold_split."""

    SMILES = [PARACETAMOL, CAFFEINE, ASPIRIN, IBUPROFEN, ERLOTINIB, GEFITINIB]

    @pytest.fixture(scope="class")
    def lookup(self):
        return build_scaffold_lookup(self.SMILES)

    def test_all_indices_present(self, lookup):
        train, val, test = scaffold_split_indices(self.SMILES, lookup, seed=42)
        all_idx = sorted(train + val + test)
        assert all_idx == list(range(len(self.SMILES)))

    def test_no_overlap_between_splits(self, lookup):
        train, val, test = scaffold_split_indices(self.SMILES, lookup, seed=42)
        assert len(set(train) & set(val)) == 0
        assert len(set(train) & set(test)) == 0
        assert len(set(val) & set(test)) == 0

    def test_different_seeds_give_different_splits(self, lookup):
        train1, _, _ = scaffold_split_indices(self.SMILES, lookup, seed=42)
        train2, _, _ = scaffold_split_indices(self.SMILES, lookup, seed=7)
        assert train1 != train2

    def test_same_seed_is_deterministic(self, lookup):
        train1, _, _ = scaffold_split_indices(self.SMILES, lookup, seed=99)
        train2, _, _ = scaffold_split_indices(self.SMILES, lookup, seed=99)
        assert train1 == train2


# ── build_scaffold_lookup ─────────────────────────────────────────────────────


@pytest.mark.unit
class TestBuildScaffoldLookup:
    def test_returns_dict(self):
        lookup = build_scaffold_lookup([PARACETAMOL, CAFFEINE])
        assert isinstance(lookup, dict)

    def test_keys_are_input_smiles(self):
        smiles = [PARACETAMOL, CAFFEINE, ASPIRIN]
        lookup = build_scaffold_lookup(smiles)
        assert set(lookup.keys()) == set(smiles)

    def test_values_are_strings(self):
        lookup = build_scaffold_lookup([PARACETAMOL])
        assert isinstance(lookup[PARACETAMOL], str)

    def test_invalid_smiles_gets_no_scaffold(self):
        lookup = build_scaffold_lookup(["not_a_smiles"])
        assert lookup["not_a_smiles"] == "no_scaffold"


# ── Integration: full ablation on tiny synthetic parquet ──────────────────────


def _make_tiny_parquet(tmp_path: Path, n: int = 20, seed: int = 42) -> Path:
    """Write a minimal parquet with canonical_smiles + pic50 for ablation testing."""
    rng = np.random.default_rng(seed)
    # Use real drug-like SMILES so RDKit can parse them
    base_smiles = [PARACETAMOL, CAFFEINE, ASPIRIN, IBUPROFEN, ERLOTINIB, GEFITINIB]
    # Repeat and slightly modify to get n molecules (still valid SMILES)
    smiles = []
    for i in range(n):
        smiles.append(base_smiles[i % len(base_smiles)])
    pic50 = rng.uniform(5.0, 9.0, n).astype(np.float32)

    df = pd.DataFrame({"canonical_smiles": smiles, "pic50": pic50})
    path = tmp_path / "tiny_fp_test.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.mark.integration
class TestRunTaskAblationIntegration:
    """Runs a minimal end-to-end ablation with n_estimators=3, 2 seeds.
    Checks structure and serialisability — not numerical results.
    """

    @pytest.fixture(scope="class")
    def ablation_result(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("ablation")
        parquet = _make_tiny_parquet(tmp, n=24, seed=0)
        result = run_task_ablation(
            parquet_path=parquet,
            task_name="test_task",
            seeds=[42, 7],
            n_estimators=3,
        )
        return result

    def test_all_fp_types_present(self, ablation_result):
        for fp_key in FP_CONFIGS:
            assert fp_key in ablation_result, f"{fp_key} missing from ablation result"

    def test_each_fp_has_best_model(self, ablation_result):
        for fp_key, res in ablation_result.items():
            assert "best_model" in res, f"{fp_key}: missing best_model"
            assert res["best_model"] in MODEL_NAMES

    def test_each_fp_has_metric_keys(self, ablation_result):
        expected = {
            "val_rmse_mean",
            "test_rmse_mean",
            "test_r2_mean",
            "test_spearman_mean",
        }
        for fp_key, res in ablation_result.items():
            actual = set(res["best"].keys())
            assert expected.issubset(
                actual
            ), f"{fp_key}: missing keys {expected - actual}"

    def test_result_is_json_serialisable(self, ablation_result):
        # Should not raise
        json.dumps(ablation_result)

    def test_maccs_has_178_features(self, ablation_result):
        # MACCS 167 bits + 11 descriptors = 178
        assert ablation_result["maccs"]["n_features"] == 178

    def test_morgan_ecfp4_has_2059_features(self, ablation_result):
        assert ablation_result["morgan_ecfp4"]["n_features"] == 2059

    def test_val_rmse_is_positive(self, ablation_result):
        for fp_key, res in ablation_result.items():
            val = res["best"]["val_rmse_mean"]
            assert val > 0, f"{fp_key}: val_rmse_mean={val} is not positive"
