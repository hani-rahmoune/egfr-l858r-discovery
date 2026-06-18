from __future__ import annotations

import numpy as np
import pytest

from src.features.descriptors import (
    DESCRIPTOR_NAMES,
    check_lipinski,
    check_veber,
    compute_descriptor_matrix,
    compute_descriptors,
)
from src.features.fingerprints import (
    compute_fingerprint,
    compute_fingerprint_matrix,
    maccs_fingerprint,
    morgan_fingerprint,
)

VALID = "CC(=O)Nc1ccc(O)cc1"  # paracetamol
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
INVALID = "not_smiles"


@pytest.mark.unit
class TestMorganFingerprint:
    def test_default_shape(self):
        assert morgan_fingerprint(VALID).shape == (2048,)

    def test_custom_length(self):
        assert morgan_fingerprint(VALID, n_bits=1024).shape == (1024,)

    def test_ecfp4_ecfp6_differ(self):
        # Different radii must produce different bit patterns for complex molecules
        assert not np.array_equal(
            morgan_fingerprint(CAFFEINE, radius=2),
            morgan_fingerprint(CAFFEINE, radius=3),
        )

    def test_invalid_returns_none(self):
        assert morgan_fingerprint(INVALID) is None

    def test_binary_values_only(self):
        assert set(morgan_fingerprint(VALID).tolist()).issubset({0, 1})


@pytest.mark.unit
class TestMaccsFingerprint:
    def test_always_167_bits(self):
        assert maccs_fingerprint(VALID).shape == (167,)

    def test_invalid_returns_none(self):
        assert maccs_fingerprint(INVALID) is None


@pytest.mark.unit
class TestComputeFingerprint:
    @pytest.mark.parametrize(
        "fp_type,expected_len",
        [
            ("morgan_ecfp4", 2048),
            ("morgan_ecfp6", 2048),
            ("maccs", 167),
            ("rdkit_topological", 2048),
            ("atom_pair", 2048),
        ],
    )
    def test_all_types_correct_shape(self, fp_type, expected_len):
        fp = compute_fingerprint(VALID, fp_type=fp_type)
        assert fp is not None and fp.shape == (expected_len,)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            compute_fingerprint(VALID, fp_type="unknown")


@pytest.mark.unit
class TestFingerprintMatrix:
    def test_shape(self):
        matrix, idx = compute_fingerprint_matrix([VALID, CAFFEINE])
        assert matrix.shape == (2, 2048) and idx == [0, 1]

    def test_invalid_excluded_correct_indices(self):
        # Index 1 is invalid, so valid_indices should be [0, 2]
        matrix, idx = compute_fingerprint_matrix([VALID, INVALID, CAFFEINE])
        assert matrix.shape[0] == 2 and idx == [0, 2]

    def test_all_invalid_returns_empty(self):
        matrix, idx = compute_fingerprint_matrix([INVALID, INVALID])
        assert matrix.shape[0] == 0 and idx == []


@pytest.mark.unit
class TestDescriptors:
    def test_has_all_keys(self):
        desc = compute_descriptors(VALID)
        for k in DESCRIPTOR_NAMES:
            assert k in desc

    def test_invalid_returns_none(self):
        assert compute_descriptors(INVALID) is None

    def test_qed_between_0_and_1(self):
        assert 0.0 <= compute_descriptors(VALID)["qed"] <= 1.0

    def test_mw_positive(self):
        assert compute_descriptors(VALID)["mol_weight"] > 0


@pytest.mark.unit
class TestDescriptorMatrix:
    def test_shape(self):
        matrix, idx = compute_descriptor_matrix([VALID, CAFFEINE])
        assert matrix.shape == (2, len(DESCRIPTOR_NAMES)) and idx == [0, 1]

    def test_invalid_excluded(self):
        matrix, idx = compute_descriptor_matrix([VALID, INVALID])
        assert matrix.shape[0] == 1 and idx == [0]


@pytest.mark.unit
class TestLipinskiVeber:
    def test_paracetamol_passes_lipinski(self):
        result = check_lipinski(VALID)
        assert result["lipinski_pass"] is True and result["violations"] == 0

    def test_invalid_fails_lipinski(self):
        assert check_lipinski(INVALID)["lipinski_pass"] is False

    def test_paracetamol_passes_veber(self):
        assert check_veber(VALID)["veber_pass"] is True
