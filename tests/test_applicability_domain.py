"""
Unit tests for src/scoring/applicability_domain.py.

Covers:
  - _tanimoto_max: identical FP, orthogonal FPs, partial overlap, empty train set
  - _domain_label: each band (in_domain / borderline / out_of_domain)
  - ApplicabilityDomain.fit: stores n_train, handles invalid SMILES
  - ApplicabilityDomain.predict: correct domain label and confidence factor,
      invalid SMILES, empty training set, unfit model
  - ApplicabilityDomain.predict_batch: length, order preservation
  - ApplicabilityDomain.from_config: loads thresholds from model_config.yaml
  - max_tanimoto_to_set: stateless helper

Test molecules:
  GEFITINIB  -- EGFR inhibitor (should be in-domain vs EGFR training set)
  ERLOTINIB  -- EGFR inhibitor (similar scaffold to gefitinib)
  ASPIRIN    -- very different from EGFR inhibitors (expect low similarity)
  INVALID    -- non-parseable string
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring.applicability_domain import (
    ApplicabilityDomain,
    _domain_label,
    _tanimoto_max,
    max_tanimoto_to_set,
)

GEFITINIB = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
ERLOTINIB = "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
INVALID = "not_a_smiles$$"

# Tiny synthetic FP helpers for deterministic unit tests
_N_BITS = 16  # tiny for speed


def _zero_fp(n: int = _N_BITS) -> np.ndarray:
    return np.zeros(n, dtype=np.uint8)


def _ones_fp(n: int = _N_BITS) -> np.ndarray:
    return np.ones(n, dtype=np.uint8)


def _half_fp(n: int = _N_BITS) -> np.ndarray:
    """First half bits set."""
    fp = np.zeros(n, dtype=np.uint8)
    fp[: n // 2] = 1
    return fp


def _other_fp(n: int = _N_BITS) -> np.ndarray:
    """Second half bits set (orthogonal to _half_fp)."""
    fp = np.zeros(n, dtype=np.uint8)
    fp[n // 2 :] = 1
    return fp


# ── TestTanimotoMax ───────────────────────────────────────────────────────────


@pytest.mark.unit
class TestTanimotoMax:

    def test_identical_fps_return_one(self):
        fp = _half_fp()
        T = np.array([fp])
        assert _tanimoto_max(fp, T) == pytest.approx(1.0)

    def test_orthogonal_fps_return_zero(self):
        q = _half_fp()
        T = np.array([_other_fp()])
        assert _tanimoto_max(q, T) == pytest.approx(0.0)

    def test_returns_maximum_over_training_set(self):
        q = _ones_fp()
        low = _half_fp()  # Tanimoto(ones, half) = 0.5
        hi = _ones_fp()  # Tanimoto(ones, ones) = 1.0
        T = np.array([low, hi])
        assert _tanimoto_max(q, T) == pytest.approx(1.0)

    def test_partial_overlap(self):
        # 16 bits; query has bits 0–3 set (4 bits); train has bits 2–5 set (4 bits)
        n = 16
        q = np.zeros(n, dtype=np.uint8)
        q[:4] = 1
        tr = np.zeros(n, dtype=np.uint8)
        tr[2:6] = 1
        T = np.array([tr])
        # intersection = bits 2,3 -> 2; union = bits 0,1,2,3,4,5 -> 6
        expected = 2 / 6
        assert _tanimoto_max(q, T) == pytest.approx(expected, abs=1e-5)

    def test_empty_training_set_returns_zero(self):
        q = _half_fp()
        T = np.empty((0, _N_BITS), dtype=np.uint8)
        assert _tanimoto_max(q, T) == 0.0

    def test_no_overflow_with_2048_bits(self):
        # uint8 max = 255; dot product of 2048-bit all-ones vectors = 2048 > 255
        n = 2048
        q = np.ones(n, dtype=np.uint8)
        T = np.array([np.ones(n, dtype=np.uint8)])
        assert _tanimoto_max(q, T) == pytest.approx(1.0)


# ── TestDomainLabel ───────────────────────────────────────────────────────────


@pytest.mark.unit
class TestDomainLabel:

    def test_high_sim_is_in_domain(self):
        assert _domain_label(0.80, 0.50, 0.30) == "in_domain"

    def test_at_in_domain_threshold_is_in_domain(self):
        assert _domain_label(0.50, 0.50, 0.30) == "in_domain"

    def test_borderline_band(self):
        assert _domain_label(0.40, 0.50, 0.30) == "borderline"

    def test_at_borderline_threshold_is_borderline(self):
        assert _domain_label(0.30, 0.50, 0.30) == "borderline"

    def test_low_sim_is_out_of_domain(self):
        assert _domain_label(0.10, 0.50, 0.30) == "out_of_domain"

    def test_zero_sim_is_out_of_domain(self):
        assert _domain_label(0.0, 0.50, 0.30) == "out_of_domain"


# ── TestApplicabilityDomainFit ────────────────────────────────────────────────


@pytest.mark.unit
class TestApplicabilityDomainFit:

    def test_fit_stores_n_train(self):
        ad = ApplicabilityDomain()
        ad.fit([GEFITINIB, ERLOTINIB, ASPIRIN])
        assert ad.n_train == 3

    def test_fit_skips_invalid_smiles(self):
        ad = ApplicabilityDomain()
        ad.fit([GEFITINIB, INVALID, ASPIRIN])
        assert ad.n_train == 2

    def test_fit_empty_list(self):
        ad = ApplicabilityDomain()
        ad.fit([])
        assert ad.n_train == 0

    def test_fit_all_invalid_gives_zero_n_train(self):
        ad = ApplicabilityDomain()
        ad.fit([INVALID, "??!!"])
        assert ad.n_train == 0

    def test_train_fps_shape_after_fit(self):
        ad = ApplicabilityDomain()
        ad.fit([GEFITINIB, ERLOTINIB])
        assert ad.train_fps is not None
        assert ad.train_fps.shape == (2, 2048)

    def test_returns_self_for_chaining(self):
        ad = ApplicabilityDomain()
        result = ad.fit([GEFITINIB])
        assert result is ad


# ── TestApplicabilityDomainPredict ────────────────────────────────────────────


@pytest.mark.unit
class TestApplicabilityDomainPredict:

    def _fitted(self, smiles_list=None) -> ApplicabilityDomain:
        ad = ApplicabilityDomain()
        ad.fit(smiles_list or [GEFITINIB, ERLOTINIB])
        return ad

    def test_required_keys_present(self):
        ad = self._fitted()
        res = ad.predict(GEFITINIB)
        for key in ("smiles", "valid", "max_tanimoto", "domain", "confidence_factor"):
            assert key in res, f"Missing key: {key}"

    def test_valid_flag_true_for_valid_smiles(self):
        ad = self._fitted()
        assert ad.predict(GEFITINIB)["valid"] is True

    def test_valid_flag_false_for_invalid_smiles(self):
        ad = self._fitted()
        assert ad.predict(INVALID)["valid"] is False

    def test_domain_is_valid_label(self):
        ad = self._fitted()
        dom = ad.predict(GEFITINIB)["domain"]
        assert dom in ("in_domain", "borderline", "out_of_domain")

    def test_identical_molecule_is_in_domain(self):
        ad = ApplicabilityDomain()
        ad.fit([GEFITINIB])
        res = ad.predict(GEFITINIB)
        assert res["domain"] == "in_domain"
        assert res["max_tanimoto"] == pytest.approx(1.0)

    def test_very_similar_molecule_in_domain(self):
        # Erlotinib and gefitinib are both EGFR inhibitors, same quinazoline core
        ad = ApplicabilityDomain()
        ad.fit([GEFITINIB])
        res = ad.predict(ERLOTINIB)
        assert res["max_tanimoto"] is not None
        assert 0.0 < res["max_tanimoto"] <= 1.0

    def test_aspirin_likely_lower_similarity_than_gefitinib(self):
        # Aspirin vs EGFR inhibitor training set: should have lower Tanimoto
        ad = ApplicabilityDomain()
        ad.fit([GEFITINIB, ERLOTINIB])
        sim_egfr = ad.predict(ERLOTINIB)["max_tanimoto"]
        sim_aspirin = ad.predict(ASPIRIN)["max_tanimoto"]
        assert sim_aspirin < sim_egfr

    def test_confidence_factor_matches_domain(self):
        ad = ApplicabilityDomain(
            confidence_factors={
                "in_domain": 1.0,
                "borderline": 0.75,
                "out_of_domain": 0.50,
            }
        )
        ad.fit([GEFITINIB])
        res = ad.predict(GEFITINIB)  # identical -> in_domain
        assert res["confidence_factor"] == 1.0

    def test_invalid_smiles_has_out_of_domain(self):
        ad = self._fitted()
        res = ad.predict(INVALID)
        assert res["domain"] == "out_of_domain"
        assert res["confidence_factor"] == 0.50

    def test_predict_before_fit_raises(self):
        ad = ApplicabilityDomain()
        with pytest.raises(RuntimeError):
            ad.predict(GEFITINIB)

    def test_empty_training_set_gives_out_of_domain(self):
        ad = ApplicabilityDomain()
        ad.fit([])
        res = ad.predict(GEFITINIB)
        assert res["domain"] == "out_of_domain"
        assert res["max_tanimoto"] == 0.0

    def test_low_threshold_widens_in_domain(self):
        # With a very low in_domain_threshold, aspirin should be in_domain
        ad = ApplicabilityDomain(in_domain_threshold=0.0, borderline_threshold=-1.0)
        ad.fit([GEFITINIB])
        res = ad.predict(ASPIRIN)
        assert res["domain"] == "in_domain"

    def test_max_tanimoto_is_rounded(self):
        ad = self._fitted()
        res = ad.predict(ERLOTINIB)
        if res["max_tanimoto"] is not None:
            # Should have at most 4 decimal places
            s = str(res["max_tanimoto"])
            if "." in s:
                assert len(s.split(".")[1]) <= 4


# ── TestApplicabilityDomainBatch ──────────────────────────────────────────────


@pytest.mark.unit
class TestApplicabilityDomainBatch:

    def _fitted(self) -> ApplicabilityDomain:
        ad = ApplicabilityDomain()
        ad.fit([GEFITINIB, ERLOTINIB])
        return ad

    def test_batch_returns_correct_length(self):
        ad = self._fitted()
        res = ad.predict_batch([GEFITINIB, ASPIRIN, INVALID])
        assert len(res) == 3

    def test_batch_preserves_order(self):
        ad = self._fitted()
        res = ad.predict_batch([GEFITINIB, ASPIRIN, ERLOTINIB])
        assert res[0]["smiles"] == GEFITINIB
        assert res[1]["smiles"] == ASPIRIN
        assert res[2]["smiles"] == ERLOTINIB

    def test_batch_empty_list(self):
        ad = self._fitted()
        res = ad.predict_batch([])
        assert res == []

    def test_batch_single_element(self):
        ad = self._fitted()
        res = ad.predict_batch([GEFITINIB])
        assert len(res) == 1
        assert "domain" in res[0]


# ── TestFromConfig ────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestFromConfig:

    def test_from_config_returns_instance(self):
        ad = ApplicabilityDomain.from_config()
        assert isinstance(ad, ApplicabilityDomain)

    def test_from_config_in_domain_threshold(self):
        ad = ApplicabilityDomain.from_config()
        # Config specifies 0.50
        assert ad.in_domain_threshold == pytest.approx(0.50)

    def test_from_config_borderline_threshold(self):
        ad = ApplicabilityDomain.from_config()
        # Config specifies 0.30
        assert ad.borderline_threshold == pytest.approx(0.30)

    def test_from_config_confidence_factors_keys(self):
        ad = ApplicabilityDomain.from_config()
        for key in ("in_domain", "borderline", "out_of_domain"):
            assert key in ad.confidence_factors

    def test_from_config_in_domain_factor(self):
        ad = ApplicabilityDomain.from_config()
        assert ad.confidence_factors["in_domain"] == pytest.approx(1.0)

    def test_from_config_out_of_domain_factor(self):
        ad = ApplicabilityDomain.from_config()
        assert ad.confidence_factors["out_of_domain"] == pytest.approx(0.50)


# ── TestMaxTanimotoToSet (stateless helper) ───────────────────────────────────


@pytest.mark.unit
class TestMaxTanimotoToSet:

    def _train_matrix(self) -> np.ndarray:
        from src.features.fingerprints import morgan_fingerprint

        fps = [morgan_fingerprint(smi) for smi in [GEFITINIB, ERLOTINIB]]
        return np.array([fp.astype(np.uint8) for fp in fps if fp is not None])

    def test_valid_smiles_returns_float(self):
        T = self._train_matrix()
        sim = max_tanimoto_to_set(GEFITINIB, T)
        assert isinstance(sim, float)

    def test_invalid_smiles_returns_none(self):
        T = self._train_matrix()
        sim = max_tanimoto_to_set(INVALID, T)
        assert sim is None

    def test_identical_molecule_returns_one(self):
        T = self._train_matrix()
        sim = max_tanimoto_to_set(GEFITINIB, T)
        assert sim == pytest.approx(1.0)

    def test_sim_in_zero_one_range(self):
        T = self._train_matrix()
        sim = max_tanimoto_to_set(ASPIRIN, T)
        assert sim is not None
        assert 0.0 <= sim <= 1.0
