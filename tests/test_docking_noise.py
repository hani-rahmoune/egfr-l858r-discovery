"""
Tests for scripts/eval_docking_noise.py.

Unit tests cover compute_noise_stats and classify_call — the two functions
that implement the noise-propagation and confidence-classification logic.
Integration tests require library_docking_results.json from a completed
dock_library.py run.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_docking_noise import (
    classify_call,
    compute_noise_stats,
    load_top_compounds,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

LIBRARY_RESULTS = PROJECT_ROOT / "models" / "qsar" / "library_docking_results.json"
_HAS_LIBRARY = LIBRARY_RESULTS.exists()


# ── TestComputeNoiseStats ─────────────────────────────────────────────────────


class TestComputeNoiseStats:

    @pytest.mark.unit
    def test_returns_none_when_l858r_empty(self):
        result = compute_noise_stats([], [-7.0, -7.1, -7.2])
        assert result is None

    @pytest.mark.unit
    def test_returns_none_when_wt_empty(self):
        result = compute_noise_stats([-7.5, -7.6], [])
        assert result is None

    @pytest.mark.unit
    def test_returns_none_when_all_none(self):
        result = compute_noise_stats([None, None], [None])
        assert result is None

    @pytest.mark.unit
    def test_returns_none_when_one_pocket_all_none(self):
        result = compute_noise_stats([-7.5, -7.6], [None, None])
        assert result is None

    @pytest.mark.unit
    def test_identical_values_zero_std(self):
        aff_l = [-7.5, -7.5, -7.5, -7.5, -7.5]
        aff_w = [-7.0, -7.0, -7.0, -7.0, -7.0]
        r = compute_noise_stats(aff_l, aff_w)
        assert r is not None
        assert r["std_l858r"] == pytest.approx(0.0, abs=1e-6)
        assert r["std_wt"] == pytest.approx(0.0, abs=1e-6)
        assert r["std_delta"] == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.unit
    def test_correct_means(self):
        aff_l = [-8.0, -8.4, -7.6]
        aff_w = [-7.0, -7.2, -6.8]
        r = compute_noise_stats(aff_l, aff_w)
        assert r["mean_l858r"] == pytest.approx(-8.0, abs=1e-3)
        assert r["mean_wt"] == pytest.approx(-7.0, abs=1e-3)

    @pytest.mark.unit
    def test_correct_delta(self):
        aff_l = [-8.0, -8.0, -8.0]
        aff_w = [-7.0, -7.0, -7.0]
        r = compute_noise_stats(aff_l, aff_w)
        assert r["delta"] == pytest.approx(-1.0, abs=1e-3)

    @pytest.mark.unit
    def test_std_delta_propagation(self):
        # std_l=0.2, std_w=0.3  ->  std_delta = sqrt(0.04+0.09) = sqrt(0.13)
        # Use 4-value datasets to get these sample stds
        # Values: mean=0, std (sample) = 0.2 => [-0.2, -0.2, 0.2, 0.2] no...
        # Actually let's just check the propagation formula holds exactly
        aff_l = [-8.0, -8.2, -7.8, -8.0]
        aff_w = [-7.0, -7.3, -6.7, -7.0]
        r = compute_noise_stats(aff_l, aff_w)
        # Manually compute expected std_delta
        l_mean = sum(aff_l) / 4
        w_mean = sum(aff_w) / 4
        std_l = math.sqrt(sum((x - l_mean) ** 2 for x in aff_l) / 3)
        std_w = math.sqrt(sum((x - w_mean) ** 2 for x in aff_w) / 3)
        expected_std_delta = math.sqrt(std_l**2 + std_w**2)
        assert r["std_delta"] == pytest.approx(expected_std_delta, abs=1e-3)

    @pytest.mark.unit
    def test_none_values_excluded(self):
        # Only 3 valid values out of 5
        aff_l = [-8.0, None, -8.2, None, -7.8]
        aff_w = [-7.0, -7.0, -7.0, None, -7.0]
        r = compute_noise_stats(aff_l, aff_w)
        assert r is not None
        assert r["n_l858r"] == 3
        assert r["n_wt"] == 4

    @pytest.mark.unit
    def test_single_value_zero_std(self):
        r = compute_noise_stats([-8.0], [-7.0])
        assert r is not None
        assert r["std_l858r"] == pytest.approx(0.0, abs=1e-6)
        assert r["std_wt"] == pytest.approx(0.0, abs=1e-6)
        assert r["n_l858r"] == 1
        assert r["n_wt"] == 1

    @pytest.mark.unit
    def test_result_keys(self):
        r = compute_noise_stats([-7.5, -7.6], [-7.0, -7.1])
        expected_keys = {
            "mean_l858r",
            "std_l858r",
            "n_l858r",
            "mean_wt",
            "std_wt",
            "n_wt",
            "delta",
            "std_delta",
        }
        assert set(r.keys()) == expected_keys

    @pytest.mark.unit
    def test_positive_std_delta_with_variation(self):
        aff_l = [-8.0, -8.3, -7.7, -8.1, -7.9]
        aff_w = [-7.0, -7.1, -6.9, -7.2, -6.8]
        r = compute_noise_stats(aff_l, aff_w)
        assert r["std_delta"] > 0.0


# ── TestClassifyCall ──────────────────────────────────────────────────────────


class TestClassifyCall:

    @pytest.mark.unit
    def test_covalent_always_low_confidence(self):
        # Even if delta is very large, covalent -> low_confidence_covalent
        call = classify_call(
            delta=-5.0, std_delta=0.1, docking_confidence="low_confidence"
        )
        assert call == "low_confidence_covalent"

    @pytest.mark.unit
    def test_covalent_with_zero_std(self):
        call = classify_call(
            delta=-2.0, std_delta=0.0, docking_confidence="low_confidence"
        )
        assert call == "low_confidence_covalent"

    @pytest.mark.unit
    def test_l858r_selective_confident(self):
        # |delta|=1.0, std_delta=0.3, threshold=1.5 -> 1.5*0.3=0.45 < 1.0 -> confident
        call = classify_call(
            delta=-1.0, std_delta=0.3, docking_confidence="standard", threshold=1.5
        )
        assert call == "L858R_selective"

    @pytest.mark.unit
    def test_wt_selective_confident(self):
        call = classify_call(
            delta=+1.0, std_delta=0.3, docking_confidence="standard", threshold=1.5
        )
        assert call == "WT_selective"

    @pytest.mark.unit
    def test_ambiguous_when_delta_within_noise(self):
        # |delta|=0.4, std_delta=0.3, threshold=1.5 -> 1.5*0.3=0.45 > 0.4 -> ambiguous
        call = classify_call(
            delta=-0.4, std_delta=0.3, docking_confidence="standard", threshold=1.5
        )
        assert call == "ambiguous"

    @pytest.mark.unit
    def test_ambiguous_just_below_threshold(self):
        # |delta|=0.44, threshold*std=1.5*0.3=0.45 -> 0.44 < 0.45 -> ambiguous
        call = classify_call(
            delta=-0.44, std_delta=0.3, docking_confidence="standard", threshold=1.5
        )
        assert call == "ambiguous"

    @pytest.mark.unit
    def test_zero_std_negative_delta(self):
        call = classify_call(delta=-0.5, std_delta=0.0, docking_confidence="standard")
        assert call == "L858R_selective"

    @pytest.mark.unit
    def test_zero_std_positive_delta(self):
        call = classify_call(delta=+0.5, std_delta=0.0, docking_confidence="standard")
        assert call == "WT_selective"

    @pytest.mark.unit
    def test_zero_std_zero_delta(self):
        call = classify_call(delta=0.0, std_delta=0.0, docking_confidence="standard")
        assert call == "ambiguous"

    @pytest.mark.unit
    def test_custom_threshold_tighter(self):
        # threshold=2.0: need |delta| > 2.0 * std_delta
        # delta=-0.6, std_delta=0.3 -> 0.6 vs 0.6 -> not strictly greater -> ambiguous
        call = classify_call(
            delta=-0.6, std_delta=0.3, docking_confidence="standard", threshold=2.0
        )
        assert call == "ambiguous"

    @pytest.mark.unit
    def test_custom_threshold_looser(self):
        # threshold=1.0: need |delta| > 1.0 * std_delta
        # delta=-0.4, std_delta=0.3 -> 0.4 > 0.3 -> L858R_selective
        call = classify_call(
            delta=-0.4, std_delta=0.3, docking_confidence="standard", threshold=1.0
        )
        assert call == "L858R_selective"


# ── TestLoadTopCompounds ──────────────────────────────────────────────────────


class TestLoadTopCompounds:

    @pytest.mark.integration
    @pytest.mark.skipif(
        not _HAS_LIBRARY, reason="library_docking_results.json not found"
    )
    def test_returns_n_compounds(self):
        rows = load_top_compounds(LIBRARY_RESULTS, n=15)
        assert len(rows) == 15

    @pytest.mark.integration
    @pytest.mark.skipif(
        not _HAS_LIBRARY, reason="library_docking_results.json not found"
    )
    def test_sorted_ascending_by_delta(self):
        rows = load_top_compounds(LIBRARY_RESULTS, n=15)
        deltas = [r["selectivity_delta"] for r in rows]
        assert deltas == sorted(deltas)

    @pytest.mark.integration
    @pytest.mark.skipif(
        not _HAS_LIBRARY, reason="library_docking_results.json not found"
    )
    def test_all_status_ok(self):
        rows = load_top_compounds(LIBRARY_RESULTS, n=15)
        for r in rows:
            assert r["docking_status"] == "ok"

    @pytest.mark.integration
    @pytest.mark.skipif(
        not _HAS_LIBRARY, reason="library_docking_results.json not found"
    )
    def test_all_have_valid_delta(self):
        rows = load_top_compounds(LIBRARY_RESULTS, n=15)
        for r in rows:
            assert r["selectivity_delta"] is not None

    @pytest.mark.integration
    @pytest.mark.skipif(
        not _HAS_LIBRARY, reason="library_docking_results.json not found"
    )
    def test_top_compound_is_most_l858r_selective(self):
        rows = load_top_compounds(LIBRARY_RESULTS, n=15)
        # cmpd_024 has the strongest delta (-0.953) from the first pass
        assert rows[0]["cid"] == "cmpd_024"

    @pytest.mark.integration
    @pytest.mark.skipif(
        not _HAS_LIBRARY, reason="library_docking_results.json not found"
    )
    def test_small_n(self):
        rows = load_top_compounds(LIBRARY_RESULTS, n=3)
        assert len(rows) == 3
        # Still the most negative delta first
        assert rows[0]["selectivity_delta"] <= rows[1]["selectivity_delta"]
