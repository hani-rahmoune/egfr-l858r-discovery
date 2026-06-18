"""
Unit tests for Phase 23 final integrated ranking (src/scoring/ranking.py).

Coverage:
  TestRankingWeights   — defaults, sum, normalize, as_tuple
  TestMinmaxNormalize  — scaling, constants, None/NaN, ordering
  TestBuildWarnings    — each warning source + combinations, no false positives
  TestRankCandidates   — formula, sign conventions, AD scaling, warnings-not-scores,
                         sorting, columns, empty input

Pure pandas/numpy — no RDKit or model artifacts needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring.ranking import (
    RANKED_COLUMNS,
    RankingWeights,
    build_warnings,
    minmax_normalize,
    rank_candidates,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────


def _record(
    cid,
    activity,
    l858r,
    wt,
    qed,
    cf=1.0,
    domain="in_domain",
    covalent=False,
    warheads=None,
    within_noise=None,
    source="known",
):
    return {
        "cid": cid,
        "source": source,
        "smiles": f"C-{cid}",
        "activity": activity,
        "l858r_score": l858r,
        "wt_score": wt,
        "selectivity_delta": (
            None if (l858r is None or wt is None) else round(l858r - wt, 3)
        ),
        "admet_qed": qed,
        "domain": domain,
        "max_tanimoto": 0.6,
        "confidence_factor": cf,
        "is_covalent": covalent,
        "warheads": warheads or ([] if not covalent else ["acrylamide"]),
        "selectivity_within_noise": within_noise,
    }


# ── TestRankingWeights ─────────────────────────────────────────────────────────


@pytest.mark.unit
class TestRankingWeights:

    def test_defaults(self):
        w = RankingWeights()
        assert w.activity == 0.30
        assert w.docking_selectivity == 0.30
        assert w.docking_affinity == 0.20
        assert w.admet == 0.20

    def test_default_sum_is_one(self):
        assert RankingWeights().sum() == pytest.approx(1.0)

    def test_normalized_sums_to_one(self):
        w = RankingWeights(
            activity=3, docking_selectivity=3, docking_affinity=2, admet=2
        ).normalized()
        assert w.sum() == pytest.approx(1.0)
        assert w.activity == pytest.approx(0.30)

    def test_normalized_zero_raises(self):
        with pytest.raises(ValueError):
            RankingWeights(0, 0, 0, 0).normalized()

    def test_as_tuple_order(self):
        w = RankingWeights(0.1, 0.2, 0.3, 0.4)
        assert w.as_tuple() == (0.1, 0.2, 0.3, 0.4)


# ── TestMinmaxNormalize ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestMinmaxNormalize:

    def test_basic_range(self):
        out = minmax_normalize([0.0, 5.0, 10.0])
        np.testing.assert_allclose(out, [0.0, 0.5, 1.0])

    def test_preserves_order(self):
        out = minmax_normalize([3.0, 1.0, 2.0])
        assert out[1] < out[2] < out[0]

    def test_constant_maps_to_half(self):
        out = minmax_normalize([4.0, 4.0, 4.0])
        np.testing.assert_allclose(out, [0.5, 0.5, 0.5])

    def test_none_maps_to_zero(self):
        out = minmax_normalize([10.0, None, 0.0])
        assert out[1] == 0.0
        assert out[0] == 1.0
        assert out[2] == 0.0

    def test_all_none_returns_zeros(self):
        out = minmax_normalize([None, None])
        np.testing.assert_allclose(out, [0.0, 0.0])

    def test_single_value_is_half(self):
        out = minmax_normalize([7.0])
        assert out[0] == 0.5

    def test_negative_values(self):
        out = minmax_normalize([-10.0, -5.0, 0.0])
        np.testing.assert_allclose(out, [0.0, 0.5, 1.0])

    def test_output_within_unit_interval(self):
        out = minmax_normalize([0.1, 99.0, -3.0, 4.0, None])
        present = out[[0, 1, 2, 3]]
        assert present.min() >= 0.0 and present.max() <= 1.0


# ── TestBuildWarnings ──────────────────────────────────────────────────────────


@pytest.mark.unit
class TestBuildWarnings:

    def test_in_domain_noncovalent_no_warnings(self):
        assert build_warnings("in_domain", False, [], False) == []

    def test_out_of_domain_warns(self):
        w = build_warnings("out_of_domain", False, [], None)
        assert len(w) == 1 and "OUT_OF_DOMAIN" in w[0]

    def test_borderline_warns(self):
        w = build_warnings("borderline", False, [], None)
        assert len(w) == 1 and "BORDERLINE" in w[0]

    def test_covalent_warns_with_warhead_name(self):
        w = build_warnings("in_domain", True, ["acrylamide"], False)
        assert len(w) == 1 and "COVALENT" in w[0] and "acrylamide" in w[0]

    def test_within_noise_warns(self):
        w = build_warnings("in_domain", False, [], True)
        assert len(w) == 1 and "WITHIN_NOISE" in w[0]

    def test_within_noise_false_no_warning(self):
        assert build_warnings("in_domain", False, [], False) == []

    def test_all_three_warnings(self):
        w = build_warnings("out_of_domain", True, ["vinyl_sulfone"], True)
        assert len(w) == 3
        assert any("OUT_OF_DOMAIN" in x for x in w)
        assert any("COVALENT" in x for x in w)
        assert any("WITHIN_NOISE" in x for x in w)

    def test_warning_order(self):
        w = build_warnings("borderline", True, ["acrylamide"], True)
        assert "BORDERLINE" in w[0]
        assert "COVALENT" in w[1]
        assert "WITHIN_NOISE" in w[2]


# ── TestRankCandidates ─────────────────────────────────────────────────────────


@pytest.mark.unit
class TestRankCandidates:

    def test_empty_returns_empty_with_columns(self):
        df = rank_candidates([])
        assert len(df) == 0
        assert list(df.columns) == RANKED_COLUMNS

    def test_columns_present(self):
        df = rank_candidates([_record("a", 8.0, -8.0, -7.0, 0.6)])
        assert list(df.columns) == RANKED_COLUMNS

    def test_rank_is_sequential(self):
        recs = [_record(f"c{i}", 7.0 + i, -7.0 - i, -7.0, 0.5) for i in range(4)]
        df = rank_candidates(recs)
        assert list(df["rank"]) == [1, 2, 3, 4]

    def test_sorted_by_final_desc(self):
        recs = [_record(f"c{i}", 7.0 + i, -7.0 - i, -7.0, 0.5) for i in range(4)]
        df = rank_candidates(recs)
        finals = list(df["final_score"])
        assert finals == sorted(finals, reverse=True)

    def test_best_in_all_components_ranks_first(self):
        # c_best: highest activity, most negative delta (selective), strongest binding, top QED
        best = _record("best", 9.0, -9.0, -6.0, 0.9)  # delta -3.0
        worst = _record("worst", 6.0, -6.0, -6.0, 0.2)  # delta  0.0
        mid = _record("mid", 7.5, -7.0, -6.5, 0.5)  # delta -0.5
        df = rank_candidates([worst, mid, best])
        assert df.iloc[0]["cid"] == "best"
        assert df.iloc[-1]["cid"] == "worst"

    def test_selectivity_sign_convention(self):
        # more negative selectivity_delta => more L858R-selective => higher selectivity_norm
        a = _record("a", 7.0, -8.0, -6.0, 0.5)  # delta -2.0 (selective)
        b = _record("b", 7.0, -8.0, -8.0, 0.5)  # delta  0.0 (not)
        df = rank_candidates([a, b]).set_index("cid")
        assert df.loc["a", "selectivity_norm"] > df.loc["b", "selectivity_norm"]

    def test_affinity_sign_convention(self):
        # more negative l858r_score => stronger binding => higher affinity_norm
        a = _record("a", 7.0, -9.0, -7.0, 0.5)
        b = _record("b", 7.0, -6.0, -7.0, 0.5)
        df = rank_candidates([a, b]).set_index("cid")
        assert df.loc["a", "affinity_norm"] > df.loc["b", "affinity_norm"]

    def test_confidence_factor_scales_final(self):
        # identical evidence, different AD confidence -> final scales by cf, bioactivity equal
        a = _record("a", 8.0, -8.0, -7.0, 0.6, cf=1.0, domain="in_domain")
        b = _record("b", 8.0, -8.0, -7.0, 0.6, cf=0.5, domain="out_of_domain")
        df = rank_candidates([a, b]).set_index("cid")
        assert df.loc["a", "bioactivity_score"] == pytest.approx(
            df.loc["b", "bioactivity_score"]
        )
        assert df.loc["b", "final_score"] == pytest.approx(
            0.5 * df.loc["b", "bioactivity_score"]
        )
        assert df.loc["a", "final_score"] > df.loc["b", "final_score"]

    def test_covalent_does_not_change_score_only_warns(self):
        # covalent vs non-covalent with identical numbers -> identical scores, covalent warned
        a = _record("a", 8.0, -8.0, -7.0, 0.6, covalent=False)
        b = _record("b", 8.0, -8.0, -7.0, 0.6, covalent=True, warheads=["acrylamide"])
        df = rank_candidates([a, b]).set_index("cid")
        assert df.loc["a", "final_score"] == pytest.approx(df.loc["b", "final_score"])
        assert df.loc["a", "warnings"] == ""
        assert "COVALENT" in df.loc["b", "warnings"]

    def test_within_noise_warns_not_scored(self):
        a = _record("a", 8.0, -8.0, -7.0, 0.6, within_noise=False)
        b = _record("b", 8.0, -8.0, -7.0, 0.6, within_noise=True)
        df = rank_candidates([a, b]).set_index("cid")
        assert df.loc["a", "final_score"] == pytest.approx(df.loc["b", "final_score"])
        assert "WITHIN_NOISE" in df.loc["b", "warnings"]
        assert "WITHIN_NOISE" not in df.loc["a", "warnings"]

    def test_formula_matches_manual(self):
        # two records so normalisation is well-defined; check the winner's arithmetic
        a = _record("a", 9.0, -9.0, -6.0, 0.8, cf=1.0)  # all best -> norms all 1.0
        b = _record("b", 6.0, -6.0, -6.0, 0.2, cf=1.0)  # all worst -> norms all 0.0
        df = rank_candidates([a, b]).set_index("cid")
        # a is best on every axis: norms = 1 -> bioactivity = sum of weights = 1.0
        assert df.loc["a", "bioactivity_score"] == pytest.approx(1.0)
        assert df.loc["a", "final_score"] == pytest.approx(1.0)
        # b is worst on every axis: norms = 0 -> bioactivity = 0
        assert df.loc["b", "bioactivity_score"] == pytest.approx(0.0)

    def test_missing_docking_penalised_not_dropped(self):
        # a candidate with no docking still appears, scored 0 on those axes
        full = _record("full", 8.0, -8.0, -7.0, 0.6)
        nodock = _record("nodock", 8.0, None, None, 0.6)
        nodock["selectivity_delta"] = None
        df = rank_candidates([full, nodock])
        assert set(df["cid"]) == {"full", "nodock"}
        nd = df.set_index("cid").loc["nodock"]
        assert nd["affinity_norm"] == 0.0
        assert nd["selectivity_norm"] == 0.0

    def test_source_column_preserved(self):
        recs = [
            _record("k", 8.0, -8.0, -7.0, 0.6, source="known"),
            _record("g", 8.0, -8.0, -7.0, 0.6, source="generated"),
        ]
        df = rank_candidates(recs).set_index("cid")
        assert df.loc["k", "source"] == "known"
        assert df.loc["g", "source"] == "generated"

    def test_custom_weights_change_ranking(self):
        # weight activity 100% -> highest-activity candidate wins regardless of others
        a = _record("a", 9.0, -6.0, -6.0, 0.1)  # top activity, worst else
        b = _record("b", 6.0, -9.0, -6.0, 0.9)  # bottom activity, best else
        w = RankingWeights(
            activity=1.0, docking_selectivity=0.0, docking_affinity=0.0, admet=0.0
        )
        df = rank_candidates([a, b], weights=w)
        assert df.iloc[0]["cid"] == "a"
