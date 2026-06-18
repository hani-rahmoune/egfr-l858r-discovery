"""
Tests for the Phase 25 dashboard data-loading + scoring-client helpers.

The Streamlit UI itself is not unit-tested (it needs a browser session); these
cover the pure functions the UI depends on:
  TestDataLoaders   — each loader returns the right shape / tolerates missing files
  TestRankingPlacement — generated-vs-known summary arithmetic
  TestApiClient     — API availability + local-registry fallback (offline, mock)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard import api_client as api
from src.dashboard import data_loaders as dl

# ── TestDataLoaders ─────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestDataLoaders:

    def test_final_ranking_loads(self):
        df = dl.load_final_ranking()
        assert df is not None
        assert {"rank", "cid", "source", "final_score", "warnings"}.issubset(df.columns)
        assert len(df) > 0

    def test_final_ranking_has_both_sources(self):
        df = dl.load_final_ranking()
        assert set(df["source"]) <= {"known", "generated"}
        assert "known" in set(df["source"])

    def test_qsar_metrics(self):
        m = dl.load_qsar_metrics()
        assert not m.empty
        assert {"model", "best", "rmse", "r2", "pearson_r"}.issubset(m.columns)
        # backbone RF, wt-proxy XGB
        bests = set(m["best"])
        assert "random_forest" in bests or "xgboost" in bests

    def test_seed_stability_shape(self):
        ss = dl.load_seed_stability()
        assert len(ss) == 2
        assert {"rmse_mean", "rmse_std", "r2_mean", "r2_std"}.issubset(ss.columns)
        # std must be present and positive (the whole point of the page)
        assert (ss["r2_std"] > 0).all()

    def test_seed_stability_is_a_copy(self):
        a = dl.load_seed_stability()
        a.loc[0, "r2_mean"] = 999
        b = dl.load_seed_stability()
        assert b.loc[0, "r2_mean"] != 999

    def test_fingerprint_ablation(self):
        abl = dl.load_fingerprint_ablation()
        assert "general" in abl and "wt_proxy" in abl
        g = abl["general"]
        assert {"fingerprint", "val_rmse", "test_rmse_mean", "test_rmse_std"}.issubset(
            g.columns
        )
        # sorted ascending by val_rmse -> first row is the winner
        assert g["val_rmse"].is_monotonic_increasing

    def test_model3_verdict(self):
        m3 = dl.load_model3_verdict()
        assert m3 is not None
        assert "backbone" in set(m3["table"]["method"])
        assert m3["n_l858r"] in (22, 19, 21)  # documented L858R count
        assert isinstance(m3["verdict"], str) and m3["verdict"]

    def test_model4_verdict(self):
        m4 = dl.load_model4_verdict()
        assert m4 is not None
        assert m4["n_pairs"] == 9
        assert m4["spearman_derived"] is not None

    def test_docking_noise(self):
        nz = dl.load_docking_noise()
        assert nz is not None
        assert {"cid", "delta", "std_delta", "call"}.issubset(nz.columns)
        assert (nz["std_delta"] >= 0).all()

    def test_sanity_check(self):
        sc = dl.load_sanity_check()
        assert sc is not None
        assert sc["verdict"] == "PASS"
        assert "delta" in sc["table"].columns

    def test_generated_docking(self):
        gd = dl.load_generated_docking()
        assert gd is not None
        assert {"cid", "selectivity_delta", "l858r_score"}.issubset(gd.columns)

    def test_rl_results(self):
        rl = dl.load_rl_results()
        assert rl is not None
        assert rl["verdict"] == "INCONCLUSIVE"
        assert len(rl["table"]) == 6

    def test_missing_file_returns_none(self, tmp_path, monkeypatch):
        # point a loader at a non-existent tree -> graceful None
        monkeypatch.setattr(dl, "_QSAR", tmp_path / "nope")
        assert dl.load_docking_noise() is None
        assert dl.load_model3_verdict() is None


# ── TestRankingPlacement ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestRankingPlacement:

    def _frame(self):
        return pd.DataFrame(
            [
                {"rank": 1, "cid": "k1", "source": "known", "final_score": 0.9},
                {"rank": 2, "cid": "k2", "source": "known", "final_score": 0.8},
                {"rank": 3, "cid": "g1", "source": "generated", "final_score": 0.7},
                {"rank": 4, "cid": "g2", "source": "generated", "final_score": 0.6},
            ]
        )

    def test_counts(self):
        p = dl.ranking_placement(self._frame())
        assert p["n_total"] == 4 and p["n_known"] == 2 and p["n_generated"] == 2

    def test_best_generated(self):
        p = dl.ranking_placement(self._frame())
        assert p["best_generated_rank"] == 3
        assert p["best_generated_cid"] == "g1"

    def test_best_known(self):
        p = dl.ranking_placement(self._frame())
        assert p["best_known_cid"] == "k1"

    def test_empty_frame(self):
        assert dl.ranking_placement(pd.DataFrame()) == {}

    def test_all_known(self):
        df = pd.DataFrame(
            [{"rank": 1, "cid": "k", "source": "known", "final_score": 1.0}]
        )
        p = dl.ranking_placement(df)
        assert p["n_generated"] == 0
        assert "best_generated_rank" not in p


# ── TestApiClient ───────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestApiClient:

    def test_api_unavailable_on_dead_port(self):
        # nothing listens here -> fast False, no exception
        assert api.api_available("http://127.0.0.1:9", timeout=0.2) is False

    def test_api_unavailable_on_garbage_url(self):
        assert api.api_available("http://nonexistent.invalid", timeout=0.2) is False

    def test_predict_via_registry_delegates(self):
        class _Reg:
            def score(self, smiles):
                return {"smiles": smiles, "valid": True, "pic50_mutant": 8.0}

        out = api.predict_via_registry(_Reg(), "CCO")
        assert out["valid"] is True
        assert out["pic50_mutant"] == 8.0
        assert out["smiles"] == "CCO"


# ── TestAppRenders (integration — runs the real Streamlit script) ───────────────


@pytest.mark.integration
class TestAppRenders:
    """Render the actual app via Streamlit's AppTest and assert no page throws.
    The default 'Single molecule' page resolves a scorer (API down -> local
    ModelRegistry load), so this needs the real artifacts."""

    @pytest.fixture(scope="class")
    def app(self):
        general = PROJECT_ROOT / "models" / "qsar" / "general" / "best_model.pkl"
        if not general.exists():
            pytest.skip("Model artifacts not present; run training first.")
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(
            str(PROJECT_ROOT / "src" / "dashboard" / "app.py"), default_timeout=90
        )
        at.run()
        return at

    def test_default_page_no_exception(self, app):
        assert not app.exception

    @pytest.mark.parametrize(
        "page",
        [
            "Final ranking",
            "Model performance",
            "Docking results",
            "Limitations",
        ],
    )
    def test_data_pages_render(self, app, page):
        app.radio[0].set_value(page).run()
        assert not app.exception
