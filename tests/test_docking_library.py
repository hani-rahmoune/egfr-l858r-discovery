"""
Tests for Phase B2 library docking modules.

Unit tests (no model files, no Vina binary):
  - select_top_candidates: validate deduplication and ranking logic using
    synthetic data
  - build_ranking: validate delta computation, covalent flag propagation,
    sort order, and failure-handling
  - report: smoke test for correct summary dict structure

Integration tests (marked 'integration', require real parquet + model files):
  - select_top_candidates on real data: correct n, dtype, uniqueness
  - covalent flags on real top-50: at least one acrylamide present
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.dock_library import (
    build_ranking,
    report,
)
from src.features.covalent import detect_warheads

# ── Synthetic helpers ─────────────────────────────────────────────────────────


def _make_candidates(smiles_and_preds: list[tuple[str, float]]) -> pd.DataFrame:
    """Build a minimal candidates DataFrame for build_ranking tests."""
    rows = []
    for i, (smi, pred) in enumerate(smiles_and_preds):
        rows.append(
            {
                "canonical_smiles": smi,
                "pic50": 6.0 + i * 0.1,
                "pred_pic50": pred,
                "mutation_flag": "unknown",
            }
        )
    return pd.DataFrame(rows)


def _make_pdbqt(tmp_path: Path, name: str, affinity: float) -> Path:
    """Write a minimal Vina PDBQT file with one scored pose."""
    p = tmp_path / name
    p.write_text(
        f"REMARK VINA RESULT: {affinity:9.3f}      0.000      0.000\n"
        "ATOM      1  C1  LIG L   1       0.000   0.000   0.000  1.00  0.00      0.000 C\n"
    )
    return p


# SMILES fixtures
_GEFITINIB = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
_OSIMERTINIB = "COc1cc2c(Nc3cccc(NC(=O)/C=C/CN(C)C)c3)ncnc2cc1NC(C)=O"
_SIMPLE_MOLECULE = "c1ccccc1"  # benzene, no warhead


# ── build_ranking ─────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestBuildRanking:
    def test_delta_computed_correctly(self, tmp_path):
        candidates = _make_candidates([(_GEFITINIB, 8.5)])
        l858r_out = _make_pdbqt(tmp_path, "cmpd_001__2ITZ_receptor_out.pdbqt", -8.0)
        wt_out = _make_pdbqt(
            tmp_path, "cmpd_001__2ITY_aligned_receptor_out.pdbqt", -7.5
        )
        docking = {"cmpd_001": {"L858R": l858r_out, "WT": wt_out}}
        rows = build_ranking(candidates, docking)
        assert len(rows) == 1
        assert rows[0]["selectivity_delta"] == pytest.approx(-0.5, abs=0.01)

    def test_covalent_compound_tagged_low_confidence(self, tmp_path):
        candidates = _make_candidates([(_OSIMERTINIB, 9.0)])
        l858r_out = _make_pdbqt(tmp_path, "cmpd_001__rec_L858R.pdbqt", -8.0)
        wt_out = _make_pdbqt(tmp_path, "cmpd_001__rec_WT.pdbqt", -7.5)
        docking = {"cmpd_001": {"L858R": l858r_out, "WT": wt_out}}
        rows = build_ranking(candidates, docking)
        assert rows[0]["docking_confidence"] == "low_confidence"
        assert "acrylamide" in rows[0]["warheads"]

    def test_non_covalent_compound_standard_confidence(self, tmp_path):
        candidates = _make_candidates([(_GEFITINIB, 8.5)])
        l858r_out = _make_pdbqt(tmp_path, "cmpd_001__r.pdbqt", -8.0)
        wt_out = _make_pdbqt(tmp_path, "cmpd_001__w.pdbqt", -7.5)
        docking = {"cmpd_001": {"L858R": l858r_out, "WT": wt_out}}
        rows = build_ranking(candidates, docking)
        assert rows[0]["docking_confidence"] == "standard"
        assert rows[0]["warheads"] == []

    def test_failed_docking_status(self, tmp_path):
        candidates = _make_candidates([(_GEFITINIB, 8.5)])
        docking = {"cmpd_001": {"L858R": None, "WT": None}}
        rows = build_ranking(candidates, docking)
        assert rows[0]["docking_status"] == "failed"
        assert rows[0]["selectivity_delta"] is None

    def test_partial_docking_status(self, tmp_path):
        candidates = _make_candidates([(_GEFITINIB, 8.5)])
        l858r_out = _make_pdbqt(tmp_path, "cmpd_001__r.pdbqt", -8.0)
        docking = {"cmpd_001": {"L858R": l858r_out, "WT": None}}
        rows = build_ranking(candidates, docking)
        assert rows[0]["docking_status"] == "partial"
        assert rows[0]["selectivity_delta"] is None

    def test_sorted_ascending_by_delta(self, tmp_path):
        # Three compounds with different deltas: -1.0, -0.2, +0.3
        candidates = _make_candidates(
            [
                (_GEFITINIB, 9.0),  # cmpd_001: delta -1.0 (best L858R)
                (_SIMPLE_MOLECULE, 8.5),  # cmpd_002: delta -0.2
                (_GEFITINIB, 8.0),  # cmpd_003: delta +0.3 (WT-selective)
            ]
        )
        docking = {
            "cmpd_001": {
                "L858R": _make_pdbqt(tmp_path, "a1.pdbqt", -8.5),
                "WT": _make_pdbqt(tmp_path, "b1.pdbqt", -7.5),
            },
            "cmpd_002": {
                "L858R": _make_pdbqt(tmp_path, "a2.pdbqt", -7.2),
                "WT": _make_pdbqt(tmp_path, "b2.pdbqt", -7.0),
            },
            "cmpd_003": {
                "L858R": _make_pdbqt(tmp_path, "a3.pdbqt", -6.5),
                "WT": _make_pdbqt(tmp_path, "b3.pdbqt", -6.8),
            },
        }
        rows = build_ranking(candidates, docking)
        deltas = [
            r["selectivity_delta"] for r in rows if r["selectivity_delta"] is not None
        ]
        assert deltas == sorted(deltas), "Rows not sorted ascending by delta"
        assert deltas[0] == pytest.approx(-1.0, abs=0.01)

    def test_failed_rows_sorted_after_ok_rows(self, tmp_path):
        candidates = _make_candidates(
            [
                (_GEFITINIB, 9.0),  # cmpd_001: ok
                (_GEFITINIB, 8.5),  # cmpd_002: failed
            ]
        )
        docking = {
            "cmpd_001": {
                "L858R": _make_pdbqt(tmp_path, "a.pdbqt", -8.0),
                "WT": _make_pdbqt(tmp_path, "b.pdbqt", -7.5),
            },
            "cmpd_002": {"L858R": None, "WT": None},
        }
        rows = build_ranking(candidates, docking)
        statuses = [r["docking_status"] for r in rows]
        ok_indices = [i for i, s in enumerate(statuses) if s == "ok"]
        fail_indices = [i for i, s in enumerate(statuses) if s == "failed"]
        if ok_indices and fail_indices:
            assert max(ok_indices) < min(fail_indices)

    def test_cid_format(self, tmp_path):
        candidates = _make_candidates([(_GEFITINIB, 8.5)])
        docking = {"cmpd_001": {"L858R": None, "WT": None}}
        rows = build_ranking(candidates, docking)
        assert rows[0]["cid"] == "cmpd_001"

    def test_returns_all_candidates(self, tmp_path):
        n = 5
        entries = [(_GEFITINIB, float(9 - i * 0.1)) for i in range(n)]
        candidates = _make_candidates(entries)
        docking = {f"cmpd_{i+1:03d}": {"L858R": None, "WT": None} for i in range(n)}
        rows = build_ranking(candidates, docking)
        assert len(rows) == n

    def test_row_has_required_keys(self, tmp_path):
        candidates = _make_candidates([(_GEFITINIB, 8.5)])
        docking = {"cmpd_001": {"L858R": None, "WT": None}}
        rows = build_ranking(candidates, docking)
        required = {
            "cid",
            "smiles",
            "pred_pic50",
            "pic50",
            "mutation_flag",
            "warheads",
            "docking_confidence",
            "l858r_score",
            "wt_score",
            "selectivity_delta",
            "docking_status",
        }
        assert set(rows[0].keys()) >= required


# ── report ────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestReport:
    def _sample_rows(self, tmp_path) -> list[dict]:
        return [
            {
                "cid": "cmpd_001",
                "smiles": _GEFITINIB,
                "pred_pic50": 9.0,
                "pic50": 8.5,
                "mutation_flag": "unknown",
                "warheads": [],
                "docking_confidence": "standard",
                "l858r_score": -8.0,
                "wt_score": -7.5,
                "selectivity_delta": -0.5,
                "docking_status": "ok",
            },
            {
                "cid": "cmpd_002",
                "smiles": _OSIMERTINIB,
                "pred_pic50": 8.8,
                "pic50": 8.0,
                "mutation_flag": "unknown",
                "warheads": ["acrylamide"],
                "docking_confidence": "low_confidence",
                "l858r_score": -7.9,
                "wt_score": -7.3,
                "selectivity_delta": -0.6,
                "docking_status": "ok",
            },
            {
                "cid": "cmpd_003",
                "smiles": _SIMPLE_MOLECULE,
                "pred_pic50": 8.5,
                "pic50": 7.0,
                "mutation_flag": "unknown",
                "warheads": [],
                "docking_confidence": "standard",
                "l858r_score": None,
                "wt_score": None,
                "selectivity_delta": None,
                "docking_status": "failed",
            },
        ]

    def test_returns_dict_with_required_keys(self, tmp_path):
        rows = self._sample_rows(tmp_path)
        summary = report(rows, n_candidates=3, n_ligand_failures=0)
        required = {
            "n_candidates",
            "n_ok",
            "n_partial",
            "n_failed",
            "n_covalent_flagged",
            "compounds",
            "note",
        }
        assert set(summary.keys()) >= required

    def test_n_ok_count(self, tmp_path):
        rows = self._sample_rows(tmp_path)
        summary = report(rows, n_candidates=3, n_ligand_failures=0)
        assert summary["n_ok"] == 2

    def test_n_failed_includes_ligand_failures(self, tmp_path):
        rows = self._sample_rows(tmp_path)
        summary = report(rows, n_candidates=3, n_ligand_failures=2)
        assert summary["n_failed"] == 3  # 1 failed docking + 2 ligand failures

    def test_n_covalent_flagged(self, tmp_path):
        rows = self._sample_rows(tmp_path)
        summary = report(rows, n_candidates=3, n_ligand_failures=0)
        assert summary["n_covalent_flagged"] == 1

    def test_note_contains_exploratory_label(self, tmp_path):
        rows = self._sample_rows(tmp_path)
        summary = report(rows, n_candidates=3, n_ligand_failures=0)
        assert "EXPLORATORY" in summary["note"]

    def test_note_mentions_covalent_limitation(self, tmp_path):
        rows = self._sample_rows(tmp_path)
        summary = report(rows, n_candidates=3, n_ligand_failures=0)
        assert "covalent" in summary["note"].lower()

    def test_compounds_in_summary(self, tmp_path):
        rows = self._sample_rows(tmp_path)
        summary = report(rows, n_candidates=3, n_ligand_failures=0)
        assert len(summary["compounds"]) == 3


# ── Integration: real parquet + model ─────────────────────────────────────────

_PARQUET = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "processed"
    / "features_egfr_general.parquet"
)
_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "qsar" / "general"

_real_data = pytest.mark.skipif(
    not (_PARQUET.exists() and (_MODEL_DIR / "best_model.pkl").exists()),
    reason="Real parquet / model not present (run train_models.py first)",
)


@pytest.mark.integration
class TestSelectTopCandidatesIntegration:
    @_real_data
    def test_returns_correct_n(self):
        from scripts.dock_library import select_top_candidates

        df = select_top_candidates(_PARQUET, _MODEL_DIR, n=50)
        assert len(df) == 50

    @_real_data
    def test_smiles_are_unique(self):
        from scripts.dock_library import select_top_candidates

        df = select_top_candidates(_PARQUET, _MODEL_DIR, n=50)
        assert df["canonical_smiles"].nunique() == 50

    @_real_data
    def test_sorted_descending_by_pred_pic50(self):
        from scripts.dock_library import select_top_candidates

        df = select_top_candidates(_PARQUET, _MODEL_DIR, n=50)
        preds = df["pred_pic50"].tolist()
        assert preds == sorted(preds, reverse=True)

    @_real_data
    def test_has_required_columns(self):
        from scripts.dock_library import select_top_candidates

        df = select_top_candidates(_PARQUET, _MODEL_DIR, n=50)
        assert set(
            ["canonical_smiles", "pic50", "pred_pic50", "mutation_flag"]
        ).issubset(df.columns)

    @_real_data
    def test_at_least_one_covalent_in_top50(self):
        """Top 50 EGFR backbone candidates should include acrylamide compounds."""
        from scripts.dock_library import select_top_candidates

        df = select_top_candidates(_PARQUET, _MODEL_DIR, n=50)
        flagged = [smi for smi in df["canonical_smiles"] if detect_warheads(smi)]
        assert (
            len(flagged) >= 1
        ), "Expected at least one covalent compound in top-50 EGFR candidates"
