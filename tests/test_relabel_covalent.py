"""
Tests for scripts/relabel_covalent.py.

Verifies that re-applying the updated WARHEAD_SMARTS to existing docking
JSON files correctly updates warheads, docking_confidence, and call fields.
Uses synthetic JSON data in tmp_path — no real model files required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.relabel_covalent import _classify_call, relabel_library, relabel_noise

# ── Synthetic JSON builders ───────────────────────────────────────────────────

_GEFITINIB = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
_OSIMERTINIB = "COc1cc2c(Nc3cccc(NC(=O)/C=C/CN(C)C)c3)ncnc2cc1NC(C)=O"
_CMPD_021 = "C=CC(=O)Oc1cc2c(Nc3ccc(Cl)c(Cl)c3F)ncnc2cc1OC"


def _library_json(compounds: list[dict]) -> dict:
    return {
        "compounds": compounds,
        "summary": {},
        "note": "test",
    }


def _noise_json(compounds: list[dict]) -> dict:
    return {
        "compounds": compounds,
        "summary": {},
        "note": "test",
    }


def _lib_compound(
    cid: str, smiles: str, warheads: list = None, confidence: str = "standard"
) -> dict:
    return {
        "cid": cid,
        "smiles": smiles,
        "pred_pic50": 8.5,
        "pic50": 9.0,
        "mutation_flag": "unknown",
        "warheads": warheads or [],
        "docking_confidence": confidence,
        "l858r_score": -8.0,
        "wt_score": -7.5,
        "selectivity_delta": -0.5,
        "docking_status": "ok",
    }


def _noise_compound(
    cid: str,
    smiles: str,
    delta: float,
    std_delta: float,
    warheads: list = None,
    confidence: str = "standard",
    call: str = "L858R_selective",
) -> dict:
    return {
        "cid": cid,
        "smiles": smiles,
        "initial_delta": delta,
        "docking_confidence": confidence,
        "warheads": warheads or [],
        "affinities_l858r": [-8.0] * 5,
        "affinities_wt": [-7.5] * 5,
        "noise_stats": {
            "mean_l858r": -8.0,
            "std_l858r": 0.05,
            "mean_wt": -7.5,
            "std_wt": 0.10,
            "delta": delta,
            "std_delta": std_delta,
        },
        "call": call,
    }


# ── _classify_call ────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestClassifyCall:
    def test_low_confidence_always_covalent(self):
        assert _classify_call(-1.0, 0.1, "low_confidence") == "low_confidence_covalent"

    def test_low_confidence_ignores_delta(self):
        assert _classify_call(+0.5, 0.01, "low_confidence") == "low_confidence_covalent"

    def test_l858r_selective_when_delta_large_enough(self):
        # |delta| = 0.5, std_delta = 0.2, threshold = 1.5 → 0.5 > 0.3 → selective
        assert _classify_call(-0.5, 0.2, "standard") == "L858R_selective"

    def test_wt_selective(self):
        assert _classify_call(+0.5, 0.2, "standard") == "WT_selective"

    def test_ambiguous_when_delta_too_small(self):
        # |delta| = 0.1, std_delta = 0.2, 1.5*0.2 = 0.3 → 0.1 < 0.3 → ambiguous
        assert _classify_call(-0.1, 0.2, "standard") == "ambiguous"

    def test_zero_std_delta_uses_sign(self):
        assert _classify_call(-0.5, 0.0, "standard") == "L858R_selective"
        assert _classify_call(+0.5, 0.0, "standard") == "WT_selective"

    def test_none_delta_returns_ambiguous(self):
        assert _classify_call(None, None, "standard") == "ambiguous"


# ── relabel_library ───────────────────────────────────────────────────────────


@pytest.mark.unit
class TestRelabelLibrary:
    def test_acrylate_ester_gets_labeled(self, tmp_path):
        p = tmp_path / "lib.json"
        p.write_text(
            json.dumps(
                _library_json(
                    [
                        _lib_compound(
                            "cmpd_021", _CMPD_021, warheads=[], confidence="standard"
                        ),
                    ]
                )
            ),
            encoding="utf-8",
        )

        changes = relabel_library(p)

        assert "cmpd_021" in changes
        result = json.loads(p.read_text(encoding="utf-8"))["compounds"][0]
        assert "acrylate_ester" in result["warheads"]
        assert result["docking_confidence"] == "low_confidence"

    def test_non_covalent_unchanged(self, tmp_path):
        p = tmp_path / "lib.json"
        p.write_text(
            json.dumps(
                _library_json(
                    [
                        _lib_compound(
                            "cmpd_001", _GEFITINIB, warheads=[], confidence="standard"
                        ),
                    ]
                )
            ),
            encoding="utf-8",
        )

        changes = relabel_library(p)

        assert "cmpd_001" not in changes
        result = json.loads(p.read_text(encoding="utf-8"))["compounds"][0]
        assert result["warheads"] == []
        assert result["docking_confidence"] == "standard"

    def test_already_labeled_acrylamide_unchanged(self, tmp_path):
        p = tmp_path / "lib.json"
        p.write_text(
            json.dumps(
                _library_json(
                    [
                        _lib_compound(
                            "cmpd_x",
                            _OSIMERTINIB,
                            warheads=["acrylamide"],
                            confidence="low_confidence",
                        ),
                    ]
                )
            ),
            encoding="utf-8",
        )

        changes = relabel_library(p)

        assert "cmpd_x" not in changes

    def test_returns_change_dict(self, tmp_path):
        p = tmp_path / "lib.json"
        p.write_text(
            json.dumps(
                _library_json(
                    [
                        _lib_compound("cmpd_021", _CMPD_021),
                    ]
                )
            ),
            encoding="utf-8",
        )
        changes = relabel_library(p)
        assert isinstance(changes, dict)

    def test_multiple_compounds_only_affected_one_changes(self, tmp_path):
        p = tmp_path / "lib.json"
        p.write_text(
            json.dumps(
                _library_json(
                    [
                        _lib_compound("cmpd_001", _GEFITINIB),
                        _lib_compound("cmpd_021", _CMPD_021),
                        _lib_compound(
                            "cmpd_x",
                            _OSIMERTINIB,
                            warheads=["acrylamide"],
                            confidence="low_confidence",
                        ),
                    ]
                )
            ),
            encoding="utf-8",
        )

        changes = relabel_library(p)

        assert set(changes.keys()) == {"cmpd_021"}


# ── relabel_noise ─────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestRelabelNoise:
    def test_acrylate_ester_call_updated(self, tmp_path):
        p = tmp_path / "noise.json"
        # cmpd_021: was 'standard' + 'L858R_selective'; should flip to 'low_confidence_covalent'
        p.write_text(
            json.dumps(
                _noise_json(
                    [
                        _noise_compound(
                            "cmpd_021",
                            _CMPD_021,
                            delta=-0.522,
                            std_delta=0.044,
                            call="L858R_selective",
                        ),
                    ]
                )
            ),
            encoding="utf-8",
        )

        changes = relabel_noise(p)

        assert "cmpd_021" in changes
        result = json.loads(p.read_text(encoding="utf-8"))["compounds"][0]
        assert result["call"] == "low_confidence_covalent"
        assert result["docking_confidence"] == "low_confidence"
        assert "acrylate_ester" in result["warheads"]

    def test_non_covalent_call_preserved(self, tmp_path):
        p = tmp_path / "noise.json"
        p.write_text(
            json.dumps(
                _noise_json(
                    [
                        _noise_compound(
                            "cmpd_001",
                            _GEFITINIB,
                            delta=-0.813,
                            std_delta=0.277,
                            call="L858R_selective",
                        ),
                    ]
                )
            ),
            encoding="utf-8",
        )

        changes = relabel_noise(p)

        assert "cmpd_001" not in changes
        result = json.loads(p.read_text(encoding="utf-8"))["compounds"][0]
        assert result["call"] == "L858R_selective"

    def test_already_covalent_unchanged(self, tmp_path):
        p = tmp_path / "noise.json"
        p.write_text(
            json.dumps(
                _noise_json(
                    [
                        _noise_compound(
                            "cmpd_x",
                            _OSIMERTINIB,
                            delta=-0.8,
                            std_delta=0.1,
                            warheads=["acrylamide"],
                            confidence="low_confidence",
                            call="low_confidence_covalent",
                        ),
                    ]
                )
            ),
            encoding="utf-8",
        )

        changes = relabel_noise(p)

        assert "cmpd_x" not in changes

    def test_change_description_contains_call(self, tmp_path):
        p = tmp_path / "noise.json"
        p.write_text(
            json.dumps(
                _noise_json(
                    [
                        _noise_compound(
                            "cmpd_021", _CMPD_021, delta=-0.522, std_delta=0.044
                        ),
                    ]
                )
            ),
            encoding="utf-8",
        )
        changes = relabel_noise(p)
        assert "call" in changes["cmpd_021"]
