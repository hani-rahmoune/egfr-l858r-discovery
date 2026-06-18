"""
Unit tests for src/admet/filters.py.

Tests cover:
  - Schema completeness (all expected keys present)
  - Lipinski violation counting
  - Veber pass/fail
  - PAINS / Brenk structural alerts
  - QED range
  - admet_status logic
  - Invalid SMILES handling
  - Batch evaluation
  - summarize_admet aggregation

Test molecules:
  aspirin     CC(=O)Oc1ccccc1C(=O)O     — clean small drug, expect pass
  acrylamide  C=CC(=O)Nc1ccccc1          — N-phenylacrylamide; Michael acceptor
                                            should trigger Brenk structural alert
  gefitinib   COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1  — reference EGFR inhibitor
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.admet.filters import (
    _SA_AVAILABLE,
    QED_FLAG_BELOW,
    SA_FLAG_ABOVE,
    evaluate_admet,
    evaluate_admet_batch,
    summarize_admet,
)

# ── Test molecules ────────────────────────────────────────────────────────────

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ACRYLAMIDE = "C=CC(=O)Nc1ccccc1"  # N-phenylacrylamide; Brenk Michael acceptor
GEFITINIB = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
INVALID = "not_a_smiles$$"


# ── Expected result keys ──────────────────────────────────────────────────────

REQUIRED_KEYS = {
    "smiles",
    "valid",
    "mw",
    "logp",
    "hbd",
    "hba",
    "tpsa",
    "rotatable_bonds",
    "lipinski_violations",
    "lipinski_pass",
    "veber_pass",
    "pains_alerts",
    "pains_flag",
    "brenk_alerts",
    "brenk_flag",
    "qed",
    "sa_score",
    "range_mw_ok",
    "range_logp_ok",
    "range_tpsa_ok",
    "range_violations",
    "flag_reasons",
    "total_flags",
    "admet_status",
}


# ── TestSchema ────────────────────────────────────────────────────────────────


class TestSchema:

    @pytest.mark.unit
    def test_all_required_keys_present_valid(self):
        r = evaluate_admet(ASPIRIN)
        assert REQUIRED_KEYS.issubset(r.keys())

    @pytest.mark.unit
    def test_all_required_keys_present_invalid(self):
        r = evaluate_admet(INVALID)
        assert REQUIRED_KEYS.issubset(r.keys())

    @pytest.mark.unit
    def test_valid_flag_true_for_valid_smiles(self):
        assert evaluate_admet(ASPIRIN)["valid"] is True

    @pytest.mark.unit
    def test_valid_flag_false_for_invalid_smiles(self):
        assert evaluate_admet(INVALID)["valid"] is False

    @pytest.mark.unit
    def test_admet_status_is_pass_or_flag(self):
        for smi in [ASPIRIN, ACRYLAMIDE, GEFITINIB, INVALID]:
            status = evaluate_admet(smi)["admet_status"]
            assert status in ("pass", "flag"), f"Unexpected status: {status}"

    @pytest.mark.unit
    def test_pains_alerts_is_list(self):
        r = evaluate_admet(ASPIRIN)
        assert isinstance(r["pains_alerts"], list)

    @pytest.mark.unit
    def test_brenk_alerts_is_list(self):
        r = evaluate_admet(ASPIRIN)
        assert isinstance(r["brenk_alerts"], list)

    @pytest.mark.unit
    def test_flag_reasons_is_list(self):
        r = evaluate_admet(ASPIRIN)
        assert isinstance(r["flag_reasons"], list)

    @pytest.mark.unit
    def test_total_flags_equals_len_flag_reasons(self):
        for smi in [ASPIRIN, ACRYLAMIDE, GEFITINIB]:
            r = evaluate_admet(smi)
            assert r["total_flags"] == len(r["flag_reasons"])


# ── TestLipinski ──────────────────────────────────────────────────────────────


class TestLipinski:

    @pytest.mark.unit
    def test_aspirin_zero_violations(self):
        # aspirin MW~180, LogP~1.2, HBD=0, HBA=4 → all within Ro5
        r = evaluate_admet(ASPIRIN)
        assert r["lipinski_violations"] == 0
        assert r["lipinski_pass"] is True

    @pytest.mark.unit
    def test_violations_non_negative(self):
        r = evaluate_admet(GEFITINIB)
        assert r["lipinski_violations"] >= 0

    @pytest.mark.unit
    def test_violations_at_most_four(self):
        # four rules: MW, LogP, HBD, HBA
        r = evaluate_admet(GEFITINIB)
        assert r["lipinski_violations"] <= 4

    @pytest.mark.unit
    def test_lipinski_pass_consistent_with_violations(self):
        for smi in [ASPIRIN, ACRYLAMIDE, GEFITINIB]:
            r = evaluate_admet(smi)
            # lipinski_pass iff violations <= 1
            expected = r["lipinski_violations"] <= 1
            assert r["lipinski_pass"] == expected

    @pytest.mark.unit
    def test_mw_is_positive(self):
        r = evaluate_admet(ASPIRIN)
        assert r["mw"] > 0

    @pytest.mark.unit
    def test_aspirin_mw_reasonable(self):
        r = evaluate_admet(ASPIRIN)
        # aspirin ExactMolWt ~180; allow a few Da tolerance
        assert 175.0 < r["mw"] < 185.0

    @pytest.mark.unit
    def test_invalid_smiles_violations_is_none(self):
        r = evaluate_admet(INVALID)
        assert r["lipinski_violations"] is None


# ── TestVeber ─────────────────────────────────────────────────────────────────


class TestVeber:

    @pytest.mark.unit
    def test_aspirin_veber_pass(self):
        # aspirin: 3 rotbonds, TPSA ~63 → well within Veber
        r = evaluate_admet(ASPIRIN)
        assert r["veber_pass"] is True

    @pytest.mark.unit
    def test_tpsa_non_negative(self):
        r = evaluate_admet(GEFITINIB)
        assert r["tpsa"] >= 0

    @pytest.mark.unit
    def test_rotatable_bonds_non_negative(self):
        r = evaluate_admet(GEFITINIB)
        assert r["rotatable_bonds"] >= 0

    @pytest.mark.unit
    def test_veber_pass_consistent_with_properties(self):
        r = evaluate_admet(GEFITINIB)
        expected = r["rotatable_bonds"] <= 10 and r["tpsa"] <= 140
        assert r["veber_pass"] == expected


# ── TestStructuralAlerts ──────────────────────────────────────────────────────


class TestStructuralAlerts:

    @pytest.mark.unit
    def test_acrylamide_triggers_brenk(self):
        # N-phenylacrylamide: acrylamide = Michael acceptor → Brenk alert expected
        r = evaluate_admet(ACRYLAMIDE)
        assert (
            r["brenk_flag"] is True
        ), "N-phenylacrylamide (Michael acceptor) should trigger Brenk alert"

    @pytest.mark.unit
    def test_acrylamide_brenk_in_flag_reasons(self):
        r = evaluate_admet(ACRYLAMIDE)
        combined = " ".join(r["flag_reasons"])
        assert "Brenk" in combined

    @pytest.mark.unit
    def test_brenk_flag_is_bool(self):
        r = evaluate_admet(ASPIRIN)
        assert isinstance(r["brenk_flag"], bool)

    @pytest.mark.unit
    def test_pains_flag_is_bool(self):
        r = evaluate_admet(ASPIRIN)
        assert isinstance(r["pains_flag"], bool)

    @pytest.mark.unit
    def test_brenk_alerts_nonempty_when_flagged(self):
        r = evaluate_admet(ACRYLAMIDE)
        if r["brenk_flag"]:
            assert len(r["brenk_alerts"]) > 0

    @pytest.mark.unit
    def test_pains_alerts_nonempty_when_flagged(self):
        r = evaluate_admet(ASPIRIN)
        if r["pains_flag"]:
            assert len(r["pains_alerts"]) > 0

    @pytest.mark.unit
    def test_alert_descriptions_are_strings(self):
        r = evaluate_admet(ACRYLAMIDE)
        for alert in r["brenk_alerts"]:
            assert isinstance(alert, str)
            assert len(alert) > 0

    @pytest.mark.unit
    def test_invalid_smiles_no_alerts(self):
        r = evaluate_admet(INVALID)
        assert r["brenk_alerts"] == []
        assert r["pains_alerts"] == []


# ── TestQED ───────────────────────────────────────────────────────────────────


class TestQED:

    @pytest.mark.unit
    def test_qed_in_zero_one_range(self):
        for smi in [ASPIRIN, ACRYLAMIDE, GEFITINIB]:
            r = evaluate_admet(smi)
            assert 0.0 <= r["qed"] <= 1.0

    @pytest.mark.unit
    def test_aspirin_qed_above_threshold(self):
        r = evaluate_admet(ASPIRIN)
        # Aspirin is a known drug; QED should be reasonable
        assert r["qed"] >= QED_FLAG_BELOW

    @pytest.mark.unit
    def test_qed_flagged_in_reasons_when_low(self):
        # Construct a scenario: if qed < threshold, it should appear in reasons
        # We test this indirectly via the flag_reasons logic check
        r = evaluate_admet(ASPIRIN)
        qed_flagged = any("QED" in reason for reason in r["flag_reasons"])
        assert qed_flagged == (r["qed"] < QED_FLAG_BELOW)

    @pytest.mark.unit
    def test_invalid_smiles_qed_is_none(self):
        r = evaluate_admet(INVALID)
        assert r["qed"] is None


# ── TestSAScore ───────────────────────────────────────────────────────────────


class TestSAScore:

    @pytest.mark.unit
    def test_sa_score_is_float_or_none(self):
        r = evaluate_admet(ASPIRIN)
        assert r["sa_score"] is None or isinstance(r["sa_score"], float)

    @pytest.mark.unit
    @pytest.mark.skipif(not _SA_AVAILABLE, reason="SA scorer not installed")
    def test_sa_score_in_valid_range(self):
        r = evaluate_admet(ASPIRIN)
        assert r["sa_score"] is not None
        assert 1.0 <= r["sa_score"] <= 10.0

    @pytest.mark.unit
    @pytest.mark.skipif(not _SA_AVAILABLE, reason="SA scorer not installed")
    def test_aspirin_sa_score_reasonable(self):
        r = evaluate_admet(ASPIRIN)
        # Aspirin is simple; SA score should be well below the "hard" threshold
        assert r["sa_score"] < SA_FLAG_ABOVE


# ── TestAdmetStatus ───────────────────────────────────────────────────────────


class TestAdmetStatus:

    @pytest.mark.unit
    def test_flagged_when_brenk_alert(self):
        r = evaluate_admet(ACRYLAMIDE)
        if r["brenk_flag"]:
            assert r["admet_status"] == "flag"

    @pytest.mark.unit
    def test_pass_implies_no_flag_reasons(self):
        r = evaluate_admet(ASPIRIN)
        if r["admet_status"] == "pass":
            assert r["flag_reasons"] == []
            assert r["total_flags"] == 0

    @pytest.mark.unit
    def test_flag_implies_nonempty_reasons(self):
        for smi in [ASPIRIN, ACRYLAMIDE, GEFITINIB]:
            r = evaluate_admet(smi)
            if r["admet_status"] == "flag":
                assert len(r["flag_reasons"]) > 0

    @pytest.mark.unit
    def test_invalid_smiles_is_flagged(self):
        r = evaluate_admet(INVALID)
        assert r["admet_status"] == "flag"
        assert "Invalid SMILES" in r["flag_reasons"][0]


# ── TestBatch ─────────────────────────────────────────────────────────────────


class TestBatch:

    @pytest.mark.unit
    def test_batch_returns_correct_length(self):
        smiles = [ASPIRIN, ACRYLAMIDE, GEFITINIB, INVALID]
        results = evaluate_admet_batch(smiles)
        assert len(results) == len(smiles)

    @pytest.mark.unit
    def test_batch_preserves_order(self):
        smiles = [ASPIRIN, INVALID, GEFITINIB]
        results = evaluate_admet_batch(smiles)
        assert results[0]["smiles"] == ASPIRIN
        assert results[1]["smiles"] == INVALID
        assert results[2]["smiles"] == GEFITINIB

    @pytest.mark.unit
    def test_batch_empty_list(self):
        assert evaluate_admet_batch([]) == []


# ── TestSummarize ─────────────────────────────────────────────────────────────


class TestSummarize:

    @pytest.mark.unit
    def test_summary_keys(self):
        results = evaluate_admet_batch([ASPIRIN, ACRYLAMIDE])
        s = summarize_admet(results)
        for key in (
            "n_total",
            "n_valid",
            "n_pass",
            "n_flag",
            "pass_rate",
            "pains_frequency",
            "brenk_frequency",
            "median_qed",
        ):
            assert key in s

    @pytest.mark.unit
    def test_n_pass_plus_n_flag_equals_n_valid(self):
        results = evaluate_admet_batch([ASPIRIN, ACRYLAMIDE, GEFITINIB, INVALID])
        s = summarize_admet(results)
        assert s["n_pass"] + s["n_flag"] == s["n_valid"]

    @pytest.mark.unit
    def test_median_qed_in_range(self):
        results = evaluate_admet_batch([ASPIRIN, GEFITINIB])
        s = summarize_admet(results)
        assert s["median_qed"] is not None
        assert 0.0 <= s["median_qed"] <= 1.0

    @pytest.mark.unit
    def test_brenk_frequency_counts_acrylamide(self):
        results = evaluate_admet_batch([ACRYLAMIDE, ACRYLAMIDE, ASPIRIN])
        s = summarize_admet(results)
        # Two acrylamide molecules: total Brenk count should be >= 2
        total_brenk = sum(s["brenk_frequency"].values())
        assert total_brenk >= 2

    @pytest.mark.unit
    def test_empty_results(self):
        s = summarize_admet([])
        assert s["n_total"] == 0
        assert s["pass_rate"] is None

    @pytest.mark.unit
    def test_all_invalid_smiles(self):
        results = evaluate_admet_batch([INVALID, INVALID])
        s = summarize_admet(results)
        assert s["n_valid"] == 0
        assert s["n_pass"] == 0
