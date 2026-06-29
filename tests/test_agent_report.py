"""
Tests for src/agent/report.py: template generation.

Verifies: report contains a limitations section, makes no forbidden experimental
claims, and handles missing data gracefully.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.guardrails import find_forbidden_claims
from src.agent.report import generate_report
from src.agent.schemas import (
    CandidateReport,
    DockingLookupResult,
    PredictToolResult,
    RankingLookupResult,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_ranking(cid: str = "cmpd_015", rank: int = 1) -> RankingLookupResult:
    return RankingLookupResult(
        found=True,
        candidate_id=cid,
        rank=rank,
        source="known",
        smiles="CN(C)c1cc2ncnc(Nc3cccc(Br)c3)c2cn1",
        final_score=0.730,
        activity_norm=0.71,
        selectivity_norm=0.83,
        affinity_norm=0.35,
        admet_norm=1.0,
        confidence_factor=1.0,
        is_covalent=False,
    )


def _make_predict(cid: str = "cmpd_015") -> PredictToolResult:
    return PredictToolResult(
        valid=True,
        smiles="CN(C)c1cc2ncnc(Nc3cccc(Br)c3)c2cn1",
        canonical_smiles="CN(C)c1cc2ncnc(Nc3cccc(Br)c3)c2cn1",
        pic50_mutant=8.816,
        pic50_wt=8.182,
        selectivity_proxy=0.634,
        covalent=False,
        warheads=[],
        admet_status="pass",
        qed=0.787,
        admet_alerts=[],
        domain="in_domain",
        confidence_factor=1.0,
        warnings=[],
    )


def _make_docking(cid: str = "cmpd_015") -> DockingLookupResult:
    return DockingLookupResult(
        found=True,
        candidate_id=cid,
        l858r_score=-7.552,
        wt_score=-6.918,
        selectivity_delta=-0.634,
        direction="L858R_favoured",
        mean_delta=-0.452,
        std_delta=0.139,
        noise_call="L858R_selective",
        docking_confidence="standard",
        data_source="library",
    )


# ── Structure tests ───────────────────────────────────────────────────────────


@pytest.mark.unit
def test_report_returns_candidate_report():
    report = generate_report(
        "cmpd_015", _make_ranking(), _make_docking(), _make_predict()
    )
    assert isinstance(report, CandidateReport)


@pytest.mark.unit
def test_report_contains_candidate_id():
    report = generate_report(
        "cmpd_015", _make_ranking(), _make_docking(), _make_predict()
    )
    assert "cmpd_015" in report.markdown


@pytest.mark.unit
def test_report_has_limitations_section():
    report = generate_report(
        "cmpd_015", _make_ranking(), _make_docking(), _make_predict()
    )
    assert "## Limitations" in report.markdown


@pytest.mark.unit
def test_report_has_exploratory_banner():
    report = generate_report(
        "cmpd_015", _make_ranking(), _make_docking(), _make_predict()
    )
    assert "EXPLORATORY" in report.markdown


@pytest.mark.unit
def test_report_has_summary_section():
    report = generate_report(
        "cmpd_015", _make_ranking(), _make_docking(), _make_predict()
    )
    assert "## Summary" in report.markdown


@pytest.mark.unit
def test_report_has_docking_section():
    report = generate_report(
        "cmpd_015", _make_ranking(), _make_docking(), _make_predict()
    )
    assert "## Docking" in report.markdown


@pytest.mark.unit
def test_report_has_admet_section():
    report = generate_report(
        "cmpd_015", _make_ranking(), _make_docking(), _make_predict()
    )
    assert "## ADMET" in report.markdown


@pytest.mark.unit
def test_report_has_ranking_section():
    report = generate_report(
        "cmpd_015", _make_ranking(), _make_docking(), _make_predict()
    )
    assert "## Composite Ranking" in report.markdown


# ── Forbidden claim tests ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_report_no_forbidden_claims_full():
    report = generate_report(
        "cmpd_015", _make_ranking(), _make_docking(), _make_predict()
    )
    claims = find_forbidden_claims(report.markdown)
    assert (
        claims == []
    ), f"Forbidden claims in report: {claims}\n\n{report.markdown[:800]}"


@pytest.mark.unit
def test_limitations_no_forbidden_claims():
    """The limitations block must itself be claim-free."""
    from src.agent.report import _LIMITATIONS_BLOCK

    claims = find_forbidden_claims(_LIMITATIONS_BLOCK)
    assert claims == [], f"Forbidden claims in limitations block: {claims}"


# ── Missing data graceful handling ────────────────────────────────────────────


@pytest.mark.unit
def test_report_none_ranking():
    """Report with no ranking data should still generate with limitations."""
    not_found = RankingLookupResult(found=False, candidate_id="cmpd_999")
    report = generate_report("cmpd_999", not_found, None, None)
    assert "## Limitations" in report.markdown
    assert "cmpd_999" in report.markdown


@pytest.mark.unit
def test_report_none_docking():
    report = generate_report("cmpd_015", _make_ranking(), None, _make_predict())
    assert "## Limitations" in report.markdown
    # Should mention no docking data
    assert (
        "No docking data" in report.markdown
        or "unavailable" in report.markdown
        or "not found" in report.markdown
    )


@pytest.mark.unit
def test_report_none_predict():
    report = generate_report("cmpd_015", _make_ranking(), _make_docking(), None)
    assert "## Limitations" in report.markdown


@pytest.mark.unit
def test_report_all_none():
    not_found = RankingLookupResult(found=False, candidate_id="cmpd_999")
    report = generate_report("cmpd_999", not_found, None, None)
    assert isinstance(report, CandidateReport)
    assert "## Limitations" in report.markdown


@pytest.mark.unit
def test_report_invalid_predict():
    """Invalid (unparseable) SMILES predict result should not crash report."""
    bad_predict = PredictToolResult(
        valid=False, smiles="bad_smiles", error="Invalid SMILES"
    )
    report = generate_report("cmpd_015", _make_ranking(), _make_docking(), bad_predict)
    assert "## Limitations" in report.markdown


# ── Numeric accuracy ──────────────────────────────────────────────────────────


@pytest.mark.unit
def test_report_includes_pic50():
    report = generate_report(
        "cmpd_015", _make_ranking(), _make_docking(), _make_predict()
    )
    assert "8.816" in report.markdown


@pytest.mark.unit
def test_report_includes_rank():
    report = generate_report(
        "cmpd_015", _make_ranking(), _make_docking(), _make_predict()
    )
    assert "1/68" in report.markdown


@pytest.mark.unit
def test_report_includes_docking_delta():
    report = generate_report(
        "cmpd_015", _make_ranking(), _make_docking(), _make_predict()
    )
    assert "-0.634" in report.markdown or "-0.452" in report.markdown


# ── Warnings passthrough ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_report_propagates_warnings():
    predict = _make_predict()
    predict.warnings = ["test warning for unit test"]
    report = generate_report("cmpd_015", _make_ranking(), _make_docking(), predict)
    assert "test warning for unit test" in report.warnings
