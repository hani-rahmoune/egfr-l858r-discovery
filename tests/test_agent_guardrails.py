"""
Tests for src/agent/guardrails.py: scientific warning injection and
forbidden-claim detection (including negation pass-through).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.guardrails import (
    _BORDERLINE_CAVEAT,
    _GENERATED_CAVEAT,
    _OOD_CAVEAT,
    _SELECTIVITY_CAVEAT,
    add_scientific_warnings,
    find_forbidden_claims,
    sanitize_text,
)
from src.agent.schemas import DockingLookupResult, PredictToolResult

# ── add_scientific_warnings ────────────────────────────────────────────────────


@pytest.mark.unit
def test_selectivity_caveat_added():
    result = PredictToolResult(
        valid=True, smiles="C", selectivity_proxy=0.5, warnings=[]
    )
    warnings = add_scientific_warnings(result)
    assert any("selectivity_proxy" in w for w in warnings)


@pytest.mark.unit
def test_selectivity_caveat_not_duplicated():
    """Caveat must not appear twice if already present."""
    result = PredictToolResult(
        valid=True, smiles="C", selectivity_proxy=0.5, warnings=[_SELECTIVITY_CAVEAT]
    )
    warnings = add_scientific_warnings(result)
    assert warnings.count(_SELECTIVITY_CAVEAT) == 1


@pytest.mark.unit
def test_no_selectivity_caveat_when_none():
    result = PredictToolResult(valid=True, smiles="C", selectivity_proxy=None)
    warnings = add_scientific_warnings(result)
    assert not any("selectivity_proxy" in w for w in warnings)


@pytest.mark.unit
def test_covalent_caveat_added():
    result = PredictToolResult(
        valid=True,
        smiles="C=CC(=O)NC",
        covalent=True,
        warheads=["acrylamide"],
        warnings=[],
    )
    warnings = add_scientific_warnings(result)
    assert any("covalent" in w.lower() or "warhead" in w.lower() for w in warnings)


@pytest.mark.unit
def test_out_of_domain_caveat():
    result = PredictToolResult(
        valid=True, smiles="C", domain="out_of_domain", warnings=[]
    )
    warnings = add_scientific_warnings(result)
    assert _OOD_CAVEAT in warnings


@pytest.mark.unit
def test_borderline_caveat():
    result = PredictToolResult(valid=True, smiles="C", domain="borderline", warnings=[])
    warnings = add_scientific_warnings(result)
    assert _BORDERLINE_CAVEAT in warnings


@pytest.mark.unit
def test_in_domain_no_domain_caveat():
    result = PredictToolResult(valid=True, smiles="C", domain="in_domain", warnings=[])
    warnings = add_scientific_warnings(result)
    assert _OOD_CAVEAT not in warnings
    assert _BORDERLINE_CAVEAT not in warnings


@pytest.mark.unit
def test_generated_caveat():
    result = PredictToolResult(valid=True, smiles="C", warnings=[])
    result.source = "generated"  # type: ignore[attr-defined]
    warnings = add_scientific_warnings(result)
    assert _GENERATED_CAVEAT in warnings


@pytest.mark.unit
def test_docking_caveat_from_docking_result():
    """DockingLookupResult with l858r_score should trigger the docking caveat."""
    result = DockingLookupResult(found=True, candidate_id="cmpd_015", l858r_score=-7.5)
    warnings = add_scientific_warnings(result)
    assert any(
        "rigid" in w.lower() or "vina" in w.lower() or "docking" in w.lower()
        for w in warnings
    )


@pytest.mark.unit
def test_warnings_list_returned_not_mutated():
    """add_scientific_warnings must return a new list, not mutate the result."""
    original = ["existing warning"]
    result = PredictToolResult(valid=True, smiles="C", warnings=original)
    new_warnings = add_scientific_warnings(result)
    assert new_warnings is not result.warnings
    assert "existing warning" in new_warnings


# ── find_forbidden_claims ─────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "text,expected_claim",
    [
        ("This compound is active against EGFR.", "is active"),
        ("The molecule is selective for L858R.", "is selective"),
        ("This is a drug candidate for NSCLC.", "drug candidate"),
        ("The result was validated in a cell assay.", "validated"),
        ("This finding is proven by experiment.", "proven"),
        ("Binding was confirmed by SPR.", "confirmed"),
    ],
)
def test_forbidden_claim_detected(text, expected_claim):
    found = find_forbidden_claims(text)
    assert expected_claim in found, f"Expected '{expected_claim}' in {found}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        "This compound is not active in any assay.",
        "The molecule is not selective for L858R.",
        "This is not a drug candidate.",
        "The result was not validated experimentally.",
        "This is unproven in cell lines.",
        "Binding was not confirmed.",
        "No confirmed activity has been established.",
        "The pipeline has not been validated experimentally.",
    ],
)
def test_negated_claim_passes(text):
    found = find_forbidden_claims(text)
    assert found == [], f"Unexpected claims found in '{text}': {found}"


@pytest.mark.unit
def test_multiple_claims_detected():
    text = "This molecule is active and is selective; it was validated in vivo."
    found = find_forbidden_claims(text)
    assert len(found) >= 2


@pytest.mark.unit
def test_clean_exploratory_text_passes():
    text = (
        "EXPLORATORY: the backbone predicts pIC50=7.5. "
        "The selectivity proxy is 0.5 but this is not validated. "
        "No experimental confirmation is available."
    )
    found = find_forbidden_claims(text)
    assert found == []


# ── sanitize_text ──────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_sanitize_returns_text_unchanged():
    text = "This is active."
    returned_text, claims = sanitize_text(text)
    assert returned_text == text


@pytest.mark.unit
def test_sanitize_reports_claims():
    _, claims = sanitize_text("This compound is active and has been validated.")
    assert "is active" in claims
    assert "validated" in claims


@pytest.mark.unit
def test_sanitize_clean_text():
    _, claims = sanitize_text("EXPLORATORY prediction only. Not validated.")
    assert claims == []
