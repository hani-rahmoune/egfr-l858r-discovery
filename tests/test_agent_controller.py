"""
Tests for src/agent/controller.py: intent classification and dispatch.

Uses a mock registry; no real model loading.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.controller import classify_intent, handle
from src.agent.schemas import AgentRequest, AgentResponse


# ── Mock registry ──────────────────────────────────────────────────────────────


class _MockRegistry:
    def score(self, smiles: str) -> dict:
        from rdkit import Chem
        from src.features.covalent import detect_warheads
        from src.scoring.ranking import build_warnings

        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return {
                "smiles": smiles, "canonical_smiles": None, "valid": False,
                "pic50_mutant": None, "pic50_wt": None, "selectivity_proxy": None,
                "covalent": False, "warheads": [],
                "admet": None, "applicability_domain": None,
                "docking_selectivity_available": False,
                "warnings": ["Invalid SMILES: RDKit could not parse the input."],
            }
        canonical = Chem.MolToSmiles(mol)
        warheads = detect_warheads(canonical)
        domain = "in_domain"
        return {
            "smiles": smiles, "canonical_smiles": canonical, "valid": True,
            "pic50_mutant": 7.5, "pic50_wt": 7.0, "selectivity_proxy": 0.5,
            "covalent": bool(warheads), "warheads": warheads,
            "admet": {"status": "pass", "qed": 0.65, "sa_score": 2.5,
                      "lipinski_pass": True, "veber_pass": True,
                      "pains_alerts": [], "brenk_alerts": [], "flag_reasons": []},
            "applicability_domain": {"domain": domain, "max_tanimoto": 0.75,
                                     "confidence_factor": 1.0},
            "docking_selectivity_available": False,
            "warnings": build_warnings(domain, bool(warheads), warheads, None),
        }


@pytest.fixture
def reg():
    return _MockRegistry()


# ── classify_intent ───────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("query,expected", [
    ("Predict the activity for this SMILES: CCO", "single_predict"),
    ("Score this molecule: c1ccccc1", "single_predict"),
    ("batch screen these SMILES", "batch_predict"),
    ("Compare cmpd_015 and cmpd_024", "comparison"),
    ("cmpd_015 vs cmpd_002 which is better", "comparison"),
    ("Look up cmpd_024 in the ranking", "candidate_lookup"),
    ("What is the rank of gen_005?", "candidate_lookup"),
    ("docking results for cmpd_015", "docking_query"),
    ("What is the Vina score for gen_005?", "docking_query"),
    ("Generate a report for cmpd_015", "report"),
    ("summary for cmpd_002", "report"),
    ("What is LOOCV?", "project_qa"),
    ("How does the scaffold split work?", "project_qa"),
    ("Why did RL fail?", "project_qa"),
    ("Explain the ADMET filters", "project_qa"),
])
def test_classify_intent(query, expected):
    assert classify_intent(query) == expected


@pytest.mark.unit
def test_classify_unknown():
    result = classify_intent("gibberish xyz zqp")
    assert result == "unknown"


# ── handle: single_predict ────────────────────────────────────────────────────


@pytest.mark.unit
def test_handle_single_predict(reg):
    req = AgentRequest(
        query="Predict activity", smiles="COc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OC"
    )
    resp = handle(req, registry=reg)
    assert isinstance(resp, AgentResponse)
    assert resp.intent == "single_predict"
    assert "EXPLORATORY" in resp.answer or "7.5" in resp.answer


@pytest.mark.unit
def test_handle_single_predict_invalid_smiles(reg):
    req = AgentRequest(query="Predict activity", smiles="not_a_smiles$$")
    resp = handle(req, registry=reg)
    assert resp.intent == "single_predict"
    assert "Invalid" in resp.answer or "invalid" in resp.answer


@pytest.mark.unit
def test_handle_single_predict_no_smiles(reg):
    req = AgentRequest(query="Predict activity for the molecule")
    resp = handle(req, registry=reg)
    assert resp.intent == "single_predict"
    # Should ask for SMILES
    assert "SMILES" in resp.answer or "smiles" in resp.answer.lower()


# ── handle: batch_predict ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_handle_batch_no_smiles(reg):
    req = AgentRequest(query="batch screen these SMILES")
    resp = handle(req, registry=reg)
    assert resp.intent == "batch_predict"
    assert "SMILES" in resp.answer or "smiles" in resp.answer.lower()


# ── handle: candidate_lookup ──────────────────────────────────────────────────


@pytest.mark.unit
def test_handle_candidate_lookup_known(reg):
    req = AgentRequest(query="Look up cmpd_015 in the ranking", candidate_ids=["cmpd_015"])
    resp = handle(req, registry=reg)
    assert resp.intent == "candidate_lookup"
    assert "cmpd_015" in resp.answer


@pytest.mark.unit
def test_handle_candidate_lookup_missing(reg):
    req = AgentRequest(query="Look up cmpd_999", candidate_ids=["cmpd_999"])
    resp = handle(req, registry=reg)
    assert resp.intent == "candidate_lookup"
    assert "cmpd_999" in resp.answer


# ── handle: docking_query ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_handle_docking_known(reg):
    req = AgentRequest(query="docking results", candidate_ids=["cmpd_024"])
    resp = handle(req, registry=reg)
    assert resp.intent == "docking_query"
    assert "cmpd_024" in resp.answer or "L858R" in resp.answer


@pytest.mark.unit
def test_handle_docking_no_cid(reg):
    req = AgentRequest(query="What are the docking results?")
    resp = handle(req, registry=reg)
    assert resp.intent == "docking_query"
    # Should ask for a candidate ID
    assert "candidate" in resp.answer.lower() or "ID" in resp.answer


# ── handle: comparison ────────────────────────────────────────────────────────


@pytest.mark.unit
def test_handle_comparison(reg):
    req = AgentRequest(
        query="Compare cmpd_015 and cmpd_024",
        candidate_ids=["cmpd_015", "cmpd_024"],
    )
    resp = handle(req, registry=reg)
    assert resp.intent == "comparison"
    assert resp.answer  # non-empty
    assert "Recommendation" in resp.answer or "Preferring" in resp.answer


@pytest.mark.unit
def test_handle_comparison_too_few_ids(reg):
    req = AgentRequest(
        query="Compare cmpd_015 vs something",
        candidate_ids=["cmpd_015"],
    )
    resp = handle(req, registry=reg)
    assert resp.intent == "comparison"
    assert "two" in resp.answer.lower() or "2" in resp.answer or "least" in resp.answer


# ── handle: report ────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_handle_report(reg):
    req = AgentRequest(
        query="Generate a report for cmpd_015",
        candidate_ids=["cmpd_015"],
    )
    resp = handle(req, registry=reg)
    assert resp.intent == "report"
    assert "## Limitations" in resp.answer
    assert "EXPLORATORY" in resp.answer


@pytest.mark.unit
def test_handle_report_no_forbidden_claims(reg):
    from src.agent.guardrails import find_forbidden_claims
    req = AgentRequest(
        query="Generate a report for cmpd_015",
        candidate_ids=["cmpd_015"],
    )
    resp = handle(req, registry=reg)
    # Guardrail warning may appear in warnings but should NOT be in the report body itself
    # (guardrail fires on the answer text; report template is clean)
    # Check only the main answer sections, not the guardrail warning in resp.warnings
    claims = find_forbidden_claims(resp.answer)
    assert claims == [], f"Forbidden claims in report answer: {claims}"


# ── handle: project_qa ────────────────────────────────────────────────────────


@pytest.mark.unit
def test_handle_project_qa(reg):
    req = AgentRequest(query="What is LOOCV and why was it used?")
    resp = handle(req, registry=reg)
    assert resp.intent == "project_qa"
    assert resp.answer
    assert len(resp.sources) > 0


@pytest.mark.unit
def test_handle_project_qa_rl(reg):
    req = AgentRequest(query="Why did the RL training fail?")
    resp = handle(req, registry=reg)
    assert resp.intent == "project_qa"
    assert resp.answer


# ── handle: unknown ───────────────────────────────────────────────────────────


@pytest.mark.unit
def test_handle_unknown(reg):
    req = AgentRequest(query="zqpxyz gibberish 12345")
    resp = handle(req, registry=reg)
    assert resp.intent == "unknown"
    assert resp.answer  # falls back to retrieval or a helpful message


# ── AgentResponse structure ───────────────────────────────────────────────────


@pytest.mark.unit
def test_response_fields_present(reg):
    req = AgentRequest(query="Look up cmpd_002", candidate_ids=["cmpd_002"])
    resp = handle(req, registry=reg)
    assert hasattr(resp, "intent")
    assert hasattr(resp, "answer")
    assert hasattr(resp, "tool_results")
    assert hasattr(resp, "warnings")
    assert hasattr(resp, "sources")
    assert isinstance(resp.warnings, list)
    assert isinstance(resp.tool_results, list)
