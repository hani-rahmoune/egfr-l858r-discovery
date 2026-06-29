"""
Tests for src/agent/tools.py: happy-path + missing-value paths.

All tests are @unit (no real model loading). The ModelRegistry is replaced with
_MockRegistry, identical in shape to the one used in test_api.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.schemas import (
    BatchPredictToolResult,
    ComparisonResult,
    DockingLookupResult,
    PredictToolResult,
    RankingLookupResult,
)
from src.agent.tools import (
    _clear_docking_index,
    batch_predict,
    compare_candidates,
    generate_candidate_report,
    lookup_docking_results,
    lookup_final_ranking,
    predict_smiles,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

GEFITINIB = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
INVALID = "not_a_smiles$$"

# Candidates that exist in the precomputed artifacts
KNOWN_CID = "cmpd_015"
KNOWN_GEN = "gen_005"
MISSING_CID = "cmpd_999"


class _MockRegistry:
    """Canned-response registry, no artifact loading required."""

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
        is_cov = bool(warheads)
        domain = "in_domain"
        warnings = build_warnings(domain, is_cov, warheads, None)
        return {
            "smiles": smiles, "canonical_smiles": canonical, "valid": True,
            "pic50_mutant": 7.5, "pic50_wt": 7.0, "selectivity_proxy": 0.5,
            "covalent": is_cov, "warheads": warheads,
            "admet": {"status": "pass", "qed": 0.65, "sa_score": 2.5,
                      "lipinski_pass": True, "veber_pass": True,
                      "pains_alerts": [], "brenk_alerts": [], "flag_reasons": []},
            "applicability_domain": {"domain": domain, "max_tanimoto": 0.75,
                                     "confidence_factor": 1.0},
            "docking_selectivity_available": False,
            "warnings": warnings,
        }


@pytest.fixture
def reg():
    return _MockRegistry()


# ── predict_smiles ─────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_predict_valid_smiles(reg):
    result = predict_smiles(GEFITINIB, registry=reg)
    assert isinstance(result, PredictToolResult)
    assert result.valid is True
    assert result.canonical_smiles is not None
    assert result.pic50_mutant == pytest.approx(7.5)
    assert result.pic50_wt == pytest.approx(7.0)
    assert result.selectivity_proxy == pytest.approx(0.5)
    assert result.domain == "in_domain"
    assert result.confidence_factor == pytest.approx(1.0)
    assert isinstance(result.warnings, list)


@pytest.mark.unit
def test_predict_invalid_smiles(reg):
    result = predict_smiles(INVALID, registry=reg)
    assert result.valid is False
    assert result.error is not None
    assert result.pic50_mutant is None
    assert result.pic50_wt is None
    assert result.selectivity_proxy is None


@pytest.mark.unit
def test_predict_empty_smiles(reg):
    result = predict_smiles("", registry=reg)
    assert result.valid is False


@pytest.mark.unit
def test_predict_covalent_smiles(reg):
    """Osimertinib has an acrylamide warhead; covalent should be True."""
    osimertinib = "C=CC(=O)Nc1cc2c(Nc3cccc(NC(=O)/C=C/CN(C)C)c3)ncnc2cc1OC"
    result = predict_smiles(osimertinib, registry=reg)
    assert result.valid is True
    assert result.covalent is True
    assert "acrylamide" in result.warheads


@pytest.mark.unit
def test_predict_warnings_populated(reg):
    """Valid molecule should carry at least the selectivity-proxy caveat."""
    result = predict_smiles(GEFITINIB, registry=reg)
    assert any("selectivity_proxy" in w or "exploratory" in w.lower() or "EXPLORATORY" in w
               for w in result.warnings), result.warnings


# ── batch_predict ──────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_batch_predict_mixed(reg):
    smiles = [GEFITINIB, INVALID]
    batch = batch_predict(smiles, registry=reg)
    assert isinstance(batch, BatchPredictToolResult)
    assert batch.n == 2
    assert batch.n_valid == 1
    assert batch.n_invalid == 1
    assert len(batch.results) == 2


@pytest.mark.unit
def test_batch_predict_cap(reg):
    from src.agent.tools import MAX_BATCH
    with pytest.raises(ValueError, match="exceeds limit"):
        batch_predict(["CC"] * (MAX_BATCH + 1), registry=reg)


@pytest.mark.unit
def test_batch_predict_all_valid(reg):
    smiles = [GEFITINIB, "CC(=O)Oc1ccccc1C(=O)O"]
    batch = batch_predict(smiles, registry=reg)
    assert batch.n_valid == 2
    assert batch.n_invalid == 0


# ── lookup_final_ranking ──────────────────────────────────────────────────────

@pytest.mark.unit
def test_lookup_ranking_known():
    result = lookup_final_ranking(KNOWN_CID)
    assert isinstance(result, RankingLookupResult)
    assert result.found is True
    assert result.candidate_id == KNOWN_CID
    assert result.rank is not None and result.rank >= 1
    assert result.final_score is not None
    assert result.source in ("known", "generated")


@pytest.mark.unit
def test_lookup_ranking_generated():
    result = lookup_final_ranking(KNOWN_GEN)
    assert result.found is True
    assert result.source == "generated"
    assert result.rank == 21


@pytest.mark.unit
def test_lookup_ranking_missing():
    result = lookup_final_ranking(MISSING_CID)
    assert result.found is False
    assert result.rank is None
    assert result.final_score is None


@pytest.mark.unit
def test_lookup_ranking_returns_smiles():
    result = lookup_final_ranking(KNOWN_CID)
    assert result.found is True
    assert result.smiles is not None and len(result.smiles) > 5


# ── lookup_docking_results ────────────────────────────────────────────────────

@pytest.mark.unit
def test_lookup_docking_known():
    _clear_docking_index()
    result = lookup_docking_results(KNOWN_CID)
    assert isinstance(result, DockingLookupResult)
    assert result.found is True
    assert result.l858r_score is not None
    assert result.wt_score is not None
    assert result.selectivity_delta is not None


@pytest.mark.unit
def test_lookup_docking_cmpd024_has_noise():
    """cmpd_024 is in the top-15 noise study."""
    _clear_docking_index()
    result = lookup_docking_results("cmpd_024")
    assert result.found is True
    assert result.mean_delta is not None
    assert result.std_delta is not None
    assert result.noise_call is not None


@pytest.mark.unit
def test_lookup_docking_generated():
    _clear_docking_index()
    result = lookup_docking_results(KNOWN_GEN)
    assert result.found is True
    assert result.data_source == "generated"


@pytest.mark.unit
def test_lookup_docking_missing():
    _clear_docking_index()
    result = lookup_docking_results(MISSING_CID)
    assert result.found is False
    assert result.l858r_score is None
    assert result.wt_score is None
    # message must say no docking data, not fabricate a number
    assert result.message is not None
    assert result.selectivity_delta is None


@pytest.mark.unit
def test_docking_direction_consistent():
    """Direction field must match sign of selectivity_delta."""
    _clear_docking_index()
    result = lookup_docking_results("cmpd_015")
    assert result.found is True
    if result.selectivity_delta is not None:
        if result.selectivity_delta < 0:
            assert result.direction == "L858R_favoured"
        else:
            assert result.direction == "WT_favoured"


# ── compare_candidates ────────────────────────────────────────────────────────

@pytest.mark.unit
def test_compare_two_known(reg):
    result = compare_candidates(["cmpd_015", "cmpd_024"], registry=reg)
    assert isinstance(result, ComparisonResult)
    assert result.recommendation in ("cmpd_015", "cmpd_024")
    assert result.reason
    assert "cmpd_015" in result.scores
    assert "cmpd_024" in result.scores


@pytest.mark.unit
def test_compare_prefers_noncovalent(reg):
    """cmpd_015 (non-covalent, rank 1) should beat cmpd_008 (covalent)."""
    result = compare_candidates(["cmpd_008", "cmpd_015"], registry=reg)
    assert result.recommendation == "cmpd_015"


@pytest.mark.unit
def test_compare_requires_two(reg):
    with pytest.raises(ValueError, match="at least 2"):
        compare_candidates(["cmpd_015"], registry=reg)


@pytest.mark.unit
def test_compare_three_candidates(reg):
    result = compare_candidates(["cmpd_015", "cmpd_002", "cmpd_024"], registry=reg)
    assert result.recommendation in ("cmpd_015", "cmpd_002", "cmpd_024")
    assert len(result.candidate_ids) == 3


# ── generate_candidate_report ─────────────────────────────────────────────────

@pytest.mark.unit
def test_report_known_candidate(reg):
    from src.agent.schemas import CandidateReport
    result = generate_candidate_report(KNOWN_CID, registry=reg)
    assert isinstance(result, CandidateReport)
    assert KNOWN_CID in result.markdown
    assert "## Limitations" in result.markdown
    assert "EXPLORATORY" in result.markdown


@pytest.mark.unit
def test_report_missing_candidate(reg):
    """Missing candidate should still return a report with a not-found note."""
    result = generate_candidate_report(MISSING_CID, registry=reg)
    assert MISSING_CID in result.markdown
    assert "## Limitations" in result.markdown


@pytest.mark.unit
def test_report_no_forbidden_claims(reg):
    from src.agent.guardrails import find_forbidden_claims
    result = generate_candidate_report(KNOWN_CID, registry=reg)
    claims = find_forbidden_claims(result.markdown)
    assert claims == [], f"Forbidden claims found: {claims}\n---\n{result.markdown[:500]}"
