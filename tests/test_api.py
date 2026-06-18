"""
Tests for the Phase 24 FastAPI fast-screen service.

Layers:
  TestSchemas          — pydantic request/response validation (no app)
  TestEndpointsMock    — full endpoint behaviour against a MOCK registry
                         (no real model artifacts loaded; fast, @unit)
  TestStructuredErrors — invalid SMILES / validation produce {error, detail}
  TestModelsUnavailable— 503 when the registry is absent
  TestEndpointsReal    — smoke tests against the REAL loaded models (@integration)

The mock registry implements the same surface the routes use (score / model_info
/ backbone / wt_proxy / ad), so endpoint wiring is exercised without artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import create_app
from src.api.schemas import (
    ADMETResult,
    BatchPredictRequest,
    PredictRequest,
    PredictResponse,
)
from src.features.covalent import detect_warheads
from src.scoring.ranking import build_warnings

GEFITINIB = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
OSIMERTINIB = "C=CC(=O)Nc1cc2c(Nc3cccc(NC(=O)/C=C/CN(C)C)c3)ncnc2cc1OC"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
INVALID = "not_a_smiles$$"


# ── Mock registry ───────────────────────────────────────────────────────────────


class _MockRegistry:
    """Artifact-free stand-in. Validity via RDKit; covalent/warnings via the real
    pure-function modules; activity numbers are canned."""

    backbone = object()
    wt_proxy = object()
    ad = object()
    n_train = 1347

    def score(self, smiles: str) -> dict:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return {
                "smiles": smiles,
                "canonical_smiles": None,
                "valid": False,
                "pic50_mutant": None,
                "pic50_wt": None,
                "selectivity_proxy": None,
                "covalent": False,
                "warheads": [],
                "admet": None,
                "applicability_domain": None,
                "docking_selectivity_available": False,
                "warnings": ["Invalid SMILES: RDKit could not parse the input."],
            }
        canonical = Chem.MolToSmiles(mol)
        warheads = detect_warheads(canonical)
        is_cov = bool(warheads)
        domain = "in_domain"
        warnings = build_warnings(domain, is_cov, warheads, None)
        return {
            "smiles": smiles,
            "canonical_smiles": canonical,
            "valid": True,
            "pic50_mutant": 8.1,
            "pic50_wt": 7.4,
            "selectivity_proxy": 0.7,
            "covalent": is_cov,
            "warheads": warheads,
            "admet": {
                "status": "pass",
                "qed": 0.62,
                "sa_score": 2.4,
                "lipinski_pass": True,
                "veber_pass": True,
                "pains_alerts": [],
                "brenk_alerts": [],
                "flag_reasons": [],
            },
            "applicability_domain": {
                "domain": domain,
                "max_tanimoto": 0.71,
                "confidence_factor": 1.0,
            },
            "docking_selectivity_available": False,
            "warnings": warnings,
        }

    def model_info(self) -> dict:
        return {
            "service": "egfr-l858r-fast-screen",
            "version": "1.0.0",
            "models": {
                "backbone_activity": {"algorithm": "random_forest"},
                "wt_proxy": {"algorithm": "xgboost"},
            },
            "score_definitions": {"pic50_mutant": "...", "selectivity_proxy": "..."},
            "exploratory_caveats": ["All activity numbers are EXPLORATORY."],
            "docking_selectivity": (
                "Docking-based selectivity is NOT computed at request "
                "time. Use the offline Vina pipeline."
            ),
        }


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(registry=_MockRegistry()))


# ── TestSchemas ─────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSchemas:

    def test_predict_request_valid(self):
        assert PredictRequest(smiles=GEFITINIB).smiles == GEFITINIB

    def test_predict_request_empty_rejected(self):
        with pytest.raises(ValidationError):
            PredictRequest(smiles="")

    def test_predict_request_too_long_rejected(self):
        with pytest.raises(ValidationError):
            PredictRequest(smiles="C" * 1001)

    def test_batch_request_valid(self):
        assert len(BatchPredictRequest(smiles=[GEFITINIB, ASPIRIN]).smiles) == 2

    def test_batch_request_empty_rejected(self):
        with pytest.raises(ValidationError):
            BatchPredictRequest(smiles=[])

    def test_batch_request_over_limit_rejected(self):
        with pytest.raises(ValidationError):
            BatchPredictRequest(smiles=["C"] * 513)

    def test_admet_defaults(self):
        a = ADMETResult(status="pass")
        assert a.pains_alerts == [] and a.brenk_alerts == [] and a.qed is None

    def test_predict_response_round_trip(self):
        r = PredictResponse(smiles="C", valid=True, covalent=False)
        assert r.docking_selectivity_available is False
        assert r.warnings == []


# ── TestEndpointsMock ───────────────────────────────────────────────────────────


@pytest.mark.unit
class TestEndpointsMock:

    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert all(body["models_loaded"].values())

    def test_predict_valid(self, client):
        r = client.post("/predict", json={"smiles": GEFITINIB})
        assert r.status_code == 200
        b = r.json()
        assert b["valid"] is True
        assert b["pic50_mutant"] == 8.1
        assert b["pic50_wt"] == 7.4
        assert b["selectivity_proxy"] == 0.7
        assert b["docking_selectivity_available"] is False
        assert b["canonical_smiles"]

    def test_predict_covalent_flagged(self, client):
        r = client.post("/predict", json={"smiles": OSIMERTINIB})
        assert r.status_code == 200
        b = r.json()
        assert b["covalent"] is True
        assert b["warheads"]
        assert any("COVALENT" in w for w in b["warnings"])

    def test_predict_noncovalent_clean(self, client):
        r = client.post("/predict", json={"smiles": GEFITINIB})
        b = r.json()
        assert b["covalent"] is False
        assert b["warnings"] == []

    def test_predict_invalid_smiles_422(self, client):
        r = client.post("/predict", json={"smiles": INVALID})
        assert r.status_code == 422
        assert r.json()["error"] == "invalid_smiles"

    def test_predict_whitespace_smiles_422(self, client):
        r = client.post("/predict", json={"smiles": "   "})
        assert r.status_code == 422

    def test_predict_missing_field_422(self, client):
        r = client.post("/predict", json={})
        assert r.status_code == 422
        assert r.json()["error"] == "validation_error"

    def test_batch_predict_mixed(self, client):
        r = client.post(
            "/batch_predict", json={"smiles": [GEFITINIB, INVALID, ASPIRIN]}
        )
        assert r.status_code == 200
        b = r.json()
        assert b["n"] == 3
        assert b["n_valid"] == 2
        assert b["n_invalid"] == 1
        # order preserved: middle one invalid
        assert b["results"][1]["valid"] is False
        assert b["results"][0]["valid"] is True

    def test_batch_predict_empty_rejected(self, client):
        r = client.post("/batch_predict", json={"smiles": []})
        assert r.status_code == 422

    def test_batch_predict_over_limit_rejected(self, client):
        r = client.post("/batch_predict", json={"smiles": ["C"] * 513})
        assert r.status_code == 422

    def test_model_info(self, client):
        r = client.get("/model-info")
        assert r.status_code == 200
        b = r.json()
        assert "docking" in b["docking_selectivity"].lower()
        assert "not computed at request time" in b["docking_selectivity"].lower()
        assert len(b["exploratory_caveats"]) >= 1
        assert "pic50_mutant" in b["score_definitions"]

    def test_model_info_lists_algorithms(self, client):
        b = client.get("/model-info").json()
        assert b["models"]["backbone_activity"]["algorithm"] == "random_forest"
        assert b["models"]["wt_proxy"]["algorithm"] == "xgboost"


# ── TestStructuredErrors ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestStructuredErrors:

    def test_invalid_smiles_has_error_and_detail(self, client):
        b = client.post("/predict", json={"smiles": INVALID}).json()
        assert set(["error", "detail"]).issubset(b.keys())

    def test_validation_error_shape(self, client):
        b = client.post("/predict", json={"smiles": 123}).json()
        assert b["error"] == "validation_error"
        assert "errors" in b


# ── TestModelsUnavailable ───────────────────────────────────────────────────────


@pytest.mark.unit
class TestModelsUnavailable:

    def test_predict_503_when_no_registry(self):
        app = create_app(registry=_MockRegistry())
        app.state.registry = None  # simulate failed startup
        c = TestClient(app)
        r = c.post("/predict", json={"smiles": GEFITINIB})
        assert r.status_code == 503
        assert r.json()["error"] == "models_unavailable"

    def test_health_degraded_when_no_registry(self):
        app = create_app(registry=_MockRegistry())
        app.state.registry = None
        c = TestClient(app)
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "degraded"


# ── TestEndpointsReal (integration — loads real artifacts) ──────────────────────


@pytest.mark.integration
class TestEndpointsReal:

    @pytest.fixture(scope="class")
    def real_client(self):
        from src.api.services import ModelRegistry

        general = PROJECT_ROOT / "models" / "qsar" / "general" / "best_model.pkl"
        if not general.exists():
            pytest.skip("Real model artifacts not present; run training first.")
        return TestClient(create_app(registry=ModelRegistry.load()))

    def test_real_health(self, real_client):
        assert real_client.get("/health").json()["status"] == "ok"

    def test_real_predict_egfr_inhibitor(self, real_client):
        b = real_client.post("/predict", json={"smiles": GEFITINIB}).json()
        assert b["valid"] is True
        assert b["pic50_mutant"] is not None
        assert b["pic50_wt"] is not None
        assert b["selectivity_proxy"] is not None
        assert b["applicability_domain"]["domain"] in (
            "in_domain",
            "borderline",
            "out_of_domain",
        )

    def test_real_model_info_algorithms(self, real_client):
        b = real_client.get("/model-info").json()
        assert b["models"]["backbone_activity"]["algorithm"] == "random_forest"
        assert b["models"]["backbone_activity"]["n_features"] == 2059

    def test_real_invalid_smiles_422(self, real_client):
        assert real_client.post("/predict", json={"smiles": INVALID}).status_code == 422
