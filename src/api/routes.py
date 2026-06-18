"""
API routes for the EGFR L858R fast-screen service (Phase 24).

Endpoints:
  GET  /health         — liveness + which artifacts are loaded
  POST /predict        — single SMILES fast screen (422 on invalid SMILES)
  POST /batch_predict  — many SMILES; invalid ones become valid=false rows
  GET  /model-info     — model versions, score meanings, exploratory caveats,
                         and the explicit "docking not computed at request time" note

The registry of loaded models is injected via the `get_registry` dependency,
which reads from app.state. Tests override this dependency with a mock so no
real artifacts are required.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from src.api.services import ModelRegistry

router = APIRouter()


def get_registry(request: Request) -> ModelRegistry:
    """Dependency: return the ModelRegistry loaded at startup (app.state)."""
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "models_unavailable",
                "detail": "Model artifacts are not loaded. The service is not ready.",
            },
        )
    return registry


@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health(request: Request) -> HealthResponse:
    """Liveness check. status='ok' only when every artifact is loaded."""
    registry = getattr(request.app.state, "registry", None)
    loaded = {
        "backbone_activity": registry is not None and registry.backbone is not None,
        "wt_proxy": registry is not None and registry.wt_proxy is not None,
        "applicability_domain": registry is not None and registry.ad is not None,
    }
    status = "ok" if registry is not None and all(loaded.values()) else "degraded"
    return HealthResponse(status=status, models_loaded=loaded)


@router.post(
    "/predict",
    response_model=PredictResponse,
    tags=["predict"],
    responses={422: {"description": "Invalid SMILES"}},
)
def predict(
    body: PredictRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> PredictResponse:
    """Fast screen for one molecule. Returns 422 with a structured error if the
    SMILES cannot be parsed by RDKit."""
    result = registry.score(body.smiles.strip())
    if not result["valid"]:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_smiles",
                "detail": f"Could not parse SMILES: {body.smiles!r}",
            },
        )
    return PredictResponse(**result)


@router.post("/batch_predict", response_model=BatchPredictResponse, tags=["predict"])
def batch_predict(
    body: BatchPredictRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> BatchPredictResponse:
    """Fast screen for many molecules. Invalid SMILES do not fail the batch —
    they come back as valid=false rows. Order matches the input."""
    results = [registry.score(s.strip()) for s in body.smiles]
    responses = [PredictResponse(**r) for r in results]
    n_valid = sum(1 for r in responses if r.valid)
    return BatchPredictResponse(
        n=len(responses),
        n_valid=n_valid,
        n_invalid=len(responses) - n_valid,
        results=responses,
    )


@router.get("/model-info", response_model=ModelInfoResponse, tags=["meta"])
def model_info(registry: ModelRegistry = Depends(get_registry)) -> ModelInfoResponse:
    """Model versions, the meaning of each score, exploratory caveats, and the
    explicit statement that docking selectivity is not computed at request time."""
    return ModelInfoResponse(**registry.model_info())
