"""
Pydantic request/response schemas for the EGFR L858R screening API (Phase 24).

The API serves PRECOMPUTED models only: backbone activity (Model 1), WT-proxy
(Model 2), the ADMET filter, the covalent-warhead detector, and the
applicability domain. No training, no docking, and no ESM-2 run at request time.

Every activity/selectivity number returned here is EXPLORATORY (in-sample
backbone, derived WT-proxy delta). Docking-based selectivity is NOT computed at
request time — it needs the offline Vina pipeline — and the response says so
explicitly via `docking_selectivity_available`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

MAX_BATCH = 512


# ── Requests ───────────────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    """Single-molecule prediction request."""

    smiles: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="A single SMILES string.",
        examples=["COc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OC"],
    )


class BatchPredictRequest(BaseModel):
    """Batch prediction request (one screen per SMILES, independent)."""

    smiles: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH,
        description=f"List of SMILES strings (1-{MAX_BATCH}).",
        examples=[["COc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OC", "CC(=O)Oc1ccccc1C(=O)O"]],
    )


# ── Response sub-objects ───────────────────────────────────────────────────────


class ADMETResult(BaseModel):
    """Approximate ADMET / drug-likeness summary (QED, alerts, pass/flag)."""

    status: str = Field(..., description='"pass" or "flag" (never hard-dropped).')
    qed: float | None = Field(None, description="QED drug-likeness [0,1].")
    sa_score: float | None = Field(
        None, description="Synthetic accessibility [1,10]; lower = easier."
    )
    lipinski_pass: bool | None = None
    veber_pass: bool | None = None
    pains_alerts: list[str] = Field(default_factory=list)
    brenk_alerts: list[str] = Field(default_factory=list)
    flag_reasons: list[str] = Field(default_factory=list)


class ApplicabilityDomainResult(BaseModel):
    """Where the molecule sits relative to the training chemical space."""

    domain: str = Field(
        ..., description='"in_domain" | "borderline" | "out_of_domain".'
    )
    max_tanimoto: float | None = Field(
        None, description="Max ECFP4 Tanimoto to training set."
    )
    confidence_factor: float = Field(
        ..., description="Score multiplier: 1.0 / 0.75 / 0.50."
    )


class PredictResponse(BaseModel):
    """Full fast-screen result for one molecule."""

    smiles: str = Field(..., description="Input SMILES (echoed).")
    canonical_smiles: str | None = Field(
        None, description="RDKit canonical SMILES, or null if invalid."
    )
    valid: bool = Field(..., description="True if RDKit parsed the molecule.")

    pic50_mutant: float | None = Field(
        None,
        description="Backbone (Model 1) predicted pIC50 on L858R/general EGFR. EXPLORATORY (in-sample).",
    )
    pic50_wt: float | None = Field(
        None, description="WT-proxy (Model 2) predicted pIC50. EXPLORATORY."
    )
    selectivity_proxy: float | None = Field(
        None,
        description="ML selectivity proxy = pic50_mutant - pic50_wt (positive = mutant-selective). "
        "EXPLORATORY; NOT the docking selectivity.",
    )

    covalent: bool = Field(
        False, description="True if an electrophilic warhead was detected."
    )
    warheads: list[str] = Field(
        default_factory=list, description="Matched warhead type names."
    )

    admet: ADMETResult | None = None
    applicability_domain: ApplicabilityDomainResult | None = None

    docking_selectivity_available: bool = Field(
        False,
        description="Always false: structure-based (Vina) selectivity is NOT computed at request "
        "time. Use the offline docking pipeline for that.",
    )

    warnings: list[str] = Field(
        default_factory=list,
        description="Advisory low-confidence flags (out-of-domain, covalent, etc.). "
        "These do not alter the numeric scores.",
    )


class BatchPredictResponse(BaseModel):
    n: int
    n_valid: int
    n_invalid: int
    results: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str = Field(..., description='"ok" when all artifacts are loaded.')
    models_loaded: dict[str, bool]


class ModelInfoResponse(BaseModel):
    service: str
    version: str
    models: dict
    score_definitions: dict[str, str]
    exploratory_caveats: list[str]
    docking_selectivity: str


class ErrorResponse(BaseModel):
    """Structured error body for 4xx responses."""

    error: str = Field(..., description="Machine-readable error code.")
    detail: str = Field(..., description="Human-readable explanation.")
