"""Typed request/response shapes for each Discovery Copilot tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── Tool results ───────────────────────────────────────────────────────────────


@dataclass
class PredictToolResult:
    """Output of predict_smiles / batch_predict row."""

    valid: bool
    smiles: str
    canonical_smiles: str | None = None
    error: str | None = None
    pic50_mutant: float | None = None
    pic50_wt: float | None = None
    selectivity_proxy: float | None = None
    covalent: bool = False
    warheads: list[str] = field(default_factory=list)
    admet_status: str | None = None
    qed: float | None = None
    admet_alerts: list[str] = field(default_factory=list)
    domain: str | None = None
    confidence_factor: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class BatchPredictToolResult:
    """Output of batch_predict."""

    n: int
    n_valid: int
    n_invalid: int
    results: list[PredictToolResult] = field(default_factory=list)


@dataclass
class RankingLookupResult:
    """Output of lookup_final_ranking."""

    found: bool
    candidate_id: str | None = None
    rank: int | None = None
    source: str | None = None
    smiles: str | None = None
    final_score: float | None = None
    activity_norm: float | None = None
    selectivity_norm: float | None = None
    affinity_norm: float | None = None
    admet_norm: float | None = None
    confidence_factor: float | None = None
    is_covalent: bool = False
    warnings: str | None = None


@dataclass
class DockingLookupResult:
    """Output of lookup_docking_results."""

    found: bool
    candidate_id: str | None = None
    message: str | None = None
    # First-pass Vina scores (initial_delta from library or seed-42 score)
    l858r_score: float | None = None
    wt_score: float | None = None
    selectivity_delta: float | None = None
    direction: str | None = None
    # Noise-study averages (top-15 only)
    mean_delta: float | None = None
    std_delta: float | None = None
    noise_call: str | None = None
    docking_confidence: str | None = None
    warheads: list[str] = field(default_factory=list)
    data_source: str = "unknown"  # "library", "generated", "sanity"


@dataclass
class ComparisonResult:
    """Output of compare_candidates."""

    candidate_ids: list[str]
    recommendation: str
    reason: str
    scores: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class CandidateReport:
    """Output of generate_candidate_report."""

    candidate_id: str
    markdown: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class RetrievalSection:
    """One retrieved documentation section."""

    source: str
    header: str
    content: str
    score: float


# ── Controller I/O ─────────────────────────────────────────────────────────────


@dataclass
class AgentRequest:
    """Incoming query to the controller."""

    query: str
    smiles: str | None = None
    candidate_ids: list[str] = field(default_factory=list)


@dataclass
class AgentResponse:
    """Unified controller output."""

    intent: str
    answer: str
    tool_results: list[Any] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
