"""
Deterministic tool functions for the Discovery Copilot.

Every function reuses existing precomputed artifacts; none reimplement science.
The registry parameter accepts a ModelRegistry instance (or mock) for testing;
when None, a module-level singleton is lazy-loaded from disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agent.guardrails import add_scientific_warnings
from src.agent.report import generate_report
from src.agent.schemas import (
    BatchPredictToolResult,
    CandidateReport,
    ComparisonResult,
    DockingLookupResult,
    PredictToolResult,
    RankingLookupResult,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_RANKING_CSV = PROJECT_ROOT / "data" / "generated" / "final_ranked_candidates.csv"
_LIBRARY_DOCKING_JSON = (
    PROJECT_ROOT / "models" / "qsar" / "library_docking_results.json"
)
_NOISE_DOCKING_JSON = PROJECT_ROOT / "models" / "qsar" / "docking_noise_results.json"
_GEN_DOCKING_JSON = (
    PROJECT_ROOT / "models" / "generator" / "generated_docking_results.json"
)

MAX_BATCH = 512

# ── Registry singleton ─────────────────────────────────────────────────────────

_registry_cache: Any = None  # ModelRegistry | None


def _get_registry(registry: Any = None) -> Any:
    global _registry_cache
    if registry is not None:
        return registry
    if _registry_cache is None:
        from src.api.services import ModelRegistry

        _registry_cache = ModelRegistry.load()
    return _registry_cache


# ── Docking index (lazy) ───────────────────────────────────────────────────────

_docking_index: dict[str, DockingLookupResult] | None = None


def _build_docking_index() -> dict[str, DockingLookupResult]:
    """
    Merge all three docking result files into a single cid -> DockingLookupResult map.

    Priority: noise study data (5-seed mean) overlays library first-pass scores.
    Generated candidates are added separately.
    """
    index: dict[str, DockingLookupResult] = {}

    # 1. Library first-pass (initial single-seed scores for all 50 candidates)
    if _LIBRARY_DOCKING_JSON.exists():
        data = json.loads(_LIBRARY_DOCKING_JSON.read_text())
        for c in data.get("compounds", []):
            cid = c.get("cid")
            if not cid:
                continue
            # Infer direction from delta sign (negative = L858R-selective)
            delta = c.get("selectivity_delta")
            direction: str | None = None
            if delta is not None:
                direction = "L858R_favoured" if delta < 0 else "WT_favoured"
            index[cid] = DockingLookupResult(
                found=True,
                candidate_id=cid,
                l858r_score=c.get("l858r_score"),
                wt_score=c.get("wt_score"),
                selectivity_delta=delta,
                direction=direction,
                docking_confidence=c.get("docking_confidence"),
                warheads=c.get("warheads") or [],
                data_source="library",
            )

    # 2. Noise study (5-seed mean/std for top-15; overwrites first-pass where present)
    if _NOISE_DOCKING_JSON.exists():
        data = json.loads(_NOISE_DOCKING_JSON.read_text())
        for c in data.get("compounds", []):
            cid = c.get("cid")
            if not cid:
                continue
            ns = c.get("noise_stats", {})
            existing = index.get(cid)
            if existing:
                # Overlay noise-study mean values and call
                existing.mean_delta = ns.get("delta")
                existing.std_delta = ns.get("std_delta")
                existing.noise_call = c.get("call")
                # Use mean scores as the canonical values when available
                if ns.get("mean_l858r") is not None:
                    existing.l858r_score = ns["mean_l858r"]
                if ns.get("mean_wt") is not None:
                    existing.wt_score = ns["mean_wt"]
                if ns.get("delta") is not None:
                    existing.selectivity_delta = ns["delta"]
                    existing.direction = (
                        "L858R_favoured" if ns["delta"] < 0 else "WT_favoured"
                    )
            else:
                delta = ns.get("delta")
                index[cid] = DockingLookupResult(
                    found=True,
                    candidate_id=cid,
                    l858r_score=ns.get("mean_l858r"),
                    wt_score=ns.get("mean_wt"),
                    selectivity_delta=delta,
                    direction="L858R_favoured" if (delta or 0) < 0 else "WT_favoured",
                    mean_delta=delta,
                    std_delta=ns.get("std_delta"),
                    noise_call=c.get("call"),
                    docking_confidence=c.get("docking_confidence"),
                    warheads=c.get("warheads") or [],
                    data_source="library",
                )

    # 3. Generated candidates
    if _GEN_DOCKING_JSON.exists():
        data = json.loads(_GEN_DOCKING_JSON.read_text())
        for c in data.get("compounds", []):
            cid = c.get("cid")
            if not cid:
                continue
            delta = c.get("selectivity_delta")
            direction = None
            if delta is not None:
                direction = "L858R_favoured" if delta < 0 else "WT_favoured"
            index[cid] = DockingLookupResult(
                found=True,
                candidate_id=cid,
                l858r_score=c.get("l858r_score"),
                wt_score=c.get("wt_score"),
                selectivity_delta=delta,
                direction=direction,
                docking_confidence=c.get("docking_confidence"),
                warheads=c.get("warheads") or [],
                data_source="generated",
            )

    return index


def _get_docking_index() -> dict[str, DockingLookupResult]:
    global _docking_index
    if _docking_index is None:
        _docking_index = _build_docking_index()
    return _docking_index


def _clear_docking_index() -> None:
    """Force a rebuild of the docking index (useful in tests)."""
    global _docking_index
    _docking_index = None


# ── Ranking index (lazy) ───────────────────────────────────────────────────────

_ranking_cache: dict[str, dict[str, Any]] | None = None


def _get_ranking_index() -> dict[str, dict[str, Any]]:
    global _ranking_cache
    if _ranking_cache is None:
        if not _RANKING_CSV.exists():
            _ranking_cache = {}
            return _ranking_cache
        import pandas as pd

        df = pd.read_csv(_RANKING_CSV)
        # Normalise: NaN -> None for downstream code
        _ranking_cache = {
            row["cid"]: row.where(row.notna(), other=None).to_dict()
            for _, row in df.iterrows()
        }
    return _ranking_cache


# ── Public tool functions ──────────────────────────────────────────────────────


def predict_smiles(smiles: str, registry: Any = None) -> PredictToolResult:
    """
    Run the fast ML screen for one SMILES via ModelRegistry.score().

    Returns PredictToolResult with valid=False and an error field for unparseable input.
    """
    reg = _get_registry(registry)
    raw = reg.score(smiles)

    if not raw.get("valid", True):
        return PredictToolResult(
            valid=False,
            smiles=smiles,
            error=(raw.get("warnings") or ["Invalid SMILES"])[0],
            warnings=list(raw.get("warnings") or []),
        )

    admet = raw.get("admet") or {}
    ad = raw.get("applicability_domain") or {}
    alerts: list[str] = list(admet.get("pains_alerts") or []) + list(
        admet.get("brenk_alerts") or []
    )

    result = PredictToolResult(
        valid=True,
        smiles=smiles,
        canonical_smiles=raw.get("canonical_smiles"),
        pic50_mutant=raw.get("pic50_mutant"),
        pic50_wt=raw.get("pic50_wt"),
        selectivity_proxy=raw.get("selectivity_proxy"),
        covalent=raw.get("covalent", False),
        warheads=list(raw.get("warheads") or []),
        admet_status=admet.get("status"),
        qed=admet.get("qed"),
        admet_alerts=alerts,
        domain=ad.get("domain"),
        confidence_factor=ad.get("confidence_factor"),
        warnings=list(raw.get("warnings") or []),
    )
    result.warnings = add_scientific_warnings(result)
    return result


def batch_predict(
    smiles_list: list[str], registry: Any = None
) -> BatchPredictToolResult:
    """Run predict_smiles on up to MAX_BATCH SMILES. Invalid rows are included, not dropped."""
    if len(smiles_list) > MAX_BATCH:
        raise ValueError(f"Batch size {len(smiles_list)} exceeds limit {MAX_BATCH}.")

    reg = _get_registry(registry)
    results: list[PredictToolResult] = [predict_smiles(s, reg) for s in smiles_list]
    n_valid = sum(1 for r in results if r.valid)
    return BatchPredictToolResult(
        n=len(results),
        n_valid=n_valid,
        n_invalid=len(results) - n_valid,
        results=results,
    )


def lookup_final_ranking(candidate_id: str) -> RankingLookupResult:
    """
    Look up a candidate in data/generated/final_ranked_candidates.csv.

    Returns found=False (not raises) if the ID is absent or the file does not exist.
    """
    idx = _get_ranking_index()
    row = idx.get(candidate_id)
    if row is None:
        return RankingLookupResult(found=False, candidate_id=candidate_id)

    return RankingLookupResult(
        found=True,
        candidate_id=candidate_id,
        rank=int(row["rank"]) if row.get("rank") is not None else None,
        source=row.get("source"),
        smiles=row.get("smiles"),
        final_score=row.get("final_score"),
        activity_norm=row.get("activity_norm"),
        selectivity_norm=row.get("selectivity_norm"),
        affinity_norm=row.get("affinity_norm"),
        admet_norm=row.get("admet_norm"),
        confidence_factor=row.get("confidence_factor"),
        is_covalent=bool(row.get("is_covalent") or False),
        warnings=str(row["warnings"]) if row.get("warnings") is not None else None,
    )


def lookup_docking_results(candidate_id: str) -> DockingLookupResult:
    """
    Look up precomputed Vina scores for a candidate.

    Checks library docking, docking noise study, and generated-candidate docking.
    Returns found=False with a message if the candidate has no docking data.
    NEVER fabricates a docking number.
    """
    result = _get_docking_index().get(candidate_id)
    if result is None:
        return DockingLookupResult(
            found=False,
            candidate_id=candidate_id,
            message=(
                f"No docking results found for '{candidate_id}'. "
                "Run scripts/dock_library.py or scripts/dock_generated_candidates.py "
                "to generate docking scores for this compound."
            ),
        )
    return result


def compare_candidates(ids: list[str], registry: Any = None) -> ComparisonResult:
    """
    Compare two or more candidates and recommend the more conservative choice.

    Conservative preference order: non-covalent > in-domain > ADMET pass > final score.
    Raises ValueError if fewer than 2 IDs are provided.
    """
    if len(ids) < 2:
        raise ValueError("compare_candidates requires at least 2 candidate IDs.")

    # Collect evidence for each candidate
    rankings = {cid: lookup_final_ranking(cid) for cid in ids}
    dockings = {cid: lookup_docking_results(cid) for cid in ids}

    def conservative_score(cid: str) -> float:
        r = rankings[cid]
        d = dockings[cid]
        score = 0.0
        # Non-covalent preferred
        if r.found and not r.is_covalent:
            score += 2.0
        elif d.found and not d.warheads:
            score += 2.0
        # In-domain preferred
        if r.found and r.confidence_factor == 1.0:
            score += 2.0
        # ADMET: higher admet_norm is better
        if r.found and r.admet_norm is not None:
            score += r.admet_norm
        # Final score
        if r.found and r.final_score is not None:
            score += r.final_score
        # Confirmed L858R-selective docking
        if d.found and d.noise_call == "L858R_selective":
            score += 0.5
        return score

    scored = sorted(ids, key=conservative_score, reverse=True)
    best = scored[0]
    runner_up = scored[1]

    # Build reason string
    b_rank = rankings[best]
    b_dock = dockings[best]
    reason_parts: list[str] = []

    if b_rank.found:
        reason_parts.append(
            f"rank {b_rank.rank}/68, final_score={b_rank.final_score:.3f}"
        )
    if not (b_rank.found and b_rank.is_covalent):
        reason_parts.append("non-covalent")
    if b_rank.found and b_rank.confidence_factor == 1.0:
        reason_parts.append("in-domain (cf=1.0)")
    if b_rank.found and b_rank.admet_norm is not None and b_rank.admet_norm > 0.5:
        reason_parts.append(f"ADMET norm={b_rank.admet_norm:.2f}")
    if b_dock.found and b_dock.noise_call == "L858R_selective":
        reason_parts.append(
            f"noise-study call: L858R_selective (mean delta={b_dock.mean_delta:.3f})"
        )

    reason = (
        f"Preferring {best} over {runner_up}: "
        + ("; ".join(reason_parts) if reason_parts else "higher conservative score")
        + ". All scores are exploratory."
    )

    scores = {cid: conservative_score(cid) for cid in ids}

    all_warnings: list[str] = []
    for cid in ids:
        r = rankings[cid]
        if r.found and r.warnings:
            all_warnings.append(f"{cid}: {r.warnings}")

    return ComparisonResult(
        candidate_ids=ids,
        recommendation=best,
        reason=reason,
        scores=scores,
        warnings=all_warnings,
    )


def generate_candidate_report(
    candidate_id: str, registry: Any = None
) -> CandidateReport:
    """
    Build a full markdown report for one candidate.

    Internally calls lookup_final_ranking, lookup_docking_results, and (if a
    registry is available) predict_smiles using the candidate's SMILES from the
    ranking CSV. Falls back gracefully when any data source is unavailable.
    """
    ranking = lookup_final_ranking(candidate_id)
    docking = lookup_docking_results(candidate_id)

    predict: PredictToolResult | None = None
    smiles = ranking.smiles if ranking.found else None
    if smiles:
        try:
            predict = predict_smiles(smiles, registry)
        except Exception:
            pass  # report will note ML data unavailable

    # Collect extra warnings from guardrails
    extra: list[str] = []
    if predict:
        extra = add_scientific_warnings(predict)

    return generate_report(
        candidate_id=candidate_id,
        ranking=ranking,
        docking=docking if docking.found else None,
        predict=predict,
        extra_warnings=extra,
    )
