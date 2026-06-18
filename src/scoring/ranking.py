"""
Phase 23 — Final integrated candidate ranking (v2 composite score).

This is the capstone scoring module that fuses the four orthogonal evidence
streams produced by the earlier phases into a single ranked table:

  bioactivity_score = 0.30 * activity          (backbone pred_pIC50)
                    + 0.30 * docking_selectivity (-(L858R - WT) Vina delta)
                    + 0.20 * docking_affinity   (-L858R Vina score)
                    + 0.20 * ADMET              (QED)

  final_score = bioactivity_score * confidence_factor   (AD module: 1.0 / 0.75 / 0.50)

Each of the four components is min-max normalised to [0, 1] ACROSS the candidate
set before weighting, so the weights act on comparable scales. Higher is better
for every normalised component (signs are flipped where the raw quantity is
"better when smaller", e.g. Vina kcal/mol).

Design rule (explicit): the ONLY thing that scales the score is the applicability-
domain confidence_factor. Two other low-confidence signals —

  * covalent warhead (rigid docking cannot model the covalent bond)
  * within-noise selectivity (|delta| <= 1.5 * std_delta from the seed-noise study)

— are surfaced as human-readable WARNINGS, never as silent score penalties. This
keeps the score interpretable: a covalent compound and a non-covalent compound
with identical evidence get identical scores, and the covalent liability is shown
next to the score for a human to weigh.

Everything here is EXPLORATORY: backbone pIC50 is in-sample, docking is rigid-
receptor Vina across two different crystal structures, ADMET filters are
approximate. The ranking orders candidates by aggregated evidence; it is not a
calibrated probability of success.

Public API
----------
RankingWeights                       weight container (.from_config(), .normalized())
minmax_normalize(values)             component normaliser, robust to constants/None
build_warnings(domain, is_covalent, warheads, selectivity_within_noise)
rank_candidates(records, weights)    -> pandas.DataFrame, sorted by final_score desc
RANKED_COLUMNS                       canonical output column order
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ── Weights ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RankingWeights:
    """v2 composite weights. Components are weighted AFTER min-max normalisation."""

    activity: float = 0.30
    docking_selectivity: float = 0.30
    docking_affinity: float = 0.20
    admet: float = 0.20

    @classmethod
    def from_config(cls) -> "RankingWeights":
        """
        Read weights from model_config.yaml > ranking.weights_v2 if present,
        otherwise fall back to the v2 defaults above. The legacy ranking.weights
        (mutant_activity/selectivity/docking/admet) is a different, four-axis
        scheme and is intentionally NOT read here.
        """
        try:
            from src.utils.config import load_model_config

            cfg = load_model_config()
            w = (cfg.get("ranking", {}) or {}).get("weights_v2", {}) or {}
        except Exception:
            w = {}
        return cls(
            activity=float(w.get("activity", cls.activity)),
            docking_selectivity=float(
                w.get("docking_selectivity", cls.docking_selectivity)
            ),
            docking_affinity=float(w.get("docking_affinity", cls.docking_affinity)),
            admet=float(w.get("admet", cls.admet)),
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (
            self.activity,
            self.docking_selectivity,
            self.docking_affinity,
            self.admet,
        )

    def sum(self) -> float:
        return float(sum(self.as_tuple()))

    def normalized(self) -> "RankingWeights":
        """Return weights rescaled to sum to 1.0 (no-op if they already do)."""
        s = self.sum()
        if s <= 0:
            raise ValueError("Ranking weights must sum to a positive value.")
        return RankingWeights(
            activity=self.activity / s,
            docking_selectivity=self.docking_selectivity / s,
            docking_affinity=self.docking_affinity / s,
            admet=self.admet / s,
        )


# ── Normalisation ──────────────────────────────────────────────────────────────


def minmax_normalize(values: Sequence[float | None]) -> np.ndarray:
    """
    Min-max normalise a sequence to [0, 1], higher = better.

    Robustness:
      * None / NaN entries map to 0.0 (treated as the worst value) so a candidate
        missing a component is penalised on that axis but not dropped.
      * If every present value is equal (max == min), all present entries map to
        0.5 (neutral) — there is no information to rank on that axis.
      * An all-missing input returns all zeros.
    """
    arr = np.array(
        [np.nan if v is None else float(v) for v in values],
        dtype=np.float64,
    )
    present = ~np.isnan(arr)
    out = np.zeros_like(arr)
    if not present.any():
        return out

    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if vmax == vmin:
        out[present] = 0.5
        return out

    out[present] = (arr[present] - vmin) / (vmax - vmin)
    return out


# ── Warnings ───────────────────────────────────────────────────────────────────


def build_warnings(
    domain: str | None,
    is_covalent: bool,
    warheads: Sequence[str] | None,
    selectivity_within_noise: bool | None,
) -> list[str]:
    """
    Build the per-molecule warning list. These are advisory only — they never
    alter the numeric score (the AD confidence_factor already does that).

    Order: domain reliability, covalent liability, selectivity confidence.
    """
    warnings: list[str] = []

    if domain == "out_of_domain":
        warnings.append(
            "OUT_OF_DOMAIN: backbone activity prediction is unreliable here "
            "(AD confidence_factor 0.50 already applied to final_score)"
        )
    elif domain == "borderline":
        warnings.append(
            "BORDERLINE_DOMAIN: backbone prediction is borderline-reliable "
            "(AD confidence_factor 0.75 already applied to final_score)"
        )

    if is_covalent:
        wh = ",".join(warheads) if warheads else "warhead"
        warnings.append(
            f"COVALENT[{wh}]: rigid non-covalent docking underestimates binding; "
            "docking_affinity/selectivity for this compound are lower-confidence"
        )

    if selectivity_within_noise:
        warnings.append(
            "SELECTIVITY_WITHIN_NOISE: |L858R-WT delta| <= 1.5x Vina seed-noise std; "
            "the selectivity direction is not statistically distinguishable from noise"
        )

    return warnings


# ── Ranking ────────────────────────────────────────────────────────────────────

RANKED_COLUMNS: list[str] = [
    "rank",
    "cid",
    "source",
    "smiles",
    "activity",
    "l858r_score",
    "wt_score",
    "selectivity_delta",
    "admet_qed",
    "domain",
    "max_tanimoto",
    "confidence_factor",
    "activity_norm",
    "selectivity_norm",
    "affinity_norm",
    "admet_norm",
    "bioactivity_score",
    "final_score",
    "is_covalent",
    "warheads",
    "selectivity_within_noise",
    "warnings",
]


def _selectivity_raw(record: dict) -> float | None:
    """Higher = more L858R-selective. selectivity_delta = L858R - WT (neg = good)."""
    d = record.get("selectivity_delta")
    return None if d is None else -float(d)


def _affinity_raw(record: dict) -> float | None:
    """Higher = stronger L858R binding. Vina score is negative kcal/mol."""
    s = record.get("l858r_score")
    return None if s is None else -float(s)


def rank_candidates(
    records: list[dict[str, Any]],
    weights: RankingWeights | None = None,
) -> pd.DataFrame:
    """
    Compute the v2 composite ranking over a list of candidate records.

    Each record should provide:
      cid, smiles, source                      (identity)
      activity            (backbone pred_pIC50, higher better)
      l858r_score, wt_score                    (Vina kcal/mol; may be None)
      selectivity_delta   (= l858r_score - wt_score; may be None)
      admet_qed           (QED in [0,1])
      domain, confidence_factor, max_tanimoto  (from AD module)
      is_covalent, warheads                    (from covalent detector)
      selectivity_within_noise (bool | None)   (from docking-noise study; None = unknown)

    Components are min-max normalised across the supplied set, weighted, and
    multiplied by the AD confidence_factor. Returns a DataFrame sorted by
    final_score descending with a 1-indexed `rank` column and RANKED_COLUMNS order.

    An empty input returns an empty DataFrame with the canonical columns.
    """
    if weights is None:
        weights = RankingWeights.from_config()
    weights = weights.normalized()

    if not records:
        return pd.DataFrame(columns=RANKED_COLUMNS)

    activity_norm = minmax_normalize([r.get("activity") for r in records])
    selectivity_norm = minmax_normalize([_selectivity_raw(r) for r in records])
    affinity_norm = minmax_normalize([_affinity_raw(r) for r in records])
    admet_norm = minmax_normalize([r.get("admet_qed") for r in records])

    rows: list[dict[str, Any]] = []
    for i, r in enumerate(records):
        bioactivity = float(
            weights.activity * activity_norm[i]
            + weights.docking_selectivity * selectivity_norm[i]
            + weights.docking_affinity * affinity_norm[i]
            + weights.admet * admet_norm[i]
        )
        cf = float(r.get("confidence_factor", 1.0))
        final = bioactivity * cf

        is_cov = bool(r.get("is_covalent", False))
        warheads = list(r.get("warheads") or [])
        within_noise = r.get("selectivity_within_noise")
        warnings = build_warnings(r.get("domain"), is_cov, warheads, within_noise)

        rows.append(
            {
                "cid": r.get("cid"),
                "source": r.get("source"),
                "smiles": r.get("smiles"),
                "activity": _round(r.get("activity")),
                "l858r_score": _round(r.get("l858r_score")),
                "wt_score": _round(r.get("wt_score")),
                "selectivity_delta": _round(r.get("selectivity_delta")),
                "admet_qed": _round(r.get("admet_qed")),
                "domain": r.get("domain"),
                "max_tanimoto": _round(r.get("max_tanimoto")),
                "confidence_factor": cf,
                "activity_norm": round(float(activity_norm[i]), 4),
                "selectivity_norm": round(float(selectivity_norm[i]), 4),
                "affinity_norm": round(float(affinity_norm[i]), 4),
                "admet_norm": round(float(admet_norm[i]), 4),
                "bioactivity_score": round(bioactivity, 4),
                "final_score": round(final, 4),
                "is_covalent": is_cov,
                "warheads": ",".join(warheads) if warheads else "",
                "selectivity_within_noise": within_noise,
                "warnings": " | ".join(warnings),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("final_score", ascending=False, kind="mergesort").reset_index(
        drop=True
    )
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df[RANKED_COLUMNS]


def _round(v: Any, ndigits: int = 4) -> Any:
    """Round floats for display; pass through None and non-numerics."""
    if v is None:
        return None
    try:
        return round(float(v), ndigits)
    except (TypeError, ValueError):
        return v
