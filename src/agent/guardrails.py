"""Scientific warning injection and forbidden-claim sanitizer for the Discovery Copilot."""

from __future__ import annotations

import re
from typing import Any

# ── Forbidden experimental claim patterns ──────────────────────────────────────

# Each tuple: (compiled regex for the claim, human-readable label)
_CLAIM_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bis\s+active\b", re.IGNORECASE), "is active"),
    (re.compile(r"\bis\s+selective\b", re.IGNORECASE), "is selective"),
    (re.compile(r"\bdrug\s+candidate\b", re.IGNORECASE), "drug candidate"),
    (re.compile(r"\bvalidated\b", re.IGNORECASE), "validated"),
    (re.compile(r"\bproven\b", re.IGNORECASE), "proven"),
    (re.compile(r"\bconfirmed\b", re.IGNORECASE), "confirmed"),
]

# Words whose presence within 30 characters before a claim negates it
_NEGATION_RE = re.compile(
    r"\b(not|no|never|un\w*|isn'?t|aren'?t|wasn'?t|cannot|can'?t|fails?\s+to)\b",
    re.IGNORECASE,
)

_SELECTIVITY_CAVEAT = (
    "selectivity_proxy is an ML-derived difference (pic50_mutant - pic50_wt), "
    "NOT statistically validated (Spearman r=0.433, p=0.244 at n=9). Treat as "
    "an exploratory rank signal only."
)
_DOCKING_CAVEAT = (
    "Docking scores are from rigid-receptor AutoDock Vina (2ITZ/2ITY); "
    "delta magnitudes underestimate the true affinity difference. "
    "Direction is more reliable than magnitude."
)
_COVALENT_DOCKING_CAVEAT = (
    "Covalent warhead detected: rigid non-covalent docking cannot model the "
    "covalent bond; docking scores for this compound are lower-confidence."
)
_OOD_CAVEAT = (
    "Out-of-domain (max Tanimoto < 0.30 to training set): activity predictions "
    "carry confidence_factor=0.50; treat with extra caution."
)
_BORDERLINE_CAVEAT = (
    "Borderline applicability domain (Tanimoto 0.30-0.50): predictions carry "
    "confidence_factor=0.75."
)
_GENERATED_CAVEAT = (
    "Generated candidate: activity is scored by the in-sample backbone (doubly "
    "exploratory). No experimental data exists for this molecule."
)


def add_scientific_warnings(result: Any) -> list[str]:
    """
    Return the result's existing warnings extended with standard scientific caveats.

    Operates on any object with optional attributes: selectivity_proxy, covalent,
    domain, source, warheads, l858r_score (presence of docking data).
    Never mutates the result in place; returns a new list.
    """
    warnings: list[str] = list(getattr(result, "warnings", []) or [])

    # selectivity proxy
    if getattr(result, "selectivity_proxy", None) is not None:
        if _SELECTIVITY_CAVEAT not in warnings:
            warnings.append(_SELECTIVITY_CAVEAT)

    # docking data present
    if getattr(result, "l858r_score", None) is not None:
        if _DOCKING_CAVEAT not in warnings:
            warnings.append(_DOCKING_CAVEAT)

    # covalent + docking combo
    covalent = getattr(result, "covalent", False)
    warheads = getattr(result, "warheads", []) or []
    if covalent or warheads:
        if _COVALENT_DOCKING_CAVEAT not in warnings:
            warnings.append(_COVALENT_DOCKING_CAVEAT)

    # applicability domain
    domain = getattr(result, "domain", None)
    if domain == "out_of_domain":
        if _OOD_CAVEAT not in warnings:
            warnings.append(_OOD_CAVEAT)
    elif domain == "borderline":
        if _BORDERLINE_CAVEAT not in warnings:
            warnings.append(_BORDERLINE_CAVEAT)

    # generated molecules
    source = getattr(result, "source", None)
    if source == "generated":
        if _GENERATED_CAVEAT not in warnings:
            warnings.append(_GENERATED_CAVEAT)

    return warnings


def find_forbidden_claims(text: str) -> list[str]:
    """
    Return a list of forbidden experimental claim labels found in text.

    Negated forms are ignored: "not validated", "no confirmed activity", etc.
    An empty list means the text is clean.
    """
    found: list[str] = []
    for pattern, label in _CLAIM_PATTERNS:
        for match in pattern.finditer(text):
            # Look back up to 30 chars for a negation word
            start = match.start()
            prefix = text[max(0, start - 30) : start]
            if not _NEGATION_RE.search(prefix):
                found.append(label)
                break  # one flag per pattern; move on
    return found


def sanitize_text(text: str) -> tuple[str, list[str]]:
    """
    Return (text, found_claims).

    text is returned unchanged; found_claims lists any forbidden phrases so the
    caller can decide whether to warn, redact, or raise.
    """
    claims = find_forbidden_claims(text)
    return text, claims
