"""
Discovery Copilot controller: classify intent, call deterministic tools, apply
guardrails, and assemble a grounded answer.

Runs in pure deterministic mode (no LLM required). An optional LLM summarizer
can be plugged in via prompts.llm_summarize; when it returns None the controller
uses the structured tool output directly.
"""

from __future__ import annotations

import re
from typing import Any

from src.agent.guardrails import find_forbidden_claims
from src.agent.prompts import SYSTEM_PROMPT, llm_summarize
from src.agent.retrieval import retrieve
from src.agent.schemas import AgentRequest, AgentResponse
from src.agent.tools import (
    batch_predict,
    compare_candidates,
    generate_candidate_report,
    lookup_docking_results,
    lookup_final_ranking,
    predict_smiles,
)

# ── Intent patterns ────────────────────────────────────────────────────────────

# Each entry: (intent_name, keyword list). First match wins.
_INTENT_RULES: list[tuple[str, list[str]]] = [
    ("report", ["report", "summarize", "summary", "full report", "generate report"]),
    ("comparison", ["compare", " vs ", " versus ", "better than", "prefer", "between"]),
    (
        "docking_query",
        ["dock", "vina", "binding", "pocket", "delta kcal", "kcal/mol", "b2", "b3"],
    ),
    ("batch_predict", ["batch", "multiple smiles", "list of smiles", "screen list"]),
    (
        "candidate_lookup",
        ["cmpd_", "gen_", "rank", "ranking", "shortlist", "final rank"],
    ),
    ("single_predict", ["smiles", "predict", "score", "pic50", "pIC50", "activity of"]),
    (
        "project_qa",
        [
            "what",
            "how",
            "why",
            "explain",
            "limitation",
            "method",
            "pipeline",
            "phase",
            "model",
            "admet",
            "loocv",
            "rl",
            "gnn",
            "scaffold",
            "tanimoto",
        ],
    ),
]

# Regex for candidate IDs embedded in a query
_CID_RE = re.compile(r"\b(cmpd_\d+|gen_\d+)\b", re.IGNORECASE)

# Rough SMILES detection: contains ring notation or chain chars but not spaces
_SMILES_RE = re.compile(r"^[A-Za-z0-9@\[\]()=#\-\+\\/%.]{6,}$")


def classify_intent(query: str) -> str:
    """
    Return the most likely intent label for a natural-language query.

    Uses keyword matching in priority order. Returns "unknown" if nothing matches.
    """
    lower = query.lower()
    for intent, keywords in _INTENT_RULES:
        if any(kw.lower() in lower for kw in keywords):
            return intent
    return "unknown"


def _extract_candidate_ids(query: str) -> list[str]:
    return [m.group(0).lower() for m in _CID_RE.finditer(query)]


def _extract_smiles(query: str) -> list[str]:
    """Pull tokens that look like SMILES strings from the query."""
    found: list[str] = []
    for token in query.split():
        # Strip surrounding punctuation
        token = token.strip(".,;:\"'()[]")
        if _SMILES_RE.match(token) and len(token) >= 8:
            found.append(token)
    return found


def _format_predict_answer(result: Any) -> str:
    if not result.valid:
        return (
            f"**Invalid SMILES**: {result.error or 'RDKit could not parse the input.'}"
        )
    lines = [
        "**Prediction (EXPLORATORY)**",
        (
            f"- Backbone pred pIC50: {result.pic50_mutant:.3f}"
            if result.pic50_mutant
            else "- Backbone pred pIC50: N/A"
        ),
        (
            f"- WT-proxy pred pIC50: {result.pic50_wt:.3f}"
            if result.pic50_wt
            else "- WT-proxy pred pIC50: N/A"
        ),
        (
            f"- Selectivity proxy (ML proxy, exploratory): {result.selectivity_proxy:.3f}"
            if result.selectivity_proxy
            else "- Selectivity proxy (ML proxy, exploratory): N/A"
        ),
        f"- Applicability domain: {result.domain or 'N/A'} (cf={result.confidence_factor})",
        f"- Covalent: {'Yes (' + ', '.join(result.warheads) + ')' if result.covalent else 'No'}",
        (
            f"- ADMET: {result.admet_status or 'N/A'}, QED={result.qed:.3f}"
            if result.qed
            else f"- ADMET: {result.admet_status or 'N/A'}"
        ),
    ]
    if result.warnings:
        lines.append("\n**Warnings**:")
        lines.extend(f"- {w}" for w in result.warnings)
    return "\n".join(lines)


def _format_ranking_answer(r: Any) -> str:
    if not r.found:
        return (
            f"Candidate `{r.candidate_id}` was not found in the final ranked shortlist."
        )
    return (
        f"**{r.candidate_id}** | Rank {r.rank}/68 | Source: {r.source} | "
        f"Final score: {r.final_score:.3f} | "
        f"Activity norm: {r.activity_norm:.3f} | "
        f"Selectivity norm: {r.selectivity_norm:.3f} | "
        f"Affinity norm: {r.affinity_norm:.3f} | "
        f"ADMET norm: {r.admet_norm:.3f} | "
        f"Covalent: {'Yes' if r.is_covalent else 'No'} | "
        f"CF: {r.confidence_factor}\n\n"
        "_All scores are EXPLORATORY (in-sample backbone, rigid-receptor docking)._"
    )


def _format_docking_answer(d: Any) -> str:
    if not d.found:
        return f"{d.message or 'No docking data found for this candidate.'}"
    parts = [
        f"**Docking results for {d.candidate_id}** (EXPLORATORY, rigid Vina)",
        (
            f"- L858R score: {d.l858r_score:.3f} kcal/mol"
            if d.l858r_score
            else "- L858R score: N/A"
        ),
        f"- WT score: {d.wt_score:.3f} kcal/mol" if d.wt_score else "- WT score: N/A",
        (
            f"- Selectivity delta: {d.selectivity_delta:.3f} kcal/mol ({d.direction or 'unknown'})"
            if d.selectivity_delta
            else "- Selectivity delta: N/A"
        ),
    ]
    if d.mean_delta is not None:
        parts += [
            f"- Mean delta (5-seed): {d.mean_delta:.3f} ± {d.std_delta:.3f} kcal/mol",
            f"- Noise-study call: {d.noise_call}",
        ]
    if d.warheads:
        parts.append(f"- Warheads: {', '.join(d.warheads)} (docking confidence: lower)")
    parts.append(
        "\n_Delta direction is reliable; magnitude underestimates the true affinity difference._"
    )
    return "\n".join(parts)


def handle(request: AgentRequest, registry: Any = None) -> AgentResponse:
    """
    Classify the request intent, call the appropriate tool(s), apply guardrails,
    and return a grounded AgentResponse.

    No LLM is required. If llm_summarize returns a non-None string it is used as
    the answer; otherwise the structured deterministic output is used directly.
    """
    query = request.query
    intent = classify_intent(query)

    tool_results: list[Any] = []
    answer_parts: list[str] = []
    all_warnings: list[str] = []
    sources: list[str] = []

    # ── Dispatch ───────────────────────────────────────────────────────────────

    if intent == "single_predict":
        smiles_candidates = (
            request.smiles and [request.smiles] or _extract_smiles(query)
        )
        if not smiles_candidates:
            answer_parts.append(
                "Please provide a SMILES string to predict. "
                "Example: `predict COc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OC`"
            )
        else:
            for smi in smiles_candidates[:1]:  # single mode: first token only
                result = predict_smiles(smi, registry)
                tool_results.append(result)
                answer_parts.append(_format_predict_answer(result))
                all_warnings.extend(result.warnings)

    elif intent == "batch_predict":
        smiles_list = (
            request.smiles.split() if request.smiles else _extract_smiles(query)
        )
        if not smiles_list:
            answer_parts.append("Please provide SMILES strings for batch prediction.")
        else:
            batch = batch_predict(smiles_list[:MAX_BATCH], registry)
            tool_results.append(batch)
            answer_parts.append(
                f"Batch prediction: {batch.n} molecules, "
                f"{batch.n_valid} valid, {batch.n_invalid} invalid (EXPLORATORY)."
            )
            for r in batch.results:
                answer_parts.append(
                    f"\n**{r.smiles[:40]}{'...' if len(r.smiles) > 40 else ''}**"
                )
                answer_parts.append(_format_predict_answer(r))
                all_warnings.extend(r.warnings)

    elif intent == "candidate_lookup":
        cids = request.candidate_ids or _extract_candidate_ids(query)
        if not cids:
            # Try to look up any cmpd-like token in the query
            answer_parts.append(
                "Please specify a candidate ID (e.g. cmpd_015 or gen_005) "
                "to look up in the final ranking."
            )
        else:
            for cid in cids[:5]:
                r = lookup_final_ranking(cid)
                tool_results.append(r)
                answer_parts.append(_format_ranking_answer(r))

    elif intent == "docking_query":
        cids = request.candidate_ids or _extract_candidate_ids(query)
        if not cids:
            answer_parts.append(
                "Please specify a candidate ID (e.g. cmpd_024) "
                "to retrieve its docking results."
            )
        else:
            for cid in cids[:5]:
                d = lookup_docking_results(cid)
                tool_results.append(d)
                answer_parts.append(_format_docking_answer(d))

    elif intent == "comparison":
        cids = request.candidate_ids or _extract_candidate_ids(query)
        if len(cids) < 2:
            answer_parts.append(
                "Please specify at least two candidate IDs to compare "
                "(e.g. cmpd_015 vs cmpd_024)."
            )
        else:
            # Surface individual lookups in the Evidence panel
            for cid in cids[:5]:
                tool_results.append(lookup_final_ranking(cid))
                tool_results.append(lookup_docking_results(cid))
            result = compare_candidates(cids[:5], registry)
            tool_results.append(result)
            answer_parts.append(
                f"**Recommendation**: {result.recommendation}\n\n" f"{result.reason}"
            )
            if result.warnings:
                all_warnings.extend(result.warnings)
            # Always emit the comparison exploratory caveat
            all_warnings.append(
                "Comparison scores are in-silico only. Conservative scoring weights: "
                "non-covalent (+2), in-domain (+2), ADMET norm, final score, "
                "noise-study L858R_selective (+0.5). No experimental validation."
            )

    elif intent == "report":
        cids = request.candidate_ids or _extract_candidate_ids(query)
        if not cids:
            answer_parts.append(
                "Please specify a candidate ID to generate a report for "
                "(e.g. 'report for cmpd_015')."
            )
        else:
            report = generate_candidate_report(cids[0], registry)
            tool_results.append(report)
            answer_parts.append(report.markdown)
            all_warnings.extend(report.warnings)

    elif intent == "project_qa":
        sections = retrieve(query, top_k=4)
        tool_results.extend(sections)
        sources = [f"{s.source} § {s.header}" for s in sections]
        if sections:
            context = "\n\n---\n\n".join(
                f"**{s.source} — {s.header}**\n{s.content[:600]}" for s in sections
            )
            answer_parts.append(
                "_Relevant documentation sections (keyword retrieval):_\n\n" + context
            )
        else:
            answer_parts.append(
                "No documentation sections matched your query. "
                "Try more specific keywords."
            )

    else:  # unknown
        sections = retrieve(query, top_k=3)
        tool_results.extend(sections)
        sources = [f"{s.source} § {s.header}" for s in sections]
        answer_parts.append(
            "I could not classify your intent. Here are the closest documentation sections:"
        )
        for s in sections:
            answer_parts.append(
                f"\n**{s.source} — {s.header}** (score={s.score:.2f})\n{s.content[:300]}"
            )

    # ── Assemble answer ────────────────────────────────────────────────────────

    raw_answer = "\n\n".join(p for p in answer_parts if p)

    # Guardrail: scan for forbidden experimental claims in the assembled answer
    _, bad_claims = _sanitize_answer(raw_answer)
    if bad_claims:
        all_warnings.append(
            "[GUARDRAIL] Potential unsupported claims detected in answer: "
            + ", ".join(f'"{c}"' for c in bad_claims)
            + ". Review before sharing."
        )

    # Optional LLM summarizer hook (returns None in deterministic v1)
    context_for_llm = raw_answer
    llm_result = llm_summarize(SYSTEM_PROMPT, context_for_llm)
    final_answer = llm_result if llm_result is not None else raw_answer

    # Deduplicate warnings
    seen: set[str] = set()
    unique_warnings: list[str] = []
    for w in all_warnings:
        if w not in seen:
            seen.add(w)
            unique_warnings.append(w)

    return AgentResponse(
        intent=intent,
        answer=final_answer,
        tool_results=tool_results,
        warnings=unique_warnings,
        sources=sources,
    )


def _sanitize_answer(text: str) -> tuple[str, list[str]]:
    """Thin wrapper so tests can patch it independently."""
    return text, find_forbidden_claims(text)


# ── Convenience alias ──────────────────────────────────────────────────────────

MAX_BATCH = 512
