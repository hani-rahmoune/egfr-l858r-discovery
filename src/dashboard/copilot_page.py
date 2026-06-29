"""
Discovery Copilot page for the EGFR L858R dashboard (Phase 27).

Wires a chat interface to controller.handle() in deterministic mode (no LLM).
Models are lazy-loaded once via @st.cache_resource; on the initial render
(empty chat history) no model loading is triggered, so the page opens instantly.

Three display panels per response:
  1. Grounded answer   -- st.markdown over resp.answer
  2. Evidence          -- which tool functions were called, and what they returned
  3. Warnings          -- guardrail caveats (scientific + forbidden-claim flags)

A markdown download button appears when the response contains a CandidateReport.
"""

from __future__ import annotations

import re
from typing import Any

import streamlit as st

# ── Example prompts ────────────────────────────────────────────────────────────

EXAMPLE_PROMPTS: list[tuple[str, str]] = [
    ("Single molecule", "Predict: COc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OC"),
    ("Compare", "Compare cmpd_015 and cmpd_024"),
    ("Candidate report", "Generate a report for gen_005"),
    ("Explain ranking", "Explain the ranking for cmpd_002"),
    ("Project QA", "Why did the RL training fail?"),
]

# Regex matching candidate IDs embedded in the query text
_CID_RE = re.compile(r"\b(cmpd_\d+|gen_\d+)\b", re.IGNORECASE)

# ── Pure helper functions (no Streamlit, fully testable) ───────────────────────


def format_evidence(tool_results: list[Any]) -> list[str]:
    """
    Convert a list of tool result objects into human-readable evidence strings.

    Uses type name matching so no imports of agent schemas are needed here.
    """
    lines: list[str] = []
    for r in tool_results:
        kind = type(r).__name__

        if kind == "PredictToolResult":
            smiles_short = (r.smiles[:40] + "...") if len(r.smiles) > 40 else r.smiles
            lines.append(
                f"predict_smiles({smiles_short!r}) "
                f"→ valid={r.valid}"
                + (f", pIC50={r.pic50_mutant:.3f}" if r.pic50_mutant else "")
            )

        elif kind == "BatchPredictToolResult":
            lines.append(
                f"batch_predict(n={r.n}) "
                f"→ {r.n_valid} valid, {r.n_invalid} invalid"
            )

        elif kind == "RankingLookupResult":
            suffix = f", rank={r.rank}/68, score={r.final_score:.3f}" if r.found else ""
            lines.append(
                f"lookup_final_ranking({r.candidate_id!r}) → found={r.found}{suffix}"
            )

        elif kind == "DockingLookupResult":
            if r.found and r.selectivity_delta is not None:
                suffix = (
                    f", delta={r.selectivity_delta:.3f} kcal/mol ({r.direction})"
                    + (f", noise={r.noise_call}" if r.noise_call else "")
                )
            else:
                suffix = ""
            lines.append(
                f"lookup_docking_results({r.candidate_id!r}) → found={r.found}{suffix}"
            )

        elif kind == "ComparisonResult":
            lines.append(
                f"compare_candidates({r.candidate_ids}) "
                f"→ recommendation={r.recommendation!r}"
            )

        elif kind == "CandidateReport":
            lines.append(
                f"generate_candidate_report({r.candidate_id!r}) "
                f"→ {len(r.markdown):,} char report"
            )

        elif kind == "RetrievalSection":
            lines.append(
                f"retrieve() → {r.source} § {r.header!r} (score={r.score:.2f})"
            )

        else:
            lines.append(f"{kind}(...)")

    return lines


def extract_report_markdown(tool_results: list[Any]) -> str | None:
    """Return the markdown string from the first CandidateReport in tool_results, or None."""
    for r in tool_results:
        if type(r).__name__ == "CandidateReport":
            return r.markdown
    return None


def get_download_filename(tool_results: list[Any]) -> str:
    """Return a descriptive download filename when a CandidateReport is present."""
    for r in tool_results:
        if type(r).__name__ == "CandidateReport":
            return f"report_{r.candidate_id}.md"
    return "report.md"


# ── Registry (lazy, cached per session) ───────────────────────────────────────


@st.cache_resource(show_spinner="Loading models for Discovery Copilot…")
def _registry():
    """Load ModelRegistry once per process; never invoked on the empty initial render."""
    from src.api.services import ModelRegistry

    return ModelRegistry.load()


# ── Internal rendering helpers ─────────────────────────────────────────────────


def _render_response(resp: Any) -> None:
    """Render the three panels for one AgentResponse."""
    # Panel 1: grounded answer
    st.markdown(resp.answer)

    # Panel 2: evidence
    evidence = format_evidence(resp.tool_results)
    if evidence:
        with st.expander(f"Evidence — {len(evidence)} tool call(s)", expanded=False):
            for line in evidence:
                st.code(line, language=None)
            if resp.sources:
                st.caption("Documentation sources: " + " | ".join(resp.sources))

    # Panel 3: warnings
    if resp.warnings:
        with st.expander(f"Warnings ({len(resp.warnings)})", expanded=False):
            for w in resp.warnings:
                if w.startswith("[GUARDRAIL]"):
                    st.error(w)
                else:
                    st.warning(w)

    # Download button (only when a CandidateReport is present)
    md = extract_report_markdown(resp.tool_results)
    if md is not None:
        fname = get_download_filename(resp.tool_results)
        st.download_button(
            label="Download report as Markdown",
            data=md,
            file_name=fname,
            mime="text/markdown",
        )


def _handle_query(query: str) -> None:
    """Run the controller on query, append result to session history, rerun."""
    from src.agent.controller import handle
    from src.agent.schemas import AgentRequest

    cids = [m.group(0).lower() for m in _CID_RE.finditer(query)]
    req = AgentRequest(query=query, candidate_ids=cids)

    with st.spinner("Consulting precomputed artifacts…"):
        try:
            resp = handle(req, registry=_registry())
        except Exception as exc:  # pragma: no cover
            st.error(f"Controller error: {exc}")
            return

    history: list[tuple[str, Any]] = st.session_state.get("copilot_history", [])
    history.append((query, resp))
    st.session_state["copilot_history"] = history
    st.rerun()


# ── Page entry point ───────────────────────────────────────────────────────────


def page_copilot() -> None:
    """Render the Discovery Copilot chat page."""
    st.header("Discovery Copilot")
    st.caption(
        "Deterministic mode: all answers come from precomputed in-silico artifacts. "
        "No LLM or API key required. All outputs are EXPLORATORY."
    )

    # Sidebar note
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "**Discovery Copilot** runs offline against precomputed results. "
        "The LLM hook is not wired (v1); answers are pure tool output."
    )

    # Example prompt buttons (always visible above the chat)
    st.subheader("Example queries")
    cols = st.columns(len(EXAMPLE_PROMPTS))
    for col, (label, prompt) in zip(cols, EXAMPLE_PROMPTS):
        if col.button(label, use_container_width=True, key=f"copilot_ex_{label}"):
            st.session_state["copilot_pending"] = prompt

    st.divider()

    # Check for a pending query set by an example button
    pending: str | None = st.session_state.pop("copilot_pending", None)
    if pending:
        _handle_query(pending)
        return  # rerun will re-enter here; history is now populated

    # Chat history (oldest first)
    history: list[tuple[str, Any]] = st.session_state.get("copilot_history", [])
    for query, resp in history:
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            _render_response(resp)

    # New chat input at the bottom
    if query := st.chat_input("Ask the Discovery Copilot…"):
        _handle_query(query)
