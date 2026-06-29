"""Keyword retrieval over project markdown documentation (no vector DB)."""

from __future__ import annotations

import re
from pathlib import Path

from src.agent.schemas import RetrievalSection

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_DOC_PATHS: list[Path] = [
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "docs" / "PROJECT_WALKTHROUGH.md",
    PROJECT_ROOT / "CLAUDE.md",
]

# Header pattern: one or more # followed by the title
_HEADER_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)


def _load_sections(path: Path) -> list[tuple[str, str, str]]:
    """
    Return list of (source_name, header, body) for each section in a file.

    Splits on markdown headers. Content before the first header is attached to a
    synthetic "Preamble" section.
    """
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8", errors="replace")
    source = path.name

    sections: list[tuple[str, str, str]] = []
    positions: list[tuple[int, str]] = []  # (char offset, header text)

    for m in _HEADER_RE.finditer(text):
        positions.append((m.start(), m.group(2).strip()))

    if not positions:
        sections.append((source, "Preamble", text.strip()))
        return sections

    # Text before first header
    if positions[0][0] > 0:
        sections.append((source, "Preamble", text[: positions[0][0]].strip()))

    for i, (start, header) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        # Body is everything after the header line
        body_start = text.index("\n", start) + 1 if "\n" in text[start:end] else end
        body = text[body_start:end].strip()
        if body:
            sections.append((source, header, body))

    return sections


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens; drop single-character tokens."""
    return {w for w in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(w) > 1}


def _score(query_tokens: set[str], header: str, body: str) -> float:
    """
    Score a section by token overlap with the query.

    Header matches are weighted 3x over body matches so that a section whose
    title names the topic ranks above one that merely mentions it in passing.
    """
    header_tokens = _tokenize(header)
    body_tokens = _tokenize(body)

    header_hits = len(query_tokens & header_tokens)
    body_hits = len(query_tokens & body_tokens)

    # Normalise against query size to prevent long sections from always winning
    n = max(len(query_tokens), 1)
    return (3 * header_hits + body_hits) / n


# Module-level cache: load once per process
_SECTION_CACHE: list[tuple[str, str, str]] | None = None


def _get_sections() -> list[tuple[str, str, str]]:
    global _SECTION_CACHE
    if _SECTION_CACHE is None:
        sections: list[tuple[str, str, str]] = []
        for path in _DOC_PATHS:
            sections.extend(_load_sections(path))
        _SECTION_CACHE = sections
    return _SECTION_CACHE


def retrieve(query: str, top_k: int = 5) -> list[RetrievalSection]:
    """
    Return up to top_k documentation sections most relevant to query.

    Scoring is keyword overlap (header weighted 3x body). No embeddings or
    vector DB required.
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scored: list[tuple[float, str, str, str]] = []
    for source, header, body in _get_sections():
        s = _score(query_tokens, header, body)
        if s > 0:
            scored.append((s, source, header, body))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        RetrievalSection(source=src, header=hdr, content=body, score=sc)
        for sc, src, hdr, body in scored[:top_k]
    ]


def clear_cache() -> None:
    """Force a reload of documentation on the next retrieve() call (useful in tests)."""
    global _SECTION_CACHE
    _SECTION_CACHE = None
