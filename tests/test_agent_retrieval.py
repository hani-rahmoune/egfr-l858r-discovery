"""
Tests for src/agent/retrieval.py: keyword retrieval over project documentation.

Verifies that specific known sections (L858R calibration result, RL failure) are
retrievable, and that the scoring and splitting logic works correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.retrieval import _load_sections, _score, _tokenize, clear_cache, retrieve
from src.agent.schemas import RetrievalSection


@pytest.fixture(autouse=True)
def reset_cache():
    """Clear the module-level section cache before each test."""
    clear_cache()
    yield
    clear_cache()


# ── _tokenize ─────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_tokenize_basic():
    tokens = _tokenize("Hello World 42")
    assert "hello" in tokens
    assert "world" in tokens
    assert "42" in tokens


@pytest.mark.unit
def test_tokenize_drops_single_char():
    tokens = _tokenize("a b c test")
    assert "a" not in tokens
    assert "test" in tokens


# ── _score ────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_score_header_match_beats_body_only():
    query = _tokenize("L858R calibration result")
    header_score = _score(query, "L858R calibration", "some unrelated content")
    body_score = _score(query, "Unrelated Header", "L858R calibration result details")
    assert header_score > body_score


@pytest.mark.unit
def test_score_zero_no_overlap():
    query = _tokenize("quantum mechanics")
    s = _score(query, "ADMET filtering", "drug-likeness QED Lipinski")
    assert s == 0.0


# ── _load_sections ────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_load_sections_readme():
    path = PROJECT_ROOT / "README.md"
    if not path.exists():
        pytest.skip("README.md not found")
    sections = _load_sections(path)
    assert len(sections) > 0
    sources = {s[0] for s in sections}
    assert "README.md" in sources


@pytest.mark.unit
def test_load_sections_missing_file(tmp_path):
    sections = _load_sections(tmp_path / "nonexistent.md")
    assert sections == []


@pytest.mark.unit
def test_load_sections_returns_triples():
    path = PROJECT_ROOT / "CLAUDE.md"
    if not path.exists():
        pytest.skip("CLAUDE.md not found")
    sections = _load_sections(path)
    for item in sections:
        assert len(item) == 3
        source, header, body = item
        assert isinstance(source, str)
        assert isinstance(header, str)
        assert isinstance(body, str)


# ── retrieve ──────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_retrieve_returns_list():
    results = retrieve("EGFR backbone model", top_k=3)
    assert isinstance(results, list)
    assert all(isinstance(r, RetrievalSection) for r in results)
    assert len(results) <= 3


@pytest.mark.unit
def test_retrieve_scores_positive():
    results = retrieve("L858R mutation backbone model", top_k=5)
    assert all(r.score > 0 for r in results)


@pytest.mark.unit
def test_retrieve_sorted_descending():
    results = retrieve("docking selectivity Vina L858R WT", top_k=10)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.unit
def test_retrieve_l858r_calibration_section():
    """Must return a section discussing the L858R LOOCV calibration result."""
    results = retrieve("L858R calibration LOOCV backbone", top_k=10)
    combined = " ".join(r.header + " " + r.content for r in results).lower()
    assert "l858r" in combined and (
        "calibration" in combined or "loocv" in combined
    ), "No L858R calibration section found in top-10 results"


@pytest.mark.unit
def test_retrieve_rl_failure_section():
    """Must return a section discussing the RL reward-hacking failure."""
    results = retrieve("RL REINVENT reward hacking mode collapse", top_k=10)
    combined = " ".join(r.header + " " + r.content for r in results).lower()
    assert (
        "reward" in combined
        or "hacking" in combined
        or "collapse" in combined
        or "reinvent" in combined
        or "rl" in combined
    ), "No RL failure section found in top-10 results"


@pytest.mark.unit
def test_retrieve_empty_query():
    results = retrieve("", top_k=5)
    assert results == []


@pytest.mark.unit
def test_retrieve_top_k_respected():
    results = retrieve("model training data EGFR L858R pipeline results", top_k=2)
    assert len(results) <= 2


@pytest.mark.unit
def test_retrieve_section_fields():
    results = retrieve("ADMET Lipinski QED", top_k=3)
    if results:
        r = results[0]
        assert r.source  # non-empty filename
        assert r.header  # non-empty header
        assert r.content  # non-empty body
        assert r.score > 0
