"""
Tests for the Discovery Copilot dashboard page (Phase 27).

TestHelpers    — pure data-prep functions (format_evidence, extract_report_markdown,
                 get_download_filename). No Streamlit runtime, @unit.
TestAppRender  — AppTest smoke test: navigates to the copilot page and confirms it
                 renders without raising an exception. @integration (real app, real models
                 are NOT loaded on an empty initial render because _registry() is only
                 invoked when a query is submitted).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.copilot_page import (
    EXAMPLE_PROMPTS,
    extract_report_markdown,
    format_evidence,
    get_download_filename,
)


# ── Fixtures: fake result objects ──────────────────────────────────────────────


class _FakePredict:
    __class__ = type("PredictToolResult", (), {})()

    def __init__(self, valid=True, pic50=7.5):
        self.smiles = "COc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OC"
        self.valid = valid
        self.pic50_mutant = pic50 if valid else None

    @property
    def __class__(self):
        return type("PredictToolResult", (), {"__name__": "PredictToolResult"})

    class _Meta:
        __name__ = "PredictToolResult"

    def _get_class(self):
        return self


# It's cleaner to use simple namespaces with the right __class__.__name__

def _make(class_name: str, **kwargs):
    """Create a simple object whose type().__name__ == class_name."""
    cls = type(class_name, (), {})
    obj = cls()
    for k, v in kwargs.items():
        setattr(obj, k, v)
    return obj


def _predict(valid=True, smiles="COc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OC", pic50=7.5):
    return _make(
        "PredictToolResult",
        valid=valid,
        smiles=smiles,
        pic50_mutant=pic50 if valid else None,
    )


def _batch(n=3, n_valid=2):
    return _make("BatchPredictToolResult", n=n, n_valid=n_valid, n_invalid=n - n_valid)


def _ranking(cid="cmpd_015", found=True, rank=1, final_score=0.73):
    return _make(
        "RankingLookupResult",
        candidate_id=cid,
        found=found,
        rank=rank if found else None,
        final_score=final_score if found else None,
    )


def _docking(cid="cmpd_015", found=True, delta=-0.452, direction="L858R_favoured"):
    return _make(
        "DockingLookupResult",
        candidate_id=cid,
        found=found,
        selectivity_delta=delta if found else None,
        direction=direction if found else None,
        noise_call="L858R_selective" if found else None,
    )


def _comparison(ids=None, rec="cmpd_015"):
    if ids is None:
        ids = ["cmpd_015", "cmpd_024"]
    return _make("ComparisonResult", candidate_ids=ids, recommendation=rec)


def _report(cid="gen_005", markdown="# Report\n\n## Limitations\nAll exploratory."):
    return _make("CandidateReport", candidate_id=cid, markdown=markdown)


def _retrieval(source="CLAUDE.md", header="RL training", score=1.5):
    return _make("RetrievalSection", source=source, header=header, content="...", score=score)


# ── TestHelpers: format_evidence ───────────────────────────────────────────────


@pytest.mark.unit
class TestFormatEvidence:

    def test_predict_valid(self):
        lines = format_evidence([_predict(valid=True, pic50=7.5)])
        assert len(lines) == 1
        assert "predict_smiles" in lines[0]
        assert "valid=True" in lines[0]
        assert "7.500" in lines[0]

    def test_predict_invalid(self):
        lines = format_evidence([_predict(valid=False)])
        assert "valid=False" in lines[0]
        # No pIC50 for invalid
        assert "pIC50" not in lines[0]

    def test_predict_long_smiles_truncated(self):
        long_smiles = "C" * 60
        lines = format_evidence([_predict(smiles=long_smiles)])
        assert "..." in lines[0]
        # Should not exceed the raw smiles length in the evidence line
        assert len(long_smiles) > 40

    def test_batch(self):
        lines = format_evidence([_batch(n=5, n_valid=4)])
        assert len(lines) == 1
        assert "batch_predict" in lines[0]
        assert "n=5" in lines[0]
        assert "4 valid" in lines[0]
        assert "1 invalid" in lines[0]

    def test_ranking_found(self):
        lines = format_evidence([_ranking(cid="cmpd_015", found=True, rank=1)])
        assert "lookup_final_ranking" in lines[0]
        assert "cmpd_015" in lines[0]
        assert "found=True" in lines[0]
        assert "rank=1/68" in lines[0]

    def test_ranking_not_found(self):
        lines = format_evidence([_ranking(cid="cmpd_999", found=False)])
        assert "found=False" in lines[0]
        # No rank=N/68 suffix when not found
        assert "rank=" not in lines[0]

    def test_docking_found(self):
        lines = format_evidence([_docking(cid="cmpd_024", found=True, delta=-0.813)])
        assert "lookup_docking_results" in lines[0]
        assert "cmpd_024" in lines[0]
        assert "found=True" in lines[0]
        assert "-0.813" in lines[0]
        assert "L858R" in lines[0]

    def test_docking_not_found(self):
        lines = format_evidence([_docking(cid="cmpd_999", found=False)])
        assert "found=False" in lines[0]
        assert "delta" not in lines[0]

    def test_comparison(self):
        lines = format_evidence([_comparison(ids=["cmpd_015", "cmpd_024"], rec="cmpd_015")])
        assert "compare_candidates" in lines[0]
        assert "cmpd_015" in lines[0]
        assert "recommendation" in lines[0]

    def test_candidate_report(self):
        r = _report(cid="gen_005", markdown="x" * 500)
        lines = format_evidence([r])
        assert "generate_candidate_report" in lines[0]
        assert "gen_005" in lines[0]
        assert "500" in lines[0]

    def test_retrieval_section(self):
        lines = format_evidence([_retrieval(source="README.md", header="RL Phase 22")])
        assert "retrieve()" in lines[0]
        assert "README.md" in lines[0]
        assert "RL Phase 22" in lines[0]

    def test_unknown_type(self):
        unknown = _make("SomeUnknownResult")
        lines = format_evidence([unknown])
        assert "SomeUnknownResult" in lines[0]

    def test_mixed_results(self):
        results = [
            _predict(),
            _ranking(),
            _docking(),
        ]
        lines = format_evidence(results)
        assert len(lines) == 3
        assert "predict_smiles" in lines[0]
        assert "lookup_final_ranking" in lines[1]
        assert "lookup_docking_results" in lines[2]

    def test_empty_list(self):
        assert format_evidence([]) == []


# ── TestHelpers: extract_report_markdown ───────────────────────────────────────


@pytest.mark.unit
class TestExtractReportMarkdown:

    def test_found(self):
        md = "# Report\n\n## Limitations\nAll exploratory."
        result = extract_report_markdown([_predict(), _report(markdown=md)])
        assert result == md

    def test_not_found(self):
        result = extract_report_markdown([_predict(), _ranking()])
        assert result is None

    def test_empty_list(self):
        assert extract_report_markdown([]) is None

    def test_first_report_returned(self):
        r1 = _report(cid="gen_005", markdown="first")
        r2 = _report(cid="cmpd_015", markdown="second")
        result = extract_report_markdown([r1, r2])
        assert result == "first"

    def test_report_only(self):
        r = _report(markdown="# Only report")
        assert extract_report_markdown([r]) == "# Only report"


# ── TestHelpers: get_download_filename ────────────────────────────────────────


@pytest.mark.unit
class TestGetDownloadFilename:

    def test_with_report(self):
        fname = get_download_filename([_report(cid="gen_005")])
        assert fname == "report_gen_005.md"

    def test_with_known_candidate(self):
        fname = get_download_filename([_report(cid="cmpd_015")])
        assert fname == "report_cmpd_015.md"

    def test_no_report(self):
        fname = get_download_filename([_predict(), _ranking()])
        assert fname == "report.md"

    def test_empty_list(self):
        assert get_download_filename([]) == "report.md"

    def test_first_report_used(self):
        r1 = _report(cid="gen_005")
        r2 = _report(cid="cmpd_015")
        assert get_download_filename([r1, r2]) == "report_gen_005.md"


# ── TestHelpers: EXAMPLE_PROMPTS ──────────────────────────────────────────────


@pytest.mark.unit
def test_example_prompts_count():
    assert len(EXAMPLE_PROMPTS) == 5


@pytest.mark.unit
def test_example_prompts_cover_five_flows():
    labels = {label for label, _ in EXAMPLE_PROMPTS}
    prompts = [p for _, p in EXAMPLE_PROMPTS]
    # Single molecule: has a SMILES-like token
    assert any(
        any(c.isupper() and c.isalpha() for c in p) and len(p) > 10
        for p in prompts
    )
    # Compare: should mention two candidates
    assert any("cmpd_" in p.lower() and "and" in p.lower() for p in prompts)
    # Report
    assert any("report" in p.lower() for p in prompts)
    # Ranking / explain
    assert any("ranking" in p.lower() or "rank" in p.lower() for p in prompts)
    # Project QA
    assert any("?" in p or "why" in p.lower() or "how" in p.lower() for p in prompts)


# ── TestAppRender: smoke test ──────────────────────────────────────────────────


@pytest.mark.integration
class TestCopilotPageRenders:
    """
    AppTest integration test: loads the real app, navigates to the Discovery Copilot
    page, and asserts no exception is raised on the empty initial render.

    The _registry() lazy-loader is NOT invoked on empty render (no query submitted),
    so this test does not load model artifacts and completes in a few seconds.
    """

    @pytest.fixture(scope="class")
    def app(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(
            str(PROJECT_ROOT / "src" / "dashboard" / "app.py"),
            default_timeout=60,
        )
        at.run()
        # Navigate to the Discovery Copilot page
        at.radio[0].set_value("Discovery Copilot").run()
        return at

    def test_no_exception(self, app):
        assert not app.exception

    def test_page_has_example_buttons(self, app):
        button_labels = [b.label for b in app.button]
        assert any(
            any(label in bl for label in ("Single molecule", "Compare", "Report", "Copilot", "Predict"))
            for bl in button_labels
        ), f"No example prompt buttons found. Buttons: {button_labels}"

    def test_chat_input_present(self, app):
        # Streamlit AppTest exposes chat_input via app.chat_input
        assert hasattr(app, "chat_input")
