"""
Unit tests for scaffold-diversity metrics in src/generation/sampler.py.

Covers scaffold_stats() and the scaffold keys folded into evaluate_metrics().
Pure RDKit (no PyTorch needed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# scaffold_stats/evaluate_metrics are pure RDKit, but they live in sampler.py
# which imports torch at module top, so importing them pulls in torch. Skip the
# whole module when torch is absent (e.g. CI), matching the other generator tests.
pytest.importorskip("torch", reason="PyTorch not installed")

from src.generation.sampler import evaluate_metrics, scaffold_stats

# Three molecules sharing ONE benzene scaffold + one with a different (pyridine) core
_BENZENES = ["c1ccccc1", "Cc1ccccc1", "CCc1ccccc1"]  # all Bemis-Murcko -> benzene
_PYRIDINE = "c1ccncc1"
_QUINAZOLINE = "c1ccc2ncncc2c1"


@pytest.mark.unit
class TestScaffoldStats:
    def test_required_keys(self):
        s = scaffold_stats(_BENZENES)
        for key in (
            "n_scaffolds",
            "scaffold_diversity",
            "n_novel_scaffolds",
            "scaffold_novelty",
        ):
            assert key in s

    def test_shared_scaffold_counts_once(self):
        # Three benzene derivatives collapse to a single Bemis-Murcko scaffold
        s = scaffold_stats(_BENZENES)
        assert s["n_scaffolds"] == 1

    def test_distinct_scaffolds_counted(self):
        s = scaffold_stats([_BENZENES[0], _PYRIDINE, _QUINAZOLINE])
        assert s["n_scaffolds"] == 3

    def test_diversity_ratio(self):
        # 3 molecules, 1 scaffold -> diversity 1/3
        s = scaffold_stats(_BENZENES)
        assert s["scaffold_diversity"] == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_diversity_all_distinct(self):
        mols = [_BENZENES[0], _PYRIDINE, _QUINAZOLINE]
        s = scaffold_stats(mols)
        assert s["scaffold_diversity"] == pytest.approx(1.0)

    def test_novelty_none_without_train(self):
        s = scaffold_stats(_BENZENES)
        assert s["n_novel_scaffolds"] is None
        assert s["scaffold_novelty"] is None

    def test_novel_scaffold_detected(self):
        # train has only benzene; pyridine + quinazoline are novel scaffolds
        train = {"c1ccccc1"}
        s = scaffold_stats([_BENZENES[0], _PYRIDINE, _QUINAZOLINE], train_smiles=train)
        assert s["n_novel_scaffolds"] == 2

    def test_no_novel_when_all_in_train(self):
        train = {"c1ccccc1"}
        s = scaffold_stats(_BENZENES, train_smiles=train)
        assert s["n_novel_scaffolds"] == 0
        assert s["scaffold_novelty"] == 0.0

    def test_empty_input(self):
        s = scaffold_stats([])
        assert s["n_scaffolds"] == 0
        assert s["scaffold_diversity"] == 0.0


@pytest.mark.unit
class TestEvaluateMetricsScaffolds:
    def test_scaffold_keys_present_by_default(self):
        m = evaluate_metrics(["c1ccccc1", "Cc1ccccc1"])
        for key in (
            "n_scaffolds",
            "scaffold_diversity",
            "n_novel_scaffolds",
            "scaffold_novelty",
        ):
            assert key in m

    def test_scaffold_keys_absent_when_disabled(self):
        m = evaluate_metrics(["c1ccccc1"], compute_scaffolds=False)
        assert "n_scaffolds" not in m

    def test_core_keys_still_present(self):
        # adding scaffold stats must not drop the original metrics
        m = evaluate_metrics(["c1ccccc1", "CCO"])
        for key in ("n_generated", "n_valid", "n_unique", "validity", "uniqueness"):
            assert key in m

    def test_scaffold_novelty_vs_train(self):
        train = {"c1ccccc1"}
        m = evaluate_metrics([_PYRIDINE, _QUINAZOLINE], train_smiles=train)
        # both generated scaffolds differ from benzene
        assert m["scaffold_novelty"] == pytest.approx(1.0)
