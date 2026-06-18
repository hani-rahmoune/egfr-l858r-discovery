"""
Unit tests for Phase 22 RL reward logic and REINVENT NLL formula.

Coverage:
  TestSigmoid            — sigmoid helper shape / monotonicity
  TestRewardComponents   — each reward component fires correctly
  TestRewardBounds       — invalid SMILES returns penalty; bounds hold
  TestMoleculeRewardBatch — batch callable: shape, order, gradient-free
  TestNLLBatch           — compute_nll_batch: shape, positivity, PAD masking
  TestREINVENTFormula    — AugNLL formula and loss semantics (no training)

Molecules:
  GEFITINIB  — EGFR inhibitor (non-covalent, drug-like)
  OSIMERTINIB — covalent 3rd-gen EGFR inhibitor
  ASPIRIN    — small clean drug, low predicted activity vs EGFR
  INVALID    — unparseable string
  TINY       — methane (extreme MW < 100)

All tests use a MockBackbone that returns a fixed pIC50, so no real model
artifact is needed.  PyTorch is required only for TestNLLBatch /
TestREINVENTFormula; those tests are skipped if torch is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.reward import (
    _DEFAULTS,
    MoleculeReward,
    _sigmoid,
    compute_reward,
)
from src.scoring.applicability_domain import ApplicabilityDomain

# ── Test molecules ─────────────────────────────────────────────────────────────

GEFITINIB = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
OSIMERTINIB = "C=CC(=O)Nc1cc2c(Nc3cccc(NC(=O)/C=C/CN(C)C)c3)ncnc2cc1OC"  # acrylamide
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
INVALID = "not_a_smiles$$"
TINY = "C"  # methane — MW=16, extreme range

# ── Fixtures ──────────────────────────────────────────────────────────────────


class _MockBackbone:
    """Returns a fixed pIC50 for any molecule."""

    def __init__(self, pic50: float = 8.0):
        self._v = pic50

    def predict(self, X):
        return np.full(len(X), self._v, dtype=np.float32)


def _make_ad(in_domain_smi: list[str]) -> ApplicabilityDomain:
    """Fit an AD on the supplied molecules (use same SMILES as query to test in_domain)."""
    ad = ApplicabilityDomain.from_config()
    ad.fit(in_domain_smi)
    return ad


def _make_reward(
    pic50: float = 8.0,
    ad_train: list[str] | None = None,
    train_smiles: set[str] | None = None,
    cfg: dict | None = None,
):
    backbone = _MockBackbone(pic50)
    ad = _make_ad(ad_train or [GEFITINIB])
    ts = train_smiles or set()
    return backbone, ad, ts, cfg or {}


# ── TestSigmoid ───────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSigmoid:

    def test_at_center_returns_half(self):
        assert _sigmoid(7.0, center=7.0) == pytest.approx(0.5, abs=1e-6)

    def test_above_center_above_half(self):
        assert _sigmoid(8.0, center=7.0) > 0.5

    def test_below_center_below_half(self):
        assert _sigmoid(5.0, center=7.0) < 0.5

    def test_monotone_increasing(self):
        vals = [_sigmoid(x, center=7.0) for x in [5, 6, 7, 8, 9, 10]]
        assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

    def test_output_in_zero_one(self):
        # use moderate values that don't saturate sigmoid to exactly 0.0 or 1.0 in float
        for x in [-5, 0, 7, 15, 20]:
            v = _sigmoid(x, center=7.0)
            assert 0.0 < v < 1.0


# ── TestRewardComponents ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestRewardComponents:

    def test_invalid_returns_penalty(self):
        backbone, ad, ts, cfg = _make_reward()
        r = compute_reward(INVALID, ts, ad, backbone, cfg)
        assert r == pytest.approx(_DEFAULTS["invalid_penalty"])

    def test_valid_molecule_returns_float(self):
        backbone, ad, ts, cfg = _make_reward(ad_train=[GEFITINIB])
        r = compute_reward(GEFITINIB, ts, ad, backbone, cfg)
        assert isinstance(r, float)

    def test_covalent_gets_lower_reward_than_noncovalent(self):
        backbone, ad, ts, _ = _make_reward(ad_train=[GEFITINIB, OSIMERTINIB], pic50=8.0)
        r_non = compute_reward(GEFITINIB, ts, ad, backbone)
        r_cov = compute_reward(OSIMERTINIB, ts, ad, backbone)
        assert r_non > r_cov

    def test_novelty_bonus_increases_reward(self):
        backbone, ad, ts_with, _ = _make_reward(
            ad_train=[GEFITINIB], train_smiles={GEFITINIB}
        )
        ts_without: set[str] = set()
        r_novel = compute_reward(GEFITINIB, ts_without, ad, backbone)
        r_seen = compute_reward(GEFITINIB, ts_with, ad, backbone)
        assert r_novel > r_seen

    def test_ood_penalty_reduces_reward(self):
        # AD fitted on ASPIRIN only → GEFITINIB will likely be borderline/OOD
        backbone = _MockBackbone(8.0)
        ad_on_aspirin = _make_ad([ASPIRIN])
        ad_on_egfr = _make_ad([GEFITINIB])
        ts: set[str] = set()
        r_in = compute_reward(GEFITINIB, ts, ad_on_egfr, backbone)
        r_out = compute_reward(GEFITINIB, ts, ad_on_aspirin, backbone)
        # In-domain should score higher than out-of-domain (or borderline)
        assert r_in > r_out

    def test_high_pic50_scores_higher(self):
        _, ad, ts, _ = _make_reward(pic50=9.5, ad_train=[GEFITINIB])
        backbone_high = _MockBackbone(9.5)
        backbone_low = _MockBackbone(4.0)
        r_high = compute_reward(GEFITINIB, ts, ad, backbone_high)
        r_low = compute_reward(GEFITINIB, ts, ad, backbone_low)
        assert r_high > r_low

    def test_tiny_molecule_range_penalty_fires(self):
        backbone, ad, ts, _ = _make_reward(ad_train=[GEFITINIB, TINY])
        r_tiny = compute_reward(TINY, ts, ad, backbone)
        r_gef = compute_reward(GEFITINIB, ts, ad, backbone)
        # Tiny MW < 100 should trigger range_penalty making tiny score lower
        assert r_tiny < r_gef

    def test_covalent_penalty_default_value(self):
        # Verify covalent penalty is applied (osimertinib has acrylamide)
        backbone = _MockBackbone(8.0)
        ad = _make_ad([GEFITINIB, OSIMERTINIB])
        cfg_zero = {  # zero out all other components for isolation
            "activity_weight": 0.0,
            "qed_weight": 0.0,
            "admet_bonus": 0.0,
            "novelty_bonus": 0.0,
            "out_of_domain_penalty": 0.0,
            "borderline_penalty": 0.0,
            "range_penalty": 0.0,
        }
        r_cov = compute_reward(OSIMERTINIB, set(), ad, backbone, cfg_zero)
        r_noncov = compute_reward(GEFITINIB, set(), ad, backbone, cfg_zero)
        assert r_cov < r_noncov


# ── TestRewardBounds ──────────────────────────────────────────────────────────


@pytest.mark.unit
class TestRewardBounds:

    def test_invalid_exactly_penalty(self):
        backbone, ad, ts, cfg = _make_reward()
        r = compute_reward(INVALID, ts, ad, backbone, cfg)
        assert r == _DEFAULTS["invalid_penalty"]

    def test_valid_reward_above_penalty(self):
        backbone, ad, ts, _ = _make_reward(ad_train=[GEFITINIB])
        r = compute_reward(GEFITINIB, ts, ad, backbone)
        assert r > _DEFAULTS["invalid_penalty"]

    def test_reward_cfg_override(self):
        backbone, ad, ts, _ = _make_reward()
        cfg_override = {
            "activity_weight": 0.0,
            "qed_weight": 0.0,
            "admet_bonus": 0.0,
            "novelty_bonus": 0.0,
            "covalent_penalty": -0.99,
        }
        r = compute_reward(OSIMERTINIB, ts, ad, backbone, cfg_override)
        assert r < 0


# ── TestMoleculeRewardBatch ───────────────────────────────────────────────────


@pytest.mark.unit
class TestMoleculeRewardBatch:

    def _make_batch_reward(self):
        backbone = _MockBackbone(8.0)
        ad = _make_ad([GEFITINIB])
        return MoleculeReward(
            backbone_model=backbone, ad=ad, train_smiles=set(), cfg={}
        )

    def test_returns_ndarray(self):
        fn = self._make_batch_reward()
        r = fn([GEFITINIB, ASPIRIN])
        assert isinstance(r, np.ndarray)

    def test_shape_matches_input(self):
        fn = self._make_batch_reward()
        smiles = [GEFITINIB, ASPIRIN, INVALID, TINY]
        r = fn(smiles)
        assert r.shape == (4,)

    def test_order_preserved(self):
        fn = self._make_batch_reward()
        r_batch = fn([GEFITINIB, INVALID])
        r_gef = fn([GEFITINIB])[0]
        r_inv = fn([INVALID])[0]
        assert r_batch[0] == pytest.approx(r_gef, abs=1e-3)
        assert r_batch[1] == pytest.approx(r_inv, abs=1e-3)

    def test_invalid_gets_penalty_in_batch(self):
        fn = self._make_batch_reward()
        r = fn([INVALID])
        assert r[0] == pytest.approx(_DEFAULTS["invalid_penalty"])

    def test_empty_list(self):
        fn = self._make_batch_reward()
        r = fn([])
        assert r.shape == (0,)


# ── TestNLLBatch ─────────────────────────────────────────────────────────────

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from src.generation.model import GRUGenerator
from src.generation.rl_trainer import (
    ScaffoldMemory,
    compare_pre_post,
    compute_nll_batch,
)
from src.generation.tokenizer import SMILESTokenizer

_TINY_SMILES = ["CCO", "CCN", "CCC", "c1ccccc1", "Cc1ccccc1"]
_ARCH = dict(embed_dim=8, hidden_dim=16, num_layers=1, dropout=0.0)


def _tiny_tok() -> SMILESTokenizer:
    return SMILESTokenizer().fit(_TINY_SMILES)


def _tiny_model(tok: SMILESTokenizer) -> GRUGenerator:
    m = GRUGenerator(vocab_size=tok.vocab_size, pad_idx=tok.pad_idx, **_ARCH)
    m.eval()
    return m


@pytest.mark.unit
class TestNLLBatch:

    def test_returns_tensor_correct_shape(self):
        tok = _tiny_tok()
        model = _tiny_model(tok)
        smiles = ["CCO", "CCN", "CCC"]
        with torch.no_grad():
            nll = compute_nll_batch(model, tok, smiles, torch.device("cpu"))
        assert nll.shape == (3,)

    def test_nll_positive(self):
        tok = _tiny_tok()
        model = _tiny_model(tok)
        with torch.no_grad():
            nll = compute_nll_batch(
                model, tok, ["CCO", "c1ccccc1"], torch.device("cpu")
            )
        assert (nll > 0).all()

    def test_single_molecule_batch(self):
        tok = _tiny_tok()
        model = _tiny_model(tok)
        with torch.no_grad():
            nll = compute_nll_batch(model, tok, ["CCO"], torch.device("cpu"))
        assert nll.shape == (1,)
        assert float(nll[0]) > 0

    def test_invalid_smiles_does_not_crash(self):
        tok = _tiny_tok()
        model = _tiny_model(tok)
        with torch.no_grad():
            nll = compute_nll_batch(model, tok, [INVALID, "CCO"], torch.device("cpu"))
        assert nll.shape == (2,)
        assert (nll >= 0).all()

    def test_different_lengths_processed_correctly(self):
        tok = _tiny_tok()
        model = _tiny_model(tok)
        # Mixed lengths: short "C" and longer "c1ccccc1"
        with torch.no_grad():
            nll = compute_nll_batch(
                model, tok, ["C", "c1ccccc1CC"], torch.device("cpu")
            )
        assert nll.shape == (2,)

    def test_gradient_flows_through_agent_nll(self):
        """NLL for the agent (no no_grad context) should have gradients."""
        tok = _tiny_tok()
        model = _tiny_model(tok)
        model.train()
        nll = compute_nll_batch(model, tok, ["CCO", "CCN"], torch.device("cpu"))
        # Sum should be differentiable
        nll.sum().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0


# ── TestREINVENTFormula ───────────────────────────────────────────────────────


@pytest.mark.unit
class TestREINVENTFormula:

    def test_aug_nll_formula_zero_score(self):
        """When score=0, AugNLL = NLL_prior, so loss = (NLL_agent - NLL_prior)^2."""
        nll_agent = torch.tensor([1.2, 0.8, 1.5])
        nll_prior = torch.tensor([1.0, 0.9, 1.3])
        sigma = 0.5
        scores = torch.zeros(3)

        aug_nll = nll_prior - sigma * scores
        loss = ((nll_agent - aug_nll) ** 2).mean()

        expected = ((nll_agent - nll_prior) ** 2).mean()
        assert loss == pytest.approx(expected.item(), abs=1e-6)

    def test_positive_score_reduces_aug_nll_target(self):
        """High reward should lower AugNLL, pulling agent NLL down (learn it)."""
        nll_prior = torch.tensor([1.0])
        sigma = 0.5
        score_hi = torch.tensor([1.0])
        score_lo = torch.tensor([0.0])

        aug_nll_hi = nll_prior - sigma * score_hi  # 1.0 - 0.5 = 0.5
        aug_nll_lo = nll_prior - sigma * score_lo  # 1.0 - 0.0 = 1.0

        assert aug_nll_hi < aug_nll_lo

    def test_negative_score_increases_aug_nll_target(self):
        """Negative reward raises AugNLL target — loss pushes agent NLL up (avoid)."""
        nll_prior = torch.tensor([1.0])
        sigma = 0.5
        score_bad = torch.tensor([-1.0])
        aug_nll = nll_prior - sigma * score_bad  # 1.0 - (-0.5) = 1.5

        assert float(aug_nll) > float(nll_prior)

    def test_zero_loss_when_agent_equals_aug_nll(self):
        """Loss = 0 when NLL_agent exactly matches the augmented target."""
        nll_prior = torch.tensor([1.0, 0.8])
        sigma = 0.5
        scores = torch.tensor([0.4, 0.2])  # aug_nll = [0.8, 0.7]
        aug_nll = nll_prior - sigma * scores
        nll_agent = aug_nll.clone()  # perfect match

        loss = ((nll_agent - aug_nll) ** 2).mean()
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_loss_scale_with_sigma(self):
        """Doubling sigma should change how much score affects aug_nll."""
        nll_prior = torch.tensor([1.0])
        scores = torch.tensor([1.0])
        nll_agent = torch.tensor([0.5])

        loss_low = ((nll_agent - (nll_prior - 0.5 * scores)) ** 2).mean()
        loss_high = ((nll_agent - (nll_prior - 1.0 * scores)) ** 2).mean()

        # With higher sigma the AugNLL target is lower → larger gap from agent NLL
        # (0.5 vs 0.0 target → different losses)
        assert loss_low != loss_high


# ── TestScaffoldMemory ────────────────────────────────────────────────────────

# Three molecules that share ONE Bemis-Murcko scaffold (benzene) and one with a
# distinct scaffold, so we can test per-scaffold bucketing.
_BENZENE_1 = "Cc1ccccc1"  # toluene  → benzene scaffold
_BENZENE_2 = "CCc1ccccc1"  # ethylbenzene → benzene scaffold
_BENZENE_3 = "CCCc1ccccc1"  # propylbenzene → benzene scaffold
_PYRIDINE = "Cc1ccccn1"  # picoline → pyridine scaffold


@pytest.mark.unit
class TestScaffoldMemory:

    def test_under_bucket_passes_through(self):
        mem = ScaffoldMemory(bucket_size=5, min_score=0.3, penalty=0.0)
        smis = [_BENZENE_1, _BENZENE_2]
        rew = np.array([0.8, 0.7], dtype=np.float32)
        out = mem.apply(smis, rew)
        np.testing.assert_allclose(out, rew)

    def test_bucket_saturates_and_penalizes(self):
        mem = ScaffoldMemory(bucket_size=2, min_score=0.3, penalty=0.0)
        # 3 benzene-scaffold molecules; bucket_size=2 → the 3rd gets penalised
        smis = [_BENZENE_1, _BENZENE_2, _BENZENE_3]
        rew = np.array([0.8, 0.8, 0.8], dtype=np.float32)
        out = mem.apply(smis, rew)
        assert out[0] == pytest.approx(0.8)
        assert out[1] == pytest.approx(0.8)
        assert out[2] == pytest.approx(0.0)  # saturated → penalty

    def test_penalty_value_used(self):
        mem = ScaffoldMemory(bucket_size=1, min_score=0.3, penalty=-0.5)
        smis = [_BENZENE_1, _BENZENE_2]
        rew = np.array([0.9, 0.9], dtype=np.float32)
        out = mem.apply(smis, rew)
        assert out[0] == pytest.approx(0.9)
        assert out[1] == pytest.approx(-0.5)

    def test_below_min_score_not_counted(self):
        mem = ScaffoldMemory(bucket_size=1, min_score=0.5, penalty=0.0)
        # first benzene scores below min_score → does NOT fill the bucket
        smis = [_BENZENE_1, _BENZENE_2]
        rew = np.array([0.2, 0.9], dtype=np.float32)
        out = mem.apply(smis, rew)
        # bucket was never filled by the low-scoring one, so the 0.9 passes through
        assert out[1] == pytest.approx(0.9)

    def test_distinct_scaffolds_independent(self):
        mem = ScaffoldMemory(bucket_size=1, min_score=0.3, penalty=0.0)
        smis = [_BENZENE_1, _PYRIDINE, _BENZENE_2]
        rew = np.array([0.8, 0.8, 0.8], dtype=np.float32)
        out = mem.apply(smis, rew)
        assert out[0] == pytest.approx(0.8)  # first benzene fills its bucket
        assert out[1] == pytest.approx(0.8)  # pyridine is a different scaffold
        assert out[2] == pytest.approx(0.0)  # second benzene → saturated

    def test_does_not_mutate_input(self):
        mem = ScaffoldMemory(bucket_size=1, min_score=0.3, penalty=0.0)
        smis = [_BENZENE_1, _BENZENE_2]
        rew = np.array([0.8, 0.8], dtype=np.float32)
        _ = mem.apply(smis, rew)
        np.testing.assert_allclose(rew, np.array([0.8, 0.8]))  # unchanged

    def test_invalid_smiles_passes_through(self):
        mem = ScaffoldMemory(bucket_size=1, min_score=0.3, penalty=0.0)
        smis = [INVALID, _BENZENE_1]
        rew = np.array([0.8, 0.8], dtype=np.float32)
        out = mem.apply(smis, rew)
        assert out[0] == pytest.approx(0.8)  # invalid → no scaffold → unchanged

    def test_persists_across_batches(self):
        mem = ScaffoldMemory(bucket_size=2, min_score=0.3, penalty=0.0)
        rew = np.array([0.8], dtype=np.float32)
        mem.apply([_BENZENE_1], rew)  # count benzene = 1
        mem.apply([_BENZENE_2], rew)  # count benzene = 2 (now full)
        out = mem.apply([_BENZENE_3], rew)  # 3rd batch → saturated
        assert out[0] == pytest.approx(0.0)

    def test_counters_report(self):
        # bucket_size=2: benzene reaches 3 (saturated, 1 penalised); pyridine stays at 1
        mem = ScaffoldMemory(bucket_size=2, min_score=0.3, penalty=0.0)
        mem.apply(
            [_BENZENE_1, _BENZENE_2, _BENZENE_3, _PYRIDINE],
            np.array([0.8, 0.8, 0.8, 0.8], dtype=np.float32),
        )
        assert mem.n_scaffolds == 2  # benzene + pyridine
        assert mem.n_saturated == 1  # only benzene reached bucket_size
        assert mem.n_penalized == 1  # the third benzene was penalised


# ── TestVerdictLogic ──────────────────────────────────────────────────────────


def _eval_dict(
    validity=0.93,
    uniqueness=0.80,
    scaffold_diversity=0.58,
    mean_pic50=6.7,
    admet_pass_rate=0.31,
    in_domain_rate=0.88,
):
    return {
        "validity": validity,
        "uniqueness": uniqueness,
        "scaffold_diversity": scaffold_diversity,
        "mean_pic50": mean_pic50,
        "admet_pass_rate": admet_pass_rate,
        "in_domain_rate": in_domain_rate,
    }


@pytest.mark.unit
class TestVerdictLogic:

    def test_diversity_collapse_is_reward_hacking(self):
        pre = _eval_dict(scaffold_diversity=0.58)
        post = _eval_dict(scaffold_diversity=0.32, mean_pic50=7.6, admet_pass_rate=0.80)
        res = compare_pre_post(pre, post)
        assert res["verdict"] == "REWARD_HACKING"

    def test_uniqueness_collapse_is_reward_hacking(self):
        # diversity held (drop < 0.10) but uniqueness fell below the 0.60 floor
        pre = _eval_dict(uniqueness=0.80, scaffold_diversity=0.58)
        post = _eval_dict(
            uniqueness=0.30,
            scaffold_diversity=0.52,
            mean_pic50=7.6,
            admet_pass_rate=0.80,
        )
        res = compare_pre_post(pre, post)
        assert res["verdict"] == "REWARD_HACKING"
        assert "uniqueness" in res["detail"].lower()

    def test_success_when_improved_and_diversity_held(self):
        pre = _eval_dict(
            uniqueness=0.80,
            scaffold_diversity=0.58,
            mean_pic50=6.7,
            admet_pass_rate=0.31,
        )
        post = _eval_dict(
            uniqueness=0.70,
            scaffold_diversity=0.55,
            mean_pic50=7.2,
            admet_pass_rate=0.45,
        )
        res = compare_pre_post(pre, post)
        assert res["verdict"] == "SUCCESS"

    def test_partial_when_admet_collapses(self):
        pre = _eval_dict(
            uniqueness=0.80,
            scaffold_diversity=0.58,
            mean_pic50=6.7,
            admet_pass_rate=0.50,
        )
        post = _eval_dict(
            uniqueness=0.70,
            scaffold_diversity=0.55,
            mean_pic50=7.2,
            admet_pass_rate=0.30,
        )  # -20pp
        res = compare_pre_post(pre, post)
        assert res["verdict"] == "PARTIAL"

    def test_inconclusive_when_pic50_flat(self):
        pre = _eval_dict(mean_pic50=6.7)
        post = _eval_dict(mean_pic50=6.72, uniqueness=0.75, scaffold_diversity=0.56)
        res = compare_pre_post(pre, post)
        assert res["verdict"] == "INCONCLUSIVE"

    def test_table_is_ascii(self):
        # Regression: the table must not contain non-ASCII arrows (Windows cp1252)
        pre = _eval_dict()
        post = _eval_dict(mean_pic50=7.6, admet_pass_rate=0.80, uniqueness=0.70)
        res = compare_pre_post(pre, post)
        res["table"].encode("cp1252")  # raises if any char is non-cp1252-encodable
