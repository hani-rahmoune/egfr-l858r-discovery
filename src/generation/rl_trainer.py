"""
REINVENT-style policy gradient for GRU SMILES generator fine-tuning.

Algorithm (Olivecrona et al. 2017, "Molecular de-novo design through deep
reinforcement learning"):

  For each training step:
    1. Generate SMILES from the agent (stochastic sampling, no grad)
    2. Score each SMILES with the multi-objective reward function
    3. Compute per-token NLL of those SMILES under the agent (teacher forcing, WITH grad)
    4. Compute per-token NLL of those SMILES under the frozen prior (no grad)
    5. AugNLL = NLL_prior - sigma * Score
    6. Loss = mean( (NLL_agent - AugNLL)^2 )
    7. Backprop through NLL_agent only; prior is frozen

The sigma hyperparameter balances reward influence vs prior regularisation.
Higher sigma = more drift from prior; lower sigma = tighter regularisation.

Public API
----------
compute_nll_batch(model, tokenizer, smiles_list, device) -> Tensor (n,)
evaluate_generator(model, tokenizer, train_smiles, ad, backbone_model, ...) -> dict
compare_pre_post(pre, post) -> dict with "table" and "verdict"
REINVENTTrainer(agent, prior, tokenizer, reward_fn, optimizer, sigma, device)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem

from src.admet.filters import evaluate_admet
from src.generation.model import GRUGenerator
from src.generation.reward import _predict_activity_batch
from src.generation.sampler import evaluate_metrics
from src.generation.tokenizer import SMILESTokenizer
from src.scoring.applicability_domain import ApplicabilityDomain
from src.splitting.scaffold_split import get_bemis_murcko_scaffold
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ── Scaffold-memory diversity filter ──────────────────────────────────────────


class ScaffoldMemory:
    """
    REINVENT-style scaffold-memory diversity filter (Blaschke et al. 2020,
    "Memory-assisted reinforcement learning for diverse molecular de novo design").

    Maintains a running count of high-reward Bemis-Murcko scaffolds across the
    whole training run. Once a scaffold's bucket is full (>= bucket_size molecules
    that scored above min_score), every subsequent molecule sharing that scaffold
    has its reward overwritten with `penalty` (default 0.0).

    This is the missing guard that the flat novelty_bonus could not provide: it
    forces the agent to keep discovering NEW high-reward scaffolds instead of
    re-generating one collapsed mode. Unlike the train-set novelty check, the
    memory grows from the agent's OWN output during this run.

    Parameters
    ----------
    bucket_size : int   — molecules per scaffold before saturation (REINVENT default 25)
    min_score   : float — only count molecules whose raw reward exceeds this
                          (so junk does not fill buckets)
    penalty     : float — reward assigned to molecules from a saturated scaffold
    """

    def __init__(
        self,
        bucket_size: int = 25,
        min_score: float = 0.30,
        penalty: float = 0.0,
    ) -> None:
        self.bucket_size = bucket_size
        self.min_score = min_score
        self.penalty = penalty
        self.counts: dict[str, int] = {}
        self.n_penalized = 0  # cumulative molecules zeroed (telemetry)

    def apply(self, smiles_list: list[str], rewards: np.ndarray) -> np.ndarray:
        """
        Penalise rewards for molecules from saturated scaffolds and update counts.

        Returns a NEW array (does not mutate the input). Invalid SMILES and
        molecules scoring at/below min_score pass through unchanged and do not
        increment any bucket.
        """
        adjusted = rewards.astype(np.float32).copy()
        for i, smi in enumerate(smiles_list):
            if rewards[i] < self.min_score:
                continue
            scaf = get_bemis_murcko_scaffold(smi)
            if scaf is None:
                continue
            count = self.counts.get(scaf, 0)
            if count >= self.bucket_size:
                adjusted[i] = self.penalty
                self.n_penalized += 1
            self.counts[scaf] = count + 1
        return adjusted

    @property
    def n_scaffolds(self) -> int:
        return len(self.counts)

    @property
    def n_saturated(self) -> int:
        return sum(1 for c in self.counts.values() if c >= self.bucket_size)


# ── Batch SMILES generation ───────────────────────────────────────────────────


def _sample_batch(
    model: GRUGenerator,
    tokenizer: SMILESTokenizer,
    n: int,
    max_len: int = 120,
    temperature: float = 1.0,
    device: torch.device | None = None,
) -> list[str]:
    """
    Generate n SMILES in parallel using batched GRU inference.

    All n molecules are processed simultaneously at each token step, giving
    ~10-30× speedup over the sequential sampler for large n on CPU.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        seqs: list[list[int]] = [[] for _ in range(n)]
        done = [False] * n
        current = torch.full((n, 1), tokenizer.sos_idx, dtype=torch.long, device=device)
        hidden = model.init_hidden(n, device)

        for _ in range(max_len):
            logits, hidden = model(current, hidden)  # (n, 1, vocab)
            scaled = logits[:, 0, :] / max(temperature, 1e-6)
            probs = torch.softmax(scaled, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)  # (n, 1)

            current = next_tok  # feed as next input (even for finished positions)
            for i, tok in enumerate(next_tok.squeeze(1).tolist()):
                if not done[i]:
                    if tok == tokenizer.eos_idx:
                        done[i] = True
                    else:
                        seqs[i].append(tok)

            if all(done):
                break

    return [tokenizer.decode(s, strip_special=True) for s in seqs]


# ── Per-token NLL (teacher forcing) ──────────────────────────────────────────


def compute_nll_batch(
    model: GRUGenerator,
    tokenizer: SMILESTokenizer,
    smiles_list: list[str],
    device: torch.device,
    max_len: int = 120,
) -> torch.Tensor:
    """
    Compute per-token mean NLL for a batch of SMILES (teacher forcing).

    Sequences are encoded with SOS/EOS and padded; PAD positions are masked.
    For invalid / empty SMILES the encoder still produces a minimal sequence
    [SOS, EOS] giving one token of CE loss.

    Call this inside torch.no_grad() when computing the prior NLL.
    For agent NLL (needs gradient), call with gradients enabled.

    Returns
    -------
    Tensor of shape (batch,) — per-token mean NLL, all values ≥ 0.
    """
    pad_idx = tokenizer.pad_idx

    # ── Encode ────────────────────────────────────────────────────────────────
    encoded: list[list[int]] = []
    for smi in smiles_list:
        try:
            ids = tokenizer.encode(smi, add_sos=True, add_eos=True)
        except Exception:
            ids = [tokenizer.sos_idx, tokenizer.eos_idx]
        if len(ids) > max_len + 1:
            ids = ids[: max_len + 1]
        encoded.append(ids)

    # ── Pad ───────────────────────────────────────────────────────────────────
    max_len_batch = max(len(e) for e in encoded)
    padded = [e + [pad_idx] * (max_len_batch - len(e)) for e in encoded]
    t = torch.tensor(padded, dtype=torch.long, device=device)

    x = t[:, :-1]  # (batch, seq)  — inputs:  SOS … t_{n-1}
    y = t[:, 1:]  # (batch, seq)  — targets: t_1 … EOS

    # ── Forward ───────────────────────────────────────────────────────────────
    hidden = model.init_hidden(len(smiles_list), device)
    logits, _ = model(x, hidden)  # (batch, seq, vocab)

    log_probs = F.log_softmax(logits, dim=-1)  # (batch, seq, vocab)
    nll_tok = -log_probs.gather(2, y.unsqueeze(2)).squeeze(2)  # (batch, seq)

    # ── Mask PAD, average over valid tokens ───────────────────────────────────
    mask = (y != pad_idx).float()
    n_tok = mask.sum(dim=1).clamp(min=1.0)
    nll_pm = (nll_tok * mask).sum(dim=1) / n_tok  # (batch,)

    return nll_pm


# ── Generator quality evaluation ──────────────────────────────────────────────


def evaluate_generator(
    model: GRUGenerator,
    tokenizer: SMILESTokenizer,
    train_smiles: set[str],
    ad: ApplicabilityDomain,
    backbone_model: Any,
    n: int = 512,
    temperature: float = 0.8,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """
    Sample n molecules and compute all six comparison metrics:
      validity, uniqueness, scaffold_diversity, mean_pic50,
      admet_pass_rate, in_domain_rate.

    This is used for pre-RL and post-RL comparison.
    """
    if isinstance(device, str):
        device = torch.device(device)

    generated = _sample_batch(
        model, tokenizer, n=n, temperature=temperature, device=device
    )
    metrics = evaluate_metrics(generated, train_smiles=train_smiles)

    # ── Filter: valid + unique + novel ────────────────────────────────────────
    seen: set[str] = set()
    candidates: list[str] = []
    for smi in generated:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        can = Chem.MolToSmiles(mol)
        if can in seen:
            continue
        seen.add(can)
        if can not in train_smiles:
            candidates.append(can)

    if not candidates:
        return {
            **metrics,
            "mean_pic50": None,
            "admet_pass_rate": None,
            "in_domain_rate": None,
            "n_candidates": 0,
        }

    # ── Backbone pIC50 (batched) ──────────────────────────────────────────────
    preds = _predict_activity_batch(candidates, backbone_model)
    valid_pred = [p for p in preds if p is not None]
    mean_pic50 = round(float(np.mean(valid_pred)), 3) if valid_pred else None

    # ── ADMET ─────────────────────────────────────────────────────────────────
    admet_res = [evaluate_admet(s) for s in candidates]
    n_admet = sum(1 for r in admet_res if r["admet_status"] == "pass")
    admet_rate = round(n_admet / len(candidates), 3)

    # ── AD ────────────────────────────────────────────────────────────────────
    ad_res = ad.predict_batch(candidates)
    n_in = sum(1 for r in ad_res if r["domain"] == "in_domain")
    in_domain = round(n_in / len(candidates), 3)

    return {
        **metrics,
        "mean_pic50": mean_pic50,
        "admet_pass_rate": admet_rate,
        "in_domain_rate": in_domain,
        "n_candidates": len(candidates),
    }


# ── Pre-vs-post comparison ────────────────────────────────────────────────────


def compare_pre_post(pre: dict, post: dict) -> dict[str, Any]:
    """
    Compare pre-RL and post-RL evaluation dicts and return a verdict.

    Verdict rules (applied in priority order):
      REWARD_HACKING : scaffold_diversity drops > 0.10  OR  uniqueness drops below
                       0.60  OR  in_domain drops > 10 pp   (any diversity collapse)
      SUCCESS        : mean_pic50 improves > 0.10 AND ADMET improves AND
                       uniqueness stays >= 0.60 AND scaffold_diversity drop < 0.10
      PARTIAL        : pIC50 improves but ADMET collapses > 15 pp (no div collapse)
      INCONCLUSIVE   : pIC50 change < 0.10 — RL did not move the needle

    The uniqueness floor and diversity-drop ceiling are the explicit anti-collapse
    guards the scaffold-memory filter is meant to satisfy.

    Also returns a formatted comparison table string.
    """
    metrics = [
        ("validity", "Validity", ".1%"),
        ("uniqueness", "Uniqueness", ".1%"),
        ("scaffold_diversity", "Scaffold diversity", ".3f"),
        ("mean_pic50", "Mean pred pIC50", ".3f"),
        ("admet_pass_rate", "ADMET pass rate", ".1%"),
        ("in_domain_rate", "In-domain rate", ".1%"),
    ]

    rows = []
    for key, label, fmt in metrics:
        pre_v = pre.get(key)
        post_v = post.get(key)
        if pre_v is None or post_v is None:
            delta_s = "  n/a"
            arrow = " "
        else:
            delta = post_v - pre_v
            delta_s = f"{delta:+.3f}"
            arrow = "^" if delta > 0.005 else ("v" if delta < -0.005 else "-")
        pre_s = format(pre_v, fmt) if pre_v is not None else "  n/a"
        post_s = format(post_v, fmt) if post_v is not None else "  n/a"
        rows.append(f"  {label:<22} {pre_s:>8}  {post_s:>8}  {delta_s:>8}  {arrow}")

    table = (
        "\n  Metric                    Pre-RL   Post-RL    Delta  Dir\n"
        "  " + "-" * 58 + "\n" + "\n".join(rows)
    )

    # ── Verdict ───────────────────────────────────────────────────────────────
    in_domain_pre = pre.get("in_domain_rate")
    in_domain_post = post.get("in_domain_rate")
    div_pre = pre.get("scaffold_diversity")
    div_post = post.get("scaffold_diversity")
    uniq_post = post.get("uniqueness")
    pic50_pre = pre.get("mean_pic50")
    pic50_post = post.get("mean_pic50")
    admet_pre = pre.get("admet_pass_rate")
    admet_post = post.get("admet_pass_rate")

    ood_hack = (
        in_domain_pre is not None
        and in_domain_post is not None
        and (in_domain_post - in_domain_pre) < -0.10
    )
    div_collapse = (
        div_pre is not None and div_post is not None and (div_post - div_pre) < -0.10
    )
    uniq_collapse = uniq_post is not None and uniq_post < 0.60
    improved = (
        pic50_pre is not None
        and pic50_post is not None
        and (pic50_post - pic50_pre) > 0.10
    )
    admet_improved = (
        admet_pre is not None
        and admet_post is not None
        and (admet_post - admet_pre) > 0.0
    )
    admet_collapse = (
        admet_pre is not None
        and admet_post is not None
        and (admet_post - admet_pre) < -0.15
    )

    if div_collapse or uniq_collapse or ood_hack:
        verdict = "REWARD_HACKING"
        if div_collapse:
            detail = (
                f"scaffold diversity collapsed {div_post - div_pre:+.3f} (> 0.10 drop)"
            )
        elif uniq_collapse:
            detail = f"uniqueness collapsed to {uniq_post:.1%} (< 60% floor)"
        else:
            detail = "in-domain rate dropped > 10 pp"
    elif improved and admet_improved:
        verdict = "SUCCESS"
        detail = (
            f"mean pIC50 +{pic50_post - pic50_pre:.3f} and ADMET "
            f"+{(admet_post - admet_pre):.1%} with diversity/uniqueness intact"
        )
    elif improved and admet_collapse:
        verdict = "PARTIAL"
        detail = "pIC50 improved but ADMET pass rate dropped > 15 pp"
    elif improved:
        verdict = "PARTIAL"
        detail = "pIC50 improved, diversity held, but ADMET did not improve"
    else:
        verdict = "INCONCLUSIVE"
        detail = "pIC50 change < 0.10 — RL did not move the needle"

    return {
        "table": table,
        "verdict": verdict,
        "detail": detail,
    }


# ── REINVENT trainer ──────────────────────────────────────────────────────────


class REINVENTTrainer:
    """
    REINVENT augmented-log-likelihood policy gradient trainer.

    The agent is updated; the prior is frozen (loaded from the same checkpoint
    as the initial agent and never modified).

    Parameters
    ----------
    agent      : GRUGenerator — policy to train (warm-started from fine-tune ckpt)
    prior      : GRUGenerator — frozen reference policy (same initial weights as agent)
    tokenizer  : SMILESTokenizer
    reward_fn  : callable(list[str]) -> np.ndarray
    optimizer  : torch.optim.Optimizer bound to agent.parameters()
    sigma      : REINVENT regularisation weight (0.5 = balanced; higher = more reward)
    device     : torch device
    max_len    : max token length for generation and NLL computation
    temperature: sampling temperature (1.0 = full stochasticity)
    scaffold_memory : optional ScaffoldMemory — penalises rewards for saturated
                      scaffolds so the agent cannot collapse onto one mode.
                      None disables the diversity filter.
    """

    def __init__(
        self,
        agent: GRUGenerator,
        prior: GRUGenerator,
        tokenizer: SMILESTokenizer,
        reward_fn: Any,  # callable(list[str]) -> np.ndarray
        optimizer: torch.optim.Optimizer,
        sigma: float = 0.5,
        device: torch.device | str = "cpu",
        max_len: int = 120,
        temperature: float = 1.0,
        scaffold_memory: ScaffoldMemory | None = None,
    ) -> None:
        if isinstance(device, str):
            device = torch.device(device)

        self.agent = agent
        self.prior = prior
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.optimizer = optimizer
        self.sigma = sigma
        self.device = device
        self.max_len = max_len
        self.temperature = temperature
        self.scaffold_memory = scaffold_memory

        # Ensure prior is frozen
        for p in self.prior.parameters():
            p.requires_grad_(False)
        self.prior.eval()

    def train_step(self, batch_size: int = 64) -> dict[str, float]:
        """
        One REINVENT training step.

        Returns dict with: loss, mean_reward, mean_nll_agent, mean_nll_prior,
        validity (fraction of parseable SMILES in batch).
        """
        # ── 1. Generate from agent (no grad) ──────────────────────────────────
        smiles = _sample_batch(
            self.agent,
            self.tokenizer,
            n=batch_size,
            max_len=self.max_len,
            temperature=self.temperature,
            device=self.device,
        )

        # ── 2. Score ──────────────────────────────────────────────────────────
        raw_scores = self.reward_fn(smiles)  # np.ndarray

        # ── 2b. Scaffold-memory diversity filter ──────────────────────────────
        # Zero the reward of molecules whose Murcko scaffold bucket is saturated,
        # forcing the agent to keep finding NEW high-reward scaffolds.
        if self.scaffold_memory is not None:
            scores = self.scaffold_memory.apply(smiles, raw_scores)
        else:
            scores = raw_scores
        scores_t = torch.tensor(scores, dtype=torch.float32, device=self.device)

        validity = sum(1 for s in smiles if Chem.MolFromSmiles(s) is not None) / len(
            smiles
        )

        # ── 3. NLL under agent (with grad) ────────────────────────────────────
        self.agent.train()
        nll_agent = compute_nll_batch(
            self.agent, self.tokenizer, smiles, self.device, self.max_len
        )  # (batch,)

        # ── 4. NLL under prior (no grad) ──────────────────────────────────────
        with torch.no_grad():
            nll_prior = compute_nll_batch(
                self.prior, self.tokenizer, smiles, self.device, self.max_len
            )  # (batch,)

        # ── 5. REINVENT loss ──────────────────────────────────────────────────
        # AugNLL = NLL_prior - σ * Score
        # Loss   = mean( (NLL_agent - AugNLL)² )
        aug_nll = nll_prior - self.sigma * scores_t
        loss = ((nll_agent - aug_nll) ** 2).mean()

        # ── 6. Backprop ───────────────────────────────────────────────────────
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
        self.optimizer.step()

        n_penalized = int((scores != raw_scores).sum())

        return {
            "loss": loss.item(),
            "mean_reward": float(scores.mean()),  # post-filter (used in loss)
            "mean_reward_raw": float(raw_scores.mean()),  # pre-filter
            "n_penalized": n_penalized,  # molecules zeroed this step
            "mean_nll_agent": float(nll_agent.mean().item()),
            "mean_nll_prior": float(nll_prior.mean().item()),
            "validity": validity,
        }
