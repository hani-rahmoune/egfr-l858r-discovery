"""
Multi-objective reward function for REINVENT-style RL fine-tuning.

Reward formula (all components are additive unless noted):
  + activity_weight * sigmoid(pred_pIC50 - activity_center) * ad_confidence_factor
  + qed_weight * QED
  + admet_bonus        if ADMET status == "pass"
  + novelty_bonus      if not in training SMILES set
  - out_of_domain_penalty  if AD domain == "out_of_domain"
  - borderline_penalty     if AD domain == "borderline"
  - covalent_penalty   if electrophilic warhead detected
  - range_penalty      if MW or LogP outside drug-like range
  (invalid SMILES returns invalid_penalty directly; all other components skipped)

The AD confidence_factor is applied as a multiplicative scale on the activity
term only — this is the primary anti-reward-hacking guard: backbone predictions
for OOD molecules are discounted before contributing to the reward.

Default weights are sensible starting points; override via config > rl > reward.

Public API
----------
compute_reward(smiles, train_smiles, ad, backbone_model, cfg) -> float
MoleculeReward                   callable on list[str] -> np.ndarray
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from src.admet.filters import evaluate_admet
from src.features.covalent import is_covalent
from src.features.descriptors import DESCRIPTOR_NAMES, compute_descriptors
from src.features.fingerprints import morgan_fingerprint
from src.scoring.applicability_domain import ApplicabilityDomain
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ── Default reward weights ─────────────────────────────────────────────────────

_DEFAULTS: dict[str, float] = {
    "activity_weight": 0.50,  # weight on sigmoid(pIC50 - center)
    "activity_center": 7.00,  # pIC50 sigmoid inflection (≈ 10 nM IC50)
    "qed_weight": 0.20,  # weight on QED score [0,1]
    "admet_bonus": 0.20,  # flat bonus when ADMET passes
    "novelty_bonus": 0.10,  # flat bonus for molecules not in train set
    "invalid_penalty": -1.00,  # returned immediately for unparseable SMILES
    "covalent_penalty": -0.30,  # warhead detected (non-covalent docking unreliable)
    "out_of_domain_penalty": -0.30,  # AD says out_of_domain (backbone prediction unsafe)
    "borderline_penalty": -0.10,  # AD says borderline
    "range_penalty": -0.10,  # extreme MW (<100 or >600) or LogP (<-2 or >5.5)
}


def _cfg(cfg: dict[str, Any] | None, key: str) -> float:
    return float((cfg or {}).get(key, _DEFAULTS[key]))


def _sigmoid(x: float, center: float = 7.0, scale: float = 1.0) -> float:
    """Logistic sigmoid centered at `center` with steepness `scale`."""
    return 1.0 / (1.0 + math.exp(-scale * (x - center)))


# ── Batch backbone prediction helper ─────────────────────────────────────────


def _predict_activity_batch(
    smiles_list: list[str],
    backbone_model: Any,
) -> list[float | None]:
    """
    Predict pIC50 for a list of canonical SMILES using the backbone model.

    Batches all feature computations and calls backbone_model.predict once,
    which is important for RF/XGB speed (parallelised internally).
    Returns None for molecules where feature computation fails.
    """
    features: list[np.ndarray] = []
    valid_indices: list[int] = []

    for i, smi in enumerate(smiles_list):
        fp = morgan_fingerprint(smi)
        if fp is None:
            continue
        desc = compute_descriptors(smi)
        if desc is None:
            continue
        desc_vec = np.array([desc[k] for k in DESCRIPTOR_NAMES], dtype=np.float32)
        features.append(np.concatenate([fp.astype(np.float32), desc_vec]))
        valid_indices.append(i)

    preds: list[float | None] = [None] * len(smiles_list)
    if features:
        batch_preds = backbone_model.predict(np.array(features))
        for j, idx in enumerate(valid_indices):
            preds[idx] = float(batch_preds[j])

    return preds


# ── Single-molecule reward ────────────────────────────────────────────────────


def compute_reward(
    smiles: str,
    train_smiles: set[str],
    ad: ApplicabilityDomain,
    backbone_model: Any,
    cfg: dict[str, Any] | None = None,
) -> float:
    """
    Compute the RL reward for one SMILES string.

    Parameters
    ----------
    smiles         : generated SMILES (may be invalid)
    train_smiles   : canonical SMILES of the reference training set
                     (used for novelty check)
    ad             : fitted ApplicabilityDomain instance
    backbone_model : QSARTrainer with .predict(X) → pIC50 array
    cfg            : reward weight overrides (keys from _DEFAULTS); None = use defaults

    Returns
    -------
    float reward (bounded roughly in [-1.0, 1.0])
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _cfg(cfg, "invalid_penalty")

    can = Chem.MolToSmiles(mol)
    reward = 0.0

    # ── Covalent warhead ──────────────────────────────────────────────────────
    if is_covalent(can):
        reward += _cfg(cfg, "covalent_penalty")

    # ── Applicability domain ─────────────────────────────────────────────────
    ad_result = ad.predict(can)
    domain = ad_result["domain"]
    cf = ad_result["confidence_factor"]

    if domain == "out_of_domain":
        reward += _cfg(cfg, "out_of_domain_penalty")
    elif domain == "borderline":
        reward += _cfg(cfg, "borderline_penalty")

    # ── Backbone activity (scaled by AD confidence) ──────────────────────────
    fp = morgan_fingerprint(can)
    if fp is not None:
        desc = compute_descriptors(can)
        if desc is not None:
            desc_vec = np.array([desc[k] for k in DESCRIPTOR_NAMES], dtype=np.float32)
            feat = np.concatenate([fp.astype(np.float32), desc_vec])
            try:
                pred_pic50 = float(backbone_model.predict([feat])[0])
                raw_act = _sigmoid(pred_pic50, center=_cfg(cfg, "activity_center"))
                reward += _cfg(cfg, "activity_weight") * raw_act * cf
            except Exception:
                pass

    # ── ADMET / QED ───────────────────────────────────────────────────────────
    admet = evaluate_admet(can)
    qed = admet.get("qed") or 0.0
    reward += _cfg(cfg, "qed_weight") * qed
    if admet.get("admet_status") == "pass":
        reward += _cfg(cfg, "admet_bonus")

    # ── Novelty ───────────────────────────────────────────────────────────────
    if can not in train_smiles:
        reward += _cfg(cfg, "novelty_bonus")

    # ── Extreme property range ────────────────────────────────────────────────
    try:
        mw = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        if mw > 600 or mw < 100 or logp > 5.5 or logp < -2.0:
            reward += _cfg(cfg, "range_penalty")
    except Exception:
        pass

    return reward


# ── Batched reward (used during RL training) ──────────────────────────────────


class MoleculeReward:
    """
    Callable reward function that batches backbone predictions for efficiency.

    Usage::

        reward_fn = MoleculeReward(backbone_model, ad, train_smiles, cfg)
        scores = reward_fn(smiles_list)   # np.ndarray, shape (n,)
    """

    def __init__(
        self,
        backbone_model: Any,
        ad: ApplicabilityDomain,
        train_smiles: set[str],
        cfg: dict[str, Any] | None = None,
    ) -> None:
        self.backbone_model = backbone_model
        self.ad = ad
        self.train_smiles = train_smiles
        self.cfg = cfg or {}

    def __call__(self, smiles_list: list[str]) -> np.ndarray:
        """
        Compute reward for each SMILES in the batch.

        Backbone predictions are batched (one sklearn call per batch).
        All other components are per-molecule Python loops (fast).

        Returns np.ndarray of shape (len(smiles_list),).
        """
        n = len(smiles_list)

        # ── Validity + canonical form ─────────────────────────────────────────
        mols: list[Chem.Mol | None] = []
        cans: list[str] = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            mols.append(mol)
            cans.append(Chem.MolToSmiles(mol) if mol is not None else "")

        # ── Batch backbone prediction (valid molecules only) ──────────────────
        valid_cans = [cans[i] for i in range(n) if mols[i] is not None]
        act_preds = (
            _predict_activity_batch(valid_cans, self.backbone_model)
            if valid_cans
            else []
        )
        pred_map: dict[int, float | None] = {}
        vi = 0
        for i in range(n):
            if mols[i] is not None:
                pred_map[i] = act_preds[vi]
                vi += 1

        # ── Batch AD prediction (valid molecules only) ────────────────────────
        vi = 0
        ad_batch = self.ad.predict_batch(valid_cans) if valid_cans else []
        ad_map: dict[int, dict] = {}
        vi = 0
        for i in range(n):
            if mols[i] is not None:
                ad_map[i] = ad_batch[vi]
                vi += 1

        # ── Per-molecule reward assembly ──────────────────────────────────────
        rewards = np.zeros(n, dtype=np.float32)
        for i in range(n):
            if mols[i] is None:
                rewards[i] = _cfg(self.cfg, "invalid_penalty")
                continue

            r = 0.0
            can = cans[i]
            mol = mols[i]
            ad_r = ad_map.get(i, {"domain": "out_of_domain", "confidence_factor": 0.5})

            # Covalent
            if is_covalent(can):
                r += _cfg(self.cfg, "covalent_penalty")

            # AD penalty + activity scaling
            domain = ad_r["domain"]
            cf = ad_r["confidence_factor"]
            if domain == "out_of_domain":
                r += _cfg(self.cfg, "out_of_domain_penalty")
            elif domain == "borderline":
                r += _cfg(self.cfg, "borderline_penalty")

            # Activity
            pred = pred_map.get(i)
            if pred is not None:
                raw_act = _sigmoid(pred, center=_cfg(self.cfg, "activity_center"))
                r += _cfg(self.cfg, "activity_weight") * raw_act * cf

            # QED / ADMET
            admet = evaluate_admet(can)
            qed = admet.get("qed") or 0.0
            r += _cfg(self.cfg, "qed_weight") * qed
            if admet.get("admet_status") == "pass":
                r += _cfg(self.cfg, "admet_bonus")

            # Novelty
            if can not in self.train_smiles:
                r += _cfg(self.cfg, "novelty_bonus")

            # Range
            try:
                mw = Descriptors.ExactMolWt(mol)
                logp = Descriptors.MolLogP(mol)
                if mw > 600 or mw < 100 or logp > 5.5 or logp < -2.0:
                    r += _cfg(self.cfg, "range_penalty")
            except Exception:
                pass

            rewards[i] = r

        return rewards
