"""
Applicability domain (AD) assessment via maximum Tanimoto similarity to training set.

Domain bands (from model_config.yaml > applicability_domain):
  in_domain_threshold  0.50 -> "in_domain"    confidence_factor 1.00
  borderline_threshold 0.30 -> "borderline"   confidence_factor 0.75
  below borderline          -> "out_of_domain" confidence_factor 0.50

Fingerprint: Morgan ECFP4 radius=2, 2048 bits (same as production models).

Public API:
  ApplicabilityDomain.from_config()            -- build with thresholds from config
  ApplicabilityDomain.fit(smiles_list)         -- store training fingerprints
  ApplicabilityDomain.predict(smiles)          -- single-molecule AD check
  ApplicabilityDomain.predict_batch(smiles_list) -- batch AD check
  max_tanimoto_to_set(smiles, train_fps)       -- stateless helper
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem

from src.features.fingerprints import morgan_fingerprint
from src.utils.config import load_model_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_IN_DOMAIN = 0.50
_DEFAULT_BORDERLINE = 0.30
_DEFAULT_CONFIDENCE: dict[str, float] = {
    "in_domain": 1.0,
    "borderline": 0.75,
    "out_of_domain": 0.50,
}


def _tanimoto_max(query_fp: np.ndarray, train_fps: np.ndarray) -> float:
    """
    Maximum Tanimoto similarity between one binary query FP and a matrix of training FPs.

    All inputs must be binary (0/1) arrays.  Converts to float32 internally to
    avoid uint8 overflow when n_bits=2048.
    """
    if len(train_fps) == 0:
        return 0.0
    q = query_fp.astype(np.float32)
    T = train_fps.astype(np.float32)
    intersect = T @ q  # (n_train,)
    q_bits = float(q.sum())
    t_bits = T.sum(axis=1)  # (n_train,)
    union = t_bits + q_bits - intersect
    sims = np.where(union > 0, intersect / union, 0.0)
    return float(sims.max())


def _domain_label(
    sim: float,
    in_domain_threshold: float,
    borderline_threshold: float,
) -> str:
    if sim >= in_domain_threshold:
        return "in_domain"
    if sim >= borderline_threshold:
        return "borderline"
    return "out_of_domain"


def max_tanimoto_to_set(smiles: str, train_fps: np.ndarray) -> float | None:
    """
    Stateless helper: compute max Tanimoto of a SMILES against a pre-built FP matrix.
    Returns None if SMILES is invalid.
    """
    fp = morgan_fingerprint(smiles)
    if fp is None:
        return None
    return _tanimoto_max(fp.astype(np.uint8), train_fps)


class ApplicabilityDomain:
    """
    Tanimoto-similarity applicability domain checker.

    Fit once on training SMILES, then call predict() or predict_batch()
    on any query molecules.
    """

    def __init__(
        self,
        in_domain_threshold: float = _DEFAULT_IN_DOMAIN,
        borderline_threshold: float = _DEFAULT_BORDERLINE,
        confidence_factors: dict[str, float] | None = None,
    ) -> None:
        self.in_domain_threshold = in_domain_threshold
        self.borderline_threshold = borderline_threshold
        self.confidence_factors: dict[str, float] = (
            confidence_factors
            if confidence_factors is not None
            else dict(_DEFAULT_CONFIDENCE)
        )
        self._train_fps: np.ndarray | None = None
        self._n_train: int = 0

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls) -> "ApplicabilityDomain":
        """Build an ApplicabilityDomain using thresholds from model_config.yaml."""
        cfg = load_model_config()
        ad_cfg = cfg.get("applicability_domain", {})
        return cls(
            in_domain_threshold=ad_cfg.get("in_domain_threshold", _DEFAULT_IN_DOMAIN),
            borderline_threshold=ad_cfg.get(
                "borderline_threshold", _DEFAULT_BORDERLINE
            ),
            confidence_factors=ad_cfg.get(
                "confidence_factors", dict(_DEFAULT_CONFIDENCE)
            ),
        )

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, smiles_list: list[str]) -> "ApplicabilityDomain":
        """Compute and cache Morgan ECFP4 fingerprints for training molecules."""
        fps: list[np.ndarray] = []
        n_invalid = 0
        for smi in smiles_list:
            fp = morgan_fingerprint(smi)
            if fp is not None:
                fps.append(fp.astype(np.uint8))
            else:
                n_invalid += 1
        if n_invalid:
            logger.warning(
                f"ApplicabilityDomain.fit: {n_invalid} invalid SMILES skipped"
            )
        self._train_fps = (
            np.array(fps, dtype=np.uint8)
            if fps
            else np.empty((0, 2048), dtype=np.uint8)
        )
        self._n_train = len(fps)
        logger.info(f"ApplicabilityDomain fitted on {self._n_train} molecules")
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, smiles: str) -> dict:
        """
        Assess one molecule's applicability domain.

        Returns:
          smiles           -- input SMILES
          valid            -- True if RDKit could parse the molecule
          max_tanimoto     -- highest Tanimoto similarity to any training molecule
          domain           -- "in_domain" | "borderline" | "out_of_domain"
          confidence_factor -- confidence scaling factor for predictions
        """
        if self._train_fps is None:
            raise RuntimeError("Call fit() before predict()")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                "smiles": smiles,
                "valid": False,
                "max_tanimoto": None,
                "domain": "out_of_domain",
                "confidence_factor": self.confidence_factors["out_of_domain"],
            }

        if self._n_train == 0:
            return {
                "smiles": smiles,
                "valid": True,
                "max_tanimoto": 0.0,
                "domain": "out_of_domain",
                "confidence_factor": self.confidence_factors["out_of_domain"],
            }

        fp = morgan_fingerprint(smiles)
        if fp is None:
            return {
                "smiles": smiles,
                "valid": True,
                "max_tanimoto": 0.0,
                "domain": "out_of_domain",
                "confidence_factor": self.confidence_factors["out_of_domain"],
            }

        sim = _tanimoto_max(fp.astype(np.uint8), self._train_fps)
        domain = _domain_label(sim, self.in_domain_threshold, self.borderline_threshold)
        return {
            "smiles": smiles,
            "valid": True,
            "max_tanimoto": round(sim, 4),
            "domain": domain,
            "confidence_factor": self.confidence_factors[domain],
        }

    def predict_batch(self, smiles_list: list[str]) -> list[dict]:
        """Batch version of predict(). Result order matches input order."""
        return [self.predict(smi) for smi in smiles_list]

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def n_train(self) -> int:
        """Number of training molecules stored after fit()."""
        return self._n_train

    @property
    def train_fps(self) -> np.ndarray | None:
        """Raw training fingerprint matrix (n_train, 2048) uint8, or None before fit."""
        return self._train_fps
