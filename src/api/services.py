"""
Model registry + fast-screen scoring for the API (Phase 24).

`ModelRegistry` loads every precomputed artifact ONCE at application startup and
exposes a single `score(smiles)` method that runs the fast screen:

  backbone pIC50 (mutant) + WT-proxy pIC50  ->  ML selectivity proxy
  ADMET (QED / pass-flag / alerts)
  covalent warhead flag
  applicability domain (band + confidence_factor)
  warning strings

No training, no docking, no ESM-2 happen here — only forward passes over saved
sklearn/XGBoost models and RDKit descriptor computation. Docking-based
selectivity is intentionally absent (it needs the offline Vina pipeline).

The class is deliberately plain (no FastAPI imports) so it can be constructed
directly in unit tests, and so a mock can be substituted via FastAPI's
dependency override in endpoint tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from src.admet.filters import evaluate_admet
from src.features.covalent import detect_warheads
from src.features.descriptors import compute_descriptors_array
from src.features.fingerprints import morgan_fingerprint
from src.models.qsar import QSARTrainer
from src.scoring.applicability_domain import ApplicabilityDomain
from src.scoring.ranking import build_warnings
from src.utils.config import load_model_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_GENERAL_DIR = PROJECT_ROOT / "models" / "qsar" / "general"
_WTPROXY_DIR = PROJECT_ROOT / "models" / "qsar" / "wt_proxy"
_EGFR_CSV = PROJECT_ROOT / "data" / "interim" / "egfr_cleaned.csv"
_ERBB2_CSV = PROJECT_ROOT / "data" / "interim" / "erbb2_cleaned.csv"

SERVICE_VERSION = "1.0.0"


def _load_train_smiles() -> list[str]:
    frames = []
    for csv in (_EGFR_CSV, _ERBB2_CSV):
        if csv.exists():
            frames.append(pd.read_csv(csv, usecols=["canonical_smiles"]))
    if not frames:
        return []
    raw = pd.concat(frames)["canonical_smiles"].dropna().tolist()
    out: set[str] = set()
    for smi in raw:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            out.add(Chem.MolToSmiles(mol))
    return sorted(out)


class ModelRegistry:
    """Holds all loaded artifacts and runs the fast screen."""

    def __init__(
        self,
        backbone: QSARTrainer,
        wt_proxy: QSARTrainer,
        ad: ApplicabilityDomain,
        backbone_meta: dict,
        wt_proxy_meta: dict,
        n_train: int,
    ) -> None:
        self.backbone = backbone
        self.wt_proxy = wt_proxy
        self.ad = ad
        self.backbone_meta = backbone_meta
        self.wt_proxy_meta = wt_proxy_meta
        self.n_train = n_train

    # ── Loading ────────────────────────────────────────────────────────────────

    @classmethod
    def load(cls) -> "ModelRegistry":
        """Load every artifact from disk once. Raises FileNotFoundError if missing."""
        import json

        cfg = load_model_config()
        logger.info("Loading backbone (Model 1) ...")
        backbone = QSARTrainer.load(_GENERAL_DIR, cfg)
        logger.info("Loading WT-proxy (Model 2) ...")
        wt_proxy = QSARTrainer.load(_WTPROXY_DIR, cfg)

        backbone_meta = json.loads((_GENERAL_DIR / "metadata.json").read_text())
        wt_proxy_meta = json.loads((_WTPROXY_DIR / "metadata.json").read_text())

        logger.info("Fitting applicability domain on training SMILES ...")
        train_smiles = _load_train_smiles()
        ad = ApplicabilityDomain.from_config()
        ad.fit(train_smiles)

        logger.info(f"ModelRegistry ready (n_train={len(train_smiles)}).")
        return cls(
            backbone, wt_proxy, ad, backbone_meta, wt_proxy_meta, len(train_smiles)
        )

    # ── Feature vector ───────────────────────────────────────────────────────────

    @staticmethod
    def _features(smiles: str) -> np.ndarray | None:
        """Morgan ECFP4 (2048) + 11 RDKit descriptors = 2059, or None on failure."""
        fp = morgan_fingerprint(smiles, radius=2, n_bits=2048, use_chirality=True)
        if fp is None:
            return None
        desc = compute_descriptors_array(smiles)
        if desc is None:
            return None
        return np.concatenate([fp.astype(np.float32), desc])

    # ── Scoring ────────────────────────────────────────────────────────────────

    def score(self, smiles: str) -> dict[str, Any]:
        """
        Run the full fast screen for one SMILES.

        Returns a dict shaped for PredictResponse. For invalid SMILES it returns
        {valid: False, ...} with null scores and an "Invalid SMILES" warning —
        callers decide whether that becomes a 422 (single) or a row (batch).
        """
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return {
                "smiles": smiles,
                "canonical_smiles": None,
                "valid": False,
                "pic50_mutant": None,
                "pic50_wt": None,
                "selectivity_proxy": None,
                "covalent": False,
                "warheads": [],
                "admet": None,
                "applicability_domain": None,
                "docking_selectivity_available": False,
                "warnings": ["Invalid SMILES: RDKit could not parse the input."],
            }

        canonical = Chem.MolToSmiles(mol)

        # Activity (backbone) + WT-proxy
        pic50_mutant = pic50_wt = selectivity = None
        feats = self._features(canonical)
        if feats is not None:
            X = np.array([feats])
            try:
                pic50_mutant = round(float(self.backbone.predict(X)[0]), 3)
                pic50_wt = round(float(self.wt_proxy.predict(X)[0]), 3)
                selectivity = round(pic50_mutant - pic50_wt, 3)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    f"Activity prediction failed for {canonical[:50]}: {exc}"
                )

        # Covalent
        warheads = detect_warheads(canonical)
        is_covalent = bool(warheads)

        # ADMET
        admet = evaluate_admet(canonical)
        admet_obj = {
            "status": admet["admet_status"],
            "qed": admet.get("qed"),
            "sa_score": admet.get("sa_score"),
            "lipinski_pass": admet.get("lipinski_pass"),
            "veber_pass": admet.get("veber_pass"),
            "pains_alerts": admet.get("pains_alerts", []),
            "brenk_alerts": admet.get("brenk_alerts", []),
            "flag_reasons": admet.get("flag_reasons", []),
        }

        # Applicability domain
        ad_r = self.ad.predict(canonical)
        ad_obj = {
            "domain": ad_r["domain"],
            "max_tanimoto": ad_r["max_tanimoto"],
            "confidence_factor": ad_r["confidence_factor"],
        }

        # Warnings (no within-noise here: docking is not run at request time)
        warnings = build_warnings(
            domain=ad_r["domain"],
            is_covalent=is_covalent,
            warheads=warheads,
            selectivity_within_noise=None,
        )

        return {
            "smiles": smiles,
            "canonical_smiles": canonical,
            "valid": True,
            "pic50_mutant": pic50_mutant,
            "pic50_wt": pic50_wt,
            "selectivity_proxy": selectivity,
            "covalent": is_covalent,
            "warheads": warheads,
            "admet": admet_obj,
            "applicability_domain": ad_obj,
            "docking_selectivity_available": False,
            "warnings": warnings,
        }

    # ── Introspection ────────────────────────────────────────────────────────────

    def model_info(self) -> dict[str, Any]:
        """Static description of the served models + their meaning + caveats."""

        def _test(meta: dict) -> dict:
            t = meta.get("test_metrics", {})
            return {
                k: (round(v, 3) if isinstance(v, (int, float)) else v)
                for k, v in t.items()
            }

        return {
            "service": "egfr-l858r-fast-screen",
            "version": SERVICE_VERSION,
            "models": {
                "backbone_activity": {
                    "role": "Model 1 — EGFR general backbone; predicts pIC50 on the "
                    "mutant/general construct.",
                    "algorithm": self.backbone_meta.get("best_model"),
                    "n_features": len(self.backbone_meta.get("feature_cols", [])),
                    "feature_vector": "Morgan ECFP4 (2048 bits) + 11 RDKit descriptors",
                    "test_metrics": _test(self.backbone_meta),
                },
                "wt_proxy": {
                    "role": "Model 2 — WT-proxy comparator; predicts pIC50 on "
                    "wild-type/unspecified EGFR.",
                    "algorithm": self.wt_proxy_meta.get("best_model"),
                    "test_metrics": _test(self.wt_proxy_meta),
                },
                "admet": {
                    "role": "Approximate ADMET / drug-likeness filter "
                    "(Lipinski, Veber, PAINS, Brenk, QED, SA).",
                    "note": "Molecules are flagged, never hard-dropped.",
                },
                "covalent_detector": {
                    "role": "SMARTS-based electrophilic warhead detector.",
                    "warhead_types": [
                        "acrylamide",
                        "acrylate_ester",
                        "propiolamide",
                        "vinyl_sulfone",
                        "chloroacetamide",
                        "epoxide",
                        "michael_enone",
                        "isocyanate",
                        "cyanamide",
                    ],
                },
                "applicability_domain": {
                    "role": "Max ECFP4 Tanimoto to the training set; 3 bands.",
                    "bands": {
                        "in_domain": ">= 0.50 (confidence_factor 1.00)",
                        "borderline": "0.30-0.50 (confidence_factor 0.75)",
                        "out_of_domain": "< 0.30 (confidence_factor 0.50)",
                    },
                    "n_train": self.n_train,
                },
            },
            "score_definitions": {
                "pic50_mutant": "Backbone (Model 1) predicted pIC50 on the L858R/general "
                "EGFR construct. Higher = more potent.",
                "pic50_wt": "WT-proxy (Model 2) predicted pIC50 on wild-type EGFR.",
                "selectivity_proxy": "pic50_mutant - pic50_wt. Positive = mutant-selective. "
                "An ML proxy only — NOT the docking selectivity.",
                "covalent": "True if an electrophilic warhead SMARTS matched.",
                "admet": "QED + Lipinski/Veber/PAINS/Brenk; status pass|flag.",
                "applicability_domain": "Reliability band + confidence_factor for the activity "
                "predictions, by Tanimoto similarity to training data.",
                "warnings": "Advisory low-confidence flags; they do not change the numbers.",
            },
            "exploratory_caveats": [
                "All activity numbers are EXPLORATORY: the backbone is in-sample for known "
                "actives and only ~22 true L858R records exist.",
                "selectivity_proxy is a derived ML difference, not statistically validated "
                "(QSAR selectivity was not significant at n=9).",
                "ADMET filters are approximate and not a substitute for experimental profiling.",
                "Out-of-domain molecules carry a reduced confidence_factor; treat their "
                "activity predictions with caution.",
            ],
            "docking_selectivity": (
                "NOT computed at request time. Structure-based (AutoDock Vina) L858R-vs-WT "
                "selectivity requires the offline docking pipeline (rigid-receptor 2ITZ/2ITY, "
                "exhaustiveness=8). This API returns the fast ML/heuristic screen only; "
                "`docking_selectivity_available` is always false in responses."
            ),
        }
