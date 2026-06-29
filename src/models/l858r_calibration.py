"""
Model 3 — L858R-adapted QSAR calibration.

** EXPLORATORY — all predictions from this module are labeled exploratory. **

Fine-tunes the general EGFR backbone on the 22 L858R molecules.
Because 22 molecules is far too few for a standalone model, calibration is
the only valid approach: we learn a correction to backbone predictions, not
a new model.

Evaluation is LOOCV (leave-one-out cross-validation) with 5 random seeds for
the backbone RF.  Primary metric: Spearman rank correlation.

Calibration methods
-------------------
MeanShiftCalibrator
    Learns a constant bias: δ = mean(y_true − y_backbone) over training
    L858R molecules.  Has 1 degree of freedom; only corrects systematic bias.

RidgeCalibrator
    Fits y_cal = α * y_backbone + β via Ridge regression.  Has 2 degrees of
    freedom; corrects both bias and scale.

Both calibrators are fitted using OOB (out-of-bag) backbone predictions for
the training L858R molecules to avoid in-sample overfitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from src.utils.logging import get_logger

logger = get_logger(__name__)

EXPLORATORY_LABEL = "[EXPLORATORY — Model 3]"


# ── Calibrators ───────────────────────────────────────────────────────────────


class MeanShiftCalibrator:
    """
    Constant shift calibrator.

    Learns the mean residual δ = mean(y_true − y_backbone) over training
    L858R molecules, then applies it to every prediction.
    """

    def __init__(self) -> None:
        self.shift_: float = 0.0
        self._n_train: int = 0

    def fit(
        self,
        y_backbone_train: np.ndarray,
        y_true_train: np.ndarray,
    ) -> "MeanShiftCalibrator":
        deltas = y_true_train - y_backbone_train
        self.shift_ = float(np.mean(deltas))
        self._n_train = len(y_true_train)
        return self

    def predict(self, y_backbone: np.ndarray) -> np.ndarray:
        return np.asarray(y_backbone, dtype=float) + self.shift_

    def __repr__(self) -> str:
        return f"MeanShiftCalibrator(shift={self.shift_:.4f}, n_train={self._n_train})"


class RidgeCalibrator:
    """
    Linear scale+shift calibrator using Ridge regression.

    Fits y_cal = α * y_backbone + β on training L858R molecules.
    Uses backbone prediction as the sole feature; Ridge regularization
    prevents overfitting with small n.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._ridge: Ridge | None = None
        self._n_train: int = 0

    def fit(
        self,
        y_backbone_train: np.ndarray,
        y_true_train: np.ndarray,
    ) -> "RidgeCalibrator":
        X = np.asarray(y_backbone_train, dtype=float).reshape(-1, 1)
        y = np.asarray(y_true_train, dtype=float)
        self._ridge = Ridge(alpha=self.alpha)
        self._ridge.fit(X, y)
        self._n_train = len(y_true_train)
        return self

    def predict(self, y_backbone: np.ndarray) -> np.ndarray:
        if self._ridge is None:
            raise RuntimeError("RidgeCalibrator.fit() must be called before predict()")
        X = np.asarray(y_backbone, dtype=float).reshape(-1, 1)
        return self._ridge.predict(X)

    def __repr__(self) -> str:
        if self._ridge is not None:
            return (
                f"RidgeCalibrator(alpha={self.alpha}, "
                f"coef={self._ridge.coef_[0]:.4f}, "
                f"intercept={self._ridge.intercept_:.4f}, "
                f"n_train={self._n_train})"
            )
        return f"RidgeCalibrator(alpha={self.alpha}, unfitted)"


# ── LOOCV runner ──────────────────────────────────────────────────────────────


def _train_rf_backbone(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: list[str],
    seed: int,
    n_estimators: int = 100,
    rf_cfg: dict | None = None,
) -> RandomForestRegressor:
    """Train RF backbone with OOB enabled.  Uses fixed hyperparameters."""
    cfg = rf_cfg or {}
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=cfg.get("max_depth", None),
        min_samples_leaf=cfg.get("min_samples_leaf", 2),
        n_jobs=cfg.get("n_jobs", -1),
        random_state=seed,
        oob_score=True,
    )
    X_df = pd.DataFrame(X, columns=feature_cols)
    rf.fit(X_df, y)
    return rf


def run_loocv(
    general_df: pd.DataFrame,
    feature_cols: list[str],
    seeds: list[int] | None = None,
    n_estimators_loo: int = 100,
    ridge_alpha: float = 1.0,
    model_cfg: dict | None = None,
) -> dict:
    """
    LOOCV across 22 L858R molecules.

    Parameters
    ----------
    general_df
        Full general EGFR feature parquet (1253 rows).  Must contain
        columns: mutation_flag, canonical_smiles, pic50, and all feature_cols.
    feature_cols
        Ordered list of feature column names (2059 entries).
    seeds
        RF random_state values; one full LOOCV per seed.  Default: [42, 7, 13, 99, 123].
    n_estimators_loo
        Trees in the LOO backbone RF.  100 (not 300) to keep runtime ≤ 10 min;
        accuracy loss vs the saved backbone is acceptable for calibration training.
    ridge_alpha
        L2 regularization for RidgeCalibrator.
    model_cfg
        model_config.yaml dict (used for RF hyperparameters).

    Returns
    -------
    dict with structure:
        {
          "per_seed": {seed: {method: float}},      # Spearman r per seed per method
          "summary": {method: {"mean": float, "std": float}},
          "pooled_predictions": {seed: {method: array(22)}},  # raw LOOCV predictions
          "y_true": array(22),                       # same for all seeds (fixed data)
        }
    """
    if seeds is None:
        seeds = [42, 7, 13, 99, 123]

    rf_cfg = (model_cfg or {}).get("qsar", {}).get("random_forest", {})

    l858r_mask = general_df["mutation_flag"] == "L858R"
    l858r_idx = general_df.index[l858r_mask].tolist()
    n_l858r = len(l858r_idx)

    if n_l858r < 5:
        raise ValueError(
            f"Expected ≥5 L858R molecules, found {n_l858r}. Check general_df."
        )

    logger.info(
        f"{EXPLORATORY_LABEL} LOOCV on {n_l858r} L858R molecules, "
        f"{len(seeds)} seeds, n_estimators_loo={n_estimators_loo}"
    )
    logger.info(
        f"{EXPLORATORY_LABEL} Calibrators: backbone (baseline), "
        f"mean-shift, ridge (alpha={ridge_alpha})"
    )

    y_true_global = general_df.loc[l858r_idx, "pic50"].values.astype(float)

    per_seed: dict[int, dict[str, float]] = {}
    pooled: dict[int, dict[str, np.ndarray]] = {}

    for seed in seeds:
        logger.info(f"{EXPLORATORY_LABEL} seed={seed}: running {n_l858r} LOO folds …")

        preds_backbone = np.zeros(n_l858r)
        preds_shift = np.zeros(n_l858r)
        preds_ridge = np.zeros(n_l858r)

        for fold_i, held_out_row in enumerate(l858r_idx):
            # ── Build backbone training set: all general molecules minus held-out ──
            backbone_mask = general_df.index != held_out_row
            backbone_df = general_df[backbone_mask]

            X_back = backbone_df[feature_cols].values.astype(np.float32)
            y_back = backbone_df["pic50"].values.astype(np.float32)

            # ── Train backbone ────────────────────────────────────────────────────
            rf = _train_rf_backbone(
                X_back,
                y_back,
                feature_cols,
                seed,
                n_estimators=n_estimators_loo,
                rf_cfg=rf_cfg,
            )

            # ── OOB predictions for the 21 TRAINING L858R molecules ───────────────
            # These are out-of-bag predictions from the backbone, avoiding the
            # in-sample bias that would distort calibration training.
            l858r_train_mask = (backbone_df["mutation_flag"] == "L858R").values
            l858r_train_positions = np.where(l858r_train_mask)[0]

            oob_preds = rf.oob_prediction_  # shape (n_backbone,)
            y_back_train = oob_preds[l858r_train_positions]
            y_true_train = y_back[l858r_train_positions]

            # ── Direct prediction for held-out molecule ───────────────────────────
            X_held = general_df.loc[[held_out_row], feature_cols].values.astype(
                np.float32
            )
            X_held_df = pd.DataFrame(X_held, columns=feature_cols)
            y_back_test = float(rf.predict(X_held_df)[0])

            # ── Fit and apply calibrators ─────────────────────────────────────────
            shift_cal = MeanShiftCalibrator().fit(y_back_train, y_true_train)
            ridge_cal = RidgeCalibrator(alpha=ridge_alpha).fit(
                y_back_train, y_true_train
            )

            preds_backbone[fold_i] = y_back_test
            preds_shift[fold_i] = float(shift_cal.predict(np.array([y_back_test]))[0])
            preds_ridge[fold_i] = float(ridge_cal.predict(np.array([y_back_test]))[0])

            if (fold_i + 1) % 5 == 0 or fold_i == n_l858r - 1:
                logger.info(
                    f"  fold {fold_i + 1}/{n_l858r}  "
                    f"y_true={y_true_global[fold_i]:.2f}  "
                    f"y_back={y_back_test:.2f}  "
                    f"y_shift={preds_shift[fold_i]:.2f}  "
                    f"y_ridge={preds_ridge[fold_i]:.2f}"
                )

        # ── Compute Spearman r ────────────────────────────────────────────────────
        spear_backbone = float(spearmanr(y_true_global, preds_backbone).statistic)
        spear_shift = float(spearmanr(y_true_global, preds_shift).statistic)
        spear_ridge = float(spearmanr(y_true_global, preds_ridge).statistic)

        rmse_backbone = float(
            np.sqrt(mean_squared_error(y_true_global, preds_backbone))
        )
        rmse_shift = float(np.sqrt(mean_squared_error(y_true_global, preds_shift)))
        rmse_ridge = float(np.sqrt(mean_squared_error(y_true_global, preds_ridge)))

        per_seed[seed] = {
            "backbone_spearman": spear_backbone,
            "shift_spearman": spear_shift,
            "ridge_spearman": spear_ridge,
            "backbone_rmse": rmse_backbone,
            "shift_rmse": rmse_shift,
            "ridge_rmse": rmse_ridge,
        }
        pooled[seed] = {
            "backbone": preds_backbone.copy(),
            "shift": preds_shift.copy(),
            "ridge": preds_ridge.copy(),
        }

        logger.info(
            f"{EXPLORATORY_LABEL} seed={seed}  "
            f"backbone r={spear_backbone:.3f}  "
            f"shift r={spear_shift:.3f}  "
            f"ridge r={spear_ridge:.3f}"
        )

    # ── Summary statistics ────────────────────────────────────────────────────
    methods = ["backbone", "shift", "ridge"]
    summary: dict[str, dict[str, float]] = {}

    for method in methods:
        spear_key = f"{method}_spearman"
        rmse_key = f"{method}_rmse"
        vals_spear = [per_seed[s][spear_key] for s in seeds]
        vals_rmse = [per_seed[s][rmse_key] for s in seeds]
        summary[method] = {
            "spearman_mean": float(np.mean(vals_spear)),
            "spearman_std": float(np.std(vals_spear)),
            "rmse_mean": float(np.mean(vals_rmse)),
            "rmse_std": float(np.std(vals_rmse)),
        }

    return {
        "per_seed": per_seed,
        "summary": summary,
        "pooled_predictions": pooled,
        "y_true": y_true_global,
        "n_l858r": n_l858r,
        "seeds": seeds,
    }


def interpret_result(summary: dict) -> str:
    """
    Plain-English verdict: does calibration beat the backbone?

    Returns a string starting with EXPLORATORY_LABEL.
    """
    back_mean = summary["backbone"]["spearman_mean"]
    back_std = summary["backbone"]["spearman_std"]

    best_cal_method = max(
        ["shift", "ridge"],
        key=lambda m: summary[m]["spearman_mean"],
    )
    cal_mean = summary[best_cal_method]["spearman_mean"]
    cal_std = summary[best_cal_method]["spearman_std"]

    delta = cal_mean - back_mean

    if delta > 0.05:
        verdict = (
            f"Calibration ({best_cal_method}) improves Spearman r by {delta:+.3f} "
            f"({back_mean:.3f} → {cal_mean:.3f}).  "
            f"L858R-specific signal is detectable at n=22."
        )
    elif delta > 0.0:
        verdict = (
            f"Calibration ({best_cal_method}) improves Spearman r by {delta:+.3f} "
            f"({back_mean:.3f} → {cal_mean:.3f}), but the effect is within noise "
            f"(backbone std={back_std:.3f}).  "
            f"Improvement is marginal; use the general backbone."
        )
    else:
        verdict = (
            f"Calibration does not improve over the backbone "
            f"(backbone r={back_mean:.3f}+/-{back_std:.3f}, "
            f"best calibration r={cal_mean:.3f}+/-{cal_std:.3f}).  "
            f"L858R-specific signal is not separable at n=22; "
            f"use the general backbone."
        )

    return f"{EXPLORATORY_LABEL} {verdict}"
