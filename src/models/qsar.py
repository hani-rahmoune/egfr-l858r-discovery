"""
QSAR regressor trainer.

Trains RF, XGBoost, and LightGBM candidates, selects the best by val RMSE,
then evaluates on a held-out test set.  Designed for the two-stage general →
WT-proxy → L858R-fine-tune pipeline.

Feature columns are everything in the parquet that is not recognised metadata.
The fixed metadata set is defined by META_COLS below.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.utils.logging import get_logger

logger = get_logger(__name__)

META_COLS: frozenset[str] = frozenset(
    {
        "pic50",
        "canonical_smiles",
        "mutation_flag",
        "source",
        "split",
        "scaffold",
        "dataset",
        "binary_label",
        "activity_class",
        "smiles_valid",
    }
)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from scipy.stats import spearmanr

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "spearman_r": float(spearmanr(y_true, y_pred).statistic),
        "n": int(len(y_true)),
    }


def _build_candidates(cfg: dict) -> dict[str, Any]:
    """Instantiate unfitted model objects from model_config.yaml."""
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    rf_cfg = cfg.get("qsar", {}).get("random_forest", {})
    xgb_cfg = cfg.get("qsar", {}).get("xgboost", {})
    lgb_cfg = cfg.get("qsar", {}).get("lightgbm", {})

    return {
        "random_forest": RandomForestRegressor(
            n_estimators=rf_cfg.get("n_estimators", 300),
            max_depth=rf_cfg.get("max_depth", None),
            min_samples_leaf=rf_cfg.get("min_samples_leaf", 2),
            n_jobs=rf_cfg.get("n_jobs", -1),
            random_state=rf_cfg.get("random_state", 42),
        ),
        "xgboost": XGBRegressor(
            n_estimators=xgb_cfg.get("n_estimators", 300),
            learning_rate=xgb_cfg.get("learning_rate", 0.05),
            max_depth=xgb_cfg.get("max_depth", 6),
            subsample=xgb_cfg.get("subsample", 0.8),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
            random_state=xgb_cfg.get("random_state", 42),
            verbosity=0,
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=lgb_cfg.get("n_estimators", 300),
            learning_rate=lgb_cfg.get("learning_rate", 0.05),
            num_leaves=lgb_cfg.get("num_leaves", 63),
            random_state=lgb_cfg.get("random_state", 42),
            verbose=-1,
        ),
    }


class QSARTrainer:
    """
    Train RF / XGBoost / LightGBM on a scaffold-split feature parquet.

    Attributes set after fit():
        best_name       name of the winning candidate
        best_model      fitted sklearn-compatible estimator
        scaler          fitted StandardScaler (saved for applicability domain)
        val_metrics     dict of metrics on val set for each candidate
        test_metrics    dict of metrics on test set for best candidate
        feature_cols    ordered list of feature column names
    """

    def __init__(self, model_cfg: dict) -> None:
        self.model_cfg = model_cfg
        self.scaler = StandardScaler()
        self.best_name: str | None = None
        self.best_model: Any = None
        self.feature_cols: list[str] = []
        self.val_metrics: dict = {}
        self.test_metrics: dict = {}

    # ── public API ──────────────────────────────────────────────────────────

    def fit_from_parquet(self, parquet_path: Path, label: str = "") -> None:
        df = pd.read_parquet(parquet_path)
        if "split" not in df.columns:
            raise ValueError(
                f"No 'split' column in {parquet_path}. Run assign_splits.py first."
            )

        self.feature_cols = get_feature_cols(df)
        logger.info(
            f"{label}: {len(df)} rows, {len(self.feature_cols)} features, "
            f"split counts: {df['split'].value_counts().to_dict()}"
        )

        train = df[df["split"] == "train"]
        val = df[df["split"] == "val"]
        test = df[df["split"] == "test"]

        X_tr = train[self.feature_cols].values.astype(np.float32)
        y_tr = train["pic50"].values.astype(np.float32)
        X_val = val[self.feature_cols].values.astype(np.float32)
        y_val = val["pic50"].values.astype(np.float32)
        X_te = test[self.feature_cols].values.astype(np.float32)
        y_te = test["pic50"].values.astype(np.float32)

        # Fit scaler on train only (persisted for inference; the tree models
        # below train on the unscaled named DataFrames).
        self.scaler.fit(X_tr)

        candidates = _build_candidates(self.model_cfg)

        # Use named DataFrames for fit/predict so LightGBM's internal feature
        # names stay consistent with sklearn's validation on predict calls.
        X_tr_df = pd.DataFrame(X_tr, columns=self.feature_cols)
        X_val_df = pd.DataFrame(X_val, columns=self.feature_cols)
        X_te_df = pd.DataFrame(X_te, columns=self.feature_cols)

        best_val_rmse = float("inf")
        for name, model in candidates.items():
            logger.info(f"{label}: fitting {name} on {len(X_tr_df)} train rows …")
            model.fit(X_tr_df, y_tr)
            val_pred = model.predict(X_val_df)
            val_m = compute_metrics(y_val, val_pred)
            self.val_metrics[name] = val_m
            logger.info(
                f"  {name} val  RMSE={val_m['rmse']:.3f}  R²={val_m['r2']:.3f}  "
                f"pearson_r={val_m['pearson_r']:.3f}"
            )
            if val_m["rmse"] < best_val_rmse:
                best_val_rmse = val_m["rmse"]
                self.best_name = name
                self.best_model = model

        logger.info(
            f"{label}: best model = {self.best_name}  (val RMSE {best_val_rmse:.3f})"
        )
        test_pred = self.best_model.predict(X_te_df)
        self.test_metrics = compute_metrics(y_te, test_pred)
        logger.info(
            f"{label}: test  RMSE={self.test_metrics['rmse']:.3f}  "
            f"R²={self.test_metrics['r2']:.3f}  "
            f"pearson_r={self.test_metrics['pearson_r']:.3f}  "
            f"n={self.test_metrics['n']}"
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.best_model.predict(X)

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.best_model, out_dir / "best_model.pkl")
        joblib.dump(self.scaler, out_dir / "scaler.pkl")
        metadata = {
            "best_model": self.best_name,
            "feature_cols": self.feature_cols,
            "val_metrics": self.val_metrics,
            "test_metrics": self.test_metrics,
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved model artifacts to {out_dir}")

    @classmethod
    def load(cls, out_dir: Path, model_cfg: dict | None = None) -> "QSARTrainer":
        trainer = cls(model_cfg or {})
        trainer.best_model = joblib.load(out_dir / "best_model.pkl")
        trainer.scaler = joblib.load(out_dir / "scaler.pkl")
        with open(out_dir / "metadata.json") as f:
            meta = json.load(f)
        trainer.best_name = meta["best_model"]
        trainer.feature_cols = meta["feature_cols"]
        trainer.val_metrics = meta["val_metrics"]
        trainer.test_metrics = meta["test_metrics"]
        return trainer
