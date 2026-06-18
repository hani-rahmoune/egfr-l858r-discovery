"""
Model 4 — Derived selectivity estimation (EXPLORATORY).

** EXPLORATORY — all predictions from this module are labeled exploratory. **
** This is NOT a selectivity model. There are only 9 paired molecules. **

Sign convention (from CLAUDE.md):
    selectivity_delta = pIC50_mutant - pIC50_wt
    Positive = mutant-selective (desired)
    Negative = WT-selective

Derived delta:
    pred_delta = backbone_pred(X) - wt_proxy_pred(X)

    The mutant predictor is the general EGFR backbone (Model 1), consistent with
    the Model 3 LOOCV verdict: L858R calibration did not improve over backbone.
    The WT predictor is the WT-proxy model (Model 2).

    IMPORTANT CAVEAT: All 9 paired molecules are in the training data for BOTH
    models (backbone was trained on their L858R pIC50; WT-proxy on their WT
    pIC50).  There is no holdout for the activity predictions — only the
    selectivity label is truly held out in the LOO loop.  Predictions from
    both models for these molecules are IN-SAMPLE.

    Additionally, the two models were trained on heavily overlapping datasets
    (backbone on all 1253 EGFR molecules, WT-proxy on 1018 of those).  The
    derived delta therefore has a strong shared-noise component and may be
    near-zero even for genuinely selective compounds.

Evaluation:
    LOO across the 9 pairs against two trivial baselines.
    Spearman rank correlation is the primary metric.
    At n=9, Spearman is extremely unstable — one rank swap can shift r by ~0.2.
    A statistically significant rank correlation requires roughly n≥25 at r≥0.4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.features.descriptors import compute_descriptor_matrix, DESCRIPTOR_NAMES
from src.features.fingerprints import compute_fingerprint_matrix
from src.utils.logging import get_logger

logger = get_logger(__name__)

EXPLORATORY_LABEL = "[EXPLORATORY — Model 4]"

_FP_COLS = [f"fp_{i}" for i in range(2048)]
FEATURE_COLS = _FP_COLS + list(DESCRIPTOR_NAMES)  # 2059 total, fixed order


def spearman_stability_note(n: int) -> str:
    """Return a canned warning about Spearman instability at small n."""
    return (
        f"STABILITY WARNING: Spearman r on n={n} is extremely unstable. "
        f"One rank swap changes r by ~{2 * 6 / (n * (n**2 - 1)):.2f}. "
        f"Statistical significance requires r>0.65 at n=9 (p<0.05, two-sided). "
        f"Do not interpret these numbers without docking/FEP orthogonal data."
    )


def compute_features_for_smiles(
    smiles_list: list[str],
) -> tuple[np.ndarray, list[int]]:
    """
    Compute Morgan ECFP4 + RDKit descriptors for a list of SMILES.
    Returns (X, valid_indices) where valid_indices maps result rows to
    the input SMILES list positions.
    Feature dimension is 2059 (2048 Morgan + 11 descriptors), fixed order.
    """
    fp_matrix, fp_valid = compute_fingerprint_matrix(
        smiles_list, fp_type="morgan_ecfp4", radius=2, n_bits=2048
    )
    desc_matrix, desc_valid = compute_descriptor_matrix(smiles_list)

    valid_set    = sorted(set(fp_valid) & set(desc_valid))
    fp_map   = {orig: i for i, orig in enumerate(fp_valid)}
    desc_map = {orig: i for i, orig in enumerate(desc_valid)}

    fp_rows   = np.array([fp_matrix[fp_map[i]]   for i in valid_set])
    desc_rows = np.array([desc_matrix[desc_map[i]] for i in valid_set])
    X = np.concatenate([fp_rows, desc_rows], axis=1).astype(np.float32)
    return X, valid_set


def predict_with_trainer(trainer, X: np.ndarray) -> np.ndarray:
    """
    Predict using a loaded QSARTrainer, passing a named DataFrame so
    tree models (LGB) don't raise feature-name warnings.
    """
    X_df = pd.DataFrame(X, columns=trainer.feature_cols)
    return trainer.best_model.predict(X_df)


def evaluate_selectivity(
    sel_df: pd.DataFrame,
    backbone_trainer,
    wt_proxy_trainer,
) -> dict:
    """
    Run the full selectivity evaluation on paired molecules.

    Parameters
    ----------
    sel_df
        DataFrame with columns: canonical_smiles, pic50_mutant, pic50_wt,
        selectivity_delta.  Sign convention: delta = mutant - wt.
    backbone_trainer
        Loaded QSARTrainer for Model 1 (general EGFR backbone).
    wt_proxy_trainer
        Loaded QSARTrainer for Model 2 (WT-proxy).

    Returns
    -------
    dict with keys:
        n_pairs, true_delta, pred_l858r, pred_wt, derived_delta,
        loomean_preds, constant_pred,
        spearman_derived, spearman_loomean, spearman_constant,
        per_pair (list of per-molecule detail dicts),
        stability_note, verdict
    """
    smiles_list  = sel_df["canonical_smiles"].tolist()
    true_delta   = sel_df["selectivity_delta"].values.astype(float)
    true_mutant  = sel_df["pic50_mutant"].values.astype(float)
    true_wt      = sel_df["pic50_wt"].values.astype(float)
    n = len(true_delta)

    logger.info(
        f"{EXPLORATORY_LABEL} computing features for {n} paired molecules ..."
    )
    X, valid_idx = compute_features_for_smiles(smiles_list)

    if len(valid_idx) < n:
        failed = sorted(set(range(n)) - set(valid_idx))
        logger.warning(
            f"{EXPLORATORY_LABEL} Feature computation failed for rows: {failed}. "
            f"Those pairs are excluded from evaluation."
        )

    # Restrict to molecules that succeeded feature computation
    X_valid      = X
    true_delta_v = true_delta[valid_idx]
    true_mutant_v = true_mutant[valid_idx]
    true_wt_v    = true_wt[valid_idx]
    smiles_valid  = [smiles_list[i] for i in valid_idx]
    n_v = len(valid_idx)

    logger.info(f"{EXPLORATORY_LABEL} predicting with backbone (Model 1) ...")
    pred_l858r = predict_with_trainer(backbone_trainer, X_valid)

    logger.info(f"{EXPLORATORY_LABEL} predicting with WT-proxy (Model 2) ...")
    pred_wt = predict_with_trainer(wt_proxy_trainer, X_valid)

    derived_delta  = pred_l858r - pred_wt
    constant_pred  = np.full(n_v, np.mean(true_delta_v))  # same for all
    loomean_preds  = np.array([
        np.mean(np.delete(true_delta_v, i)) for i in range(n_v)
    ])

    # Spearman r (suppress nan_policy issues via try/except)
    def _spearman(a, b):
        res = spearmanr(a, b)
        return float(res.statistic), float(res.pvalue)

    r_derived,  p_derived  = _spearman(true_delta_v, derived_delta)
    r_loomean,  p_loomean  = _spearman(true_delta_v, loomean_preds)
    r_constant, p_constant = _spearman(true_delta_v, constant_pred)

    per_pair = []
    orig_idxs = list(sel_df.index) if hasattr(sel_df, 'index') else list(range(n))
    for k, vi in enumerate(valid_idx):
        per_pair.append({
            "orig_idx":     vi,
            "smiles":       smiles_valid[k],
            "true_mutant":  float(true_mutant_v[k]),
            "true_wt":      float(true_wt_v[k]),
            "true_delta":   float(true_delta_v[k]),
            "pred_l858r":   float(pred_l858r[k]),
            "pred_wt":      float(pred_wt[k]),
            "derived_delta": float(derived_delta[k]),
            "loomean_pred": float(loomean_preds[k]),
            "err_backbone": float(pred_l858r[k] - true_mutant_v[k]),
            "err_wt_proxy": float(pred_wt[k] - true_wt_v[k]),
        })

    stability_note = spearman_stability_note(n_v)
    verdict        = _build_verdict(r_derived, r_loomean, r_constant, n_v)

    return {
        "n_pairs":           n_v,
        "true_delta":        true_delta_v,
        "pred_l858r":        pred_l858r,
        "pred_wt":           pred_wt,
        "derived_delta":     derived_delta,
        "loomean_preds":     loomean_preds,
        "constant_pred":     constant_pred,
        "spearman_derived":  r_derived,
        "pvalue_derived":    p_derived,
        "spearman_loomean":  r_loomean,
        "pvalue_loomean":    p_loomean,
        "spearman_constant": r_constant,
        "pvalue_constant":   p_constant,
        "per_pair":          per_pair,
        "stability_note":    stability_note,
        "verdict":           verdict,
    }


def _build_verdict(r_derived: float, r_loomean: float, r_constant: float, n: int) -> str:
    """
    Plain-English verdict on whether derived selectivity is informative.
    Returns a string beginning with EXPLORATORY_LABEL.
    """
    # Significance threshold for n-sized Spearman (approx p<0.05 two-sided)
    # Critical r for n=9 is ~0.683; for n=8 ~0.738; for n=7 ~0.786
    # Using the t-distribution approximation: t = r*sqrt((n-2)/(1-r^2)), df=n-2
    # We hard-code conservative thresholds.
    from scipy.stats import t as t_dist
    df = max(n - 2, 1)
    t_crit = float(t_dist.ppf(0.975, df=df))  # two-sided 5%

    def _t_from_r(r):
        r2 = min(abs(r), 0.9999) ** 2
        return abs(r) * np.sqrt((n - 2) / (1 - r2))

    is_significant = _t_from_r(r_derived) > t_crit

    if r_derived > r_loomean and r_derived > r_constant and is_significant:
        verdict = (
            f"Derived selectivity (r={r_derived:.3f}, p<0.05) beats both trivial "
            f"baselines (LOO-mean r={r_loomean:.3f}, constant r={r_constant:.3f}).  "
            f"Some rank signal exists at n={n}.  "
            f"Still exploratory; validate with docking or FEP."
        )
    else:
        reason = []
        if r_derived <= r_loomean:
            reason.append(f"does not beat LOO-mean baseline (r={r_loomean:.3f})")
        if r_derived <= r_constant:
            reason.append(f"does not beat constant baseline (r={r_constant:.3f})")
        if not is_significant:
            reason.append(f"not statistically significant at n={n} (r={r_derived:.3f} < threshold)")
        verdict = (
            f"Derived selectivity {'; '.join(reason)}.  "
            f"Selectivity cannot be modeled at n={n}; "
            f"structure-based methods (docking, FEP) are the path.  "
            f"Treat the {n} deltas as exploratory reference data, not a model."
        )

    return f"{EXPLORATORY_LABEL} {verdict}"
