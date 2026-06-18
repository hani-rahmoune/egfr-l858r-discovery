"""
Optional fingerprint ablation study (Phase 5).

Benchmarks six fingerprint representations separately against both
EGFR general (Model 1) and WT-proxy (Model 2) tasks using
RF / XGB / LGB, 5 scaffold-split seeds.

Representations evaluated (FP bits + 11 RDKit descriptors):
  morgan_ecfp4       2048 + 11 = 2059  PRIMARY (used in production models)
  morgan_ecfp6       2048 + 11 = 2059
  maccs               167 + 11 =  178
  rdkit_topological  2048 + 11 = 2059
  atom_pair          2048 + 11 = 2059
  topological_torsion 2048 + 11 = 2059

n_estimators = 100 (vs 300 in saved artifacts) for runtime;
same choice as L858R LOOCV.

Design notes:
- Scaffold splits are computed ONCE per task (not per FP type) to avoid
  repeating Murcko scaffold computation 30 times (6 FPs x 5 seeds).
- FP matrices are computed ONCE per FP type (not per seed).
- Model type (RF/XGB/LGB) and FP type are independently selected per task.
- Winner = lowest mean val RMSE across seeds.
- Do NOT concatenate fingerprints. Do NOT build ensemble of FP types.

Run:
    PYTHONPATH=. .venv/Scripts/python.exe scripts/fingerprint_ablation.py

Output:
    models/qsar/fingerprint_ablation_results.json
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.features.descriptors import DESCRIPTOR_NAMES, compute_descriptor_matrix
from src.features.fingerprints import compute_fingerprint_matrix
from src.splitting.scaffold_split import get_bemis_murcko_scaffold
from src.utils.config import get_project_root, load_model_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

ROOT = get_project_root()
PROCESSED = ROOT / "data" / "processed"
OUT_DIR = ROOT / "models" / "qsar"

# ── Configuration ──────────────────────────────────────────────────────────────

FP_CONFIGS: dict[str, dict] = {
    "morgan_ecfp4": {"fp_type": "morgan_ecfp4", "radius": 2, "n_bits": 2048},
    "morgan_ecfp6": {"fp_type": "morgan_ecfp6", "radius": 3, "n_bits": 2048},
    "maccs": {"fp_type": "maccs", "n_bits": 167},
    "rdkit_topological": {"fp_type": "rdkit_topological", "n_bits": 2048},
    "atom_pair": {"fp_type": "atom_pair", "n_bits": 2048},
    "topological_torsion": {"fp_type": "topological_torsion", "n_bits": 2048},
}

SEEDS = [42, 7, 13, 99, 123]
N_ESTIMATORS = 100
MODEL_NAMES = ["random_forest", "xgboost", "lightgbm"]

TASKS: dict[str, str] = {
    "general": "features_egfr_general",
    "wt_proxy": "features_wt_proxy",
}


# ── Model construction ────────────────────────────────────────────────────────


def _build_model(model_name: str, seed: int, n_estimators: int):
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        )
    if model_name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbosity=0,
        )
    if model_name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=63,
            random_state=seed,
            verbose=-1,
        )
    raise ValueError(f"Unknown model: {model_name}")


# ── Feature computation ───────────────────────────────────────────────────────


def compute_fp_desc_matrix(
    smiles_list: list[str],
    fp_type: str,
    n_bits: int,
    radius: int = 2,
) -> tuple[np.ndarray, list[int]]:
    """
    Compute FP + 11 RDKit descriptor feature matrix.
    Returns (X, valid_indices) where valid_indices are positions in smiles_list.
    """
    fp_kwargs: dict = {"fp_type": fp_type, "n_bits": n_bits}
    if fp_type in ("morgan_ecfp4", "morgan_ecfp6"):
        fp_kwargs["radius"] = radius

    fp_mat, fp_valid = compute_fingerprint_matrix(smiles_list, **fp_kwargs)
    desc_mat, desc_valid = compute_descriptor_matrix(smiles_list)

    valid_set = sorted(set(fp_valid) & set(desc_valid))
    if not valid_set:
        return np.empty((0, n_bits + len(DESCRIPTOR_NAMES)), dtype=np.float32), []

    fp_pos = {v: i for i, v in enumerate(fp_valid)}
    desc_pos = {v: i for i, v in enumerate(desc_valid)}

    fp_rows = fp_mat[[fp_pos[i] for i in valid_set]]
    desc_rows = desc_mat[[desc_pos[i] for i in valid_set]]

    X = np.hstack([fp_rows, desc_rows]).astype(np.float32)
    return X, valid_set


# ── Scaffold split (index-based, pre-computed scaffolds) ───────────────────────


def build_scaffold_lookup(smiles_list: list[str]) -> dict[str, str]:
    """Compute Bemis-Murcko scaffold for each SMILES. Cached as a dict."""
    lookup: dict[str, str] = {}
    for smi in smiles_list:
        sc = get_bemis_murcko_scaffold(smi)
        lookup[smi] = sc if sc is not None else "no_scaffold"
    return lookup


def scaffold_split_indices(
    smiles_list: list[str],
    scaffold_lookup: dict[str, str],
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[list[int], list[int], list[int]]:
    """
    Scaffold split using pre-computed scaffold lookup.
    Returns (train_idx, val_idx, test_idx) as positional indices into smiles_list.
    Same algorithm as src.splitting.scaffold_split.scaffold_split, but reuses
    pre-computed scaffolds to avoid repeating Murcko computation per seed.
    """
    scaffold_to_pos: dict[str, list[int]] = defaultdict(list)
    for pos, smi in enumerate(smiles_list):
        scaffold_to_pos[scaffold_lookup.get(smi, "no_scaffold")].append(pos)

    groups = sorted(scaffold_to_pos.values(), key=len, reverse=True)
    np.random.default_rng(seed).shuffle(groups)

    n = len(smiles_list)
    train_cut = int(n * train_ratio)
    val_cut = int(n * (train_ratio + val_ratio))

    train_idx, val_idx, test_idx = [], [], []
    for group in groups:
        if len(train_idx) < train_cut:
            train_idx.extend(group)
        elif len(train_idx) + len(val_idx) < val_cut:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    return sorted(train_idx), sorted(val_idx), sorted(test_idx)


# ── Per-combo runner ──────────────────────────────────────────────────────────


def run_one_seed(
    X: np.ndarray,
    y: np.ndarray,
    smiles_valid: list[str],
    scaffold_lookup: dict[str, str],
    seed: int,
    n_estimators: int,
) -> dict[str, dict[str, float]]:
    """
    Train RF/XGB/LGB for one (FP type, seed).
    Returns {model_name: {val_rmse, test_rmse, test_r2, test_spearman}}.
    """
    train_idx, val_idx, test_idx = scaffold_split_indices(
        smiles_valid, scaffold_lookup, seed
    )
    if not train_idx or not val_idx or not test_idx:
        return {}

    feat_cols = [f"f{i}" for i in range(X.shape[1])]

    X_tr = pd.DataFrame(X[train_idx], columns=feat_cols)
    X_va = pd.DataFrame(X[val_idx], columns=feat_cols)
    X_te = pd.DataFrame(X[test_idx], columns=feat_cols)
    y_tr, y_va, y_te = y[train_idx], y[val_idx], y[test_idx]

    results: dict[str, dict[str, float]] = {}
    for mn in MODEL_NAMES:
        model = _build_model(mn, seed, n_estimators)
        model.fit(X_tr, y_tr)

        va_pred = model.predict(X_va)
        te_pred = model.predict(X_te)

        results[mn] = {
            "val_rmse": float(np.sqrt(mean_squared_error(y_va, va_pred))),
            "test_rmse": float(np.sqrt(mean_squared_error(y_te, te_pred))),
            "test_r2": float(r2_score(y_te, te_pred)),
            "test_spearman": float(spearmanr(y_te, te_pred).statistic),
            "test_n": int(len(y_te)),
        }
    return results


# ── Per-task ablation ─────────────────────────────────────────────────────────


def run_task_ablation(
    parquet_path: Path,
    task_name: str,
    seeds: list[int],
    n_estimators: int,
) -> dict:
    """
    Run all FP types for one task. Returns nested results dict.

    Scaffold computation: once per task.
    FP matrix computation: once per FP type.
    """
    df = pd.read_parquet(parquet_path)
    smiles_all = df["canonical_smiles"].tolist()
    y_all = df["pic50"].values.astype(np.float32)

    logger.info(
        f"{task_name}: computing Bemis-Murcko scaffolds for {len(smiles_all)} molecules ..."
    )
    t0 = time.time()
    scaffold_lookup = build_scaffold_lookup(smiles_all)
    logger.info(f"  scaffolds done in {time.time()-t0:.1f}s")

    task_results: dict[str, dict] = {}

    for fp_key, fp_cfg in FP_CONFIGS.items():
        logger.info(f"{task_name} | {fp_key}: computing feature matrix ...")
        t1 = time.time()
        X, valid_set = compute_fp_desc_matrix(
            smiles_all,
            fp_type=fp_cfg["fp_type"],
            n_bits=fp_cfg["n_bits"],
            radius=fp_cfg.get("radius", 2),
        )
        n_features = X.shape[1]
        n_mols = len(valid_set)
        logger.info(
            f"  {fp_key}: {n_mols} molecules, {n_features} features, "
            f"computed in {time.time()-t1:.1f}s"
        )

        y_valid = y_all[valid_set]
        smiles_valid = [smiles_all[i] for i in valid_set]

        per_seed: list[dict] = []
        for seed in seeds:
            seed_result = run_one_seed(
                X, y_valid, smiles_valid, scaffold_lookup, seed, n_estimators
            )
            if seed_result:
                per_seed.append(seed_result)

        if not per_seed:
            logger.warning(f"  {fp_key}: all seeds failed, skipping")
            continue

        # Aggregate across seeds: for each model, compute mean/std of each metric
        model_agg: dict[str, dict[str, float]] = {}
        for mn in MODEL_NAMES:
            vals = {
                k: [s[mn][k] for s in per_seed if mn in s]
                for k in ["val_rmse", "test_rmse", "test_r2", "test_spearman"]
            }
            model_agg[mn] = {f"{k}_mean": float(np.mean(v)) for k, v in vals.items()}
            model_agg[mn].update(
                {f"{k}_std": float(np.std(v)) for k, v in vals.items()}
            )
            model_agg[mn]["test_n_mean"] = float(
                np.mean([s[mn]["test_n"] for s in per_seed if mn in s])
            )

        # Pick best model for this FP type by mean val_rmse
        best_model = min(MODEL_NAMES, key=lambda m: model_agg[m]["val_rmse_mean"])

        task_results[fp_key] = {
            "n_features": n_features,
            "n_mols": n_mols,
            "best_model": best_model,
            "models": model_agg,
            "best": model_agg[best_model],  # shorthand for reporting
        }

        logger.info(
            f"  {fp_key}: best={best_model}  "
            f"val_RMSE={model_agg[best_model]['val_rmse_mean']:.3f}  "
            f"test_RMSE={model_agg[best_model]['test_rmse_mean']:.3f}+/-"
            f"{model_agg[best_model]['test_rmse_std']:.3f}  "
            f"R2={model_agg[best_model]['test_r2_mean']:.3f}  "
            f"Spearman={model_agg[best_model]['test_spearman_mean']:.3f}"
        )

    return task_results


# ── Reporting ─────────────────────────────────────────────────────────────────


def _print_task_table(task_name: str, results: dict) -> None:
    """Print formatted ablation table for one task."""
    print()
    print(f"  Task: {task_name}")

    hdr = (
        f"  {'Fingerprint':<22}  {'n_feat':>6}  {'Best model':<14}"
        f"  {'val RMSE':>9}  {'test RMSE':>13}  {'test R2':>10}  {'Spearman r':>12}"
    )
    sep = "  " + "-" * (len(hdr) - 2)
    print(sep)
    print(hdr)
    print(sep)

    # Sort by val_rmse_mean so winner is first
    sorted_fps = sorted(
        results.items(),
        key=lambda kv: kv[1]["best"]["val_rmse_mean"],
    )

    for fp_key, res in sorted_fps:
        b = res["best"]
        print(
            f"  {fp_key:<22}  {res['n_features']:>6}  {res['best_model']:<14}"
            f"  {b['val_rmse_mean']:>9.3f}"
            f"  {b['test_rmse_mean']:>6.3f}+/-{b['test_rmse_std']:.3f}"
            f"  {b['test_r2_mean']:>6.3f}+/-{b['test_r2_std']:.3f}"
            f"  {b['test_spearman_mean']:>7.3f}+/-{b['test_spearman_std']:.3f}"
        )
    print(sep)

    winner_fp, winner_res = sorted_fps[0]
    print(
        f"\n  Winner for {task_name}: {winner_fp} "
        f"({winner_res['best_model']}, "
        f"val RMSE={winner_res['best']['val_rmse_mean']:.3f})"
    )


def _print_full_model_breakdown(task_name: str, fp_key: str, results: dict) -> None:
    """Print all 3 models for one (task, FP) combo — for the winner only."""
    res = results[fp_key]
    print(
        f"\n  {task_name} | {fp_key} — all models (mean +/- std across {len(SEEDS)} seeds):"
    )
    hdr = (
        f"  {'Model':<14}  {'val RMSE':>9}  {'test RMSE':>13}"
        f"  {'test R2':>10}  {'Spearman r':>12}"
    )
    sep = "  " + "-" * (len(hdr) - 2)
    print(sep)
    print(hdr)
    print(sep)
    for mn in MODEL_NAMES:
        m = res["models"][mn]
        marker = " *" if mn == res["best_model"] else "  "
        print(
            f"  {mn:<14}{marker}"
            f"  {m['val_rmse_mean']:>7.3f}"
            f"  {m['test_rmse_mean']:>6.3f}+/-{m['test_rmse_std']:.3f}"
            f"  {m['test_r2_mean']:>6.3f}+/-{m['test_r2_std']:.3f}"
            f"  {m['test_spearman_mean']:>7.3f}+/-{m['test_spearman_std']:.3f}"
        )
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    load_model_config()
    t_all = time.time()

    all_results: dict[str, dict] = {}
    winners: dict[str, dict] = {}

    for task_name, parquet_stem in TASKS.items():
        parquet_path = PROCESSED / f"{parquet_stem}.parquet"
        if not parquet_path.exists():
            logger.error(
                f"{parquet_path} not found. "
                f"Run compute_features.py and assign_splits.py first."
            )
            continue

        logger.info("=" * 70)
        logger.info(
            f"FINGERPRINT ABLATION  task={task_name}  seeds={SEEDS}  n_est={N_ESTIMATORS}"
        )
        logger.info("=" * 70)

        task_results = run_task_ablation(
            parquet_path=parquet_path,
            task_name=task_name,
            seeds=SEEDS,
            n_estimators=N_ESTIMATORS,
        )
        all_results[task_name] = task_results

        if task_results:
            winner_fp = min(
                task_results,
                key=lambda k: task_results[k]["best"]["val_rmse_mean"],
            )
            winners[task_name] = {
                "fp": winner_fp,
                "model": task_results[winner_fp]["best_model"],
                "val_rmse_mean": task_results[winner_fp]["best"]["val_rmse_mean"],
            }

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = OUT_DIR / "fingerprint_ablation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": all_results,
                "winners": winners,
                "seeds": SEEDS,
                "n_estimators": N_ESTIMATORS,
                "fp_configs": FP_CONFIGS,
            },
            f,
            indent=2,
        )
    logger.info(f"Results saved to {out_path}")

    # ── Print tables ───────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  FINGERPRINT ABLATION RESULTS")
    print(f"  n_estimators={N_ESTIMATORS}  seeds={SEEDS}")
    print("  Winner = lowest val RMSE (mean across seeds)")
    print("=" * 70)

    for task_name, task_results in all_results.items():
        if not task_results:
            continue
        _print_task_table(task_name, task_results)

        winner_fp = winners.get(task_name, {}).get("fp")
        if winner_fp and winner_fp in task_results:
            _print_full_model_breakdown(task_name, winner_fp, task_results)

    print()
    print("  WINNERS:")
    for task_name, w in winners.items():
        print(
            f"    {task_name:<12} -> {w['fp']:<22}  "
            f"{w['model']}  val_RMSE={w['val_rmse_mean']:.3f}"
        )

    print()
    print(f"  Total runtime: {(time.time()-t_all)/60:.1f} min")
    print()


if __name__ == "__main__":
    main()
