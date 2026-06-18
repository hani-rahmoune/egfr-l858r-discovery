"""
Data-loading helpers for the Streamlit dashboard (Phase 25).

Every function here is PURE (file read -> DataFrame/dict) and tolerant of a
missing artifact (returns None or an empty frame) so the dashboard degrades
gracefully instead of crashing. These are the functions covered by tests.

All numbers surfaced are EXPLORATORY — see the Limitations page.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_QSAR = PROJECT_ROOT / "models" / "qsar"
_GEN = PROJECT_ROOT / "models" / "generator"

# 5-seed scaffold-split stability (seeds 42,7,13,99,123). Source:
# scripts/eval_seed_stability.py, documented in CLAUDE.md. Not persisted as JSON,
# so encoded here as the authoritative recorded values.
SEED_STABILITY = pd.DataFrame(
    [
        {
            "model": "Model 1 — EGFR general backbone",
            "rmse_mean": 1.010,
            "rmse_std": 0.167,
            "r2_mean": 0.438,
            "r2_std": 0.143,
            "pearson_mean": 0.672,
            "pearson_std": 0.105,
        },
        {
            "model": "Model 2 — WT-proxy",
            "rmse_mean": 0.942,
            "rmse_std": 0.061,
            "r2_mean": 0.507,
            "r2_std": 0.063,
            "pearson_mean": 0.717,
            "pearson_std": 0.047,
        },
    ]
)


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


# ── Final ranking ──────────────────────────────────────────────────────────────


def load_final_ranking() -> pd.DataFrame | None:
    """Phase 23 final ranked candidates (known + generated)."""
    path = PROJECT_ROOT / "data" / "generated" / "final_ranked_candidates.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def ranking_placement(df: pd.DataFrame) -> dict:
    """Summarise where generated molecules land vs known in a ranked frame."""
    if df is None or df.empty or "source" not in df.columns:
        return {}
    n = len(df)
    gen = df[df["source"] == "generated"]
    known = df[df["source"] == "known"]
    out = {
        "n_total": n,
        "n_known": len(known),
        "n_generated": len(gen),
    }
    if not gen.empty:
        out.update(
            {
                "best_generated_rank": int(gen["rank"].min()),
                "best_generated_cid": gen.iloc[0]["cid"],
                "best_generated_final": float(gen.iloc[0]["final_score"]),
                "median_generated_rank": int(gen["rank"].median()),
                "generated_in_top10": int((gen["rank"] <= 10).sum()),
                "generated_in_top20": int((gen["rank"] <= 20).sum()),
            }
        )
    if not known.empty:
        out["best_known_cid"] = known.iloc[0]["cid"]
        out["best_known_final"] = float(known.iloc[0]["final_score"])
    return out


# ── QSAR model performance ──────────────────────────────────────────────────────


def load_qsar_metrics() -> pd.DataFrame:
    """Single-split test metrics for Models 1 and 2 from their metadata.json."""
    rows = []
    for label, sub in [
        ("Model 1 — general backbone", "general"),
        ("Model 2 — WT-proxy", "wt_proxy"),
    ]:
        meta = _read_json(_QSAR / sub / "metadata.json")
        if meta is None:
            continue
        t = meta.get("test_metrics", {})
        rows.append(
            {
                "model": label,
                "best": meta.get("best_model"),
                "rmse": t.get("rmse"),
                "r2": t.get("r2"),
                "pearson_r": t.get("pearson_r"),
                "n_test": t.get("n"),
            }
        )
    return pd.DataFrame(rows)


def load_seed_stability() -> pd.DataFrame:
    """5-seed scaffold-split stability (mean ± std)."""
    return SEED_STABILITY.copy()


def load_fingerprint_ablation() -> dict[str, pd.DataFrame]:
    """Per-task ({general, wt_proxy}) fingerprint ablation tables, best-model rows."""
    d = _read_json(_QSAR / "fingerprint_ablation_results.json")
    if d is None:
        return {}
    out: dict[str, pd.DataFrame] = {}
    for task, fps in d.get("results", {}).items():
        rows = []
        for fp_name, entry in fps.items():
            best = entry.get("best", {})
            rows.append(
                {
                    "fingerprint": fp_name,
                    "n_feat": entry.get("n_features"),
                    "best_model": entry.get("best_model"),
                    "val_rmse": round(best.get("val_rmse_mean", float("nan")), 4),
                    "test_rmse_mean": round(
                        best.get("test_rmse_mean", float("nan")), 3
                    ),
                    "test_rmse_std": round(best.get("test_rmse_std", float("nan")), 3),
                    "test_r2_mean": round(best.get("test_r2_mean", float("nan")), 3),
                    "spearman_mean": round(
                        best.get("test_spearman_mean", float("nan")), 3
                    ),
                }
            )
        frame = pd.DataFrame(rows).sort_values("val_rmse").reset_index(drop=True)
        out[task] = frame
    return out


def load_model3_verdict() -> dict | None:
    """Model 3 L858R LOOCV summary + verdict (calibration vs backbone)."""
    d = _read_json(_QSAR / "l858r" / "loocv_results.json")
    if d is None:
        return None
    summary = d.get("summary", {})
    table = pd.DataFrame(
        [
            {
                "method": k,
                "spearman_mean": v.get("spearman_mean"),
                "spearman_std": v.get("spearman_std"),
                "rmse_mean": v.get("rmse_mean"),
                "rmse_std": v.get("rmse_std"),
            }
            for k, v in summary.items()
        ]
    )
    return {
        "verdict": d.get("verdict"),
        "n_l858r": d.get("n_l858r"),
        "seeds": d.get("seeds"),
        "table": table,
    }


def load_model4_verdict() -> dict | None:
    """Model 4 derived-selectivity significance summary."""
    d = _read_json(_QSAR / "selectivity" / "selectivity_results.json")
    if d is None:
        return None
    return {
        "n_pairs": d.get("n_pairs"),
        "spearman_derived": d.get("spearman_derived"),
        "pvalue_derived": d.get("pvalue_derived"),
        "stability_note": d.get("stability_note"),
    }


# ── Docking ──────────────────────────────────────────────────────────────────


def load_docking_noise() -> pd.DataFrame | None:
    """Top-15 seed-noise selectivity shortlist with error bars + call."""
    d = _read_json(_QSAR / "docking_noise_results.json")
    if d is None:
        return None
    rows = []
    for c in d.get("compounds", []):
        ns = c.get("noise_stats", {})
        rows.append(
            {
                "cid": c.get("cid"),
                "delta": ns.get("delta"),
                "std_delta": ns.get("std_delta"),
                "mean_l858r": ns.get("mean_l858r"),
                "std_l858r": ns.get("std_l858r"),
                "mean_wt": ns.get("mean_wt"),
                "std_wt": ns.get("std_wt"),
                "call": c.get("call"),
                "warheads": ",".join(c.get("warheads", [])),
            }
        )
    return pd.DataFrame(rows)


def load_sanity_check() -> dict | None:
    """B2 sanity-check docking table (3 known inhibitors) + verdict."""
    d = _read_json(_QSAR / "sanity_check_docking.json")
    if d is None:
        return None
    table = pd.DataFrame(d.get("compounds", []))
    return {
        "verdict": d.get("verdict"),
        "verdict_detail": d.get("verdict_detail"),
        "table": table,
    }


def load_generated_docking() -> pd.DataFrame | None:
    """Post-hoc docking results for generated candidates."""
    d = _read_json(_GEN / "generated_docking_results.json")
    if d is None:
        return None
    rows = [c for c in d.get("compounds", []) if c.get("docking_status") == "ok"]
    return pd.DataFrame(rows)


# ── RL ───────────────────────────────────────────────────────────────────────


def load_rl_results() -> dict | None:
    """RL fine-tuning Run-2 pre/post metrics + verdict."""
    d = _read_json(_GEN / "rl_results.json")
    if d is None:
        return None
    pre, post = d.get("pre_rl", {}), d.get("post_rl", {})
    keys = [
        ("validity", "Validity"),
        ("uniqueness", "Uniqueness"),
        ("scaffold_diversity", "Scaffold diversity"),
        ("mean_pic50", "Mean pred pIC50"),
        ("admet_pass_rate", "ADMET pass rate"),
        ("in_domain_rate", "In-domain rate"),
    ]
    table = pd.DataFrame(
        [
            {
                "metric": label,
                "pre_rl": pre.get(k),
                "post_rl": post.get(k),
                "delta": (
                    None
                    if pre.get(k) is None or post.get(k) is None
                    else round(post.get(k) - pre.get(k), 3)
                ),
            }
            for k, label in keys
        ]
    )
    return {
        "verdict": d.get("comparison", {}).get("verdict"),
        "detail": d.get("comparison", {}).get("detail"),
        "sigma": d.get("sigma"),
        "n_steps": d.get("n_steps"),
        "diversity_filter": d.get("diversity_filter", {}),
        "table": table,
    }
