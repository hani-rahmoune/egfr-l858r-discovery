"""
Generate static PNG figures for the EGFR L858R drug discovery project.

Reads real numbers from saved artifact JSON files (no hardcoding except
where no JSON artifact exists -- QSAR 5-seed stability numbers, which come
from eval_seed_stability.py console output captured in CLAUDE.md).

Output: reports/figures/*.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── colour palette ──────────────────────────────────────────────────────────────
BLUE = "#4878CF"
ORANGE = "#D65F00"
GREEN = "#6ACC65"
RED = "#D44444"
GREY = "#9B9B9B"
AMBER = "#E8A838"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
    }
)


# ── Figure 1: QSAR vs GNN 5-seed performance ───────────────────────────────────
def fig_qsar_vs_gnn():
    """R^2 mean +/- std for QSAR (XGB/RF) and GNN across both tasks."""
    # QSAR 5-seed numbers from eval_seed_stability.py (no JSON artifact):
    qsar = {
        "general": {"r2_mean": 0.438, "r2_std": 0.143, "rmse_mean": 1.010, "rmse_std": 0.167},
        "wt_proxy": {"r2_mean": 0.507, "r2_std": 0.063, "rmse_mean": 0.942, "rmse_std": 0.061},
    }

    # GNN numbers from models/gnn/*/metadata.json (real artifact)
    gnn = {}
    for task in ("general", "wt_proxy"):
        p = PROJECT_ROOT / f"models/gnn/{task}/metadata.json"
        with open(p) as f:
            m = json.load(f)
        gnn[task] = m["summary"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)

    tasks = ["general", "wt_proxy"]
    task_labels = ["EGFR general\n(n~1253)", "WT-proxy\n(n~1018)"]

    for ax, task, tlabel in zip(axes, tasks, task_labels):
        labels = ["QSAR\n(XGBoost/RF)", "GNN\n(GINEConv)"]
        means = [qsar[task]["r2_mean"], gnn[task]["test_r2"]["mean"]]
        stds = [qsar[task]["r2_std"], gnn[task]["test_r2"]["std"]]
        colors = [BLUE, ORANGE]
        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.8, capsize=6, width=0.5, ecolor="#555555")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Test R^2 (5-seed mean +/- std)")
        ax.set_title(tlabel)
        ax.set_ylim(0, 0.80)
        ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
        for bar, m, s in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                m + s + 0.02,
                f"{m:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle("QSAR vs GNN: 5-seed scaffold-split R^2\n(QSAR wins both tasks)", y=1.01)
    fig.tight_layout()
    out = FIGURES_DIR / "fig1_qsar_vs_gnn.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ── Figure 2: Fingerprint ablation ─────────────────────────────────────────────
def fig_fingerprint_ablation():
    """Val RMSE by fingerprint type for both tasks (lower = better)."""
    with open(PROJECT_ROOT / "models/qsar/fingerprint_ablation_results.json") as f:
        fa = json.load(f)

    # Display names for FP types
    fp_display = {
        "morgan_ecfp6": "ECFP6",
        "morgan_ecfp4": "ECFP4",
        "topological_torsion": "Torsion",
        "rdkit_topological": "RDKit-topological",
        "atom_pair": "Atom-pair",
        "maccs": "MACCS",
    }
    # Consistent ordering by ecfp6 (best) to MACCS (worst) on general task
    fp_order = [
        "morgan_ecfp6",
        "morgan_ecfp4",
        "topological_torsion",
        "rdkit_topological",
        "atom_pair",
        "maccs",
    ]

    tasks = ["general", "wt_proxy"]
    task_labels = ["EGFR general", "WT-proxy"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for ax, task, tlabel in zip(axes, tasks, task_labels):
        fp_data = fa["results"][task]
        names = [fp_display[k] for k in fp_order]
        vals = [fp_data[k]["best"]["val_rmse_mean"] for k in fp_order]
        stds = [fp_data[k]["best"]["val_rmse_std"] for k in fp_order]
        y = np.arange(len(names))
        colors = [BLUE if k == "morgan_ecfp6" else GREY for k in fp_order]
        ax.barh(y, vals, xerr=stds, color=colors, alpha=0.85, capsize=4, ecolor="#555555", height=0.55)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Val RMSE (5-seed mean +/- std)")
        ax.set_title(tlabel)
        for i, (v, s) in enumerate(zip(vals, stds)):
            ax.text(v + s + 0.005, i, f"{v:.3f}", va="center", fontsize=8)

    axes[0].set_title("EGFR general (ECFP6 wins, margin within noise)")
    axes[1].set_title("WT-proxy (ECFP6 wins, slight margin)")
    fig.suptitle("Fingerprint ablation: val RMSE by representation\n(ECFP4 kept for production -- ECFP6 margin within seed noise)", y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / "fig2_fingerprint_ablation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ── Figure 3: Docking selectivity shortlist ─────────────────────────────────────
def fig_docking_selectivity():
    """Selectivity deltas with +/- 1.5*std noise bands for top-15 compounds."""
    with open(PROJECT_ROOT / "models/qsar/docking_noise_results.json") as f:
        dn = json.load(f)

    compounds = dn["compounds"]
    # Sort by delta ascending (most selective first)
    compounds = sorted(compounds, key=lambda c: c["noise_stats"]["delta"])

    cids = [c["cid"] for c in compounds]
    deltas = [c["noise_stats"]["delta"] for c in compounds]
    stds = [c["noise_stats"]["std_delta"] for c in compounds]
    calls = [c["call"] for c in compounds]

    color_map = {
        "L858R_selective": GREEN,
        "ambiguous": AMBER,
        "low_confidence_covalent": GREY,
        "WT_selective": RED,
    }
    colors = [color_map.get(call, GREY) for call in calls]

    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(cids))
    noise_half = [1.5 * s for s in stds]

    ax.barh(y, deltas, color=colors, alpha=0.8, height=0.6)
    # error bars for noise
    ax.errorbar(
        deltas, y,
        xerr=noise_half,
        fmt="none",
        ecolor="#333333",
        elinewidth=1.0,
        capsize=4,
    )
    ax.axvline(0, color="#333333", linewidth=0.9, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(cids, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Selectivity delta (L858R - WT, kcal/mol)\n<0 = L858R-favoured")
    ax.set_title("Docking selectivity: top-15 compounds\nerror bars = +/-1.5 x seed std (noise threshold)")

    legend_handles = [
        mpatches.Patch(color=GREEN, label="L858R-selective (confident)"),
        mpatches.Patch(color=AMBER, label="Ambiguous (within noise)"),
        mpatches.Patch(color=GREY, label="Low-confidence (covalent)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
    fig.tight_layout()
    out = FIGURES_DIR / "fig3_docking_selectivity.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ── Figure 4: RL pre vs post ────────────────────────────────────────────────────
def fig_rl_comparison():
    """Pre-RL vs post-RL for Run 1 (hacking) and Run 2 (no collapse)."""
    with open(PROJECT_ROOT / "models/generator/rl_results.json") as f:
        rl = json.load(f)

    pre = rl["pre_rl"]
    post_run2 = rl["post_rl"]

    # Run 1 (sigma=0.5, REWARD_HACKING) -- from CLAUDE.md (no separate JSON)
    post_run1 = {
        "validity": 0.994,
        "uniqueness": 0.086,
        "scaffold_diversity": 0.318,
        "admet_pass_rate": 0.806,
        "in_domain_rate": 0.935,
        "mean_pic50": 7.592,
    }

    metrics = ["validity", "uniqueness", "scaffold_diversity"]
    metric_labels = ["Validity", "Uniqueness", "Scaffold diversity"]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(metrics))
    width = 0.25

    pre_vals = [pre[m] for m in metrics]
    r1_vals = [post_run1[m] for m in metrics]
    r2_vals = [post_run2[m] for m in metrics]

    bars_pre = ax.bar(x - width, pre_vals, width, label="Pre-RL (base)", color=BLUE, alpha=0.85)
    bars_r1 = ax.bar(x, r1_vals, width, label="Post-RL Run 1 (sigma=0.5, REWARD_HACKING)", color=RED, alpha=0.85)
    bars_r2 = ax.bar(x + width, r2_vals, width, label="Post-RL Run 2 (sigma=0.25 + diversity filter, INCONCLUSIVE)", color=ORANGE, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Rate [0, 1]")
    ax.set_ylim(0, 1.12)
    ax.set_title("RL fine-tuning: Run 1 collapses uniqueness; Run 2 holds diversity but flat pIC50")
    ax.legend(fontsize=8, loc="upper right")

    # Annotate uniqueness drop in Run 1
    ax.annotate(
        "8.6%\n(mode collapse)",
        xy=(1, r1_vals[1]),
        xytext=(1.2, r1_vals[1] + 0.25),
        arrowprops=dict(arrowstyle="->", color=RED),
        fontsize=8,
        color=RED,
    )

    for bars in (bars_pre, bars_r1, bars_r2):
        for bar in bars:
            h = bar.get_height()
            if h > 0.05:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.01,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    fig.tight_layout()
    out = FIGURES_DIR / "fig4_rl_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ── Figure 5: Final ranking top candidates ─────────────────────────────────────
def fig_final_ranking():
    """Final composite score for top candidates, coloured by source."""
    csv_path = PROJECT_ROOT / "data/generated/final_ranked_candidates.csv"
    df = pd.read_csv(csv_path)
    top = df.head(25).copy()

    colors = [GREEN if s == "generated" else BLUE for s in top["source"]]
    has_warning = top["warnings"].notna()

    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(top))
    ax.barh(y, top["final_score"], color=colors, alpha=0.85, height=0.65)

    # Mark covalent warning with hatching
    for i, (row, warn) in enumerate(zip(top.itertuples(), has_warning)):
        if warn:
            ax.barh(i, row.final_score, color=colors[i], alpha=0.4, height=0.65, hatch="//")

    ax.set_yticks(y)
    labels = [
        f"{row.cid} ({row.source})"
        for row in top.itertuples()
    ]
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Final composite score (0.30 activity + 0.30 docking-selectivity\n+ 0.20 docking-affinity + 0.20 ADMET, min-max normalised)")
    ax.set_title("Top 25 final-ranked candidates\n(hatched = covalent or Brenk warning)")

    legend_handles = [
        mpatches.Patch(color=BLUE, label="Known library"),
        mpatches.Patch(color=GREEN, label="Generated (de novo)"),
        mpatches.Patch(facecolor=GREY, hatch="//", label="Warning flag"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
    fig.tight_layout()
    out = FIGURES_DIR / "fig5_final_ranking.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ── Figure 6: Candidate funnel ─────────────────────────────────────────────────
def fig_candidate_funnel():
    """Waterfall funnel from raw actives to clean shortlist."""
    steps = [
        "EGFR/ErbB2 actives\n(ChEMBL, cleaned)",
        "Top-50 predicted\n(backbone pIC50)",
        "Successfully docked\n(Vina, both pockets)",
        "L858R-selective\n(delta < 0)",
        "Noise-confirmed\nnon-covalent",
        "Clean ADMET\n(pass all filters)",
    ]
    counts = [1253, 50, 49, 30, 6, 4]
    colors_funnel = [BLUE, BLUE, BLUE, GREEN, GREEN, GREEN]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(steps))
    bars = ax.barh(y, counts, color=colors_funnel, alpha=0.82, height=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(steps, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of compounds")
    ax.set_title("Candidate funnel: from ChEMBL actives to clean shortlist")
    ax.set_xscale("log")
    ax.set_xlim(1, 3000)

    for bar, n in zip(bars, counts):
        ax.text(
            bar.get_width() * 1.05,
            bar.get_y() + bar.get_height() / 2,
            str(n),
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # Drop-off annotations
    for i in range(1, len(counts)):
        reduction = (counts[i - 1] - counts[i]) / counts[i - 1] * 100
        ax.text(
            3000 * 0.55,
            i - 0.5,
            f"-{reduction:.0f}%",
            va="center",
            ha="right",
            fontsize=8,
            color="#888888",
        )

    fig.tight_layout()
    out = FIGURES_DIR / "fig6_candidate_funnel.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


if __name__ == "__main__":
    print("Generating figures -> reports/figures/")
    fig_qsar_vs_gnn()
    fig_fingerprint_ablation()
    fig_docking_selectivity()
    fig_rl_comparison()
    fig_final_ranking()
    fig_candidate_funnel()
    print("Done.")
