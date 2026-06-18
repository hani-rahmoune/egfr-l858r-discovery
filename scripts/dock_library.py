"""
Phase B2 ranked-library docking: first pass.

Selects the top-50 unique compounds from the EGFR general dataset ranked by
the backbone model (RandomForest, seed=42), docks each into both pockets
(L858R = 2ITZ_receptor.pdbqt, WT = 2ITY_aligned_receptor.pdbqt) with Vina,
and produces an exploratory selectivity ranking.

Covalent warhead detection (SMARTS-based) is applied before docking; flagged
compounds are tagged 'low_confidence' because non-covalent rigid docking
cannot model the covalent bond.

All output is EXPLORATORY:
  - Rigid receptor (no induced-fit).
  - Vina scoring function is a coarse approximation.
  - Selectivity delta (L858R - WT) uses two different crystal structures.
  - n=50 compounds, not a statistically powered set.
  - No ADMET filter applied.

This is a pipeline-validation pass, not a final filtered library.

Run:
  PYTHONPATH=. .venv/Scripts/python.exe scripts/dock_library.py

Runtime: ~50-100 min (50 compounds × 2 pockets, exhaustiveness=8, all CPU cores).
Prerequisites:
  - scripts/train_models.py  (models/qsar/general/ must exist)
  - scripts/sanity_check_docking.py  (aligned 2ITY receptor must exist)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.docking.parse_results import best_affinity
from src.docking.prepare_ligands import smiles_to_pdbqt
from src.docking.vina_runner import run_vina
from src.features.covalent import covalent_confidence, detect_warheads
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

N_CANDIDATES = 50
EXHAUSTIVENESS = 8
N_POSES = 9
SEED = 42

RECEPTORS = {
    "L858R": PROJECT_ROOT / "data" / "docking" / "protein" / "2ITZ_receptor.pdbqt",
    "WT": PROJECT_ROOT / "data" / "docking" / "protein" / "2ITY_aligned_receptor.pdbqt",
}

MODEL_DIR = PROJECT_ROOT / "models" / "qsar" / "general"
PARQUET = PROJECT_ROOT / "data" / "processed" / "features_egfr_general.parquet"
RESULTS_OUT = PROJECT_ROOT / "models" / "qsar" / "library_docking_results.json"


# ── Step 1: select top candidates ─────────────────────────────────────────────


def select_top_candidates(
    parquet_path: Path,
    model_dir: Path,
    n: int = N_CANDIDATES,
) -> pd.DataFrame:
    """
    Rank all molecules in the EGFR general parquet by backbone prediction,
    deduplicate by canonical SMILES (keep highest predicted pIC50), and return
    the top-n as a DataFrame with columns:
      canonical_smiles, pic50, pred_pic50, mutation_flag

    Parameters
    ----------
    parquet_path : path to features_egfr_general.parquet
    model_dir    : directory containing best_model.pkl + metadata.json
    n            : number of unique candidates to select

    Returns
    -------
    DataFrame, shape (n, 4), sorted descending by pred_pic50.
    """
    import joblib

    meta_path = model_dir / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    feat_cols = meta["feature_cols"]

    model = joblib.load(model_dir / "best_model.pkl")
    df = pd.read_parquet(parquet_path)

    X = pd.DataFrame(df[feat_cols].values.astype(np.float32), columns=feat_cols)
    df = df.copy()
    df["pred_pic50"] = model.predict(X)

    # Deduplicate: keep the row with the highest prediction for each SMILES
    df_dedup = df.sort_values("pred_pic50", ascending=False).drop_duplicates(
        "canonical_smiles"
    )
    top = df_dedup.nlargest(n, "pred_pic50")[
        ["canonical_smiles", "pic50", "pred_pic50", "mutation_flag"]
    ].reset_index(drop=True)
    return top


# ── Step 2: prepare ligand PDBQTs ─────────────────────────────────────────────


def prepare_library_ligands(
    candidates: pd.DataFrame,
    ligand_dir: Path,
    seed: int = SEED,
) -> dict[str, Path]:
    """
    Generate flexible PDBQTs for each candidate SMILES.

    Returns {compound_id: pdbqt_path}.  compound_id is 'cmpd_{rank}' (1-indexed).
    Failed conformer generations are logged and skipped.
    """
    ligand_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    failures: list[str] = []

    for idx, row in candidates.iterrows():
        cid = f"cmpd_{idx + 1:03d}"
        smiles = row["canonical_smiles"]
        out = ligand_dir / f"{cid}.pdbqt"

        if out.exists():
            logger.info(f"Ligand PDBQT already exists: {out.name}")
            paths[cid] = out
            continue

        try:
            smiles_to_pdbqt(smiles, out, name=cid, seed=seed)
            paths[cid] = out
        except Exception as exc:
            logger.warning(f"Ligand prep failed for {cid} ({smiles[:50]}): {exc}")
            failures.append(cid)

    if failures:
        logger.warning(f"Ligand prep failures ({len(failures)}): {failures}")
    return paths


# ── Step 3: dock all compounds into both pockets ──────────────────────────────


def dock_library(
    ligand_paths: dict[str, Path],
    box: dict[str, float],
    out_dir: Path,
    exhaustiveness: int = EXHAUSTIVENESS,
    n_poses: int = N_POSES,
    seed: int = SEED,
) -> dict[str, dict[str, Path | None]]:
    """
    Dock each compound into L858R and WT pockets.

    Returns {cid: {"L858R": out_pdbqt_or_None, "WT": out_pdbqt_or_None}}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Path | None]] = {}

    total = len(ligand_paths) * len(RECEPTORS)
    done = 0
    for cid, lig_path in ligand_paths.items():
        results[cid] = {}
        for pocket, rec_path in RECEPTORS.items():
            done += 1
            logger.info(f"[{done}/{total}] Docking {cid} into {pocket} ...")
            try:
                out_pdbqt, _ = run_vina(
                    receptor=rec_path,
                    ligand=lig_path,
                    out_dir=out_dir,
                    box=box,
                    n_poses=n_poses,
                    exhaustiveness=exhaustiveness,
                    seed=seed,
                )
                results[cid][pocket] = out_pdbqt
            except Exception as exc:
                logger.warning(f"  Vina failed: {cid}/{pocket}: {exc}")
                results[cid][pocket] = None

    return results


# ── Step 4: build selectivity ranking ─────────────────────────────────────────


def build_ranking(
    candidates: pd.DataFrame,
    docking_results: dict[str, dict[str, Path | None]],
) -> list[dict]:
    """
    Compute per-compound Vina scores and L858R-minus-WT selectivity delta.

    Returns list of dicts sorted by selectivity_delta (ascending = more L858R
    selective), with failed/missing dockings handled gracefully.

    Fields per row:
      cid, smiles, pred_pic50, pic50, mutation_flag,
      warheads, docking_confidence,
      l858r_score, wt_score, selectivity_delta,
      docking_status ('ok' | 'partial' | 'failed')
    """
    rows = []

    for idx, cand in candidates.iterrows():
        cid = f"cmpd_{idx + 1:03d}"
        smiles = cand["canonical_smiles"]

        warheads = detect_warheads(smiles)
        confidence = covalent_confidence(smiles)

        pocket_results = docking_results.get(cid, {})
        l858r_path = pocket_results.get("L858R")
        wt_path = pocket_results.get("WT")

        l858r_score = best_affinity(l858r_path) if l858r_path else None
        wt_score = best_affinity(wt_path) if wt_path else None

        if l858r_score is not None and wt_score is not None:
            delta = round(l858r_score - wt_score, 3)
            status = "ok"
        elif l858r_score is not None or wt_score is not None:
            delta = None
            status = "partial"
        else:
            delta = None
            status = "failed"

        rows.append(
            {
                "cid": cid,
                "smiles": smiles,
                "pred_pic50": round(float(cand["pred_pic50"]), 3),
                "pic50": round(float(cand["pic50"]), 3),
                "mutation_flag": cand["mutation_flag"],
                "warheads": warheads,
                "docking_confidence": confidence,
                "l858r_score": l858r_score,
                "wt_score": wt_score,
                "selectivity_delta": delta,
                "docking_status": status,
            }
        )

    # Sort by selectivity_delta ascending (most L858R-selective first)
    # Push None deltas to the end
    rows.sort(
        key=lambda r: (r["selectivity_delta"] is None, r["selectivity_delta"] or 0)
    )
    return rows


# ── Step 5: report ─────────────────────────────────────────────────────────────


def report(
    rows: list[dict],
    n_candidates: int,
    n_ligand_failures: int,
) -> dict:
    """Print ranked table and return summary dict."""
    ok_rows = [r for r in rows if r["docking_status"] == "ok"]
    partial_rows = [r for r in rows if r["docking_status"] == "partial"]
    failed_rows = [r for r in rows if r["docking_status"] == "failed"]
    covalent_rows = [r for r in rows if r["docking_confidence"] == "low_confidence"]

    logger.info("=" * 80)
    logger.info("PHASE B2 LIBRARY DOCKING -- first pass (EXPLORATORY, rigid receptor)")
    logger.info(
        f"n_candidates={n_candidates}  ok={len(ok_rows)}  "
        f"partial={len(partial_rows)}  failed={len(failed_rows) + n_ligand_failures}  "
        f"covalent_flagged={len(covalent_rows)}"
    )
    logger.info("-" * 80)
    logger.info(
        f"{'Rank':<5} {'CID':<10} {'L858R':>7} {'WT':>7} {'delta':>7} "
        f"{'pred_pIC50':>11} {'confid.':<16} {'warheads'}"
    )
    logger.info("-" * 80)

    shown = 0
    for rank, row in enumerate(rows, 1):
        if row["docking_status"] != "ok":
            continue
        shown += 1
        delta_str = f"{row['selectivity_delta']:+.3f}"
        wh = ",".join(row["warheads"]) or "—"
        logger.info(
            f"{rank:<5d} {row['cid']:<10} "
            f"{row['l858r_score']:>7.3f} "
            f"{row['wt_score']:>7.3f} "
            f"{delta_str:>7} "
            f"{row['pred_pic50']:>11.3f} "
            f"{row['docking_confidence']:<16} "
            f"{wh}"
        )

    logger.info("=" * 80)

    # Top 5 most L858R-selective (most-negative delta)
    top5_l858r = [r for r in ok_rows if r["selectivity_delta"] is not None][:5]
    if top5_l858r:
        logger.info("Top 5 most L858R-selective (most-negative delta):")
        for r in top5_l858r:
            wh = ",".join(r["warheads"]) or "—"
            logger.info(
                f"  {r['cid']}  delta={r['selectivity_delta']:+.3f}  "
                f"L858R={r['l858r_score']:.3f}  WT={r['wt_score']:.3f}  "
                f"pred_pIC50={r['pred_pic50']:.3f}  conf={r['docking_confidence']}  "
                f"warheads={wh}"
            )

    return {
        "n_candidates": n_candidates,
        "n_ok": len(ok_rows),
        "n_partial": len(partial_rows),
        "n_failed": len(failed_rows) + n_ligand_failures,
        "n_covalent_flagged": len(covalent_rows),
        "compounds": rows,
        "note": (
            "EXPLORATORY. Rigid receptor (2ITZ L858R / 2ITY_aligned WT). "
            "Vina 1.2.7, exhaustiveness=8, seed=42. "
            "Top-50 candidates ranked by general EGFR backbone (RandomForest). "
            "Covalent warhead compounds tagged low_confidence: non-covalent docking "
            "cannot model covalent bond. selectivity_delta = L858R - WT (kcal/mol); "
            "negative = L858R-selective. No ADMET filter applied. Pipeline-validation pass."
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def _load_box() -> dict[str, float]:
    import yaml

    cfg_path = PROJECT_ROOT / "config" / "docking_config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return {
        k: float(cfg["box"][k])
        for k in ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z")
    }


def main() -> None:
    ligand_dir = PROJECT_ROOT / "data" / "docking" / "ligands" / "library"
    dock_dir = PROJECT_ROOT / "data" / "docking" / "results" / "library"
    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Phase B2 library docking — first pass")
    logger.info(f"Selecting top {N_CANDIDATES} candidates from backbone model ...")

    # Step 1 — candidate selection
    candidates = select_top_candidates(PARQUET, MODEL_DIR, n=N_CANDIDATES)
    logger.info(
        f"Selected {len(candidates)} candidates  "
        f"pred_pIC50 range: {candidates['pred_pic50'].min():.3f}–{candidates['pred_pic50'].max():.3f}"
    )
    n_cov_pre = sum(1 for smi in candidates["canonical_smiles"] if detect_warheads(smi))
    logger.info(f"Covalent warhead flags before docking: {n_cov_pre}/{len(candidates)}")

    # Step 2 — ligand preparation
    logger.info("Preparing ligand PDBQTs (meeko ETKDGv3 + MMFF94) ...")
    ligand_paths = prepare_library_ligands(candidates, ligand_dir)
    n_ligand_failures = len(candidates) - len(ligand_paths)

    # Step 3 — docking
    box = _load_box()
    logger.info(
        f"Box: center=({box['center_x']:.3f}, {box['center_y']:.3f}, {box['center_z']:.3f})  "
        f"size={box['size_x']:.1f} A^3"
    )
    logger.info(
        f"Docking {len(ligand_paths)} compounds × 2 pockets "
        f"(exhaustiveness={EXHAUSTIVENESS}, seed={SEED}) ..."
    )
    docking_results = dock_library(ligand_paths, box, dock_dir)

    # Step 4 — ranking
    rows = build_ranking(candidates, docking_results)

    # Step 5 — report
    summary = report(
        rows, n_candidates=len(candidates), n_ligand_failures=n_ligand_failures
    )

    with open(RESULTS_OUT, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved: {RESULTS_OUT}")

    logger.info(
        f"Done. {summary['n_ok']}/{summary['n_candidates']} compounds docked successfully. "
        f"{summary['n_covalent_flagged']} covalent-flagged. "
        f"See {RESULTS_OUT} for full ranking."
    )


if __name__ == "__main__":
    main()
