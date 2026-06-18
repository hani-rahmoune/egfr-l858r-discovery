"""
Phase B3: CNN rescoring of B2 sanity-check poses with GNINA v1.0.

Rescores the best Vina pose for each of the 3 sanity compounds (gefitinib,
erlotinib, osimertinib) in both pockets (L858R = 2ITZ, WT = 2ITY_aligned).

Route: GNINA v1.0 Linux binary via WSL2 Ubuntu (CPU-only; no CUDA required).
``pip install vina`` and the v1.3.2 binary both fail on Windows without
NVIDIA GPU drivers; v1.0 runs cleanly on WSL2 glibc 2.39 without libcuda.

All output is EXPLORATORY. Rigid receptor; CNN trained on PDBbind (not EGFR-
specific); n=3 compounds; CPU scoring without GPU-accelerated ensemble.

Run:
  PYTHONPATH=. .venv/Scripts/python.exe scripts/rescore_sanity_poses.py

Prerequisites:
  - B2 sanity check done (scripts/sanity_check_docking.py already run)
  - WSL2 Ubuntu running
  - data/docking/tools/gnina_v1.0 present (see CLAUDE.md for download)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.docking.gnina_runner import extract_best_pose, rescore_pose
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ── Compound table (must match B2 sanity check) ───────────────────────────────

COMPOUNDS = ["gefitinib", "erlotinib", "osimertinib"]

RECEPTORS = {
    "L858R": PROJECT_ROOT / "data" / "docking" / "protein" / "2ITZ_receptor.pdbqt",
    "WT": PROJECT_ROOT / "data" / "docking" / "protein" / "2ITY_aligned_receptor.pdbqt",
}

VINA_POCKET_STEM = {
    "L858R": "2ITZ_receptor",
    "WT": "2ITY_aligned_receptor",
}

# Known Vina scores from B2 (used only for delta comparison in the report)
VINA_SCORES = {
    "gefitinib": {"L858R": -7.860, "WT": -7.492},
    "erlotinib": {"L858R": -7.666, "WT": -7.263},
    "osimertinib": {"L858R": -7.944, "WT": -7.306},
}

RT_LN10 = 1.3626  # kcal/mol per pKd unit at 300 K (RT ln 10)


# ── Step 1: extract best Vina poses ──────────────────────────────────────────


def step_extract_poses(
    sanity_dir: Path, best_pose_dir: Path
) -> dict[str, dict[str, Path]]:
    """
    Extract MODEL 1 from each Vina multi-MODEL PDBQT.
    Returns {compound: {pocket: path_to_single_pose_pdbqt}}.
    """
    best_pose_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, dict[str, Path]] = {}

    for compound in COMPOUNDS:
        paths[compound] = {}
        for pocket, stem in VINA_POCKET_STEM.items():
            vina_out = sanity_dir / f"{compound}__{stem}_out.pdbqt"
            best_out = best_pose_dir / f"{compound}__{pocket}_best.pdbqt"
            if best_out.exists():
                logger.info(f"Best pose already extracted: {best_out.name}")
            else:
                extract_best_pose(vina_out, best_out)
                logger.info(f"Extracted: {best_out.name}")
            paths[compound][pocket] = best_out

    return paths


# ── Step 2: CNN rescoring ─────────────────────────────────────────────────────


def step_rescore(
    pose_paths: dict[str, dict[str, Path]],
    gnina_out_dir: Path,
) -> dict[str, dict[str, dict]]:
    """
    Run GNINA --score_only on each (compound, pocket) pair.
    Returns {compound: {pocket: cnn_scores_dict}}.
    """
    all_scores: dict[str, dict[str, dict]] = {}

    for compound in COMPOUNDS:
        all_scores[compound] = {}
        for pocket, rec_path in RECEPTORS.items():
            best_pose = pose_paths[compound][pocket]
            scores = rescore_pose(
                receptor=rec_path,
                best_pose_pdbqt=best_pose,
                out_dir=gnina_out_dir,
            )
            all_scores[compound][pocket] = scores

    return all_scores


# ── Step 3: report and verdict ────────────────────────────────────────────────


def report_and_verdict(cnn_scores: dict[str, dict[str, dict]]) -> dict:
    """
    Build per-compound rows, compute deltas, and issue a validation verdict.

    Validation criteria (from Phase B3 task spec):
      - CNN must favour L858R (delta_cnn_affinity > 0) for all 3 compounds.
      - If all pass: VALIDATED (use CNN as supplementary scorer alongside Vina).
      - If any fail: NOT_VALIDATED (keep Vina-only as coarse filter).

    Note: CNNaffinity is in pKd units.  To convert to kcal/mol: pKd × RT ln 10
    (= pKd × 1.363 at 300 K).
    """
    rows = []
    for compound in COMPOUNDS:
        l = cnn_scores[compound]["L858R"]
        w = cnn_scores[compound]["WT"]
        vina_l = VINA_SCORES[compound]["L858R"]
        vina_w = VINA_SCORES[compound]["WT"]

        delta_cnn_aff = round(l["cnn_affinity_pkd"] - w["cnn_affinity_pkd"], 4)
        delta_cnn_score = round(l["cnn_score"] - w["cnn_score"], 4)
        delta_vina = round(vina_l - vina_w, 3)

        # Convert delta_cnn_aff to kcal/mol for direct comparison with Vina
        delta_cnn_kcal = round(delta_cnn_aff * RT_LN10, 3)

        direction = (
            "L858R_favoured"
            if delta_cnn_aff > 0
            else "WT_favoured" if delta_cnn_aff < 0 else "tied"
        )

        rows.append(
            {
                "compound": compound,
                "L858R": {
                    "affinity_kcal": l["affinity_kcal"],
                    "cnn_score": l["cnn_score"],
                    "cnn_affinity_pkd": l["cnn_affinity_pkd"],
                    "cnn_variance": l["cnn_variance"],
                },
                "WT": {
                    "affinity_kcal": w["affinity_kcal"],
                    "cnn_score": w["cnn_score"],
                    "cnn_affinity_pkd": w["cnn_affinity_pkd"],
                    "cnn_variance": w["cnn_variance"],
                },
                "delta_cnn_affinity_pkd": delta_cnn_aff,
                "delta_cnn_affinity_kcal": delta_cnn_kcal,
                "delta_cnn_score": delta_cnn_score,
                "delta_vina_kcal": delta_vina,
                "direction_cnn_affinity": direction,
            }
        )

    # Print table
    logger.info("=" * 72)
    logger.info("PHASE B3 -- CNN rescoring (EXPLORATORY)")
    logger.info(
        f"{'Compound':<14} {'L858R_cnn':>10} {'WT_cnn':>8} {'D_pkd':>7} "
        f"{'D_kcal(cnn)':>12} {'D_kcal(vina)':>13}  direction"
    )
    logger.info("-" * 72)
    for r in rows:
        logger.info(
            f"{r['compound']:<14} "
            f"{r['L858R']['cnn_affinity_pkd']:>10.3f} "
            f"{r['WT']['cnn_affinity_pkd']:>8.3f} "
            f"{r['delta_cnn_affinity_pkd']:>7.3f} "
            f"{r['delta_cnn_affinity_kcal']:>12.3f} "
            f"{r['delta_vina_kcal']:>13.3f}  "
            f"{r['direction_cnn_affinity']}"
        )
    logger.info("=" * 72)

    # Verdict: all three must show L858R preference on CNNaffinity
    n_pass = sum(1 for r in rows if r["direction_cnn_affinity"] == "L858R_favoured")

    if n_pass == 3:
        verdict = "VALIDATED"
        verdict_detail = (
            "CNN favours L858R on all 3 compounds (direction criterion met). "
            "Use CNNaffinity as supplementary score alongside Vina."
        )
    elif n_pass >= 2:
        verdict = "BORDERLINE"
        failing = [
            r["compound"]
            for r in rows
            if r["direction_cnn_affinity"] != "L858R_favoured"
        ]
        verdict_detail = (
            f"CNN favours L858R on {n_pass}/3 compounds. "
            f"Fails on: {', '.join(failing)} (delta near zero). "
            f"Direction criterion NOT fully met. Treat CNNaffinity as unreliable; keep Vina-only."
        )
    else:
        verdict = "NOT_VALIDATED"
        verdict_detail = (
            f"CNN favours L858R on only {n_pass}/3 compounds. "
            "Direction criterion not met. Vina-only scoring remains the coarse filter."
        )

    logger.info(f"VERDICT: {verdict}")
    logger.info(f"DETAIL:  {verdict_detail}")

    return {
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "compounds": rows,
        "note": (
            "EXPLORATORY. GNINA v1.0 CPU (WSL2), no GPU. CNN not EGFR-specific. "
            "n=3 compounds. Best Vina pose rescored (no re-docking). "
            "CNNaffinity in pKd; multiply by 1.363 kcal/mol per pKd unit for kcal/mol."
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    sanity_dir = PROJECT_ROOT / "data" / "docking" / "results" / "sanity"
    best_pose_dir = sanity_dir / "best_poses"
    gnina_out_dir = PROJECT_ROOT / "data" / "docking" / "results" / "gnina_rescore"
    results_out = PROJECT_ROOT / "models" / "qsar" / "gnina_rescore_sanity.json"

    logger.info("Phase B3: GNINA CNN rescoring of B2 sanity poses")
    logger.info("Route: WSL2 Ubuntu / gnina v1.0 (CPU-only)")

    # Step 1 -- extract best poses
    logger.info("Step 1: extracting best Vina poses ...")
    pose_paths = step_extract_poses(sanity_dir, best_pose_dir)

    # Step 2 -- CNN rescore
    logger.info("Step 2: GNINA rescoring (6 runs, ~30 s each on CPU) ...")
    cnn_scores = step_rescore(pose_paths, gnina_out_dir)

    # Step 3 -- report
    summary = report_and_verdict(cnn_scores)

    # Save
    results_out.parent.mkdir(parents=True, exist_ok=True)
    with open(results_out, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved: {results_out}")

    if summary["verdict"] == "NOT_VALIDATED":
        logger.warning("CNN rescoring not validated. Proceed with Vina-only scoring.")


if __name__ == "__main__":
    main()
