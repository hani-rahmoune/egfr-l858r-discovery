"""
Phase B2 sanity check: dock gefitinib, erlotinib, and osimertinib into both
EGFR pockets (L858R = 2ITZ, WT = 2ITY) and verify the directional preference.

Literature anchor (Yun et al. 2007 Cancer Cell PMID:17376547):
  gefitinib binds L858R roughly 20-fold tighter than WT (delta Kd ~ 1.3 kcal/mol).
  Docking should show a more-negative (better) score for the L858R pocket.

Pipeline:
  1. Align 2ITY onto 2ITZ on Ca atoms; write 2ITY_aligned_receptor.pdbqt
  2. Prepare sanity-check ligands (meeko, ETKDGv3 + MMFF94)
  3. Dock 3 ligands x 2 pockets (6 Vina runs, seed=42, exhaustiveness=8)
  4. Report affinity table and L858R-minus-WT delta per compound
  5. Sanity-check verdict: gefitinib delta must be negative (L858R preferred)

All output is EXPLORATORY. Rigid receptor; scoring-function approximation of
binding free energy; n=3 compounds.

Run:
  PYTHONPATH=. .venv/Scripts/python.exe scripts/sanity_check_docking.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.docking.align_structures import align_wt_to_l858r, verify_box_coverage
from src.docking.parse_results import best_affinity
from src.docking.prepare_ligands import smiles_to_pdbqt
from src.docking.prepare_protein import write_receptor_pdbqt
from src.docking.vina_runner import run_vina
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ── Sanity-check compound library ─────────────────────────────────────────────

SANITY_COMPOUNDS = {
    "gefitinib": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "erlotinib": "C#Cc1cccc(Nc2ncnc3cc(OCC)c(OCC)cc23)c1",
    "osimertinib": "COc1cc2c(Nc3cccc(NC(=O)/C=C/CN(C)C)c3)ncnc2cc1NC(C)=O",
}


# ── Helper: load box from docking_config.yaml ─────────────────────────────────


def _load_box() -> dict[str, float]:
    cfg_path = PROJECT_ROOT / "config" / "docking_config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return {
        k: float(cfg["box"][k])
        for k in ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z")
    }


# ── Step 1: align 2ITY onto 2ITZ ─────────────────────────────────────────────


def step_align(protein_dir: Path) -> tuple[Path, Path]:
    """
    Superpose 2ITY_prepared.pdb onto 2ITZ_prepared.pdb, write the aligned
    PDB and its receptor PDBQT.  Returns (aligned_pdb, aligned_pdbqt).
    """
    l858r_pdb = protein_dir / "2ITZ_prepared.pdb"
    wt_pdb = protein_dir / "2ITY_prepared.pdb"
    aligned_pdb = protein_dir / "2ITY_aligned.pdb"
    aligned_pdbqt = protein_dir / "2ITY_aligned_receptor.pdbqt"

    if aligned_pdbqt.exists():
        logger.info("Aligned 2ITY PDBQT already exists, skipping alignment.")
        return aligned_pdb, aligned_pdbqt

    rmsd = align_wt_to_l858r(l858r_pdb, wt_pdb, aligned_pdb)
    logger.info(f"Ca alignment RMSD: {rmsd:.3f} A")

    write_receptor_pdbqt(aligned_pdb, aligned_pdbqt)
    return aligned_pdb, aligned_pdbqt


# ── Step 2: prepare sanity-check ligand PDBQTs ────────────────────────────────


def step_prepare_ligands(ligand_dir: Path) -> dict[str, Path]:
    """Generate flexible PDBQTs for gefitinib, erlotinib, osimertinib."""
    ligand_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, smiles in SANITY_COMPOUNDS.items():
        out = ligand_dir / f"{name}.pdbqt"
        if out.exists():
            logger.info(f"Ligand PDBQT already exists: {out.name}")
        else:
            smiles_to_pdbqt(smiles, out, name=name)
        paths[name] = out
    return paths


# ── Step 3: run docking ───────────────────────────────────────────────────────


def step_dock(
    ligand_paths: dict[str, Path],
    receptors: dict[str, Path],
    box: dict[str, float],
    out_dir: Path,
) -> dict[str, dict[str, Path]]:
    """
    Dock each ligand into each receptor.
    Returns {compound: {pocket_label: out_pdbqt}}.
    """
    results: dict[str, dict[str, Path]] = {n: {} for n in ligand_paths}

    for compound, lig_path in ligand_paths.items():
        for pocket_label, rec_path in receptors.items():
            out_pdbqt, log_file = run_vina(
                receptor=rec_path,
                ligand=lig_path,
                out_dir=out_dir,
                box=box,
                exhaustiveness=8,
                seed=42,
            )
            results[compound][pocket_label] = out_pdbqt
            logger.info(f"  {compound} / {pocket_label}: {out_pdbqt.name}")

    return results


# ── Step 4: report and verdict ────────────────────────────────────────────────


def report_and_verdict(docking_results: dict[str, dict[str, Path]]) -> dict:
    """
    Extract best affinities, compute L858R-minus-WT delta, and assess the
    literature anchor (gefitinib must prefer L858R).
    """
    rows = []
    for compound, pockets in docking_results.items():
        l858r_score = best_affinity(pockets["L858R"]) if "L858R" in pockets else None
        wt_score = best_affinity(pockets["WT"]) if "WT" in pockets else None
        delta = None
        if l858r_score is not None and wt_score is not None:
            delta = round(l858r_score - wt_score, 3)
        rows.append(
            {
                "compound": compound,
                "L858R_score": l858r_score,
                "WT_score": wt_score,
                "delta": delta,
            }
        )

    # Print table
    logger.info("=" * 60)
    logger.info("SANITY CHECK -- docking affinities (kcal/mol, EXPLORATORY)")
    logger.info(f"{'Compound':<16} {'L858R':>8} {'WT':>8} {'delta':>8}  interpretation")
    logger.info("-" * 60)
    for r in rows:
        interp = ""
        if r["delta"] is not None:
            if r["delta"] < 0:
                interp = "L858R favoured (correct direction)"
            else:
                interp = "WT favoured (unexpected)"
        logger.info(
            f"{r['compound']:<16} {str(r['L858R_score']):>8} "
            f"{str(r['WT_score']):>8} {str(r['delta']):>8}  {interp}"
        )
    logger.info("=" * 60)

    # Verdict: gefitinib delta must be negative (L858R preferred)
    gef_row = next((r for r in rows if r["compound"] == "gefitinib"), None)
    if gef_row is None or gef_row["delta"] is None:
        verdict = "INCONCLUSIVE"
        verdict_detail = "Gefitinib docking failed; cannot assess literature anchor."
    elif gef_row["delta"] < 0:
        verdict = "PASS"
        verdict_detail = (
            f"Gefitinib favours L858R (delta={gef_row['delta']:.3f} kcal/mol < 0). "
            f"Directionally consistent with Yun et al. 2007."
        )
    else:
        verdict = "FAIL"
        verdict_detail = (
            f"Gefitinib does NOT favour L858R (delta={gef_row['delta']:.3f} >= 0). "
            f"Pipeline not trustworthy for selectivity scoring. "
            f"Check box placement, receptor prep, and aligned PDBQT."
        )

    logger.info(f"VERDICT: {verdict} -- {verdict_detail}")

    return {
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "compounds": rows,
        "note": "EXPLORATORY. Rigid receptor. n=3 compounds. Scoring-function approximation.",
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    protein_dir = PROJECT_ROOT / "data" / "docking" / "protein"
    ligand_dir = PROJECT_ROOT / "data" / "docking" / "ligands" / "sanity"
    dock_dir = PROJECT_ROOT / "data" / "docking" / "results" / "sanity"
    results_out = PROJECT_ROOT / "models" / "qsar" / "sanity_check_docking.json"

    box = _load_box()
    logger.info(
        f"Box: center=({box['center_x']}, {box['center_y']}, {box['center_z']})  "
        f"size={box['size_x']}x{box['size_y']}x{box['size_z']} A"
    )

    # Step 1 -- align WT receptor
    logger.info("Step 1: aligning 2ITY onto 2ITZ ...")
    aligned_pdb, aligned_pdbqt = step_align(protein_dir)

    # Verify box coverage for both receptors
    l858r_pdbqt = protein_dir / "2ITZ_receptor.pdbqt"
    for label, pdb in [
        ("2ITZ (L858R)", protein_dir / "2ITZ_prepared.pdb"),
        ("2ITY aligned (WT)", aligned_pdb),
    ]:
        ok = verify_box_coverage(pdb, box)
        logger.info(
            f"Box coverage {label}: {'OK' if ok else 'WARN -- outside extended box'}"
        )

    # Step 2 -- prepare sanity-check ligands
    logger.info("Step 2: preparing sanity-check ligands ...")
    ligand_paths = step_prepare_ligands(ligand_dir)

    # Step 3 -- dock
    receptors = {
        "L858R": l858r_pdbqt,
        "WT": aligned_pdbqt,
    }
    logger.info("Step 3: running 6 Vina dockings (3 ligands x 2 pockets) ...")
    docking_results = step_dock(ligand_paths, receptors, box, dock_dir)

    # Step 4 -- report and verdict
    logger.info("Step 4: computing affinities and verdict ...")
    summary = report_and_verdict(docking_results)

    # Save JSON
    results_out.parent.mkdir(parents=True, exist_ok=True)
    with open(results_out, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved: {results_out}")

    if summary["verdict"] == "FAIL":
        logger.error(
            "SANITY CHECK FAILED. Do not proceed to ranked-library docking. "
            "Investigate box placement and receptor preparation."
        )
        sys.exit(1)
    elif summary["verdict"] == "PASS":
        logger.info(
            "Sanity check passed. Proceed to ranked-library docking (Phase B2 step 4)."
        )


if __name__ == "__main__":
    main()
