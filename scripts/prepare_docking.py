"""
Phase B — Step B1: Structure preparation for AutoDock Vina docking.

Downloads 2ITZ (EGFR L858R + gefitinib) and 2ITY (EGFR WT + gefitinib),
prepares them identically (single chain, hydrogens at pH 7.4), writes
receptor PDBQT files, computes the docking box from the IRE centroid in 2ITZ,
and persists all parameters to config/docking_config.yaml.

Does NOT perform any docking — that is Phase B Step B2.

Run:
    PYTHONPATH=. .venv/Scripts/python.exe scripts/prepare_docking.py

Output (data/docking/protein/):
    2ITZ_raw.pdb            2ITY_raw.pdb
    2ITZ_ligand.pdb         2ITY_ligand.pdb
    2ITZ_protein_clean.pdb  2ITY_protein_clean.pdb
    2ITZ_prepared.pdb       2ITY_prepared.pdb
    2ITZ_receptor.pdbqt     2ITY_receptor.pdbqt
    2ITZ_ligand_rigid.pdbqt 2ITY_ligand_rigid.pdbqt

Config updated:
    config/docking_config.yaml  (box.center_x/y/z filled in)
"""

from __future__ import annotations

import yaml

from src.docking.prepare_ligands import ligand_pdb_to_pdbqt_rigid
from src.docking.prepare_protein import prepare_receptor
from src.utils.config import get_project_root
from src.utils.logging import get_logger

logger = get_logger(__name__)

ROOT = get_project_root()
DOCK_DIR = ROOT / "data" / "docking" / "protein"
CFG_PATH = ROOT / "config" / "docking_config.yaml"

# Structure pair — confirmed Yun et al. 2007, same construct EGFR 696-1022
STRUCTURES = [
    {"pdb_id": "2ITZ", "mutation": "L858R"},
    {"pdb_id": "2ITY", "mutation": "wild_type"},
]
CHAIN = "A"
LIGAND_RESNAME = "IRE"  # gefitinib (Iressa) residue name in RCSB PDB
PH = 7.4
BOX_SIZE = 22.5  # Angstrom per axis, covers ATP site with ~5 A buffer


def _update_config(cx: float, cy: float, cz: float) -> None:
    """Write the computed box center back to docking_config.yaml."""
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["box"]["center_x"] = round(cx, 3)
    cfg["box"]["center_y"] = round(cy, 3)
    cfg["box"]["center_z"] = round(cz, 3)

    with open(CFG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    logger.info(f"docking_config.yaml updated: center=({cx:.3f}, {cy:.3f}, {cz:.3f})")


def main() -> None:
    DOCK_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for s in STRUCTURES:
        pid = s["pdb_id"]
        logger.info("=" * 60)
        logger.info(f"Preparing {pid} ({s['mutation']}) ...")
        logger.info("=" * 60)

        result = prepare_receptor(
            pdb_id=pid,
            chain_id=CHAIN,
            ligand_resname=LIGAND_RESNAME,
            out_dir=DOCK_DIR,
            ph=PH,
        )
        results[pid] = result

        # Convert crystal ligand to rigid PDBQT for reference
        ligand_pdbqt = DOCK_DIR / f"{pid}_ligand_rigid.pdbqt"
        ligand_pdb = result["ligand_pdb"]
        if ligand_pdb.exists() and ligand_pdb.stat().st_size > 0:
            try:
                ligand_pdb_to_pdbqt_rigid(ligand_pdb, ligand_pdbqt, resname="IRE")
            except Exception as e:
                logger.warning(f"Ligand PDBQT conversion failed for {pid}: {e}")
        else:
            logger.warning(f"Ligand PDB not found or empty: {ligand_pdb}")

    # Box center from 2ITZ IRE centroid — applied identically to both structures
    cx, cy, cz = results["2ITZ"]["ligand_centroid"]

    logger.info("=" * 60)
    logger.info("DOCKING BOX PARAMETERS")
    logger.info("=" * 60)
    logger.info(
        f"  Center (from 2ITZ IRE centroid): x={cx:.3f}  y={cy:.3f}  z={cz:.3f}"
    )
    logger.info(f"  Size: {BOX_SIZE} x {BOX_SIZE} x {BOX_SIZE} A")
    logger.info("  (identical box applied to both 2ITZ and 2ITY)")

    _update_config(cx, cy, cz)

    # Summary
    logger.info("=" * 60)
    logger.info("B1 PREPARATION COMPLETE")
    logger.info("=" * 60)
    for pid, res in results.items():
        pdbqt = res["receptor_pdbqt"]
        size_kb = pdbqt.stat().st_size // 1024 if pdbqt.exists() else 0
        logger.info(f"  {pid}  receptor={pdbqt.name} ({size_kb} KB)")
    logger.info("  Next step: B2 (docking) -- not run here.")


if __name__ == "__main__":
    main()
