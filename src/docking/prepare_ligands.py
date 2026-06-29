"""
Ligand preparation utilities for AutoDock Vina docking.

Provides:
- ligand_pdb_to_pdbqt_rigid: converts a ligand PDB (from crystal structure or
  3D generation) to a rigid PDBQT using RDKit Gasteiger charges.  Rigid means
  no ROOT/BRANCH torsion records — the ligand is treated as a rigid body.
  For full flexible docking, add torsion records with meeko or openbabel.
- centroid_from_heavy_atoms: computes the geometric centroid of heavy atoms,
  used to define the docking box center.

Notes:
- The co-crystal gefitinib (IRE) is extracted by prepare_protein.prepare_receptor
  and written as a plain PDB.  ligand_pdb_to_pdbqt_rigid converts it to PDBQT
  for reference (not the primary docking ligand).
- The primary docking ligands (compound library) will be prepared separately.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ── Centroid utility ──────────────────────────────────────────────────────────


def centroid_from_heavy_atoms(
    coords: list[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Return the geometric centroid (mean xyz) of a list of (x, y, z) tuples."""
    if not coords:
        raise ValueError("coords must be non-empty")
    arr = np.array(coords, dtype=float)
    cx, cy, cz = arr.mean(axis=0)
    return float(cx), float(cy), float(cz)


# ── PDBQT line formatter (ligand variant with Gasteiger charge) ───────────────


def _ligand_pdbqt_line(
    serial: int,
    atname: str,
    resname: str,
    x: float,
    y: float,
    z: float,
    charge: float,
    atype: str,
) -> str:
    """Format a single PDBQT HETATM record for a ligand atom."""
    an = atname.strip()
    name_field = f" {an:<3s}" if len(an) < 4 else f"{an:<4s}"
    return (
        f"{'HETATM':<6s}{serial:5d} {name_field} "
        f"{resname:>3s}  {'L':1s}{'1':>4s}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{'1.00':>6s}{'0.00':>6s}    "
        f"{charge:6.3f} {atype:<2s}"
    )


# ── Element → AutoDock type for ligand atoms ──────────────────────────────────


def _ligand_autodock_type(atomic_num: int) -> str:
    """Map RDKit atomic number to AutoDock atom type for common ligand elements."""
    _MAP = {
        1: "HD",  # H (all ligand H are potential polar)
        6: "C",  # C (aromaticity handled below)
        7: "NA",  # N (all ligand N have lone pairs)
        8: "OA",  # O
        9: "F",
        15: "P",
        16: "SA",  # S
        17: "Cl",
        35: "Br",
        53: "I",
    }
    return _MAP.get(atomic_num, "C")  # fallback to C


# ── Rigid PDBQT conversion ────────────────────────────────────────────────────


def ligand_pdb_to_pdbqt_rigid(
    input_pdb: Path,
    output_pdbqt: Path,
    resname: str = "LIG",
) -> Path:
    """
    Convert a ligand PDB file to a rigid PDBQT using RDKit Gasteiger charges.

    Rigid = no ROOT/BRANCH torsion records.  Suitable for:
    - Reference co-crystal ligand (box definition)
    - Rigid docking (all bonds fixed)

    For flexible docking of compound library, add torsion records with meeko.

    Parameters
    ----------
    input_pdb
        Path to the input PDB file (single ligand).
    output_pdbqt
        Path for the output PDBQT file.
    resname
        Residue name to write in the PDBQT.

    Returns
    -------
    Path to the written PDBQT file.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromPDBFile(str(input_pdb), removeHs=False, sanitize=True)
    if mol is None:
        # Try without hydrogen enforcement
        mol = Chem.MolFromPDBFile(str(input_pdb), removeHs=True, sanitize=False)
        if mol is None:
            raise ValueError(f"RDKit could not parse ligand PDB: {input_pdb}")
        mol = Chem.AddHs(mol, addCoords=True)

    # Compute Gasteiger partial charges
    AllChem.ComputeGasteigerCharges(mol)

    conf = mol.GetConformer()
    lines = [
        "REMARK  Rigid PDBQT prepared by prepare_ligands.py",
        "REMARK  Gasteiger partial charges; no torsion records (rigid body)",
        "ROOT",
    ]
    serial = 1
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        x, y, z = pos.x, pos.y, pos.z
        charge = float(atom.GetDoubleProp("_GasteigerCharge") or 0.0)
        if np.isnan(charge) or np.isinf(charge):
            charge = 0.0

        atype = _ligand_autodock_type(atom.GetAtomicNum())
        # Aromatic carbons
        if atom.GetAtomicNum() == 6 and atom.GetIsAromatic():
            atype = "A"
        # Non-polar H → type "H"
        if atom.GetAtomicNum() == 1:
            nbr_elements = [n.GetAtomicNum() for n in atom.GetNeighbors()]
            if all(n == 6 for n in nbr_elements):
                atype = "H"

        atname = atom.GetSymbol() + str(atom.GetIdx() + 1)
        line = _ligand_pdbqt_line(serial, atname, resname, x, y, z, charge, atype)
        lines.append(line)
        serial += 1

    lines.append("ENDROOT")
    lines.append("TORSDOF 0")

    output_pdbqt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(
        f"Rigid ligand PDBQT saved: {output_pdbqt}  ({serial - 1} atoms, TORSDOF=0)"
    )
    return output_pdbqt


# ── SMILES → flexible PDBQT (meeko) ──────────────────────────────────────────


def smiles_to_pdbqt(
    smiles: str,
    output_pdbqt: Path,
    name: str = "LIG",
    seed: int = 42,
) -> Path:
    """
    Generate a 3D conformer from SMILES and write a flexible PDBQT using meeko.

    Uses RDKit ETKDGv3 + MMFF94 optimisation for conformer generation, then
    meeko MoleculePreparation for torsion-aware PDBQT output.

    Parameters
    ----------
    smiles       : input SMILES string
    output_pdbqt : path for the output PDBQT file
    name         : compound name used in log messages
    seed         : random seed for ETKDGv3

    Returns
    -------
    Path to the written PDBQT file.
    """
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"smiles_to_pdbqt: RDKit could not parse SMILES for {name}")

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) == -1:
        raise RuntimeError(
            f"smiles_to_pdbqt: ETKDGv3 conformer generation failed for {name}"
        )

    AllChem.MMFFOptimizeMolecule(mol)

    preparator = MoleculePreparation()
    mol_setups = preparator.prepare(mol)
    if not mol_setups:
        raise RuntimeError(f"smiles_to_pdbqt: meeko returned no setups for {name}")

    pdbqt_str, is_ok, err_msg = PDBQTWriterLegacy.write_string(mol_setups[0])
    if not is_ok:
        raise RuntimeError(
            f"smiles_to_pdbqt: meeko write_string failed for {name}: {err_msg}"
        )

    output_pdbqt = Path(output_pdbqt)
    output_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    output_pdbqt.write_text(pdbqt_str, encoding="utf-8")
    n_lines = len(pdbqt_str.splitlines())
    logger.info(f"Flexible ligand PDBQT saved: {output_pdbqt}  ({n_lines} lines)")
    return output_pdbqt
