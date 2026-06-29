"""
Protein structure preparation for AutoDock Vina docking.

Downloads EGFR crystal structures (2ITZ L858R, 2ITY WT) from RCSB,
cleans them with Biopython, adds hydrogens via PDBFixer at physiological pH,
and writes receptor PDBQT files suitable for AutoDock Vina.

Design notes:
- Chain A only; all HETATM except the co-crystal ligand are removed before
  pdbfixer so it never tries to template non-standard residues.
- pdbfixer adds missing heavy-atom side-chains and hydrogens at pH 7.4.
- PDBQT writer includes heavy atoms and polar H (bonded to N/O/S within
  1.15 A) using AutoDock4/Vina atom type conventions.
- Receptor partial charges are set to 0.000 (Vina ignores receptor charges).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ── AutoDock atom type tables ─────────────────────────────────────────────────

# Carbons in aromatic rings of standard amino acids
_AROMATIC_C = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2"},
    "TRP": {"CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "HIS": {"CG", "CD2", "CE1"},
}

# Nitrogens that are H-bond acceptors (ring lone pair available)
_AROMATIC_NA = {
    "HIS": {"ND1", "NE2"},
    "TRP": {"NE1"},
}

# Two-char element → AutoDock type
_ELEMENT2_MAP = {
    "ZN": "Zn",
    "FE": "Fe",
    "MG": "Mg",
    "CA": "Ca",
    "MN": "Mn",
    "CU": "Cu",
    "CL": "Cl",
    "BR": "Br",
}


def _autodock_type(element: str, resname: str, atname: str) -> str:
    """Return the AutoDock4/Vina receptor atom type string."""
    el = element.strip().capitalize()
    an = atname.strip()
    rn = resname.strip()

    if el == "C":
        return "A" if (rn in _AROMATIC_C and an in _AROMATIC_C[rn]) else "C"

    if el == "N":
        return "NA" if (rn in _AROMATIC_NA and an in _AROMATIC_NA[rn]) else "N"

    if el == "O":
        return "OA"

    if el == "S":
        return "SA"

    if el in ("H", "D"):
        return "HD"

    if el == "P":
        return "P"

    if el == "F":
        return "F"

    if el == "I":
        return "I"

    two = el.upper()[:2]
    if two in _ELEMENT2_MAP:
        return _ELEMENT2_MAP[two]

    return el[:2]  # fallback: first two chars of element


# ── PDBQT line formatter ──────────────────────────────────────────────────────


def _pdbqt_atom_line(
    serial: int,
    atname: str,
    altloc: str,
    resname: str,
    chain: str,
    resseq: int,
    icode: str,
    x: float,
    y: float,
    z: float,
    occ: float,
    bfac: float,
    charge: float,
    atype: str,
) -> str:
    """Format one PDBQT ATOM record (79 chars, no newline)."""
    an = atname.strip()
    # 4-char atom-name field: 1-char elements start at column 14 (space at 13)
    name_field = f" {an:<3s}" if len(an) < 4 else f"{an:<4s}"
    altloc_c = (altloc.strip() or " ")[:1]
    icode_c = (icode.strip() or " ")[:1]
    return (
        f"{'ATOM':<6s}{serial:5d} {name_field}{altloc_c:1s}"
        f"{resname:>3s} {chain:1s}{resseq:4d}{icode_c:1s}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occ:6.2f}{bfac:6.2f}    "
        f"{charge:6.3f} {atype:<2s}"
    )


# ── Download ──────────────────────────────────────────────────────────────────


def download_pdb(pdb_id: str, out_dir: Path) -> Path:
    """
    Download a PDB file from RCSB.  Returns the local path.
    Skips download if the file already exists.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pdb_id.upper()}_raw.pdb"

    if out_path.exists():
        logger.info(f"PDB {pdb_id} already downloaded: {out_path}")
        return out_path

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    logger.info(f"Downloading {pdb_id} from {url} ...")
    urllib.request.urlretrieve(url, str(out_path))
    logger.info(f"Saved {pdb_id} to {out_path}")
    return out_path


# ── Biopython cleaning ────────────────────────────────────────────────────────


def _save_protein_only_pdb(
    raw_pdb: Path,
    chain_id: str,
    out_path: Path,
) -> Path:
    """
    Write a PDB containing only standard ATOM residues for one chain.
    Removes all HETATM (waters, ions, ligands).
    """
    from Bio.PDB import PDBIO, PDBParser, Select

    class _ProteinSelect(Select):
        def accept_chain(self, chain):
            return chain.id == chain_id

        def accept_residue(self, residue):
            return residue.get_id()[0] == " "  # ATOM records only

        def accept_atom(self, atom):
            # Skip alternate locations other than the primary
            return atom.get_altloc() in (" ", "A")

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("prot", str(raw_pdb))

    io = PDBIO()
    io.set_structure(struct)
    io.save(str(out_path), select=_ProteinSelect())
    logger.info(f"Protein-only PDB saved: {out_path}")
    return out_path


def _save_ligand_pdb(
    raw_pdb: Path,
    ligand_resname: str,
    chain_id: str,
    out_path: Path,
) -> Path:
    """
    Extract the co-crystal ligand residue and write it to a separate PDB.
    """
    from Bio.PDB import PDBIO, PDBParser, Select

    class _LigandSelect(Select):
        def accept_chain(self, chain):
            return chain.id == chain_id

        def accept_residue(self, residue):
            return residue.get_resname().strip() == ligand_resname

        def accept_atom(self, atom):
            return atom.get_altloc() in (" ", "A")

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("lig", str(raw_pdb))

    io = PDBIO()
    io.set_structure(struct)
    io.save(str(out_path), select=_LigandSelect())
    logger.info(f"Ligand PDB ({ligand_resname}) saved: {out_path}")
    return out_path


# ── PDBFixer preparation ──────────────────────────────────────────────────────


def prepare_with_pdbfixer(
    input_pdb: Path,
    output_pdb: Path,
    ph: float = 7.4,
) -> Path:
    """
    Add missing side-chain atoms and hydrogens at the given pH using PDBFixer.

    Calls findMissingResidues to initialize internal state, then clears
    missingResidues to suppress loop modelling — we only want side-chain
    completion and H addition, not building unresolved loops.
    """
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=str(input_pdb))
    # findMissingResidues must be called to initialise fixer.missingResidues
    # before findMissingAtoms.  Clear afterwards to skip loop building.
    fixer.findMissingResidues()
    fixer.missingResidues = {}  # suppress loop modelling
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)

    with open(str(output_pdb), "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

    logger.info(f"pdbfixer-prepared PDB (pH={ph}) saved: {output_pdb}")
    return output_pdb


# ── Ligand centroid ───────────────────────────────────────────────────────────


def get_ligand_centroid(
    ligand_pdb: Path,
    ligand_resname: str,
    chain_id: str,
) -> tuple[float, float, float]:
    """
    Compute the centroid (mean xyz) of heavy atoms in the co-crystal ligand.
    Returns (cx, cy, cz) in Angstrom.
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("lig", str(ligand_pdb))

    coords = []
    for model in struct:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if residue.get_resname().strip() != ligand_resname:
                    continue
                for atom in residue:
                    if (atom.element or "").strip().upper() not in ("H", "D", ""):
                        coords.append(atom.get_coord())

    if not coords:
        raise ValueError(
            f"No heavy atoms found for ligand '{ligand_resname}' in "
            f"chain '{chain_id}' of {ligand_pdb}"
        )

    arr = np.array(coords, dtype=float)
    cx, cy, cz = arr.mean(axis=0)
    logger.info(
        f"IRE centroid in {ligand_pdb.name}: "
        f"x={cx:.3f}  y={cy:.3f}  z={cz:.3f}  "
        f"(from {len(coords)} heavy atoms)"
    )
    return float(cx), float(cy), float(cz)


# ── PDBQT writer ─────────────────────────────────────────────────────────────


def write_receptor_pdbqt(
    protein_pdb: Path,
    output_pdbqt: Path,
) -> Path:
    """
    Convert a pdbfixer-prepared PDB to receptor PDBQT for AutoDock Vina.

    Includes heavy atoms and polar hydrogens (H bonded to N/O/S within 1.15 A).
    Non-polar H (bonded to C) are omitted — Vina handles them implicitly.
    Partial charges are set to 0.000 (Vina ignores receptor charges).
    """
    from Bio.PDB import NeighborSearch, PDBParser

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("prot", str(protein_pdb))
    model = struct[0]

    # Build neighbor search from heavy atoms for polar-H classification
    heavy_atoms = [
        a
        for a in model.get_atoms()
        if (a.element or "").strip().upper() not in ("H", "D", "")
    ]
    ns = NeighborSearch(heavy_atoms)

    lines = ["REMARK  PDBQT receptor prepared by prepare_protein.py"]
    serial = 1

    for chain in model:
        for residue in chain:
            resname = residue.get_resname().strip()
            chain_id = chain.id
            resseq = residue.get_id()[1]
            icode = residue.get_id()[2]

            for atom in residue:
                # Skip alternate locations (keep primary " " or "A")
                if atom.get_altloc() not in (" ", "A"):
                    continue

                element = (atom.element or "").strip().upper()
                coord = atom.get_coord()

                if element in ("H", "D", ""):
                    # Polar-H check: must be bonded to N, O, or S
                    nearby = ns.search(coord, 1.15)
                    if not any(
                        (a.element or "").strip().upper() in ("N", "O", "S")
                        for a in nearby
                    ):
                        continue  # non-polar, omit
                    atype = "HD"
                else:
                    atype = _autodock_type(element, resname, atom.get_name())

                x, y, z = coord
                occ = atom.get_occupancy() if atom.get_occupancy() is not None else 1.0
                bfac = atom.get_bfactor() if atom.get_bfactor() is not None else 0.0

                line = _pdbqt_atom_line(
                    serial=serial,
                    atname=atom.get_name(),
                    altloc=atom.get_altloc(),
                    resname=resname,
                    chain=chain_id,
                    resseq=resseq,
                    icode=icode,
                    x=x,
                    y=y,
                    z=z,
                    occ=occ,
                    bfac=bfac,
                    charge=0.0,
                    atype=atype,
                )
                lines.append(line)
                serial += 1

    lines.append("END")
    output_pdbqt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_atoms = serial - 1
    logger.info(f"Receptor PDBQT saved: {output_pdbqt}  ({n_atoms} atoms)")
    return output_pdbqt


# ── Full preparation pipeline ─────────────────────────────────────────────────


def prepare_receptor(
    pdb_id: str,
    chain_id: str,
    ligand_resname: str,
    out_dir: Path,
    ph: float = 7.4,
) -> dict[str, Any]:
    """
    Full preparation pipeline for one crystal structure.

    1. Download PDB from RCSB (skips if already present).
    2. Extract ligand coords (for box definition).
    3. Save protein-only PDB (no HETATM).
    4. pdbfixer: add missing atoms + H at pH 7.4.
    5. Write receptor PDBQT.

    Returns a dict with paths and centroid.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pid = pdb_id.upper()

    raw_pdb = download_pdb(pid, out_dir)
    ligand_pdb = out_dir / f"{pid}_ligand.pdb"
    protein_pdb = out_dir / f"{pid}_protein_clean.pdb"
    prepared_pdb = out_dir / f"{pid}_prepared.pdb"
    receptor_pdbqt = out_dir / f"{pid}_receptor.pdbqt"

    _save_ligand_pdb(raw_pdb, ligand_resname, chain_id, ligand_pdb)
    _save_protein_only_pdb(raw_pdb, chain_id, protein_pdb)
    prepare_with_pdbfixer(protein_pdb, prepared_pdb, ph=ph)
    write_receptor_pdbqt(prepared_pdb, receptor_pdbqt)

    centroid = get_ligand_centroid(ligand_pdb, ligand_resname, chain_id)

    return {
        "pdb_id": pid,
        "raw_pdb": raw_pdb,
        "ligand_pdb": ligand_pdb,
        "protein_clean_pdb": protein_pdb,
        "prepared_pdb": prepared_pdb,
        "receptor_pdbqt": receptor_pdbqt,
        "ligand_centroid": centroid,
    }
