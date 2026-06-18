"""
Structural alignment of WT receptor (2ITY) onto L858R receptor (2ITZ).

Uses Biopython Superimposer on common Ca atoms.  The transformation is applied
to all atoms so the shared docking box (defined from the 2ITZ IRE centroid)
covers the aligned ATP-binding cleft of both receptors without the ~1.1 A
crystal-frame offset.
"""

from __future__ import annotations

from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _collect_ca(struct, chain_id: str = "A") -> dict[int, object]:
    """Return {residue_number: CA_atom} for one chain."""
    ca: dict[int, object] = {}
    for chain in struct[0]:
        if chain.id != chain_id:
            continue
        for residue in chain:
            if "CA" in residue:
                ca[residue.get_id()[1]] = residue["CA"]
    return ca


def align_wt_to_l858r(
    l858r_prepared: Path,
    wt_prepared: Path,
    aligned_out: Path,
    chain_id: str = "A",
    min_common_ca: int = 50,
) -> float:
    """
    Superpose 2ITY (WT) onto 2ITZ (L858R) on common Ca atoms.

    Writes the aligned WT structure to ``aligned_out`` in PDB format.
    Returns the Ca backbone RMSD in Angstrom.

    Parameters
    ----------
    l858r_prepared : pdbfixer-prepared PDB for 2ITZ (fixed / reference)
    wt_prepared    : pdbfixer-prepared PDB for 2ITY (mobile)
    aligned_out    : output path for the aligned WT PDB
    chain_id       : chain used for Ca selection (both structures)
    min_common_ca  : minimum shared Ca residues required
    """
    from Bio.PDB import PDBIO, PDBParser, Superimposer

    parser = PDBParser(QUIET=True)
    struct_ref = parser.get_structure("l858r", str(l858r_prepared))
    struct_mov = parser.get_structure("wt", str(wt_prepared))

    ref_ca = _collect_ca(struct_ref, chain_id)
    mov_ca = _collect_ca(struct_mov, chain_id)

    common = sorted(set(ref_ca) & set(mov_ca))
    if len(common) < min_common_ca:
        raise ValueError(
            f"align_wt_to_l858r: only {len(common)} common Ca atoms "
            f"(need >= {min_common_ca})"
        )

    ref_atoms = [ref_ca[i] for i in common]
    mov_atoms = [mov_ca[i] for i in common]

    sup = Superimposer()
    sup.set_atoms(ref_atoms, mov_atoms)
    sup.apply(list(struct_mov[0].get_atoms()))

    aligned_out = Path(aligned_out)
    aligned_out.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(struct_mov)
    io.save(str(aligned_out))

    rmsd = float(sup.rms)
    logger.info(f"Aligned 2ITY onto 2ITZ on {len(common)} Ca atoms  RMSD={rmsd:.3f} A")
    logger.info(f"Aligned WT PDB saved: {aligned_out}")
    return rmsd


def verify_box_coverage(
    pdb_path: Path,
    box: dict[str, float],
    chain_id: str = "A",
    pad: float = 2.0,
) -> bool:
    """
    Check that the docking box comfortably covers the Ca centroid of the receptor.

    Uses the Ca centroid as a proxy for the kinase-domain centre and confirms
    it lies within the box extended by ``pad`` Angstrom on each face.  Returns
    True if coverage is adequate, False otherwise.
    """
    import numpy as np
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("s", str(pdb_path))
    ca_coords = [
        a.get_coord()
        for chain in struct[0]
        if chain.id == chain_id
        for res in chain
        if "CA" in res
        for a in [res["CA"]]
    ]
    if not ca_coords:
        logger.warning(f"verify_box_coverage: no Ca found in {pdb_path}")
        return False

    cx_struct, cy_struct, cz_struct = np.mean(ca_coords, axis=0)
    cx = box["center_x"]
    cy = box["center_y"]
    cz = box["center_z"]
    hx = box["size_x"] / 2 + pad
    hy = box["size_y"] / 2 + pad
    hz = box["size_z"] / 2 + pad

    in_box = (
        abs(cx_struct - cx) < hx
        and abs(cy_struct - cy) < hy
        and abs(cz_struct - cz) < hz
    )
    if not in_box:
        logger.warning(
            f"verify_box_coverage: Ca centroid ({cx_struct:.1f}, {cy_struct:.1f}, "
            f"{cz_struct:.1f}) outside box+pad={pad} A for {pdb_path.name}"
        )
    return in_box
