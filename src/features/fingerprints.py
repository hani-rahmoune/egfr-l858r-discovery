"""Morgan/ECFP, MACCS, RDKit-topological, atom-pair, and torsion fingerprints."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator, rdMolDescriptors

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _mol(smiles: str):
    """Parse SMILES, return None if invalid."""
    if not isinstance(smiles, str):
        return None
    return Chem.MolFromSmiles(smiles)


def morgan_fingerprint(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = True,
) -> np.ndarray | None:
    """Morgan/ECFP fingerprint. radius=2 -> ECFP4, radius=3 -> ECFP6."""
    mol = _mol(smiles)
    if mol is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits, includeChirality=use_chirality
    )
    return gen.GetFingerprintAsNumPy(mol)


def maccs_fingerprint(smiles: str) -> np.ndarray | None:
    """MACCS keys, always 167 bits."""
    mol = _mol(smiles)
    if mol is None:
        return None
    return np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.uint8)


def rdkit_topological_fingerprint(smiles: str, n_bits: int = 2048) -> np.ndarray | None:
    """RDKit path-based topological fingerprint."""
    mol = _mol(smiles)
    if mol is None:
        return None
    return np.array(Chem.RDKFingerprint(mol, fpSize=n_bits), dtype=np.uint8)


def atom_pair_fingerprint(smiles: str, n_bits: int = 2048) -> np.ndarray | None:
    """Atom pair fingerprint, encodes pairs of atoms and their distances."""
    mol = _mol(smiles)
    if mol is None:
        return None
    fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
    return np.array(fp, dtype=np.uint8)


def topological_torsion_fingerprint(
    smiles: str, n_bits: int = 2048
) -> np.ndarray | None:
    """Topological torsion fingerprint (hashed bit vector).

    Encodes sequences of 4 atoms along paths in the molecular graph.
    Complementary to atom pairs and Morgan FPs for branching/shape information.
    """
    mol = _mol(smiles)
    if mol is None:
        return None
    fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
        mol, nBits=n_bits
    )
    return np.array(fp, dtype=np.uint8)


def compute_fingerprint(
    smiles: str,
    fp_type: str = "morgan_ecfp4",
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = True,
) -> np.ndarray | None:
    """Unified dispatcher. fp_type must match keys in model_config.yaml."""
    if fp_type in ("morgan_ecfp4", "morgan_ecfp6", "morgan"):
        return morgan_fingerprint(
            smiles, radius=radius, n_bits=n_bits, use_chirality=use_chirality
        )
    elif fp_type == "maccs":
        return maccs_fingerprint(smiles)
    elif fp_type in ("rdkit_topological", "rdkit"):
        return rdkit_topological_fingerprint(smiles, n_bits=n_bits)
    elif fp_type == "atom_pair":
        return atom_pair_fingerprint(smiles, n_bits=n_bits)
    elif fp_type == "topological_torsion":
        return topological_torsion_fingerprint(smiles, n_bits=n_bits)
    else:
        raise ValueError(f"Unknown fingerprint type: {fp_type}")


def compute_fingerprint_matrix(
    smiles_list: list[str],
    fp_type: str = "morgan_ecfp4",
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = True,
) -> tuple[np.ndarray, list[int]]:
    """
    Compute fingerprints for a list of SMILES.
    Returns (matrix, valid_indices) so callers know which rows succeeded.
    Invalid SMILES are silently skipped.
    """
    fps, valid_indices = [], []
    for i, smi in enumerate(smiles_list):
        fp = compute_fingerprint(
            smi,
            fp_type=fp_type,
            radius=radius,
            n_bits=n_bits,
            use_chirality=use_chirality,
        )
        if fp is not None:
            fps.append(fp)
            valid_indices.append(i)
    n_failed = len(smiles_list) - len(valid_indices)
    if n_failed > 0:
        logger.warning(f"compute_fingerprint_matrix: {n_failed} invalid SMILES skipped")
    if not fps:
        return np.empty((0, n_bits), dtype=np.float32), []
    return np.array(fps, dtype=np.float32), valid_indices
