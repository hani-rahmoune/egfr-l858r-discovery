from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Fixed order, never change — models depend on this exact sequence
DESCRIPTOR_NAMES = [
    "mol_weight", "logp", "tpsa", "hbd", "hba",
    "rotatable_bonds", "aromatic_rings", "ring_count",
    "fraction_csp3", "formal_charge", "qed",
]


def compute_descriptors(smiles: str) -> dict[str, float] | None:
    """Compute all physicochemical descriptors for one molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return {
            "mol_weight": Descriptors.ExactMolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "tpsa": Descriptors.TPSA(mol),
            "hbd": rdMolDescriptors.CalcNumHBD(mol),
            "hba": rdMolDescriptors.CalcNumHBA(mol),
            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "ring_count": rdMolDescriptors.CalcNumRings(mol),
            "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
            "formal_charge": Chem.GetFormalCharge(mol),
            "qed": QED.qed(mol),  # 0-1 drug-likeness score
        }
    except Exception:
        return None


def compute_descriptors_array(smiles: str) -> np.ndarray | None:
    """Same as compute_descriptors but returns a fixed-length numpy array."""
    desc = compute_descriptors(smiles)
    if desc is None:
        return None
    return np.array([desc[k] for k in DESCRIPTOR_NAMES], dtype=np.float32)


def compute_descriptor_matrix(
    smiles_list: list[str],
) -> tuple[np.ndarray, list[int]]:
    """
    Compute descriptors for a list of SMILES.
    Returns (matrix, valid_indices), same pattern as compute_fingerprint_matrix.
    """
    descs, valid_indices = [], []
    for i, smi in enumerate(smiles_list):
        arr = compute_descriptors_array(smi)
        if arr is not None:
            descs.append(arr)
            valid_indices.append(i)
    if not descs:
        return np.empty((0, len(DESCRIPTOR_NAMES)), dtype=np.float32), []
    return np.array(descs, dtype=np.float32), valid_indices


def check_lipinski(smiles: str) -> dict:
    """
    Lipinski Rule of Five. Allow 1 violation (common pharma practice).
    MW<=500, LogP<=5, HBD<=5, HBA<=10.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"lipinski_pass": False, "violations": 4}
    mw = Descriptors.ExactMolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return {
        "mol_weight": mw, "logp": logp, "hbd": hbd, "hba": hba,
        "violations": violations, "lipinski_pass": violations <= 1,
    }


def check_veber(smiles: str) -> dict:
    """Veber oral bioavailability rules: TPSA<=140, rotatable bonds<=10."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"veber_pass": False}
    tpsa = Descriptors.TPSA(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    return {"tpsa": tpsa, "rotatable_bonds": rot, "veber_pass": tpsa <= 140 and rot <= 10}