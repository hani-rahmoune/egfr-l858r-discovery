"""
SMARTS-based covalent warhead detector.

Identifies electrophilic warheads that can form covalent bonds with cysteine
(C797 in EGFR) and other nucleophilic residues.  Non-covalent docking (Vina,
GNINA v1.0) cannot model the covalent bond; flagged compounds receive a
'low_confidence' docking selectivity tag.

Warhead definitions follow the ChEMBL electrophilic fragment vocabulary and
standard medicinal chemistry references.  All SMARTS are intentionally
conservative: false negatives are acceptable; false positives are not.

Public API
----------
detect_warheads(smiles)   -> list[str]   names of matched warhead types
is_covalent(smiles)       -> bool
covalent_confidence(smiles) -> str       'low_confidence' | 'standard'
"""

from __future__ import annotations

from rdkit import Chem

# ── SMARTS definitions ────────────────────────────────────────────────────────
# Each value is a SMARTS string; the key is the human-readable warhead name.
# Ordered from most-specific to most-general so the first match is the most
# informative when reporting.

WARHEAD_SMARTS: dict[str, str] = {
    # α,β-unsaturated amide (Michael acceptor): the dominant warhead in 3rd-gen
    # EGFR inhibitors (osimertinib, afatinib, neratinib, dacomitinib)
    "acrylamide": "[NX3]C(=O)C=C",
    # α,β-unsaturated ester (acrylate ester Michael acceptor): vinyl carbon
    # directly bonded to an ester carbonyl.  [OX2H0] = ester O (not carboxylic
    # acid OH).  Catches cmpd_021 (aryl acrylate), methacrylates, etc.
    # False negative vs acrylamide: erlotinib arene-alkyne is not caught (correct).
    "acrylate_ester": "C=CC(=O)[OX2H0]",
    # Ynalamide (alkynyl amide): propiolamide warhead; reacts slower than acryl
    "propiolamide": "[NX3]C(=O)C#C",
    # Vinyl sulfone: strong Michael acceptor, less common in approved drugs
    "vinyl_sulfone": "C=CS(=O)(=O)",
    # Haloacetamide (chloro- or bromo-): SN2 alkylator
    "chloroacetamide": "[Cl,Br][CH2]C(=O)[NX3]",
    # Epoxide: strained ring SN2/BAl2 alkylator
    "epoxide": "[OX2r3]1CC1",
    # α,β-unsaturated ketone (enone): Michael acceptor without amide nitrogen
    "michael_enone": "[CH2]=[CH]C(=O)[CX4,CX3;!$([CX3]=O)]",
    # Isocyanate: reacts with amines/hydroxyls
    "isocyanate": "[NX2]=[CX2]=[OX1]",
    # Cyanamide: mild electrophile (covalent kinase inhibitor scaffold)
    "cyanamide": "[NX3][CX2]#[NX1]",
}

# Pre-compiled patterns (module-level, singleton)
_COMPILED: dict[str, Chem.Mol] = {}


def _get_patterns() -> dict[str, Chem.Mol]:
    if not _COMPILED:
        for name, smarts in WARHEAD_SMARTS.items():
            pat = Chem.MolFromSmarts(smarts)
            if pat is None:
                raise RuntimeError(
                    f"covalent.py: invalid SMARTS for '{name}': {smarts}"
                )
            _COMPILED[name] = pat
    return _COMPILED


def detect_warheads(smiles: str) -> list[str]:
    """
    Return a list of warhead type names matched in the molecule.

    Returns an empty list for non-covalent molecules or unparseable SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    patterns = _get_patterns()
    return [name for name, pat in patterns.items() if mol.HasSubstructMatch(pat)]


def is_covalent(smiles: str) -> bool:
    """Return True if any electrophilic warhead is detected."""
    return bool(detect_warheads(smiles))


def covalent_confidence(smiles: str) -> str:
    """
    Return 'low_confidence' if a covalent warhead is detected, else 'standard'.

    Use this tag when annotating docking selectivity scores: non-covalent Vina
    docking cannot model the covalent bond and will underestimate (or misrank)
    binding for flagged compounds.
    """
    return "low_confidence" if is_covalent(smiles) else "standard"
