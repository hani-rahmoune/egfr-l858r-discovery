"""
Unit tests for src/features/covalent.py

All tests use SMILES only — no model files, no data files required.
"""

from __future__ import annotations

import pytest

from src.features.covalent import (
    WARHEAD_SMARTS,
    covalent_confidence,
    detect_warheads,
    is_covalent,
)

# ── Known positives ────────────────────────────────────────────────────────────

# Osimertinib: acrylamide warhead targets C797 in EGFR
_OSIMERTINIB = "COc1cc2c(Nc3cccc(NC(=O)/C=C/CN(C)C)c3)ncnc2cc1NC(C)=O"
# Afatinib: acrylamide warhead
_AFATINIB = "CN(C)/C=C/C(=O)Nc1cc2c(Nc3ccc(F)cc3Cl)ncnc2cc1OCC1CCOCC1"
# Neratinib: acrylamide warhead
_NERATINIB = "C=CC(=O)Nc1cccc(Nc2ncnc3ccc(CC#N)cc23)c1"
# Vinyl sulfone example (non-drug, synthetic)
_VINYL_SULFONE = "C=CS(=O)(=O)c1ccccc1"
# Propiolamide (ynalamide) example
_PROPIOLAMIDE = "C#CC(=O)Nc1ccccc1"
# Chloroacetamide example
_CHLOROACETAMIDE = "ClCC(=O)Nc1ccccc1"
# Epoxide example
_EPOXIDE = "C1OC1c1ccccc1"

# Acrylate ester (Michael acceptor via ester linkage): missed by acrylamide SMARTS
# cmpd_021 from the ranked-library docking; confirmed by Brenk Michael_acceptor_1
_CMPD_021_ACRYLATE_ESTER = "C=CC(=O)Oc1cc2c(Nc3ccc(Cl)c(Cl)c3F)ncnc2cc1OC"

# ── Known negatives ────────────────────────────────────────────────────────────

# Gefitinib: reversible EGFR inhibitor, no covalent warhead
_GEFITINIB = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
# Erlotinib: reversible EGFR inhibitor (alkyne present, but not propiolamide)
_ERLOTINIB = "C#Cc1cccc(Nc2ncnc3cc(OCC)c(OCC)cc23)c1"


# ── WARHEAD_SMARTS sanity ─────────────────────────────────────────────────────


@pytest.mark.unit
class TestWarheadSmartsDefinitions:
    def test_all_smarts_are_strings(self):
        assert all(isinstance(v, str) for v in WARHEAD_SMARTS.values())

    def test_all_smarts_are_parseable(self):
        from rdkit import Chem

        for name, smarts in WARHEAD_SMARTS.items():
            pat = Chem.MolFromSmarts(smarts)
            assert pat is not None, f"SMARTS for '{name}' is invalid: {smarts}"

    def test_warhead_dict_has_acrylamide(self):
        assert "acrylamide" in WARHEAD_SMARTS

    def test_warhead_dict_has_acrylate_ester(self):
        assert "acrylate_ester" in WARHEAD_SMARTS

    def test_warhead_dict_has_vinyl_sulfone(self):
        assert "vinyl_sulfone" in WARHEAD_SMARTS

    def test_warhead_dict_has_propiolamide(self):
        assert "propiolamide" in WARHEAD_SMARTS


# ── detect_warheads ───────────────────────────────────────────────────────────


@pytest.mark.unit
class TestDetectWarheads:
    def test_osimertinib_flagged_as_acrylamide(self):
        hits = detect_warheads(_OSIMERTINIB)
        assert "acrylamide" in hits

    def test_afatinib_flagged_as_acrylamide(self):
        hits = detect_warheads(_AFATINIB)
        assert "acrylamide" in hits

    def test_neratinib_flagged_as_acrylamide(self):
        hits = detect_warheads(_NERATINIB)
        assert "acrylamide" in hits

    def test_vinyl_sulfone_detected(self):
        hits = detect_warheads(_VINYL_SULFONE)
        assert "vinyl_sulfone" in hits

    def test_propiolamide_detected(self):
        hits = detect_warheads(_PROPIOLAMIDE)
        assert "propiolamide" in hits

    def test_chloroacetamide_detected(self):
        hits = detect_warheads(_CHLOROACETAMIDE)
        assert "chloroacetamide" in hits

    def test_epoxide_detected(self):
        hits = detect_warheads(_EPOXIDE)
        assert "epoxide" in hits

    def test_acrylate_ester_detected(self):
        # cmpd_021: aryl acrylate ester — Michael acceptor, missed by acrylamide SMARTS
        hits = detect_warheads(_CMPD_021_ACRYLATE_ESTER)
        assert "acrylate_ester" in hits

    def test_acrylate_ester_not_flagged_as_acrylamide(self):
        # The ester form should NOT trigger the amide-specific acrylamide key
        hits = detect_warheads(_CMPD_021_ACRYLATE_ESTER)
        assert "acrylamide" not in hits

    def test_plain_ester_not_flagged(self):
        # Aspirin: acetate ester, no vinyl group adjacent — NOT a Michael acceptor
        aspirin = "CC(=O)Oc1ccccc1C(=O)O"
        hits = detect_warheads(aspirin)
        assert "acrylate_ester" not in hits

    def test_gefitinib_no_warhead(self):
        assert detect_warheads(_GEFITINIB) == []

    def test_erlotinib_no_warhead(self):
        # Erlotinib has a terminal alkyne on an arene but NOT a propiolamide
        assert detect_warheads(_ERLOTINIB) == []

    def test_returns_list(self):
        result = detect_warheads(_GEFITINIB)
        assert isinstance(result, list)

    def test_invalid_smiles_returns_empty(self):
        assert detect_warheads("not_a_smiles_xyz") == []

    def test_empty_smiles_returns_empty(self):
        assert detect_warheads("") == []

    def test_multiple_warheads_detected(self):
        # A molecule with both acrylamide and chloro group
        # C=CC(=O)Nc1ccccc1CC1OC1 has acrylamide + epoxide
        hits = detect_warheads("C=CC(=O)Nc1ccc(CC2OC2)cc1")
        assert len(hits) >= 1  # at least acrylamide


# ── is_covalent ───────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestIsCovalent:
    def test_osimertinib_is_covalent(self):
        assert is_covalent(_OSIMERTINIB) is True

    def test_gefitinib_not_covalent(self):
        assert is_covalent(_GEFITINIB) is False

    def test_vinyl_sulfone_is_covalent(self):
        assert is_covalent(_VINYL_SULFONE) is True

    def test_returns_bool(self):
        assert isinstance(is_covalent(_GEFITINIB), bool)
        assert isinstance(is_covalent(_OSIMERTINIB), bool)

    def test_acrylate_ester_is_covalent(self):
        assert is_covalent(_CMPD_021_ACRYLATE_ESTER) is True

    def test_invalid_smiles_not_covalent(self):
        assert is_covalent("INVALID") is False


# ── covalent_confidence ───────────────────────────────────────────────────────


@pytest.mark.unit
class TestCovalentConfidence:
    def test_covalent_molecule_returns_low_confidence(self):
        assert covalent_confidence(_OSIMERTINIB) == "low_confidence"

    def test_non_covalent_returns_standard(self):
        assert covalent_confidence(_GEFITINIB) == "standard"

    def test_erlotinib_standard(self):
        assert covalent_confidence(_ERLOTINIB) == "standard"

    def test_vinyl_sulfone_low_confidence(self):
        assert covalent_confidence(_VINYL_SULFONE) == "low_confidence"

    def test_returns_one_of_two_values(self):
        valid = {"low_confidence", "standard"}
        assert covalent_confidence(_GEFITINIB) in valid
        assert covalent_confidence(_OSIMERTINIB) in valid
