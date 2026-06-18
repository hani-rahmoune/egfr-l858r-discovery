"""
Tests for Phase B Step B1 — structure preparation.

Unit tests (no network, no files):
  - AutoDock atom type assignments
  - PDBQT line format
  - Centroid arithmetic
  - docking_config.yaml schema

Integration tests (marked 'integration', skip if protein/ files absent):
  - Receptor PDBQT files exist after prepare_docking.py
  - Single chain retained in PDBQT
  - Box center reproducibility across runs
  - No T790M structures in config
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.docking.prepare_ligands import centroid_from_heavy_atoms
from src.docking.prepare_protein import (
    _autodock_type,
    _pdbqt_atom_line,
    get_ligand_centroid,
)

ROOT = Path(__file__).resolve().parents[1]
DOCK_DIR = ROOT / "data" / "docking" / "protein"
CFG_PATH = ROOT / "config" / "docking_config.yaml"


# ── AutoDock type assignment ──────────────────────────────────────────────────


@pytest.mark.unit
class TestAutodockTypeAssignment:
    """Verify element → AutoDock atom type for common receptor atoms."""

    def test_aliphatic_carbon_is_C(self):
        assert _autodock_type("C", "ALA", "CB") == "C"

    def test_backbone_carbon_is_C(self):
        assert _autodock_type("C", "GLY", "CA") == "C"

    def test_aromatic_carbon_phe_is_A(self):
        assert _autodock_type("C", "PHE", "CG") == "A"
        assert _autodock_type("C", "PHE", "CE1") == "A"
        assert _autodock_type("C", "PHE", "CZ") == "A"

    def test_aromatic_carbon_tyr_is_A(self):
        assert _autodock_type("C", "TYR", "CD1") == "A"
        assert _autodock_type("C", "TYR", "CE2") == "A"

    def test_aromatic_carbon_trp_is_A(self):
        assert _autodock_type("C", "TRP", "CG") == "A"
        assert _autodock_type("C", "TRP", "CE3") == "A"

    def test_non_ring_carbon_phe_is_C(self):
        # CB is not in the ring
        assert _autodock_type("C", "PHE", "CB") == "C"

    def test_nitrogen_is_N(self):
        assert _autodock_type("N", "ALA", "N") == "N"
        assert _autodock_type("N", "LYS", "NZ") == "N"

    def test_his_ring_nitrogen_is_NA(self):
        assert _autodock_type("N", "HIS", "ND1") == "NA"
        assert _autodock_type("N", "HIS", "NE2") == "NA"

    def test_oxygen_is_OA(self):
        assert _autodock_type("O", "ALA", "O") == "OA"
        assert _autodock_type("O", "SER", "OG") == "OA"
        assert _autodock_type("O", "GLU", "OE1") == "OA"

    def test_sulfur_is_SA(self):
        assert _autodock_type("S", "CYS", "SG") == "SA"
        assert _autodock_type("S", "MET", "SD") == "SA"

    def test_hydrogen_is_HD(self):
        assert _autodock_type("H", "ALA", "H") == "HD"

    def test_phosphorus_is_P(self):
        assert _autodock_type("P", "ANY", "P") == "P"

    def test_zinc_maps_correctly(self):
        assert _autodock_type("ZN", "ZN", "ZN") == "Zn"

    def test_chlorine_maps_correctly(self):
        assert _autodock_type("CL", "ANY", "CL") == "Cl"


# ── PDBQT line format ─────────────────────────────────────────────────────────


@pytest.mark.unit
class TestPdbqtLineFormat:
    """Verify the PDBQT atom record format matches AutoDock conventions."""

    def test_line_contains_atom_keyword(self):
        line = _pdbqt_atom_line(
            1, "N", " ", "ALA", "A", 1, " ", 5.959, 73.201, 8.729, 1.0, 0.0, 0.0, "N"
        )
        assert line.startswith("ATOM  ")

    def test_coordinates_present(self):
        line = _pdbqt_atom_line(
            1, "CA", " ", "ALA", "A", 1, " ", 12.345, -6.789, 0.001, 1.0, 0.0, 0.0, "C"
        )
        assert "12.345" in line
        assert "-6.789" in line
        assert "0.001" in line

    def test_atom_type_at_end(self):
        line = _pdbqt_atom_line(
            1, "OG", " ", "SER", "A", 5, " ", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, "OA"
        )
        assert line.endswith("OA") or line.endswith("OA ")

    def test_charge_present(self):
        line = _pdbqt_atom_line(
            1, "N", " ", "ALA", "A", 1, " ", 0.0, 0.0, 0.0, 1.0, 0.0, -0.347, "N"
        )
        assert "-0.347" in line

    def test_serial_number_increments(self):
        line1 = _pdbqt_atom_line(
            1, "N", " ", "ALA", "A", 1, " ", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, "N"
        )
        line2 = _pdbqt_atom_line(
            2, "CA", " ", "ALA", "A", 1, " ", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, "C"
        )
        # Serial 1 is at positions 7-11 in the record
        assert "    1" in line1
        assert "    2" in line2

    def test_long_atom_name_no_padding(self):
        # Atom names with 4 chars should not get a leading space
        line = _pdbqt_atom_line(
            1, "HG11", " ", "VAL", "A", 1, " ", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, "HD"
        )
        assert "HG11" in line

    def test_short_atom_name_gets_space_padding(self):
        # "N" → " N  " in the 4-char name field
        line = _pdbqt_atom_line(
            1, "N", " ", "ALA", "A", 1, " ", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, "N"
        )
        assert " N  " in line or " N " in line

    def test_chain_id_present(self):
        line = _pdbqt_atom_line(
            1, "CA", " ", "ALA", "B", 1, " ", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, "C"
        )
        assert "B" in line


# ── Centroid arithmetic ───────────────────────────────────────────────────────


@pytest.mark.unit
class TestCentroidArithmetic:
    """Verify centroid computation for box definition."""

    def test_single_atom_centroid(self):
        cx, cy, cz = centroid_from_heavy_atoms([(3.0, 4.0, 5.0)])
        assert abs(cx - 3.0) < 1e-9
        assert abs(cy - 4.0) < 1e-9
        assert abs(cz - 5.0) < 1e-9

    def test_symmetric_pair_centroid_is_midpoint(self):
        cx, cy, cz = centroid_from_heavy_atoms([(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
        assert abs(cx) < 1e-9
        assert abs(cy) < 1e-9
        assert abs(cz) < 1e-9

    def test_five_atom_centroid(self):
        coords = [
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
            (5.0, 0.0, 0.0),
        ]
        cx, cy, cz = centroid_from_heavy_atoms(coords)
        assert abs(cx - 3.0) < 1e-9

    def test_empty_coords_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            centroid_from_heavy_atoms([])

    def test_centroid_returns_floats(self):
        cx, cy, cz = centroid_from_heavy_atoms([(1, 2, 3)])
        assert isinstance(cx, float)
        assert isinstance(cy, float)
        assert isinstance(cz, float)


# ── docking_config.yaml schema ────────────────────────────────────────────────


@pytest.mark.unit
class TestDockingConfigSchema:
    """Verify config/docking_config.yaml has the expected structure."""

    @pytest.fixture(scope="class")
    def cfg(self):
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_structures_key_present(self, cfg):
        assert "structures" in cfg

    def test_l858r_structure_defined(self, cfg):
        assert "l858r" in cfg["structures"]
        s = cfg["structures"]["l858r"]
        assert s["pdb_id"] == "2ITZ"
        assert s["mutation"] == "L858R"
        assert s["chain"] == "A"
        assert s["co_crystal_ligand"] == "IRE"

    def test_wt_structure_defined(self, cfg):
        assert "wild_type" in cfg["structures"]
        s = cfg["structures"]["wild_type"]
        assert s["pdb_id"] == "2ITY"
        assert s["mutation"] == "wild_type"
        assert s["chain"] == "A"
        assert s["co_crystal_ligand"] == "IRE"

    def test_box_size_defined(self, cfg):
        box = cfg["box"]
        assert box["size_x"] == pytest.approx(22.5)
        assert box["size_y"] == pytest.approx(22.5)
        assert box["size_z"] == pytest.approx(22.5)

    def test_excluded_structures_listed(self, cfg):
        excluded = cfg.get("excluded_structures", [])
        for bad in ("5UGA", "5UG8", "5UGC", "5UWD", "4I21"):
            assert bad in excluded, f"{bad} missing from excluded_structures"

    def test_no_t790m_structures_in_active_list(self, cfg):
        t790m_ids = {"5UGA", "5UG8", "5UGC", "5UWD", "4I21"}
        for key, s in cfg["structures"].items():
            assert (
                s["pdb_id"] not in t790m_ids
            ), f"Structure {key} uses excluded T790M PDB {s['pdb_id']}"

    def test_preparation_ph(self, cfg):
        assert cfg["preparation"]["ph"] == pytest.approx(7.4)


# ── Integration tests (require prepared files) ────────────────────────────────


def _pdbqt_exists(pdb_id: str) -> bool:
    return (DOCK_DIR / f"{pdb_id}_receptor.pdbqt").exists()


@pytest.mark.integration
class TestPreparedFiles:
    """Integration tests: verify files produced by prepare_docking.py."""

    @pytest.fixture(autouse=True)
    def require_2itz(self):
        if not _pdbqt_exists("2ITZ"):
            pytest.skip(
                "2ITZ_receptor.pdbqt not found — run scripts/prepare_docking.py first"
            )

    def test_2itz_receptor_pdbqt_exists(self):
        assert (DOCK_DIR / "2ITZ_receptor.pdbqt").exists()

    def test_2ity_receptor_pdbqt_exists(self):
        if not _pdbqt_exists("2ITY"):
            pytest.skip("2ITY not prepared")
        assert (DOCK_DIR / "2ITY_receptor.pdbqt").exists()

    def test_pdbqt_contains_atom_records(self):
        pdbqt = (DOCK_DIR / "2ITZ_receptor.pdbqt").read_text(encoding="utf-8")
        atom_lines = [l for l in pdbqt.splitlines() if l.startswith("ATOM")]
        assert len(atom_lines) > 100, f"Expected >100 ATOM lines, got {len(atom_lines)}"

    def test_single_chain_in_pdbqt(self):
        pdbqt = (DOCK_DIR / "2ITZ_receptor.pdbqt").read_text(encoding="utf-8")
        atom_lines = [l for l in pdbqt.splitlines() if l.startswith("ATOM")]
        # Chain ID is at column 22 (0-indexed: col 21)
        chains = {line[21] for line in atom_lines if len(line) > 21}
        assert chains == {"A"}, f"Expected only chain A, found: {chains}"

    def test_no_water_in_pdbqt(self):
        pdbqt = (DOCK_DIR / "2ITZ_receptor.pdbqt").read_text(encoding="utf-8")
        # HOH residue name is columns 17-20
        assert "HOH" not in pdbqt, "Water (HOH) found in receptor PDBQT"

    def test_no_ire_in_receptor_pdbqt(self):
        pdbqt = (DOCK_DIR / "2ITZ_receptor.pdbqt").read_text(encoding="utf-8")
        assert "IRE" not in pdbqt, "Gefitinib (IRE) found in receptor PDBQT"

    def test_2itz_ligand_pdb_exists(self):
        assert (DOCK_DIR / "2ITZ_ligand.pdb").exists()

    def test_box_center_populated_in_config(self):
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        box = cfg["box"]
        assert (
            box["center_x"] is not None
        ), "box.center_x is null — run prepare_docking.py"
        assert box["center_y"] is not None
        assert box["center_z"] is not None

    def test_box_center_is_numeric(self):
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        box = cfg["box"]
        for axis in ("center_x", "center_y", "center_z"):
            val = box[axis]
            assert isinstance(val, (int, float)), f"{axis} is not numeric: {val!r}"

    def test_box_center_reproducible(self):
        # Recompute from the ligand PDB and verify it matches config

        ligand_pdb = DOCK_DIR / "2ITZ_ligand.pdb"
        if not ligand_pdb.exists():
            pytest.skip("2ITZ_ligand.pdb not found")

        cx, cy, cz = get_ligand_centroid(ligand_pdb, "IRE", "A")

        with open(CFG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        box = cfg["box"]

        assert (
            abs(cx - box["center_x"]) < 0.01
        ), f"x centroid mismatch: computed {cx:.3f}, config {box['center_x']}"
        assert (
            abs(cy - box["center_y"]) < 0.01
        ), f"y centroid mismatch: computed {cy:.3f}, config {box['center_y']}"
        assert (
            abs(cz - box["center_z"]) < 0.01
        ), f"z centroid mismatch: computed {cz:.3f}, config {box['center_z']}"
