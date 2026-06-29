"""
Tests for Phase B2 docking modules.

Unit tests (no Vina binary, no real PDB files):
  - align_structures: _collect_ca, verify_box_coverage logic
  - vina_runner: CLI argument assembly, error propagation
  - parse_results: parse REMARK VINA RESULT lines

Integration tests (marked 'integration', require prepared PDB files):
  - Alignment of 2ITY onto 2ITZ produces RMSD < 2.0 A
  - Aligned PDBQT written with correct atom count range
  - smiles_to_pdbqt generates a valid PDBQT for gefitinib
"""

from __future__ import annotations

import importlib.util
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Biopython (Bio) backs verify_box_coverage but is not a CI dependency, so skip
# the structure-alignment unit tests when it is absent.
_HAS_BIO = importlib.util.find_spec("Bio") is not None

from src.docking.parse_results import best_affinity, parse_vina_output
from src.docking.vina_runner import get_vina_exe, run_vina

# ── parse_results ─────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestParseVinaOutput:
    SAMPLE_PDBQT = textwrap.dedent(
        """\
        REMARK VINA RESULT:      -8.500      0.000      0.000
        REMARK VINA RESULT:      -8.200      1.543      2.311
        REMARK VINA RESULT:      -7.900      2.101      3.450
        ATOM      1  C1  LIG L   1       1.000   2.000   3.000  1.00  0.00      0.000 C
    """
    )

    def test_parse_returns_three_poses(self, tmp_path):
        f = tmp_path / "out.pdbqt"
        f.write_text(self.SAMPLE_PDBQT)
        poses = parse_vina_output(f)
        assert len(poses) == 3

    def test_mode_numbering_starts_at_1(self, tmp_path):
        f = tmp_path / "out.pdbqt"
        f.write_text(self.SAMPLE_PDBQT)
        poses = parse_vina_output(f)
        assert poses[0]["mode"] == 1
        assert poses[2]["mode"] == 3

    def test_affinity_values(self, tmp_path):
        f = tmp_path / "out.pdbqt"
        f.write_text(self.SAMPLE_PDBQT)
        poses = parse_vina_output(f)
        assert poses[0]["affinity"] == pytest.approx(-8.5)
        assert poses[1]["affinity"] == pytest.approx(-8.2)

    def test_rmsd_best_mode_is_zero(self, tmp_path):
        f = tmp_path / "out.pdbqt"
        f.write_text(self.SAMPLE_PDBQT)
        poses = parse_vina_output(f)
        assert poses[0]["rmsd_lb"] == pytest.approx(0.0)
        assert poses[0]["rmsd_ub"] == pytest.approx(0.0)

    def test_best_affinity_returns_most_negative(self, tmp_path):
        f = tmp_path / "out.pdbqt"
        f.write_text(self.SAMPLE_PDBQT)
        assert best_affinity(f) == pytest.approx(-8.5)

    def test_empty_file_returns_empty_list(self, tmp_path):
        f = tmp_path / "empty.pdbqt"
        f.write_text("ATOM  1  C1  LIG L   1  1.0  2.0  3.0  1.00 0.00  0.000 C\n")
        poses = parse_vina_output(f)
        assert poses == []

    def test_best_affinity_empty_returns_none(self, tmp_path):
        f = tmp_path / "empty.pdbqt"
        f.write_text("no results here\n")
        assert best_affinity(f) is None


# ── vina_runner ───────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestVinaRunner:
    BOX = {
        "center_x": -51.654,
        "center_y": -1.266,
        "center_z": -21.945,
        "size_x": 22.5,
        "size_y": 22.5,
        "size_z": 22.5,
    }

    def _fake_pdbqt(self, tmp_path: Path, name: str) -> Path:
        p = tmp_path / name
        p.write_text(
            "REMARK VINA RESULT:      -8.500      0.000      0.000\n"
            "ATOM      1  C1  LIG L   1       0.0   0.0   0.0  1.00  0.00      0.000 C\n"
        )
        return p

    def test_get_vina_exe_raises_if_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Vina binary not found"):
            get_vina_exe(tmp_path / "nonexistent.exe")

    def test_run_vina_assembles_correct_flags(self, tmp_path):
        receptor = self._fake_pdbqt(tmp_path, "rec.pdbqt")
        ligand = self._fake_pdbqt(tmp_path, "lig.pdbqt")
        fake_exe = tmp_path / "vina.exe"
        fake_exe.touch()

        captured_cmd = []

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            # Create expected output file so run_vina doesn't raise
            out_dir = Path(tmp_path) / "out"
            out_dir.mkdir(exist_ok=True)
            stem = f"{ligand.stem}__{receptor.stem}"
            (out_dir / f"{stem}_out.pdbqt").write_text(
                "REMARK VINA RESULT:      -8.500      0.000      0.000\n"
            )
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""
            return mock_result

        with patch("src.docking.vina_runner.subprocess.run", side_effect=fake_run):
            run_vina(
                receptor=receptor,
                ligand=ligand,
                out_dir=tmp_path / "out",
                box=self.BOX,
                vina_exe=fake_exe,
            )

        assert "--receptor" in captured_cmd
        assert "--ligand" in captured_cmd
        assert "--center_x" in captured_cmd
        assert "--size_x" in captured_cmd
        assert "--seed" in captured_cmd

    def test_run_vina_raises_on_nonzero_exit(self, tmp_path):
        receptor = self._fake_pdbqt(tmp_path, "rec.pdbqt")
        ligand = self._fake_pdbqt(tmp_path, "lig.pdbqt")
        fake_exe = tmp_path / "vina.exe"
        fake_exe.touch()

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "some error"

        with patch("src.docking.vina_runner.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="Vina failed"):
                run_vina(
                    receptor, ligand, tmp_path / "out", self.BOX, vina_exe=fake_exe
                )

    def test_center_x_formatted_to_3dp(self, tmp_path):
        receptor = self._fake_pdbqt(tmp_path, "rec.pdbqt")
        ligand = self._fake_pdbqt(tmp_path, "lig.pdbqt")
        fake_exe = tmp_path / "vina.exe"
        fake_exe.touch()

        captured_cmd = []

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            out_dir = tmp_path / "out"
            out_dir.mkdir(exist_ok=True)
            stem = f"{ligand.stem}__{receptor.stem}"
            (out_dir / f"{stem}_out.pdbqt").write_text(
                "REMARK VINA RESULT:      -7.000      0.000      0.000\n"
            )
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""
            return mock_result

        with patch("src.docking.vina_runner.subprocess.run", side_effect=fake_run):
            run_vina(receptor, ligand, tmp_path / "out", self.BOX, vina_exe=fake_exe)

        cx_idx = captured_cmd.index("--center_x") + 1
        assert captured_cmd[cx_idx] == "-51.654"


# ── align_structures (unit) ───────────────────────────────────────────────────


@pytest.mark.skipif(not _HAS_BIO, reason="Biopython not installed")
@pytest.mark.unit
class TestAlignStructuresUnit:
    def test_verify_box_coverage_centroid_inside(self, tmp_path):
        """Ca centroid well inside the box passes."""
        from src.docking.align_structures import verify_box_coverage

        # Create a minimal PDB with one CA atom at the box center
        pdb = tmp_path / "single_ca.pdb"
        pdb.write_text(
            "ATOM      1  CA  ALA A   1     -51.654  -1.266 -21.945  1.00  0.00           C\n"
            "END\n"
        )
        box = {
            "center_x": -51.654,
            "center_y": -1.266,
            "center_z": -21.945,
            "size_x": 22.5,
            "size_y": 22.5,
            "size_z": 22.5,
        }
        assert verify_box_coverage(pdb, box)

    def test_verify_box_coverage_centroid_outside(self, tmp_path):
        """Ca centroid far outside the box fails."""
        from src.docking.align_structures import verify_box_coverage

        pdb = tmp_path / "far_ca.pdb"
        pdb.write_text(
            "ATOM      1  CA  ALA A   1     100.000 100.000 100.000  1.00  0.00           C\n"
            "END\n"
        )
        box = {
            "center_x": -51.654,
            "center_y": -1.266,
            "center_z": -21.945,
            "size_x": 22.5,
            "size_y": 22.5,
            "size_z": 22.5,
        }
        assert not verify_box_coverage(pdb, box)


# ── Integration: alignment on real PDB files ──────────────────────────────────

PROTEIN_DIR = Path(__file__).resolve().parent.parent / "data" / "docking" / "protein"
_REAL_FILES = pytest.mark.skipif(
    not (PROTEIN_DIR / "2ITZ_prepared.pdb").exists()
    or not (PROTEIN_DIR / "2ITY_prepared.pdb").exists(),
    reason="Prepared PDB files not present (run prepare_docking.py first)",
)


@pytest.mark.integration
class TestAlignmentIntegration:
    @_REAL_FILES
    def test_alignment_rmsd_below_2_angstrom(self, tmp_path):
        """2ITY vs 2ITZ should have Ca RMSD < 2 A (same construct, similar conformation)."""
        from src.docking.align_structures import align_wt_to_l858r

        aligned = tmp_path / "2ITY_aligned.pdb"
        rmsd = align_wt_to_l858r(
            PROTEIN_DIR / "2ITZ_prepared.pdb",
            PROTEIN_DIR / "2ITY_prepared.pdb",
            aligned,
        )
        assert aligned.exists()
        assert 0.0 < rmsd < 2.0, f"Unexpected Ca RMSD: {rmsd:.3f} A"

    @_REAL_FILES
    def test_aligned_pdbqt_atom_count_reasonable(self, tmp_path):
        """Aligned receptor PDBQT should have ~2300-2600 atoms (similar to 2ITY_receptor.pdbqt)."""
        from src.docking.align_structures import align_wt_to_l858r
        from src.docking.prepare_protein import write_receptor_pdbqt

        aligned_pdb = tmp_path / "2ITY_aligned.pdb"
        aligned_pdbqt = tmp_path / "2ITY_aligned_receptor.pdbqt"

        align_wt_to_l858r(
            PROTEIN_DIR / "2ITZ_prepared.pdb",
            PROTEIN_DIR / "2ITY_prepared.pdb",
            aligned_pdb,
        )
        write_receptor_pdbqt(aligned_pdb, aligned_pdbqt)

        text = aligned_pdbqt.read_text()
        n_atoms = sum(1 for l in text.splitlines() if l.startswith("ATOM"))
        assert 2000 < n_atoms < 3000, f"Unexpected atom count: {n_atoms}"

    @_REAL_FILES
    def test_box_covers_both_receptors_after_alignment(self, tmp_path):
        """After alignment, the shared box should cover both receptor Ca centroids."""
        import yaml

        from src.docking.align_structures import align_wt_to_l858r, verify_box_coverage

        cfg_path = (
            Path(__file__).resolve().parent.parent / "config" / "docking_config.yaml"
        )
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        box = {
            k: float(cfg["box"][k])
            for k in ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z")
        }

        aligned_pdb = tmp_path / "2ITY_aligned.pdb"
        align_wt_to_l858r(
            PROTEIN_DIR / "2ITZ_prepared.pdb",
            PROTEIN_DIR / "2ITY_prepared.pdb",
            aligned_pdb,
        )

        assert verify_box_coverage(PROTEIN_DIR / "2ITZ_prepared.pdb", box)
        assert verify_box_coverage(aligned_pdb, box)


# ── Integration: smiles_to_pdbqt ─────────────────────────────────────────────


@pytest.mark.integration
class TestSmilesToPdbqtIntegration:
    GEFITINIB = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"

    def test_gefitinib_pdbqt_written(self, tmp_path):
        from src.docking.prepare_ligands import smiles_to_pdbqt

        out = tmp_path / "gefitinib.pdbqt"
        smiles_to_pdbqt(self.GEFITINIB, out, name="gefitinib")
        assert out.exists()
        text = out.read_text()
        assert len(text.splitlines()) > 10

    def test_pdbqt_contains_torsion_record(self, tmp_path):
        from src.docking.prepare_ligands import smiles_to_pdbqt

        out = tmp_path / "gefitinib.pdbqt"
        smiles_to_pdbqt(self.GEFITINIB, out, name="gefitinib")
        text = out.read_text()
        assert "TORSDOF" in text

    def test_invalid_smiles_raises(self, tmp_path):
        from src.docking.prepare_ligands import smiles_to_pdbqt

        with pytest.raises((ValueError, RuntimeError)):
            smiles_to_pdbqt("not_a_smiles", tmp_path / "bad.pdbqt")
