"""
Tests for Phase B3 GNINA rescoring modules.

Unit tests (no WSL2, no binary required):
  - win_to_wsl: path conversion correctness
  - extract_best_pose: MODEL 1 extraction, null-byte stripping, meeko REMARK removal
  - parse_gnina_stdout: parse known stdout strings
  - parse_gnina_stdout: raises on incomplete output

Integration tests (marked 'integration', require WSL2 + gnina_v1.0 binary):
  - rescore_pose: produces expected keys for one real receptor/pose pair
  - CNNscore is in (0, 1)
  - CNNaffinity is positive (pKd, expected range 4–10 for drug-like molecules)
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.docking.gnina_runner import (
    extract_best_pose,
    parse_gnina_stdout,
    win_to_wsl,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_VINA_PDBQT = textwrap.dedent(
    """\
    MODEL 1
    REMARK VINA RESULT:    -7.860      0.000      0.000
    REMARK INTER + INTRA:         -12.884
    REMARK INTER:                 -11.536
    REMARK INTRA:                  -1.347
    REMARK UNBOUND:                -1.347
    REMARK SMILES COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1
    REMARK SMILES IDX 22 1 21 2
    REMARK H PARENT 10 15
    ROOT
    ATOM      1  O   UNL     1     -49.318  -1.286 -19.916  1.00  0.00    -0.490 OA
    ENDROOT
    BRANCH   1   2
    ATOM      2  C   UNL     1     -50.520  -0.726 -20.262  1.00  0.00     0.162 A
    ENDBRANCH   1   2
    TORSDOF 1
    ENDMDL
    MODEL 2
    REMARK VINA RESULT:    -7.500      1.200      2.100
    ROOT
    ATOM      1  O   UNL     1     -48.000  -0.900 -18.500  1.00  0.00    -0.490 OA
    ENDROOT
    TORSDOF 1
    ENDMDL
"""
)

SAMPLE_GNINA_STDOUT = textwrap.dedent(
    """\
    WARNING: No GPU detected. CNN scoring will be slow.
    gnina v1.0 HEAD:6381355   Built Mar  6 2021.

    Affinity: -7.86185 (kcal/mol)
    CNNscore: 0.47940
    CNNaffinity: 6.69269
    CNNvariance: 1.46309
    Intramolecular energy: -1.34836
"""
)


# ── win_to_wsl ────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestWinToWsl:
    def test_c_drive_conversion(self):
        p = Path(r"C:\Users\foo\bar.pdbqt")
        result = win_to_wsl(p)
        assert result == "/mnt/c/Users/foo/bar.pdbqt"

    def test_d_drive_conversion(self):
        p = Path(r"D:\data\file.txt")
        result = win_to_wsl(p)
        assert result.startswith("/mnt/d/")

    def test_backslashes_converted(self):
        p = Path(r"C:\a\b\c.pdbqt")
        assert "\\" not in win_to_wsl(p)

    def test_drive_letter_lowercased(self):
        p = Path(r"C:\foo")
        assert win_to_wsl(p).startswith("/mnt/c/")


# ── extract_best_pose ─────────────────────────────────────────────────────────


@pytest.mark.unit
class TestExtractBestPose:
    def test_extracts_model1_only(self, tmp_path):
        infile = tmp_path / "multi.pdbqt"
        infile.write_text(SAMPLE_VINA_PDBQT, encoding="utf-8")
        out = tmp_path / "best.pdbqt"
        extract_best_pose(infile, out)
        text = out.read_text()
        assert "MODEL 1" not in text
        assert "MODEL 2" not in text
        assert "ENDMDL" not in text

    def test_contains_atom_records(self, tmp_path):
        infile = tmp_path / "multi.pdbqt"
        infile.write_text(SAMPLE_VINA_PDBQT, encoding="utf-8")
        out = tmp_path / "best.pdbqt"
        extract_best_pose(infile, out)
        lines = out.read_text().splitlines()
        assert any(l.startswith("ATOM") for l in lines)

    def test_strips_meeko_smiles_remark(self, tmp_path):
        infile = tmp_path / "multi.pdbqt"
        infile.write_text(SAMPLE_VINA_PDBQT, encoding="utf-8")
        out = tmp_path / "best.pdbqt"
        extract_best_pose(infile, out)
        text = out.read_text()
        assert "REMARK SMILES" not in text
        assert "REMARK H PARENT" not in text

    def test_strips_null_bytes(self, tmp_path):
        raw = SAMPLE_VINA_PDBQT.encode("utf-8")
        # Inject null bytes as meeko does on Windows
        raw_with_nulls = raw + b"\x00" * 64
        infile = tmp_path / "nulls.pdbqt"
        infile.write_bytes(raw_with_nulls)
        out = tmp_path / "best.pdbqt"
        extract_best_pose(infile, out)
        content = out.read_bytes()
        assert b"\x00" not in content

    def test_keeps_vina_remark_result(self, tmp_path):
        infile = tmp_path / "multi.pdbqt"
        infile.write_text(SAMPLE_VINA_PDBQT, encoding="utf-8")
        out = tmp_path / "best.pdbqt"
        extract_best_pose(infile, out)
        assert "REMARK VINA RESULT" in out.read_text()

    def test_keeps_root_and_branch(self, tmp_path):
        infile = tmp_path / "multi.pdbqt"
        infile.write_text(SAMPLE_VINA_PDBQT, encoding="utf-8")
        out = tmp_path / "best.pdbqt"
        extract_best_pose(infile, out)
        text = out.read_text()
        assert "ROOT" in text
        assert "BRANCH" in text

    def test_raises_on_empty_model1(self, tmp_path):
        pdbqt = tmp_path / "bad.pdbqt"
        pdbqt.write_text("MODEL 2\nATOM 1 C UNL 1 0 0 0 1.00 0.00 0.000 C\nENDMDL\n")
        with pytest.raises(ValueError, match="MODEL 1"):
            extract_best_pose(pdbqt, tmp_path / "out.pdbqt")

    def test_second_model_atoms_excluded(self, tmp_path):
        infile = tmp_path / "multi.pdbqt"
        infile.write_text(SAMPLE_VINA_PDBQT, encoding="utf-8")
        out = tmp_path / "best.pdbqt"
        extract_best_pose(infile, out)
        text = out.read_text()
        # The WT O atom in MODEL 2 has x=-48.000 — should not appear
        assert "-48.000" not in text
        # The MODEL 1 O atom has x=-49.318 — must appear
        assert "-49.318" in text


# ── parse_gnina_stdout ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestParseGninaStdout:
    def test_parses_affinity(self):
        result = parse_gnina_stdout(SAMPLE_GNINA_STDOUT)
        assert result["affinity_kcal"] == pytest.approx(-7.86185)

    def test_parses_cnn_score(self):
        result = parse_gnina_stdout(SAMPLE_GNINA_STDOUT)
        assert result["cnn_score"] == pytest.approx(0.47940, abs=1e-4)

    def test_parses_cnn_affinity(self):
        result = parse_gnina_stdout(SAMPLE_GNINA_STDOUT)
        assert result["cnn_affinity_pkd"] == pytest.approx(6.69269, abs=1e-4)

    def test_parses_cnn_variance(self):
        result = parse_gnina_stdout(SAMPLE_GNINA_STDOUT)
        assert result["cnn_variance"] == pytest.approx(1.46309, abs=1e-4)

    def test_raises_on_missing_cnn_score(self):
        stdout_no_cnn = (
            "Affinity: -7.5 (kcal/mol)\nCNNaffinity: 6.0\nCNNvariance: 1.0\n"
        )
        with pytest.raises(ValueError, match="missing fields"):
            parse_gnina_stdout(stdout_no_cnn)

    def test_raises_on_empty_stdout(self):
        with pytest.raises(ValueError, match="missing fields"):
            parse_gnina_stdout("")

    def test_returns_all_four_keys(self):
        result = parse_gnina_stdout(SAMPLE_GNINA_STDOUT)
        assert set(result.keys()) == {
            "affinity_kcal",
            "cnn_score",
            "cnn_affinity_pkd",
            "cnn_variance",
        }

    def test_all_values_are_float(self):
        result = parse_gnina_stdout(SAMPLE_GNINA_STDOUT)
        for v in result.values():
            assert isinstance(v, float)


# ── Integration: real rescoring ───────────────────────────────────────────────

_PROTEIN_DIR = Path(__file__).resolve().parent.parent / "data" / "docking" / "protein"
_SANITY_DIR = (
    Path(__file__).resolve().parent.parent / "data" / "docking" / "results" / "sanity"
)
_GNINA_BIN = (
    Path(__file__).resolve().parent.parent / "data" / "docking" / "tools" / "gnina_v1.0"
)

_real_files = pytest.mark.skipif(
    not (
        _GNINA_BIN.exists()
        and (_SANITY_DIR / "gefitinib__2ITZ_receptor_out.pdbqt").exists()
    ),
    reason="gnina_v1.0 binary or B2 Vina output not present",
)


@pytest.mark.integration
class TestGninaRescoringIntegration:
    @_real_files
    def test_rescore_returns_expected_keys(self, tmp_path):
        from src.docking.gnina_runner import rescore_pose

        vina_out = _SANITY_DIR / "gefitinib__2ITZ_receptor_out.pdbqt"
        best_pose = tmp_path / "best.pdbqt"
        extract_best_pose(vina_out, best_pose)

        scores = rescore_pose(
            receptor=_PROTEIN_DIR / "2ITZ_receptor.pdbqt",
            best_pose_pdbqt=best_pose,
            out_dir=tmp_path,
        )
        assert set(scores.keys()) == {
            "affinity_kcal",
            "cnn_score",
            "cnn_affinity_pkd",
            "cnn_variance",
        }

    @_real_files
    def test_cnn_score_in_0_to_1(self, tmp_path):
        from src.docking.gnina_runner import rescore_pose

        vina_out = _SANITY_DIR / "gefitinib__2ITZ_receptor_out.pdbqt"
        best_pose = tmp_path / "best.pdbqt"
        extract_best_pose(vina_out, best_pose)

        scores = rescore_pose(
            receptor=_PROTEIN_DIR / "2ITZ_receptor.pdbqt",
            best_pose_pdbqt=best_pose,
            out_dir=tmp_path,
        )
        assert 0.0 <= scores["cnn_score"] <= 1.0

    @_real_files
    def test_cnn_affinity_positive_pkd_range(self, tmp_path):
        """Drug-like binders should have pKd in roughly 4–12 range."""
        from src.docking.gnina_runner import rescore_pose

        vina_out = _SANITY_DIR / "gefitinib__2ITZ_receptor_out.pdbqt"
        best_pose = tmp_path / "best.pdbqt"
        extract_best_pose(vina_out, best_pose)

        scores = rescore_pose(
            receptor=_PROTEIN_DIR / "2ITZ_receptor.pdbqt",
            best_pose_pdbqt=best_pose,
            out_dir=tmp_path,
        )
        assert 3.0 < scores["cnn_affinity_pkd"] < 15.0

    @_real_files
    def test_l858r_gefitinib_cnn_affinity_above_wt(self, tmp_path):
        """Gefitinib should have higher CNNaffinity (pKd) on L858R than WT.

        CNNscore is near 0.48 for both pockets and does not reliably distinguish
        them (B3 verdict: BORDERLINE).  CNNaffinity (+0.303 pKd in favour of L858R)
        is the metric that shows consistent L858R preference.
        """
        from src.docking.gnina_runner import rescore_pose

        for pocket, stem in [
            ("L858R", "2ITZ_receptor"),
            ("WT", "2ITY_aligned_receptor"),
        ]:
            vina_out = _SANITY_DIR / f"gefitinib__{stem}_out.pdbqt"
            best_pose = tmp_path / f"gef_{pocket}.pdbqt"
            extract_best_pose(vina_out, best_pose)

        rec_l858r = _PROTEIN_DIR / "2ITZ_receptor.pdbqt"
        rec_wt = _PROTEIN_DIR / "2ITY_aligned_receptor.pdbqt"

        s_l858r = rescore_pose(rec_l858r, tmp_path / "gef_L858R.pdbqt", tmp_path)
        s_wt = rescore_pose(rec_wt, tmp_path / "gef_WT.pdbqt", tmp_path)

        assert s_l858r["cnn_affinity_pkd"] > s_wt["cnn_affinity_pkd"], (
            f"Gefitinib CNNaffinity: L858R={s_l858r['cnn_affinity_pkd']:.3f} pKd "
            f"not > WT={s_wt['cnn_affinity_pkd']:.3f} pKd"
        )
