"""
GNINA CNN rescoring wrapper for Windows / WSL2.

GNINA (https://github.com/gnina/gnina) is a pretrained CNN pose-rescorer.
The official Linux binary has no Windows build; on this machine it runs via
WSL2 Ubuntu, called with ``wsl -d Ubuntu -- /mnt/c/... gnina ...``.

Route selected after feasibility check (June 2026):
- gnina v1.3.2 CPU build requires libcudnn.so.9 — missing in WSL2 without GPU drivers.
- gnina v1.0 (Dec 2021) is statically linked against all available libs in Ubuntu 24.04
  WSL2 and runs on CPU with no CUDA requirement.
- Binary stored at data/docking/tools/gnina_v1.0 (535 MB).

Key outputs per pose (from gnina stdout):
  Affinity:     smina/Vina-compatible score (kcal/mol, more negative = better)
  CNNscore:     CNN-predicted binding probability (0–1, higher = better)
  CNNaffinity:  CNN-predicted pKd (higher = tighter; 1 pKd unit ≈ 1.36 kcal/mol)
  CNNvariance:  model ensemble variance (high value = low confidence)
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_GNINA_WIN = _PROJECT_ROOT / "data" / "docking" / "tools" / "gnina_v1.0"
_WSL_DISTRO = "Ubuntu"

# Meeko REMARK tags that confuse gnina v1.0
_MEEKO_SKIP_PREFIXES = (
    "REMARK SMILES",
    "REMARK H PARENT",
    "REMARK INTER",
    "REMARK UNBOUND",
)


# ── Path conversion ───────────────────────────────────────────────────────────


def win_to_wsl(path: Path) -> str:
    """Convert an absolute Windows path to its WSL2 /mnt/<drive>/... equivalent."""
    p = Path(path).resolve()
    drive = p.drive.lower().rstrip(":")  # "c:" -> "c"
    rest = str(p)[len(p.drive) :].replace("\\", "/")  # "\Users\..." -> "/Users/..."
    return f"/mnt/{drive}{rest}"


# ── Pose extraction ───────────────────────────────────────────────────────────


def extract_best_pose(vina_pdbqt: Path, out_path: Path) -> Path:
    """
    Extract MODEL 1 from a multi-MODEL Vina output PDBQT and write a
    single-pose PDBQT compatible with gnina --score_only.

    Strips:
    - null bytes (meeko artifact from writing via Windows filesystem)
    - meeko-specific REMARK lines (SMILES IDX, H PARENT, INTER, UNBOUND)
    - END / ENDMDL records (not accepted by gnina v1.0)
    - empty / whitespace-only lines
    """
    raw = Path(vina_pdbqt).read_bytes().replace(b"\x00", b"")
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()

    out_lines: list[str] = []
    in_model1 = False
    for line in lines:
        s = line.strip()
        if s == "MODEL 1":
            in_model1 = True
            continue
        if in_model1:
            if s == "ENDMDL":
                break
            if not s or s == "END":
                continue
            if any(s.startswith(p) for p in _MEEKO_SKIP_PREFIXES):
                continue
            out_lines.append(s + "\n")

    if not out_lines:
        raise ValueError(f"extract_best_pose: no MODEL 1 content found in {vina_pdbqt}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(out_lines), encoding="utf-8")
    return out_path


# ── GNINA stdout parser ───────────────────────────────────────────────────────

_AFFINITY_RE = re.compile(r"^Affinity:\s*([-\d.]+)")
_CNN_SCORE_RE = re.compile(r"^CNNscore:\s*([\d.]+)")
_CNN_AFF_RE = re.compile(r"^CNNaffinity:\s*([-\d.]+)")
_CNN_VAR_RE = re.compile(r"^CNNvariance:\s*([\d.]+)")


def parse_gnina_stdout(stdout: str) -> dict:
    """
    Parse gnina --score_only stdout and return a dict with numeric scores.

    Returns
    -------
    dict with keys: affinity_kcal, cnn_score, cnn_affinity_pkd, cnn_variance
    All values are float.  Raises ValueError if any key is missing.
    """
    result: dict[str, float] = {}
    for line in stdout.splitlines():
        m = _AFFINITY_RE.match(line)
        if m:
            result["affinity_kcal"] = float(m.group(1))
        m = _CNN_SCORE_RE.match(line)
        if m:
            result["cnn_score"] = float(m.group(1))
        m = _CNN_AFF_RE.match(line)
        if m:
            result["cnn_affinity_pkd"] = float(m.group(1))
        m = _CNN_VAR_RE.match(line)
        if m:
            result["cnn_variance"] = float(m.group(1))

    required = {"affinity_kcal", "cnn_score", "cnn_affinity_pkd", "cnn_variance"}
    missing = required - result.keys()
    if missing:
        raise ValueError(
            f"parse_gnina_stdout: missing fields {missing}.\n"
            f"GNINA stdout was:\n{stdout[:800]}"
        )
    return result


# ── Main rescoring function ───────────────────────────────────────────────────


def rescore_pose(
    receptor: Path,
    best_pose_pdbqt: Path,
    out_dir: Path,
    gnina_win_path: Path | None = None,
    wsl_distro: str = _WSL_DISTRO,
    cnn_scoring: str = "rescore",
    timeout: int = 120,
) -> dict:
    """
    Rescore a single-pose PDBQT with GNINA via WSL2 and return CNN scores.

    Parameters
    ----------
    receptor        : PDBQT receptor file (Windows path, converted to WSL2 internally)
    best_pose_pdbqt : single-pose PDBQT extracted by extract_best_pose()
    out_dir         : directory for GNINA output PDBQT (created if missing)
    gnina_win_path  : Windows path to gnina binary; defaults to gnina_v1.0
    wsl_distro      : WSL2 distro name (default "Ubuntu")
    cnn_scoring     : gnina --cnn_scoring value (default "rescore" = ensemble)
    timeout         : subprocess timeout in seconds

    Returns
    -------
    dict: affinity_kcal, cnn_score, cnn_affinity_pkd, cnn_variance
    """
    gnina_win = Path(gnina_win_path) if gnina_win_path else _GNINA_WIN
    if not gnina_win.exists():
        raise FileNotFoundError(
            f"GNINA binary not found: {gnina_win}\n"
            f"Download gnina v1.0 from https://github.com/gnina/gnina/releases/tag/v1.0\n"
            f"and place at data/docking/tools/gnina_v1.0"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(best_pose_pdbqt).stem}__{Path(receptor).stem}"
    out_pdbqt = out_dir / f"{stem}_gnina_out.pdbqt"

    # Convert Windows paths to WSL2 /mnt/c/... paths
    gnina_wsl = win_to_wsl(gnina_win)
    rec_wsl = win_to_wsl(receptor)
    lig_wsl = win_to_wsl(best_pose_pdbqt)
    out_wsl = win_to_wsl(out_pdbqt)

    cmd = [
        "wsl",
        "-d",
        wsl_distro,
        "--",
        gnina_wsl,
        "--receptor",
        rec_wsl,
        "--ligand",
        lig_wsl,
        "--score_only",
        "--cnn_scoring",
        cnn_scoring,
        "--out",
        out_wsl,
    ]

    logger.info(f"GNINA rescore: {Path(best_pose_pdbqt).name} x {Path(receptor).name}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"GNINA timed out after {timeout}s: {Path(best_pose_pdbqt).name}"
        )

    stdout_all = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(
            f"GNINA failed (exit {result.returncode})\n"
            f"CMD: {' '.join(cmd)}\n"
            f"OUTPUT: {stdout_all[:600]}"
        )

    scores = parse_gnina_stdout(stdout_all)
    logger.info(
        f"  affinity={scores['affinity_kcal']:.3f} kcal/mol  "
        f"CNNscore={scores['cnn_score']:.3f}  "
        f"CNNaffinity={scores['cnn_affinity_pkd']:.3f} pKd"
    )
    return scores
