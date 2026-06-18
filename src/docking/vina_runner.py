"""
AutoDock Vina subprocess wrapper.

Uses the pre-built vina.exe binary at data/docking/tools/vina.exe via subprocess
rather than the vina Python package, which requires Boost C++ headers and fails
to compile on Windows without Visual Studio build tools.

Download the binary from:
  https://github.com/ccsb-scripps/AutoDock-Vina/releases
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)

_VINA_EXE_DEFAULT = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "docking"
    / "tools"
    / "vina.exe"
)


def get_vina_exe(override: Path | None = None) -> Path:
    """Return the Vina executable path, raising FileNotFoundError if absent."""
    exe = Path(override) if override else _VINA_EXE_DEFAULT
    if not exe.exists():
        raise FileNotFoundError(
            f"Vina binary not found: {exe}\n"
            f"Download from https://github.com/ccsb-scripps/AutoDock-Vina/releases"
        )
    return exe


def run_vina(
    receptor: Path,
    ligand: Path,
    out_dir: Path,
    box: dict[str, float],
    n_poses: int = 9,
    exhaustiveness: int = 8,
    seed: int = 42,
    vina_exe: Path | None = None,
    timeout: int = 300,
) -> tuple[Path, Path]:
    """
    Run AutoDock Vina on one receptor-ligand pair via subprocess.

    Parameters
    ----------
    receptor      : PDBQT receptor file
    ligand        : PDBQT ligand file
    out_dir       : directory for output files (created if missing)
    box           : dict with center_x/y/z and size_x/y/z (Angstrom)
    n_poses       : number of binding poses to generate
    exhaustiveness: search exhaustiveness (Vina default = 8)
    seed          : random seed for reproducible poses
    vina_exe      : path to vina binary; defaults to data/docking/tools/vina.exe
    timeout       : subprocess timeout in seconds

    Returns
    -------
    (out_pdbqt, log_file) Path objects
    """
    exe = get_vina_exe(vina_exe)
    receptor = Path(receptor)
    ligand = Path(ligand)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{ligand.stem}__{receptor.stem}"
    out_pdbqt = out_dir / f"{stem}_out.pdbqt"
    log_file = out_dir / f"{stem}.log"

    cmd = [
        str(exe),
        "--receptor",
        str(receptor),
        "--ligand",
        str(ligand),
        "--center_x",
        f"{box['center_x']:.3f}",
        "--center_y",
        f"{box['center_y']:.3f}",
        "--center_z",
        f"{box['center_z']:.3f}",
        "--size_x",
        f"{box['size_x']:.3f}",
        "--size_y",
        f"{box['size_y']:.3f}",
        "--size_z",
        f"{box['size_z']:.3f}",
        "--out",
        str(out_pdbqt),
        "--exhaustiveness",
        str(exhaustiveness),
        "--num_modes",
        str(n_poses),
        "--seed",
        str(seed),
    ]

    logger.info(f"Vina: {ligand.name} x {receptor.name}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Vina timed out after {timeout}s: {ligand.name} x {receptor.name}"
        )

    # Vina 1.2.7 removed --log; write captured stdout to the log file
    log_output = result.stdout + result.stderr
    log_file.write_text(log_output, encoding="utf-8")

    if result.returncode != 0:
        raise RuntimeError(
            f"Vina failed (exit {result.returncode})\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT: {result.stdout[:500]}\n"
            f"STDERR: {result.stderr[:500]}"
        )

    if not out_pdbqt.exists():
        raise RuntimeError(f"Vina exited 0 but output not found: {out_pdbqt}")

    logger.info(f"Vina OK -> {out_pdbqt.name}")
    return out_pdbqt, log_file
