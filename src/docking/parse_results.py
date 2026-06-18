"""
Parse AutoDock Vina output PDBQT files.

Vina writes one REMARK VINA RESULT line per pose in the output PDBQT:

  REMARK VINA RESULT:      -8.500      0.000      0.000
  REMARK VINA RESULT:      -8.200      1.543      2.311

Fields: affinity (kcal/mol), rmsd_lower_bound, rmsd_upper_bound.
The best pose (mode 1) always has rmsd = 0.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)

_RESULT_RE = re.compile(r"REMARK VINA RESULT:\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)")


def parse_vina_output(pdbqt_path: Path) -> list[dict]:
    """
    Parse a Vina output PDBQT file and return a list of pose dicts.

    Each dict contains:
      mode     : int  (1 = best pose)
      affinity : float  kcal/mol; more negative = more favorable
      rmsd_lb  : float  RMSD lower bound from best mode (A)
      rmsd_ub  : float  RMSD upper bound from best mode (A)

    Returns an empty list if no VINA RESULT lines are found (failed job).
    """
    text = Path(pdbqt_path).read_text(encoding="utf-8", errors="replace")
    poses = []
    for i, m in enumerate(_RESULT_RE.finditer(text), start=1):
        poses.append(
            {
                "mode": i,
                "affinity": float(m.group(1)),
                "rmsd_lb": float(m.group(2)),
                "rmsd_ub": float(m.group(3)),
            }
        )
    if not poses:
        logger.warning(f"parse_vina_output: no VINA RESULT lines in {pdbqt_path}")
    return poses


def best_affinity(pdbqt_path: Path) -> float | None:
    """
    Return the best (most negative) docking affinity in kcal/mol, or None on failure.
    """
    poses = parse_vina_output(pdbqt_path)
    if not poses:
        return None
    return min(p["affinity"] for p in poses)
