"""
Re-apply covalent warhead detection to existing docking result JSONs.

Run after any change to src/features/covalent.py WARHEAD_SMARTS to propagate
updated confidence labels without re-running expensive docking.

Updates in place:
    models/qsar/library_docking_results.json
        compound.warheads, compound.docking_confidence

    models/qsar/docking_noise_results.json
        compound.warheads, compound.docking_confidence, compound.call
        (call is re-derived from noise_stats + new docking_confidence)

Run:
    PYTHONPATH=. .venv/Scripts/python.exe scripts/relabel_covalent.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.covalent import covalent_confidence, detect_warheads
from src.utils.logging import get_logger

logger = get_logger(__name__)

LIBRARY_PATH = PROJECT_ROOT / "models" / "qsar" / "library_docking_results.json"
NOISE_PATH = PROJECT_ROOT / "models" / "qsar" / "docking_noise_results.json"

# Must match THRESHOLD in eval_docking_noise.py
_THRESHOLD = 1.5


def _classify_call(
    delta: float | None, std_delta: float | None, docking_confidence: str
) -> str:
    """Re-derive the selectivity call given updated docking_confidence."""
    if docking_confidence == "low_confidence":
        return "low_confidence_covalent"
    if delta is None or std_delta is None:
        return "ambiguous"
    if std_delta == 0.0:
        return (
            "L858R_selective"
            if delta < 0
            else ("WT_selective" if delta > 0 else "ambiguous")
        )
    if abs(delta) > _THRESHOLD * std_delta:
        return "L858R_selective" if delta < 0 else "WT_selective"
    return "ambiguous"


def relabel_library(path: Path) -> dict[str, str]:
    """
    Re-apply warhead detection to library docking results.

    Returns a dict mapping cid -> change description for compounds that changed.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    changes: dict[str, str] = {}

    for c in data["compounds"]:
        smiles = c["smiles"]
        old_wh = c.get("warheads", [])
        old_conf = c.get("docking_confidence", "standard")

        new_wh = detect_warheads(smiles)
        new_conf = covalent_confidence(smiles)

        if sorted(new_wh) != sorted(old_wh) or new_conf != old_conf:
            changes[c["cid"]] = (
                f"warheads {old_wh!r} -> {new_wh!r}; "
                f"confidence {old_conf!r} -> {new_conf!r}"
            )
            c["warheads"] = new_wh
            c["docking_confidence"] = new_conf

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return changes


def relabel_noise(path: Path) -> dict[str, str]:
    """
    Re-apply warhead detection + re-derive selectivity call for noise results.

    Returns a dict mapping cid -> change description for compounds that changed.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    changes: dict[str, str] = {}

    for c in data["compounds"]:
        smiles = c["smiles"]
        old_wh = c.get("warheads", [])
        old_conf = c.get("docking_confidence", "standard")
        old_call = c.get("call", "")

        new_wh = detect_warheads(smiles)
        new_conf = covalent_confidence(smiles)

        stats = c.get("noise_stats") or {}
        delta = stats.get("delta")
        std_d = stats.get("std_delta")
        new_call = _classify_call(delta, std_d, new_conf)

        if (
            sorted(new_wh) != sorted(old_wh)
            or new_conf != old_conf
            or new_call != old_call
        ):
            changes[c["cid"]] = (
                f"warheads {old_wh!r} -> {new_wh!r}; "
                f"confidence {old_conf!r} -> {new_conf!r}; "
                f"call {old_call!r} -> {new_call!r}"
            )
            c["warheads"] = new_wh
            c["docking_confidence"] = new_conf
            c["call"] = new_call

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return changes


def main() -> None:
    logger.info("=== relabel_covalent: re-applying warhead labels ===")

    for p in (LIBRARY_PATH, NOISE_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}. Run docking scripts first.")

    lib_changes = relabel_library(LIBRARY_PATH)
    noise_changes = relabel_noise(NOISE_PATH)

    if lib_changes:
        print(f"\nLibrary docking ({LIBRARY_PATH.name}) — {len(lib_changes)} changed:")
        for cid, desc in sorted(lib_changes.items()):
            print(f"  {cid}: {desc}")
    else:
        print("\nLibrary docking: no changes.")

    if noise_changes:
        print(f"\nNoise eval ({NOISE_PATH.name}) — {len(noise_changes)} changed:")
        for cid, desc in sorted(noise_changes.items()):
            print(f"  {cid}: {desc}")
    else:
        print("\nNoise eval: no changes.")

    total = len(lib_changes) + len(noise_changes)
    print(f"\nTotal compounds relabeled: {total}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
