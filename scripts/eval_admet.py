"""
ADMET evaluation for the EGFR docked candidate sets.

Applies src/admet/filters to two candidate pools:

  1. Shortlist (n=7): non-covalent L858R-selective compounds that cleared the
     docking noise filter (call="L858R_selective", docking_confidence="standard")
     from eval_docking_noise.py.

  2. Top-50 library (n<=50): all compounds from the first-pass library docking
     (library_docking_results.json).

All ADMET results are APPROXIMATE — computational estimates only.
admet_status ("pass"/"flag") is a soft ranking signal (composite weight 0.20),
not a hard exclusion criterion.

Run:
    PYTHONPATH=. .venv/Scripts/python.exe scripts/eval_admet.py

Prerequisites:
    scripts/dock_library.py       (library_docking_results.json)
    scripts/eval_docking_noise.py (docking_noise_results.json)

Output:
    models/qsar/admet_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.admet.filters import evaluate_admet, summarize_admet
from src.utils.logging import get_logger

logger = get_logger(__name__)

LIBRARY_RESULTS = PROJECT_ROOT / "models" / "qsar" / "library_docking_results.json"
NOISE_RESULTS = PROJECT_ROOT / "models" / "qsar" / "docking_noise_results.json"
OUTPUT_PATH = PROJECT_ROOT / "models" / "qsar" / "admet_results.json"


# ── Loaders ───────────────────────────────────────────────────────────────────


def load_shortlist(noise_path: Path) -> list[dict]:
    """
    Return the 7-compound confidence-filtered shortlist.

    Compounds must have call="L858R_selective" and docking_confidence="standard"
    (covalent compounds get "low_confidence_covalent" and are excluded).
    """
    data = json.loads(noise_path.read_text(encoding="utf-8"))
    return [
        c
        for c in data["compounds"]
        if c["call"] == "L858R_selective"
        and c.get("docking_confidence", "") == "standard"
    ]


def load_top50(library_path: Path) -> list[dict]:
    """Return all compounds from the library docking first pass."""
    data = json.loads(library_path.read_text(encoding="utf-8"))
    return data["compounds"]


# ── Reporting ─────────────────────────────────────────────────────────────────


def _print_shortlist_table(rows: list[dict]) -> None:
    hdr = f"{'CID':<10}  {'MW':>5}  {'LogP':>5}  {'QED':>5}  {'SA':>5}  {'LipoV':>5}  {'PAINS':>5}  {'Brenk':>5}  status"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        res = r["admet"]
        sa = f"{res['sa_score']:.1f}" if res["sa_score"] else "N/A"
        print(
            f"{r['cid']:<10}  {res['mw']:>5.0f}  {res['logp']:>5.1f}  "
            f"{res['qed']:>5.3f}  {sa:>5}  "
            f"{res['lipinski_violations']:>5}  "
            f"{'Y' if res['pains_flag'] else 'N':>5}  "
            f"{'Y' if res['brenk_flag'] else 'N':>5}  "
            f"{res['admet_status']}"
        )
        if res["flag_reasons"]:
            for reason in res["flag_reasons"]:
                print(f"  {'':10}  -> {reason}")


def _print_top50_summary(summary: dict) -> None:
    print(
        f"\n  n_pass      : {summary['n_pass']} / {summary['n_valid']}"
        f"  (pass rate {summary['pass_rate']:.1%})"
    )
    print(f"  n_flag      : {summary['n_flag']}")
    print(f"  median QED  : {summary['median_qed']}")
    if summary["median_sa"] is not None:
        print(f"  median SA   : {summary['median_sa']}")

    if summary["brenk_frequency"]:
        print("\n  Most common Brenk alerts:")
        for alert, count in list(summary["brenk_frequency"].items())[:8]:
            print(f"    {count:3d}x  {alert}")

    if summary["pains_frequency"]:
        print("\n  Most common PAINS alerts:")
        for alert, count in list(summary["pains_frequency"].items())[:5]:
            print(f"    {count:3d}x  {alert}")


def report(shortlist_rows: list[dict], top50_summary: dict) -> None:
    print(
        f"\n=== ADMET evaluation: confidence-filtered shortlist (n={len(shortlist_rows)}) ===\n"
    )
    _print_shortlist_table(shortlist_rows)

    print("\n=== ADMET evaluation: top-50 library candidates ===")
    _print_top50_summary(top50_summary)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    logger.info("=== eval_admet: ADMET filter evaluation ===")

    for path, label in [
        (LIBRARY_RESULTS, "library_docking_results.json"),
        (NOISE_RESULTS, "docking_noise_results.json"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found: {path}\n"
                f"Run the relevant docking scripts first."
            )

    # ── Shortlist ─────────────────────────────────────────────────────────────
    shortlist = load_shortlist(NOISE_RESULTS)
    logger.info(f"Shortlist: {len(shortlist)} compounds")
    shortlist_rows = []
    for cand in shortlist:
        res = evaluate_admet(cand["smiles"])
        shortlist_rows.append({**cand, "admet": res})

    # ── Top-50 ────────────────────────────────────────────────────────────────
    top50 = load_top50(LIBRARY_RESULTS)
    logger.info(f"Top-50 library: {len(top50)} compounds")
    top50_results = []
    for cand in top50:
        res = evaluate_admet(cand["smiles"])
        top50_results.append({**cand, "admet": res})

    # ── Summaries ─────────────────────────────────────────────────────────────
    shortlist_summary = summarize_admet([r["admet"] for r in shortlist_rows])
    top50_summary = summarize_admet([r["admet"] for r in top50_results])

    report(shortlist_rows, top50_summary)

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "shortlist": {
            "n": len(shortlist_rows),
            "summary": shortlist_summary,
            "compounds": [
                {
                    "cid": r["cid"],
                    "initial_delta": r["initial_delta"],
                    "noise_delta": (
                        r.get("noise_stats", {}).get("delta")
                        if r.get("noise_stats")
                        else None
                    ),
                    "docking_confidence": r["docking_confidence"],
                    **r["admet"],
                }
                for r in shortlist_rows
            ],
        },
        "top50": {
            "n": len(top50_results),
            "summary": top50_summary,
            "compounds": [
                {
                    "cid": r["cid"],
                    "selectivity_delta": r["selectivity_delta"],
                    "docking_confidence": r["docking_confidence"],
                    **r["admet"],
                }
                for r in top50_results
            ],
        },
        "note": (
            "APPROXIMATE. Computational drug-likeness and structural liability "
            "estimates only. Not a substitute for experimental ADMET profiling. "
            "admet_status is 'pass' or 'flag' — molecules are never hard-dropped. "
            "Brenk alerts on covalent-warhead compounds (acrylamides, vinyl sulfones) "
            "are expected and consistent with their low_confidence docking tag."
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
