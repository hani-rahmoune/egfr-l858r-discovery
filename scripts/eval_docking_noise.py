"""
Phase B2 docking noise quantification.

Takes the top N_TOP compounds by initial selectivity delta from the first-pass
library docking (dock_library.py) and re-docks each compound into both pockets
N_SEEDS times with different random seeds.  Propagates seed-to-seed variability
into a delta uncertainty:

    std_delta = sqrt(std_L858R^2 + std_WT^2)

A compound is classified as a confident call only when:

    |delta| > THRESHOLD * std_delta

Covalent compounds (docking_confidence="low_confidence") are excluded from
confident calls regardless of the noise-quantified delta.

All output is EXPLORATORY: rigid receptor, Vina scoring function, n=15.

Run:
    PYTHONPATH=. .venv/Scripts/python.exe scripts/eval_docking_noise.py

Runtime: ~60-90 min (15 compounds x 2 pockets x 5 seeds = 150 Vina runs).
Prerequisites:
    - scripts/dock_library.py must be complete  (library_docking_results.json)
    - Ligand PDBQTs in data/docking/ligands/library/
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.docking.parse_results import best_affinity
from src.docking.vina_runner import run_vina
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

N_TOP = 15
SEEDS = [42, 7, 13, 99, 123]
EXHAUSTIVENESS = 8
N_POSES = 9
THRESHOLD = 1.5  # |delta| must exceed THRESHOLD * std_delta for a confident call

RECEPTORS = {
    "L858R": PROJECT_ROOT / "data" / "docking" / "protein" / "2ITZ_receptor.pdbqt",
    "WT": PROJECT_ROOT / "data" / "docking" / "protein" / "2ITY_aligned_receptor.pdbqt",
}

LIBRARY_RESULTS_PATH = PROJECT_ROOT / "models" / "qsar" / "library_docking_results.json"
LIGAND_DIR = PROJECT_ROOT / "data" / "docking" / "ligands" / "library"
NOISE_DIR = PROJECT_ROOT / "data" / "docking" / "results" / "noise_eval"
OUTPUT_PATH = PROJECT_ROOT / "models" / "qsar" / "docking_noise_results.json"

# ── Core functions (importable for tests) ─────────────────────────────────────


def load_top_compounds(results_path: Path, n: int = N_TOP) -> list[dict]:
    """
    Load top-n compounds by selectivity_delta from library_docking_results.json.

    Only includes compounds with docking_status="ok" and a valid (non-None)
    selectivity_delta.  Returns rows sorted ascending by delta (most L858R-
    selective first).
    """
    data = json.loads(Path(results_path).read_text(encoding="utf-8"))
    ok = [
        row
        for row in data["compounds"]
        if row["docking_status"] == "ok" and row["selectivity_delta"] is not None
    ]
    ok.sort(key=lambda r: r["selectivity_delta"])
    return ok[:n]


def compute_noise_stats(
    affinities_l858r: list[float | None],
    affinities_wt: list[float | None],
) -> dict | None:
    """
    Compute noise statistics from multi-seed Vina runs.

    Parameters
    ----------
    affinities_l858r : per-seed best affinity (kcal/mol) for the L858R pocket.
                       None entries (failed runs) are excluded from statistics.
    affinities_wt    : same for the WT pocket.

    Returns
    -------
    dict with keys:
        mean_l858r, std_l858r, n_l858r
        mean_wt, std_wt, n_wt
        delta        = mean_l858r - mean_wt
        std_delta    = sqrt(std_l858r^2 + std_wt^2)   (error propagation)
    Returns None when either pocket has zero successful runs.
    """
    l_vals = [a for a in affinities_l858r if a is not None]
    w_vals = [a for a in affinities_wt if a is not None]

    if not l_vals or not w_vals:
        return None

    n_l = len(l_vals)
    n_w = len(w_vals)
    mean_l = sum(l_vals) / n_l
    mean_w = sum(w_vals) / n_w

    # Population std when n=1 (not undefined); ddof=1 otherwise
    if n_l > 1:
        std_l = math.sqrt(sum((x - mean_l) ** 2 for x in l_vals) / (n_l - 1))
    else:
        std_l = 0.0

    if n_w > 1:
        std_w = math.sqrt(sum((x - mean_w) ** 2 for x in w_vals) / (n_w - 1))
    else:
        std_w = 0.0

    delta = mean_l - mean_w
    std_delta = math.sqrt(std_l**2 + std_w**2)

    return {
        "mean_l858r": round(mean_l, 3),
        "std_l858r": round(std_l, 3),
        "n_l858r": n_l,
        "mean_wt": round(mean_w, 3),
        "std_wt": round(std_w, 3),
        "n_wt": n_w,
        "delta": round(delta, 3),
        "std_delta": round(std_delta, 3),
    }


def classify_call(
    delta: float,
    std_delta: float,
    docking_confidence: str,
    threshold: float = THRESHOLD,
) -> str:
    """
    Assign a selectivity call from noise-quantified delta.

    Returns one of:
        "L858R_selective"       — delta < 0 and |delta| > threshold * std_delta
        "WT_selective"          — delta > 0 and |delta| > threshold * std_delta
        "ambiguous"             — |delta| <= threshold * std_delta  (within noise)
        "low_confidence_covalent" — compound is covalent regardless of delta
    """
    if docking_confidence == "low_confidence":
        return "low_confidence_covalent"
    if std_delta == 0.0:
        if delta < 0:
            return "L858R_selective"
        elif delta > 0:
            return "WT_selective"
        return "ambiguous"
    if abs(delta) > threshold * std_delta:
        return "L858R_selective" if delta < 0 else "WT_selective"
    return "ambiguous"


# ── Docking helpers ───────────────────────────────────────────────────────────


def _load_box() -> dict[str, float]:
    """Read docking box from docking_config.yaml."""
    import yaml

    cfg_path = PROJECT_ROOT / "config" / "docking_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    box_cfg = cfg["box"]
    return {
        "center_x": float(box_cfg["center_x"]),
        "center_y": float(box_cfg["center_y"]),
        "center_z": float(box_cfg["center_z"]),
        "size_x": float(box_cfg["size_x"]),
        "size_y": float(box_cfg["size_y"]),
        "size_z": float(box_cfg["size_z"]),
    }


def _run_pocket_multi_seed(
    cid: str,
    lig_path: Path,
    rec_path: Path,
    pocket_label: str,
    box: dict[str, float],
    noise_dir: Path,
    seeds: list[int],
    exhaustiveness: int,
) -> list[float | None]:
    """
    Run Vina for one (ligand, receptor) pair across all seeds.

    Output files are cached per seed directory so re-runs skip completed jobs.
    Returns a list of best affinities (kcal/mol), None for any failed seed.
    """
    affinities: list[float | None] = []
    rec_stem = rec_path.stem

    for seed in seeds:
        seed_dir = noise_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        expected_out = seed_dir / f"{cid}__{rec_stem}_out.pdbqt"

        if expected_out.exists():
            aff = best_affinity(expected_out)
            logger.info(f"  [{pocket_label}/seed={seed}] cached -> {aff}")
        else:
            try:
                out_pdbqt, _ = run_vina(
                    receptor=rec_path,
                    ligand=lig_path,
                    out_dir=seed_dir,
                    box=box,
                    n_poses=N_POSES,
                    exhaustiveness=exhaustiveness,
                    seed=seed,
                )
                aff = best_affinity(out_pdbqt)
                logger.info(f"  [{pocket_label}/seed={seed}] -> {aff}")
            except Exception as exc:
                logger.warning(f"  [{pocket_label}/seed={seed}] FAILED: {exc}")
                aff = None

        affinities.append(aff)

    return affinities


# ── Report ────────────────────────────────────────────────────────────────────


def _print_table(rows: list[dict]) -> None:
    header = (
        f"{'Rank':>4}  {'CID':<10}  {'delta±std':>12}  "
        f"{'L858R(mean±std)':>16}  {'WT(mean±std)':>13}  "
        f"{'conf':>8}  call"
    )
    print(header)
    print("-" * len(header))
    for i, r in enumerate(rows, 1):
        stats = r.get("noise_stats")
        if stats:
            delta_str = f"{stats['delta']:+.3f}±{stats['std_delta']:.3f}"
            l858r_str = f"{stats['mean_l858r']:.3f}±{stats['std_l858r']:.3f}"
            wt_str = f"{stats['mean_wt']:.3f}±{stats['std_wt']:.3f}"
        else:
            delta_str = l858r_str = wt_str = "N/A"
        print(
            f"{i:>4}  {r['cid']:<10}  {delta_str:>12}  "
            f"{l858r_str:>16}  {wt_str:>13}  "
            f"{r['docking_confidence']:>8}  {r['call']}"
        )


def report(rows: list[dict], threshold: float) -> dict:
    """Print ranked table and return summary dict."""
    rows_sorted = sorted(
        rows,
        key=lambda r: (r["noise_stats"]["delta"] if r["noise_stats"] else float("inf")),
    )

    print("\n=== Docking noise quantification — re-ranked table ===\n")
    _print_table(rows_sorted)

    calls = [r["call"] for r in rows_sorted]
    n_l858r = calls.count("L858R_selective")
    n_wt = calls.count("WT_selective")
    n_ambig = calls.count("ambiguous")
    n_covalent = calls.count("low_confidence_covalent")

    print(f"\nThreshold: |delta| > {threshold} × std_delta")
    print(f"  Confident L858R-selective : {n_l858r}")
    print(f"  Confident WT-selective    : {n_wt}")
    print(f"  Ambiguous / within noise  : {n_ambig}")
    print(f"  Low-confidence (covalent) : {n_covalent}")
    print(f"  Total                     : {len(rows_sorted)}\n")

    return {
        "n_confident_l858r": n_l858r,
        "n_confident_wt": n_wt,
        "n_ambiguous": n_ambig,
        "n_low_conf_covalent": n_covalent,
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    logger.info("=== eval_docking_noise: Phase B2 noise quantification ===")

    # Load box from config
    box = _load_box()
    logger.info(
        f"Box: center ({box['center_x']}, {box['center_y']}, {box['center_z']}) "
        f"size {box['size_x']}x{box['size_y']}x{box['size_z']} A"
    )

    # Validate receptors
    for label, rec_path in RECEPTORS.items():
        if not rec_path.exists():
            raise FileNotFoundError(
                f"Receptor not found: {rec_path}\n"
                f"Run scripts/prepare_docking.py and scripts/sanity_check_docking.py first."
            )

    # Load top-N candidates from library docking first pass
    if not LIBRARY_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Library docking results not found: {LIBRARY_RESULTS_PATH}\n"
            f"Run scripts/dock_library.py first."
        )
    candidates = load_top_compounds(LIBRARY_RESULTS_PATH, N_TOP)
    logger.info(
        f"Top {len(candidates)} compounds by selectivity delta "
        f"(range: {candidates[0]['selectivity_delta']:.3f} to "
        f"{candidates[-1]['selectivity_delta']:.3f} kcal/mol)"
    )

    # Run multi-seed docking for each compound
    NOISE_DIR.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict] = []

    for rank, cand in enumerate(candidates, 1):
        cid = cand["cid"]
        smiles = cand["smiles"]
        lig_path = LIGAND_DIR / f"{cid}.pdbqt"

        logger.info(
            f"\n[{rank}/{len(candidates)}] {cid} "
            f"(initial delta={cand['selectivity_delta']:.3f}, "
            f"conf={cand['docking_confidence']})"
        )

        if not lig_path.exists():
            logger.warning(f"  Ligand PDBQT not found, skipping: {lig_path}")
            result_rows.append(
                {
                    "cid": cid,
                    "smiles": smiles,
                    "initial_delta": cand["selectivity_delta"],
                    "docking_confidence": cand["docking_confidence"],
                    "warheads": cand["warheads"],
                    "affinities_l858r": [],
                    "affinities_wt": [],
                    "noise_stats": None,
                    "call": "missing_ligand",
                }
            )
            continue

        aff_l = _run_pocket_multi_seed(
            cid,
            lig_path,
            RECEPTORS["L858R"],
            "L858R",
            box,
            NOISE_DIR,
            SEEDS,
            EXHAUSTIVENESS,
        )
        aff_w = _run_pocket_multi_seed(
            cid,
            lig_path,
            RECEPTORS["WT"],
            "WT",
            box,
            NOISE_DIR,
            SEEDS,
            EXHAUSTIVENESS,
        )

        stats = compute_noise_stats(aff_l, aff_w)
        call = (
            classify_call(
                delta=stats["delta"] if stats else 0.0,
                std_delta=stats["std_delta"] if stats else 0.0,
                docking_confidence=cand["docking_confidence"],
            )
            if stats
            else "insufficient_data"
        )

        result_rows.append(
            {
                "cid": cid,
                "smiles": smiles,
                "initial_delta": cand["selectivity_delta"],
                "docking_confidence": cand["docking_confidence"],
                "warheads": cand["warheads"],
                "affinities_l858r": aff_l,
                "affinities_wt": aff_w,
                "noise_stats": stats,
                "call": call,
            }
        )

        if stats:
            logger.info(
                f"  delta={stats['delta']:+.3f} ± {stats['std_delta']:.3f}  "
                f"(L858R: {stats['mean_l858r']:.3f}±{stats['std_l858r']:.3f}  "
                f"WT: {stats['mean_wt']:.3f}±{stats['std_wt']:.3f})  "
                f"call={call}"
            )

    # Report
    summary = report(result_rows, THRESHOLD)

    # Save results
    output = {
        "n_compounds": len(candidates),
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "exhaustiveness": EXHAUSTIVENESS,
        "threshold": THRESHOLD,
        **summary,
        "compounds": result_rows,
        "note": (
            "EXPLORATORY. Rigid receptor (2ITZ L858R / 2ITY_aligned WT). "
            "Vina 1.2.7 scoring function. "
            f"Noise quantified over {len(SEEDS)} seeds; "
            f"std_delta = sqrt(std_L858R^2 + std_WT^2). "
            f"Confident call criterion: |delta| > {THRESHOLD} * std_delta. "
            "Covalent compounds (docking_confidence=low_confidence) are "
            "excluded from confident calls regardless of noise-quantified delta."
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
