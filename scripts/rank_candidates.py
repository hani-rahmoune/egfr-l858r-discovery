"""
Phase 23 — Final integrated candidate ranking (capstone).

Fuses the four orthogonal evidence streams via the v2 composite score
(src/scoring/ranking.py):

  bioactivity = 0.30*activity + 0.30*docking_selectivity
              + 0.20*docking_affinity + 0.20*ADMET            (each min-max normalised)
  final_score = bioactivity * AD_confidence_factor

Covalent warhead and within-noise selectivity become WARNINGS, not score penalties.

Sources combined:
  * Known library candidates  — models/qsar/library_docking_results.json
      (+ ADMET QED from admet_results.json, + within-noise call from
       docking_noise_results.json, + AD computed here)
  * Generated candidates      — models/generator/generated_docking_results.json
      (already carry ADMET QED + AD from the screen; produced by
       scripts/dock_generated_candidates.py)

If the generated docking file is absent, the script ranks the known library
alone (Phase 23 part 2, the immediate no-new-compute deliverable). When the
generated file is present it ranks both together (parts 3-4).

All output is EXPLORATORY.

Run:
  PYTHONPATH=. .venv/Scripts/python.exe scripts/rank_candidates.py
  PYTHONPATH=. .venv/Scripts/python.exe scripts/rank_candidates.py --library-only

Output: data/generated/final_ranked_candidates.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from rdkit import Chem

from src.scoring.applicability_domain import ApplicabilityDomain
from src.scoring.ranking import RankingWeights, rank_candidates
from src.utils.logging import get_logger

logger = get_logger(__name__)

_LIBRARY_JSON = PROJECT_ROOT / "models" / "qsar" / "library_docking_results.json"
_ADMET_JSON = PROJECT_ROOT / "models" / "qsar" / "admet_results.json"
_NOISE_JSON = PROJECT_ROOT / "models" / "qsar" / "docking_noise_results.json"
_GENERATED_JSON = (
    PROJECT_ROOT / "models" / "generator" / "generated_docking_results.json"
)
_EGFR_CSV = PROJECT_ROOT / "data" / "interim" / "egfr_cleaned.csv"
_ERBB2_CSV = PROJECT_ROOT / "data" / "interim" / "erbb2_cleaned.csv"
_OUT_CSV = PROJECT_ROOT / "data" / "generated" / "final_ranked_candidates.csv"


# ── Loaders ────────────────────────────────────────────────────────────────────


def _load_train_smiles() -> set[str]:
    frames = []
    for csv in (_EGFR_CSV, _ERBB2_CSV):
        if csv.exists():
            frames.append(pd.read_csv(csv, usecols=["canonical_smiles"]))
    if not frames:
        return set()
    raw = pd.concat(frames)["canonical_smiles"].dropna().tolist()
    out: set[str] = set()
    for smi in raw:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            out.add(Chem.MolToSmiles(mol))
    return out


def _qed_by_cid() -> dict[str, float | None]:
    """Map library cid -> QED from admet_results.json (top50)."""
    if not _ADMET_JSON.exists():
        return {}
    d = json.loads(_ADMET_JSON.read_text())
    out: dict[str, float | None] = {}
    for c in d.get("top50", {}).get("compounds", []):
        out[c["cid"]] = c.get("qed")
    return out


def _within_noise_by_cid() -> dict[str, bool | None]:
    """
    Map library cid -> selectivity-within-noise flag from the seed-noise study.
      call == 'ambiguous'                  -> True  (within noise)
      call in {'L858R_selective','WT_selective'} -> False (distinguishable)
      call == 'low_confidence_covalent'    -> None  (covalent warning dominates)
    Compounds not in the noise study (only the top-15 were studied) -> absent (None).
    """
    if not _NOISE_JSON.exists():
        return {}
    d = json.loads(_NOISE_JSON.read_text())
    out: dict[str, bool | None] = {}
    for c in d.get("compounds", []):
        call = c.get("call")
        if call == "ambiguous":
            out[c["cid"]] = True
        elif call in ("L858R_selective", "WT_selective"):
            out[c["cid"]] = False
        else:  # low_confidence_covalent or unknown
            out[c["cid"]] = None
    return out


# ── Record builders ────────────────────────────────────────────────────────────


def build_library_records(ad: ApplicabilityDomain) -> list[dict]:
    """Build ranking records for the docked known-library compounds (status ok)."""
    d = json.loads(_LIBRARY_JSON.read_text())
    qed_map = _qed_by_cid()
    noise_map = _within_noise_by_cid()

    records: list[dict] = []
    for c in d["compounds"]:
        if c.get("docking_status") != "ok":
            continue
        smi = c["smiles"]
        ad_r = ad.predict(smi)
        warheads = list(c.get("warheads") or [])
        records.append(
            {
                "cid": c["cid"],
                "source": "known",
                "smiles": smi,
                "activity": c.get("pred_pic50"),
                "l858r_score": c.get("l858r_score"),
                "wt_score": c.get("wt_score"),
                "selectivity_delta": c.get("selectivity_delta"),
                "admet_qed": qed_map.get(c["cid"]),
                "domain": ad_r["domain"],
                "max_tanimoto": ad_r["max_tanimoto"],
                "confidence_factor": ad_r["confidence_factor"],
                "is_covalent": bool(warheads),
                "warheads": warheads,
                "selectivity_within_noise": noise_map.get(c["cid"]),
            }
        )
    return records


def build_generated_records() -> list[dict]:
    """Build ranking records for the docked generated compounds (status ok)."""
    if not _GENERATED_JSON.exists():
        return []
    d = json.loads(_GENERATED_JSON.read_text())
    records: list[dict] = []
    for c in d["compounds"]:
        if c.get("docking_status") != "ok":
            continue
        warheads = list(c.get("warheads") or [])
        records.append(
            {
                "cid": c["cid"],
                "source": "generated",
                "smiles": c["smiles"],
                "activity": c.get("pred_pic50"),
                "l858r_score": c.get("l858r_score"),
                "wt_score": c.get("wt_score"),
                "selectivity_delta": c.get("selectivity_delta"),
                "admet_qed": c.get("admet_qed"),
                "domain": c.get("domain"),
                "max_tanimoto": c.get("max_tanimoto"),
                "confidence_factor": c.get("confidence_factor", 1.0),
                "is_covalent": bool(warheads),
                "warheads": warheads,
                "selectivity_within_noise": None,  # generated set was not noise-studied
            }
        )
    return records


# ── Reporting ──────────────────────────────────────────────────────────────────


def print_table(df: pd.DataFrame, n: int = 20) -> None:
    logger.info("=" * 118)
    logger.info("FINAL INTEGRATED RANKING (v2 composite, EXPLORATORY)")
    logger.info("-" * 118)
    logger.info(
        f"{'#':<4}{'cid':<10}{'src':<10}{'final':>7} {'bioact':>7} "
        f"{'act':>6}{'sel':>6}{'aff':>6}{'qed':>6} {'cf':>5} {'dom':<13}{'warnings'}"
    )
    logger.info("-" * 118)
    for _, r in df.head(n).iterrows():
        warn = r["warnings"]
        warn_short = (warn[:48] + "...") if len(warn) > 51 else warn
        logger.info(
            f"{int(r['rank']):<4}{r['cid']:<10}{r['source']:<10}"
            f"{r['final_score']:>7.4f} {r['bioactivity_score']:>7.4f} "
            f"{r['activity_norm']:>6.2f}{r['selectivity_norm']:>6.2f}"
            f"{r['affinity_norm']:>6.2f}{r['admet_norm']:>6.2f} "
            f"{r['confidence_factor']:>5.2f} {str(r['domain']):<13}{warn_short}"
        )
    logger.info("=" * 118)


def report_generated_placement(df: pd.DataFrame) -> None:
    if "generated" not in set(df["source"]):
        return
    n = len(df)
    gen = df[df["source"] == "generated"]
    known = df[df["source"] == "known"]
    logger.info("")
    logger.info("GENERATED-vs-KNOWN placement:")
    logger.info(f"  {len(gen)} generated, {len(known)} known, {n} total")
    logger.info(
        f"  best generated rank: #{int(gen['rank'].min())} "
        f"(final={gen.iloc[0]['final_score']:.4f}, cid={gen.iloc[0]['cid']})"
    )
    logger.info(f"  median generated rank: {int(gen['rank'].median())} / {n}")

    def in_top(k):
        return int((gen["rank"] <= k).sum())

    logger.info(
        f"  generated in top-10: {in_top(10)}   top-20: {in_top(20)}   "
        f"top-{n//2}: {in_top(n//2)}"
    )
    best_known = known.iloc[0]
    logger.info(
        f"  best known: #{int(best_known['rank'])} "
        f"{best_known['cid']} (final={best_known['final_score']:.4f})"
    )


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Final integrated ranking")
    parser.add_argument(
        "--library-only",
        action="store_true",
        help="Rank only the known library (skip generated even if present)",
    )
    parser.add_argument("--top", type=int, default=20, help="Rows to print")
    args = parser.parse_args()

    logger.info("Fitting applicability domain on EGFR/ErbB2 training set ...")
    ad = ApplicabilityDomain.from_config()
    ad.fit(list(_load_train_smiles()))

    records = build_library_records(ad)
    logger.info(f"Known library: {len(records)} docked candidates")

    if not args.library_only:
        gen = build_generated_records()
        if gen:
            logger.info(f"Generated: {len(gen)} docked candidates")
            records += gen
        else:
            logger.info(
                "No generated docking results found "
                f"({_GENERATED_JSON.name}); ranking library only. "
                "Run scripts/dock_generated_candidates.py to include them."
            )

    weights = RankingWeights.from_config()
    logger.info(f"v2 weights (normalised): {weights.normalized()}")

    df = rank_candidates(records, weights)

    print_table(df, n=args.top)
    report_generated_placement(df)

    _OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_OUT_CSV, index=False)
    logger.info(f"\nExported {len(df)} ranked candidates -> {_OUT_CSV}")


if __name__ == "__main__":
    main()
