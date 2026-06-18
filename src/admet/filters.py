"""
ADMET drug-likeness and structural liability filters.

All results are APPROXIMATE — computational estimates only, not a substitute
for experimental safety and pharmacokinetic profiling.

admet_status is "pass" or "flag"; molecules are never hard-dropped.
Use admet_status as a soft ranking signal (weight 0.20 in the composite score)
not as an exclusion criterion.

Filters:
    Lipinski Ro5  MW <= 500, LogP <= 5, HBD <= 5, HBA <= 10
                  (pass = <=1 violation; exact MW via Descriptors.ExactMolWt)
    Veber         RotBonds <= 10, TPSA <= 140
    PAINS         pan-assay interference (RDKit FilterCatalog, all subsets)
    Brenk         metabolic/reactive structural alerts (RDKit FilterCatalog)
    QED           quantitative drug-likeness [0, 1]  (flag < 0.25)
    SA score      synthetic accessibility [1, 10]     (flag > 6.0 if available)
    Range checks  MW 100-600, LogP -2 to 5.5, TPSA 20-140  (informational only,
                  do NOT contribute to admet_status)

Dependencies:
    rdkit (standard)
    rdkit.Contrib.SA_Score (optional; sa_score=None if absent)

Reuses check_lipinski and check_veber from src.features.descriptors.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from rdkit import Chem
from rdkit.Chem import QED as _QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from src.features.descriptors import check_lipinski, check_veber
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ── SA score (optional) ───────────────────────────────────────────────────────
try:
    from rdkit.Contrib.SA_Score import sascorer as _sa

    _SA_AVAILABLE: bool = True
    logger.info("SA score available (rdkit.Contrib.SA_Score)")
except Exception:
    _SA_AVAILABLE = False
    _sa = None  # type: ignore[assignment]

# ── Filter catalogs (built once at import) ────────────────────────────────────


def _make_pains_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    # PAINS is the combined A+B+C catalog; fall back to individual subsets if
    # the combined entry is absent in the installed RDKit version.
    try:
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    except AttributeError:
        for sub in ("PAINS_A", "PAINS_B", "PAINS_C"):
            try:
                params.AddCatalog(getattr(FilterCatalogParams.FilterCatalogs, sub))
            except AttributeError:
                pass
    return FilterCatalog(params)


def _make_brenk_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog(params)


_PAINS_CATALOG: FilterCatalog = _make_pains_catalog()
_BRENK_CATALOG: FilterCatalog = _make_brenk_catalog()

# ── Thresholds ────────────────────────────────────────────────────────────────

# Lipinski and Veber thresholds live in check_lipinski / check_veber
QED_FLAG_BELOW = 0.25  # QED below this is flagged
SA_FLAG_ABOVE = 6.0  # SA score above this = hard to synthesize; flag
# (10 = most complex; only evaluated if SA available)

# Range checks (informational — do NOT affect admet_status)
RANGE_MW_MIN = 100.0
RANGE_MW_MAX = 600.0
RANGE_LOGP_MIN = -2.0
RANGE_LOGP_MAX = 5.5
RANGE_TPSA_MIN = 20.0
RANGE_TPSA_MAX = 140.0


# ── Public API ────────────────────────────────────────────────────────────────


def evaluate_admet(smiles: str) -> dict[str, Any]:
    """
    Evaluate ADMET filters for one SMILES string.

    Returns a flat dict with per-filter results, violation counts, boolean alert
    flags, QED, SA score, and admet_status ("pass" or "flag").

    Parameters
    ----------
    smiles : SMILES string (canonical or raw)

    Returns
    -------
    dict — see module docstring for all keys.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _invalid_result(smiles)

    # ── Lipinski (reuse existing function) ────────────────────────────────────
    lip = check_lipinski(smiles)
    lipo_viols = lip["violations"]
    lipinski_pass = lip["lipinski_pass"]
    mw = lip["mol_weight"]
    logp = lip["logp"]
    hbd = lip["hbd"]
    hba = lip["hba"]

    # ── Veber (reuse existing function) ───────────────────────────────────────
    veb = check_veber(smiles)
    veber_pass = veb["veber_pass"]
    tpsa = veb["tpsa"]
    rot_bonds = veb["rotatable_bonds"]

    # ── PAINS ─────────────────────────────────────────────────────────────────
    pains_alerts = _get_alerts(_PAINS_CATALOG, mol, "PAINS")
    pains_flag = len(pains_alerts) > 0

    # ── Brenk ─────────────────────────────────────────────────────────────────
    brenk_alerts = _get_alerts(_BRENK_CATALOG, mol, "Brenk")
    brenk_flag = len(brenk_alerts) > 0

    # ── QED ───────────────────────────────────────────────────────────────────
    qed = round(_QED.qed(mol), 3)

    # ── SA score ──────────────────────────────────────────────────────────────
    sa_score: float | None = None
    if _SA_AVAILABLE and _sa is not None:
        try:
            sa_score = round(float(_sa.calculateScore(mol)), 2)
        except Exception as exc:
            logger.debug(f"SA score failed for {smiles[:40]}: {exc}")

    # ── Range checks (informational only) ────────────────────────────────────
    range_mw_ok = RANGE_MW_MIN <= mw <= RANGE_MW_MAX
    range_logp_ok = RANGE_LOGP_MIN <= logp <= RANGE_LOGP_MAX
    range_tpsa_ok = RANGE_TPSA_MIN <= tpsa <= RANGE_TPSA_MAX
    range_violations = (
        int(not range_mw_ok) + int(not range_logp_ok) + int(not range_tpsa_ok)
    )

    # ── Flag reasons → admet_status ───────────────────────────────────────────
    flag_reasons: list[str] = []
    if not lipinski_pass:
        flag_reasons.append(
            f"Lipinski {lipo_viols} violation(s)"
            f" (MW={mw:.0f}, LogP={logp:.1f}, HBD={hbd}, HBA={hba})"
        )
    if not veber_pass:
        flag_reasons.append(f"Veber (RotBonds={rot_bonds}, TPSA={tpsa:.0f})")
    if pains_flag:
        flag_reasons.append(f"PAINS: {'; '.join(pains_alerts)}")
    if brenk_flag:
        flag_reasons.append(f"Brenk: {'; '.join(brenk_alerts)}")
    if qed < QED_FLAG_BELOW:
        flag_reasons.append(f"QED={qed:.3f} (below {QED_FLAG_BELOW})")
    if sa_score is not None and sa_score > SA_FLAG_ABOVE:
        flag_reasons.append(f"SA={sa_score:.1f} (hard to synthesize)")

    return {
        "smiles": smiles,
        "valid": True,
        # Physicochemical
        "mw": round(float(mw), 2),
        "logp": round(float(logp), 2),
        "hbd": int(hbd),
        "hba": int(hba),
        "tpsa": round(float(tpsa), 1),
        "rotatable_bonds": int(rot_bonds),
        # Lipinski
        "lipinski_violations": int(lipo_viols),
        "lipinski_pass": bool(lipinski_pass),
        # Veber
        "veber_pass": bool(veber_pass),
        # PAINS
        "pains_alerts": pains_alerts,
        "pains_flag": pains_flag,
        # Brenk
        "brenk_alerts": brenk_alerts,
        "brenk_flag": brenk_flag,
        # QED
        "qed": qed,
        # SA score
        "sa_score": sa_score,
        # Range (informational)
        "range_mw_ok": bool(range_mw_ok),
        "range_logp_ok": bool(range_logp_ok),
        "range_tpsa_ok": bool(range_tpsa_ok),
        "range_violations": int(range_violations),
        # Summary
        "flag_reasons": flag_reasons,
        "total_flags": len(flag_reasons),
        "admet_status": "flag" if flag_reasons else "pass",
    }


def evaluate_admet_batch(smiles_list: list[str]) -> list[dict[str, Any]]:
    """Run evaluate_admet for each SMILES and return results in order."""
    return [evaluate_admet(s) for s in smiles_list]


def summarize_admet(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate statistics from a list of evaluate_admet dicts.

    Returns pass/flag counts, most common alerts, and median QED/SA.
    """
    valid = [r for r in results if r["valid"]]
    n_pass = sum(r["admet_status"] == "pass" for r in valid)

    pains_ctr: Counter = Counter()
    brenk_ctr: Counter = Counter()
    for r in valid:
        for a in r["pains_alerts"]:
            pains_ctr[a] += 1
        for a in r["brenk_alerts"]:
            brenk_ctr[a] += 1

    qed_vals = sorted(r["qed"] for r in valid if r["qed"] is not None)
    sa_vals = sorted(r["sa_score"] for r in valid if r["sa_score"] is not None)

    return {
        "n_total": len(results),
        "n_valid": len(valid),
        "n_pass": n_pass,
        "n_flag": len(valid) - n_pass,
        "pass_rate": round(n_pass / len(valid), 3) if valid else None,
        "pains_frequency": dict(pains_ctr.most_common()),
        "brenk_frequency": dict(brenk_ctr.most_common()),
        "median_qed": round(qed_vals[len(qed_vals) // 2], 3) if qed_vals else None,
        "median_sa": round(sa_vals[len(sa_vals) // 2], 2) if sa_vals else None,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────


def _get_alerts(catalog: FilterCatalog, mol: Chem.Mol, label: str) -> list[str]:
    """Return sorted list of unique alert descriptions from a FilterCatalog."""
    try:
        matches = catalog.GetMatches(mol)
        return sorted({m.GetDescription() for m in matches})
    except Exception as exc:
        logger.warning(f"{label} catalog query failed: {exc}")
        return []


def _invalid_result(smiles: str) -> dict[str, Any]:
    return {
        "smiles": smiles,
        "valid": False,
        "mw": None,
        "logp": None,
        "hbd": None,
        "hba": None,
        "tpsa": None,
        "rotatable_bonds": None,
        "lipinski_violations": None,
        "lipinski_pass": False,
        "veber_pass": False,
        "pains_alerts": [],
        "pains_flag": False,
        "brenk_alerts": [],
        "brenk_flag": False,
        "qed": None,
        "sa_score": None,
        "range_mw_ok": False,
        "range_logp_ok": False,
        "range_tpsa_ok": False,
        "range_violations": 3,
        "flag_reasons": ["Invalid SMILES"],
        "total_flags": 1,
        "admet_status": "flag",
    }
