"""Template-based markdown report assembler for the Discovery Copilot."""

from __future__ import annotations

from src.agent.schemas import (
    CandidateReport,
    DockingLookupResult,
    PredictToolResult,
    RankingLookupResult,
)

_LIMITATIONS_BLOCK = """\
## Limitations

- **Activity predictions are EXPLORATORY**: the backbone model is trained on ~1,253 general
  EGFR molecules and only ~22 true L858R records exist. Predictions for known candidates
  are in-sample; for generated candidates they are doubly exploratory.
- **Selectivity proxy is NOT statistically validated**: the ML delta (pic50_mutant - pic50_wt)
  showed Spearman r=0.433, p=0.244 at n=9 pairs, which is below significance.
- **Docking is rigid-receptor only** (AutoDock Vina 1.2.7, 2ITZ/2ITY crystal structures).
  Delta magnitudes underestimate the true affinity difference; direction is more reliable
  than magnitude.
- **ADMET filters are approximate**: Lipinski/Veber/PAINS/Brenk/QED rules are computational
  proxies and not a substitute for experimental profiling.
- **No experimental validation has been performed** on any candidate in this pipeline.
- For a full discussion see `docs/PROJECT_WALKTHROUGH.md` and the Limitations page of
  the Streamlit dashboard.
"""

_EXPLORATORY_BANNER = (
    "> **EXPLORATORY**: All numbers in this report come from precomputed in-silico "
    "models only. No experimental data exists for this candidate unless otherwise noted. "
    "Do not interpret any score as proof of biological activity."
)


def _fmt_float(value: float | None, dp: int = 3) -> str:
    return f"{value:.{dp}f}" if value is not None else "N/A"


def _fmt_admet(predict: PredictToolResult | None) -> str:
    if predict is None or not predict.valid:
        return "_ADMET data unavailable._\n"
    lines = [
        "| Property | Value |",
        "|---|---|",
        f"| Status | {predict.admet_status or 'N/A'} |",
        f"| QED | {_fmt_float(predict.qed, 3)} |",
        f"| Covalent | {'Yes (' + ', '.join(predict.warheads) + ')' if predict.covalent else 'No'} |",
    ]
    if predict.admet_alerts:
        lines.append(f"| Alerts | {', '.join(predict.admet_alerts)} |")
    return "\n".join(lines) + "\n"


def _fmt_docking(docking: DockingLookupResult | None) -> str:
    if docking is None or not docking.found:
        return (
            "_No docking data found for this candidate in the precomputed results. "
            "Run `scripts/dock_library.py` or `scripts/dock_generated_candidates.py` "
            "to generate docking scores._\n"
        )
    lines = [
        "| Metric | Value |",
        "|---|---|",
        f"| L858R Vina score | {_fmt_float(docking.l858r_score)} kcal/mol |",
        f"| WT Vina score | {_fmt_float(docking.wt_score)} kcal/mol |",
        f"| Selectivity delta (structure-based, docking) | {_fmt_float(docking.selectivity_delta)} kcal/mol |",
        f"| Direction | {docking.direction or 'N/A'} |",
        f"| Docking confidence | {docking.docking_confidence or 'N/A'} |",
        f"| Data source | {docking.data_source} |",
    ]
    if docking.mean_delta is not None:
        lines += [
            f"| Mean delta (5-seed noise study) | {_fmt_float(docking.mean_delta)} kcal/mol |",
            f"| Std delta | {_fmt_float(docking.std_delta)} kcal/mol |",
            f"| Noise-study call | {docking.noise_call or 'N/A'} |",
        ]
    return "\n".join(lines) + "\n"


def _fmt_ranking(ranking: RankingLookupResult | None) -> str:
    if ranking is None or not ranking.found:
        return "_This candidate is not in the final ranked shortlist._\n"
    lines = [
        "| Metric | Value |",
        "|---|---|",
        f"| Rank | {ranking.rank} / 68 |",
        f"| Source | {ranking.source} |",
        f"| Final score | {_fmt_float(ranking.final_score)} |",
        f"| Activity (norm) | {_fmt_float(ranking.activity_norm)} |",
        f"| Selectivity (norm) | {_fmt_float(ranking.selectivity_norm)} |",
        f"| Docking affinity (norm) | {_fmt_float(ranking.affinity_norm)} |",
        f"| ADMET (norm) | {_fmt_float(ranking.admet_norm)} |",
        f"| Confidence factor | {_fmt_float(ranking.confidence_factor)} |",
        f"| Covalent | {'Yes' if ranking.is_covalent else 'No'} |",
    ]
    return "\n".join(lines) + "\n"


def generate_report(
    candidate_id: str,
    ranking: RankingLookupResult | None,
    docking: DockingLookupResult | None,
    predict: PredictToolResult | None,
    extra_warnings: list[str] | None = None,
) -> CandidateReport:
    """
    Assemble a template-based markdown report for one candidate.

    All input objects may be None (report notes missing data rather than crashing).
    """
    warnings: list[str] = list(extra_warnings or [])

    # Collect all warnings from inputs
    for src in (predict, docking):
        if src is not None:
            warnings.extend(getattr(src, "warnings", []) or [])

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_warnings: list[str] = []
    for w in warnings:
        if w not in seen:
            seen.add(w)
            unique_warnings.append(w)

    smiles_line = ""
    if predict and predict.canonical_smiles:
        smiles_line = f"\n**SMILES**: `{predict.canonical_smiles}`\n"
    elif ranking and ranking.smiles:
        smiles_line = f"\n**SMILES**: `{ranking.smiles}`\n"

    # Summary table
    pic50 = _fmt_float(predict.pic50_mutant if predict and predict.valid else None)
    sel = _fmt_float(predict.selectivity_proxy if predict and predict.valid else None)
    domain = (predict.domain if predict and predict.valid else None) or "N/A"
    rank_str = (
        str(ranking.rank) + "/68" if (ranking and ranking.found) else "not ranked"
    )
    final = _fmt_float(ranking.final_score if ranking and ranking.found else None)

    summary_table = (
        f"| Key metric | Value |\n"
        f"|---|---|\n"
        f"| Rank (composite) | {rank_str} |\n"
        f"| Final score | {final} |\n"
        f"| Backbone pred pIC50 | {pic50} (EXPLORATORY) |\n"
        f"| Selectivity proxy (ML proxy, exploratory) | {sel} (not validated at n=9) |\n"
        f"| Applicability domain | {domain} |\n"
    )

    # Warnings section
    warnings_md = ""
    if unique_warnings:
        items = "\n".join(f"- {w}" for w in unique_warnings)
        warnings_md = f"## Warnings\n\n{items}\n\n"

    md = f"""\
# Candidate Report: {candidate_id}

{_EXPLORATORY_BANNER}
{smiles_line}
## Summary

{summary_table}

## Composite Ranking

{_fmt_ranking(ranking)}

## Activity Prediction (EXPLORATORY)

Backbone (Model 1, RandomForest) predicts pIC50 on the general EGFR/L858R construct.
WT-proxy (Model 2, XGBoost) predicts pIC50 on wild-type/unspecified EGFR.
The selectivity proxy is their difference; it was NOT statistically significant at n=9.

| Score | Value |
|---|---|
| pred pIC50 (L858R backbone) | {pic50} |
| pred pIC50 (WT-proxy) | {_fmt_float(predict.pic50_wt if predict and predict.valid else None)} |
| Selectivity proxy (ML proxy, exploratory) | {sel} |
| Applicability domain | {domain} |
| Confidence factor | {_fmt_float(predict.confidence_factor if predict and predict.valid else None)} |

## Docking Results (EXPLORATORY)

Scores are from AutoDock Vina 1.2.7 with rigid receptors 2ITZ (L858R) and 2ITY (WT).
Negative delta = L858R-favoured. Delta direction is reliable; magnitude is not.

{_fmt_docking(docking)}

## ADMET Profile (APPROXIMATE)

{_fmt_admet(predict)}

{warnings_md}{_LIMITATIONS_BLOCK}"""

    return CandidateReport(
        candidate_id=candidate_id,
        markdown=md,
        warnings=unique_warnings,
    )
