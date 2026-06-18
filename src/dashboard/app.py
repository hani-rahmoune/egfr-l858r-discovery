"""
Streamlit dashboard for the EGFR L858R drug-discovery platform (Phase 25).

Six pages:
  1. Single molecule   — /predict (API) or local registry fallback
  2. Batch screening    — /batch_predict over uploaded/pasted SMILES
  3. Final ranking      — Phase 23 fused ranking; generated vs known
  4. Model performance  — QSAR scaffold-split metrics (seed std), FP ablation, Model 3/4 verdicts
  5. Docking results    — error-bar'd selectivity shortlist + sanity check + generated docking
  6. Limitations        — the honest findings, stated plainly

Scoring prefers the FastAPI service if it is up (GET /health), else falls back to
scoring locally through the same ModelRegistry. Everything is EXPLORATORY.

Run:
  PYTHONPATH=. .venv/Scripts/python.exe -m streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.dashboard import api_client as api
from src.dashboard import data_loaders as dl

st.set_page_config(page_title="EGFR L858R Discovery", page_icon="🧬", layout="wide")

EXPLORATORY_BANNER = (
    "⚠️ **All scores are EXPLORATORY.** Backbone activity is in-sample for known "
    "actives (~22 true L858R records), selectivity is not validated, docking is "
    "rigid-receptor, and nothing here is experimentally confirmed. See **Limitations**."
)


# ── Scorer resolution (API preferred, local fallback) ──────────────────────────


@st.cache_resource(show_spinner="Loading local models (API not reachable)…")
def _local_registry():
    from src.api.services import ModelRegistry

    return ModelRegistry.load()


def resolve_scorer(base_url: str):
    """Return (score_fn, batch_fn, mode_label)."""
    if api.api_available(base_url):
        return (
            lambda s: api.predict_via_api(base_url, s),
            lambda lst: api.batch_predict_via_api(base_url, lst)["results"],
            f"🟢 API · {base_url}",
        )
    reg = _local_registry()
    return (
        lambda s: api.predict_via_registry(reg, s),
        lambda lst: [reg.score(s) for s in lst],
        "🟡 Local registry (API down)",
    )


# ── Rendering helpers ──────────────────────────────────────────────────────────


def _render_prediction(res: dict) -> None:
    if not res.get("valid", False):
        st.error("Invalid SMILES — RDKit could not parse the input.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("pIC50 mutant (backbone)", res["pic50_mutant"])
    c2.metric("pIC50 WT (WT-proxy)", res["pic50_wt"])
    sel = res["selectivity_proxy"]
    c3.metric(
        "Selectivity proxy (mut − WT)",
        sel,
        help="EXPLORATORY ML proxy. Positive = mutant-selective. NOT docking selectivity.",
    )
    st.caption(
        "Selectivity proxy is an exploratory ML difference, not the structure-based "
        "docking selectivity (which is not computed at request time)."
    )

    c4, c5, c6 = st.columns(3)
    admet = res.get("admet") or {}
    c4.metric("ADMET", (admet.get("status") or "—").upper())
    c5.metric("QED", admet.get("qed"))
    ad = res.get("applicability_domain") or {}
    c6.metric(
        "Applicability domain",
        ad.get("domain", "—"),
        help=f"max Tanimoto {ad.get('max_tanimoto')} · confidence_factor {ad.get('confidence_factor')}",
    )

    cov = res.get("covalent", False)
    st.write(
        f"**Covalent warhead:** {'🔴 yes — ' + ', '.join(res.get('warheads', [])) if cov else '⚪ none detected'}"
    )
    if admet.get("flag_reasons"):
        st.write("**ADMET flags:** " + "; ".join(admet["flag_reasons"]))

    for w in res.get("warnings", []):
        st.warning(w)
    st.info(
        f"`docking_selectivity_available = {res.get('docking_selectivity_available', False)}` "
        "— structure-based selectivity needs the offline Vina pipeline."
    )


def _parse_smiles_input(text: str, uploaded) -> list[str]:
    smiles: list[str] = []
    if uploaded is not None:
        raw = uploaded.getvalue().decode("utf-8", errors="ignore")
        if uploaded.name.lower().endswith(".csv"):
            from io import StringIO

            df = pd.read_csv(StringIO(raw))
            col = next(
                (c for c in df.columns if c.lower() in ("smiles", "canonical_smiles")),
                df.columns[0],
            )
            smiles += [str(x) for x in df[col].dropna().tolist()]
        else:
            smiles += [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if text.strip():
        smiles += [ln.strip() for ln in text.splitlines() if ln.strip()]
    return smiles


# ── Pages ──────────────────────────────────────────────────────────────────────


def page_single(score_fn) -> None:
    st.header("Single molecule")
    st.info(EXPLORATORY_BANNER)
    smiles = st.text_input("SMILES", value="COc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OC")
    if st.button("Screen", type="primary") and smiles.strip():
        try:
            res = score_fn(smiles.strip())
        except Exception as exc:  # API may raise structured 422 for invalid SMILES
            st.error(f"Invalid SMILES or service error: {exc}")
            return
        _render_prediction(res)


def page_batch(batch_fn) -> None:
    st.header("Batch screening")
    st.info(EXPLORATORY_BANNER)
    uploaded = st.file_uploader(
        "Upload SMILES (.smi/.txt one per line, or .csv)", type=["smi", "txt", "csv"]
    )
    text = st.text_area("…or paste SMILES (one per line)", height=120)
    if st.button("Screen batch", type="primary"):
        smiles = _parse_smiles_input(text, uploaded)
        if not smiles:
            st.warning("No SMILES provided.")
            return
        smiles = smiles[:512]
        with st.spinner(f"Scoring {len(smiles)} molecules…"):
            results = batch_fn(smiles)
        rows = []
        for r in results:
            admet = r.get("admet") or {}
            ad = r.get("applicability_domain") or {}
            rows.append(
                {
                    "smiles": r.get("smiles"),
                    "valid": r.get("valid"),
                    "pic50_mutant": r.get("pic50_mutant"),
                    "pic50_wt": r.get("pic50_wt"),
                    "selectivity_proxy": r.get("selectivity_proxy"),
                    "covalent": r.get("covalent"),
                    "admet": admet.get("status"),
                    "qed": admet.get("qed"),
                    "domain": ad.get("domain"),
                    "warnings": " | ".join(r.get("warnings", [])),
                }
            )
        df = pd.DataFrame(rows)
        n_valid = int(df["valid"].sum())
        st.success(f"Scored {len(df)} — {n_valid} valid, {len(df) - n_valid} invalid.")
        st.dataframe(df, width="stretch")
        st.download_button(
            "Download CSV", df.to_csv(index=False), "batch_screen.csv", "text/csv"
        )


def page_ranking() -> None:
    st.header("Final integrated ranking (Phase 23)")
    st.info(EXPLORATORY_BANNER)
    df = dl.load_final_ranking()
    if df is None:
        st.warning(
            "final_ranked_candidates.csv not found. Run scripts/rank_candidates.py."
        )
        return

    place = dl.ranking_placement(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total candidates", place.get("n_total"))
    c2.metric("Known", place.get("n_known"))
    c3.metric("Generated", place.get("n_generated"))
    c4.metric(
        "Best generated rank",
        (
            f"#{place.get('best_generated_rank')}"
            if place.get("best_generated_rank")
            else "—"
        ),
    )

    if place.get("best_generated_rank"):
        st.markdown(
            f"**Generated vs known:** best generated **{place['best_generated_cid']}** at "
            f"**#{place['best_generated_rank']}** (final {place['best_generated_final']:.3f}); "
            f"median generated rank {place['median_generated_rank']}/{place['n_total']}; "
            f"{place['generated_in_top20']} in the top-20. "
            f"Best known **{place['best_known_cid']}** (final {place['best_known_final']:.3f}). "
            "De novo molecules rank *alongside but not above* the best known actives."
        )

    try:
        import altair as alt

        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("rank:Q", title="rank"),
                y=alt.Y("final_score:Q", title="final score"),
                color=alt.Color(
                    "source:N",
                    scale=alt.Scale(
                        domain=["known", "generated"], range=["#4C78A8", "#E45756"]
                    ),
                ),
                tooltip=["rank", "cid", "source", "final_score", "warnings"],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.bar_chart(df.set_index("rank")["final_score"])

    only_warned = st.checkbox("Show only rows with warnings", value=False)
    view = df[df["warnings"].astype(str).str.len() > 0] if only_warned else df
    st.dataframe(view, width="stretch", height=420)
    st.download_button(
        "Download ranking CSV",
        df.to_csv(index=False),
        "final_ranked_candidates.csv",
        "text/csv",
    )


def page_performance() -> None:
    st.header("Model performance")
    st.info(EXPLORATORY_BANNER)

    st.subheader("QSAR — single scaffold split (test set)")
    m = dl.load_qsar_metrics()
    if not m.empty:
        st.dataframe(m, width="stretch", hide_index=True)
    st.caption(
        "Single-split metrics on ~150–180 molecules are high-variance under scaffold "
        "splitting — do not over-read one seed."
    )

    st.subheader("QSAR — 5-seed scaffold-split stability (mean ± std)")
    ss = dl.load_seed_stability()
    st.dataframe(ss, width="stretch", hide_index=True)
    try:
        import altair as alt

        long = ss.melt(
            id_vars="model", value_vars=["r2_mean"], var_name="metric", value_name="R2"
        )
        long["std"] = ss["r2_std"].values
        base = alt.Chart(long).encode(x=alt.X("model:N", title=None))
        bar = base.mark_bar(color="#4C78A8").encode(
            y=alt.Y("R2:Q", title="R² (mean ± std)")
        )
        err = base.mark_errorbar().encode(y=alt.Y("R2:Q"), yError="std:Q")
        st.altair_chart((bar + err).properties(height=240), use_container_width=True)
    except Exception:
        pass
    st.caption(
        "Model 1 R² std ±0.143 is structural (scaffold partition variance at seed 99). "
        "Model 2 is steadier. Reliable Model 2 R² ≈ 0.507 ± 0.063, not the lucky 0.604."
    )

    st.subheader("Fingerprint ablation (winner by val RMSE)")
    abl = dl.load_fingerprint_ablation()
    for task, frame in abl.items():
        st.markdown(f"**{task}**")
        st.dataframe(frame, width="stretch", hide_index=True)
    st.caption(
        "morgan_ecfp6 nominally wins but the margin sits inside seed noise on the general "
        "task; production keeps Morgan ECFP4."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Model 3 — L858R calibration")
        m3 = dl.load_model3_verdict()
        if m3:
            st.dataframe(m3["table"], width="stretch", hide_index=True)
            st.error(f"**Verdict (n={m3['n_l858r']}):** {m3['verdict']}")
        else:
            st.info("loocv_results.json not found.")
    with c2:
        st.subheader("Model 4 — derived selectivity")
        m4 = dl.load_model4_verdict()
        if m4:
            st.metric(
                "Spearman r (n=%s pairs)" % m4["n_pairs"],
                round(m4["spearman_derived"], 3),
                help=f"p = {m4['pvalue_derived']:.3f}",
            )
            st.error(
                "**Verdict:** not significant at n=9 — selectivity is not modelable; "
                "structure-based methods (docking/FEP) are the path."
            )
        else:
            st.info("selectivity_results.json not found.")


def page_docking() -> None:
    st.header("Docking results")
    st.info(EXPLORATORY_BANNER)

    st.subheader("B2 sanity check — clinical inhibitors favour the L858R pocket")
    sc = dl.load_sanity_check()
    if sc:
        st.success(f"Verdict: **{sc['verdict']}** — {sc.get('verdict_detail', '')}")
        st.dataframe(sc["table"], width="stretch", hide_index=True)
    st.caption(
        "delta = L858R − WT (kcal/mol); negative = L858R-favoured. Rigid docking "
        "recovers the direction (~0.4–0.6) but underestimates the ~1.7 magnitude."
    )

    st.subheader("Seed-noise selectivity shortlist (top-15, 5 seeds)")
    nz = dl.load_docking_noise()
    if nz is not None:
        try:
            import altair as alt

            nz2 = nz.copy()
            nz2["lo"] = nz2["delta"] - 1.5 * nz2["std_delta"]
            nz2["hi"] = nz2["delta"] + 1.5 * nz2["std_delta"]
            base = alt.Chart(nz2).encode(y=alt.Y("cid:N", sort="x", title=None))
            pts = base.mark_point(filled=True, size=70, color="#E45756").encode(
                x=alt.X(
                    "delta:Q", title="selectivity delta (kcal/mol, − = L858R-selective)"
                ),
                tooltip=["cid", "delta", "std_delta", "call"],
            )
            bars = base.mark_rule().encode(x="lo:Q", x2="hi:Q")
            rule0 = (
                alt.Chart(pd.DataFrame({"x": [0]}))
                .mark_rule(strokeDash=[4, 4], color="gray")
                .encode(x="x:Q")
            )
            st.altair_chart(
                (bars + pts + rule0).properties(height=380), use_container_width=True
            )
        except Exception:
            st.dataframe(nz, width="stretch", hide_index=True)
        st.dataframe(nz, width="stretch", hide_index=True)
        st.caption(
            "Error bars = ±1.5×std_delta (the confident-call threshold). 7/9 non-covalent "
            "clear it; cmpd_010 was a single-seed artifact; covalent hits stay low-confidence."
        )

    st.subheader("Generated candidates — post-hoc docking")
    gd = dl.load_generated_docking()
    if gd is not None and not gd.empty:
        n_sel = int((gd["selectivity_delta"] < 0).sum())
        st.metric("L858R-selective (generated)", f"{n_sel}/{len(gd)}")
        st.dataframe(
            gd[
                [
                    "cid",
                    "pred_pic50",
                    "l858r_score",
                    "wt_score",
                    "selectivity_delta",
                    "admet_qed",
                    "domain",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Generated molecules were strongly directional (17/19 L858R-selective), "
            "but docking remains rigid-receptor and coarse."
        )


def page_limitations() -> None:
    st.header("Limitations — the honest findings")
    st.markdown(
        """
This platform is a **methodology demonstrator built around extreme data scarcity**, not a
validated discovery engine. Stated plainly:

- **L858R ML does not beat the backbone at n=22.** Transfer-learning calibration on the 22 true
  L858R records (LOOCV) did **not** improve rank correlation over the general backbone
  (Spearman r 0.620 ± 0.008). There is no separable L858R-specific signal at this size — the
  general backbone is what you should use, and all L858R output is labeled exploratory.

- **Selectivity is not modelable at n=9.** The derived L858R-vs-WT selectivity over the 9 paired
  molecules is not statistically significant (Spearman r 0.433, p ≈ 0.24). Structure-based methods
  (docking, and ideally FEP) are the only credible path; the 9 deltas are reference data, not a model.

- **RL either reward-hacks or stalls.** At σ=0.5 with no diversity filter the generator
  **reward-hacked** — collapsing onto ~14 scaffolds (uniqueness 80%→8.6%). Adding a scaffold-memory
  filter and dropping to σ=0.25 **prevented the collapse** but then the activity signal was too weak
  to move pIC50 (+0.004, **inconclusive**). Activity and diversity are in direct tension at this
  corpus size. The production generator stays the non-RL fine-tuned checkpoint.

- **Docking is rigid-receptor and coarse.** Both pockets (2ITZ L858R / 2ITY WT) are treated as rigid,
  across two different crystal structures. The direction of the known inhibitors is correct, but the
  magnitude is underestimated (~0.4 vs ~1.7 kcal/mol expected) and seed noise (±0.1–0.3) swamps small
  deltas. CNN rescoring (GNINA) was borderline and is not used for ranking.

- **The backbone is in-sample for the known library.** The top-ranked known candidates were selected
  as the highest backbone-predicted actives, so their activity scores are not held-out estimates.
  Generated molecules are genuinely novel but are scored by that same in-sample backbone — doubly
  exploratory.

- **No experimental validation.** Nothing here has been synthesised or assayed. Every pIC50, every
  selectivity delta, every ADMET flag is a computational estimate. The composite ranking orders
  candidates by *aggregated evidence*; it is **not** a calibrated probability of success.

The value of the project is the **discipline**: honest negative results (Models 3/4, RL), uncertainty
quantification (seed std, docking error bars), anti-reward-hacking guards, and confidence-aware
ranking where covalent/within-noise liabilities are surfaced as warnings rather than silently buried.
"""
    )


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    st.sidebar.title("🧬 EGFR L858R Discovery")
    base_url = st.sidebar.text_input("API base URL", api.DEFAULT_BASE_URL)

    page = st.sidebar.radio(
        "Page",
        [
            "Single molecule",
            "Batch screening",
            "Final ranking",
            "Model performance",
            "Docking results",
            "Limitations",
        ],
    )

    # Only the scoring pages need a live scorer.
    score_fn = batch_fn = None
    mode = "—"
    if page in ("Single molecule", "Batch screening"):
        score_fn, batch_fn, mode = resolve_scorer(base_url)
    st.sidebar.caption(f"Scoring backend: {mode}")
    st.sidebar.caption("All outputs EXPLORATORY — see Limitations.")

    if page == "Single molecule":
        page_single(score_fn)
    elif page == "Batch screening":
        page_batch(batch_fn)
    elif page == "Final ranking":
        page_ranking()
    elif page == "Model performance":
        page_performance()
    elif page == "Docking results":
        page_docking()
    elif page == "Limitations":
        page_limitations()


if __name__ == "__main__":
    main()
