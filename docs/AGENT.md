# Discovery Copilot

A bounded, deterministic AI copilot that orchestrates the EGFR L858R pipeline's
existing tools and explains their precomputed outputs. No LLM is required for v1.

> **Scope**: this module routes queries to deterministic functions and returns
> grounded answers. It never recomputes science, never fabricates numbers, and
> always labels outputs as EXPLORATORY.

---

## Architecture

```
src/agent/
  __init__.py      module docstring
  schemas.py       typed dataclasses for every tool input/output
  guardrails.py    scientific warning injection + forbidden-claim sanitizer
  retrieval.py     keyword retrieval over project markdown docs (no vector DB)
  report.py        template-based markdown report assembler
  tools.py         deterministic tool functions (reuse existing artifacts only)
  prompts.py       system prompt + LLM hook placeholder
  controller.py    intent classifier + tool dispatcher + answer assembler
```

Dependency order (no circular imports):

```
schemas  <-  guardrails, retrieval, report
report   <-  tools
tools    <-  controller
retrieval, guardrails, prompts  <-  controller
```

---

## Design principles

**Deterministic-first**: every answer comes from precomputed artifacts on disk
(`data/generated/final_ranked_candidates.csv`, `models/qsar/*.json`,
`models/generator/*.json`). No training, no docking, no LLM call at query time.

**LLM is optional, never required**: `prompts.llm_summarize()` currently returns
`None`. When it returns a string, the controller uses it as the answer. When it
returns `None`, the structured deterministic output is used directly. The hook is
the only place an LLM call belongs; it must never invent numbers.

**Never fabricate**: `lookup_docking_results` returns `found=False` with an
explanatory `message` when a candidate has no docking data. It never produces a
synthetic score. `lookup_final_ranking` behaves identically.

**Warnings, not suppression**: the guardrail layer appends scientific caveats to
results. It never hides or modifies numeric scores.

---

## Tool reference

### `predict_smiles(smiles, registry=None)`

Calls `ModelRegistry.score(smiles)` and returns a `PredictToolResult`.

| Field | Description |
|---|---|
| `valid` | False if RDKit cannot parse the SMILES |
| `pic50_mutant` | Backbone (Model 1, RandomForest) pred pIC50, EXPLORATORY |
| `pic50_wt` | WT-proxy (Model 2, XGBoost) pred pIC50, EXPLORATORY |
| `selectivity_proxy` | `pic50_mutant - pic50_wt`, ML only, not statistically validated |
| `covalent` / `warheads` | SMARTS-based electrophilic warhead detection |
| `admet_status` / `qed` | Lipinski/Veber/PAINS/Brenk/QED approximate filter |
| `domain` / `confidence_factor` | Applicability domain band (in/borderline/out) |
| `warnings` | List of scientific caveats appended by guardrails |

### `batch_predict(smiles_list, registry=None)`

Calls `predict_smiles` for each entry; invalid rows are included, not dropped.
Cap: 512 SMILES per call.

### `lookup_final_ranking(candidate_id)`

Reads `data/generated/final_ranked_candidates.csv`. Returns `RankingLookupResult`
with `found=False` for unknown IDs. Fields: rank (out of 68), source (known /
generated), final_score, activity/selectivity/affinity/admet norms (each 0-1),
confidence_factor, is_covalent, warnings string.

### `lookup_docking_results(candidate_id)`

Merges three docking artifact files in priority order:

1. `models/qsar/library_docking_results.json` (50 known candidates, seed-42 scores)
2. `models/qsar/docking_noise_results.json` (top-15 compounds, 5-seed mean/std)
3. `models/generator/generated_docking_results.json` (19 generated candidates)

Returns `DockingLookupResult`. Noise-study fields (`mean_delta`, `std_delta`,
`noise_call`) are populated only for the top-15 compounds. Never fabricates a score.

### `compare_candidates(ids, registry=None)`

Ranks two or more candidates by a conservative score:

```
+2  non-covalent
+2  in_domain (confidence_factor == 1.0)
+[0-1]  admet_norm from ranking CSV
+[0-1]  final_score from ranking CSV
+0.5  noise-study call: L858R_selective (mean delta populated in docking index)
```

Returns `ComparisonResult` with `recommendation` (ID of preferred candidate) and
`reason` (plain English, EXPLORATORY caveat appended).

### `generate_candidate_report(candidate_id, registry=None)`

Calls `lookup_final_ranking`, `lookup_docking_results`, and (if the candidate's
SMILES is in the ranking CSV) `predict_smiles`. Assembles a markdown document via
`report.generate_report()`. Sections: Summary, Composite Ranking, Activity
Prediction, Docking Results, ADMET Profile, Warnings, Limitations.

---

## Guardrails

### `add_scientific_warnings(result)`

Appends standard caveats to the `warnings` list of any result object. Rules:

| Trigger | Caveat appended |
|---|---|
| `selectivity_proxy` is not None | ML-derived proxy, not validated at n=9 |
| `l858r_score` is not None | Rigid-receptor docking caveat |
| `covalent=True` or `warheads` non-empty | Rigid docking underestimates covalent binding |
| `domain == "out_of_domain"` | confidence_factor=0.50, treat with caution |
| `domain == "borderline"` | confidence_factor=0.75 |
| `source == "generated"` | In-sample backbone, doubly exploratory |

### `find_forbidden_claims(text)` / `sanitize_text(text)`

Scans text for experimental claims that exceed what the pipeline can support.
Returns a list of found claim labels; an empty list means the text is clean.

**Forbidden phrases** (when not negated): "is active", "is selective",
"drug candidate", "validated", "proven", "confirmed".

**Negation detection**: a negation word ("not", "no", "never", "un...", "isn't",
etc.) within 30 characters before a forbidden phrase causes it to be ignored.

Examples that **pass** the sanitizer:
- "not validated experimentally"
- "no confirmed activity"
- "this is not a drug candidate"
- "unproven in cell lines"

Examples that are **flagged**:
- "This compound is active against EGFR."
- "Binding was confirmed by SPR."
- "This is a drug candidate for NSCLC."

---

## Retrieval

`retrieval.retrieve(query, top_k=5)` scores every section of the three
documentation files (README.md, docs/PROJECT_WALKTHROUGH.md, CLAUDE.md) by
keyword overlap with the query:

```
score = (3 * header_hits + body_hits) / len(query_tokens)
```

Header tokens count 3x so that a section whose title names the topic ranks above
one that merely mentions it in passing. No embeddings or vector DB required.
Sections are cached in memory after the first load.

---

## Controller: intent classification

`classify_intent(query)` matches the lowercased query against keyword lists in
priority order:

| Intent | Keywords |
|---|---|
| `report` | "report", "summarize", "summary" |
| `comparison` | "compare", " vs ", "better than", "prefer" |
| `docking_query` | "dock", "vina", "binding", "pocket", "kcal/mol" |
| `batch_predict` | "batch", "multiple smiles", "screen list" |
| `candidate_lookup` | "cmpd_", "gen_", "rank", "ranking", "shortlist" |
| `single_predict` | "smiles", "predict", "score", "pic50", "activity of" |
| `project_qa` | "what", "how", "why", "explain", "limitation", "phase", etc. |
| `unknown` | no match |

`handle(request, registry=None)` dispatches to tools, collects warnings, optionally
calls `llm_summarize`, and returns an `AgentResponse(intent, answer, tool_results,
warnings, sources)`.

---

## Running

```bash
# Verify all agent unit tests pass
PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_agent_tools.py \
  tests/test_agent_guardrails.py tests/test_agent_retrieval.py \
  tests/test_agent_report.py tests/test_agent_controller.py -v

# Quick programmatic use (deterministic mode, no LLM)
PYTHONPATH=. .venv/Scripts/python.exe - <<'EOF'
from src.agent.controller import handle
from src.agent.schemas import AgentRequest

resp = handle(AgentRequest(query="Look up cmpd_015", candidate_ids=["cmpd_015"]))
print(resp.answer)
EOF

# With the real ModelRegistry loaded (requires model artifacts)
PYTHONPATH=. .venv/Scripts/python.exe - <<'EOF'
from src.agent.tools import predict_smiles

result = predict_smiles("COc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OC")
print(f"pIC50={result.pic50_mutant}, domain={result.domain}")
for w in result.warnings:
    print(" *", w)
EOF
```

---

## Wiring an LLM (v2, optional)

1. Implement the body of `prompts.llm_summarize(system_prompt, context)` to call
   the Anthropic API (or any other provider). Return the model's response string.
2. The controller passes `SYSTEM_PROMPT` and the full deterministic context to it;
   the LLM may only rephrase, not invent numbers.
3. The guardrail layer scans the final answer for forbidden claims before it is
   returned, regardless of whether the LLM or deterministic path produced it.
4. No other files need to change for LLM integration.

See `src/agent/prompts.py` for the full system prompt. Key constraints it encodes:
every activity result must be labeled EXPLORATORY; forbidden claim list is enforced;
docking direction is reliable but magnitude is not; missing values must be stated,
not estimated.

---

## Streamlit Dashboard Page (Phase 27)

`src/dashboard/copilot_page.py` adds a **Discovery Copilot** tab to the existing
six-page dashboard (`src/dashboard/app.py`).

### Running

```bash
# Start the dashboard (includes the Copilot page)
PYTHONPATH=. .venv/Scripts/python.exe -m streamlit run src/dashboard/app.py
# → http://localhost:8501, select "Discovery Copilot" in the sidebar
```

The page does **not** require the FastAPI service — it loads `ModelRegistry` locally
via `@st.cache_resource`, exactly like the existing dashboard pages. The API-or-local
fallback is preserved for the Single Molecule and Batch pages; the Copilot page always
uses the local registry.

### UI layout

```
Discovery Copilot
  [5 example-prompt buttons]
  ─────────────────────────
  [chat history, oldest first]
    user: …query…
    assistant:
      Panel 1: grounded answer (markdown)
      Panel 2: Evidence ▶  (tool calls + what they returned, collapsible)
      Panel 3: Warnings ▶  (guardrail caveats, collapsible)
      [Download report as Markdown]  ← appears only when a CandidateReport is in results
  ─────────────────────────
  [chat_input: "Ask the Discovery Copilot…"]
```

Example prompts (label → query):

| Label | Query |
|---|---|
| Single molecule | `Predict: COc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OC` |
| Compare | `Compare cmpd_015 and cmpd_024` |
| Candidate report | `Generate a report for gen_005` |
| Explain ranking | `Explain the ranking for cmpd_002` |
| Project QA | `Why did the RL training fail?` |

### Lazy registry loading

`_registry()` is decorated with `@st.cache_resource`. It is invoked **only when a
query is submitted**, not on the initial empty page render. This keeps the cold-start
time for the copilot page near-instant (< 1 s) even when model artifacts are large.

### Pure helper functions (`copilot_page.py`)

These functions have no Streamlit dependency and are fully unit-tested:

| Function | Purpose |
|---|---|
| `format_evidence(tool_results)` | Convert result objects to one string per tool call |
| `extract_report_markdown(tool_results)` | Return markdown from first `CandidateReport` in list, or `None` |
| `get_download_filename(tool_results)` | Return `report_<cid>.md` or `report.md` |

`format_evidence` uses `type(r).__name__` matching — it does not import agent schema
classes, keeping the dashboard dependency graph clean.

### Tests

`tests/test_agent_copilot.py` — 29 tests total:

- **26 `@unit`**: `TestFormatEvidence` (13 tests), `TestExtractReportMarkdown` (5),
  `TestGetDownloadFilename` (5), `test_example_prompts_count`,
  `test_example_prompts_cover_five_flows`. All use lightweight `_make(class_name, ...)` 
  objects — no Streamlit runtime, no model artifacts.
- **3 `@integration`**: `TestCopilotPageRenders` — runs the real app via Streamlit's
  `AppTest`, navigates to the Discovery Copilot page, and asserts no exception on the
  empty initial render. The registry is **not** loaded (no query submitted), so this
  test completes in a few seconds without model artifacts.

---

## Limitations and future work

- **No session memory**: each `handle()` call is stateless. Conversation context
  (e.g. "compare it to the previous one") requires the caller to re-specify IDs.
- **Retrieval is keyword-only**: dense retrieval (embeddings) would improve recall
  for paraphrased questions.
- **Docking index covers ~70 candidates**: compounds not in the precomputed files
  return `found=False`. Adding new candidates requires re-running the docking pipeline.
- **LLM hook is a stub**: v1 returns raw structured text. Wire `prompts.llm_summarize`
  to the Anthropic API for natural-language reformatting (see Wiring section above).
