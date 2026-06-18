# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Mutation-aware AI drug discovery platform for EGFR L858R non-small cell lung cancer (NSCLC). The critical constraint throughout: **only ~19 labeled L858R records exist**, which is insufficient for a standalone model. The entire modeling strategy is designed around this scarcity.

Modeling strategy (four-model cascade):
1. **EGFR general backbone** — trained on ~1253 all-mutation records
2. **WT-proxy comparator** — trained on ~1018 molecules (61 explicit WT + 957 unspecified-mutation ChEMBL EGFR, which in practice are WT)
3. **L858R-adapted** — general model + LOOCV calibration on 22 records (transfer learning mandatory; see re-scan note below)
4. **T790M** (planned) — genuine T790M single-mutant records only (see re-scan note below)

### L858R re-scan findings (fuzzy assay-description scan, June 2026)

The original `mutation_flag` in `data/raw/chembl_egfr_bioactivity.csv` was assigned in the Colab notebook and contains two known errors:

- **+3 relabeled pure L858R** (of 5 identified): assay CHEMBL4380726 ("Binding affinity to human wild-type partial length EGFR L858R mutant, KINOMEscan") is flagged `wild_type` but measures Kd on the L858R construct. The "wild-type" refers to the protein backbone, not the mutation. Fix is applied in `scripts/clean_bioactivity_data.py` (keyed on assay ID, idempotent). 2 of the 5 records (CHEMBL4529558, CHEMBL4578319) are PROTAC/degrader molecules with 74 heavy atoms — legitimately filtered by the `max_heavy=70` cleaning cutoff. Net L858R count: **19 → 22**. wild_type: **64 → 61**.

- **T790M bucket is contaminated**: 130 of the 211 `T790M`-flagged records are actually compound mutants:
  - 92 records = **L858R/T790M double mutant** (should get label `L858R_T790M`)
  - 38 records = **L858R/T790M/C797S triple mutant** (should get label `L858R_T790M_C797S`)
  - Only 81 records are genuine T790M single-mutant
  - This matters for Model 4 (T790M) — do the relabeling before building it.

- **No H3255 or Ba/F3 cell lines** found anywhere in the raw file. No exon-21 mentions. The 8 PC9 hits are all del19, correctly labeled.

## Environment setup

```bash
# Install base dependencies
pip install -r requirements/base.txt

# For model training (XGBoost, LightGBM, MLflow, Optuna)
pip install -r requirements/ml.txt

# For GNN (PyTorch + PyG)
pip install -r requirements/gnn.txt

# For development (pytest, ruff, black)
pip install -r requirements/dev.txt

# For Google Cloud (optional)
pip install -r requirements/cloud.txt
```

Copy `.env.example` to `.env` and configure. Key env vars:
- `USE_GCS=false` — toggle local vs. GCS artifact storage
- `MLFLOW_TRACKING_URI=mlruns` — MLflow experiment tracking
- `RANDOM_SEED=42` — global seed

## Commands

```bash
# Lint
ruff check src/ scripts/ tests/

# Format
black src/ scripts/ tests/

# Run all tests
pytest

# Run a single test
pytest tests/path/to/test_file.py::TestClass::test_function -v

# Run by marker (see pyproject.toml for all markers)
pytest -m unit          # fast, no training
pytest -m integration   # requires real data files
pytest -m "not slow and not heavy"

# Data pipeline (run in order)
# NOTE: the package is not installed; always prefix with PYTHONPATH=. and use the venv python
PYTHONPATH=. .venv/Scripts/python.exe scripts/clean_bioactivity_data.py   # raw -> data/interim/egfr_cleaned.csv
PYTHONPATH=. .venv/Scripts/python.exe scripts/build_egfr_dataset.py       # interim -> data/processed/ (3 split datasets)
PYTHONPATH=. .venv/Scripts/python.exe scripts/compute_features.py         # processed -> .parquet feature matrices (no split column)
PYTHONPATH=. .venv/Scripts/python.exe scripts/assign_splits.py            # adds split column to general/wt_proxy/erbb2 parquets (not L858R)
PYTHONPATH=. .venv/Scripts/python.exe scripts/train_models.py            # trains Models 1 (general) and 2 (WT-proxy), saves to models/qsar/
PYTHONPATH=. .venv/Scripts/python.exe scripts/train_l858r_model.py      # Model 3 LOOCV (~20 min); results in models/qsar/l858r/loocv_results.json
PYTHONPATH=. .venv/Scripts/python.exe scripts/eval_selectivity.py       # Model 4 selectivity; results in models/qsar/selectivity/selectivity_results.json
PYTHONPATH=. .venv/Scripts/python.exe scripts/eval_seed_stability.py    # 5-seed stability eval for Models 1 and 2 (~8 min)
PYTHONPATH=. .venv/Scripts/python.exe scripts/prepare_docking.py        # Phase B1: download + prepare 2ITZ/2ITY, write PDBQT, set box center (~30 s)
PYTHONPATH=. .venv/Scripts/python.exe scripts/sanity_check_docking.py  # Phase B2: align WT, prep ligands, 6 Vina runs, verdict (~3 min)
PYTHONPATH=. .venv/Scripts/python.exe scripts/rescore_sanity_poses.py  # Phase B3: GNINA CNN rescore of B2 poses via WSL2 (~3 min)
PYTHONPATH=. .venv/Scripts/python.exe scripts/dock_library.py          # Phase B2 library: dock top-50 backbone candidates, selectivity ranking (~50-100 min)
PYTHONPATH=. .venv/Scripts/python.exe scripts/eval_docking_noise.py   # Phase B2 noise eval: top-15 x 5 seeds x 2 pockets, delta error bars (~60-90 min)
```

## Architecture

```
src/
  data/         # cleaning.py, standardization.py — SMILES validation, canonicalization, dedup
  features/     # fingerprints.py (Morgan/MACCS/RDKit/AtomPair), descriptors.py (11 RDKit descriptors)
                # covalent.py — SMARTS-based electrophilic warhead detector (acrylamide, vinyl sulfone, etc.)
  splitting/    # scaffold_split.py — Bemis-Murcko scaffold splitting to prevent leakage
  models/       # qsar.py — QSARTrainer (RF/XGB/LGB, val-based selection, save/load)
  generation/   # (to be implemented) GRU SMILES generator + RL fine-tuning
  scoring/      # (to be implemented) composite ranking
  docking/      # prepare_protein.py (download + pdbfixer + PDBQT), prepare_ligands.py (rigid PDBQT)
  admet/        # (to be implemented) ADMET filters
  api/          # (to be implemented) FastAPI service
  dashboard/    # (to be implemented) Streamlit dashboard
  utils/        # config.py, logging.py, seeds.py

scripts/        # Entry-point data pipeline scripts (importable src modules, not standalone)
config/
  paths.yaml           # all file paths as dot-navigable keys (use get_path("data.processed.mutant"))
  model_config.yaml    # hyperparameters, fingerprint configs, scoring weights
  docking_config.yaml  # structure choices, box center/size, excluded structures (updated by prepare_docking.py)
```

## Key design decisions

**Feature vector**: Morgan ECFP4 (2048 bits) + 11 RDKit descriptors = **2059 features total**. Column order in `DESCRIPTOR_NAMES` ([src/features/descriptors.py](src/features/descriptors.py):12) is fixed — never reorder, models depend on this.

**Fingerprint type names**: must match keys in `model_config.yaml` (`morgan_ecfp4`, `morgan_ecfp6`, `maccs`, `rdkit_topological`, `atom_pair`). The primary type is `morgan_ecfp4_desc` (ECFP4 + descriptors).

**Scaffold splitting**: always use Bemis-Murcko scaffold split (`src/splitting/scaffold_split.py`) — random splits leak chemically similar molecules between train/test.

**Config access**: use `src.utils.config` functions rather than hardcoding paths. `get_path("data.processed.mutant")` returns an absolute `Path` resolved against project root. `load_model_config()` for hyperparameters.

**WT-proxy naming**: the WT dataset contains explicit WT + unspecified-mutation records. Call it "WT-proxy", never "WT-only".

**Selectivity delta**: `selectivity_delta = pIC50_mutant - pIC50_wt`. Positive = mutant-selective (desired). Final composite score weights: mutant_activity 0.35, selectivity 0.25, docking 0.20, admet 0.20.

**Applicability domain**: Tanimoto similarity threshold 0.50 for "in-domain", 0.30 for "borderline". Confidence factors applied to predictions outside domain.

### Trained model artifacts

Artifacts on disk were trained at seed=42.  Single-split metrics on ~150-180 test molecules are **high-variance** under scaffold splitting; do not compare single-seed numbers between models or pipeline versions.

| Model | Artifact dir | Best candidate | Test RMSE | Test R² | Pearson r | Test n |
|---|---|---|---|---|---|---|
| 1 — EGFR general backbone | `models/qsar/general/` | RandomForest | 1.076 | 0.446 | 0.698 | 183 |
| 2 — WT-proxy | `models/qsar/wt_proxy/` | XGBoost | 0.868 | 0.604 | 0.792 | 153 |

Each artifact dir contains `best_model.pkl`, `scaler.pkl`, `metadata.json` (val metrics for all candidates + test metrics for winner). Load via `QSARTrainer.load(path, cfg)`.

#### 5-seed scaffold-split stability (seeds 42, 7, 13, 99, 123)

Run `PYTHONPATH=. .venv/Scripts/python.exe scripts/eval_seed_stability.py` to reproduce.

| Model | RMSE (mean ± std) | R² (mean ± std) | Pearson r (mean ± std) |
|---|---|---|---|
| 1 — EGFR general backbone | 1.010 ± 0.167 | 0.438 ± 0.143 | 0.672 ± 0.105 |
| 2 — WT-proxy | 0.942 ± 0.061 | 0.507 ± 0.063 | 0.717 ± 0.047 |

The high variance on Model 1 (R² std ±0.143) is structural: the scaffold partition at seed=99 places structurally unusual test compounds that neither model was exposed to. Model 2's lower variance reflects the more homogeneous kinase-inhibitor chemical space. The seed=42 value of R²=0.604 for Model 2 is at the lucky end of the ±0.063 range; the reliable estimate is **0.507 ± 0.063**.

**The 3-record L858R relabeling did not meaningfully affect Model 1 or 2 performance.** Those 3 molecules moved from wild_type to L858R; they are absent from the WT-proxy training set in both cases because L858R-flagged records are excluded from it. Any difference between a pre-relabel and post-relabel single-split run is scaffold-partition variance, not a model quality change.

#### Model 3 — L858R calibration (EXPLORATORY, LOOCV, seeds 42, 7, 13, 99, 123)

Run `PYTHONPATH=. .venv/Scripts/python.exe scripts/train_l858r_model.py` to reproduce (~20 min).
Results saved to `models/qsar/l858r/loocv_results.json` (no `best_model.pkl` — there is no single deployable artifact).

**Design:** In each LOO fold, the backbone is retrained on all 1252 general EGFR molecules *excluding the held-out L858R molecule*, preventing any leakage. Calibrators train on OOB backbone predictions for the 21 training L858R molecules to avoid in-sample bias.

| Method | Spearman r (mean ± std) | RMSE (mean ± std) |
|---|---|---|
| Backbone (baseline) | **0.620 ± 0.008** | 0.941 ± 0.014 |
| Mean-shift calibration | 0.599 ± 0.012 | 0.827 ± 0.012 |
| Ridge calibration | 0.593 ± 0.017 | 0.873 ± 0.013 |

**Verdict:** Calibration does not improve rank correlation (backbone Spearman r 0.620 beats both calibrators). The mean-shift reduces RMSE (0.827 vs 0.941) because the backbone systematically underpredicts L858R pIC50, but the uniform shift doesn't improve compound prioritization. **L858R-specific signal is not separable at n=22; use the general backbone.**

All downstream use of Model 3 must be labeled exploratory.

#### Model 4 — Derived selectivity (EXPLORATORY)

Run `PYTHONPATH=. .venv/Scripts/python.exe scripts/eval_selectivity.py` to reproduce (~5 s).
Results saved to `models/qsar/selectivity/selectivity_results.json`.

**Sign convention (confirmed):** `selectivity_delta = pIC50_mutant - pIC50_wt`. Positive = mutant-selective (desired). Verified against file contents before analysis.

**Data:** 9 paired molecules with both L858R and WT measurements (`data/processed/egfr_selectivity_dataset.csv`).

**Predictors used:** backbone (Model 1, RandomForest) for mutant pIC50; WT-proxy (Model 2, XGBoost) for WT pIC50. This is consistent with the Model 3 verdict (calibration did not improve over backbone).

**Critical caveats:**
- All 9 pairs are **in-sample** for both activity models — there is no holdout for the activity predictions.
- Backbone and WT-proxy share ~80% of their training data; the derived delta has a large shared-noise component.
- At n=9, Spearman r is statistically significant only for r > 0.67 (p < 0.05, two-sided).

| Method | Spearman r | p-value |
|---|---|---|
| **Derived delta (backbone − WT-proxy)** | **0.433** | 0.244 |
| LOO mean baseline | −1.000* | 0.000 |
| Constant mean baseline | n/a | n/a |

*LOO mean r = −1.000 is a mathematical artifact: excluding each molecule's true delta from the mean always produces predictions in exact reverse rank order. Low p-value does not indicate a useful predictor.

**Verdict:** Derived selectivity is not statistically significant (r=0.433, p=0.244) at n=9. **Selectivity cannot be modeled at n=9; structure-based methods (docking, FEP) are the path. Treat the 9 deltas as exploratory reference data, not a model.**

Note: the per-pair table reveals that the backbone substantially underpredicts high-selectivity compounds (e.g., pair 3: true delta +1.505, derived delta −0.052), because both models make large individual pIC50 errors that cancel out differently for each compound, washing out the selectivity signal.

## Fingerprint ablation (Phase 5, optional, DONE)

Results saved to `models/qsar/fingerprint_ablation_results.json`.
Run: `PYTHONPATH=. .venv/Scripts/python.exe scripts/fingerprint_ablation.py`

Settings: n_estimators=100, seeds=[42,7,13,99,123], RF/XGB/LGB; winner = lowest mean val RMSE.
Feature vector = FP bits + 11 RDKit descriptors for each representation.

### General task (EGFR backbone, n≈1253)

| Fingerprint | n feat | Best model | val RMSE | test RMSE (mean±std) | test R² (mean±std) | Spearman r (mean±std) |
|---|---|---|---|---|---|---|
| **morgan_ecfp6** | 2059 | XGBoost | **0.949** | 1.021 ± 0.181 | 0.424 ± 0.165 | 0.646 ± 0.137 |
| morgan_ecfp4 | 2059 | XGBoost | 0.973 | 1.039 ± 0.185 | 0.403 ± 0.175 | 0.639 ± 0.139 |
| topological_torsion | 2059 | XGBoost | 0.975 | 1.011 ± 0.142 | 0.433 ± 0.137 | 0.646 ± 0.126 |
| rdkit_topological | 2059 | XGBoost | 0.978 | 1.019 ± 0.171 | 0.424 ± 0.166 | 0.630 ± 0.136 |
| atom_pair | 2059 | XGBoost | 1.006 | 1.068 ± 0.136 | 0.374 ± 0.107 | 0.642 ± 0.078 |
| maccs | 178 | XGBoost | 1.059 | 1.077 ± 0.131 | 0.368 ± 0.068 | 0.617 ± 0.049 |

### WT-proxy task (n≈1018)

| Fingerprint | n feat | Best model | val RMSE | test RMSE (mean±std) | test R² (mean±std) | Spearman r (mean±std) |
|---|---|---|---|---|---|---|
| **morgan_ecfp6** | 2059 | XGBoost | **0.855** | 0.942 ± 0.059 | 0.506 ± 0.067 | 0.725 ± 0.049 |
| topological_torsion | 2059 | LightGBM | 0.875 | 0.945 ± 0.042 | 0.504 ± 0.061 | 0.711 ± 0.048 |
| rdkit_topological | 2059 | XGBoost | 0.890 | 0.949 ± 0.046 | 0.501 ± 0.049 | 0.729 ± 0.053 |
| morgan_ecfp4 | 2059 | LightGBM | 0.893 | 0.948 ± 0.049 | 0.502 ± 0.050 | 0.713 ± 0.034 |
| atom_pair | 2059 | LightGBM | 0.926 | 1.012 ± 0.048 | 0.434 ± 0.042 | 0.663 ± 0.033 |
| maccs | 178 | LightGBM | 0.950 | 1.006 ± 0.068 | 0.436 ± 0.087 | 0.671 ± 0.077 |

### Findings

- **Winners by val RMSE**: morgan_ecfp6 (XGBoost) on both tasks.
- **Margin over morgan_ecfp4**: 0.024 val RMSE (general) and 0.038 (WT-proxy). Both margins fall inside the test RMSE std (±0.18 and ±0.06 respectively), so the difference is **within seed noise** for general; slightly above noise for WT-proxy.
- **Top 4 FPs are statistically indistinguishable** on general (val RMSE 0.949–0.978). On WT-proxy the top 3 are similarly clustered (0.855–0.890).
- **MACCS is consistently weakest** on both tasks — 167 bits are too coarse for pIC50 regression on kinase inhibitors.
- **atom_pair** underperforms despite 2048 bits, suggesting distance-based atom-pair encoding adds less signal than topological paths for this chemical space.
- **Production models (Morgan ECFP4) remain the right default**: the nominal winner (ECFP6) does not clear the seed-noise bar on the general task, and the WT-proxy gap (0.038) does not justify rebuilding all downstream artifacts.

## Phase B — Structure-based docking

### Structure pair (B1 — DONE)

Matched pair from Yun et al. 2007 Cancer Cell (same construct, same ligand):

| Role | PDB ID | Resolution | Mutation | Ligand |
|---|---|---|---|---|
| L858R | 2ITZ | 2.8 A | L858R | IRE (gefitinib) |
| WT | 2ITY | 3.42 A | wild_type | IRE (gefitinib) |

- Both: EGFR kinase domain 696-1022, chain A.
- Excluded (T790M-containing): 5UGA, 5UG8, 5UGC, 5UWD, 4I21 — listed in `excluded_structures` in `config/docking_config.yaml`.

### Preparation (B1 — DONE)

`scripts/prepare_docking.py` applies identically to both structures:
1. Downloads PDB from RCSB into `data/docking/protein/`.
2. Biopython: extracts chain A, removes all HETATM except IRE, saves protein-only PDB.
3. pdbfixer: `findMissingResidues` (clears `missingResidues` to skip loop modelling) → `findMissingAtoms` → `addMissingAtoms` → `addMissingHydrogens(7.4)`.
4. PDBQT writer: heavy atoms + polar H (bonded to N/O/S within 1.15 A, Biopython NeighborSearch). Atom types from AutoDock4/Vina convention (A for aromatic C, OA for O, SA for S, NA for ring N acceptors, HD for polar H). Receptor charges = 0.000 (Vina ignores them).
5. IRE (gefitinib) extracted separately as PDB + rigid PDBQT (RDKit Gasteiger charges, TORSDOF=0).

Prepared files: `data/docking/protein/2ITZ_receptor.pdbqt` (2437 atoms), `data/docking/protein/2ITY_receptor.pdbqt` (2406 atoms).

### Docking box (B1 — DONE)

Center from heavy-atom centroid of IRE in 2ITZ chain A:
- **center_x = -51.654, center_y = -1.266, center_z = -21.945** (Angstrom)
- **size = 22.5 x 22.5 x 22.5 A** (covers ATP binding cleft with ~5 A buffer)
- Identical box used for both 2ITZ and 2ITY so scores are directly comparable.
- All values persisted in `config/docking_config.yaml`; tests verify reproducibility.

### Alignment (B2 — DONE)

`scripts/sanity_check_docking.py` step 1 superposes 2ITY onto 2ITZ on 300 common Ca atoms (Biopython Superimposer). Ca RMSD = **1.643 A** (expected for same construct with one point mutation). Aligned output:
- `data/docking/protein/2ITY_aligned.pdb` — Biopython PDBIO output with transformation applied to all atoms
- `data/docking/protein/2ITY_aligned_receptor.pdbqt` — 2406 atoms; same PDBQT writer as B1

Box coverage verified for both 2ITZ_prepared.pdb and 2ITY_aligned.pdb: Ca centroids are inside the shared box + 2 A pad.

### B2 sanity check (DONE — PASS)

Run `PYTHONPATH=. .venv/Scripts/python.exe scripts/sanity_check_docking.py` to reproduce (~3 min).
Results saved to `models/qsar/sanity_check_docking.json`.

Ligands: gefitinib, erlotinib, osimertinib prepared from SMILES using RDKit ETKDGv3 + MMFF94 + meeko 0.7.1 (flexible PDBQT). AutoDock Vina 1.2.7 binary (`data/docking/tools/vina.exe`, Windows pre-built); exhaustiveness=8, seed=42, 9 poses.

**Note: `pip install vina` FAILS on Windows** ("Boost library not found"). Use the pre-built binary from GitHub releases instead. `src/docking/vina_runner.py` calls it via subprocess. Meeko installs fine: `pip install meeko gemmi`.

| Compound | L858R (kcal/mol) | WT (kcal/mol) | delta (L858R - WT) | direction |
|---|---|---|---|---|
| gefitinib | -7.860 | -7.492 | -0.368 | L858R favoured |
| erlotinib | -7.666 | -7.263 | -0.403 | L858R favoured |
| osimertinib | -7.944 | -7.306 | -0.638 | L858R favoured |

**VERDICT: PASS.** All three clinically validated EGFR inhibitors favour the L858R pocket, consistent with the Yun et al. 2007 literature anchor (gefitinib binds L858R ~20-fold tighter than WT). Delta magnitudes (0.4-0.6 kcal/mol) are modest compared to the ~1.7 kcal/mol expected from a 20-fold affinity difference — rigid-receptor docking underestimates the effect, but the direction is correct.

### B3 CNN rescoring — BORDERLINE (EXPLORATORY)

Run `PYTHONPATH=. .venv/Scripts/python.exe scripts/rescore_sanity_poses.py` to reproduce (~3 min).
Results saved to `models/qsar/gnina_rescore_sanity.json`.

**Route:** GNINA v1.3.2 (CPU build) requires `libcudnn.so.9` — absent in WSL2 without NVIDIA GPU drivers. GNINA v1.0 (Dec 2021, 535 MB, `data/docking/tools/gnina_v1.0`) runs cleanly on WSL2 Ubuntu glibc 2.39 with no CUDA dependency. Called via `wsl -d Ubuntu --` subprocess from Windows Python (`src/docking/gnina_runner.py`). ~12 s per pose on CPU.

**Preprocessing:** meeko output PDBQTs contain null bytes and REMARK SMILES/H PARENT/INTER/UNBOUND tags that cause GNINA v1.0 parse errors. `extract_best_pose()` strips these and extracts MODEL 1.

CNN outputs: `CNNscore` (0-1 probability), `CNNaffinity` (pKd; 1 unit = 1.363 kcal/mol at 300 K), `CNNvariance` (ensemble uncertainty).

| Compound | L858R CNNaff (pKd) | WT CNNaff (pKd) | delta_pKd | delta_kcal(CNN) | delta_kcal(Vina) | CNN direction |
|---|---|---|---|---|---|---|
| gefitinib | 6.693 | 6.389 | +0.303 | +0.413 | -0.368 | L858R favoured |
| erlotinib | 5.909 | 5.670 | +0.239 | +0.326 | -0.403 | L858R favoured |
| osimertinib | 5.708 | 5.756 | -0.048 | -0.066 | -0.638 | WT favoured (near zero) |

**VERDICT: BORDERLINE.** Gefitinib and erlotinib: CNN correctly favours L858R with delta ~0.3-0.4 kcal/mol. Osimertinib: CNN gives WT a tiny advantage (-0.048 pKd, -0.066 kcal/mol), within ensemble noise (CNNvariance=1.66). Direction criterion (all 3 compounds) NOT fully met. **The CNN does not reliably improve on Vina for selectivity scoring. Keep Vina-only as the docking filter. Do not use CNNaffinity for ranked-library scoring until a validated CNN reproduces all 3 compounds.**

**Why osimertinib fails:** Osimertinib is a covalent 3rd-generation inhibitor (C797-targeting acrylamide). GNINA v1.0 CNN was trained on non-covalent PDBbind complexes; it may not recognize the covalent pharmacophore.

### LIMITATIONS (docking)

- **Dual-pocket comparability**: 2ITZ and 2ITY have slightly different crystal packing (2.8 A vs 3.42 A resolution, and the L858R substitution shifts helix C). Docking score differences between the two receptors reflect both the mutation effect AND differences in how the scoring function handles the two pocket geometries. Interpret L858R vs WT score deltas with caution.
- **Rigid receptor**: both receptors are treated as rigid. Induced-fit effects (especially in the glycine-rich loop and helix aC) are ignored.
- **Box centering**: the box is centered on 2ITZ IRE. After alignment, the 2ITY pocket is in the same frame, so the same box is valid. No re-centering needed.
- **Delta magnitude**: rigid Vina underestimates the 20-fold gefitinib affinity difference (~1.7 kcal/mol theoretical) to ~0.4 kcal/mol. Direction is correct; magnitude is not quantitatively reliable.
- **CNN not EGFR-specific**: GNINA v1.0 default models were trained on PDBbind cross-docking data, not EGFR. The CNN variance is high (~1.0-1.7) reflecting genuine model uncertainty. A finer-grained EGFR-specific CNN or FEP would be needed to recover the full 1.7 kcal/mol signal.

## Covalent warhead detection (`src/features/covalent.py`)

SMARTS-based heuristic detector for electrophilic warheads. Applied before any docking: flagged compounds receive `docking_confidence = "low_confidence"` because non-covalent rigid docking cannot model the covalent bond.

Public API:
- `detect_warheads(smiles)` → `list[str]` — names of matched warhead types
- `is_covalent(smiles)` → `bool`
- `covalent_confidence(smiles)` → `"low_confidence"` or `"standard"`

Warhead patterns (conservative; false negatives preferred over false positives):

| Key | SMARTS | Covers |
|---|---|---|
| `acrylamide` | `[NX3]C(=O)C=C` | osimertinib, afatinib, neratinib, dacomitinib |
| `acrylate_ester` | `C=CC(=O)[OX2H0]` | aryl/alkyl acrylate esters (cmpd_021); `[OX2H0]` = ester O (not carboxylic acid OH) |
| `propiolamide` | `[NX3]C(=O)C#C` | ynalamide warhead |
| `vinyl_sulfone` | `C=CS(=O)(=O)` | vinyl sulfone electrophile |
| `chloroacetamide` | `[Cl,Br][CH2]C(=O)[NX3]` | haloacetamide SN2 alkylator |
| `epoxide` | `[OX2r3]1CC1` | strained-ring alkylator |
| `michael_enone` | `[CH2]=[CH]C(=O)[CX4,CX3;!$([CX3]=O)]` | enone Michael acceptor (ketone, not amide/ester) |
| `isocyanate` | `[NX2]=[CX2]=[OX1]` | isocyanate |
| `cyanamide` | `[NX3][CX2]#[NX1]` | covalent kinase inhibitor scaffold |

Known false-negative: erlotinib has a terminal alkyne on arene but NOT a propiolamide (no carbonyl), correctly returns non-covalent.

**Gap fixed (June 2026):** cmpd_021 (`C=CC(=O)Oc1...`) has an acrylate ESTER warhead, not an acrylamide. The `acrylamide` SMARTS `[NX3]C(=O)C=C` requires an amide nitrogen and therefore missed it. Added `acrylate_ester` key. Relabeling script: `scripts/relabel_covalent.py` (re-applies labels without re-running docking).

**In the top-50 backbone candidates: 22/50 are covalent-flagged** (21 acrylamide + 1 acrylate ester). Docking scores for these compounds should be interpreted with extra caution.

## Phase B2 ranked-library docking (first pass, EXPLORATORY)

Run: `PYTHONPATH=. .venv/Scripts/python.exe scripts/dock_library.py` (~50-100 min).
Results: `models/qsar/library_docking_results.json`.

**Candidate selection** (`select_top_candidates` in `scripts/dock_library.py`):
- Load backbone model (`models/qsar/general/`), predict pIC50 for all 1253 molecules.
- Deduplicate by canonical SMILES (keep highest prediction per SMILES): 1253 rows → ~900 unique SMILES.
- Take top 50 by `pred_pic50`.
- Predicted pIC50 range: ~8.4–9.3.

**Docking**: Vina 1.2.7, exhaustiveness=8, seed=42, 9 poses, shared box (center −51.654/−1.266/−21.945, size 22.5 Å³).

**Covalent flag**: applied to all compounds before docking; tagged `docking_confidence = "low_confidence"` in results.

**Selectivity delta**: `selectivity_delta = L858R_score - WT_score` (kcal/mol). Negative = L858R-selective (desired). Same sign convention as the B2 sanity check.

**Results** (seed=42, exhaustiveness=8, 62-min wall-clock):

| Summary stat | Value |
|---|---|
| n_candidates | 50 |
| n_ok | 49 |
| n_partial | 0 |
| n_failed | 1 (cmpd_041 — 123-line PDBQT macrocycle, timed out 300 s both pockets) |
| n_covalent_flagged | 21 |
| L858R-selective (delta < 0) | 30 / 49 |
| WT-selective (delta > 0) | 18 / 49 |
| tied | 1 |

Top 5 most L858R-selective compounds (sorted by delta ascending, most selective first):

| Rank | CID | delta (kcal/mol) | L858R | WT | pred pIC50 | confidence | warheads |
|---|---|---|---|---|---|---|---|
| 1 | cmpd_024 | **−0.953** | −7.439 | −6.486 | 8.684 | standard | — |
| 2 | cmpd_008 | −0.856 | −8.865 | −8.009 | 8.959 | **low_confidence** | acrylamide |
| 3 | cmpd_033 | −0.723 | −8.829 | −8.106 | 8.589 | **low_confidence** | acrylamide |
| 4 | cmpd_015 | −0.634 | −7.552 | −6.918 | 8.816 | standard | — |
| 5 | cmpd_012 | −0.591 | −7.895 | −7.304 | 8.856 | standard | — |

Top non-covalent (standard-confidence) L858R-selective compounds:
- **cmpd_024** (delta −0.953): sulfonamide-containing EGFR inhibitor analog, strongest L858R preference in set
- **cmpd_015** (delta −0.634): dimethylamino-substituted quinazoline analog
- **cmpd_012** (delta −0.591): histidine-tethered EGFR inhibitor
- **cmpd_037** (delta −0.419): fused indazole-quinazoline scaffold
- **cmpd_010** (delta −0.392): imidazole-tethered EGFR inhibitor

Direction consistency (61% L858R-selective) is reasonable for rigid Vina on two different crystal structures — the sanity-check inhibitors showed 100% but those are well-validated co-crystal poses.

**LIMITATIONS**: same as B2 sanity check (rigid receptor, dual-pocket comparability, delta magnitude). Additionally: all 50 candidates are in-sample for the backbone model — their predicted pIC50 values are not held-out estimates.

## Phase B2 docking noise quantification (EXPLORATORY)

Run: `PYTHONPATH=. .venv/Scripts/python.exe scripts/eval_docking_noise.py` (~60-90 min).
Results: `models/qsar/docking_noise_results.json`.

**Motivation:** The first-pass selectivity deltas are point estimates of a difference between two noisy Vina scores. Seed-to-seed variability quantifies whether the observed delta is distinguishable from scoring noise.

**Method:**
- Take top 15 compounds by initial selectivity delta (range −0.953 to −0.332 kcal/mol).
- For each compound × each pocket (L858R and WT): run Vina 5 times with seeds [42, 7, 13, 99, 123].
- Propagate: `std_delta = sqrt(std_L858R² + std_WT²)` (quadrature error propagation).
- Confident call criterion: `|delta| > 1.5 × std_delta`. Otherwise: "ambiguous / within noise".
- Covalent compounds remain `low_confidence_covalent` regardless of noise analysis.
- Total new Vina runs: 15 × 2 × 5 = 150. Resume-capable (existing output PDBQTs are reused).

**Classification scheme:**
| Call | Criterion |
|---|---|
| `L858R_selective` | `delta < 0` AND `|delta| > 1.5 × std_delta` AND non-covalent |
| `WT_selective` | `delta > 0` AND `|delta| > 1.5 × std_delta` AND non-covalent |
| `ambiguous` | `|delta| <= 1.5 × std_delta` (within noise band) AND non-covalent |
| `low_confidence_covalent` | `docking_confidence = "low_confidence"` (warhead detected), any delta |

**Results** (seeds [42, 7, 13, 99, 123], exhaustiveness=8):

| Rank | CID | delta±std | L858R(mean±std) | WT(mean±std) | call |
|---|---|---|---|---|---|
| 1 | cmpd_024 | **−0.813±0.277** | −7.391±0.133 | −6.577±0.243 | **L858R_selective** |
| 2 | cmpd_008 | −0.663±0.393 | −8.886±0.060 | −8.223±0.389 | low_confidence_covalent |
| 3 | cmpd_033 | −0.650±0.077 | −8.784±0.043 | −8.134±0.064 | low_confidence_covalent |
| 4 | cmpd_012 | **−0.546±0.196** | −7.956±0.073 | −7.410±0.182 | **L858R_selective** |
| 5 | cmpd_021 | −0.522±0.044 | −7.632±0.041 | −7.110±0.016 | low_confidence_covalent *(acrylate ester; relabeled)* |
| 6 | cmpd_011 | −0.513±0.044 | −8.596±0.025 | −8.083±0.036 | low_confidence_covalent |
| 7 | cmpd_015 | **−0.452±0.139** | −7.428±0.073 | −6.975±0.118 | **L858R_selective** |
| 8 | cmpd_048 | **−0.430±0.198** | −7.548±0.123 | −7.117±0.155 | **L858R_selective** |
| 9 | cmpd_038 | −0.383±0.290 | −7.190±0.075 | −6.807±0.280 | **ambiguous** |
| 10 | cmpd_037 | **−0.357±0.042** | −8.262±0.027 | −7.905±0.032 | **L858R_selective** |
| 11 | cmpd_002 | **−0.342±0.048** | −7.326±0.048 | −6.984±0.009 | **L858R_selective** |
| 12 | cmpd_014 | −0.306±0.083 | −7.456±0.015 | −7.150±0.081 | low_confidence_covalent |
| 13 | cmpd_030 | −0.279±0.060 | −7.834±0.053 | −7.555±0.028 | low_confidence_covalent |
| 14 | cmpd_022 | −0.198±0.130 | −8.379±0.093 | −8.181±0.090 | low_confidence_covalent |
| 15 | cmpd_010 | −0.093±0.321 | −6.949±0.296 | −6.856±0.125 | **ambiguous** |

Summary: 6 confident L858R-selective | 0 confident WT-selective | 2 ambiguous | 7 low-confidence covalent (6 acrylamide + 1 acrylate ester [cmpd_021]; relabeled by scripts/relabel_covalent.py).

**Key findings:**
- **7/9 non-covalent compounds clear the noise threshold** (|delta| > 1.5×std_delta). The L858R pocket generally has lower seed noise (std ≈ 0.03–0.13) than the WT pocket (std ≈ 0.01–0.28).
- **cmpd_010 is an artifact**: initial delta −0.392 becomes mean delta −0.093 over 5 seeds (L858R score ranged −6.577 to −7.195). The single-seed estimate was a lucky outlier; this compound has no reliable L858R preference.
- **cmpd_038** is also ambiguous: WT pocket is highly variable (std 0.280), absorbing the signal.
- **cmpd_024 retains top rank** with delta −0.813±0.277 despite high WT-pocket noise — the large delta comfortably exceeds the 1.5×std_delta=0.416 threshold.
- **Confidence-filtered shortlist** (standard-confidence, L858R_selective, sorted by delta): cmpd_024 (−0.813), cmpd_012 (−0.546), cmpd_015 (−0.452), cmpd_048 (−0.430), cmpd_037 (−0.357), cmpd_002 (−0.342). cmpd_021 (−0.522) reclassified to `low_confidence_covalent` (acrylate ester warhead) and removed from this list.

**LIMITATIONS**: same as Phase B2 first pass. Additionally: Vina seed noise (±0.1–0.3 kcal/mol per pocket typical) means the 1.5×std threshold may filter most or all small deltas; compounds that survive this filter are more credible but still subject to rigid-receptor limitations. Covalent compounds that appear L858R-selective are excluded from confident calls because the docking score reflects non-covalent geometry only.

## ADMET filtering (APPROXIMATE)

Run: `PYTHONPATH=. .venv/Scripts/python.exe scripts/eval_admet.py` (~5 s).
Results: `models/qsar/admet_results.json`.

Module: `src/admet/filters.py`. Reuses `check_lipinski` / `check_veber` from `src/features/descriptors.py`.

**Filters** (all approximate — not a substitute for experimental profiling):

| Filter | Rule | Flag condition |
|---|---|---|
| Lipinski Ro5 | MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10 | > 1 violation |
| Veber | RotBonds ≤ 10, TPSA ≤ 140 | either violated |
| PAINS | RDKit FilterCatalog (all subsets) | any match |
| Brenk | RDKit FilterCatalog (metabolic/reactive alerts) | any match |
| QED | RDKit QED.qed() [0, 1] | < 0.25 |
| SA score | rdkit.Contrib.SA_Score [1, 10] | > 6.0 (if available) |
| Range checks | MW 100–600, LogP −2 to 5.5, TPSA 20–140 | **informational only** |

`admet_status` = "pass" or "flag"; **molecules are never hard-dropped**.

**Shortlist results** (6 non-covalent L858R-selective compounds; cmpd_021 moved to low_confidence_covalent after acrylate_ester SMARTS fix):

| CID | MW | LogP | QED | SA | Brenk | PAINS | status |
|---|---|---|---|---|---|---|---|
| cmpd_015 | 343 | 3.6 | 0.787 | 2.4 | N | N | **pass** |
| cmpd_012 | 409 | 3.9 | 0.447 | 2.8 | N | N | **pass** |
| cmpd_037 | 353 | 4.0 | 0.591 | 2.4 | N | N | **pass** |
| cmpd_002 | 359 | 4.2 | 0.757 | 2.0 | N | N | **pass** |
| cmpd_024 | 423 | 2.8 | 0.518 | 2.5 | Y (Sulfonic_acid_2) | N | flag |
| cmpd_048 | 359 | 3.8 | 0.489 | 2.5 | Y (Aliphatic_long_chain) | N | flag |

cmpd_021 (delta −0.522, acrylate ester): now `low_confidence_covalent` — excluded from shortlist. Its `call` in `docking_noise_results.json` is `low_confidence_covalent`. Do not use its docking delta without covalent-aware scoring.

**Top-50 library summary**: 16/50 pass (32%). Median QED 0.489. Most common Brenk: Michael_acceptor_1 (18×), Aliphatic_long_chain (7×), diketo_group (6×), halogenated_ring_2 (5×).

**Key findings:**
- **Clean shortlist** (confirmed, unchanged): cmpd_015, cmpd_012, cmpd_037, cmpd_002 pass all filters. These four are the highest-priority candidates (confirmed L858R-selective by docking noise analysis AND clean ADMET profile).
- **cmpd_024** (strongest docking hit, delta −0.813): Brenk `Sulfonic_acid_2` — the sulfonamide scaffold carries a free sulfonic acid (`-SO3H`). Concerns: low membrane permeability. Still worth considering; bioisostere replacement of -SO3H could rescue it.
- **Michael_acceptor_1 (18×)** dominates the Brenk top-50 alerts. This is consistent with the 22 covalent-flagged compounds identified by the warhead detector; Brenk is independently confirming those as structural liabilities.
- **SA scores** are all 2.0–2.8 — synthetically accessible by conventional chemistry.
- **No PAINS alerts** in the shortlist. 2 PAINS hits in the full top-50 (anil_di_alk_A).

## De novo generator — two-stage GRU (Phase 21, EXPLORATORY)

Pipeline: **base pretrain (drug-like grammar) → EGFR fine-tune → temperature-tuned sampling**.
The single-corpus generator only reached 56.3% validity because 1347 molecules is too
few to learn SMILES grammar. A broad drug-like base fixes that; fine-tuning recovers EGFR chemistry.

```bash
# 1. build the drug-like corpus (once; ~84 MB download, cached)
PYTHONPATH=. .venv/Scripts/python.exe scripts/download_drug_corpus.py        # -> data/interim/drug_like_corpus.smi (150k)
# 2. pretrain the base on the drug-like corpus (CPU-bound; see note)
PYTHONPATH=. .venv/Scripts/python.exe scripts/pretrain_generator.py          # -> models/generator/pretrained_base_gru.pt + tokenizer.json
# 3. fine-tune on the 1347 EGFR/ErbB2 actives (~20 min CPU)
PYTHONPATH=. .venv/Scripts/python.exe scripts/finetune_generator.py          # -> models/generator/egfr_finetuned_gru.pt
# 4. (optional) pick sampling temperature
PYTHONPATH=. .venv/Scripts/python.exe scripts/sample_temperature_sweep.py    # -> models/generator/temperature_sweep.json
# 5. screen generated molecules through backbone + covalent + ADMET
PYTHONPATH=. .venv/Scripts/python.exe scripts/screen_generated.py
```

**Architecture** (base and fine-tune share it): char-level GRU, embed 128, hidden 512 × 3 layers,
dropout 0.1, max_len 120, teacher forcing + CE (PAD-ignored), Adam, ReduceLROnPlateau.
Tokenizer is regex-based (`[brackets]` > `Br/Cl` > single-char) and fit on the **union** of the
drug-like + EGFR corpora so the same vocab (43 tokens) covers both stages — required for warm-start.

**Corpora**:
- Base: MOSES `dataset_v1.csv` (1.94M), MW/LogP-filtered (150–500 / −1–5), deduped → 150k drug-like
  SMILES (`data/interim/drug_like_corpus.smi`), **capped to 80k** for CPU training feasibility.
- Fine-tune: `egfr_cleaned.csv` + `erbb2_cleaned.csv` → 1347 unique canonical SMILES.

**Training runs (seed=42, CPU)**:
- Base: best val_loss **0.586 @ epoch 6** (`pretrained_base_gru.pt`). NOTE: a full 8-epoch run on 80k
  is ~3.5 h/epoch and **hung after epoch 6** on this 6-thread machine; epoch 6 is a sufficient grammar
  base. **Do not redo base pretrain on CPU** — fine-tune from the existing checkpoint.
- Fine-tune: warm-started from the base, best val_loss **0.366 @ epoch 22** (`egfr_finetuned_gru.pt`).

**Temperature sweep** (1000 samples each, novelty/scaffold-novelty vs the EGFR corpus):

| temp | validity | uniqueness | novelty | scaffold div (n) | scaffold novelty |
|---|---|---|---|---|---|
| 1.0 | 81.0% | 87.3% | 62.7% | 0.523 (370) | 49.7% |
| 0.9 | 84.2% | 80.5% | 58.6% | 0.516 (350) | 44.0% |
| **0.8** | **92.3%** | 73.8% | 54.8% | 0.501 (341) | 37.5% |
| 0.7 | 94.5% | 63.0% | 50.8% | 0.472 (281) | 31.7% |

**Operating point = temperature 0.8** (`generator.sample_temperature` in config; default in
`screen_generated.py`): clears the 90% validity goal (**92.3%**, up from 56.3% single-corpus) while
keeping strong exploration — 341 distinct Bemis-Murcko scaffolds over 681 unique molecules, 37.5% of
scaffolds absent from the EGFR corpus. Temp 0.7 buys 2 more validity points but collapses uniqueness to
63% (mode collapse), so 0.8 is the pick. Scaffold-diversity metric: `sampler.scaffold_stats()` /
`evaluate_metrics(..., compute_scaffolds=True)`.

**Checkpoints** (`models/generator/`): `pretrained_base_gru.pt` (drug-like base),
`egfr_finetuned_gru.pt` (EGFR fine-tuned, the production checkpoint), `tokenizer.json` (shared union vocab).
Results JSON: `pretrain_base_results.json`, `finetune_results.json`, `temperature_sweep.json`.

**Screening pipeline** (`scripts/screen_generated.py`, default temp 0.8): generates → RDKit validity →
uniqueness dedup → novelty filter → backbone pIC50 (Model 1, 2048-bit Morgan ECFP4 + 11 descriptors) +
covalent flag + ADMET + **applicability domain** → ranked table. No docking at this stage.

**Applicability domain** (`src/scoring/applicability_domain.py`): max Tanimoto similarity to the EGFR/ErbB2
training set (Morgan ECFP4, 2048 bits). Three bands: `in_domain` (sim ≥ 0.50, cf 1.0), `borderline`
(0.30–0.50, cf 0.75), `out_of_domain` (<0.30, cf 0.50). Thresholds and confidence factors from
`model_config.yaml > applicability_domain`. Public API: `ApplicabilityDomain.from_config()`,
`.fit(smiles_list)`, `.predict(smiles)`, `.predict_batch(smiles_list)`.

**Refreshed screen results** (temp 0.8, `egfr_finetuned_gru.pt`, 1000 samples, June 2026):
`models/generator/screen_results.json`

| Metric | Value |
|---|---|
| Validity | 91.0% |
| Uniqueness | 71.5% |
| Novelty | 52.8% |
| Scaffold diversity | 0.484 (315 scaffolds) |
| Screened | 344 |
| ADMET pass | 99 (29%) |
| Covalent-flagged | 102 (30%) |
| in_domain | **310 (90%)** |
| borderline | 29 (8%) |
| out_of_domain | 5 (1%) |

90% of generated molecules are in-domain (max Tanimoto ≥ 0.50 to training set) — expected for an EGFR-fine-tuned
model. The 5 out-of-domain molecules are structurally novel but predictions should be discounted by cf=0.50.
Top non-covalent ADMET-pass hit: pred_pIC50 8.436, in_domain (sim 0.759).

**Limitations**:
- Base was trained on 80k (CPU cap) for only 6 epochs (hung before epoch 8). Validity ceiling at temp 1.0
  is ~81%; temperature 0.8 is what lifts it past 90%. A GPU run on the full 150k for more epochs would
  raise temp-1.0 validity and let you sample hotter for more diversity.
- Generated SMILES are scored by the general backbone (in-sample for activity) — **doubly exploratory**.
- RL fine-tuning (reward shaping toward L858R selectivity) is the next generation phase.

**Modules** (`src/generation/`):
- `tokenizer.py` — `tokenize()`, `SMILESTokenizer` (fit/encode/decode/save/load)
- `model.py` — `GRUGenerator(nn.Module)`
- `trainer.py` — `SMILESDataset`, `train_epoch`, `validate`, `train_model` (`init_ckpt`/`ckpt_name` warm-start), `finetune_model`
- `sampler.py` — `sample_smiles`, `evaluate_metrics`, `scaffold_stats`, `load_checkpoint`

## Phase 22 — REINVENT RL fine-tuning (EXPLORATORY)

Run: `PYTHONPATH=. .venv/Scripts/python.exe scripts/train_rl_generator.py`
Results: `models/generator/rl_results.json`. Step log: `logs/rl_training.log` (CSV).

**Algorithm** (Olivecrona et al. 2017):
1. Generate batch of SMILES from agent (no grad)
2. Score each SMILES with multi-objective reward
3. Compute per-token mean NLL under agent (teacher forcing, with grad)
4. Compute per-token mean NLL under frozen prior (same initial weights, no grad)
5. `AugNLL = NLL_prior − σ × Score`
6. `Loss = mean((NLL_agent − AugNLL)²)`; backprop through agent only

**σ (sigma)** balances reward influence vs prior regularisation. Lower σ = tighter stay near prior, less exploitation; higher σ = more drift, more reward shaping. Two runs were done: **Run 1 at σ=0.5 with no diversity filter (REWARD_HACKING)** and **Run 2 at σ=0.25 with a scaffold-memory filter (INCONCLUSIVE, no collapse)** — see Results.

**Scaffold-memory diversity filter** (`ScaffoldMemory` in `src/generation/rl_trainer.py`; Blaschke et al. 2020): maintains a running count of high-reward Bemis-Murcko scaffolds across the whole run. Once a scaffold's bucket fills (≥ `bucket_size` molecules scoring above `min_score`), every later molecule sharing that scaffold has its reward overwritten with `penalty` (0.0). Unlike the flat novelty_bonus (which only checks the *training* set), this grows from the agent's **own** output and is what actually prevents mode collapse. Config: `generator.rl.diversity_filter` (enabled, bucket_size=25, min_score=0.30, penalty=0.0).

**Reward components** (from `config/model_config.yaml > generator.rl.reward`):

| Component | Default | Notes |
|---|---|---|
| `activity_weight × sigmoid(pIC50 − 7.0) × AD_cf` | 0.50 × [0,1] × cf | AD confidence_factor is the primary anti-hacking guard |
| `qed_weight × QED` | 0.20 × [0,1] | |
| `admet_bonus` if ADMET pass | +0.20 flat | |
| `novelty_bonus` if not in train set | +0.10 flat | |
| `invalid_penalty` (immediate return) | −1.00 | |
| `covalent_penalty` if warhead | −0.30 | |
| `out_of_domain_penalty` | −0.30 | additive guard; activity also discounted by cf=0.50 |
| `borderline_penalty` | −0.10 | cf=0.75 |
| `range_penalty` (MW<100/>600 or LogP<−2/>5.5) | −0.10 | |

**Anti-reward-hacking double guard**: OOD molecules pay the additive penalty AND their activity reward is multiplied by cf=0.50. Two independent mechanisms prevent the agent from gaming the backbone outside the training distribution.

**Checkpoint**: `models/generator/rl_finetuned_gru.pt` — warm-started from `egfr_finetuned_gru.pt`. **Not used for downstream screening** (see verdict). The production generator remains `egfr_finetuned_gru.pt`.

**Modules**: `src/generation/reward.py` (`compute_reward`, `MoleculeReward`, `_sigmoid`, `_DEFAULTS`), `src/generation/rl_trainer.py` (`compute_nll_batch`, `evaluate_generator`, `compare_pre_post`, `REINVENTTrainer`, `ScaffoldMemory`). 47 unit tests in `tests/test_rl_reward.py` (reward components, NLL batch, REINVENT formula, scaffold-memory bucketing, verdict logic — no full training).

**Verdict criteria** (applied in `compare_pre_post`):
- `REWARD_HACKING`: scaffold_diversity drops >0.10 **OR** uniqueness falls below 0.60 **OR** in_domain drops >10 pp (any diversity/domain collapse).
- `SUCCESS`: mean_pic50 improves >0.10 **AND** ADMET improves **AND** uniqueness ≥0.60 **AND** diversity drop <0.10.
- `PARTIAL`: pIC50 improves but ADMET collapses >15 pp (or ADMET fails to improve), no diversity collapse.
- `INCONCLUSIVE`: mean_pic50 change <0.10 — RL did not move the activity needle.

### Run 1 — σ=0.5, no diversity filter (REWARD_HACKING)

n_steps=100, batch=64, σ=0.5, seed=42, eval at temp=0.8 on 512 samples, 634s.

| Metric | Pre-RL | Post-RL | Delta |
|---|---|---|---|
| Validity | 93.0% | 99.4% | +6.4pp |
| Uniqueness | **80.0%** | **8.6%** | **−71.4pp** |
| Scaffold diversity | **0.580** | **0.318** | **−0.262** |
| Mean pred pIC50 | 6.712 | 7.592 | +0.880 |
| ADMET pass rate | 31.2% | 80.6% | +49.4pp |
| In-domain rate | 88.4% | 93.5% | +5.1pp |

**REWARD_HACKING.** The agent collapsed onto ~14 Bemis-Murcko scaffolds. The pIC50 (+0.88) and ADMET (+49pp) gains are artifacts of mode collapse — the agent re-generated the same high-scoring quinazoline/indazole scaffolds rather than exploring. The OOD guard *held* (in-domain rose), so the hack was diversity collapse onto in-domain scaffolds, not backbone gaming. σ=0.5 overpowered the prior; the flat novelty_bonus (+0.10) was too weak to stop repetition.

### Run 2 — σ=0.25 + scaffold-memory filter (INCONCLUSIVE, no collapse)

n_steps=50, batch=64, σ=0.25, diversity_filter on (bucket=25, min_score=0.30), seed=42, eval at temp=0.8 on 512 samples, 330s. During training: **494 distinct scaffolds seen, only 12 saturated, 412 molecules penalised** (vs Run 1's collapse to 14 scaffolds).

| Metric | Pre-RL | Post-RL | Delta | |
|---|---|---|---|---|
| Validity | 93.0% | 96.1% | +3.1pp | ^ |
| Uniqueness | 80.0% | 67.7% | −12.4pp | above 60% floor |
| Scaffold diversity | 0.580 | 0.568 | **−0.012** | held (≪0.10) |
| Mean pred pIC50 | 6.712 | 6.716 | +0.004 | flat |
| ADMET pass rate | 31.2% | 48.3% | +17.1pp | ^ |
| In-domain rate | 88.4% | 87.1% | −1.3pp | held |

**INCONCLUSIVE — but the diversity guard worked.** The scaffold-memory filter + lower σ prevented mode collapse: diversity held (−0.012), uniqueness stayed above the 60% floor, in-domain held. ADMET improved +17pp. But once σ was low enough to stop hacking, the activity signal was too weak to move pIC50 (+0.004 < 0.10), so this is not a deployable activity improvement. **Net read: RL can be made non-hacking here, but at σ=0.25/50 steps it does not improve activity. The two are in tension at this corpus size.**

**Production decision**: keep `egfr_finetuned_gru.pt` as the generator. Run 2 did not collapse but did not improve activity either; Run 1 improved activity only by hacking. **Do NOT use `rl_finetuned_gru.pt`.** RL is documented here as a controlled experiment, not a shipped improvement.

**If pursuing RL further**: the activity/diversity tension suggests (a) a larger fine-tune corpus so the prior covers more high-activity chemistry, (b) a curriculum that raises σ gradually while the diversity filter holds, or (c) replacing the in-sample backbone reward with a held-out or docking-based signal so activity gains are real rather than backbone-gamed.

## Phase 23 — Final integrated ranking (capstone, EXPLORATORY)

Module: `src/scoring/ranking.py`. Scripts: `scripts/rank_candidates.py` (ranking + export),
`scripts/dock_generated_candidates.py` (post-hoc docking of generated hits).
Export: `data/generated/final_ranked_candidates.csv`. 35 tests in `tests/test_ranking.py`.

**v2 composite formula** (`config/model_config.yaml > ranking.weights_v2`):

```
bioactivity_score = 0.30*activity + 0.30*docking_selectivity
                  + 0.20*docking_affinity + 0.20*ADMET     (each min-max normalised)
final_score       = bioactivity_score * AD_confidence_factor
```

- **activity** = backbone pred_pIC50. **docking_selectivity** = −(L858R−WT) Vina delta (so more L858R-selective → higher). **docking_affinity** = −L858R Vina score (stronger binding → higher). **ADMET** = QED.
- Each component is **min-max normalised across the candidate set** before weighting, so weights act on comparable [0,1] scales; higher is better on every axis.
- **The only multiplier on the score is the AD confidence_factor** (1.0 / 0.75 / 0.50). Covalent warhead and within-noise selectivity are surfaced as **WARNINGS, never silent score penalties** — a covalent and a non-covalent compound with identical evidence get identical scores, and the liability is shown beside the score. Warning sources: `OUT_OF_DOMAIN`/`BORDERLINE_DOMAIN` (also explains the cf), `COVALENT[warheads]`, `SELECTIVITY_WITHIN_NOISE` (|delta| ≤ 1.5×std from the seed-noise study; only the noise-studied top-15 can trigger this).

**Pipeline**: `rank_candidates.py` ranks the docked known library always; folds in generated candidates if `models/generator/generated_docking_results.json` exists (run `--library-only` for the no-new-compute library deliverable). AD is fit on the 1347 EGFR/ErbB2 training SMILES and applied to every candidate. `dock_generated_candidates.py` samples 2000 from `egfr_finetuned_gru.pt` (temp 0.8), screens (valid/unique/novel + ADMET pass + in_domain, non-covalent preferred), selects the top 20 by pred_pIC50, and docks each into both pockets (same box/engine as the library: Vina 1.2.7, exhaustiveness=8, seed=42).

**Final ranking** (68 candidates = 49 known library + 19 generated; 1 generated ligand failed prep):

| Rank | CID | Source | final | act | sel | aff | qed | warnings |
|---|---|---|---|---|---|---|---|---|
| 1 | cmpd_015 | known | 0.730 | 0.71 | 0.83 | 0.35 | 1.00 | — |
| 2 | cmpd_002 | known | 0.718 | 0.90 | 0.67 | 0.28 | 0.95 | — |
| 3 | cmpd_008 | known | 0.697 | 0.79 | 0.95 | 0.72 | 0.16 | COVALENT[acrylamide] |
| 4 | cmpd_024 | known | 0.669 | 0.63 | 1.00 | 0.31 | 0.58 | — |
| 5 | cmpd_011 | known | 0.647 | 0.73 | 0.76 | 0.64 | 0.35 | COVALENT[acrylamide] |
| … | | | | | | | | |
| 21 | **gen_005** | **generated** | **0.597** | 0.29 | 0.72 | 0.48 | 0.98 | — |

- **Top non-covalent known hits**: cmpd_015, cmpd_002, cmpd_024, cmpd_012 (#9), cmpd_037 (#12) — the same clean-ADMET shortlist the docking-noise + ADMET phases converged on. The v2 score reproduces that shortlist from independent evidence fusion.
- **cmpd_008** (#3) is the strongest by docking (sel 0.95, aff 0.72) but carries the covalent warning and has poor QED (0.16); the score keeps it high (no silent penalty) and the warning flags it for human judgement.
- **Generated placement**: best generated **gen_005 at #21** (`COc1cc2ncnc(Nc3cccc(Cl)c3F)c2cc1OC`, a methoxy-quinazoline, activity 8.10, L858R delta −0.43, QED 0.78, in-domain, no warnings) — lands immediately after the known top-20. Median generated rank 54/68; 3 generated in the top half, 0 in the top-20. Generated docking was strongly directional: **17/19 L858R-selective**.
- **Why known dominate the top**: the library candidates were selected as the highest backbone-predicted actives (in-sample), so they hold the activity axis (cmpd_001 act_norm 1.00). The generated molecules are genuinely novel, so their activity is lower; gen_005 is competitive purely on selectivity + QED. Honest read: **de novo generation produces credible, drug-like, L858R-selective candidates that rank alongside — but not above — the best known actives on an activity-weighted score.**
- All 68 candidates are `in_domain` (both sets are in-domain by construction: known from training, generated pre-filtered to in_domain), so the OOD/borderline cf branch and warning are exercised by tests but do not trigger on this particular set. Two noise-studied library compounds (cmpd_010, cmpd_038) carry `SELECTIVITY_WITHIN_NOISE`.

**LIMITATIONS**: every component is exploratory — backbone pIC50 is in-sample for the known library, docking is rigid-receptor Vina across two crystal structures, ADMET (QED) is approximate. The score orders candidates by aggregated evidence; it is not a calibrated success probability. Generated candidates add an extra layer (novel chemistry scored by an in-sample backbone).

## Phase 24 — FastAPI fast-screen backend (serves precomputed models only)

Package: `src/api/` (`main.py`, `routes.py`, `schemas.py`, `services.py`).
Deps: `requirements/api.txt` (fastapi, uvicorn, httpx). 28 tests in `tests/test_api.py`.

Run locally:
```bash
PYTHONPATH=. .venv/Scripts/python.exe -m uvicorn src.api.main:app --reload   # docs at /docs
```

**Hard constraint**: serves **precomputed artifacts only** — no training, no docking, no ESM-2 at request time. `ModelRegistry.load()` (in `services.py`) loads everything ONCE at startup (FastAPI `lifespan`) into `app.state.registry`: backbone activity (Model 1, RandomForest), WT-proxy (Model 2, XGBoost), the ADMET filter, the covalent detector, and the applicability domain (fit on the 1347 EGFR/ErbB2 training SMILES). Each request is forward passes over saved sklearn/XGBoost models + RDKit descriptor computation.

**Endpoints**:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness; `status` = `ok` only when all artifacts loaded, else `degraded` |
| POST | `/predict` | single SMILES fast screen; **422** structured error on invalid SMILES |
| POST | `/batch_predict` | up to 512 SMILES; invalid ones return as `valid:false` rows (batch never fails) |
| GET | `/model-info` | model versions/algorithms, score definitions, exploratory caveats, docking note |

**`/predict` response** (`PredictResponse`): `pic50_mutant` (backbone), `pic50_wt` (WT-proxy), `selectivity_proxy` (= pic50_mutant − pic50_wt; positive = mutant-selective), `covalent` + `warheads`, `admet` (status/qed/sa/alerts), `applicability_domain` (band + confidence_factor), `warnings[]`, and `docking_selectivity_available` (**always false**).

**Docking is explicitly NOT computed at request time** — structure-based (Vina) L858R-vs-WT selectivity needs the offline pipeline. This is stated in `GET /model-info` (`docking_selectivity` field) and surfaced on every prediction via `docking_selectivity_available: false`. The `selectivity_proxy` is the **fast ML proxy only** (labeled exploratory; QSAR selectivity was not significant at n=9 — see Model 4).

**Errors**: structured `{error, detail}` bodies for all 4xx/5xx (handlers in `main.py`). Invalid SMILES → 422 `invalid_smiles`; bad request body → 422 `validation_error`; registry absent → 503 `models_unavailable`. No HTML forms (JSON only).

**Testing**: `create_app(registry=...)` accepts an injected registry, so endpoint tests run against an artifact-free `_MockRegistry` (fast, `@unit`); 4 `@integration` tests load the real models and verify live predictions. Reuses `build_warnings` from `src/scoring/ranking.py` (covalent/OOD warnings; within-noise is N/A since no docking runs).

## Phase 25 — Streamlit dashboard (over the API, with local fallback)

Package: `src/dashboard/` (`app.py`, `data_loaders.py`, `api_client.py`).
Deps: `requirements/dashboard.txt` (streamlit, altair, requests). 26 tests in `tests/test_dashboard.py`.

Run locally:
```bash
PYTHONPATH=. .venv/Scripts/python.exe -m streamlit run src/dashboard/app.py   # http://localhost:8501
```

**Scoring backend resolution** (`api_client.py`): the two interactive pages prefer the FastAPI service — `api_available()` does a 0.5 s `GET /health`; if it returns `status:ok` the dashboard calls `/predict` and `/batch_predict`. Otherwise it **falls back to scoring locally** through the same `ModelRegistry` the API uses (loaded once via `st.cache_resource`). The sidebar shows which mode is live (`🟢 API` vs `🟡 Local registry`). Identical fast-screen dict either way.

**Six pages** (`app.py`):

| Page | Source | Live? |
|---|---|---|
| Single molecule | `/predict` or local registry | needs API up **or** local artifacts |
| Batch screening | `/batch_predict` or local loop; upload `.smi/.txt/.csv` or paste | same |
| Final ranking | `data/generated/final_ranked_candidates.csv` | yes (static artifact) |
| Model performance | QSAR metadata + `SEED_STABILITY` const + `fingerprint_ablation_results.json` + Model 3/4 verdicts | yes |
| Docking results | `sanity_check_docking.json` + `docking_noise_results.json` (±1.5×std error bars) + `generated_docking_results.json` | yes |
| Limitations | hard-coded honest findings | yes |

**Data loaders** (`data_loaders.py`) are pure file→DataFrame/dict functions, each tolerant of a missing artifact (returns `None`/empty so the page shows an info message instead of crashing). These are what the tests cover. The 5-seed scaffold-split stability numbers are not persisted as JSON anywhere, so they live as the `SEED_STABILITY` constant (sourced from `eval_seed_stability.py` / CLAUDE.md) — keep it in sync if those are re-run.

**Visuals**: Altair bar chart of `final_score` coloured by source (known vs generated) on the ranking page; R² mean±std error-bar chart and a docking selectivity dot-plot with ±1.5×std rules. Every page carries the EXPLORATORY banner; the Limitations page states the negative results plainly (L858R ML ≯ backbone at n=22, selectivity not modelable at n=9, RL hacks-or-stalls, rigid-receptor docking, no experimental validation).

**Testing**: `data_loaders` + `ranking_placement` + `api_client` fallback logic are `@unit` (offline; `api_available` on a dead port returns False fast, local fallback is a mock registry). `TestAppRenders` is `@integration` — it runs the real `app.py` via Streamlit's `AppTest`, loads the real registry, and asserts the default page + all four data pages render with no exception.

## Phase 26 — Docker packaging

Run locally without Docker: see the commands in `## Commands` above.
Run with Docker:
```bash
docker compose up --build   # http://localhost:8000  (API) + http://localhost:8501 (dashboard)
docker compose down
```

### Images

| Service | Dockerfile | Base | Key serving deps |
|---|---|---|---|
| `api` | `docker/api/Dockerfile` | `python:3.12-slim` | serving.txt + api.txt (fastapi, uvicorn, httpx) |
| `dashboard` | `docker/dashboard/Dockerfile` | `python:3.12-slim` | serving.txt + dashboard.txt (streamlit, altair, requests) |

**`requirements/serving.txt`** — shared ML/chem core: numpy, pandas, scipy, joblib, scikit-learn, xgboost, rdkit, pydantic, pyyaml, python-dotenv. Explicitly excludes: torch, lightgbm, optuna, mlflow, umap-learn, matplotlib, seaborn, tqdm, Vina, GNINA (all offline-only).

**XGBoost GPU dep (nvidia-nccl-cu12)**: XGBoost 3.x pulls in `nvidia-nccl-cu12` (303 MB) as a required dep on Linux, even for CPU-only use. The Dockerfiles uninstall it in the same RUN layer as the install (`pip install ... && pip uninstall -y nvidia-nccl-cu12 || true`) so the binary is never committed to a layer. Result: API image = **1.19 GB**, dashboard = **1.53 GB** (would be ~300 MB larger each without the fix). XGBoost CPU inference continues to work; NCCL is only used for distributed/GPU collective training.

**Startup warnings (non-blocking)**:
- XGBoost version mismatch: "If you are loading a serialized model generated by an older version" — the WT-proxy was trained with an older XGBoost; the pickle still loads and predicts correctly. Fix: re-save the model with `booster.save_model()` after the next training run.
- RDKit MorganGenerator deprecation: `[23:17:09] DEPRECATION WARNING: please use MorganGenerator` — from the old `GetMorganFingerprintAsBitVect` API in `src/features/fingerprints.py`. Harmless; fix: migrate to `rdkit.Chem.rdMorganFingerprint.GetMorganGenerator`.

### Volume mounts (required)

Both containers read artifacts from bind mounts (not baked in):
```
./models -> /app/models:ro   backbone pkl + WT-proxy pkl + all JSON result files
./data   -> /app/data:ro     interim CSVs (for AD fitting) + generated/final_ranked_candidates.csv
```

The `models/generator/*.pt` torch checkpoints are excluded from the Docker context (`.dockerignore`) — they are large and not needed for serving.

### Environment variables

| Var | API default | Dashboard default |
|---|---|---|
| `USE_GCS` | `false` | `false` |
| `API_BASE_URL` | n/a | `http://api:8000` (service name inside compose network) |

`src/dashboard/api_client.py::DEFAULT_BASE_URL` reads `$API_BASE_URL` at import time, so override works for local non-Docker use too (still falls back to `http://127.0.0.1:8000`).

### Health and startup order

The API runs `ModelRegistry.load()` at startup (fits AD on ~1347 SMILES). `start_period=60s` in the health check accounts for this. The dashboard `depends_on: service_healthy`, so compose waits for the API `/health` to return `status:ok` before starting the dashboard process. The dashboard also has an in-process fallback to local registry if the API is not reachable at request time.

### MLflow stub

A commented stub for an MLflow service is in `docker-compose.yml`. Uncomment to add MLflow as a compose service.

## Phase 11 — MLflow experiment tracking (DONE)

Module: `src/models/mlflow_utils.py`. Experiment: `EGFR_QSAR_benchmark`.

**API:**
- `get_or_create_experiment()` → experiment_id (uses `mlflow.set_experiment()` — MLflow 3.x compatible)
- `start_run(task, model, seed)` — context manager; auto-sets `task`/`model`/`seed` tags
- `log_seed_summary(task, model, per_seed, params)` — logs mean/std summary metrics + per-seed JSON artifact

**Integrated into:**
- `scripts/eval_seed_stability.py` — logs QSAR 5-seed stability results on every run
- `scripts/train_gnn.py` — logs each GNN seed run + summary

**View results:** `mlflow ui --backend-store-uri mlruns` (or `MLFLOW_TRACKING_URI=mlruns mlflow ui`)

## Phase 13 — GNN benchmark (DONE, QSAR wins)

Run: `PYTHONPATH=. .venv/Scripts/python.exe scripts/train_gnn.py [--task general|wt_proxy|both] [--device cpu|cuda]`
Results: `models/gnn/general/metadata.json`, `models/gnn/wt_proxy/metadata.json`.

**Architecture** (`src/models/gnn_models.py`): SMILES → 41-dim atom features + 6-dim bond features → Linear embedding (41→128, 6→128) → GINEConv × 4 (BatchNorm + ReLU + Dropout 0.2) → GlobalMeanPool → MLP (128→64→1). ~200k parameters.

**Atom features (41-dim):** atomic_num one-hot (13) + degree (8) + hybridization (6) + formal_charge (6) + num_hs (6) + is_aromatic (1) + is_in_ring (1).
**Bond features (6-dim):** bond_type one-hot (4) + is_conjugated (1) + is_in_ring (1).
**Single-atom molecules** (e.g., methane): padded with self-loops so the graph has at least one edge.

**Training:** same 5 seeds and scaffold splits as QSAR. Early stopping patience=15, ReduceLROnPlateau (factor=0.5, patience=5), gradient clipping max_norm=1.0. ~95 epochs avg per seed, ~18 min CPU per task.

**5-seed scaffold-split results:**

| Task | Model | RMSE (mean ± std) | R² (mean ± std) | Spearman r (mean ± std) |
|---|---|---|---|---|
| EGFR general | **QSAR (XGBoost/RF)** | **1.010 ± 0.167** | **0.438 ± 0.143** | **0.672 ± 0.105** |
| EGFR general | GNN / GINEConv | 1.130 ± 0.067 | 0.291 ± 0.112 | 0.571 ± 0.075 |
| WT-proxy | **QSAR (XGBoost/RF)** | **0.942 ± 0.061** | **0.507 ± 0.063** | **0.717 ± 0.047** |
| WT-proxy | GNN / GINEConv | 1.061 ± 0.052 | 0.377 ± 0.049 | 0.659 ± 0.027 |

**Verdict: QSAR wins on both tasks.** GNN is consistently behind by ~0.12 RMSE on both tasks. This is the literature-expected result at n≈1k: gradient-boosted trees on fingerprints routinely beat GNNs below ~10k molecules because the fingerprint encodes expert chemical knowledge the GNN must learn from scratch. GNN does exhibit lower variance (±0.067 vs ±0.167 on the general task), reflecting a more stable representation, but does not overcome the accuracy gap at this scale.

**Production decision:** QSAR models (Models 1 and 2) remain in production. GNN artifacts in `models/gnn/` are available for reference but are not used in the API, dashboard, or ranking pipeline. A GPU run on a larger corpus (10k+ molecules, CHEMBL + PubChem EGFR actives) would be the right path to revisit the GNN.

**Public API** (`src/models/gnn_models.py`):
- `N_ATOM_FEATS = 41`, `N_BOND_FEATS = 6`
- `atom_features(atom)` → list[int]
- `bond_features(bond)` → list[int]
- `featurize(smiles, y=None)` → PyG Data | None
- `featurize_batch(smiles_list, y_list=None)` → (data_list, valid_indices)
- `build_gin_predictor(in_channels, edge_dim, hidden_channels, num_layers, dropout)` → nn.Module

**Tests:** 20 unit tests in `tests/test_gnn_models.py` — featurizer shapes, invalid SMILES, single-atom self-loop, batch API, forward pass shapes, finite output, determinism in eval mode, gradient flow, loss decreases. All guarded with `pytest.importorskip("torch_geometric")` — CI skips gracefully.

## Roadmap (remaining stages, in order)

- **Model 3 — L858R calibration**: **done** — LOOCV result is negative; backbone Spearman r 0.620 ± 0.008 beats calibration. Use general backbone for L858R predictions. All output exploratory.
- **Model 4 — selectivity / compound mutants**: **done** — see below. Selectivity cannot be modeled at n=9; use docking/FEP.
- **Phase B1 — structure prep**: **done** — 2ITZ (L858R) and 2ITY (WT) prepared, PDBQT written, box defined. See docking section above.
- **Fingerprint ablation (Phase 5)**: **done** — morgan_ecfp6 wins by val RMSE on both tasks, but margin is within seed noise on the general task. Production models keep Morgan ECFP4.
- **Phase B2 sanity check**: **done** — PASS. All 3 inhibitors favour L858R pocket (delta < 0). Ranked-library docking is next.
- **Phase B3 CNN rescoring**: **done** — BORDERLINE. 2/3 compounds pass direction criterion. Vina-only scoring retained for ranked-library docking. Do not use CNNaffinity for library scoring until validated on a 3rd-gen inhibitor without covalent bias.
- **Phase B2 ranked-library docking (first pass)**: **done** — covalent warhead detector added (`src/features/covalent.py`). 49/50 docked (1 timeout). 30/49 L858R-selective. Top non-covalent hit: cmpd_024 (delta −0.953). Results at `models/qsar/library_docking_results.json`.
- **Phase B2 docking noise quantification**: **done** — 15×2×5=150 Vina runs. 7 confident L858R-selective / 2 ambiguous / 6 covalent. cmpd_010 identified as single-seed artifact (mean delta −0.093). Confidence-filtered shortlist: cmpd_024, cmpd_012, cmpd_021, cmpd_015, cmpd_048, cmpd_037, cmpd_002. Results at `models/qsar/docking_noise_results.json`.
- **ADMET filtering**: **done** — `src/admet/filters.py` (Lipinski/Veber/PAINS/Brenk/QED/SA). Shortlist: 4/7 pass (cmpd_015, cmpd_012, cmpd_037, cmpd_002). Top-50: 16/50 pass. Michael_acceptor_1 (18×) dominates Brenk alerts. cmpd_021 identified as acrylate ester (not amide — missed by covalent SMARTS, caught by Brenk). Results at `models/qsar/admet_results.json`.
- **GNN (Phase 13) + MLflow (Phase 11)**: **done** — see sections below. QSAR wins on both tasks; production models stay XGBoost/RF. 628 unit tests passing.
- **De novo generation**: **two-stage done** — GRU pretrained on a 80k MOSES drug-like base then fine-tuned on the 1347 EGFR/ErbB2 actives. Validity 56.3% (single-corpus) → **92.3% at temp 0.8** (clears the 90% goal), scaffold diversity 0.50 (341 scaffolds, 37.5% novel). See section above. Generated hits screened through backbone + covalent + ADMET + **applicability domain** (no docking yet).
- **Applicability domain (Phase 15)**: **done** — `src/scoring/applicability_domain.py` (max Tanimoto, 3 bands). Integrated into `screen_generated.py`. 90% of generated molecules in-domain vs EGFR training set. Results in `models/generator/screen_results.json` (includes `applicability_domain` key). 532 tests passing.
- **RL fine-tuning (Phase 22)**: **done — two runs.** Run 1 (σ=0.5, no diversity filter) REWARD_HACKED (collapse to ~14 scaffolds, uniqueness 80%→8.6%). Run 2 (σ=0.25 + `ScaffoldMemory` filter, 50 steps) did NOT collapse (diversity held −0.012, 494 scaffolds seen) but is INCONCLUSIVE on activity (pIC50 +0.004). Activity and diversity are in tension at this corpus size. **Do NOT use `rl_finetuned_gru.pt`; production generator stays `egfr_finetuned_gru.pt`.** Results: `models/generator/rl_results.json` (Run 2; Run 1 numbers in CLAUDE.md). 579 tests passing.
- **Final integrated ranking (Phase 23)**: **done — capstone.** `src/scoring/ranking.py` v2 composite (0.30 activity + 0.30 docking-selectivity + 0.20 docking-affinity + 0.20 ADMET, each min-max normalised, × AD confidence_factor). Covalent + within-noise selectivity are WARNINGS, not score penalties. 49 known + 19 generated docked candidates ranked together. Top: cmpd_015, cmpd_002, cmpd_008 (covalent). Best generated gen_005 at #21 (just after the known top-20). Export: `data/generated/final_ranked_candidates.csv`. 35 tests in `tests/test_ranking.py`. See Phase 23 section above.
- **FastAPI backend (Phase 24)**: **done.** `src/api/` serves precomputed models only (no training/docking/ESM-2 at request time). Endpoints: `/health`, `/predict`, `/batch_predict`, `/model-info`. Returns fast ML/heuristic screen (backbone pIC50, WT-proxy, selectivity_proxy, ADMET, covalent, AD band, warnings); `docking_selectivity_available` always false. 28 tests (`tests/test_api.py`). See Phase 24 section above.
- **Streamlit dashboard (Phase 25)**: **done.** `src/dashboard/` (app + data_loaders + api_client). 6 pages (single, batch, ranking, performance, docking, limitations). Prefers the FastAPI service, falls back to local `ModelRegistry`. Altair visuals + error bars; honest Limitations page. 26 tests (`tests/test_dashboard.py`, incl. an `AppTest` render smoke test). See Phase 25 section above. 668 tests passing.
- **Docker packaging (Phase 26)**: **done.** `docker/api/Dockerfile` + `docker/dashboard/Dockerfile` (both `python:3.12-slim`). `requirements/serving.txt` — shared deps: numpy/pandas/scipy/joblib/sklearn/xgboost/rdkit/pydantic/pyyaml/python-dotenv; no torch, no lightgbm, no Vina/GNINA. `docker-compose.yml`: api (port 8000, healthcheck) + dashboard (port 8501, `depends_on: service_healthy`); models/ and data/ volume-mounted read-only; `API_BASE_URL=http://api:8000` so dashboard reaches API by service name; `USE_GCS=false` default; commented MLflow stub. `src/dashboard/api_client.py` DEFAULT_BASE_URL now reads `$API_BASE_URL` env var (falls back to localhost for non-Docker use). `.dockerignore` excludes `.venv/`, `data/docking/`, `data/raw/`, `data/processed/`, `models/generator/*.pt`, `tests/`, `scripts/`. 608 unit tests passing. `docker compose up --build` requires Docker Desktop to be running.
- **MLflow (Phase 11)**: **done** — `src/models/mlflow_utils.py`. Experiment "EGFR_QSAR_benchmark". `start_run(task, model, seed)` context manager. `log_seed_summary()` logs mean/std + per-seed JSON artifact. Integrated into `eval_seed_stability.py` and `scripts/train_gnn.py`. MLflow 3.x fix: use `mlflow.set_experiment()` instead of removed `get_experiment_by_name`. Run `mlflow ui --backend-store-uri mlruns` to browse.
