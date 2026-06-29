# Project Walkthrough: EGFR L858R Drug Discovery Pipeline

This document walks through every stage of the MutantScope pipeline in plain language, from the
biology that motivates it to the final ranked list of candidates. It is written for someone who
has never seen the project. Code details live in [CLAUDE.md](../CLAUDE.md); this document
explains the reasoning.

---

## Table of contents

1. [The biological problem](#1-the-biological-problem)
2. [The data reality: why this is hard](#2-the-data-reality-why-this-is-hard)
3. [Getting and cleaning the data](#3-getting-and-cleaning-the-data)
4. [Turning molecules into numbers: feature engineering](#4-turning-molecules-into-numbers-feature-engineering)
5. [Scaffold splitting: preventing a common mistake](#5-scaffold-splitting-preventing-a-common-mistake)
6. [The four-model cascade](#6-the-four-model-cascade)
7. [Checking the models are stable: seed and fingerprint studies](#7-checking-the-models-are-stable-seed-and-fingerprint-studies)
8. [The GNN benchmark: when does deep learning help?](#8-the-gnn-benchmark-when-does-deep-learning-help)
9. [Why the models are not enough: the pivot to structure-based docking](#9-why-the-models-are-not-enough-the-pivot-to-structure-based-docking)
10. [Docking: sanity check, library screen, and noise analysis](#10-docking-sanity-check-library-screen-and-noise-analysis)
11. [CNN rescoring with GNINA](#11-cnn-rescoring-with-gnina)
12. [ADMET filtering: drug-like properties](#12-admet-filtering-drug-like-properties)
13. [De novo molecule generation](#13-de-novo-molecule-generation)
14. [Reinforcement learning fine-tuning: what went wrong and why](#14-reinforcement-learning-fine-tuning-what-went-wrong-and-why)
15. [Applicability domain: knowing when not to trust the model](#15-applicability-domain-knowing-when-not-to-trust-the-model)
16. [Final integrated ranking](#16-final-integrated-ranking)
17. [The serving layer: FastAPI, Streamlit, and Docker](#17-the-serving-layer-fastapi-streamlit-and-docker)
18. [Discovery Copilot: a deterministic query and explanation layer](#18-discovery-copilot-a-deterministic-query-and-explanation-layer)
19. [Follow one molecule: gefitinib and gen_005](#19-follow-one-molecule-gefitinib-and-gen_005)
20. [Limitations](#20-limitations)
21. [Future work](#21-future-work)

---

## 1. The biological problem

### EGFR and lung cancer

EGFR (epidermal growth factor receptor) is a protein that sits on the surface of cells and sends
"grow and divide" signals when it receives a growth factor. In healthy tissue this is tightly
regulated. In non-small cell lung cancer (NSCLC), mutations in the gene that encodes EGFR can
cause the receptor to stay permanently switched on, driving uncontrolled tumour growth.

NSCLC accounts for roughly 85% of all lung cancers, making it one of the most common cancers
worldwide. A subset of NSCLC patients, around 10-15% in Western populations and up to 40% in East
Asian populations, carry activating EGFR mutations, and targeted therapy against mutant EGFR is
one of the best-validated strategies in precision oncology.

### The L858R mutation

The L858R mutation replaces a leucine (L) with an arginine (R) at position 858 of the EGFR protein.
This single amino acid change locks the kinase domain in an active conformation, making the receptor
hypersensitive and constitutively active. It is one of the two most common drug-sensitising EGFR
mutations (the other is an exon-19 deletion), and together they account for around 85% of all
EGFR-mutant NSCLC.

First-generation EGFR inhibitors such as gefitinib and erlotinib were among the first precision
oncology drugs. Yun et al. (Cancer Cell, 2007) showed that gefitinib binds the L858R mutant
receptor approximately 20 times more tightly than it binds the wild-type receptor (Kd ~7 nM vs
~142 nM), which is what makes selective treatment possible.

### Why selectivity matters

The wild-type EGFR receptor is expressed in normal skin, gut lining, and other healthy tissues.
A drug that inhibits wild-type EGFR as strongly as the mutant causes the rash, diarrhoea, and
other epithelial toxicities that are the main side effects of EGFR inhibitors. Compounds that
preferentially bind the mutant receptor over the wild-type can potentially achieve the same tumour
control with less toxicity.

This project defines selectivity as:

```
selectivity_delta = pIC50_mutant - pIC50_wt
```

A positive value means the compound is more potent against the L858R mutant than the wild-type
receptor. Finding and ranking compounds with a positive, large selectivity delta is the central
goal.

### Resistance and T790M

Patients on first-generation EGFR inhibitors almost always develop resistance within one to two
years. The most common resistance mechanism is a second mutation, T790M (threonine to methionine at
position 790), which sterically blocks gefitinib and erlotinib. Third-generation inhibitors such as
osimertinib were designed to overcome T790M resistance. Understanding the L858R mutation, including
without T790M, is therefore an important step in the larger resistance landscape, and the project
is designed to keep L858R, T790M single-mutant, and compound mutant records as separate categories.

---

## 2. The data reality: why this is hard

### Where the data comes from

All bioactivity data comes from ChEMBL, the largest publicly available database of drug-like
molecules and their biological activities. Queries for EGFR activity measurements return
IC50 and Kd values for thousands of molecules, but not all are equally useful.

The raw download contains 1,280 records. After filtering for valid SMILES, appropriate molecular
size (heavy atom count up to 70), valid activity units, and duplicate removal, 1,253 records remain.

### The scarcity problem

Those 1,253 records span every mutation type, and most are of unknown mutation status because
older assays did not specify which EGFR construct was used. When you separate out records where
the assay explicitly measured the L858R mutant:

| Category | Records |
|---|---|
| Unknown mutation (WT in practice) | 957 |
| T790M | 200 |
| Wild-type (explicit) | 61 |
| L858R | **22** |
| del19 | 13 |

Only 22 labelled L858R records exist. That is far too few to train a standalone model. Most machine
learning models need hundreds to thousands of examples to generalise. With 22, you can fit but
cannot generalise: any test you run on held-out data is dominated by which 1-2 compounds happened
to end up in the test set.

### The consequence: everything is exploratory

The entire modelling strategy is designed around this scarcity. Transfer learning from the large
(1,253-record) general model to the small L858R set is the only credible path. Even then, the
key negative results, that L858R-specific calibration does not improve on the backbone and that
selectivity cannot be modelled from 9 paired measurements, are themselves findings, not failures.
This document, and every result in this project, should be read as a research and portfolio
demonstration, not a validated drug-discovery platform. No candidates have been synthesised or
tested experimentally.

---

## 3. Getting and cleaning the data

### Raw data acquisition

The raw ChEMBL CSV (`data/raw/chembl_egfr_bioactivity.csv`) is queried from ChEMBL's REST API
targeting EGFR (CHEMBL203) and ErbB2/HER2 (CHEMBL1824, a closely related kinase used for the
generator training set). Activity values are in IC50 or Kd units and are converted to pIC50
(negative log of the molar concentration) for a linear scale.

### Cleaning steps (scripts/clean_bioactivity_data.py)

Each record passes through four checks:

1. **SMILES validation.** RDKit attempts to parse the SMILES string. Any string that fails to
   parse is dropped.

2. **Canonicalisation.** RDKit converts every valid SMILES to its canonical form so that the same
   molecule written in different ways resolves to the same string. This prevents counting the same
   compound twice because one record uses a salt form.

3. **Size filter.** Molecules with more than 70 heavy atoms are dropped. Most drug-like EGFR
   inhibitors are smaller; very large molecules are likely data artefacts or PROTACs that behave
   differently.

4. **Deduplication.** When the same canonical SMILES appears multiple times (usually because the
   same compound was assayed in multiple labs), the record with the highest pIC50 is kept.

After cleaning: 1,253 EGFR records and 604 ErbB2 records remain. The ErbB2 set is used only for
pre-training the molecular generator, not for the QSAR models.

### The L858R re-labelling story

The original mutation labels assigned during the ChEMBL download contained a subtle error. Assay
CHEMBL4380726 was described as "binding affinity to human wild-type partial length EGFR L858R
mutant, KINOMEscan." The phrase "wild-type" refers to the protein backbone structure, not to the
mutation status: the actual construct measured is the L858R mutant. This assay was originally
flagged as wild-type, mislabelling 5 records.

Of those 5 records, 2 correspond to PROTAC degrader molecules with 74 heavy atoms, which are
legitimately dropped by the heavy-atom filter. The remaining 3 are correctly re-labelled from
wild-type to L858R in the cleaning script, keyed by assay ID so the fix is idempotent.

Net result: L858R count goes from 19 to 22. Wild-type count goes from 64 to 61.

A separate scan found that 130 of the 211 T790M-flagged records are actually compound mutants
(92 are L858R/T790M double mutants, 38 are L858R/T790M/C797S triple mutants). Those re-labelled
records are kept in the general training set under corrected flags, but the genuine T790M single
mutant count drops to 81. This matters if a future Model 4 targeting T790M is built.

---

## 4. Turning molecules into numbers: feature engineering

Machine learning models need numbers. A molecule is a graph of atoms and bonds, which needs to be
converted into a fixed-length numerical vector before a model can process it.

### Morgan fingerprints (ECFP4)

The primary representation is the Morgan circular fingerprint at radius 2, commonly called ECFP4.
The algorithm works by:

1. Assigning an initial identifier to each atom based on its local chemical environment (atomic
   number, charge, whether it is aromatic, etc.).
2. For each atom, hashing together its own identifier with the identifiers of its immediate
   neighbours, then the neighbours' neighbours (out to radius 2), and so on.
3. Recording which of 2,048 "bit positions" are set. Each bit position corresponds to the presence
   of a particular circular chemical substructure somewhere in the molecule.

The result is a 2,048-dimensional binary vector. Molecules that share structural features have
similar fingerprints, and Tanimoto similarity (the fraction of bits set in either molecule that are
set in both) is a well-validated measure of structural similarity for drug-like compounds.

### RDKit descriptors

Eleven physicochemical descriptors are appended to the fingerprint:

| Descriptor | What it measures |
|---|---|
| Molecular weight | Size |
| LogP | Lipophilicity (tendency to partition into fats vs water) |
| TPSA | Topological polar surface area (membrane permeability proxy) |
| HBD | Hydrogen bond donors |
| HBA | Hydrogen bond acceptors |
| Rotatable bonds | Flexibility |
| Ring count | Structural rigidity |
| Aromatic ring count | Aromaticity |
| Fraction Csp3 | Proportion of sp3 (non-aromatic, non-flat) carbons |
| Lipinski violations | How many Lipinski Ro5 rules are violated |
| QED | Quantitative estimate of drug-likeness (0 to 1) |

The final feature vector is 2,048 + 11 = 2,059 dimensions. The column order is fixed across all
models and must never be reordered.

### Why not raw atomic coordinates?

Protein structures are 3-dimensional, but small-molecule SMILES strings only encode connectivity,
not 3D geometry. Generating reliable 3D conformations for every molecule is expensive and adds
noise. Fingerprints capture the connectivity pattern that actually drives binding in QSAR studies,
and at dataset sizes of ~1k molecules they consistently outperform 3D approaches for pIC50
regression. The docking stages (Phases B1-B3) use 3D geometry explicitly, but only for the final
shortlist.

---

## 5. Scaffold splitting: preventing a common mistake

### The problem with random splitting

The standard way to evaluate a model is to hold back a random 20% of the data, train on 80%, and
report performance on the held-out 20%. For molecules this approach is dangerously optimistic
because structurally very similar compounds (with nearly identical fingerprints) end up on both
sides of the split. The model effectively memorises a template and interpolates across small
modifications. The reported accuracy on the test set vastly overstates how the model will perform
on genuinely novel compounds.

### Bemis-Murcko scaffold splitting

A scaffold is the core ring system of a molecule. The Bemis-Murcko algorithm extracts the ring
system and the chains connecting rings, discarding peripheral substituents. Two kinase inhibitors
that differ only in their side chains share the same scaffold.

Scaffold splitting places all molecules sharing the same scaffold into the same split (train, val,
or test). A compound in the test set is guaranteed to have a scaffold not seen during training. This
is a much harder and more realistic evaluation.

The cost is variance. A single scaffold can contain many of the most active or least active
compounds in the dataset; wherever it lands determines a lot of the split's difficulty. The solution
is to run multiple seeds (the scaffold assignment is seeded so different seeds produce different
partitions) and report mean and standard deviation across seeds.

In this project all reported metrics are 5-seed scaffold split averages.

---

## 6. The four-model cascade

The modelling strategy is designed to extract signal from a large dataset and carefully transfer
it to the tiny L858R subset. There are four models, each with a specific role.

### Model 1: EGFR general backbone

**What it is:** A regression model trained on all 1,253 cleaned EGFR records to predict pIC50
across all mutation types.

**Why it exists:** The 1,253 records include L858R, T790M, wild-type, and unknown-mutation records.
The idea is that EGFR kinase pharmacology is shared across mutations: the ATP binding pocket
geometry, the key binding interactions with the kinase hinge region, and the relationship between
molecular features and potency are mostly conserved. A model trained on all records learns the
general structure-activity relationship for EGFR inhibitors, even if it does not know which
mutation it is predicting for.

**Algorithm selection:** Three algorithms are evaluated by 5-fold validation RMSE: Random Forest,
XGBoost, and LightGBM. The winner is the one with the best validation RMSE. For the general
backbone the winner is Random Forest.

**Results (5-seed scaffold split):**

| Metric | Mean | Std |
|---|---|---|
| Test RMSE | 1.010 | 0.167 |
| Test R^2 | 0.438 | 0.143 |
| Pearson r | 0.672 | 0.105 |

The high R^2 variance (+/-0.143) is structural: at seed 99, the scaffold split happens to place
structurally unusual compounds in the test set that are very different from the training set. A
model that cannot extrapolate to structurally novel scaffolds looks poor on that partition
regardless of its actual quality. The 5-seed average (0.438) is the reliable estimate.

The single-seed artifact (models/qsar/general/metadata.json, seed 42, R^2=0.446) is stored as
the deployed artifact only because seed 42 is the pipeline seed; it does not represent a best-case
estimate.

### Model 2: WT-proxy comparator

**What it is:** A separate regression model trained on ~1,018 records: the 61 explicit wild-type
records plus the 957 unknown-mutation records. The unknown records are predominantly wild-type
because most EGFR assays in ChEMBL did not report a specific mutation, and the wild-type was the
dominant construct in early drug discovery.

**Why it is called "WT-proxy" not "WT":** Because 957 of its 1,018 training records are
technically of uncertain mutation status. It is an approximation of wild-type activity, not a
precise measurement. Calling it "WT-only" would be misleading.

**Algorithm selection:** XGBoost wins for the WT-proxy task.

**Results (5-seed scaffold split):**

| Metric | Mean | Std |
|---|---|---|
| Test RMSE | 0.942 | 0.061 |
| Test R^2 | 0.507 | 0.063 |
| Pearson r | 0.717 | 0.047 |

The WT-proxy performs better and with lower variance than the general backbone because the chemical
space is more homogeneous: most kinase-inhibitor ChEMBL data comes from similar quinazoline and
indazole-based scaffolds. The general backbone includes structural outliers from multiple mutation
types.

### Model 3: L858R calibration (LOOCV, EXPLORATORY)

**What it tries to do:** Use the 22 labelled L858R records to calibrate the backbone's predictions
so they are better tuned to L858R activity specifically.

**Why LOOCV (leave-one-out cross-validation)?** With only 22 records, a normal train/test split
would leave either too few training examples or too few test examples to be meaningful. LOOCV
holds out exactly one record, trains on the remaining 21, and predicts the held-out one. This is
repeated 22 times. Each seed retrain also excludes the held-out molecule from the backbone
training set to prevent leakage.

**Two calibration strategies were tested:**
- Mean-shift: add a constant offset equal to the mean residual between backbone predictions and
  true L858R pIC50 values.
- Ridge regression: train a linear model to map backbone predictions to L858R pIC50.

**Results (5 backbone seeds, LOOCV):**

| Method | Spearman r mean | Spearman r std | RMSE mean |
|---|---|---|---|
| Backbone (baseline) | 0.620 | 0.008 | 0.941 |
| Mean-shift | 0.599 | 0.012 | 0.827 |
| Ridge | 0.593 | 0.017 | 0.873 |

**The negative result:** Calibration does not improve rank correlation over the backbone. Mean-shift
reduces RMSE (0.827 vs 0.941) because the backbone systematically underpredicts L858R pIC50 and a
uniform shift corrects the bias on average, but it does not help rank compounds from best to worst.
Ridge performs no better.

**Verdict:** L858R-specific signal is not separable at n=22. The general backbone is used for all
L858R activity predictions. All L858R-specific outputs are labelled exploratory.

### Model 4: Derived selectivity (EXPLORATORY)

**What it tries to do:** Predict the selectivity delta (pIC50 L858R minus pIC50 WT) for a compound
using the two-model derived difference: backbone prediction minus WT-proxy prediction.

**Why this matters:** A model that predicts selectivity directly could be used to screen large
libraries for L858R-selective compounds without running expensive docking. But this only works if
the signal-to-noise ratio is high enough.

**The dataset:** 9 molecules that have both a labelled L858R measurement and a labelled wild-type
measurement in ChEMBL.

**Results:**

| Method | Spearman r | p-value |
|---|---|---|
| Derived delta (backbone minus WT-proxy) | 0.433 | 0.244 |

Spearman r=0.433 with p=0.244 is not statistically significant. The 0.05 significance threshold
for n=9 requires r > 0.67.

**Why it fails:** Both the backbone and WT-proxy make substantial individual pIC50 errors (RMSE ~1
pIC50 unit). Those errors have both shared components (features both models mis-score in the same
direction) and independent components. The shared components cancel in the difference; the
independent components add. With only 9 pairs and per-prediction noise of ~1 unit, the 3-5 pIC50
unit selectivity range of real kinase inhibitors is buried in noise.

**Verdict:** Selectivity cannot be modelled from 9 data points with imperfect individual models.
Structure-based docking is the path. The 9 deltas are reference data, not a model output.

---

## 7. Checking the models are stable: seed and fingerprint studies

### 5-seed stability analysis

A concern with scaffold splitting is that a single train-test partition might be unusually lucky or
unlucky. Running 5 different scaffold splits with different random seeds and reporting mean and
standard deviation gives a realistic estimate of the range of performance the model can achieve.

The results (QSAR seed stability, reproduced by running
`scripts/eval_seed_stability.py`) show:
- General backbone: R^2 ranges from ~0.29 to ~0.58 across seeds. High variance is genuine, not a
  model defect.
- WT-proxy: R^2 ranges from ~0.44 to ~0.57. Lower variance because the chemical space is more
  homogeneous.

The single-seed artifact values in `models/qsar/general/metadata.json` (seed 42, R^2=0.446) and
`models/qsar/wt_proxy/metadata.json` (seed 42, R^2=0.604) fall within these ranges. The WT-proxy
seed-42 value of 0.604 is at the upper end of its 0.507 +/- 0.063 range; the reliable estimate is
the mean.

### Fingerprint ablation

Six fingerprint types were compared across both tasks (5 seeds each, best of RF/XGB/LGB per type):
ECFP6 (radius 3), ECFP4 (radius 2), topological torsion, RDKit topological, atom-pair, and MACCS.

The winner by validation RMSE is ECFP6 on both tasks (0.949 vs 0.973 for ECFP4 on general, 0.855
vs 0.893 for WT-proxy). However, the margin falls within the 5-seed test RMSE standard deviation
for the general task (+/-0.18), meaning it could be noise. Production models keep ECFP4, avoiding
the need to rebuild all downstream artifacts for a marginal gain.

MACCS keys (167 bits) are consistently the worst representation. They were designed for
substructure searching, not regression, and they are too coarse to capture the fine structural
distinctions between similar kinase inhibitors.

---

## 8. The GNN benchmark: when does deep learning help?

### Graph neural networks for molecules

A molecule is naturally represented as a graph: atoms are nodes, bonds are edges. Graph neural
networks (GNNs) learn directly on this structure by iteratively aggregating information from
neighbouring atoms, building up representations of progressively larger subgraphs. In principle
a GNN can learn arbitrary structural patterns from raw atomic graphs without needing hand-crafted
fingerprints.

The specific architecture tested is GINEConv (Graph Isomorphism Network with Edge features), four
convolution layers with batch normalisation and dropout, a global mean pooling step, and a two-layer
MLP that outputs a single pIC50 prediction. It has approximately 200,000 parameters.

### Why test it here?

At large scale (>10k molecules), GNNs frequently match or beat fingerprint-based models because
they can learn task-specific representations. Testing whether a GNN improves over QSAR at ~1k
molecules is a genuine question worth answering.

### Results (5-seed scaffold split)

| Model | Task | RMSE mean +/- std | R^2 mean +/- std |
|---|---|---|---|
| QSAR (XGBoost/RF) | General | 1.010 +/- 0.167 | 0.438 +/- 0.143 |
| GNN (GINEConv) | General | 1.130 +/- 0.067 | 0.291 +/- 0.112 |
| QSAR (XGBoost/RF) | WT-proxy | 0.942 +/- 0.061 | 0.507 +/- 0.063 |
| GNN (GINEConv) | WT-proxy | 1.061 +/- 0.052 | 0.377 +/- 0.049 |

**Verdict:** QSAR wins on both tasks. The GNN RMSE is consistently higher by about 0.12 on both
tasks. The GNN shows lower variance (general: +/-0.067 vs +/-0.167) because its graph
representation is inherently more stable than a fingerprint under scaffold partition shifts, but
lower variance at worse accuracy is not an improvement.

This is the expected result from the literature. Below approximately 10,000 molecules, fingerprints
encode expert chemical knowledge that the GNN must learn from scratch. The GNN simply does not have
enough data to learn what ECFP4 already encodes by design. Production models remain XGBoost/RF.

---

## 9. Why the models are not enough: the pivot to structure-based docking

### The selectivity problem revisited

Models 1 through 4 cannot reliably model selectivity. The backbone predicts L858R pIC50 reasonably
well (Spearman r ~0.62), and the WT-proxy predicts wild-type pIC50 (Spearman r ~0.72), but the
per-compound error is ~1 pIC50 unit for each. When you subtract two noisy predictions, the noise
adds in quadrature. The expected selectivity delta of 1-3 pIC50 units for a good selective compound
is the same order as the combined noise.

### What structure-based docking can add

Docking places a small molecule into the 3-dimensional binding pocket of a protein and scores the
interaction energy. Instead of asking "what does this molecule's fingerprint suggest about potency?"
docking asks "how well does this molecule physically fit into the ATP binding cleft, and does it
make the hydrogen bonds and hydrophobic contacts the receptor expects?"

By docking the same compound into both the L858R and wild-type receptor structures, you get a
direct comparison of how well it fits each pocket. The pocket geometries differ because the L858R
substitution rearranges helix C and shifts the glycine-rich loop, creating a slightly different
binding profile. Compounds that exploit those differences will show a larger selectivity delta in
docking.

Docking is not perfect (it uses a rigid receptor, ignores induced-fit effects, and its scoring
function is an approximation), but it provides orthogonal evidence to the QSAR models. Combining
both, you are more confident in a compound that scores well on both.

---

## 10. Docking: sanity check, library screen, and noise analysis

### Phase B1: Structure selection and preparation

A matched pair of crystal structures from the same paper and same construct was chosen:

| Role | PDB ID | Resolution | Ligand |
|---|---|---|---|
| L858R | 2ITZ | 2.8 Angstrom | IRE (gefitinib) |
| Wild-type | 2ITY | 3.42 Angstrom | IRE (gefitinib) |

Both are from Yun et al. (Cancer Cell, 2007), both cover the EGFR kinase domain residues 696-1022,
and both co-crystallise the same ligand (gefitinib). Using a matched pair anchors the comparison:
any score difference between the two pockets reflects the L858R mutation rather than differences
in the co-crystal ligand or experimental conditions.

Several structures containing T790M (5UGA, 5UG8, 5UGC, 5UWD, 4I21) were excluded to avoid
confounding L858R selectivity with resistance-mutation effects.

**Preparation pipeline (identical for both structures):**
1. Download from RCSB PDB.
2. Extract chain A, strip all HETATM atoms except the co-crystal ligand (IRE/gefitinib).
3. pdbfixer: add missing hydrogen atoms at physiological pH 7.4, add any missing heavy atoms to
   incomplete residues (loop modelling is skipped, only incomplete side chains are fixed).
4. Convert to PDBQT format (required by AutoDock Vina): assign AutoDock4 atom types (aromatic
   carbons as "A", hydrogen-bond donor hydrogens as "HD", etc.), set receptor charges to zero
   (Vina's scoring function does not use receptor charges).

The wild-type structure (2ITY) is structurally aligned to the L858R structure (2ITZ) on 300
common C-alpha atoms using Biopython's Superimposer. The C-alpha RMSD after alignment is 1.643
Angstrom, consistent with the two structures differing by one point mutation and slightly different
crystal packing.

**Docking box:** Centred on the heavy-atom centroid of the co-crystal gefitinib in 2ITZ:
- Centre: x=-51.654, y=-1.266, z=-21.945 Angstrom
- Size: 22.5 x 22.5 x 22.5 Angstrom (covers the ATP binding cleft with ~5 Angstrom buffer)

The same box is used for both structures because the 2ITY pocket is in the same reference frame
after alignment.

### Phase B2: Sanity check

Before screening a library, it is essential to verify that the docking setup reproduces known
biology. Three clinically validated EGFR inhibitors were docked into both structures:

| Compound | L858R (kcal/mol) | WT (kcal/mol) | Delta | Direction |
|---|---|---|---|---|
| Gefitinib | -7.860 | -7.492 | -0.368 | L858R favoured |
| Erlotinib | -7.666 | -7.263 | -0.403 | L858R favoured |
| Osimertinib | -7.944 | -7.306 | -0.638 | L858R favoured |

All three favour the L858R pocket (delta < 0 means L858R scores more negative, i.e. tighter
binding by Vina's convention). This is the correct direction: all three compounds are clinically
active against L858R tumours, and gefitinib is known from Yun et al. (2007) to bind ~20-fold
tighter to L858R than to wild-type.

The delta magnitudes (0.4-0.6 kcal/mol) are much smaller than the 1.7 kcal/mol expected from a
20-fold affinity difference (RT ln(20) at 300 K). This underestimation is expected for rigid
receptor docking: the L858R mutation causes subtle conformational changes that rigid Vina cannot
fully capture. Direction correct, magnitude not quantitatively reliable. Verdict: PASS.

**AutoDock Vina settings:** version 1.2.7, pre-built Windows binary (pip install fails on Windows
because it requires a Boost library build), exhaustiveness=8, seed=42, 9 output poses.

### Phase B2: Library docking

The top 50 candidates by backbone-predicted pIC50 were selected from the 1,253 EGFR actives.
Before docking, each compound was checked for covalent warheads (electrophilic groups that form
a covalent bond with a cysteine residue, most commonly C797 in EGFR inhibitors) using SMARTS
pattern matching. Covalent compounds are flagged as "low_confidence" because rigid non-covalent
docking cannot represent the actual binding mode.

Results:
- 49 of 50 compounds docked successfully (1 macrocycle, cmpd_041, timed out at 300 s)
- 30 of 49 were L858R-selective (delta < 0)
- 22 of 50 were covalent-flagged (21 acrylamide warheads + 1 acrylate ester)

Top non-covalent L858R-selective hits: cmpd_024 (delta -0.953 kcal/mol), cmpd_015 (-0.634),
cmpd_012 (-0.591), cmpd_048 (-0.430), cmpd_037 (-0.419), cmpd_002 (-0.392).

### Phase B2: Docking noise quantification

A single Vina docking run is a point estimate. Vina uses a Monte Carlo search algorithm that
depends on a random seed, and the binding score can vary by 0.1-0.3 kcal/mol across seeds for
the same compound. A selectivity delta is the difference between two such estimates. A delta of
-0.3 kcal/mol with per-pocket noise of 0.2 kcal/mol is within noise; a delta of -0.8 kcal/mol
with per-pocket noise of 0.15 kcal/mol is not.

The noise study ran 5 different Vina seeds for each of the top 15 compounds in each pocket
(15 compounds x 2 pockets x 5 seeds = 150 total Vina runs). The seed-to-seed standard deviation
was measured per pocket, and the delta standard deviation was propagated as:

```
std_delta = sqrt(std_L858R^2 + std_WT^2)
```

A "confident" call requires |delta| > 1.5 x std_delta.

**Results:**
- 6 compounds: confidently L858R-selective (non-covalent, |delta| > 1.5 x std)
- 2 compounds: ambiguous (within noise)
- 7 compounds: low-confidence covalent

A notable finding: cmpd_010 appeared to have delta -0.392 at seed 42 but has a mean delta of
-0.093 across 5 seeds, with the L858R score ranging from -6.577 to -7.195. The initial estimate
was a lucky outlier. This is precisely the kind of artefact the noise study was designed to catch.

**Confidence-filtered shortlist (standard confidence, L858R-selective):**
cmpd_024 (-0.813 +/- 0.277), cmpd_012 (-0.546 +/- 0.196), cmpd_015 (-0.452 +/- 0.139),
cmpd_048 (-0.430 +/- 0.198), cmpd_037 (-0.357 +/- 0.042), cmpd_002 (-0.342 +/- 0.048).

---

## 11. CNN rescoring with GNINA

AutoDock Vina uses a physics-inspired scoring function based on shape complementarity and
electrostatic interactions. GNINA (a fork of AutoDock Vina) replaces the scoring function with
a convolutional neural network (CNN) trained on crystallographic protein-ligand complexes from
the PDBbind database. In principle a CNN that has seen thousands of real binding poses should
score them more accurately than an analytical function.

**Setup:** GNINA v1.0 (December 2021) runs under WSL2 on Linux. The newer v1.3.2 requires CUDA
GPU libraries that are not available in a CPU-only WSL2 environment, so v1.0 is used.

**Results:**

| Compound | L858R CNN affinity (pKd) | WT CNN affinity (pKd) | CNN delta | Vina delta |
|---|---|---|---|---|
| Gefitinib | 6.693 | 6.389 | +0.303 pKd | -0.368 kcal/mol |
| Erlotinib | 5.909 | 5.670 | +0.239 pKd | -0.403 kcal/mol |
| Osimertinib | 5.708 | 5.756 | -0.048 pKd | -0.638 kcal/mol |

Gefitinib and erlotinib: CNN correctly favours L858R. Osimertinib: CNN gives wild-type a tiny
advantage (-0.048 pKd, effectively zero given the ensemble uncertainty CNNvariance=1.66). Osimertinib
is a covalent inhibitor targeting C797 via an acrylamide warhead, and the GNINA CNN was trained on
non-covalent PDBbind complexes. It does not recognise the covalent pharmacophore.

**Verdict: BORDERLINE.** 2 of 3 compounds pass the direction criterion. Vina-only scoring is
retained for library ranking. The CNN would need to correctly score all 3 before being used for
ranked library scoring.

---

## 12. ADMET filtering: drug-like properties

ADMET stands for absorption, distribution, metabolism, excretion, and toxicity. Even a compound
with excellent predicted potency is useless as a drug if it cannot reach its target in the body
or if it causes off-target toxicity.

The ADMET filters applied here are approximate computational proxies, not experimental
measurements. A flag does not disqualify a compound; it raises a concern for further investigation.

**Filters applied:**

| Filter | Rule |
|---|---|
| Lipinski Rule of 5 | MW <= 500, LogP <= 5, HBD <= 5, HBA <= 10; flag if > 1 violation |
| Veber | Rotatable bonds <= 10, TPSA <= 140; flag if either violated |
| PAINS | RDKit structural alert database; flag if any match |
| Brenk | RDKit metabolic/reactive alert database; flag if any match |
| QED | Flag if < 0.25 (very poor drug-likeness) |
| SA score | Flag if > 6.0 (very difficult synthesis) |

**Shortlist results:**

| Compound | MW | LogP | QED | SA | Brenk | Status |
|---|---|---|---|---|---|---|
| cmpd_015 | 343 | 3.6 | 0.787 | 2.4 | no | pass |
| cmpd_012 | 409 | 3.9 | 0.447 | 2.8 | no | pass |
| cmpd_037 | 353 | 4.0 | 0.591 | 2.4 | no | pass |
| cmpd_002 | 359 | 4.2 | 0.757 | 2.0 | no | pass |
| cmpd_024 | 423 | 2.8 | 0.518 | 2.5 | Sulfonic_acid_2 | flag |
| cmpd_048 | 359 | 3.8 | 0.489 | 2.5 | Aliphatic_long_chain | flag |

**Clean shortlist:** cmpd_015, cmpd_012, cmpd_037, cmpd_002. All four are drug-like, synthetically
accessible (SA 2.0-2.8), and have no PAINS or Brenk alerts. SA scores of 2.0-2.8 correspond to
standard medicinal chemistry scaffolds.

cmpd_024 has the strongest docking selectivity (-0.813 kcal/mol) but carries a Brenk
"Sulfonic_acid_2" alert: the sulfonamide scaffold contains a free sulfonic acid (-SO3H) which
reduces membrane permeability. A bioisostere substitution (replacing -SO3H with -SO2NH2 or a
tetrazole) could rescue this compound.

---

## 13. De novo molecule generation

### Why generate new molecules?

The top-50 library candidates come from known EGFR actives in ChEMBL. They provide good
evidence of binding activity but are all known compounds. To genuinely discover novel chemical
matter, a molecular generator is needed.

### The single-corpus failure

The first attempt trained a character-level GRU (gated recurrent unit) language model on the
1,347 EGFR/ErbB2 actives. A character-level model generates SMILES strings one character at a
time, treating molecule generation as next-character prediction. The problem: 1,347 molecules
is far too few to learn the grammar of valid SMILES. Valid SMILES must balance parentheses, rings,
and aromatic systems in precise ways. At 1,347 examples, 56.3% of generated strings were valid
SMILES. This is too low for practical screening (most of your compute is wasted on invalid strings).

### Two-stage training

The fix is to pretrain on a large drug-like corpus first, then fine-tune on the EGFR-specific
molecules.

**Stage 1: Base pretraining.** The MOSES benchmark dataset contains 1.94 million drug-like SMILES
(filtered for MW 150-500, LogP -1 to 5). The first 80,000 molecules (capped for CPU feasibility)
were used to pretrain the GRU for 6 epochs. After pretraining, the model has learned the grammar
of drug-like SMILES: it knows how to balance rings, generate aromatic systems correctly, and
produce chemically plausible strings. Validation loss: 0.586 at epoch 6.

**Stage 2: EGFR fine-tuning.** The pretrained model was warm-started and fine-tuned on the 1,347
EGFR/ErbB2 actives for 22 epochs. The model now generates SMILES that are both grammatically
valid and chemically similar to known EGFR inhibitors. Validation loss: 0.366 at epoch 22.

**Architecture:** Character-level GRU, embedding dim 128, hidden 512 x 3 layers, dropout 0.1,
max length 120 characters, teacher forcing during training, Adam optimiser with
ReduceLROnPlateau. The tokenizer is regex-based, fitting on the union of both corpora (43 tokens
total) so that the same vocabulary covers both stages.

**Sampling temperature.** A softmax temperature > 1.0 flattens the probability distribution,
producing more diverse but less valid strings. Temperature < 1.0 sharpens it, producing more
valid but more repetitive strings. The sweet spot:

| Temp | Validity | Uniqueness | Scaffold diversity |
|---|---|---|---|
| 1.0 | 81.0% | 87.3% | 0.523 |
| 0.9 | 84.2% | 80.5% | 0.516 |
| **0.8** | **92.3%** | 73.8% | 0.501 |
| 0.7 | 94.5% | 63.0% | 0.472 |

Temperature 0.8 is the operating point: validity clears 90%, 341 distinct scaffolds out of 681
unique molecules, 37.5% of scaffolds absent from the EGFR training set.

**Screening pipeline.** Generated molecules pass through: RDKit validity check, canonical SMILES
deduplication, novelty filter (must not be in the EGFR training set), backbone pIC50 prediction,
covalent warhead detection, ADMET filtering, and applicability domain assessment.

---

## 14. Reinforcement learning fine-tuning: what went wrong and why

> **Terminology note**: in this section, "agent" refers to the REINVENT GRU generator, the
> neural network whose policy is updated by the RL training loop. This is distinct from the
> Discovery Copilot orchestration layer in `src/agent/` (see section 18), which is a conventional
> software module that routes queries to deterministic functions.

### The goal

RL fine-tuning aims to steer the generator away from average drug-like molecules toward EGFR
inhibitors that are specifically active and selective. The generator policy (the weights of the
GRU) is treated as an agent. At each step the agent generates a molecule, a reward function
evaluates it, and the policy is updated to make high-reward molecules more likely.

### REINVENT

The algorithm used is REINVENT (Olivecrona et al., 2017), an augmented negative log-likelihood
objective. The prior (frozen copy of the base generator) regularises the agent: if the agent drifts
too far from the prior's distribution it is penalised. The sigma parameter controls the balance,
with higher sigma meaning the reward signal has more influence over the prior.

**Reward components:**

| Component | Weight | Notes |
|---|---|---|
| Activity: sigmoid(pIC50 - 7.0) x AD confidence | 0.50 | pIC50 from backbone; AD guards OOD exploitation |
| QED (drug-likeness) | 0.20 | Continuous 0-1 score |
| ADMET pass bonus | +0.20 flat | |
| Novelty bonus | +0.10 flat | Not in training set |
| Invalid SMILES penalty | -1.00 | Immediate return |
| Covalent warhead penalty | -0.30 | Warhead detected by SMARTS |
| Out-of-domain penalty | -0.30 | OOD molecules also have activity x 0.50 |

### Run 1: sigma=0.5, reward hacking

With sigma=0.5 and no diversity filter, after 100 steps:

| Metric | Before RL | After RL |
|---|---|---|
| Validity | 93.0% | 99.4% |
| Uniqueness | 80.0% | **8.6%** |
| Scaffold diversity | 0.580 | **0.318** |
| Mean predicted pIC50 | 6.712 | **7.592** |
| ADMET pass rate | 31.2% | 80.6% |

The activity and ADMET numbers look impressive but are artefacts. The agent collapsed onto
approximately 14 Bemis-Murcko scaffolds, generating essentially the same molecule over and over
with small side-chain variations. Uniqueness fell from 80% to 8.6%. The pIC50 gain of 0.88 units
reflects the model generating repeatedly the highest-backbone-scoring quinazoline/indazole templates.
This is reward hacking, not improved chemistry.

The out-of-domain guard held (in-domain rate actually rose from 88.4% to 93.5%): the agent did
not escape the training distribution, it just collapsed to the densest part of it. Mode collapse
under the reward signal is a known failure mode of REINVENT when sigma is too large relative to
the diversity pressure.

### Run 2: sigma=0.25 + scaffold-memory diversity filter

A scaffold-memory filter (ScaffoldMemory, after Blaschke et al. 2020) maintains a running count
of generated Bemis-Murcko scaffolds. Once a scaffold's bucket fills (25 molecules scoring above
reward threshold 0.30), every new molecule sharing that scaffold has its reward replaced with 0.0.
This directly prevents the mode collapse seen in Run 1 by making high-exploitation of any one
scaffold unprofitable after 25 uses.

With sigma=0.25 and the diversity filter, after 50 steps:

| Metric | Before RL | After RL |
|---|---|---|
| Validity | 93.0% | 96.1% |
| Uniqueness | 80.0% | 67.7% |
| Scaffold diversity | 0.580 | **0.568** (held) |
| Mean predicted pIC50 | 6.712 | **6.716** (flat) |
| ADMET pass rate | 31.2% | 48.3% |

During training: 494 distinct scaffolds were explored, only 12 were saturated, and 412 molecules
were penalised by the diversity filter. Uniqueness stayed above the 60% floor. The diversity guard
worked. But once sigma was low enough to prevent hacking, the activity signal became too weak to
move the pIC50 needle: +0.004 units over 50 steps, below the 0.10 significance threshold.

**Verdict:** RL at this corpus size is either hacking or stalling. The activity gain requires
sigma high enough to cause mode collapse. The production generator remains `egfr_finetuned_gru.pt`.

The fundamental tension is that the backbone reward is in-sample: the backbone was trained on
all 1,347 molecules, and the generator has already been fine-tuned on the same set. There is no
meaningful unexploited region of chemical space the backbone scores highly that the fine-tuned
generator has not already found. RL can only exploit the backbone's in-sample biases. The path
forward is either a larger fine-tune corpus or a held-out or docking-based reward signal.

---

## 15. Applicability domain: knowing when not to trust the model

### The problem

The backbone RandomForest was trained on 1,347 molecules. When a new molecule is very different
from all training molecules, the model's prediction is essentially interpolating between distant
training points, or extrapolating entirely, and the prediction is unreliable.

### Max Tanimoto similarity

For each new molecule the applicability domain check computes its Tanimoto similarity (based on
ECFP4 fingerprints) to every molecule in the training set and reports the maximum. A molecule
that looks like something the model has seen (max Tanimoto >= 0.50) gets full confidence. A
molecule that looks like nothing in the training set (max Tanimoto < 0.30) gets half confidence.

| Band | Max Tanimoto | Confidence factor |
|---|---|---|
| in_domain | >= 0.50 | 1.00 |
| borderline | 0.30-0.50 | 0.75 |
| out_of_domain | < 0.30 | 0.50 |

The confidence factor is a multiplier on the composite score in the final ranking, not a hard
filter. An out-of-domain compound with a confidence factor of 0.50 still appears in the ranking,
but lower.

In the screening run (1,000 generated molecules at temperature 0.8), 90% of generated molecules
are in-domain: the fine-tuned generator stays close to the EGFR training distribution, as
expected. The 5 out-of-domain molecules represent genuinely novel structural motifs whose
predictions should be interpreted with caution.

---

## 16. Final integrated ranking

### The v2 composite formula

The final ranking combines four sources of evidence for each compound:

```
bioactivity_score = 0.30 x activity_norm
                  + 0.30 x selectivity_norm
                  + 0.20 x affinity_norm
                  + 0.20 x admet_norm

final_score = bioactivity_score x confidence_factor
```

Where:
- **activity_norm**: backbone-predicted pIC50, min-max normalised across all candidates
- **selectivity_norm**: normalised -(L858R - WT Vina score), so more L858R-selective means higher
- **affinity_norm**: normalised -(L858R Vina score), so tighter binding means higher
- **admet_norm**: QED score, min-max normalised
- **confidence_factor**: from applicability domain (1.0, 0.75, or 0.50)

Each component is independently normalised to [0, 1] across the candidate set so that the weights
apply to comparable scales. Covalent warhead and within-noise selectivity are surfaced as text
warnings alongside the score, not as silent score deductions.

### The 68-candidate field

49 known-library candidates (the docked top-50 minus cmpd_041 which timed out) plus 19 generated
candidates (20 were selected for docking, 1 failed PDBQT preparation) were ranked together.

**Top 5:**

| Rank | CID | Source | Final score | Notable |
|---|---|---|---|---|
| 1 | cmpd_015 | known | 0.730 | Clean ADMET, L858R-selective |
| 2 | cmpd_002 | known | 0.718 | Highest activity norm of clean shortlist |
| 3 | cmpd_008 | known | 0.697 | Best docking, COVALENT warning |
| 4 | cmpd_024 | known | 0.669 | Strongest selectivity delta, Brenk flag |
| 5 | cmpd_011 | known | 0.647 | COVALENT warning |

Best generated candidate: **gen_005 at rank 21** (final score 0.597). This is competitive: the
generated molecule lands immediately after the known top-20, demonstrating that the fine-tuned
generator can produce credible, drug-like, L858R-selective candidates. The known compounds dominate
the top because they were selected as the highest backbone-predicted actives in the first place.

---

## 17. The serving layer: FastAPI, Streamlit, and Docker

### FastAPI backend (Phase 24)

The FastAPI service (`src/api/`) loads all precomputed model artifacts once at startup and
serves them via four endpoints:

- `GET /health`: returns `status: ok` when all artifacts are loaded
- `POST /predict`: fast screen for a single SMILES string
- `POST /batch_predict`: screen up to 512 SMILES in one request
- `GET /model-info`: model versions, algorithms, score definitions, and caveats

The `/predict` response includes backbone pIC50, WT-proxy pIC50, selectivity proxy (pIC50
difference, labelled exploratory), covalent flag, ADMET status (QED, SA, Brenk, PAINS),
applicability domain band and confidence factor, and structured warnings.

**What the API does not do:** docking is not computed at request time. AutoDock Vina takes 30-120
seconds per compound per pocket. The API is designed for fast pre-screening (milliseconds per
compound), and docking is a separate offline pipeline step. The `docking_selectivity_available`
field in every response is always `false`.

### Streamlit dashboard (Phase 25)

Seven pages:

1. **Single molecule:** type or paste a SMILES, see the full fast screen result
2. **Batch screening:** upload a .smi/.txt/.csv file, screen up to 512 molecules
3. **Final ranking:** the 68-candidate ranked table with Altair bar charts coloured by source
4. **Model performance:** R^2 and RMSE tables, error bars, fingerprint ablation chart
5. **Docking results:** sanity check table, noise error bars for the top-15
6. **Limitations:** a plain-language statement of what the pipeline cannot do
7. **Discovery Copilot:** a chat interface to the deterministic orchestration layer (see section 18)

The dashboard first tries the FastAPI service; if the API is not reachable it falls back to scoring
locally via the same ModelRegistry. The sidebar shows which mode is active.

### Docker (Phase 26)

Two images, both from python:3.12-slim, share a common set of ML and chemistry dependencies
(`requirements/serving.txt`). PyTorch, LightGBM, Optuna, MLflow, Vina, and GNINA are all excluded
from the serving images (training is offline, serving needs only sklearn, XGBoost, and RDKit).
XGBoost 3.x on Linux pulls in an NVIDIA NCCL library (303 MB) even for CPU-only use; this is
uninstalled in the same Docker layer to keep image sizes manageable (API: 1.19 GB, dashboard: 1.53
GB).

Model artifacts are bind-mounted read-only from the host (`./models` and `./data`), not baked
into the images. This means the same Docker images can be used with different trained models
without rebuilding.

```bash
docker compose up --build   # http://localhost:8000 (API) + http://localhost:8501 (dashboard)
```

---

## 18. Discovery Copilot: a deterministic query and explanation layer

### Why it exists

By the end of section 17 the pipeline has many moving parts: four models, three docking JSON
files, ADMET results, applicability domain bands, and a 68-candidate ranking table. Answering a
question like "which of cmpd_015 and cmpd_024 should I prioritise, and why?" requires
cross-referencing multiple JSON and CSV files and interpreting them together.

The Discovery Copilot (`src/agent/`, `src/dashboard/copilot_page.py`) routes such queries to
the appropriate precomputed-artifact tools, assembles the answer, and shows the reasoning. It does
not make the science more certain. It makes the pipeline easier to run, audit, and explain without
strengthening any claim.

### Deterministic-first design

No LLM is wired in v1. The controller (`src/agent/controller.py`) classifies the query by keyword
matching into eight intent classes in priority order: `report`, `comparison`, `docking_query`,
`batch_predict`, `candidate_lookup`, `single_predict`, `project_qa`, `unknown`. It then calls the
relevant tool functions and assembles the structured output directly. An LLM hook
(`src/agent/prompts.py`) returns None in v1 and can be wired to any API for natural-language
reformatting without changing any other module.

Tools compute; the orchestration layer explains. Every number in the answer comes from a file on
disk, not from a language model.

### The six tools (`src/agent/tools.py`)

| Tool | What it reads |
|---|---|
| `predict_smiles` | Runs backbone (Model 1) and WT-proxy (Model 2) on any SMILES |
| `batch_predict` | Same for up to 512 SMILES; invalid rows are included, not dropped |
| `lookup_final_ranking` | `data/generated/final_ranked_candidates.csv` (68 candidates) |
| `lookup_docking_results` | Three docking JSON files merged in priority order; never fabricates a score |
| `compare_candidates` | Ranks two or more candidates by a conservative score (non-covalent, in-domain, ADMET, final score, noise-study call) |
| `generate_candidate_report` | Calls the above tools and assembles a seven-section markdown document |

### Guardrails (`src/agent/guardrails.py`)

Two mechanisms:

**Scientific warning injection** (`add_scientific_warnings`): appends caveats to any result that
has a selectivity proxy (ML-derived, not statistically validated at n=9), a docking score
(rigid receptor), a covalent flag, or an out-of-domain label.

**Forbidden-claim sanitizer** (`find_forbidden_claims`): scans the assembled answer for phrases
that assert experimental evidence the pipeline does not have. Forbidden phrases when not preceded
by a negation within 30 characters: "is active", "is selective", "drug candidate", "validated",
"proven", "confirmed". A negation before the phrase causes it to be ignored: "not validated
experimentally" passes; "binding was confirmed by SPR" is flagged.

Selectivity labels are enforced in output: ML-derived pIC50 difference is always labeled "ML
proxy, exploratory"; docking-based delta is always labeled "structure-based (docking)".

### Three-panel Streamlit UI

```
Discovery Copilot
  [5 example-prompt buttons]
  ───────────────────────────────────────────────
  user:      compare cmpd_015 and cmpd_024
  assistant: Panel 1  grounded answer (markdown, recommendation + reason)
             Panel 2  Evidence [collapsible]: tools called and what they returned
             Panel 3  Warnings [collapsible]: guardrail caveats
             [Download report as Markdown]  (appears when a report is generated)
  ───────────────────────────────────────────────
  [chat input: "Ask the Discovery Copilot..."]
```

The registry is loaded lazily: `_registry()` is invoked only when a query is submitted, so the
cold-start page render is near-instant. No API key is required. No LLM is called in v1.

Every candidate report includes a Limitations section by template, regardless of what the query
asks. The copilot does not run docking, train models, or update any artifact at query time.

### Tests

119 tests across six files: `test_agent_tools.py`, `test_agent_guardrails.py`,
`test_agent_retrieval.py`, `test_agent_report.py`, `test_agent_controller.py`,
`test_agent_copilot.py`. 773 total unit tests.

---

## 19. Follow one molecule: gefitinib and gen_005

This section traces two molecules through every pipeline step with the real numbers produced
by this codebase. Gefitinib is a well-known clinical drug present in the training data.
gen_005 is the best de novo generated hit, a new molecule that did not exist in the input.

---

### Gefitinib

**SMILES:** `COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1`

Gefitinib is a first-generation EGFR inhibitor approved for NSCLC in 2003. It is a quinazoline
(the two fused rings containing nitrogen), which is the canonical scaffold for EGFR inhibitors.
The fluorine and chlorine on the aniline ring and the morpholine side chain are the structural
features that distinguish gefitinib from other quinazolines.

#### Step 1: Raw data

Gefitinib appears in the ChEMBL EGFR download with molecule ID CHEMBL939. Its mutation flag in
the raw data is "unknown", meaning the assay did not specify which EGFR construct was used. The
measured pIC50 is 6.29, which corresponds to an IC50 of roughly 510 nM, a moderate-potency value.

After cleaning: canonical SMILES matches exactly, heavy atom count is within limit, it is assigned
to the backbone training set (mutation_flag=unknown, so included in Model 1 and the WT-proxy
training sets).

#### Step 2: Feature vector

The Morgan ECFP4 fingerprint (radius 2, 2,048 bits) captures the quinazoline core, the aniline
substituent, and the morpholine side chain as bit patterns. RDKit computes:
- MW 446 g/mol (within Lipinski limit of 500)
- LogP 4.28
- TPSA 68.7 Angstrom squared
- HBD 1, HBA 7
- Rotatable bonds 8
- QED 0.518

Final feature vector: 2,048 + 11 = 2,059 values.

#### Step 3: Scaffold split assignment

Gefitinib's Bemis-Murcko scaffold is the quinazoline with the aniline substituent. Because it is
in the training set for both the backbone (Model 1) and the WT-proxy (Model 2), its scaffold
is in-domain for all training-time evaluations.

#### Step 4: Model predictions

- Backbone (Model 1, RandomForest): **pred pIC50 = 6.936**
- WT-proxy (Model 2, XGBoost): **pred pIC50 = 6.988**
- Selectivity proxy: 6.936 - 6.988 = **-0.052** (effectively neutral, within model noise)

The backbone slightly overpredicts relative to the measured 6.29. This is expected: in-sample
predictions from ensemble models tend to regress toward the mean.

#### Step 5: ADMET

Status: **flag**. One Brenk alert: "Aliphatic_long_chain" (the propyl-morpholine chain).
All other properties are within limits (Lipinski pass, Veber pass, no PAINS). QED=0.518, SA=2.34.
No covalent warheads detected.

This flag is mild. The Brenk alert is a metabolic liability concern, not a toxicity alert.
Gefitinib is a clinical drug precisely because its real-world ADMET profile was acceptable despite
this computational flag.

#### Step 6: Applicability domain

Max Tanimoto against the training set: **1.000**. Gefitinib is in the training set, so there is
an identical molecule with similarity 1.0. Band: **in_domain**. Confidence factor: **1.0**.

#### Step 7: Docking (sanity check)

Gefitinib was docked into both structures as part of the Phase B2 sanity check:
- L858R (2ITZ): **-7.860 kcal/mol**
- WT (2ITY): **-7.492 kcal/mol**
- Delta: **-0.368 kcal/mol** (L858R favoured)

The negative delta confirms the expected selectivity direction. The magnitude (0.37 kcal/mol)
is smaller than the 1.7 kcal/mol expected from a 20-fold affinity difference because the rigid
receptor cannot capture the full induced-fit effect.

GNINA CNN rescoring:
- L858R: CNNaffinity 6.693 pKd
- WT: CNNaffinity 6.389 pKd
- CNN delta: +0.303 pKd (L858R favoured)

Both scoring functions agree: gefitinib prefers the L858R pocket.

#### Step 8: Final ranking

Gefitinib is not in the top-50 backbone-predicted candidates (predicted pIC50 6.94 vs the
shortlist range of 8.4-9.3) and was therefore not included in the library docking step. It
appears only in the sanity check, serving as the anchoring reference point for the entire docking
analysis.

If gefitinib were included in the final ranking purely from the fast-screen evidence:
- Activity score (pIC50 6.936) would place it well below the known shortlist
- Selectivity proxy (-0.052) is flat
- ADMET: flag (Brenk)

It would rank in the lower half of the 68-candidate field. The clinical drug in this cohort is
not a top computational hit, because the computational models were trained on in-sample predictions
and the backbone predicts gefitinib's pIC50 as moderate (which it is: 6.29 measured). This is
an honest result: the pipeline is not gaming known drug scores.

---

### gen_005

**SMILES:** `COc1cc2ncnc(Nc3cccc(Cl)c3F)c2cc1OC`

gen_005 is a methoxy-quinazoline generated by the fine-tuned GRU at temperature 0.8. It was not
in the training data. Its structure is a simplified quinazoline: the core scaffold is the same
as gefitinib (quinazoline with aniline), but the morpholine side chain is replaced with nothing,
and the aniline substituents are rearranged (the chloro and fluoro groups are at different
positions, and there is no other side chain).

#### Step 1: Generation

The GRU generates SMILES one character at a time. gen_005 was sampled during
`scripts/dock_generated_candidates.py` which draws 2,000 molecules from the fine-tuned checkpoint
at temperature 0.8, filters for validity, uniqueness, novelty, ADMET pass, in-domain status, and
non-covalent structure, then selects the top 20 by backbone-predicted pIC50 for docking. gen_005
passed all filters and was selected for docking.

#### Step 2: Validation and novelty

RDKit confirms the SMILES is valid. Canonical SMILES is unique (not in the EGFR/ErbB2 training
set). gen_005 is genuinely novel.

#### Step 3: Feature vector and ADMET

- MW **333 g/mol** (well below Lipinski limit)
- LogP **4.18**
- TPSA **56.3** Angstrom squared
- HBD 1, HBA 5
- Rotatable bonds 4
- QED **0.776** (good drug-likeness)
- SA score **2.11** (synthetically very accessible)
- No PAINS alerts, no Brenk alerts
- No covalent warheads
- ADMET status: **pass**

The low rotatable bond count (4 vs gefitinib's 8) and smaller molecular weight make gen_005 more
drug-like by most computational criteria.

#### Step 4: Applicability domain

Max Tanimoto against the training set: **0.712**. The nearest training molecule has 71.2%
fingerprint similarity. Band: **in_domain** (above the 0.50 threshold). Confidence factor: **1.0**.

The similarity of 0.712 reflects that gen_005 has the same quinazoline core as many known EGFR
inhibitors, but its specific substitution pattern and the absence of a side chain make it distinct
enough that it is not a near-duplicate of any training compound.

#### Step 5: Model predictions

- Backbone (Model 1, RandomForest): **pred pIC50 = 8.102**
- WT-proxy (Model 2, XGBoost): **pred pIC50 = 8.630**
- Selectivity proxy: 8.102 - 8.630 = **-0.528** (moderate L858R preference by ML proxy)

The backbone prediction of 8.102 corresponds to an IC50 of approximately 8 nM if the prediction
were accurate. This should be treated with caution: the model has not seen this exact compound
during training, so the prediction is interpolated from similar training examples. The applicability
domain check confirmed in-domain status, so the interpolation is likely reasonable, but it has
not been validated experimentally.

#### Step 6: Docking

gen_005 was docked into both structures:
- L858R (2ITZ): **-8.016 kcal/mol**
- WT (2ITY): **-7.586 kcal/mol**
- Delta: **-0.430 kcal/mol** (L858R favoured)

The L858R Vina score of -8.016 is stronger than gefitinib's -7.860 in the same pocket, which is
consistent with the higher predicted pIC50. The selectivity delta of -0.430 is within the range of
the confirmed non-covalent shortlist compounds (cmpd_002 was -0.342, cmpd_048 was -0.430).

#### Step 7: Final composite score

With all components:
- activity_norm: 0.294 (backbone 8.102, normalised across all 68 candidates)
- selectivity_norm: 0.721 (delta -0.430, normalised)
- affinity_norm: 0.478 (L858R -8.016, normalised)
- admet_norm: 0.983 (QED 0.776, normalised)
- confidence_factor: 1.00 (in_domain)

```
bioactivity_score = 0.30 x 0.294 + 0.30 x 0.721 + 0.20 x 0.478 + 0.20 x 0.983
                  = 0.088 + 0.216 + 0.096 + 0.197
                  = 0.597
final_score       = 0.597 x 1.00 = 0.597
```

**Final rank: 21 out of 68 candidates.** No warnings.

The activity component (0.294 normalised) is lower than the known shortlist compounds because
the known compounds were selected as the top-50 backbone predictions: cmpd_002 has activity_norm
0.900. gen_005 compensates with strong selectivity (0.721) and excellent ADMET (0.983), landing
immediately after the known top-20. This is the honest read: a generated novel compound can
compete with the best known candidates on selectivity and drug-likeness, but cannot beat in-sample
activity predictions.

---

## 20. Limitations

These are not caveats buried at the end. They are central to interpreting every number in this
project.

**Data scarcity is structural, not fixable by better algorithms.** 22 labelled L858R records is
genuinely insufficient for a standalone model. The LOOCV result (backbone beats calibration) is
not a failure of calibration methods; it is the correct conclusion at n=22.

**All QSAR predictions are in-sample for the known library.** The top-50 candidates were selected
as the highest backbone-predicted actives, and the backbone was trained on those same compounds.
Their predicted pIC50 values are not held-out estimates. A compound selected for synthesis on the
basis of these predictions would likely show lower potency than predicted.

**Selectivity cannot be modelled at n=9.** The derived selectivity Spearman r=0.433, p=0.244 is
not statistically significant. The selectivity proxy in the fast API screen is labelled exploratory
for this reason.

**Rigid receptor docking underestimates affinity differences.** The 20-fold gefitinib selectivity
from Yun et al. (1.7 kcal/mol) is underestimated to 0.37 kcal/mol by rigid Vina. The direction is
correct for the known inhibitors; the magnitude is not quantitatively reliable.

**Dual-pocket comparability is imperfect.** 2ITZ (2.8 Angstrom resolution) and 2ITY (3.42
Angstrom) have slightly different crystal packing. Docking score differences reflect both the
mutation effect and differences in how the scoring function handles the two pocket geometries.

**RL activity gains were mode collapse artefacts.** Run 1's mean pIC50 increase of +0.88 units
came from collapsing to 14 scaffolds, not from discovering better chemistry. Run 2 did not improve
activity. The production generator is the fine-tuned model without RL.

**Generated candidates are doubly exploratory.** They are scored by an in-sample backbone. Their
docking was performed with the same rigid-receptor limitations as the library screen.

**No experimental data.** No compound in this pipeline has been synthesised or tested in a
biochemical assay, cell line, or animal model. Every number is computational.

---

## 21. Future work

The negative results and limitations above point directly to the next steps. Phase 27
(Discovery Copilot, a deterministic query and explanation layer, `src/agent/`) has been
implemented; see section 18.

**More L858R data.** The most important thing. Systematic mining of patent databases (Google
Patents, Espacenet) and preprint servers for L858R binding data, combined with FEP-annotated
virtual screening results from an existing L858R co-crystal library, could grow the labelled set
from 22 to 100+, which would change what is possible in LOOCV calibration.

**GPU pretraining for the generator.** The base GRU was pretrained on 80,000 molecules for 6
epochs on CPU (the training hung before epoch 8). A GPU run on the full 150,000 MOSES molecules
for 15+ epochs would raise the temperature-1.0 validity ceiling from 81% to >90%, allowing hotter
sampling for more diverse generation.

**Docking-based or FEP-based reward for RL.** The fundamental problem with the backbone reward in
RL is that it is in-sample. Using Vina scores (or FEP estimates) as the RL reward signal would
provide held-out evidence that is not gameable by exploiting the backbone's training distribution.
Each RL step would require two Vina runs per molecule, roughly 1-2 minutes of compute per compound,
so this requires either GPU-accelerated docking or a surrogate model for the docking score.

**Free energy perturbation (FEP).** Relative binding free energy calculations (FEP+, RBFE) on the
confident docking shortlist would provide quantitative affinity estimates with physical accuracy
that rigid Vina cannot. FEP is the gold standard for computational selectivity estimation; the
docking shortlist of 4-6 compounds is the right input size for an FEP campaign.

**T790M Model 4.** After correctly relabelling the 130 compound-mutant records in the T790M bucket,
81 genuine T790M single-mutant records remain. This is enough for a standalone model (similar scale
to the WT-proxy), and T790M selectivity is an important clinical question.

**Cloud deployment.** The Docker images are production-ready for deployment on any cloud provider.
The two main artifacts that are missing for production are: GCS or S3 model storage (the `USE_GCS`
flag in the API is already wired, just needs credentials), and a CI/CD pipeline that rebuilds and
pushes the Docker images when new training artifacts are committed.

---

*Document generated from artifact data (June 2026). Real numbers sourced from:
`models/qsar/*/metadata.json`, `models/qsar/l858r/loocv_results.json`,
`models/qsar/selectivity/selectivity_results.json`,
`models/qsar/docking_noise_results.json`, `models/qsar/sanity_check_docking.json`,
`models/qsar/gnina_rescore_sanity.json`, `models/qsar/admet_results.json`,
`models/generator/rl_results.json`, `data/generated/final_ranked_candidates.csv`,
and live ModelRegistry predictions for gefitinib and gen_005.*
