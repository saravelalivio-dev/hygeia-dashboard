# HYGEIA — Interactive Research Dashboard

**An Interpretable Deep Learning and Multi-Objective Pipeline for *De Novo*
Antimicrobial Peptide Design targeting *Klebsiella pneumoniae***

`app.py` is a four-tab Streamlit companion to the HYGEIA proposal. Every
descriptor, attribution, objective and elastic-network metric is recomputed
from sequence or coordinates inside the app — nothing is hard-coded except the
published reference values used for cross-checking.

| Tab | Contents |
|-----|----------|
| ① Generation & Gate Screening | Template-primed sampling, temperature scaling, descriptor panel (Henderson–Hasselbalch charge, μH, GRAVY, aliphatic index, pI), Random-Forest gate card (`p_active`, MCC/AUROC/AUPRC + confusion-matrix metrics, Youden's J threshold) |
| ② Explainable AI (SHAP) | Exact Shapley values over all 2⁹ descriptor coalitions: waterfall, φ<sub>j</sub> bars, global mean \|φ\|, directional beeswarm, Pearson/Spearman correlation table |
| ③ Pareto Optimization (ACPE) | 4-objective non-dominated front (maximise activity & stability, minimise toxicity & hemolysis or steric penalty), 2-D/3-D scatter, parallel coordinates, exact hypervolume, ideal-point ranking; `Peptide_03` and `Peptide_07` highlighted by default |
| ④ Structural Dynamics & Continuum Mechanics | Elastic-network Kirchhoff matrix, normal-mode analysis, B-factor & deformability profiles, dynamic cross-correlation map, harmonic stiffness tensor (continuum fabric tensor), 3-D docked complex, HPEPDOCK docking summary (OmpA 2K0L / OmpW 8QXP / LptDE 5IV9) |

## Run locally

```bash
pip install -r requirements-app.txt     # slim set; requirements.txt has the full pipelines
streamlit run app.py
```

## Verification against the published tables

```bash
python verify_hygeia.py
```

Recomputes proposal Tables 5, 6, 9 and 10 from sequence alone and prints a
pass/fail verdict. Current status: **ALL CHECKS PASS** — charges and pI match to
≤0.004, GRAVY and aliphatic index to ≤0.001, toxicity/hemolysis to ≤0.001, and
the Pareto membership (Peptide_01–07 non-dominated, 08/09 dominated) is exact.

Two conventions are worth noting, both documented in `app.py`:

* `KD["T"] = -0.7` — the value `feature_extraction.py` and published Table 6
  use (the 1982 paper prints −0.9).
* `INCLUDE_C_TERM = False` — the published numbers were produced by the RDKit
  titration in `amp_generation_25_pepfoundry_rdkit.py`, which left the
  C-terminal carboxyl uncounted. Excluding it reproduces all nine Table 5
  charges and all eight Table 6 pI values exactly; including it misses by up
  to 0.72. Peptide_02 has no acidic residue at all, so its charge never
  crosses zero and its pI is reported as *not recorded* — matching the
  proposal's "eight candidates with a recorded isoelectric point".

## Data

`app.py` trains its gate on `kpneumoniae_features_math.csv` (619) plus
`sample_data.csv` (1125) — 1698 curated sequences after filtering to the
10–30-residue design window. If neither file is present the app falls back to
a small embedded set, so a partial checkout still starts.

## Deploy

See [Streamlit Community Cloud](https://share.streamlit.io) — point the app at
`app.py` and the *Requirements file* field at `requirements-app.txt`.
