# -*- coding: utf-8 -*-
"""
Klebsiella pneumoniae antimicrobial peptides
PepFoundry graph features + RDKit 2D descriptors -> leakage-free ML -> TreeSHAP.

Single-file version of Klebsiella_AMP_PepFoundry_RDKit_ML_Colab.ipynb
(44 cells concatenated in order; markdown became comments).

Colab:
    from google.colab import files
    up = files.upload()                        # pick this .py
    exec(compile(open(next(iter(up)), encoding="utf-8").read(), "pipeline", "exec"))

Local:
    python Klebsiella_AMP_PepFoundry_RDKit.py
"""



# ============================================================================
# CELL 00 (markdown, commented out)
# ============================================================================
# # Antimicrobial Peptide Classification against *Klebsiella pneumoniae*
# ## PepFoundry graph features + RDKit 2D descriptors  ->  Machine Learning  ->  TreeSHAP
#
# **Dataset** `natural-modified-combined.csv` - natural + modified/synthetic peptides, ~600 rows, binary `Active` / `Inactive` MIC labels.
#
# | Stage | Proposal | Content |
# |---|---|---|
# | 0 | - | Environment (Colab) |
# | 1 | - | **Which ML family?** Options, trade-offs and a recommendation for n ~ 600 |
# | 2 | 2.2 | Data loading + cleaning (mojibake units, non-canonical residues, conflicting duplicate rows) |
# | 3 | 2.2.2 | Leakage control: exact dedup + CD-HIT style 90% identity clustering |
# | 4 | **2.3.1** | Learned stream - **PepFoundry** atom/bond tensor features (24 + 16 dims), pooled and cached |
# | 5 | **2.3.1** | Deterministic stream - **RDKit** 2D descriptors on the PepFoundry-built molecule |
# | 6 | **2.3.1** | AMP-specific peptide descriptors (net charge, hydrophobic moment, amphipathicity) from the same molecule |
# | 7b | **2.3.2** | **Physicochemical descriptors assigned onto the dataset**, row by row, id-verified |
# | 8 | **2.4.1** | Feature matrix construction: `X = [g_PepFoundry , p_RDKit]` + `feature_manifest.json` |
# | 9 | **2.4.3** | Benchmarking suite - the proposal's six architectures plus extras, as leakage-free Pipelines |
# | 10 | **2.4.2 / 2.4.3** | Train/test ladder 100/0 -> 10/90, 5 seeds; metrics Acc, **MCC**, **AUROC**, AUPRC, F1, Sens, Spec, BACC |
# | 11 | **2.4.3** | Hyper-parameter search (`n_estimators`, `max_depth`, `min_samples_split`, `learning_rate`) + 5-fold CV |
# | 12 | **2.4.4** | Master screening gate, 5-fold CV, **Youden's index** threshold, bootstrap CIs, headline table |
# | 13 | **2.4.4** | Post-hoc TreeSHAP: physicochemical beeswarm, graph block, **per-residue** attribution, SHAP pattern table |
# | 14 | **2.5.4** | Artefacts, `predict_peptides()`, `score_candidates()` and the 2.5.3 novelty reference |
# | 15 | extension | (Optional) GIN graph neural network on the PepFoundry atom graphs |
#
# > **Why PepFoundry + RDKit instead of ESM-2 + ProtParam?** The proposal's 2.3.1 asks for two parallel streams - a pretrained learned representation plus deterministic physicochemical descriptors. PepFoundry (Garzon Otero *et al.*, *JCIM* 2026, `doi:10.1021/acs.jcim.5c02629`) is the pretrained model for the first stream: it builds atom-mapped RDKit molecules from peptide sequences through a 157-entry CHUCKLES library, so it handles natural, D-, and non-canonical residues with the same code path, and it returns 24-d atom and 16-d bond features. RDKit supplies the second stream, with descriptors for MW, net charge, hydrophobic moment, amphipathicity and the aliphatic-index family that 2.4.1 names - computed on a real molecule, not a string. The paper's central claim - atomic-level representations beat sequence-level ones for modified peptides - is exactly what this dataset needs, and it removes the ESM-2 tokeniser's canonical-only restriction, so the modified block of the dataset stays in.
# >
# > **Runtime tip** Colab > Runtime > Change runtime type > **T4 GPU** (optional: only the section 15 GNN needs it). Sections 4-6 are pure CPU and take about a minute for 600 peptides.
# >
# > **Headline caveat** this dataset contains the *same peptide* listed twice and a few sequences labelled both Active and Inactive. Cell 3 fixes that; without it every metric below is inflated by several MCC points.


# ============================================================================
# CELL 01
# ============================================================================
# @title 0. Install packages
import sys, subprocess, importlib


def _pip(*pkgs):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *pkgs], check=False)


_pip('pandas', 'numpy', 'scipy', 'scikit-learn', 'matplotlib', 'seaborn', 'biopython', 'joblib', 'openpyxl')
_pip('rdkit')                       # 2D descriptors + molecule objects
_pip('openbabel-wheel')             # PepFoundry imports it at module level
_pip('git+https://github.com/BilodeauGroup/PepFoundry.git')   # not on PyPI
_pip('torch')                       # PepFoundry returns torch tensors (CPU is enough here)
_pip('xgboost', 'lightgbm', 'catboost')
_pip('shap', 'optuna')

for m in ('numpy', 'pandas', 'sklearn', 'Bio', 'torch', 'shap', 'xgboost', 'lightgbm'):
    try:
        mod = importlib.import_module(m)
        print(f'{m:12s} OK   {getattr(mod, "__version__", "?")}')
    except Exception as e:
        print(f'{m:12s} FAIL {e}')

try:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, rdMolDescriptors
    print('rdkit        OK  ', rdkit.__version__)
except Exception as e:
    print('rdkit        FAIL (install it: pip install rdkit):', e)

try:
    import pepfoundry
    from pepfoundry.interface import PepFoundry
    print('pepfoundry   OK  ', getattr(pepfoundry, '__version__', '2.0.0 (github)'))
    print('-> building the 157-residue library takes ~1 min on first import')
except Exception as e:
    print('pepfoundry   FAIL (install it: pip install git+https://github.com/BilodeauGroup/PepFoundry.git):', repr(e)[:200])


# ============================================================================
# CELL 02 (markdown, commented out)
# ============================================================================
# # 1. Which machine-learning approach? (n ~ 600, d = 230)
#
# Two feature blocks are built from the same PepFoundry molecule: **RDKit** (~70 interpretable 2D descriptors) and **PepFoundry** (160 pooled atom/bond graph features). With ~600 samples and p >> n, the **bias-variance trade-off is the whole game**: anything with high capacity memorises the training set and collapses MCC on held-out data. The families below are ordered from *safest* to *most powerful* for this regime.
#
# ## 1.1 Options
#
# | # | Family | Concrete models | Why it fits n ~ 600 | Expected MCC / AUROC | Risk |
# |---|---|---|---|---|---|
# | A | **Regularised linear** | Logistic Regression (L2 / ElasticNet), Linear SVM | n << d is exactly the regime L2-regularised linear models were designed for; lowest variance; calibrated probabilities so AUROC is trustworthy | 0.45-0.60 / 0.80-0.88 | under-fits long-range position effects |
# | B | **Kernel SVM** | SVC (RBF) on scaled features | usually the strongest *classical single model* in small-n AMP/QSAR benchmarks; tolerates p >> n | 0.55-0.70 / 0.85-0.92 | needs `C` / `gamma` tuning, slow, probabilities need Platt scaling |
# | C | **Tree ensembles** (the proposal) | **Random Forest**, ExtraTrees, GradientBoosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost | non-linear, no scaling needed, native TreeSHAP, robust against wide feature blocks | 0.55-0.70 / 0.85-0.92 | at d = 160 the graph block still favours noise, so it **must** be compressed (SVD/PLS) first |
# | D | **Discriminant / naive Bayes** | LDA with shrinkage, GaussianNB, ComplementNB, PLS-DA | almost no overfitting, seconds to fit, a defensible baseline row | 0.35-0.55 / 0.75-0.87 | weak non-linear capacity |
# | E | **Instance based** | k-NN (k = 5-15) on reduced features | interpretable, zero training cost, answers "does this resemble a known active peptide?" | 0.30-0.50 / 0.65-0.80 | collapses on the 10/90 split |
# | F | **Neural** | MLP, 1-2 hidden layers, heavy weight decay + early stopping | small enough for 600 rows if regularised; mirrors the head used for the GNN option | 0.50-0.65 / 0.83-0.90 | unstable on very small training fractions |
# | G | **Ensembles (expected best)** | soft Voting / Stacking of SVC-RBF + RF/ExtraTrees + XGB/LGBM + LogReg | averaging decorrelated errors usually gives the best *and most stable* MCC and AUROC; a stacking meta-learner on out-of-fold probabilities is the natural top model at n ~ 600 | **0.60-0.75 / 0.88-0.94** | several times the compute; must be trained on OOF predictions |
# | H | **Graph neural network** | GIN / GCN on the PepFoundry atom graph (24-d node, 16-d edge features) | the only family that sees the *connectivity* rather than a pooled summary; the PepFoundry paper's own claim is that atomic-level input beats sequence-level input | 0.60-0.80 / 0.88-0.95 | needs GPU, high seed variance - only defensible with repeated nested CV (section 15) |
#
# ## 1.2 Recommendation for this dataset
#
# 1. **Core results table = A, C, G** (LogReg, SVC-RBF, RF, ExtraTrees, XGBoost/LightGBM, soft-voting ensemble). Keep D as a sanity baseline, E as a 90/10 vs 10/90 sensitivity check, F optional, H only as a clearly labelled extension.
# 2. **Compress the graph block inside the CV loop** (`TruncatedSVD`/`SafeSVD`, 32-128 components) for tree models; give the full 160-d vector to SVC/LogReg after `StandardScaler`. Never fit the SVD on the whole dataset - that is leakage.
# 3. **Report the ablation** RDKit vs PepFoundry vs Combined. RDKit alone (d << n) is often within noise of the graph block; if so, the cheap interpretable model is the more defensible one.
# 4. **Optimise MCC, not accuracy.** On this dataset accuracy saturates near 0.85 and hides failure. Use MCC for model selection and report AUROC + AUPRC + MCC + balanced accuracy + Sens/Spec.
# 5. **De-duplicate and cluster-group the splits.** The file holds exact duplicate sequences with *conflicting* labels and near-identical clusters of up to 4 members. The notebook evaluates random **and** cluster-grouped splits so the published numbers survive review.
# 6. **Never tune on the test set.** All hyper-parameter search happens by 5-fold CV *inside* the training portion; the outer split is scored once.


# ============================================================================
# CELL 03
# ============================================================================
# @title 2. Configuration
import os, json, math, random, warnings, time, hashlib
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 200)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DRIVE_ROOT = '/content/drive/MyDrive'
WORKDIR = (os.path.join(DRIVE_ROOT, 'amp_klebsiella_pepfoundry')
           if os.path.isdir(DRIVE_ROOT) else '/content/amp_klebsiella')
os.makedirs(WORKDIR, exist_ok=True)
print('artefacts go to', WORKDIR, '(Drive)' if WORKDIR.startswith(DRIVE_ROOT) else '(session only)')

# ---------------- data ----------------
CSV_CANDIDATES = [
    '/content/drive/MyDrive/Klebsiella pneumoniae/natural-modified-combined.csv',
    '/content/drive/MyDrive/natural-modified-combined.csv',
    '/content/natural-modified-combined.csv',
    'natural-modified-combined.csv',
]

# ---------------- leakage control ----------------
DEDUP_EXACT = True             # collapse identical sequences
DROP_LABEL_CONFLICTS = True    # drop sequences recorded both Active and Inactive
CLUSTER_IDENTITY = 0.90        # CD-HIT style identity for near-duplicate groups
BUILD_CLUSTERS = True
GROUP_AWARE_SPLIT = True       # a cluster never straddles the train/test border

# ---------------- PepFoundry (section 2.3.1) ----------------
PF_CHIRALITIES = False          # include chiral tags in the atom features
PF_POOL_STATS = ('mean', 'std', 'max', 'sum')   # -> 4 x 24 atom + 4 x 16 bond features
PF_DEVICE = 'cpu'              # feature construction is cheap; 'cuda' is also fine
PF_KEEP_MOLS = False           # keep the RDKit molecules in memory (needed for inference demos)

# ---------------- RDKit (section 2.3.2) ----------------
USE_RDKIT = True
USE_PEPFOUNDRY = True
FORCE_RECOMPUTE = False        # True -> ignore the feature caches in cells 5.1 and 6.1
SVD_COMPONENTS = 128           # compression of the graph block for tree models
SHOW_PCA = True

# ---------------- evaluation ----------------
SPLIT_PAIRS = [(1.00, 0.00), (0.90, 0.10), (0.80, 0.20), (0.70, 0.30), (0.60, 0.40),
               (0.50, 0.50), (0.40, 0.60), (0.30, 0.70), (0.20, 0.80), (0.10, 0.90)]
# NOTE: the 8th ratio in the proposal was written 30/40; it is read here as 30 train / 70 test.
SEEDS = [42, 7, 123, 2024, 99]         # full evaluation
SCREEN_SEEDS = [42, 7, 123]            # cheaper model screening
CV_FOLDS = 5
CV_REPEATS = 2                         # only used by the 100/0 (no test set) case
TUNING_N_ITER = 40                     # randomized search iterations per model
PRIMARY_METRIC = 'mcc'                 # what the hyper-parameter search maximises
N_BOOTSTRAP = 2000
CV_REPORT_BLOCKS = ['RDKit', 'PepFoundry', 'Combined']

print('config ok | workdir:', WORKDIR)
print('n splits:', len(SPLIT_PAIRS), '| seeds:', SEEDS)


# ============================================================================
# CELL 04 (markdown, commented out)
# ============================================================================
# # 3. Load the dataset
#
# Put the CSV in Google Drive (for example `MyDrive/Klebsiella pneumoniae/`) and run the mount cell, or use the upload fallback. The file holds two stacked blocks - natural peptides with CD-HIT cluster columns, then DBAASP synthetic/modified peptides - separated by one empty row.


# ============================================================================
# CELL 05
# ============================================================================
# @title 3.1 Mount Google Drive (or upload the CSV)
from google.colab import drive, files
import os

drive.mount('/content/drive')


def resolve_csv(candidates):
    for p in candidates:
        if os.path.exists(p):
            print('found:', p)
            return p
    print('none of the configured paths exist -> falling back to the file uploader')
    up = files.upload()
    for name in up:
        p = os.path.join('/content', name)
        print('uploaded:', p)
        return p
    raise FileNotFoundError('CSV not found and nothing was uploaded')


CSV_PATH = resolve_csv(CSV_CANDIDATES)


# ============================================================================
# CELL 06
# ============================================================================
# @title 3.2 Raw load and inspection
raw = pd.read_csv(CSV_PATH, encoding='latin-1', encoding_errors='replace')
raw.columns = [str(c).strip() for c in raw.columns]

print('shape:', raw.shape)
print('columns:', list(raw.columns))
print()
print(raw.head(3).to_string())
print()
for c in ['Activity', 'Activity Measure', 'Medium', 'synthesis type (for modified)']:
    if c in raw.columns:
        print('--- value counts:', c)
        print(raw[c].value_counts(dropna=False).to_string())
        print()
if 'sequence' in raw.columns:
    print('unique raw sequences:', raw['sequence'].nunique())
    print('length range:', raw['sequence'].dropna().astype(str).str.len().min(),
          '-', raw['sequence'].dropna().astype(str).str.len().max())


# ============================================================================
# CELL 07
# ============================================================================
# @title 3.3 Clean labels, sequences, non-canonical residues, provenance
STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')
AA_ALIAS = {'U': 'C', 'O': 'K', 'B': 'X', 'Z': 'X', 'J': 'X', 'X': 'X'}


def clean_seq(s):
    # uppercase, drop whitespace and gap characters, map non-canonical residues
    if not isinstance(s, str):
        return '', ''
    raw_s = s.strip().upper().replace(' ', '')
    mapped, noncanon = [], []
    for ch in raw_s:
        if ch in '-*.':
            continue
        if ch in STANDARD_AA:
            mapped.append(ch)
        else:
            noncanon.append(ch)
            mapped.append(AA_ALIAS.get(ch, 'X'))
    return ''.join(mapped), ''.join(sorted(set(noncanon)))


def clean_label(v):
    if not isinstance(v, str):
        return np.nan
    v = v.strip().lower()
    if v.startswith('activ'):
        return 1
    if v.startswith('inactiv') or v.startswith('not activ'):
        return 0
    return np.nan


df = raw.copy()
seq_clean = df['sequence'].map(clean_seq)
df['sequence'] = [s for s, _ in seq_clean]
df['noncanonical_residues'] = [n for _, n in seq_clean]
df['n_noncanonical'] = df['noncanonical_residues'].map(len)
df['label'] = df['Activity'].map(clean_label)
df['length'] = df['sequence'].str.len()

df['pep_id'] = df['Peptide ID'].astype(str).str.strip()
df['source_block'] = np.where(df['pep_id'].str.match(r'^DBAASP', case=False), 'modified_DBAASP', 'natural')
df['strain'] = df['Target Species'].astype(str)
df['synthesis_type'] = df['synthesis type (for modified)'].fillna('natural').astype(str).str.strip().str.lower()
df['medium'] = df['Medium'].astype(str).str.strip()
df['unit'] = df['Unit'].astype(str).str.strip()

n0 = len(df)
df = df.dropna(subset=['sequence', 'label'])
df = df[df['sequence'].str.len() >= 5]
df['label'] = df['label'].astype(int)
df = df.reset_index(drop=True)

print(f'rows: {n0} -> {len(df)}  (dropped {n0 - len(df)} unusable)')
print('label counts:')
print(df['label'].value_counts().to_string())
print('class balance: %.1f%% active' % (100 * df['label'].mean()))
print('source block:')
print(df['source_block'].value_counts().to_string())
print('synthesis type:')
print(df['synthesis_type'].value_counts().to_string())
print('medium:')
print(df['medium'].value_counts().to_string())
print('length: min %d  median %d  max %d' % (df['length'].min(), df['length'].median(), df['length'].max()))
print('sequences with non-canonical residues:', int((df['n_noncanonical'] > 0).sum()))
if (df['n_noncanonical'] > 0).any():
    print(df.loc[df['n_noncanonical'] > 0, 'noncanonical_residues'].value_counts().to_string())
print()
print('label x source block:')
print(pd.crosstab(df['source_block'], df['label']).to_string())


# ============================================================================
# CELL 08 (markdown, commented out)
# ============================================================================
# # 4. Leakage control: duplicate sequences and label conflicts
#
# The natural block and the DBAASP block contain overlapping records, and a few sequences carry *both* labels (e.g. the same peptide is `Inactive` in one record and `Active` in another, and two records for peptide 4294 disagree). With a random split such a pair is trivially memorised, which inflates every metric.
#
# * **4.1** collapse exact duplicates (majority label), drop conflicting sequences, and keep a `n_records` count.
# * **4.2** build CD-HIT style groups at 90% identity with a 3-mer prefilter plus Biopython local alignment, so `GROUP_AWARE_SPLIT` can keep a group on one side of the split.
# * **4.3** drop clusters that still contain both labels.


# ============================================================================
# CELL 09
# ============================================================================
# @title 4.1 Exact duplicate consolidation
before = len(df)

if DEDUP_EXACT:
    agg = df.groupby('sequence')['label'].agg(['nunique', 'mean', 'size'])
    conflict_seqs = set(agg.index[agg['nunique'] > 1])
    print(f'{int((agg["size"] > 1).sum())} sequences occur more than once; '
          f'{len(conflict_seqs)} of them carry conflicting labels')
    if conflict_seqs:
        show = df[df['sequence'].isin(conflict_seqs)][['pep_id', 'source_block', 'sequence', 'label']]
        print(show.sort_values('sequence').to_string(index=False))
    if DROP_LABEL_CONFLICTS and conflict_seqs:
        df = df[~df['sequence'].isin(conflict_seqs)].copy().reset_index(drop=True)
        print('dropped', len(conflict_seqs), 'conflicting sequences')
        agg = df.groupby('sequence')['label'].agg(['nunique', 'mean', 'size'])
    keep = agg['mean'].round().astype(int).to_frame('label')
    keep['n_records'] = agg['size']
    meta = df.groupby('sequence').agg(
        pep_id=('pep_id', 'first'),
        source_block=('source_block', 'first'),
        strain=('strain', 'first'),
        synthesis_type=('synthesis_type', 'first'),
        medium=('medium', 'first'),
        unit=('unit', 'first'),
        length=('length', 'first'),
        noncanonical_residues=('noncanonical_residues', 'first'),
        n_noncanonical=('n_noncanonical', 'first'),
    )
    df = meta.join(keep).reset_index()

df['label'] = df['label'].astype(int)
df = df.reset_index(drop=True)

print(f'rows: {before} -> {len(df)} | unique sequences: {df["sequence"].nunique()}')
print('active / inactive:', int(df['label'].sum()), '/', int((1 - df['label']).sum()))
df.head()


# ============================================================================
# CELL 10
# ============================================================================
# @title 4.2 CD-HIT style clustering (k-mer prefilter + Biopython alignment)
from Bio import Align
import time

K = 3


def kmer_vec(seq, k=K):
    if len(seq) >= k:
        return set(seq[i:i + k] for i in range(len(seq) - k + 1))
    return {seq}


def identity_aligned(a_seq, b_seq, aligner, min_len):
    # local alignment identity = identical aligned positions / shortest sequence
    # NOTE: Aln.aligned yields (start, end) coordinate pairs per block, not strings,
    # so the aligned substrings are sliced out of the originals.
    aln = aligner.align(a_seq, b_seq)[0]
    blocks_a, blocks_b = aln.aligned
    matched = 0
    for (a0, a1), (b0, b1) in zip(blocks_a, blocks_b):
        matched += sum(1 for x, y in zip(a_seq[a0:a1], b_seq[b0:b1]) if x == y)
    return matched / max(1, min_len)


def build_clusters(seqs, identity=0.90):
    t0 = time.time()
    n = len(seqs)
    aligner = Align.PairwiseAligner()
    aligner.mode = 'local'
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    aligner.substitution_matrix = Align.substitution_matrices.load('BLOSUM62')

    kmers = [kmer_vec(s) for s in seqs]
    order = np.argsort([-len(s) for s in seqs])      # longest first -> cluster representatives
    clusters = np.full(n, -1, dtype=int)
    rep_idx, rep_km, cid = [], [], 0
    n_align = 0
    prefilter = max(0.5, identity - 0.35)            # cheap gate before the real alignment
    for i in order:
        si, li = seqs[i], len(seqs[i])
        ki = kmers[i]
        placed = False
        for k, r in enumerate(rep_idx):
            lr = len(seqs[r])
            if abs(li - lr) / max(li, lr) > (1 - identity):
                continue                              # length filter
            if len(ki & rep_km[k]) / max(1, min(len(ki), len(rep_km[k]))) < prefilter:
                continue                              # cheap k-mer prefilter
            n_align += 1
            if identity_aligned(si, seqs[r], aligner, min(li, lr)) >= identity:
                clusters[i] = clusters[r]
                placed = True
                break
        if not placed:
            clusters[i] = cid
            rep_idx.append(i)
            rep_km.append(ki)
            cid += 1
    print('clustered %d sequences at identity >= %.2f: %d clusters (%d redundant), '
          '%d real alignments, %.1fs' % (n, identity, cid, n - cid, n_align, time.time() - t0))
    return clusters


if BUILD_CLUSTERS:
    df['cluster'] = build_clusters(df['sequence'].tolist(), CLUSTER_IDENTITY)
else:
    df['cluster'] = np.arange(len(df))

print('cluster size distribution (size : n_clusters)')
print(df['cluster'].value_counts().value_counts().sort_index().to_string())


# ============================================================================
# CELL 11
# ============================================================================
# @title 4.3 Drop clusters with mixed labels, then persist
bad = set(df.groupby('cluster')['label'].nunique().pipe(lambda s: s[s > 1]).index)
if bad:
    print('dropping', len(bad), 'clusters that still contain both labels')
    print(df[df['cluster'].isin(bad)][['pep_id', 'cluster', 'sequence', 'label']]
          .sort_values('cluster').to_string(index=False))
    df = df[~df['cluster'].isin(bad)].copy().reset_index(drop=True)

    print('FINAL modelling set: n = %d | active = %d | inactive = %d | clusters = %d | largest cluster = %d'
          % (len(df), int(df['label'].sum()), int((1 - df['label']).sum()),
             df['cluster'].nunique(), int(df['cluster'].value_counts().max())))
    print('balance: %.1f%% active' % (100 * df['label'].mean()))
    print('length: min %d  median %d  max %d' % (df['length'].min(), df['length'].median(), df['length'].max()))

seqs = df['sequence'].tolist()
df.to_csv(os.path.join(WORKDIR, '01_dataset_clean.csv'), index=False)


# ============================================================================
# CELL 12 (markdown, commented out)
# ============================================================================
# # 5. Feature extraction 2.3.1 - PepFoundry graph features
#
# `PepFoundry` (Garzon Otero *et al.*, *JCIM* 2026, `doi:10.1021/acs.jcim.5c02629`) turns a peptide sequence into an **atom-mapped RDKit molecule** by concatenating the CHUCKLES SMILES of its residues, so natural, D- and non-canonical residues all travel through the same code path. From that molecule it returns the paper's graph tensors:
#
# * **atom features** - one 24-d vector per heavy atom (element, degree, formal charge, aromaticity, ring membership, chirality tags ...)
# * **bond features** - one 16-d vector per bond (bond type, conjugation, ring membership, stereo ...)
#
# Graph size varies with peptide length (10-30 residues), so each tensor is summarised into a **fixed-length vector** by the four pooling statistics in `PF_POOL_STATS` (`mean`, `std`, `max`, `sum`): `4 x 24 + 4 x 16 = 160` features. The same atom matrices are reused in section 13 for exact per-residue attribution and in section 15 for the optional GNN, and results are cached to `.npz`.


# ============================================================================
# CELL 13
# ============================================================================
# @title 5.1 PepFoundry molecules + atom/bond tensor features (cached)
import torch
from pepfoundry.interface import PepFoundry

PF = None            # built lazily: loading the 157-entry CHUCKLES library takes ~1 min


def pf_engine():
    global PF
    if PF is None:
        t0 = time.time()
        PF = PepFoundry()
        print('PepFoundry library ready in %.0fs' % (time.time() - t0))
    return PF


def pf_mol(seq):
    # atom-mapped RDKit molecule; MolFromSequence is the fallback if a token is unknown
    mol = pf_engine().get_peptide(seq, plot_peptide=False)
    if mol is None:
        from rdkit import Chem
        mol = Chem.MolFromSequence(seq)
    return mol


def pf_tensors(mol, chiralities=PF_CHIRALITIES, device=PF_DEVICE):
    atom_t, bond_t = pf_engine().get_peptide_tensor_atomic_features(
        mol, chiralities=chiralities, device=device)
    return (np.asarray(atom_t.detach().cpu(), dtype=float),
            np.asarray(bond_t.detach().cpu(), dtype=float))


_POOL_FN = {'mean': lambda a: a.mean(0), 'std': lambda a: a.std(0),
            'max': lambda a: a.max(0), 'sum': lambda a: a.sum(0)}


def pool_stats(A, stats=PF_POOL_STATS):
    # length-invariant summary of one [n_nodes, d] feature matrix
    if A is None or len(A) == 0:
        return np.zeros(len(stats) * (A.shape[1] if A is not None and A.ndim == 2 else 0))
    return np.concatenate([_POOL_FN[s](A) for s in stats])


def pf_pool(atom_t, bond_t, stats=PF_POOL_STATS):
    return np.concatenate([pool_stats(atom_t, stats), pool_stats(bond_t, stats)])


PF_STAT_NAMES = (['atom_%s_%02d' % (s, i) for s in PF_POOL_STATS for i in range(24)] +
                 ['bond_%s_%02d' % (s, i) for s in PF_POOL_STATS for i in range(16)])


def pepfoundry_features(seq):
    # -> (pooled vector, molecule, atom matrix [n_atoms, 24], bond matrix [n_bonds, 16])
    mol = pf_mol(seq)
    if mol is None:
        raise ValueError('no molecule could be built')
    a, b = pf_tensors(mol)
    return pf_pool(a, b), mol, a, b


cache_key = hashlib.md5(('|'.join(seqs) + str(PF_POOL_STATS) + str(PF_CHIRALITIES)).encode()).hexdigest()[:10]
cache = os.path.join(WORKDIR, 'pepfoundry_%s.npz' % cache_key)

if os.path.exists(cache) and USE_PEPFOUNDRY:
    X_pf = np.load(cache)['X']
    n_atoms = np.load(cache)['atoms'].tolist()
    print('loaded cache', cache, X_pf.shape)
else:
    rows, mols, failed, n_atoms, n_bonds = [], [], [], [], []
    t0 = time.time()
    for i, s in enumerate(seqs):
        try:
            f, m, a, b = pepfoundry_features(s)
            rows.append(f)
            mols.append(m)
            n_atoms.append(a.shape[0])
            n_bonds.append(b.shape[0])
        except Exception as e:
            print('  FAIL %s (%s): %s' % (df['pep_id'].iloc[i], s, repr(e)[:110]))
            failed.append(i)
    if failed:
        raise RuntimeError('%d of %d peptides could not be featurised - fix the input before modelling'
                           % (len(failed), len(seqs)))
    X_pf = np.vstack(rows).astype(np.float32)
    if PF_KEEP_MOLS:
        MOL_OBJECTS = mols
    np.savez_compressed(cache, X=X_pf, sequences=np.array(seqs), stat_names=np.array(PF_STAT_NAMES),
                        atoms=np.array(n_atoms), bonds=np.array(n_bonds))
    print('PepFoundry features for %d peptides in %.0fs' % (len(seqs), time.time() - t0))

PF_DIM = X_pf.shape[1]
print('PepFoundry block:', X_pf.shape, '| atoms per peptide: min %d median %d max %d'
      % (min(n_atoms), int(np.median(n_atoms)), max(n_atoms)))
print('nan:', int(np.isnan(X_pf).sum()), '| inf:', int(np.isinf(X_pf).sum()))
print('feature names e.g.:', PF_STAT_NAMES[:2], '...', PF_STAT_NAMES[-2:])


# ============================================================================
# CELL 14 (markdown, commented out)
# ============================================================================
# # 6. Feature extraction 2.3.2a - RDKit 2D descriptors
#
# Every descriptor is computed on the **same PepFoundry-built molecule**, so the chemistry is real peptide chemistry (capping, D-residues and non-canonical side chains included) rather than a string-level approximation:
#
# * **bulk / lipophilicity** - `MolWt`, `ExactMolWt`, `MolLogP`, `MolMR`, `TPSA`, `LabuteASA`, `SPS`
# * **H-bonding and charge** - donors, acceptors, rotatable bonds, `FormalCharge`, `NHOHCount`, `NOCount`, partial charges
# * **shape / topology** - `BertzCT`, `BalabanJ`, `Chi0v-4v`, `Kappa1-3`, `HallKierAlpha`, ring and aromatic counts
# * **drug-likeness / fragments** - `QED`, `SlogP_VSA*`, `PEOE_VSA*`, E-state extremes
# * **stereochemistry** - `NumAtomStereoCenters` and `NumUnspecifiedAtomStereoCenters`, which is what separates L- from D-peptides in a 2D graph
# * **amino-acid composition** - the 20 fractional counts, so a SHAP beeswarm on this block stays readable
#
# Descriptor names are resolved defensively, so anything a given RDKit build renames or drops is skipped instead of crashing the run.


# ============================================================================
# CELL 15 (markdown, commented out)
# ============================================================================
# # 7. Feature extraction 2.3.2b - AMP pharmacophore descriptors
#
# Three descriptors carry the antimicrobial-pharmacophore story, and all three now come from the PepFoundry molecule or its residue assignment:
#
# * **net charge** - the sequence-level Henderson-Hasselbalch count at pH 7 (D/E = -1, K/R = +1, His = +0.1), which is the membrane-binding driver. RDKit's `FormalCharge` of the built molecule is also computed, but PepFoundry builds **neutral** molecules, so that column is constant and gets dropped in cell 8.1 - the sequence count is the feature you actually want.
# * **hydrophobic moment** - Eisenberg consensus and Kyte-Doolittle hydropathy projected on a helical wheel (3.6 residues per turn, 11-residue window, maximum over windows): amphipathicity, the driver of membrane disruption.
# * **amphipathicity** - that moment normalised by mean hydrophobicity, plus the most cationic 7-residue window fraction and a Boman-style hydrophobic-residue fraction.


# ============================================================================
# CELL 16
# ============================================================================
# @title 6.1 + 7.1 RDKit 2D descriptors, composition and AMP pharmacophore
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, rdMolDescriptors

RDLogger.DisableLog('rdApp.*')

KD = dict(zip('AVILPFLMYWSTGC', [1.8, 4.2, 4.5, 3.8, -1.6, 2.8, -0.9, 1.9, -0.4, -0.8, -0.7, -0.1, -1.3, -0.5, 1.9, -0.8]))
EI_raw = dict(zip('AVLIPFMDWKSTYRNQCEG',
                  [0.62, 0.46, 0.30, 0.37, 0.26, 0.35, 0.21, 0.10, 0.61, 0.42, 0.29,
                   0.27, 0.28, 0.14, 0.61, 0.25, 0.15, 0.10, 0.28, 0.29]))
_mu, _sd = float(np.mean(list(EI_raw.values()))), float(np.std(list(EI_raw.values())))
EI = {k: (v - _mu) / _sd for k, v in EI_raw.items()}
CHARGE_PH = {**{k: 0.0 for k in 'ACDEFGHILNPQSTVWY'}, 'K': 1.0, 'R': 1.0, 'H': 0.1, 'D': -1.0, 'E': -1.0}

RDKIT_2D = [
    ('MolWt', lambda m: Descriptors.MolWt(m)),
    ('HeavyAtomMolWt', lambda m: Descriptors.HeavyAtomMolWt(m)),
    ('ExactMolWt', lambda m: Descriptors.ExactMolWt(m)),
    ('MolLogP', lambda m: Crippen.MolLogP(m)),
    ('MolMR', lambda m: Crippen.MolMR(m)),
    ('TPSA', lambda m: Descriptors.TPSA(m)),
    ('LabuteASA', lambda m: Descriptors.LabuteASA(m)),
    ('SPS', lambda m: Descriptors.SPS(m)),
    ('NumHDonors', lambda m: Lipinski.NumHDonors(m)),
    ('NumHAcceptors', lambda m: Lipinski.NumHAcceptors(m)),
    ('NumRotatableBonds', lambda m: Lipinski.NumRotatableBonds(m)),
    ('FractionCSP3', lambda m: Descriptors.FractionCSP3(m)),
    ('RingCount', lambda m: Lipinski.RingCount(m)),
    ('NumAromaticRings', lambda m: Lipinski.NumAromaticRings(m)),
    ('NumHeteroatoms', lambda m: Lipinski.NumHeteroatoms(m)),
    ('HeavyAtomCount', lambda m: m.GetNumHeavyAtoms()),
    ('NHOHCount', lambda m: Descriptors.NHOHCount(m)),
    ('NOCount', lambda m: Descriptors.NOCount(m)),
    ('FormalCharge', lambda m: Chem.GetFormalCharge(m)),
    ('NumRadicalElectrons', lambda m: Descriptors.NumRadicalElectrons(m)),
    ('BalabanJ', lambda m: Descriptors.BalabanJ(m)),
    ('BertzCT', lambda m: Descriptors.BertzCT(m)),
    ('Chi0v', lambda m: Descriptors.Chi0v(m)),
    ('Chi1v', lambda m: Descriptors.Chi1v(m)),
    ('Chi2v', lambda m: Descriptors.Chi2v(m)),
    ('Chi3v', lambda m: Descriptors.Chi3v(m)),
    ('Chi4v', lambda m: Descriptors.Chi4v(m)),
    ('Kappa1', lambda m: Descriptors.Kappa1(m)),
    ('Kappa2', lambda m: Descriptors.Kappa2(m)),
    ('Kappa3', lambda m: Descriptors.Kappa3(m)),
    ('HallKierAlpha', lambda m: Descriptors.HallKierAlpha(m)),
    # Ipc is deliberately absent: it overflows on peptides (1e141) and costs ~5 s per molecule
    ('MaxPartialCharge', lambda m: Descriptors.MaxPartialCharge(m)),
    ('MinPartialCharge', lambda m: Descriptors.MinPartialCharge(m)),
    ('MaxAbsPartialCharge', lambda m: Descriptors.MaxAbsPartialCharge(m)),
    ('MinAbsPartialCharge', lambda m: Descriptors.MinAbsPartialCharge(m)),
    ('MaxEStateIndex', lambda m: Descriptors.MaxEStateIndex(m)),
    ('MaxAbsEStateIndex', lambda m: Descriptors.MaxAbsEStateIndex(m)),
    ('MinEStateIndex', lambda m: Descriptors.MinEStateIndex(m)),
    ('MinAbsEStateIndex', lambda m: Descriptors.MinAbsEStateIndex(m)),
    ('NumAtomStereoCenters', lambda m: rdMolDescriptors.CalcNumAtomStereoCenters(m)),
    ('NumUnspecAtomStereoCenters', lambda m: rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(m)),
    ('qed', lambda m: QED.qed(m)),
    ('SlogP_VSA1', lambda m: rdMolDescriptors.SlogP_VSA_(m, 1)),
    ('SlogP_VSA2', lambda m: rdMolDescriptors.SlogP_VSA_(m, 2)),
    ('SlogP_VSA5', lambda m: rdMolDescriptors.SlogP_VSA_(m, 5)),
    ('PEOE_VSA1', lambda m: rdMolDescriptors.PEOE_VSA_(m, 1)),
    ('PEOE_VSA4', lambda m: rdMolDescriptors.PEOE_VSA_(m, 4)),
]

_wheel_cache = {}


def helicoidal_moment(seq, scale, window=11):
    # hydrophobic moment on a helical wheel (3.6 residues per turn), max over all windows
    key = (seq, window, tuple(sorted(scale.items())))
    if key in _wheel_cache:
        return _wheel_cache[key]
    v = np.array([scale.get(a, 0.0) for a in seq], dtype=float)
    n = len(v)
    best = 0.0
    if n:
        w = min(window, n)
        ang = 2 * np.pi * np.arange(w) / w
        for s in range(n - w + 1):
            seg = v[s:s + w]
            mu = np.hypot((seg * np.cos(ang)).sum(), (seg * np.sin(ang)).sum()) / w
            best = max(best, mu)
    best = float(best)
    _wheel_cache[key] = best
    return best


def amp_pharmacophore(seq, mol):
    # section 7: molecule-level charge plus two helical wheels
    n = max(1, len(seq))
    v = np.array([KD.get(a, 0.0) for a in seq], dtype=float)
    f = {}
    f['amp_net_charge_rdkit'] = float(Chem.GetFormalCharge(mol)) if mol is not None else np.nan
    f['amp_net_charge_pH7_seq'] = float(sum(CHARGE_PH.get(a, 0.0) for a in seq))
    f['amp_charge_density'] = f['amp_net_charge_pH7_seq'] / n
    f['amp_hydrophobic_moment_kd'] = helicoidal_moment(seq, KD, 11)
    f['amp_hydrophobic_moment_eis'] = helicoidal_moment(seq, EI, 11)
    f['amp_mean_hydropathy'] = float(v.mean())
    f['amp_hydropathy_range'] = float(v.max() - v.min())
    f['amp_amphipathicity'] = f['amp_hydrophobic_moment_kd'] / (abs(v.mean()) + 0.1)
    f['amp_amphipathicity_norm'] = f['amp_hydrophobic_moment_kd'] / (0.1 + float(np.abs(v).mean()))
    f['amp_boman_frac'] = float(np.mean([1.0 if a in 'AILVFWMY' else 0.0 for a in seq]))
    win = 7
    f['amp_max_cationic_window'] = float(max(
        [sum(1.0 for a in seq[i:i + win] if a in 'KR') / max(1, len(seq[i:i + win]))
         for i in range(max(1, n - win + 1))] or [0.0]))
    return f


AMP_FEATURE_NAMES = ['amp_net_charge_rdkit', 'amp_net_charge_pH7_seq', 'amp_charge_density',
                     'amp_hydrophobic_moment_kd', 'amp_hydrophobic_moment_eis',
                     'amp_mean_hydropathy', 'amp_hydropathy_range', 'amp_amphipathicity',
                     'amp_amphipathicity_norm', 'amp_boman_frac', 'amp_max_cationic_window']


def rdkit_features(seq, mol):
    f = {}
    for name, fn in RDKIT_2D:
        try:
            f['rdkit_' + name] = float(fn(mol)) if mol is not None else np.nan
        except Exception:
            f['rdkit_' + name] = np.nan
    for aa in 'ACDEFGHIKLMNPQRSTVWY':
        f['aac_' + aa] = float(seq.count(aa)) / max(1, len(seq))
    f.update(amp_pharmacophore(seq, mol))
    return f


RDKIT_CACHE = os.path.join(WORKDIR, 'rdkit_block.npz')
rd_key = hashlib.md5(('|'.join([n for n, _ in RDKIT_2D]) + '#' + '|'.join(AMP_FEATURE_NAMES)
                      + '#' + '|'.join(seqs)).encode()).hexdigest()[:10]

X_rdkit = None
if os.path.exists(RDKIT_CACHE) and not FORCE_RECOMPUTE:
    try:
        z = np.load(RDKIT_CACHE, allow_pickle=True)
        if str(z['key']) == rd_key:
            X_rdkit = pd.DataFrame(z['X'], columns=[str(c) for c in z['cols']])
            print('loaded cache', RDKIT_CACHE, X_rdkit.shape)
    except Exception as e:
        print('cache ignored:', repr(e)[:120])

if X_rdkit is None:
    rows = []
    t0 = time.time()
    for i, s in enumerate(seqs):
        try:
            m = pf_mol(s) if USE_PEPFOUNDRY else Chem.MolFromSequence(s)
            if m is None:
                m = Chem.MolFromSequence(s)
            rows.append(rdkit_features(s, m))
        except Exception as e:
            print('  FAIL %s (%s): %s' % (df['pep_id'].iloc[i], s, repr(e)[:110]))
            rows.append({})

    X_rdkit = pd.DataFrame(rows)
    X_rdkit = X_rdkit.replace([np.inf, -np.inf], np.nan)
    np.savez_compressed(RDKIT_CACHE, X=X_rdkit.values.astype(float), cols=np.array(list(X_rdkit.columns)),
                        key=np.array(rd_key))
    print('RDKit block: %d features in %.0fs' % (X_rdkit.shape[1], time.time() - t0))

RDKIT_ALL_COLS = list(X_rdkit.columns)
RDKIT_2D_COLS = [c for c in RDKIT_ALL_COLS if c.startswith('rdkit_')]
COMP_COLS = [c for c in RDKIT_ALL_COLS if c.startswith('aac_')]
AMP_COLS = [c for c in RDKIT_ALL_COLS if c.startswith('amp_')]
key_cols = [c for c in ['rdkit_MolWt', 'rdkit_MolLogP', 'rdkit_TPSA',
                        'rdkit_NumAtomStereoCenters', 'rdkit_qed',
                        'amp_net_charge_pH7_seq', 'amp_hydrophobic_moment_kd'] if c in X_rdkit.columns]
print(X_rdkit[key_cols].describe().T.round(3).to_string())
print()
print('peptides carrying atom stereocentres (chiral):',
      int((X_rdkit['rdkit_NumAtomStereoCenters'] > 0).sum()))
print('note: rdkit_FormalCharge is 0 for every peptide - PepFoundry builds neutral molecules,')
print('      so the usable charge feature is the pH 7 sequence count (amp_net_charge_pH7_seq);')
print('      the formal-charge column is kept in the table but dropped as constant in cell 8.1.')
print('net charge at pH 7 (sequence) describe:')
print(X_rdkit['amp_net_charge_pH7_seq'].describe().round(2).to_string())


# ============================================================================
# CELL 17 (markdown, commented out)
# ============================================================================
# # 7b. Proposal 2.3.2 - assigning the physicochemical descriptors to the dataset
#
# The descriptors above are computed **per peptide row**, so they are now bound back onto the curated dataset: every row of `natural-modified-combined` gains its own physicochemical column block instead of the matrix being a detached array. The join is on `pep_id` with `validate='one_to_one'`, followed by three assertions (same length, same `pep_id` order, no row without descriptors), so a silent row shift between `X_rdkit` and `df` is impossible.
#
# The result is saved as **`02_dataset_with_physicochemical_descriptors.csv`**: sequence, label, provenance, cluster id, length - plus the 78 physicochemical columns. This is the table that proposal 2.4.1 concatenates onto the learned representation, and the same columns are what section 2.5.2 filters de novo candidates on.
#
# Descriptor groups, matching the proposal wording:
#
# | group | columns | proposal role |
# |---|---|---|
# | `rdkit_*` | 47 | molecular weight, net charge, hydrophobicity (LogP / SlogP / LabuteASA), amphipathicity proxies, aliphatic-index family, TPSA / H-bonding, topology and chirality |
# | `aac_*` | 20 | fractional composition of the 20 canonical amino acids |
# | `amp_*` | 11 | AMP pharmacophore: net charge at pH 7, Eisenberg and Kyte-Doolittle hydrophobic moment, amphipathicity, Boman fraction, most-cationic window |


# ============================================================================
# CELL 18
# ============================================================================
# @title 7b.1 Physicochemical descriptors merged onto the dataset (id-verified)
DESC_DF = X_rdkit.copy()
DESC_DF.insert(0, 'pep_id', df['pep_id'].values)

DATASET_WITH_DESC = df.merge(DESC_DF, on='pep_id', how='left', validate='one_to_one', sort=False)
assert len(DATASET_WITH_DESC) == len(df), 'descriptor merge changed the row count'
assert DATASET_WITH_DESC['pep_id'].is_unique, 'duplicate pep_id after the descriptor merge'
assert DATASET_WITH_DESC['pep_id'].tolist() == df['pep_id'].tolist(), 'descriptor rows are misaligned'
assert DATASET_WITH_DESC[RDKIT_ALL_COLS].notna().any(axis=1).all(), 'a row ended up with no descriptor'

DATASET_WITH_DESC.to_csv(os.path.join(WORKDIR, '02_dataset_with_physicochemical_descriptors.csv'),
                          index=False)

# univariate screen: which descriptors actually separate the two classes, and in which
# direction. This is the deterministic half of what section 2.5.2 asks TreeSHAP to confirm.
from scipy.stats import pointbiserialr

_aucs = []
for c in RDKIT_ALL_COLS:
    v = DATASET_WITH_DESC[c].values.astype(float)
    if not np.isfinite(v).all() or np.nanstd(v) == 0:
        _aucs.append(dict(descriptor=c, auc=np.nan, r_pb=np.nan, mean_active=np.nan, mean_inactive=np.nan))
        continue
    r, _p = pointbiserialr(DATASET_WITH_DESC['label'].values.astype(int), v)
    _rank = pd.Series(v).rank().values
    _n1, _n0 = int((DATASET_WITH_DESC['label'] == 1).sum()), int((DATASET_WITH_DESC['label'] == 0).sum())
    _auc = (_rank[DATASET_WITH_DESC['label'].values == 1].sum() - _n1 * (_n1 + 1) / 2) / (_n1 * _n0)
    _aucs.append(dict(descriptor=c, auc=float(_auc), r_pb=float(r),
                      mean_active=float(v[DATASET_WITH_DESC['label'].values == 1].mean()),
                      mean_inactive=float(v[DATASET_WITH_DESC['label'].values == 0].mean())))

DESC_STATS = (pd.DataFrame(_aucs)
              .assign(group=lambda d: d['descriptor'].str.split('_').str[0],
                      abs_auc=lambda d: d['auc'].sub(0.5).abs())
              .sort_values('abs_auc', ascending=False))
DESC_STATS.to_csv(os.path.join(WORKDIR, 'descriptor_univariate_stats.csv'), index=False)

print('descriptor table assigned: %d rows x %d descriptor columns -> %s'
      % (len(DATASET_WITH_DESC), len(RDKIT_ALL_COLS),
         os.path.join(WORKDIR, '02_dataset_with_physicochemical_descriptors.csv')))
print('groups:', {g: int((DESC_STATS.group == g).sum()) for g in ['rdkit', 'aac', 'amp']})
print()
print('=== top 15 physicochemical descriptors by |AUC - 0.5| (univariate, on the curated set) ===')
print(DESC_STATS.head(15).round(3).to_string(index=False))


# ============================================================================
# CELL 19 (markdown, commented out)
# ============================================================================
# # 8. Proposal 2.4.1 - feature matrix construction (X) and structural dimensions
#
# * **p_RDKit** - physicochemical descriptors (2D + composition + AMP pharmacophore), interpretable, `d << n`
# * **g_PepFoundry** - pooled atom/bond graph features (160-d raw), compressed to `SVD_COMPONENTS` *inside* every CV fold
# * **Combined** - the two blocks horizontally concatenated, i.e.
#
# ```
# X_i = [ g_PepFoundry(i) , p_RDKit(i) ]        in R^D
#       103 columns        69 columns          D = 172
# ```
#
# which is the PepFoundry/RDKit form of the proposal's `X_i = [e_ESM-2, p_RDKit] in R^1300`: one learned representation stream plus the deterministic physicochemical block, instead of 1280 transformer columns plus descriptors. Raw widths before the constant-column drop are 160 + 78 = 238.
#
# `SafeSVD` clamps the number of components to `min(k, n_samples-1, n_features-1)` so the pipeline survives the 10/90 split where a training fold holds fewer than 128 samples.
#
# One cleaning step happens here: any column whose standard deviation is 0 is **dropped** and listed. That removes `FormalCharge` (PepFoundry builds neutral molecules) and most of the `max`-pooled atom/bond channels - a per-atom one-hot channel has the same maximum in every peptide, so `max` pooling destroys it. That is a result, not a bug: the informative pooling statistics for this dataset are `mean`, `std` and `sum`, and the model blocks are built from the surviving columns only.


# ============================================================================
# CELL 20
# ============================================================================
# @title 8.1 SafeSVD + assembly of the three feature blocks
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD


class SafeSVD(BaseEstimator, TransformerMixin):
    # TruncatedSVD that adapts n_components to tiny folds (small-n regime)

    def __init__(self, n_components=128, random_state=0):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        k = int(max(2, min(self.n_components, X.shape[0] - 1, X.shape[1] - 1)))
        self.svd_ = TruncatedSVD(n_components=k, random_state=self.random_state)
        self.svd_.fit(X)
        self.n_components_ = k
        return self

    def transform(self, X):
        return self.svd_.transform(X)


RDKIT_RAW = X_rdkit[RDKIT_ALL_COLS].values.astype(float)
PF_RAW = X_pf.astype(float)

# a column that never varies carries no signal and breaks the scaling of the linear
# models, so drop the constant ones explicitly (e.g. FormalCharge, which is always 0
# because PepFoundry builds neutral molecules)
keep_rd = np.where(RDKIT_RAW.std(axis=0) > 0)[0]
keep_pf = np.where(PF_RAW.std(axis=0) > 0)[0]
DROPPED_RDKIT = [RDKIT_ALL_COLS[i] for i in range(len(RDKIT_ALL_COLS)) if i not in set(keep_rd.tolist())]
DROPPED_PF = [PF_STAT_NAMES[i] for i in range(PF_DIM) if i not in set(keep_pf.tolist())]
RDKIT_DIM = len(keep_rd)          # width actually used by the models (after the constant drop)
# the kept column positions, reused by the explainer and the inference helper so that
# they always see exactly the columns the models were trained on
PF_KEEP_IDX = keep_pf.tolist()
RDKIT_KEEP_COLS = [RDKIT_ALL_COLS[i] for i in keep_rd]

X_blocks = {}
if USE_RDKIT:
    X_blocks['RDKit'] = RDKIT_RAW[:, keep_rd]
if USE_PEPFOUNDRY:
    X_blocks['PepFoundry'] = PF_RAW[:, keep_pf]
if USE_RDKIT and USE_PEPFOUNDRY:
    X_blocks['Combined'] = np.hstack([RDKIT_RAW[:, keep_rd], PF_RAW[:, keep_pf]])

y = df['label'].values.astype(int)

print('n =', len(y), '| d_RDKit = %d -> %d | d_PepFoundry = %d -> %d'
      % (RDKIT_RAW.shape[1], len(keep_rd), PF_RAW.shape[1], len(keep_pf)))
if DROPPED_RDKIT:
    print('  dropped constant RDKit columns  :', ', '.join(DROPPED_RDKIT))
if DROPPED_PF:
    print('  dropped constant PepFoundry cols:', ', '.join(DROPPED_PF))
for k, v in X_blocks.items():
    print('  block %-12s %s | nan cells %d' % (k, v.shape, int(np.isnan(v).sum())))

np.save(os.path.join(WORKDIR, 'y.npy'), y)
for k, v in X_blocks.items():
    np.save(os.path.join(WORKDIR, 'X_%s.npy' % k), v)
df.to_csv(os.path.join(WORKDIR, '02_dataset_final.csv'), index=False)

# manifest of the modelling matrix: column order, provenance and the physicochemical
# names, so section 2.5 can rebuild the identical vector for a de novo candidate
import json

FEATURE_MANIFEST = {
    'proposal_section': '2.4.1',
    'representation': {'learned_stream': 'PepFoundry pooled atom/bond graph statistics',
                       'deterministic_stream': 'RDKit 2D + composition + AMP pharmacophore'},
    'row_count': int(len(y)),
    'blocks': {k: int(v.shape[1]) for k, v in X_blocks.items()},
    'raw_widths': {'rdkit': int(RDKIT_RAW.shape[1]), 'pepfoundry': int(PF_RAW.shape[1])},
    'columns': {
        'RDKit': list(RDKIT_KEEP_COLS),
        'PepFoundry': [PF_STAT_NAMES[i] for i in PF_KEEP_IDX],
        'Combined': list(RDKIT_KEEP_COLS) + [PF_STAT_NAMES[i] for i in PF_KEEP_IDX],
    },
    'physicochemical': {
        'all_columns': list(RDKIT_ALL_COLS),
        'rdkit_2d': list(RDKIT_2D_COLS),
        'composition': list(COMP_COLS),
        'amp_pharmacophore': list(AMP_COLS),
        'dropped_constant': list(DROPPED_RDKIT),
        'kept': list(RDKIT_KEEP_COLS),
    },
    'pepfoundry': {'pool_stats': list(PF_POOL_STATS), 'chiralities': bool(PF_CHIRALITIES),
                   'raw_dim': int(PF_RAW.shape[1]), 'kept': len(PF_KEEP_IDX),
                   'svd_components_in_fold': int(SVD_COMPONENTS)},
}
with open(os.path.join(WORKDIR, 'feature_manifest.json'), 'w', encoding='utf-8') as f:
    json.dump(FEATURE_MANIFEST, f, indent=2)

print()
print('X = [g_PepFoundry , p_RDKit] -> Combined is %d columns (%d learned + %d physicochemical)'
      % (X_blocks['Combined'].shape[1], len(PF_KEEP_IDX), len(RDKIT_KEEP_COLS)))
print('wrote feature_manifest.json (consumed by section 2.5)')

if SHOW_PCA:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    D = StandardScaler().fit_transform(X_blocks['RDKit'])
    pca = PCA(n_components=2).fit(D)
    P = pca.transform(D)
    fig, ax = plt.subplots(figsize=(6.2, 5))
    for lab, col, name in [(0, 'tab:orange', 'Inactive'), (1, 'tab:blue', 'Active')]:
        m = y == lab
        ax.scatter(P[m, 0], P[m, 1], s=24, alpha=0.75, c=col, label='%s (n=%d)' % (name, int(m.sum())))
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('RDKit descriptor space (PCA) - %.0f%% of variance' % (100 * pca.explained_variance_ratio_.sum()))
    ax.legend()
    plt.tight_layout()
    plt.show()


# ============================================================================
# CELL 21 (markdown, commented out)
# ============================================================================
# # 9. Proposal 2.4.3 - the benchmarking suite
#
# Every model is a `Pipeline`, so imputation / scaling / SVD are fitted **inside** each CV fold and can never see the test data. Linear and distance-based learners receive the *full* graph vector after standardisation; tree ensembles receive the SafeSVD-compressed block (at `d = 160` a tree can only split on noise otherwise). `score_for()` returns an AUROC-compatible score even for models without `predict_proba` (LinearSVC, k-NN, GaussianNB).
#
# The proposal names six architectures. All six are here, plus extras so the winner is not a foregone conclusion:
#
# | proposal 2.4.3 | here | role |
# |---|---|---|
# | 1. L2-penalized Logistic Regression | `LogReg` | linear lower bound on the decision boundary |
# | 2. SVM with an RBF kernel | `SVC-RBF` | non-linear hyperplanes in the concatenated space |
# | 3. Decision Trees (single-tree baseline) | `DecisionTree` | unconstrained split logic, standardised features, **no** SVD compression so the splits are read on the real columns |
# | 4. XGBoost | `XGBoost` | regularised gradient-boosted trees |
# | 5. LightGBM | `LightGBM` | leaf-wise boosted trees |
# | 6. Random Forest | `RandomForest` | bootstrap ensemble over the wide feature vector |
# | - | `LinearSVC`, `ExtraTrees`, `GradientBoosting`, `CatBoost`, `LDA-shrunk`, `GaussianNB`, `kNN`, `MLP`, `Voting-soft` | additional single-model and ensemble references |


# ============================================================================
# CELL 22
# ============================================================================
# @title 9.1 Model definitions
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, VotingClassifier)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False
try:
    from lightgbm import LGBMClassifier
    HAVE_LGBM = True
except Exception:
    HAVE_LGBM = False
try:
    from catboost import CatBoostClassifier
    HAVE_CAT = True
except Exception:
    HAVE_CAT = False

K_PF = SVD_COMPONENTS
LINEAR_MODELS = {'LogReg', 'LinearSVC', 'SVC-RBF', 'LDA-shrunk', 'GaussianNB', 'kNN', 'MLP'}
TREE_MODELS = {'RandomForest', 'ExtraTrees', 'GradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost'}
# proposal 2.4.3 architecture 3: a single tree is a diagnostic, not a predictor, so it is
# deliberately left out of TREE_MODELS - it then sees the standardised full-width feature
# matrix (no SafeSVD), which is the only way its split logic is interpretable
SMALL_TREE_MODELS = {'DecisionTree'}

BASE_MODELS = {}


def _reg():
    return [('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]


def _tree_reg():
    return [('impute', SimpleImputer(strategy='median'))]


def _pf_reg(compress=True):
    steps = [('impute', SimpleImputer(strategy='median'))]
    if compress:
        steps.append(('svd', SafeSVD(n_components=K_PF, random_state=SEED)))
    steps.append(('scale', StandardScaler()))
    return steps


def _clf_of(fn):
    return fn()['clf'] if isinstance(fn, dict) else fn


BASE_MODELS['LogReg'] = lambda: Pipeline(_reg() + [('clf', LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced'))])
BASE_MODELS['DecisionTree'] = lambda: Pipeline(_reg() + [('clf', DecisionTreeClassifier(criterion='gini', min_samples_leaf=5, class_weight='balanced', random_state=SEED))])
BASE_MODELS['LinearSVC'] = lambda: Pipeline(_reg() + [('clf', LinearSVC(C=0.1, class_weight='balanced', max_iter=20000, random_state=SEED))])
BASE_MODELS['SVC-RBF'] = lambda: Pipeline(_reg() + [('clf', SVC(C=10, gamma='scale', probability=True, class_weight='balanced', random_state=SEED))])
BASE_MODELS['RandomForest'] = lambda: Pipeline(_tree_reg() + [('clf', RandomForestClassifier(n_estimators=500, max_features='sqrt', min_samples_leaf=2, class_weight='balanced', n_jobs=-1, random_state=SEED))])
BASE_MODELS['ExtraTrees'] = lambda: Pipeline(_tree_reg() + [('clf', ExtraTreesClassifier(n_estimators=500, max_features='sqrt', min_samples_leaf=2, class_weight='balanced', n_jobs=-1, random_state=SEED))])
BASE_MODELS['GradientBoosting'] = lambda: Pipeline(_tree_reg() + [('clf', GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=SEED))])
BASE_MODELS['LDA-shrunk'] = lambda: Pipeline(_reg() + [('clf', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))])
BASE_MODELS['GaussianNB'] = lambda: Pipeline(_reg() + [('clf', GaussianNB())])
BASE_MODELS['kNN'] = lambda: Pipeline(_reg() + [('clf', KNeighborsClassifier(n_neighbors=7, weights='distance'))])
BASE_MODELS['MLP'] = lambda: Pipeline(_reg() + [('clf', MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-2, max_iter=800, early_stopping=True, random_state=SEED))])
if HAVE_XGB:
    BASE_MODELS['XGBoost'] = lambda: Pipeline(_tree_reg() + [('clf', XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5, reg_lambda=2.0, min_child_weight=2, eval_metric='logloss', n_jobs=-1, random_state=SEED))])
if HAVE_LGBM:
    BASE_MODELS['LightGBM'] = lambda: Pipeline(_tree_reg() + [('clf', LGBMClassifier(n_estimators=400, num_leaves=15, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5, reg_lambda=2.0, min_child_samples=5, class_weight='balanced', n_jobs=-1, random_state=SEED, verbose=-1))])
if HAVE_CAT:
    BASE_MODELS['CatBoost'] = lambda: Pipeline(_tree_reg() + [('clf', CatBoostClassifier(iterations=400, depth=4, learning_rate=0.05, l2_leaf_reg=3, verbose=0, random_seed=SEED, allow_writing_files=False))])


def build_zoo(feature_block, models=None, include_ensemble=True, ensemble_members=('SVC-RBF', 'RandomForest', 'XGBoost', 'LightGBM', 'LogReg')):
    # every returned model is a leakage-free Pipeline for the requested block
    if models is None:
        models = list(BASE_MODELS)
    zoo = {}
    for name in models:
        if name not in BASE_MODELS:
            continue
        clf = BASE_MODELS[name]()['clf']
        if feature_block == 'RDKit':
            steps = _reg()
        elif feature_block == 'PepFoundry':
            steps = _pf_reg(compress=(name in TREE_MODELS))
        else:
            if name in TREE_MODELS:
                # RDKit descriptors scaled + graph block compressed, then concatenated
                ct = ColumnTransformer([
                    ('rdkit', Pipeline(_reg()), slice(0, RDKIT_DIM)),
                    ('graph', Pipeline(_pf_reg(compress=True)), slice(RDKIT_DIM, None))])
                steps = [('features', ct)]
            else:
                # keep the descriptors: impute + scale descriptors and the full graph block together
                steps = _reg()
        zoo[name] = Pipeline(steps + [('clf', clf)])
    if include_ensemble:
        members = [m for m in ensemble_members if m in zoo]
        if len(members) >= 3:
            # every ensemble member must see the same transformation, so borrow the
            # steps of the first non-tree member (plain impute + scale of all features)
            ref = next((m for m in members if m not in TREE_MODELS), members[0])
            steps = [('feat_' + n, clone(s)) for n, s in zoo[ref].steps[:-1]]
            zoo['Voting-soft'] = Pipeline(steps + [('clf', VotingEstimator_safe(members, zoo))])
    return zoo


def VotingEstimator_safe(members, zoo):
    ests = [(m, clone(zoo[m]['clf'])) for m in members]
    return VotingClassifier(estimators=ests, voting='soft', n_jobs=-1)


zoo_preview = build_zoo('Combined')
print('models available:', list(zoo_preview))
print('example Combined RandomForest steps:', [n for n, _ in zoo_preview['RandomForest'].steps])
print('example Combined SVC-RBF steps:', [n for n, _ in zoo_preview['SVC-RBF'].steps])


# ============================================================================
# CELL 23
# ============================================================================
# @title 10.1 Metrics
from sklearn.metrics import (accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score,
                             f1_score, balanced_accuracy_score, confusion_matrix, roc_curve)


def score_for(pipe, X):
    if hasattr(pipe, 'predict_proba'):
        return np.asarray(pipe.predict_proba(X)[:, 1], dtype=float)
    if hasattr(pipe, 'decision_function'):
        return np.asarray(pipe.decision_function(X), dtype=float)
    return np.asarray(pipe.predict(X), dtype=float)


def compute_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    out = {}
    out['acc'] = accuracy_score(y_true, y_pred)
    out['mcc'] = matthews_corrcoef(y_true, y_pred)
    out['f1'] = f1_score(y_true, y_pred, zero_division=0)
    out['bacc'] = balanced_accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out['sens'] = tp / (tp + fn) if (tp + fn) else np.nan
    out['spec'] = tn / (tn + fp) if (tn + fp) else np.nan
    if len(np.unique(y_true)) > 1:
        out['auroc'] = roc_auc_score(y_true, y_score)
        out['auprc'] = average_precision_score(y_true, y_score)
    else:
        out['auroc'] = np.nan
        out['auprc'] = np.nan
    out['y_pred'] = y_pred
    out['y_score'] = y_score
    return out


METRIC_KEYS = ['acc', 'mcc', 'auroc', 'auprc', 'f1', 'bacc', 'sens', 'spec']
# our short metric names -> the names sklearn's search / importance API expects
SKLEARN_SCORER = {'mcc': 'matthews_corrcoef', 'auroc': 'roc_auc', 'acc': 'accuracy',
                  'bacc': 'balanced_accuracy', 'auprc': 'average_precision', 'f1': 'f1'}
assert PRIMARY_METRIC in SKLEARN_SCORER
print('metrics ready:', METRIC_KEYS)


# ============================================================================
# CELL 24
# ============================================================================
# @title 10.2 Split factory (random or cluster-group aware)
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold

groups = df['cluster'].values if GROUP_AWARE_SPLIT else None


def _group_stratified_split(y_, groups_, test_frac, seed, max_rounds=60):
    # greedy cluster assignment that hits the requested test size and keeps both classes present
    n = len(y_)
    target = int(round(test_frac * n))
    want_pos = target * (y_.sum() / n)
    for attempt in range(max_rounds):
        rng = np.random.default_rng(seed * 1000 + attempt)
        uq = np.unique(groups_)
        rng.shuffle(uq)
        chosen, cur, cur_pos = [], 0, 0
        for g in uq:
            if cur >= target:
                break
            sz = int((groups_ == g).sum())
            np_ = int(y_[groups_ == g].sum())
            if cur >= want_pos and np_ == 0 and len(chosen) >= 2:
                continue
            chosen.append(g)
            cur += sz
            cur_pos += np_
        if cur_pos == 0 or cur_pos == y_.sum():
            continue
        mask = np.isin(groups_, chosen)
        te = np.where(mask)[0]
        tr = np.where(~mask)[0]
        if len(te) == 0 or len(tr) == 0:
            continue
        if len(np.unique(y_[te])) == 2 and len(np.unique(y_[tr])) == 2:
            return tr, te
    return train_test_split(np.arange(n), test_size=test_frac, stratify=y_, random_state=seed)


def make_split(train_frac, test_frac, seed):
    # returns (train_idx, test_idx); test_idx is None for the 100/0 case
    idx = np.arange(len(y))
    if test_frac <= 0:
        return idx, None
    if groups is not None:
        tr, te = _group_stratified_split(y, groups, test_frac, seed)
    else:
        tr, te = train_test_split(idx, test_size=test_frac, stratify=y, random_state=seed)
    return np.asarray(tr), np.asarray(te)


print('split type: %s' % ('cluster-group aware' if GROUP_AWARE_SPLIT else 'random'))
print('%9s %8s %7s %11s %10s' % ('split', 'n_train', 'n_test', 'train act%', 'test act%'))
for tf_tr, tf_te in SPLIT_PAIRS:
    tr, te = make_split(tf_tr, tf_te, SEED)
    s_tr = '%.1f%%' % (100 * y[tr].mean())
    s_te = '%.1f%%' % (100 * y[te].mean()) if te is not None else 'CV only'
    print('%4d/%-4d %8d %7s %11s %10s'
          % (int(tf_tr * 100), int(tf_te * 100), len(tr), '-' if te is None else len(te), s_tr, s_te))


# ============================================================================
# CELL 25
# ============================================================================
# @title 10.3 Stage A - screening harness (all splits x seeds x models)
from sklearn.base import clone


def evaluate_model_cv(model, X, y_, seed, repeats=CV_REPEATS, folds=CV_FOLDS):
    # used for the 100/0 case: repeated stratified CV, out-of-fold scores pooled
    oof = np.zeros(len(y_), dtype=float)
    cnt = np.zeros(len(y_), dtype=float)
    for rep in range(repeats):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + 1000 * rep)
        for tr, va in skf.split(X, y_):
            m = clone(model)
            m.fit(X[tr], y_[tr])
            oof[va] += score_for(m, X[va])
            cnt[va] += 1
    return compute_metrics(y_, oof / np.maximum(cnt, 1))


def run_benchmark(feature_block, models, tag='screenA', split_pairs=None, seeds=None):
    X = X_blocks[feature_block]
    zoo_ = build_zoo(feature_block, models=models)
    split_pairs = split_pairs or SPLIT_PAIRS
    seeds = seeds or SCREEN_SEEDS
    rows = []
    t0 = time.time()
    for tf_tr, tf_te in split_pairs:
        for seed in seeds:
            tr, te = make_split(tf_tr, tf_te, seed)
            for mname, proto in zoo_.items():
                try:
                    if te is None:
                        m = evaluate_model_cv(proto, X[tr], y[tr], seed)
                    else:
                        mdl = clone(proto)
                        mdl.fit(X[tr], y[tr])
                        m = compute_metrics(y[te], score_for(mdl, X[te]))
                    rows.append(dict(feature_block=feature_block, model=mname, train_frac=tf_tr,
                                     test_frac=tf_te, seed=seed, n_train=len(tr),
                                     n_test=(0 if te is None else len(te)),
                                     **{k: m[k] for k in METRIC_KEYS}))
                except Exception as e:
                    print('  !! %s on %s/%s seed %s: %s' % (mname, tf_tr, tf_te, seed, repr(e)[:130]))
        print('  split %d/%d finished (%.0fs)' % (int(tf_tr * 100), int(tf_te * 100), time.time() - t0), flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(WORKDIR, 'results_%s_%s.csv' % (tag, feature_block)), index=False)
    return res


SCREEN_MODELS = [m for m in ['LogReg', 'SVC-RBF', 'DecisionTree', 'RandomForest', 'ExtraTrees',
                             'GradientBoosting', 'LDA-shrunk', 'GaussianNB', 'kNN', 'MLP',
                             'XGBoost', 'LightGBM', 'CatBoost', 'Voting-soft']
                 if (m in BASE_MODELS or m == 'Voting-soft')]
print('screening models:', SCREEN_MODELS)
print('fits to run:', len(SPLIT_PAIRS) * len(SCREEN_SEEDS) * len(SCREEN_MODELS) * len(CV_REPORT_BLOCKS))


# ============================================================================
# CELL 26
# ============================================================================
# @title 10.4 Run the screening benchmark
results = {}
for block in CV_REPORT_BLOCKS:
    if block not in X_blocks:
        continue
    print('=== feature block:', block)
    results[block] = run_benchmark(block, SCREEN_MODELS, tag='screenA')
    print()

res_all = pd.concat(results.values(), ignore_index=True)
res_all.to_csv(os.path.join(WORKDIR, 'results_screenA_all.csv'), index=False)
print('total result rows:', len(res_all))


# ============================================================================
# CELL 27
# ============================================================================
# @title 10.5 Results: mean +/- std over seeds, ranked by MCC
def summarise(res, index=('feature_block', 'model')):
    g = res.groupby(list(index))[METRIC_KEYS]
    out = g.agg(['mean', 'std'])
    out.columns = ['%s_%s' % (a, b) for a, b in out.columns]
    return out.sort_values('mcc_mean', ascending=False)


def fmt(summary, cols=('mcc_mean', 'auroc_mean', 'acc_mean')):
    d = summary.copy()
    for c in cols:
        base = c.replace('_mean', '')
        s = d.get(base + '_std', pd.Series(np.nan, index=d.index))
        d[c] = ['%.3f+-%.3f' % (m, 0.0 if pd.isna(sd) else sd) for m, sd in zip(d[c], s)]
    return d[list(cols)]


summary_all = summarise(res_all)
print('=== overall ranking, all splits pooled ===')
print(fmt(summary_all).to_string())
print()
for block in res_all['feature_block'].unique():
    print('=== feature-block ablation: %s ===' % block)
    print(fmt(summarise(res_all[res_all.feature_block == block])).to_string())
    print()


# ============================================================================
# CELL 28
# ============================================================================
# @title 10.6 Plots: metric vs split, model x split heatmap, ROC, confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns


def plot_metric_vs_split(res, metric='mcc', top=8, block_tag=''):
    piv = res.pivot_table(index='model', columns='test_frac', values=metric, aggfunc='mean')
    order = res.groupby('model')[metric].mean().sort_values(ascending=False).head(top).index
    order = [m for m in order if m in piv.index]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for m in order:
        ax[0].plot([c * 100 for c in piv.columns], piv.loc[m], marker='o', label=m)
    ax[0].set_xlabel('test fraction (%)')
    ax[0].set_ylabel(metric.upper())
    ax[0].set_title('%s vs train/test split [%s] (mean of seeds)' % (metric.upper(), block_tag))
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    sns.heatmap(piv.loc[order], annot=True, fmt='.2f', cmap='viridis', ax=ax[1], cbar_kws={'label': metric.upper()})
    ax[1].set_xlabel('test fraction')
    ax[1].set_title('model x split')
    plt.tight_layout()
    plt.show()


main_block = res_all['feature_block'].value_counts().index[0]
sub = res_all[res_all.feature_block == main_block]
plot_metric_vs_split(sub, 'mcc', top=8, block_tag=main_block)
plot_metric_vs_split(sub, 'auroc', top=8, block_tag=main_block)

BEST_MODEL = summary_all.index[0][1]
BEST_BLOCK = summary_all.index[0][0]
print('best pooled MCC:', BEST_BLOCK, '/', BEST_MODEL)

tr, te = make_split(0.80, 0.20, SEED)
Xb = X_blocks[BEST_BLOCK]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
for mname in [m for m in SCREEN_MODELS if m != 'Voting-soft']:
    try:
        mdl = clone(build_zoo(BEST_BLOCK, models=[mname], include_ensemble=False)[mname])
        mdl.fit(Xb[tr], y[tr])
        s = score_for(mdl, Xb[te])
        fpr, tpr, _ = roc_curve(y[te], s)
        axes[0].plot(fpr, tpr, lw=1.4, label='%s (AUC %.3f)' % (mname, roc_auc_score(y[te], s)))
    except Exception as e:
        print(mname, repr(e)[:100])
axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
axes[0].set_xlabel('false positive rate')
axes[0].set_ylabel('true positive rate')
axes[0].set_title('ROC on the 80/20 hold-out')
axes[0].legend(fontsize=7, loc='lower right')

mdl = clone(build_zoo(BEST_BLOCK, models=[BEST_MODEL] if BEST_MODEL in BASE_MODELS else None,
                      include_ensemble=True)[BEST_MODEL])
mdl.fit(Xb[tr], y[tr])
m = compute_metrics(y[te], score_for(mdl, Xb[te]))
cm = confusion_matrix(y[te], m['y_pred'], labels=[0, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[1],
            xticklabels=['Inactive', 'Active'], yticklabels=['Inactive', 'Active'])
axes[1].set_title('%s: MCC %.3f | AUROC %.3f' % (BEST_MODEL, m['mcc'], m['auroc']))
plt.tight_layout()
plt.show()
print({k: round(m[k], 4) for k in METRIC_KEYS})


# ============================================================================
# CELL 29 (markdown, commented out)
# ============================================================================
# # 11. Proposal 2.4.3 - hyper-parameter optimization by 5-fold grid search
#
# Randomized search with `CV_FOLDS = 5` cross-validation, executed **only on the training portion of the 80/20 split**, maximising `PRIMARY_METRIC` (`'mcc'`; set it to `'roc_auc'` to optimise AUROC instead). The tuned pipelines are then re-scored on the whole split ladder with 5 seeds. The outer test set is therefore never used for any decision.
#
# The four axes the proposal requires are all searched, with the library-specific equivalent named where they differ:
#
# | proposal axis | RandomForest | XGBoost | LightGBM |
# |---|---|---|---|
# | `n_estimators` | `n_estimators` | `n_estimators` | `n_estimators` |
# | `max_depth` | `max_depth` | `max_depth` | `max_depth` |
# | `min_samples_split` | `min_samples_split` | `min_child_weight` | `min_child_samples` |
# | `learning_rate` | n/a (bagging) | `learning_rate` | `learning_rate` |


# ============================================================================
# CELL 30
# ============================================================================
# @title 11.1 Parameter grids + randomized search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint, uniform

PARAM_GRIDS = {
    'DecisionTree': {'clf__max_depth': [2, 3, 4, 6, 8, 12, None],
                     'clf__min_samples_split': randint(2, 20),
                     'clf__min_samples_leaf': randint(1, 8),
                     'clf__criterion': ['gini', 'entropy'],
                     'clf__class_weight': [None, 'balanced']},
    'RandomForest': {'clf__n_estimators': [300, 500, 800, 1200],
                     'clf__max_depth': [3, 5, 8, 12, None],
                     'clf__min_samples_split': randint(2, 12),
                     'clf__min_samples_leaf': randint(1, 6),
                     'clf__max_features': ['sqrt', 'log2', 0.1, 0.3, 0.6]},
    'ExtraTrees': {'clf__n_estimators': [300, 500, 800, 1200],
                   'clf__max_depth': [3, 5, 8, 12, None],
                   'clf__min_samples_leaf': randint(1, 6),
                   'clf__max_features': ['sqrt', 0.1, 0.3, 0.6]},
    'SVC-RBF': {'clf__C': loguniform(1e-1, 1e3),
                'clf__gamma': ['scale', 'auto', loguniform(1e-4, 1e-1)],
                'clf__class_weight': [None, 'balanced']},
    'LogReg': {'clf__C': loguniform(1e-3, 1e3),
               'clf__class_weight': [None, 'balanced']},
    'XGBoost': {'clf__n_estimators': [200, 400, 800],
                'clf__max_depth': randint(2, 8),
                'clf__learning_rate': loguniform(0.01, 0.3),
                'clf__subsample': uniform(0.6, 0.4),
                'clf__colsample_bytree': uniform(0.2, 0.8),
                'clf__reg_lambda': loguniform(1e-2, 1e2),
                'clf__min_child_weight': randint(1, 10)},
    'LightGBM': {'clf__n_estimators': [200, 400, 800],
                 'clf__num_leaves': randint(7, 63),
                 'clf__max_depth': randint(3, 12),
                 'clf__learning_rate': loguniform(0.01, 0.3),
                 'clf__subsample': uniform(0.6, 0.4),
                 'clf__colsample_bytree': uniform(0.2, 0.8),
                 'clf__min_child_samples': randint(3, 20),
                 'clf__reg_lambda': loguniform(1e-2, 1e2)},
    'CatBoost': {'clf__iterations': [200, 400, 800],
                 'clf__depth': randint(3, 8),
                 'clf__learning_rate': loguniform(0.01, 0.3),
                 'clf__l2_leaf_reg': loguniform(1, 30)},
    'MLP': {'clf__hidden_layer_sizes': [(64,), (128,), (128, 64), (256, 128)],
            'clf__alpha': loguniform(1e-4, 1e0),
            'clf__learning_rate_init': loguniform(1e-4, 1e-2)},
    'GradientBoosting': {'clf__n_estimators': [100, 200, 400],
                         'clf__max_depth': randint(2, 6),
                         'clf__learning_rate': loguniform(0.01, 0.3),
                         'clf__subsample': uniform(0.6, 0.4)},
}


def tune_model(block, model_name, seed=SEED, n_iter=TUNING_N_ITER, train_frac=0.80):
    X = X_blocks[block]
    tr, te = make_split(train_frac, 1.0 - train_frac, seed)
    pipe = build_zoo(block, models=[model_name], include_ensemble=False)[model_name]
    grid = PARAM_GRIDS.get(model_name)
    if grid is None:
        return pipe, None, None
    search = RandomizedSearchCV(pipe, grid, n_iter=n_iter, scoring=SKLEARN_SCORER[PRIMARY_METRIC],
                                cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed),
                                refit=True, n_jobs=-1, random_state=seed)
    if GROUP_AWARE_SPLIT:
        try:
            search.set_params(cv=StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed))
            search.fit(X[tr], y[tr], groups=df['cluster'].values[tr])
        except Exception as e:
            print('grouped CV failed for %s (%s) -> StratifiedKFold' % (model_name, repr(e)[:70]))
            search.set_params(cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed))
            search.fit(X[tr], y[tr])
    else:
        search.fit(X[tr], y[tr])
    return search.best_estimator_, search.best_params_, float(search.best_score_)


TOP_MODELS = [m for m in dict.fromkeys(list(summary_all.index.get_level_values('model'))[:4]) if m in PARAM_GRIDS]
print('tuning these models:', TOP_MODELS, '| block:', BEST_BLOCK, '| metric:', PRIMARY_METRIC)

tuned, tuned_params, tuned_scores = {}, {}, {}
for mname in TOP_MODELS:
    t0 = time.time()
    est, params, score = tune_model(BEST_BLOCK, mname)
    tuned[mname] = est
    tuned_params[mname] = params
    tuned_scores[mname] = score
    print('  %-16s best CV %s = %.4f  (%.0fs)' % (mname, PRIMARY_METRIC, score, time.time() - t0))
    print('     ', params)

pd.DataFrame([{'model': k, 'best_params': str(v), 'cv_%s' % PRIMARY_METRIC: s}
              for k, v, s in zip(tuned_params, tuned_params.values(), tuned_scores.values())]) \
    .to_csv(os.path.join(WORKDIR, 'tuned_params.csv'), index=False)


# ============================================================================
# CELL 31
# ============================================================================
# @title 11.2 Re-evaluate the tuned pipelines on the full split ladder (5 seeds)
def run_tuned(pipeline, block, tag):
    X = X_blocks[block]
    rows = []
    t0 = time.time()
    for tf_tr, tf_te in SPLIT_PAIRS:
        for seed in SEEDS:
            tr, te = make_split(tf_tr, tf_te, seed)
            if te is None:
                m = evaluate_model_cv(pipeline, X[tr], y[tr], seed)
            else:
                mdl = clone(pipeline)
                mdl.fit(X[tr], y[tr])
                m = compute_metrics(y[te], score_for(mdl, X[te]))
            rows.append(dict(feature_block=block, model=tag, train_frac=tf_tr, test_frac=tf_te,
                             seed=seed, n_train=len(tr), n_test=(0 if te is None else len(te)),
                             **{k: m[k] for k in METRIC_KEYS}))
        print('  split %d/%d finished (%.0fs)' % (int(tf_tr * 100), int(tf_te * 100), time.time() - t0), flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(WORKDIR, 'results_%s_%s.csv' % (tag, block)), index=False)
    return res


res_tuned = {}
for mname, est in tuned.items():
    print('=== tuned %s' % mname)
    res_tuned[mname] = run_tuned(est, BEST_BLOCK, 'tuned-' + mname)
res_tuned_all = pd.concat(res_tuned.values(), ignore_index=True)
res_tuned_all.to_csv(os.path.join(WORKDIR, 'results_tuned_all.csv'), index=False)

print()
print('=== tuned models, MCC per split (mean over 5 seeds) ===')
print(res_tuned_all.pivot_table(index='model', columns='test_frac', values='mcc', aggfunc='mean').round(3).to_string())
print()
print('=== tuned models, AUROC per split (mean over 5 seeds) ===')
print(res_tuned_all.pivot_table(index='model', columns='test_frac', values='auroc', aggfunc='mean').round(3).to_string())


# ============================================================================
# CELL 32 (markdown, commented out)
# ============================================================================
# # 12. Proposal 2.4.4 - primary pipeline screening gate
#
# The final model is the tuned pipeline with the best mean MCC across the whole split ladder, refitted on all rows. This single model is the **master prediction gate** for section 2.5. The headline table below is the one to paste into the proposal.
#
# The gate needs a decision threshold. The proposal fixes 0.70 as the nominal value and, in 2.5.4, asks for the threshold that **maximises Youden's index on the validation ROC**. Both are computed here from the pooled out-of-fold scores - never from the training fit - and written to `screening_gate.json`, so 2.5 filters de novo candidates with a threshold that came out of validation, not out of the air.


# ============================================================================
# CELL 33
# ============================================================================
# @title 12.1 Select the final pipeline and build the headline table
rank_tuned = res_tuned_all.groupby('model')[METRIC_KEYS].mean().sort_values('mcc', ascending=False)
print('=== tuned ranking, mean over all splits and seeds ===')
print(rank_tuned.round(4).to_string())
print()

WINNER = rank_tuned.index[0]
FINAL_PIPELINE = tuned[WINNER.replace('tuned-', '')]
print('selected final model:', WINNER, 'on feature block', BEST_BLOCK)
print('best hyperparameters:', tuned_params[WINNER.replace('tuned-', '')])
print()

headline = res_tuned_all[res_tuned_all.model == WINNER]
tab = headline.groupby(['train_frac', 'test_frac']).agg(
    seeds=('mcc', 'size'),
    ACC=('acc', 'mean'), ACC_sd=('acc', 'std'),
    MCC=('mcc', 'mean'), MCC_sd=('mcc', 'std'),
    AUROC=('auroc', 'mean'), AUROC_sd=('auroc', 'std'),
    AUPRC=('auprc', 'mean'), F1=('f1', 'mean'),
    Sens=('sens', 'mean'), Spec=('spec', 'mean'), BACC=('bacc', 'mean')).round(4)
print('=== headline table: %s ===' % WINNER)
print(tab.to_string())
tab.to_csv(os.path.join(WORKDIR, 'headline_table.csv'))
print()

appendix = res_tuned_all.groupby(['model', 'train_frac', 'test_frac'])[METRIC_KEYS].agg(['mean', 'std']).round(4)
appendix.to_csv(os.path.join(WORKDIR, 'appendix_table.csv'))
print('=== appendix: all tuned models x all splits (MCC / AUROC / ACC) ===')
print(appendix[['mcc', 'auroc', 'acc']].to_string())


# ============================================================================
# CELL 34
# ============================================================================
# @title 12.2 Refit on all data, 5-fold CV, Youden threshold, bootstrap CIs, save artefacts
import joblib

X_best = X_blocks[BEST_BLOCK]

cv_out = {k: [] for k in ['mcc', 'auroc', 'acc', 'auprc']}
oof = np.zeros(len(y))
oof_cnt = np.zeros(len(y))
for tr, va in StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED).split(X_best, y):
    mdl = clone(FINAL_PIPELINE)
    mdl.fit(X_best[tr], y[tr])
    s = score_for(mdl, X_best[va])
    oof[va] += s
    oof_cnt[va] += 1
    mm = compute_metrics(y[va], s)
    for k in cv_out:
        cv_out[k].append(mm[k])
oof_pred = oof / np.maximum(oof_cnt, 1)
m_oof = compute_metrics(y, oof_pred)

# --- proposal 2.4.4 / 2.5.4: the screening threshold ------------------------------
# Youden's index J(t) = TPR(t) - FPR(t) maximised on the pooled out-of-fold ROC, so the
# threshold is chosen on validation predictions and never on the refit training set
fpr_oof, tpr_oof, thr_oof = roc_curve(y, oof_pred)
youden = tpr_oof - fpr_oof
THR_YOUDEN = float(thr_oof[int(np.argmax(youden))]) if len(thr_oof) else 0.5
THR_PROPOSAL = 0.70
m_youden = compute_metrics(y, oof_pred, threshold=THR_YOUDEN)

FINAL_PIPELINE.fit(X_best, y)
joblib.dump({'pipeline': FINAL_PIPELINE, 'feature_block': BEST_BLOCK,
             'rdkit_cols': RDKIT_ALL_COLS, 'pf_stat_names': PF_STAT_NAMES,
             'pf_pool_stats': list(PF_POOL_STATS), 'pf_chiralities': bool(PF_CHIRALITIES),
             'cluster_identity': CLUSTER_IDENTITY, 'std_scaler': 'in-pipeline',
             'threshold_youden': THR_YOUDEN, 'threshold_proposal': THR_PROPOSAL,
             'feature_manifest': FEATURE_MANIFEST},
            os.path.join(WORKDIR, 'final_model.joblib'))

with open(os.path.join(WORKDIR, 'screening_gate.json'), 'w', encoding='utf-8') as f:
    json.dump({'model': WINNER, 'feature_block': BEST_BLOCK,
               'threshold_youden': THR_YOUDEN,
               'youden_index_at_threshold': float(np.max(youden)) if len(youden) else None,
               'threshold_proposal_nominal': THR_PROPOSAL,
               'oof_mcc_at_youden': m_youden['mcc'], 'oof_auroc': m_oof['auroc'],
               'oof_sens_at_youden': m_youden['sens'], 'oof_spec_at_youden': m_youden['spec'],
               'n_train_oof': int(len(y))}, f, indent=2)

print('=== %d-fold CV of the final pipeline (n = %d) ===' % (CV_FOLDS, len(y)))
for k, v in cv_out.items():
    print('  %-6s %.3f +- %.3f' % (k, float(np.nanmean(v)), float(np.nanstd(v))))
print()
print('=== pooled out-of-fold metrics (single 5-fold run) ===')
print({k: round(m_oof[k], 4) for k in METRIC_KEYS})
print()
print('=== screening gate (proposal 2.4.4 / 2.5.4) ===')
print('  Youden-optimal threshold on the out-of-fold ROC : %.4f  (J = %.3f)' % (THR_YOUDEN, float(np.max(youden))))
print('  at that threshold -> MCC %.3f | sens %.3f | spec %.3f'
      % (m_youden['mcc'], m_youden['sens'], m_youden['spec']))
print('  proposal nominal threshold                      : %.2f' % THR_PROPOSAL)
print('  saved screening_gate.json - section 2.5 filters with these numbers')
print()
print(pd.DataFrame({'y_true': y, 'p_active': oof_pred}).assign(correct=lambda d: (d.p_active >= 0.5).astype(int) == d.y_true).head(10).to_string(index=False))

rng = np.random.default_rng(SEED)
boot = {k: [] for k in ['mcc', 'auroc', 'acc']}
s_full = score_for(FINAL_PIPELINE, X_best)
for _ in range(N_BOOTSTRAP):
    idx = rng.integers(0, len(y), len(y))
    if len(np.unique(y[idx])) < 2:
        continue
    mb = compute_metrics(y[idx], s_full[idx])
    for k in boot:
        boot[k].append(mb[k])
print()
print('=== in-sample bootstrap 95% CIs (optimistic - sanity check only) ===')
for k, v in boot.items():
    print('  %-6s %.3f - %.3f' % (k, float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))))


# ============================================================================
# CELL 35 (markdown, commented out)
# ============================================================================
# # 13. Proposal 2.4.4 - post-hoc SHAP interpretability
#
# Three views of the same Random Forest:
#
# 1. **TreeSHAP on the physicochemical block** - a human readable ranking (net charge, hydrophobic moment, LogP, TPSA, aromaticity, chiral centres ...) with a beeswarm plot showing the *direction* of every effect. This is the panel that supports the chemistry narrative in the proposal, and it is the panel section 2.5.2 filters on.
# 2. **TreeSHAP on the PepFoundry graph block** - ranks the 160 pooled graph statistics, and the top SVD components are mapped back to the underlying atom/bond statistics so a black-box split becomes an interpretable chemical statement.
# 3. **Per-residue attribution** - PepFoundry's atom mapping assigns every atom of the molecule to a residue, so the atoms of residue *i* are aggregated into a residue-level vector. The contribution of residue *i* is the drop in predicted activity when its atoms are replaced by the dataset-average residue atoms and the pooling is recomputed. This is the exact mapping from graph feature back to physical sequence position requested in the proposal.
#
# Cell 13.1 additionally writes **`shap_physicochemical_patterns.csv`**: per descriptor, the mean signed SHAP in each class, the value range of each class, and the direction the model wants. That file is the machine-readable version of "high prediction probabilities are driven by valid biological and physicochemical properties", and it is what 2.5.2 turns into pass/fail rules for de novo candidates.


# ============================================================================
# CELL 36
# ============================================================================
# @title 13.1 TreeSHAP global explanations
import shap
from sklearn.inspection import permutation_importance

TREE_TYPES = ('RandomForestClassifier', 'ExtraTreesClassifier', 'GradientBoostingClassifier',
              'LGBMClassifier', 'XGBClassifier', 'CatBoostClassifier', 'DecisionTreeClassifier')


def shap_explain(pipeline, X, feature_names=None, sample=300, seed=SEED, title=''):
    Xt = np.asarray(pipeline[:-1].transform(X))
    clf = pipeline['clf']
    if feature_names is None or len(feature_names) != Xt.shape[1]:
        # the graph block is compressed by SafeSVD, so name the *transformed* columns
        feature_names = ['%s_f%d' % (title.split()[0].lower(), i) for i in range(Xt.shape[1])]
        print('%s: design matrix is %d columns wide -> using %s'
              % (title, Xt.shape[1], 'SVD component' if Xt.shape[1] < X.shape[1] else 'raw feature'))
    if type(clf).__name__ in TREE_TYPES:
        expl = shap.TreeExplainer(clf)
        sv = expl.shap_values(Xt[:sample])
        base = float(np.ravel(expl.expected_value)[-1]) if np.size(expl.expected_value) > 1 else float(expl.expected_value)
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, -1]       # shap >= 0.45 returns (n_samples, n_features, n_classes)
        elif sv.ndim == 1:
            sv = sv.reshape(Xt[:sample].shape[0], -1)
    else:
        # non-tree fallback: permutation importance on the same design matrix
        pi = permutation_importance(clf, Xt, y, n_repeats=10, random_state=seed, scoring='roc_auc')
        imp = pd.Series(pi.importances_mean, index=feature_names).sort_values(ascending=False)
        print('--- %s: %s is not tree based -> permutation importance shown instead ---' % (title, type(clf).__name__))
        print(imp.head(25).to_string())
        plt.figure(figsize=(8, 6))
        plt.barh(imp.head(25).index[::-1], imp.head(25).values[::-1], color='tab:blue')
        plt.xlabel('permutation importance (AUROC drop)')
        plt.title('Feature importance: %s' % title)
        plt.tight_layout()
        plt.show()
        return None, imp
    if isinstance(sv, list):
        sv = np.asarray(sv[-1])
    print('--- SHAP for %s | base value (expected output) = %.4f ---' % (title, base))
    shap.summary_plot(sv, Xt[:sample], feature_names=feature_names, max_display=25, show=False)
    plt.title('TreeSHAP attribution: %s' % title)
    plt.tight_layout()
    plt.show()
    imp = pd.Series(np.abs(sv).mean(axis=0), index=feature_names).sort_values(ascending=False)
    print('mean(|SHAP|), top 15:')
    print(imp.head(15).to_string())
    return sv, imp


shap_tables = {}
sv_d = None   # physicochemical SHAP matrix, kept for the 2.5.2 pattern table

rf_proto = tuned.get('RandomForest', build_zoo(BEST_BLOCK, models=['RandomForest'],
                                               include_ensemble=False)['RandomForest'])


def retarget(pipe, block):
    # keep the tuned classifier, swap in the preprocessing of another feature block
    base = build_zoo(block, models=['RandomForest'], include_ensemble=False)['RandomForest']
    return Pipeline(list(base.steps[:-1]) + [('clf', clone(pipe['clf']))])


# (a) RDKit block
try:
    pipe_d = retarget(rf_proto, 'RDKit')
    pipe_d.fit(X_blocks['RDKit'], y)
    sv_d, imp_d = shap_explain(pipe_d, X_blocks['RDKit'], RDKIT_ALL_COLS, title='RDKit')
    shap_tables['rdkit'] = imp_d
except Exception as e:
    print('RDKit SHAP failed:', repr(e)[:200])

# (b) PepFoundry graph block (compressed by SafeSVD -> the SHAP values belong to SVD components)
if USE_PEPFOUNDRY:
    try:
        pipe_e = retarget(rf_proto, 'PepFoundry')
        pipe_e.fit(X_blocks['PepFoundry'], y)
        sv_e, imp_e = shap_explain(pipe_e, X_blocks['PepFoundry'], title='PepFoundry')
        shap_tables['pepfoundry'] = imp_e
        # map the most important SVD components back to the pooled atom/bond statistics
        comps = pipe_e['svd'].svd_.components_
        print()
        print('=== top SVD components mapped back to the pooled PepFoundry statistics ===')
        for fname in imp_e.head(8).index:
            k = int(fname.split('_f')[-1])
            load = comps[k]
            top_dims = np.argsort(-np.abs(load))[:8]
            print('%-18s (explained variance %.3f) <- %s'
                  % (fname, pipe_e['svd'].svd_.explained_variance_ratio_[k],
                     [PF_STAT_NAMES[PF_KEEP_IDX[i]] for i in top_dims]))
    except Exception as e:
        print('PepFoundry SHAP failed:', repr(e)[:200])

if shap_tables:
    pd.concat(shap_tables, axis=1).to_csv(os.path.join(WORKDIR, 'shap_importance.csv'))
    print('\nsaved shap_importance.csv')

# --- proposal 2.5.2: the physicochemical attribution pattern, as a filter table -------
# For every physicochemical descriptor: how much the model pushes a peptide towards
# Active or Inactive, and which direction in descriptor space it wants to move in.
if sv_d is not None:
    _sv = np.asarray(sv_d)
    _y = DATASET_WITH_DESC['label'].values.astype(int)
    _raw = X_rdkit[RDKIT_ALL_COLS].values.astype(float)
    _pat = pd.DataFrame({
        'descriptor': list(RDKIT_ALL_COLS),
        'mean_shap_active': [_sv[_y == 1, j].mean() if (_y == 1).any() else np.nan for j in range(_sv.shape[1])],
        'mean_shap_inactive': [_sv[_y == 0, j].mean() if (_y == 0).any() else np.nan for j in range(_sv.shape[1])],
        'mean_abs_shap': np.abs(_sv).mean(axis=0),
    })
    _pat['shap_gap'] = _pat['mean_shap_active'] - _pat['mean_shap_inactive']
    _pat['favours'] = np.where(_pat['shap_gap'] > 0, 'Active', 'Inactive')
    _pat['active_mean'] = [_raw[_y == 1, j].mean() if (_y == 1).any() else np.nan for j in range(_raw.shape[1])]
    _pat['inactive_mean'] = [_raw[_y == 0, j].mean() if (_y == 0).any() else np.nan for j in range(_raw.shape[1])]
    _pat['direction'] = np.where(_pat['active_mean'] > _pat['inactive_mean'],
                                  'higher favours Active', 'lower favours Active')
    _pat['group'] = [c.split('_')[0] for c in RDKIT_ALL_COLS]
    SHAP_PATTERNS = _pat.sort_values('mean_abs_shap', ascending=False)
    SHAP_PATTERNS.to_csv(os.path.join(WORKDIR, 'shap_physicochemical_patterns.csv'))
    pd.DataFrame(AMP_COLS, columns=['descriptor']).merge(SHAP_PATTERNS, on='descriptor', how='left') \
        .to_csv(os.path.join(WORKDIR, 'shap_amp_pharmacophore_patterns.csv'), index=False)
    print()
    print('=== saved shap_physicochemical_patterns.csv (section 2.5.2 filter rules) ===')
    print(SHAP_PATTERNS.head(15).round(4).to_string(index=False))


# ============================================================================
# CELL 37
# ============================================================================
# @title 13.2 Per-residue attribution: PepFoundry atom mapping -> sequence positions
def residue_atom_groups(mol, n_res):
    # PepFoundry assigns every heavy atom to a residue, so occlusion can be done per residue
    amap = np.asarray(pf_engine().get_amino_acid_atom_mapping(mol)).ravel()
    return [np.where(amap == i)[0] for i in range(n_res)]


class PerResidueExplainer:
    # occlusion of residue i: its atoms are replaced by the average atom profile of that
    # residue position in the dataset, the pooling is recomputed, and the drop in
    # predicted activity is the contribution of that residue.
    def __init__(self, pipeline, block, base_X, n_avg=60, seed=SEED):
        self.pipe = pipeline
        self.block = block
        self.d_rdkit = X_blocks['RDKit'].shape[1] if block == 'Combined' else 0
        self.base = np.asarray(base_X, dtype=float)
        self.pf_keep = list(PF_KEEP_IDX)
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(seqs), size=min(n_avg, len(seqs)), replace=False)
        ref = [[] for _ in range(max(len(s) for s in seqs))]
        atom_dim = 24
        t0 = time.time()
        for i in pick:
            s = seqs[i]
            m = pf_mol(s)
            a, _ = pf_tensors(m)
            atom_dim = a.shape[1]
            groups = residue_atom_groups(m, len(s))
            for j, rows in enumerate(groups):
                ref[j].append(a[rows].mean(axis=0) if len(rows) else np.zeros(atom_dim))
        self.avg_atom = [np.mean(v, axis=0) if v else np.zeros(atom_dim) for v in ref]
        print('reference residue profiles from %d peptides in %.0fs' % (len(pick), time.time() - t0))

    def explain(self, row_idx):
        seq = seqs[row_idx]
        m = pf_mol(seq)
        A, B = pf_tensors(m)
        groups = residue_atom_groups(m, len(seq))
        p0 = float(score_for(self.pipe, self.base[[row_idx]])[0])
        contrib = np.zeros(len(seq))
        for i, rows in enumerate(groups):
            A_mod = A.copy()
            if len(rows):
                A_mod[rows] = self.avg_atom[i]              # occlude this residue
            pooled = pf_pool(A_mod, B)[self.pf_keep]         # same columns as training
            Xmod = self.base[[row_idx]].copy()
            if self.block == 'Combined':
                Xmod[0, self.d_rdkit:] = pooled
            else:
                Xmod[0, :] = pooled
            contrib[i] = p0 - float(score_for(self.pipe, Xmod)[0])
        return seq, p0, contrib


if 'PepFoundry' not in X_blocks:
    print('per-residue attribution needs the PepFoundry block -> cell skipped')
else:
    if BEST_BLOCK == 'RDKit':
        # the winning model ignores the graph block, so attribute a graph model instead
        ATTR_BLOCK = 'PepFoundry'
        ATTR_PIPE = build_zoo('PepFoundry', models=['RandomForest'], include_ensemble=False)['RandomForest']
        ATTR_PIPE.fit(X_blocks['PepFoundry'], y)
        print('BEST_BLOCK is RDKit -> per-residue attribution uses a RandomForest on the PepFoundry block')
    else:
        ATTR_BLOCK = BEST_BLOCK
        FINAL_PIPELINE.fit(X_blocks[BEST_BLOCK], y)
        ATTR_PIPE = FINAL_PIPELINE
    explainer = PerResidueExplainer(ATTR_PIPE, ATTR_BLOCK, X_blocks[ATTR_BLOCK])
    demo_idx = list(np.argsort(-y)[:3]) + list(np.argsort(y)[:3])   # 3 actives + 3 inactives
    records = []
    for i in demo_idx:
        seq, p0, c = explainer.explain(i)
        print()
        print('peptide %s | label=%d | len=%d | p(active)=%.3f' % (df['pep_id'].iloc[i], y[i], len(seq), p0))
        print('seq     :', seq)
        print('contrib :', np.array2string(c, precision=4, max_line_width=250, separator=', '))
        records.append((df['pep_id'].iloc[i], int(y[i]), seq, p0, c))

    fig, axes = plt.subplots(len(records), 1, figsize=(14, 2.3 * len(records)), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, (pid, lab, seq, p0, c) in zip(axes, records):
        ax.bar(range(len(c)), c, color=['tab:red' if v < 0 else 'tab:blue' for v in c])
        ax.set_title('%s | label=%d | p(active)=%.2f | blue raises activity, red suppresses'
                     % (pid, lab, p0), fontsize=9)
        ax.set_ylabel('delta p')
        ax.set_xticks(range(len(seq)))
        ax.set_xticklabels(list(seq), fontsize=7)
        ax.axhline(0, color='k', lw=0.8)
    plt.tight_layout()
    plt.show()


# ============================================================================
# CELL 38
# ============================================================================
# @title 13.3 Where along the sequence does the model look?
rows = []
for pid, lab, seq, p0, c in records:
    for i, (aa, v) in enumerate(zip(seq, c)):
        rows.append(dict(pep=pid, label=lab, pos=i, rel_pos=i / max(1, len(seq) - 1), aa=aa, contrib=v))
pos_df = pd.DataFrame(rows)
pos_df['bin'] = pd.cut(pos_df['rel_pos'], bins=10, labels=False)

fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
sns.barplot(data=pos_df, x='bin', y='contrib', ax=axes[0], color='tab:blue', errorbar=None)
axes[0].set_xlabel('relative position decile (0 = N-terminus, 9 = C-terminus)')
axes[0].set_title('mean signed contribution by position')
axes[0].axhline(0, color='k', lw=0.8)

aa_tab = pos_df.groupby('aa')['contrib'].agg(['mean', 'count'])
aa_tab['mean_abs'] = pos_df.groupby('aa')['contrib'].apply(lambda s: s.abs().mean())
aa_tab = aa_tab[aa_tab['count'] >= 2].sort_values('mean_abs', ascending=False).head(15)
sns.barplot(data=aa_tab, x='aa', y='mean_abs', ax=axes[1], color='tab:orange')
axes[1].set_xlabel('residue type')
axes[1].set_title('mean |contribution| by residue type')
axes[1].tick_params(axis='x', rotation=45)

sns.boxplot(data=pos_df, x='label', y='contrib', ax=axes[2], palette=['tab:orange', 'tab:blue'])
axes[2].set_xticklabels(['Inactive (0)', 'Active (1)'])
axes[2].set_title('per-residue contribution by activity class')
axes[2].axhline(0, color='k', lw=0.8)
plt.tight_layout()
plt.show()

pos_df.to_csv(os.path.join(WORKDIR, 'per_residue_attribution.csv'), index=False)
print(pos_df.groupby('bin')['contrib'].mean().round(4).to_string())


# ============================================================================
# CELL 39 (markdown, commented out)
# ============================================================================
# # 14. Proposal 2.5.4 - save artefacts, re-score candidates, hand-off to 2.5
#
# `final_model.joblib` holds the fitted pipeline, the physicochemical column order, the PepFoundry pooling statistics, both decision thresholds and the full feature manifest. `predict_peptides()` takes raw peptide strings, rebuilds the identical molecule with PepFoundry, re-extracts the identical physicochemical + graph features, and returns the activity probability **together with every physicochemical descriptor**, i.e. exactly the "secondary feature extraction and re-evaluation" step 2.5.4 prescribes.
#
# `score_candidates()` is the section 2.5 entry point. Given a candidate pool it returns one row per peptide with
#
# * `p_active` from the 2.4.4 master gate,
# * `passes_youden` / `passes_proposal_0.70` from the two thresholds,
# * the physicochemical descriptor columns, so 2.5.2 can apply the TreeSHAP pattern rules,
# * `in_training_set`, a cheap novelty flag against the curated sequences (2.5.3),
#
# and writes `02_5_candidate_scores.csv`. Nothing in this cell refits anything - the model is frozen, exactly as 2.5 requires.


# ============================================================================
# CELL 40
# ============================================================================
# @title 14.1 Inference helper (secondary feature extraction + re-evaluation)
bundle = joblib.load(os.path.join(WORKDIR, 'final_model.joblib'))
PIPE = bundle['pipeline']
BLOCK = bundle['feature_block']
THR_YOUDEN = float(bundle.get('threshold_youden', 0.5))
THR_PROPOSAL = float(bundle.get('threshold_proposal', 0.70))


def build_features(sequences):
    # identical feature pipeline as training, but for unseen sequences
    mols = [pf_mol(s) for s in sequences]
    rd = pd.DataFrame([rdkit_features(s, m) for s, m in zip(sequences, mols)])
    rd = rd.reindex(columns=RDKIT_ALL_COLS)
    graph = np.vstack([pf_pool(*pf_tensors(m)) if m is not None else np.full(PF_DIM, np.nan)
                       for m in mols])
    rd = rd.astype(float)
    rd = rd[RDKIT_KEEP_COLS].values
    graph = graph[:, PF_KEEP_IDX]
    if BLOCK == 'RDKit':
        return rd
    if BLOCK == 'PepFoundry':
        return graph
    return np.hstack([rd, graph])


def predict_peptides(sequences, threshold=None):
    threshold = THR_YOUDEN if threshold is None else threshold
    seqs_ = [clean_seq(s)[0] for s in sequences]
    Xn = build_features(seqs_)
    p = score_for(PIPE, Xn)
    info = pd.DataFrame([rdkit_features(s, pf_mol(s)) for s in seqs_]).reindex(columns=RDKIT_ALL_COLS)
    out = pd.DataFrame({'sequence': seqs_, 'length': [len(s) for s in seqs_], 'p_active': p,
                        'prediction': np.where(p >= threshold, 'Active', 'Inactive'),
                        'passes_youden_%.2f' % THR_YOUDEN: p >= THR_YOUDEN,
                        'passes_proposal_0.70': p >= THR_PROPOSAL})
    # every physicochemical descriptor travels with the prediction (2.5.2 filters on these)
    for c in AMP_COLS:
        out[c] = info[c].values
    return out


def score_candidates(sequences, tag='02_5_candidate_scores'):
    # section 2.5 entry point: frozen model + full descriptor table + novelty flag
    tab = predict_peptides(sequences)
    known = set(df['sequence'])
    tab['in_training_set'] = [s in known for s in tab['sequence']]
    tab = tab.sort_values('p_active', ascending=False).reset_index(drop=True)
    path = os.path.join(WORKDIR, '%s.csv' % tag)
    tab.to_csv(path, index=False)
    print('%d candidates scored | %d pass the Youden gate | %d pass p >= %.2f | %d seen in training'
          % (len(tab), int(tab['passes_youden_%.2f' % THR_YOUDEN].sum()),
             int(tab['passes_proposal_0.70'].sum()), THR_PROPOSAL, int(tab['in_training_set'].sum())))
    return tab


demo_new = ['KKLIKVWAKGFKKAKKLFKGIG', 'GSGSGSGSLKKIFKKPMVIGVTIPF', 'FLSLIPHAINAVSAIAKHN']
try:
    out = predict_peptides(demo_new)
    print(out[['sequence', 'length', 'p_active', 'prediction',
               'passes_proposal_0.70', 'amp_net_charge_pH7_seq',
               'amp_hydrophobic_moment_kd']].round(4).to_string(index=False))
    truth = df.set_index('sequence')['label']
    for s in demo_new:
        s = clean_seq(s)[0]
        if s in truth.index:
            print('observed label for %s: %s' % (s, 'Active' if truth[s] == 1 else 'Inactive'))
except Exception as e:
    print('inference demo failed:', repr(e)[:300])

# proposal 2.5.3: the reference set a de novo candidate is checked against for novelty
df[['pep_id', 'sequence', 'label', 'cluster']].to_csv(
    os.path.join(WORKDIR, '02_5_novelty_reference.csv'), index=False)
print('saved 02_5_novelty_reference.csv (%d curated sequences for the 2.5.3 identity check)' % len(df))

print()
print('artefacts in', WORKDIR)
for f in sorted(os.listdir(WORKDIR)):
    print('  ', f, round(os.path.getsize(os.path.join(WORKDIR, f)) / 1024, 1), 'KB')


# ============================================================================
# CELL 41 (markdown, commented out)
# ============================================================================
# # 15. (Optional) graph neural network on the PepFoundry atom graphs
#
# Everything above uses *pooled* graph statistics, which throws away connectivity. The PepFoundry paper's strongest claim - atomic-level representations beat sequence-level ones for modified peptides - is usually demonstrated with a GNN on the atom graph, so this cell provides exactly that: a **GIN** (graph isomorphism network) with 24-d node features and 16-d edge features, mean-pooled to a peptide-level prediction.
#
# It is written in plain PyTorch (no `torch_geometric` dependency) because the graphs are small (66-272 atoms). The cell is **off by default**: set `RUN_GNN = True` after the main results are saved, keep the same cluster-grouped 80/20 split, and report at least 3 seeds - a 600-row dataset makes single-run numbers unreliable.


# ============================================================================
# CELL 42
# ============================================================================
# @title 15.1 GIN on the PepFoundry atom graphs
import torch
from torch import nn

RUN_GNN = False
GNN_HIDDEN = 64
GNN_LAYERS = 3
GNN_EPOCHS = 60
GNN_LR = 3e-3
GNN_DROPOUT = 0.3
GNN_SEEDS = [42, 7, 123]


def build_graphs(sequences):
    # -> list of (node_features [n,24], edge_index [2,E] undirected, edge_features [E,16])
    out = []
    for s in sequences:
        m = pf_mol(s)
        a, b = pf_tensors(m)
        ei = np.asarray(pf_engine().get_peptide_atomic_adjacency_matrix(m))
        n = a.shape[0]
        src, dst = int(ei[0].min()), int(ei[1].min())      # keep both directions of every bond
        row = np.concatenate([ei[0] - src, ei[1] - dst])
        col = np.concatenate([ei[1] - dst, ei[0] - src])
        eidx = np.stack([np.clip(row, 0, n - 1), np.clip(col, 0, n - 1)])
        eb = np.concatenate([b, b], axis=0)                # undirected: same bond features
        out.append((a.astype(np.float32), eidx.astype(np.int64), eb.astype(np.float32)))
    return out


def collate(graphs, labels=None, device='cpu'):
    xs, eis, ebs, batch = [], [], [], []
    off = 0
    for gi, (a, eidx, eb) in enumerate(graphs):
        xs.append(torch.tensor(a))
        eis.append(torch.tensor(eidx) + off)
        ebs.append(torch.tensor(eb))
        batch.append(torch.full((a.shape[0],), gi, dtype=torch.long))
        off += a.shape[0]
    x = torch.cat(xs).to(device)
    ei = torch.cat(eis, axis=1).to(device)
    eb = torch.cat(bs_ := ebs, dim=0).to(device)
    bt = torch.cat(batch).to(device)
    yb = torch.tensor(labels, dtype=torch.long).to(device) if labels is not None else None
    return x, ei, eb, bt, yb


class GINLayer(nn.Module):
    def __init__(self, din, dout, edge_dim, dropout):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.edge_lin = nn.Linear(edge_dim, din)          # bond features -> node space
        self.mlp = nn.Sequential(nn.Linear(din, dout), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(dout, dout))

    def forward(self, x, ei, eb):
        # h_i' = MLP((1 + eps) * h_i + sum_{j in N(i)} (h_j + e_ij))
        row, col = ei[0], ei[1]
        msg = x.index_select(0, col) + self.edge_lin(eb)
        agg = torch.zeros_like(x).index_add_(0, row, msg)
        return self.mlp((1.0 + self.eps) * x + agg)


class PeptideGIN(nn.Module):
    def __init__(self, node_dim=24, edge_dim=16, hidden=GNN_HIDDEN, layers=GNN_LAYERS,
                 dropout=GNN_DROPOUT):
        super().__init__()
        dims = [node_dim] + [hidden] * layers
        self.blocks = nn.ModuleList([GINLayer(dims[i], dims[i + 1], edge_dim, dropout)
                                     for i in range(layers)])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, x, ei, eb, bt, n_graphs):
        for blk in self.blocks:
            x = blk(x, ei, eb)
        # mean-pool the final node states, one vector per peptide
        sums = torch.zeros(n_graphs, x.shape[1], device=x.device, dtype=x.dtype)
        sums = sums.index_add(0, bt, x)
        counts = torch.zeros(n_graphs, device=x.device, dtype=x.dtype)
        counts = counts.index_add(0, bt, torch.ones_like(bt, dtype=x.dtype))
        return self.head(sums / counts.clamp(min=1).unsqueeze(1))


if RUN_GNN:
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    t0 = time.time()
    GRAPHS = build_graphs(seqs)
    print('built %d atom graphs in %.0fs (nodes %d..%d)'
          % (len(GRAPHS), time.time() - t0,
             min(g[0].shape[0] for g in GRAPHS), max(g[0].shape[0] for g in GRAPHS)))

    gnn_scores = []
    for gseed in GNN_SEEDS:
        torch.manual_seed(gseed)
        np.random.seed(gseed)
        tr, te = make_split(0.80, 0.20, gseed)
        net = PeptideGIN().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=GNN_LR, weight_decay=1e-2)
        lossf = nn.CrossEntropyLoss()
        best_mcc, best_pred, best_ep = -2, None, -1
        for ep in range(GNN_EPOCHS):
            net.train()
            order = np.random.permutation(tr)
            for s in range(0, len(order), 32):
                b = order[s:s + 32]
                x, ei, eb, bt, yb = collate([GRAPHS[i] for i in b], y[b], dev)
                logits = net(x, ei, eb, bt, len(b))
                loss = lossf(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            preds = []
            with torch.no_grad():
                for s in range(0, len(te), 32):
                    b = te[s:s + 32]
                    x, ei, eb, bt, _ = collate([GRAPHS[i] for i in b], None, dev)
                    logits = net(x, ei, eb, bt, len(b))
                    preds.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
            p = np.concatenate(preds)
            mm = compute_metrics(y[te], p)
            if mm['mcc'] > best_mcc:
                best_mcc, best_pred, best_ep = mm['mcc'], p, ep + 1
            if ep % 5 == 0 or ep == GNN_EPOCHS - 1:
                print('seed %d epoch %2d | loss %.4f | MCC %.3f | AUROC %.3f'
                      % (gseed, ep + 1, float(loss), mm['mcc'], mm['auroc']))
        mm = compute_metrics(y[te], best_pred)
        gnn_scores.append(mm)
        print('seed %d -> best MCC %.3f (epoch %d) | AUROC %.3f'
              % (gseed, best_mcc, best_ep, mm['auroc']))
        torch.save({'state_dict': net.state_dict(), 'epoch': best_ep},
                   os.path.join(WORKDIR, 'gnn_seed%d.pt' % gseed))

    gnn_tab = pd.DataFrame([{k: m[k] for k in METRIC_KEYS} for m in gnn_scores])
    gnn_tab.insert(0, 'model', 'GIN (PepFoundry atom graph)')
    print()
    print('=== GIN, 80/20 cluster-grouped, mean over %d seeds ===' % len(GNN_SEEDS))
    print(gnn_tab.agg({k: ['mean', 'std'] for k in METRIC_KEYS}).round(4).to_string())
    gnn_tab.to_csv(os.path.join(WORKDIR, 'results_gnn.csv'), index=False)
else:
    print('RUN_GNN = False -> cell skipped (results above come from the pooled-feature pipelines)')


# ============================================================================
# CELL 43 (markdown, commented out)
# ============================================================================
# # 16. Reporting checklist, mapped to the proposal
#
# * **2.3.1** describe the two streams as they are implemented: PepFoundry pooled graph statistics (24-d atom, 16-d bond, mean/std/max/sum) and the RDKit physicochemical block computed on the same molecule. State the pooling statistics and the resulting `D`.
# * **2.3.2** ship `02_dataset_with_physicochemical_descriptors.csv` and `descriptor_univariate_stats.csv`: the descriptors are assigned to the dataset, not floating beside it.
# * **2.4.1** quote the realised dimensions from `feature_manifest.json` (raw 160 + 78, kept 103 + 69, `D = 172`) rather than the proposal's placeholder 1280 + 20.
# * **2.4.2** report every split in the ladder, not one split, and say which training fraction saturates the metric.
# * **2.4.3** report the six named architectures with their tuned parameters; `headline_table.csv` and `appendix_table.csv` are ready to paste.
# * **2.4.4** report **MCC** as the primary metric with AUROC, accuracy, balanced accuracy, sensitivity and specificity. State the split type (random vs cluster-grouped) and that exact duplicates were collapsed. Show the **ablation** (RDKit / PepFoundry / Combined) - if RDKit is within noise of Combined that is a finding, not a failure, and it is the more defensible model. Report **mean +- std over seeds**, never a single run.
# * **2.5.2** quote `shap_physicochemical_patterns.csv` to justify which physicochemical properties are positively contributing (e.g. positive net charge, favourable hydrophobic moment) and use 13.2-13.3 to point at the residues that drive the prediction.
# * **2.5.3** note the split is at 90% identity while the proposal text says 80%: either state the deviation and why (a tighter threshold is the conservative choice for leakage control) or re-run with `CLUSTER_IDENTITY = 0.80`.
# * **2.5.4** report both thresholds from `screening_gate.json` - the Youden-optimal one and the proposal's nominal 0.70 - and use the same `score_candidates()` path for the generated pool, so the training and screening feature pipelines are provably the same code.
# * Cite the representation sources: PepFoundry (Garzon Otero *et al.*, *JCIM* 2026, `doi:10.1021/acs.jcim.5c02629`, AGPL-3.0) and RDKit (`doi:10.1002/jcc.30155`).
# * If the optional GIN beats the pooled models, say so explicitly - but report it as a multi-seed, clearly labelled extension, not as the headline number.
