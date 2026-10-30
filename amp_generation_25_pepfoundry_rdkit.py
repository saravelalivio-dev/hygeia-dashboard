"""Section 2.5 - Tree-SHAP-guided de novo peptide generation and in silico filters.

Toolchain substitution applied in this module
----------------------------------------------
    ESM-2 transformer embeddings  ->  PepFoundry pooled atom/bond graph features
    ProtParam physicochemical     ->  RDKit 2D descriptors on the PepFoundry molecule
    modlAMP GlobalDescriptor /    ->  RDKit SMARTS titration (net charge, theoretical pI)
    modlAMP PeptideDescriptor        helical-wheel projections (hydrophobic moment,
                                    amphipathicity) + RDKit composition indices

Nothing else in section 2.5 changes: 2.5.1 ProtGPT2 generation, 2.5.3 novelty / QC
and 2.5.4 re-evaluation with the frozen 2.4.4 master gate are the same operations.

Descriptor substitution table (proposal wording -> what this module computes)
----------------------------------------------------------------------------
    peptide length              length, rdkit_HeavyAtomCount
    net charge                  amp_net_charge_pH7          (RDKit SMARTS titration)
    hydrophobic moment          amp_hydrophobic_moment_kd   (helical wheel)
                               amp_hydrophobic_moment_eis  (Eisenberg consensus)
    GRAVY index                 amp_mean_hydropathy         (Kyte-Doolittle mean)
                               amp_hydropathy_per_atom     (Crippen MolLogP / heavy atom)
    aliphatic index             aliphatic_index             (classical index, RDKit composition)
    instability index           amp_flexibility             (rotatable bonds / heavy atom)
                               rdkit_BertzCT, rdkit_FractionCSP3, rdkit_qed
    theoretical isoelectric pt  theoretical_pi              (bisection on the titration)

The ProtParam instability index is a dipeptide-composition regression with no RDKit
counterpart, so it is deliberately NOT re-implemented here. Its role - penalising
peptides that are cheap to unfold - is filled by the RDKit conformational-flexibility
and complexity block, and the direction it acts in is taken from TreeSHAP (layer 2 of
the 2.5.2 filter) rather than from a hand-picked cutoff.

Two filter layers, both traceable to artefacts written by 2.3/2.4:
    layer 1  physicochemical gates named in 2.5.2, bounded by the 10th-90th
             percentile window of the *active* curated peptides (literature defaults
             if the dataset is unavailable)
    layer 2  per-descriptor pass/fail rules read out of shap_physicochemical_patterns.csv,
             applied to every remaining descriptor in the top decile of mean|SHAP|

Usage
-----
    python amp_generation_25_pepfoundry_rdkit.py

    Colab:
        !pip install -q rdkit openbabel-wheel torch transformers shap biopython joblib
        !pip install -q git+https://github.com/BilodeauGroup/PepFoundry.git
"""

import os
import json
import re
import argparse
import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, rdMolDescriptors

RDLogger.DisableLog('rdApp.*')

# ----------------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

PROTGPT2_ID = 'nferruz/ProtGPT2'
PF_CHIRALITIES = False          # must match the value used when the gate was trained
PF_POOL_STATS = ('mean', 'std', 'max', 'sum')
PF_DEVICE = 'cpu'

MIN_LEN, MAX_LEN = 10, 30      # proposal 2.2.2 canonical length window
N_TEMPLATES = 8                # top wild-type sequences used as ProtGPT2 seeds
SAMPLES_PER_TEMPLATE = 60
GEN_TEMPERATURE = 0.8
GEN_TOP_K = 950
GEN_REPETITION_PENALTY = 1.15

NOVELTY_IDENTITY = 0.80        # proposal 2.2.2 CD-HIT identity, reused as the 2.5.3 cutoff
NOVELTY_KMER = 3
PROB_CUTOFF_NOMINAL = 0.70     # proposal 2.5.4 nominal screening threshold
SHAP_RULE_QUANTILE = 0.90      # top decile of mean|SHAP| feeds the layer-2 rules
DESCRIPTOR_NAN_IS_PASS = True  # a descriptor that cannot be computed never rejects

VALID_AA = re.compile(r'^[ACDEFGHIKLMNPQRSTVWY]+$')
STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')

DRIVE_ROOT = '/content/drive/MyDrive'
ARTIFACT_DIRS = [
    os.path.join(DRIVE_ROOT, 'amp_klebsiella_pepfoundry'),
    os.path.join(DRIVE_ROOT, 'Klebsiella pneumoniae', 'amp_klebsiella_pepfoundry'),
    '/content/amp_klebsiella',
    '.',
]

# ----------------------------------------------------------------------------------
# hydropathy / charge scales
# ----------------------------------------------------------------------------------
KD = dict(zip('AVILPFLMYWSTGC', [1.8, 4.2, 4.5, 3.8, -1.6, 2.8, -0.9, 1.9,
                                 -0.4, -0.8, -0.7, -0.1, -1.3, -0.5, 1.9, -0.8]))
EI_raw = dict(zip('AVLIPFMDWKSTYRNQCEG',
                  [0.62, 0.46, 0.30, 0.37, 0.26, 0.35, 0.21, 0.10, 0.61, 0.42, 0.29,
                   0.27, 0.28, 0.14, 0.61, 0.25, 0.15, 0.10, 0.28, 0.29]))
_mu, _sd = float(np.mean(list(EI_raw.values()))), float(np.std(list(EI_raw.values())))
EI = {k: (v - _mu) / _sd for k, v in EI_raw.items()}
CHARGE_PH = {**{k: 0.0 for k in 'ACDEFGHILNPQSTVWY'},
             'K': 1.0, 'R': 1.0, 'H': 0.1, 'D': -1.0, 'E': -1.0}

# ionisable groups: name, (base|acid), pKa - the same table ProtParam uses, applied
# either to the groups RDKit finds in the PepFoundry molecule or to the sequence.
IONISABLE = [
    ('N_term', 'base', 8.60), ('Lys', 'base', 10.53), ('Arg', 'base', 12.48),
    ('His', 'base', 6.04), ('Cys', 'acid', 8.37), ('Tyr', 'acid', 10.46),
    ('Asp', 'acid', 3.90), ('Glu', 'acid', 4.07), ('C_term', 'acid', 3.55),
]
SEQ_GROUPS = {'K': ('Lys', 1), 'R': ('Arg', 1), 'H': ('His', 1), 'C': ('Cys', 1),
              'Y': ('Tyr', 1), 'D': ('Asp', 1), 'E': ('Glu', 1)}
# RDKit SMARTS for the same groups. Side-chain Asp/Glu are counted as
# "free carboxylic acids minus the C-terminal one", which is exact for a linear
# peptide and needs no atom-map bookkeeping.
SMARTS_GROUPS = [
    ('N_term', '[NX3;H2,H3;D1]', 1),
    ('Arg', '[NX3;H2;D3;$([NX3][CX3]=[NX2])]', 1),
    ('Lys', '[NX3;H2;H1;D3;!$([NX3][CX3]=[NX2,NX3,OX1,SX1])]', 1),
    ('His', 'c1cnc[nH]1', 1),
    ('Cys', '[SX2H;D1]', 1),
    ('Tyr', '[OX2H;$(Oc1ccccc1)]', 1),
]
CARBOXYL_SMARTS = '[CX3](=[OX1])[OX2;D1]'

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
    ('PEOE_VVA4_PLACEHOLDER', lambda m: np.nan),   # placeholder, replaced below
]
RDKIT_2D[-1] = ('PEOE_VSA4', lambda m: rdMolDescriptors.PEOE_VSA_(m, 4))

AMP_PHARMACOPHORE_COLS = [
    'amp_net_charge_pH7', 'amp_charge_density', 'amp_hydrophobic_moment_kd',
    'amp_hydrophobic_moment_eis', 'amp_mean_hydropathy', 'amp_hydropathy_range',
    'amp_hydropathy_per_atom', 'amp_amphipathicity', 'amp_amphipathicity_norm',
    'amp_boman_frac', 'amp_max_cationic_window', 'amp_flexibility',
    'amp_gross_charge_pH7', 'theoretical_pi', 'aliphatic_index',
]

# layer-1 gates named in proposal 2.5.2; (lo, hi) are filled from the active-class
# window when the curated dataset is present, otherwise from these defaults
DEFAULT_BOUNDS = {
    'amp_net_charge_pH7': (1.0, 12.0),
    'amp_hydrophobic_moment_kd': (0.20, None),
    'amp_mean_hydropathy': (-1.20, 2.20),
    'aliphatic_index': (40.0, None),
    'theoretical_pi': (7.0, 11.5),
    'amp_flexibility': (None, 0.35),
    'rdkit_MolWt': (None, 4200.0),
}
LAYER1_RULES = [
    ('length', 'ge', MIN_LEN, 'proposal 2.2.2 length window'),
    ('length', 'le', MAX_LEN, 'proposal 2.2.2 length window'),
    ('amp_net_charge_pH7', 'ge', 0.0, 'cationic attraction to the anionic envelope'),
    ('amp_hydrophobic_moment_kd', 'ge', 0.0, 'amphipathic insertion interface'),
    ('amp_mean_hydropathy', 'in', None, 'GRAVY band of active peptides'),
    ('aliphatic_index', 'ge', None, 'helical propensity'),
    ('theoretical_pi', 'in', None, 'away from the insolubility edges'),
    ('amp_flexibility', 'le', None, 'conformational stability'),
    ('rdkit_MolWt', 'le', None, 'synthetic cost ceiling'),
]


# ----------------------------------------------------------------------------------
# sequence utilities
# ----------------------------------------------------------------------------------
def clean_seq(s):
    if not isinstance(s, str):
        return ''
    return ''.join(ch for ch in s.strip().upper() if ch in '-* .' and False) or \
        ''.join(ch for ch in s.strip().upper() if ch in STANDARD_AA or ch in 'X')


def is_canonical(seq):
    return bool(VALID_AA.match(seq))


# ----------------------------------------------------------------------------------
# RDKit / PepFoundry featuriser - must stay bit-identical to the 2.3 / 2.4 pipeline
# ----------------------------------------------------------------------------------
_PF = None


def pf_engine():
    global _PF
    if _PF is None:
        from pepfoundry.interface import PepFoundry
        _PF = PepFoundry()
    return _PF


def pf_mol(seq):
    try:
        mol = pf_engine().get_peptide(seq, plot_peptide=False)
    except Exception:
        mol = None
    if mol is None:
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
    if A is None or len(A) == 0:
        return np.zeros(0)
    return np.concatenate([_POOL_FN[s](A) for s in stats])


def pf_pool(atom_t, bond_t, stats=PF_POOL_STATS):
    return np.concatenate([pool_stats(atom_t, stats), pool_stats(bond_t, stats)])


def pf_stat_names(stats=PF_POOL_STATS, d_atom=24, d_bond=16):
    return (['atom_%s_%02d' % (s, i) for s in stats for i in range(d_atom)] +
            ['bond_%s_%02d' % (s, i) for s in stats for i in range(d_bond)])


def graph_vector(mol, names=pf_stat_names()):
    if mol is None:
        return np.full(len(names), np.nan)
    a, b = pf_tensors(mol)
    return pf_pool(a, b)


# ----------------------------------------------------------------------------------
# charge / pI by RDKit SMARTS titration
# ----------------------------------------------------------------------------------
def _smarts(pattern):
    patt = Chem.MolFromSmarts(pattern)
    return None if patt is None else patt


def group_counts_from_mol(mol):
    """Count ionisable groups on the RDKit molecule; None if anything is unusable."""
    if mol is None:
        return None
    try:
        counts = {}
        for name, pattern in SMARTS_GROUPS:
            patt = _smarts(pattern)
            if patt is None:
                return None
            counts[name] = len(mol.GetSubstructMatches(patt))
        patt = _smarts(CARBOXYL_SMARTS)
        if patt is None:
            return None
        n_acid = len(mol.GetSubstructMatches(patt))
        counts['Asp'] = counts['Glu'] = max(0, n_acid - 1)
        counts['C_term'] = 1 if n_acid else 0
        return counts
    except Exception:
        return None


def group_counts_from_seq(seq):
    counts = {name: 0 for name, _, _ in IONISABLE}
    counts['N_term'] = 1 if seq else 0
    counts['C_term'] = 1 if seq else 0
    for ch in seq:
        if ch in SEQ_GROUPS:
            counts[SEQ_GROUPS[ch][0]] += 1
    return counts


def net_charge_at_pH(counts, pH):
    q = 0.0
    for name, kind, pka in IONISABLE:
        n = counts.get(name, 0)
        if not n:
            continue
        q += n / (1.0 + 10.0 ** (pH - pka)) if kind == 'base' else -n / (1.0 + 10.0 ** (pka - pH))
    return q


def theoretical_pi(counts):
    """Bisection on the titration: the pH at which the net charge crosses zero."""
    lo, hi = 0.0, 14.0
    if net_charge_at_pH(counts, lo) <= 0:
        return float('nan')
    if net_charge_at_pH(counts, hi) >= 0:
        return float('nan')
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if net_charge_at_pH(counts, mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------------
# AMP pharmacophore + classical indices (the ProtParam / modlAMP replacements)
# ----------------------------------------------------------------------------------
def helicoidal_moment(seq, scale, window=11):
    v = np.array([scale.get(a, 0.0) for a in seq], dtype=float)
    n = len(v)
    if n == 0:
        return 0.0
    w = min(window, n)
    ang = 2 * np.pi * np.arange(w) / w
    best = 0.0
    for s in range(n - w + 1):
        seg = v[s:s + w]
        best = max(best, float(np.hypot((seg * np.cos(ang)).sum(),
                                        (seg * np.sin(ang)).sum()) / w))
    return best


def aliphatic_index(seq):
    """Classical index: 100*(2.9*Xa + 3.9*Xb/(Xa+Xb-1) - 1); undefined when Xa+Xb <= 1."""
    n = max(1, len(seq))
    xa = sum(seq.count(a) for a in 'ACEGILV') / n
    xb = sum(seq.count(a) for a in 'ILV') / n
    denom = xa + xb - 1.0
    if denom <= 1e-6:
        return float('nan')
    return 100.0 * (2.9 * xa + 3.9 * xb / denom - 1.0)


def pharmacophore(seq, mol, f2d):
    n = max(1, len(seq))
    counts = group_counts_from_mol(mol)
    source = 'rdkit_smarts'
    if counts is None:
        counts, source = group_counts_from_seq(seq), 'sequence_hh'

    v = np.array([KD.get(a, 0.0) for a in seq], dtype=float)
    heavy = max(1.0, f2d['rdkit_HeavyAtomCount'])

    out = {
        'amp_net_charge_pH7': float(net_charge_at_pH(counts, 7.0)),
        'amp_gross_charge_pH7': float(sum(CHARGE_PH.get(a, 0.0) for a in seq)),
        'amp_charge_density': float(net_charge_at_pH(counts, 7.0)) / n,
        'amp_hydrophobic_moment_kd': helicoidal_moment(seq, KD, 11),
        'amp_hydrophobic_moment_eis': helicoidal_moment(seq, EI, 11),
        'amp_mean_hydropathy': float(v.mean()) if n else 0.0,
        'amp_hydropathy_range': float(v.max() - v.min()) if n else 0.0,
        'amp_hydropathy_per_atom': float(f2d['rdkit_MolLogP']) / heavy,
        'amp_amphipathicity': helicoidal_moment(seq, KD, 11) / (abs(v.mean()) + 0.1) if n else 0.0,
        'amp_amphipathicity_norm': helicoidal_moment(seq, KD, 11) / (0.1 + float(np.abs(v).mean()))
        if n else 0.0,
        'amp_boman_frac': float(np.mean([1.0 if a in 'AILVFWMY' else 0.0 for a in seq])),
        'amp_max_cationic_window': float(max(
            [sum(1.0 for a in seq[i:i + 7] if a in 'KR') / max(1, len(seq[i:i + 7]))
             for i in range(max(1, n - 6))] or [0.0])),
        'amp_flexibility': float(f2d['rdkit_NumRotatableBonds']) / heavy,
        'theoretical_pi': theoretical_pi(counts),
        'aliphatic_index': aliphatic_index(seq),
        'amp_charge_source': source,
    }
    return out


def rdkit_features(seq, mol):
    f = {}
    for name, fn in RDKIT_2D:
        try:
            f['rdkit_' + name] = float(fn(mol)) if mol is not None else np.nan
        except Exception:
            f['rdkit_' + name] = np.nan
    for aa in 'ACDEFGHIKLMNPQRSTVWY':
        f['aac_' + aa] = float(seq.count(aa)) / max(1, len(seq))
    f.update(pharmacophore(seq, mol, f))
    f['length'] = float(len(seq))
    return f


def featurise(sequences, want_graph=True):
    """-> (descriptor DataFrame, molecules list, pooled graph matrix or None)"""
    mols = [pf_mol(s) if want_graph else Chem.MolFromSequence(s) for s in sequences]
    desc = pd.DataFrame([rdkit_features(s, m) for s, m in zip(sequences, mols)])
    graph = np.vstack([graph_vector(m) for m in mols]) if want_graph else None
    return desc, mols, graph


# ----------------------------------------------------------------------------------
# artefact loading (gate, thresholds, reference set, SHAP pattern table)
# ----------------------------------------------------------------------------------
def find_workdir(explicit=None):
    if explicit:
        return explicit
    for d in ARTIFACT_DIRS:
        if os.path.exists(os.path.join(d, 'final_model.joblib')):
            return d
    return ARTIFACT_DIRS[-1]


class Context(object):
    """Everything 2.5 needs from the 2.3 / 2.4 run."""

    def __init__(self, workdir):
        self.workdir = workdir
        self.bundle = None
        self.pipe = None
        self.block = 'Combined'
        self.thr_youden = PROB_CUTOFF_NOMINAL
        self.rdkit_cols = None
        self.pf_names = pf_stat_names()
        self.pf_keep_idx = None
        self.rdkit_keep_cols = None
        self.dataset = None
        self.patterns = None
        self.reference = []

    def path(self, name):
        return os.path.join(self.workdir, name)

    def load(self):
        import joblib
        try:
            self.bundle = joblib.load(self.path('final_model.joblib'))
            self.pipe = self.bundle['pipeline']
            self.block = self.bundle.get('feature_block', 'Combined')
            self.thr_youden = float(self.bundle.get('threshold_youden', PROB_CUTOFF_NOMINAL))
            self.rdkit_cols = list(self.bundle['rdkit_cols'])
            self.pf_names = list(self.bundle.get('pf_stat_names', self.pf_names))
            self.pf_pool_stats = tuple(self.bundle.get('pf_pool_stats', PF_POOL_STATS))
            print('loaded master gate: block=%s, Youden threshold=%.4f' % (self.block, self.thr_youden))
        except Exception as e:
            raise RuntimeError('final_model.joblib not found in %s - run 2.3/2.4 first (%s)'
                               % (self.workdir, repr(e)[:120]))

        self._recover_columns()

        try:
            self.dataset = pd.read_csv(self.path('02_dataset_with_physicochemical_descriptors.csv'))
        except Exception:
            for alt in ('02_dataset_final.csv',):
                try:
                    self.dataset = pd.read_csv(self.path(alt))
                    break
                except Exception:
                    pass
        try:
            self.patterns = pd.read_csv(self.path('shap_physicochemical_patterns.csv'))
        except Exception:
            self.patterns = None
        try:
            ref = pd.read_csv(self.path('02_5_novelty_reference.csv'))
            self.reference = ref['sequence'].astype(str).tolist()
        except Exception:
            self.reference = (self.dataset['sequence'].astype(str).tolist()
                              if self.dataset is not None else [])
        print('curated dataset: %s | novelty reference pool: %d sequences | SHAP table: %s'
              % ('%d rows' % len(self.dataset) if self.dataset is not None else 'absent',
                 len(self.reference), 'loaded' if self.patterns is not None else 'absent'))
        return self

    def _recover_columns(self):
        """Reproduce the constant-column drop of 2.4.1 so the vector widths match."""
        n_expected = int(getattr(self.pipe, 'n_features_in_', 0) or 0)
        keep_rd, keep_pf = None, None
        try:
            with open(self.path('feature_manifest.json'), encoding='utf-8') as f:
                man = json.load(f)
            keep_rd = list(man['physicochemical']['kept'])
        except Exception:
            pass
        if self.dataset is not None and self.rdkit_cols:
            try:
                cols = [c for c in self.rdkit_cols if c in self.dataset.columns]
                raw = self.dataset[cols].values.astype(float)
                keep_rd = keep_rd or [c for c in cols if np.nanstd(raw[:, cols.index(c)]) > 0]
            except Exception:
                pass
        if keep_rd is None:
            keep_rd = list(self.rdkit_cols)
        self.rdkit_keep_cols = keep_rd

        try:
            pf_raw = np.load(self.path('X_PepFoundry.npy'))
            keep_pf = np.where(np.nanstd(pf_raw, axis=0) > 0)[0].tolist()
        except Exception:
            keep_pf = list(range(len(self.pf_names)))
        self.pf_keep_idx = keep_pf

        width = {'RDKit': len(keep_rd), 'PepFoundry': len(keep_pf)}.get(
            self.block, len(keep_rd) + len(keep_pf))
        print('feature widths: RDKit=%d PepFoundry=%d -> block %s = %d columns'
              % (len(keep_rd), len(keep_pf), self.block, width))
        if n_expected and width != n_expected:
            print('WARNING: recovered width %d != gate n_features_in_ %d; check the '
                  'artefact directory' % (width, n_expected))

    # -- the gate itself ---------------------------------------------------------
    def build_matrix(self, sequences):
        desc, mols, graph = featurise(sequences, want_graph=self.block != 'RDKit')
        rd = desc.reindex(columns=self.rdkit_cols)[self.rdkit_keep_cols].values.astype(float)
        if self.block == 'RDKit':
            X = rd
        elif self.block == 'PepFoundry':
            X = graph[:, self.pf_keep_idx]
        else:
            X = np.hstack([rd, graph[:, self.pf_keep_idx]])
        return X, desc, mols

    def design_matrix(self, X):
        return self.pipe[:-1].transform(X)

    def feature_names(self, X):
        if self.block == 'RDKit':
            return list(self.rdkit_keep_cols)
        if self.block == 'PepFoundry':
            return [self.pf_names[i] for i in self.pf_keep_idx]
        return list(self.rdkit_keep_cols) + [self.pf_names[i] for i in self.pf_keep_idx]

    def scores(self, X):
        if hasattr(self.pipe, 'predict_proba'):
            return self.pipe.predict_proba(X)[:, 1]
        return self.pipe.decision_function(X)


# ----------------------------------------------------------------------------------
# TreeSHAP attribution
# ----------------------------------------------------------------------------------
TREE_TYPES = ('RandomForestClassifier', 'ExtraTreesClassifier', 'GradientBoostingClassifier',
              'LGBMClassifier', 'XGBClassifier', 'CatBoostClassifier', 'DecisionTreeClassifier')


def shap_matrix(explainer, X, name=''):
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = np.asarray(sv[-1])
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    elif sv.ndim == 1:
        sv = sv.reshape(X.shape[0], -1)
    return sv


def gate_shap(ctx, X):
    """TreeSHAP on the master gate. Returns (values, column names) or (None, None)."""
    try:
        import shap
    except Exception:
        return None, None
    clf = ctx.pipe['clf'] if 'clf' in getattr(ctx.pipe, 'named_steps', {}) else ctx.pipe.steps[-1][1]
    if type(clf).__name__ not in TREE_TYPES:
        return None, None
    Xt = ctx.design_matrix(X)
    names = ctx.feature_names(X)
    if names is None or len(names) != Xt.shape[1]:
        names = ['component_%03d' % i for i in range(Xt.shape[1])]
    sv = shap_matrix(shap.TreeExplainer(clf), Xt)
    return sv, names


def shap_patterns(ctx, desc, y):
    """Per-descriptor attribution pattern, i.e. the 2.5.2 rule source.

    Uses shap_physicochemical_patterns.csv when 2.4.4 wrote it, otherwise refits the
    same RandomForest on the RDKit block of the curated set and recomputes the table.
    """
    if ctx.patterns is not None:
        return ctx.patterns.copy()

    if ctx.dataset is None or y is None:
        print('no SHAP pattern table and no curated labels -> falling back to layer-1 bounds only')
        return None

    from sklearn.ensemble import RandomForestClassifier
    cols = [c for c in ctx.rdkit_cols if c in ctx.dataset.columns]
    X = ctx.dataset[cols].values.astype(float)
    keep = [c for c in cols if np.nanstd(X[:, cols.index(c)]) > 0]
    X = ctx.dataset[keep].values.astype(float)
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight='balanced',
                                n_jobs=-1, random_state=SEED)
    rf.fit(X, y)
    import shap
    sv = shap_matrix(shap.TreeExplainer(rf), X)
    tab = pd.DataFrame({
        'descriptor': keep,
        'mean_shap_active': [sv[y == 1, j].mean() for j in range(sv.shape[1])],
        'mean_shap_inactive': [sv[y == 0, j].mean() for j in range(sv.shape[1])],
        'mean_abs_shap': np.abs(sv).mean(axis=0),
    })
    tab['shap_gap'] = tab['mean_shap_active'] - tab['mean_shap_inactive']
    tab['favours'] = np.where(tab['shap_gap'] > 0, 'Active', 'Inactive')
    tab['active_mean'] = [X[y == 1, j].mean() for j in range(X.shape[1])]
    tab['inactive_mean'] = [X[y == 0, j].mean() for j in range(X.shape[1])]
    tab['direction'] = np.where(tab['active_mean'] > tab['inactive_mean'],
                                'higher favours Active', 'lower favours Active')
    tab['group'] = [c.split('_')[0] for c in keep]
    return tab.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)


def build_rules(ctx, patterns, y_active):
    """Layer-1 (physicochemical gates) + layer-2 (TreeSHAP rules) as one table."""
    rows = []

    # layer 1 -------------------------------------------------------------------
    bounds = dict(DEFAULT_BOUNDS)
    if y_active is not None and len(y_active):
        qs = {}
        for c in DEFAULT_BOUNDS:
            if c in y_active.columns:
                v = pd.to_numeric(y_active[c], errors='coerce').dropna()
                if len(v) >= 10:
                    qs[c] = (float(v.quantile(0.10)), float(v.quantile(0.90)),
                             float(v.median()))
        for c, (lo_q, hi_q, med) in qs.items():
            d_lo, d_hi = DEFAULT_BOUNDS.get(c, (None, None))
            bounds[c] = (lo_q if d_lo is not None else d_lo,
                         hi_q if d_hi is not None else d_hi)
            if c in ('amp_hydrophobic_moment_kd', 'aliphatic_index'):
                bounds[c] = (med, d_hi)
        print('layer-1 bounds read from %d active curated peptides' % len(y_active))

    for name, op, override, why in LAYER1_RULES:
        b_lo, b_hi = bounds.get(name, (None, None))
        lo, hi = (override if override is not None else b_lo), b_hi
        rows.append(dict(layer=1, descriptor=name, rule='%s' % op, lo=lo, hi=hi,
                         shap_gap=np.nan, mean_abs_shap=np.nan, rationale=why))

    # layer 2 -------------------------------------------------------------------
    if patterns is not None and len(patterns):
        covered = {r['descriptor'] for r in rows}
        pats = patterns[patterns['descriptor'].isin(set(ctx.rdkit_cols))].copy()
        if len(pats):
            cut = float(pats['mean_abs_shap'].quantile(SHAP_RULE_QUANTILE))
            pats = pats[pats['mean_abs_shap'] >= cut]
        for _, r in pats.iterrows():
            if r['descriptor'] in covered:
                continue
            if r['favours'] == 'Active':
                op, lo, hi = ('ge' if r['direction'].startswith('higher') else 'le',
                              r['active_mean'] if r['direction'].startswith('higher') else None,
                              r['active_mean'] if r['direction'].startswith('lower') else None)
            else:
                op, lo, hi = ('le' if r['direction'].startswith('higher') else 'ge',
                              r['active_mean'] if r['direction'].startswith('higher') else None,
                              r['active_mean'] if r['direction'].startswith('lower') else None)
            rows.append(dict(layer=2, descriptor=r['descriptor'], rule=op, lo=lo, hi=hi,
                             shap_gap=r['shap_gap'], mean_abs_shap=r['mean_abs_shap'],
                             rationale='%s (TreeSHAP gap %.3f)' % (r['direction'], r['shap_gap'])))
    rules = pd.DataFrame(rows)
    print('filter rules: %d physicochemical gates + %d TreeSHAP rules'
          % ((rules['layer'] == 1).sum(), (rules['layer'] == 2).sum()))
    return rules


def apply_rules(desc_row, rules):
    """-> (passed, failed_rule) for one candidate."""
    for _, r in rules.iterrows():
        d = r['descriptor']
        v = desc_row.get(d, np.nan)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            if not DESCRIPTOR_NAN_IS_PASS:
                return False, '%s (undefined)' % d
            continue
        v = float(v)
        op = r['rule']
        if op == 'ge' and r['lo'] is not None and np.isfinite(r['lo']) and v < float(r['lo']):
            return False, '%s=%.4f < %.4f' % (d, v, float(r['lo']))
        if op == 'le' and r['hi'] is not None and np.isfinite(r['hi']) and v > float(r['hi']):
            return False, '%s=%.4f > %.4f' % (d, v, float(r['hi']))
    return True, ''


# ----------------------------------------------------------------------------------
# 2.5.1 ProtGPT2 generation
# ----------------------------------------------------------------------------------
def pick_templates(ctx, n=N_TEMPLATES):
    if ctx.dataset is None or 'sequence' not in ctx.dataset.columns:
        return None
    d = ctx.dataset.copy()
    d = d[d['sequence'].astype(str).str.match(VALID_AA.pattern)]
    if 'label' in d.columns:
        d = d[d['label'].astype(int) == 1]
    if 'cluster' in d.columns and len(d):
        d = d.drop_duplicates('cluster')
    d = d.sort_values('length').head(n)
    return d['sequence'].astype(str).tolist()


def generate_candidates(templates, n_templates=N_TEMPLATES, per_template=SAMPLES_PER_TEMPLATE,
                        free_run=0, seed=SEED, model_id=PROTGPT2_ID):
    """ProtGPT2 de novo generation, optionally conditioned on wild-type templates.

    Templates are fed as whitespace-separated residues, which is the input format
    ProtGPT2 was trained on; every decoded character is then filtered to the 20
    canonical residues so a stray token cannot corrupt a candidate.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print('[2.5.1] loading %s ...' % model_id)
    tok = AutoTokenizer.from_pretrained(model_id)
    lm = AutoModelForCausalLM.from_pretrained(model_id)
    lm.eval()
    torch.manual_seed(seed)

    bos = tok('<|endoftext|>', add_special_tokens=False).input_ids
    out = []

    def draw(prompt_ids, k, tag):
        if k <= 0:
            return
        ids = torch.tensor([prompt_ids])
        with torch.no_grad():
            gen = lm.generate(ids, max_new_tokens=MAX_LEN + 10, do_sample=True, top_k=GEN_TOP_K,
                              temperature=GEN_TEMPERATURE, repetition_penalty=GEN_REPETITION_PENALTY,
                              num_return_sequences=k)
        for g in gen:
            text = tok.decode(g, skip_special_tokens=True).upper()
            seq = ''.join(ch for ch in text if ch in STANDARD_AA)
            if tag and seq.startswith(tag):
                seq = seq[len(tag):]
            out.append(seq)
        print('   %-28s -> %d decoded' % (tag or 'free running', len(gen)))

    for i, t in enumerate(templates or []):
        draw(bos + tok(' '.join(list(t)), add_special_tokens=False).input_ids,
             per_template, t)
    if free_run:
        draw(bos, free_run, '')

    print('[2.5.1] ProtGPT2 returned %d raw sequences' % len(out))
    return out


# ----------------------------------------------------------------------------------
# 2.5.3 novelty - CD-HIT style identity against the known peptide pool
# ----------------------------------------------------------------------------------
def _aligner():
    from Bio import Align
    a = Align.PairwiseAligner()
    a.mode = 'local'
    a.open_gap_score = -11
    a.extend_gap_score = -1
    try:
        a.substitution_matrix = Align.substitution_matrices.load('BLOSUM62')
    except Exception:
        pass
    return a


def _kmer(s, k=NOVELTY_KMER):
    return {s[i:i + k] for i in range(max(1, len(s) - k + 1))} if len(s) >= k else {s}


class NoveltyIndex(object):
    """k-mer prefiltered local-alignment identity search (same rule as 2.2.2)."""

    def __init__(self, references, identity=NOVELTY_IDENTITY):
        self.identity = identity
        self.refs = list(dict.fromkeys(references))
        self.kmers = [_kmer(r) for r in self.refs]
        self.aligner = _aligner()
        self.prefilter = max(0.3, identity - 0.35)

    def max_identity(self, seq):
        best, best_ref = 0.0, ''
        kk = _kmer(seq)
        for ref, rk in zip(self.refs, self.kmers):
            if abs(len(seq) - len(ref)) / max(len(seq), len(ref)) > (1 - self.identity):
                continue
            if len(kk & rk) / max(1, min(len(kk), len(rk))) < self.prefilter:
                continue
            ident = self._identity(seq, ref)
            if ident > best:
                best, best_ref = ident, ref
                if best >= 1.0:
                    break
        return best, best_ref

    def _identity(self, a, b):
        try:
            aln = self.aligner.align(a, b)[0]
            ba, bb = aln.aligned
            matched = 0
            for (a0, a1), (b0, b1) in zip(ba, bb):
                matched += sum(1 for x, y in zip(a[a0:a1], b[b0:b1]) if x == y)
            return matched / max(1, min(len(a), len(b)))
        except Exception:
            from difflib import SequenceMatcher
            return SequenceMatcher(None, a, b).ratio()


# ----------------------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------------------
def run(workdir=None, n_templates=N_TEMPLATES, per_template=SAMPLES_PER_TEMPLATE,
        free_run=0, skip_generation=False, use_rules=None, no_shap=False):
    ctx = Context(find_workdir(workdir)).load()
    os.makedirs(ctx.workdir, exist_ok=True)

    y_active = None
    if ctx.dataset is not None and 'label' in ctx.dataset.columns:
        y_active = ctx.dataset[ctx.dataset['label'].astype(int) == 1]
    y = ctx.dataset['label'].astype(int).values if y_active is not None else None

    patterns = None if no_shap else shap_patterns(ctx, None, y)
    if patterns is not None:
        patterns.to_csv(ctx.path('02_5_shap_pattern_table.csv'), index=False)
    rules = build_rules(ctx, patterns, y_active)
    if use_rules:
        rules = rules[rules['descriptor'].isin(use_rules) | (rules['layer'] == 1)]
    rules.to_csv(ctx.path('02_5_filter_rules.csv'), index=False)
    print(rules.to_string(index=False))

    templates = pick_templates(ctx, n_templates)
    print('ProtGPT2 seed templates (%d): %s' % (len(templates or []), templates))

    if skip_generation:
        raw = pd.read_csv(ctx.path('02_5_1_generated_raw.csv'))['sequence'].astype(str).tolist()
        print('[2.5.1] reusing %d cached sequences' % len(raw))
    else:
        raw = generate_candidates(templates, n_templates, per_template, free_run)
    raw = [clean_seq(s) for s in raw]
    pd.DataFrame({'sequence': raw, 'length': [len(s) for s in raw],
                  'is_canonical': [is_canonical(s) for s in raw]}).to_csv(
        ctx.path('02_5_1_generated_raw.csv'), index=False)

    # 2.5.3 part 1: canonical residues and length, before any expensive featurisation
    qc = [(i, s) for i, s in enumerate(raw) if is_canonical(s) and MIN_LEN <= len(s) <= MAX_LEN]
    print('[2.5.3] %d/%d survive the canonical-residue and %d-%d length check'
          % (len(qc), len(raw), MIN_LEN, MAX_LEN))
    if not qc:
        print('nothing to featurise - raise per_template or free_run')
        return
    seqs = list(dict.fromkeys(s for _, s in qc))

    X, desc, mols = ctx.build_matrix(seqs)
    print('featurised %d unique candidates -> X %s (block %s)'
          % (len(seqs), X.shape, ctx.block))

    # 2.5.2 physicochemical filtering, RDKit descriptors + TreeSHAP rules
    audit = [apply_rules(desc.iloc[i], rules) for i in range(len(seqs))]
    keep_idx = [i for i, (ok, _) in enumerate(audit) if ok]
    fail_counts = {}
    for _, why in audit:
        if why:
            fail_counts[why.split('=')[0].split(' ')[0]] = fail_counts.get(
                why.split('=')[0].split(' ')[0], 0) + 1
    print('[2.5.2] %d/%d candidates pass the physicochemical + TreeSHAP filters'
          % (len(keep_idx), len(seqs)))
    if fail_counts:
        print('   most common rejections: %s'
              % sorted(fail_counts.items(), key=lambda kv: -kv[1])[:6])
    pd.DataFrame([{'sequence': seqs[i], 'passes_2_5_2': ok, 'failed_rule': why}
                  for i, (ok, why) in enumerate(audit)]).to_csv(
        ctx.path('02_5_2_filter_audit.csv'), index=False)
    if not keep_idx:
        print('no candidate survived 2.5.2 - relax the layer-1 bounds or generate more')
        return
    filtered = [seqs[i] for i in keep_idx]
    desc_f = desc.iloc[keep_idx].reset_index(drop=True)
    X_f = X[keep_idx]

    # 2.5.3 part 2: novelty against the known peptide pool
    index = NoveltyIndex(ctx.reference) if ctx.reference else None
    if index is None:
        novel = list(range(len(filtered)))
        print('[2.5.3] no reference pool available - novelty check skipped')
    else:
        novel, ident = [], []
        for i, s in enumerate(filtered):
            m, ref = index.max_identity(s)
            ident.append(m)
            if m <= NOVELTY_IDENTITY:
                novel.append(i)
        print('[2.5.3] %d/%d candidates are <= %.0f%% identical to a known peptide '
              '(max identity seen %.1f%%)'
              % (len(novel), len(filtered), 100 * NOVELTY_IDENTITY,
                 100 * max(ident) if ident else 0.0))
    desc_f['max_identity_to_known'] = [novelty_identities.get(i, np.nan)] if False else None
    if index is not None:
        vals = []
        for s in filtered:
            m, ref = index.max_identity(s)
            vals.append((m, ref))
        desc_f['max_identity_to_known'] = [v[0] for v in vals]
        desc_f['nearest_known_peptide'] = [v[1] for v in vals]
    else:
        desc_f['max_identity_to_known'] = np.nan
        desc_f['nearest_known_peptide'] = ''
    desc_f.to_csv(ctx.path('02_5_3_novelty.csv'), index=False)
    if not novel:
        print('every filtered candidate is too close to a reported peptide - stopping')
        return
    filtered = [filtered[i] for i in novel]
    desc_f = desc_f.iloc[novel].reset_index(drop=True)
    X_f = X_f[novel]

    # 2.5.4 secondary feature extraction + re-evaluation with the frozen gate
    p = ctx.scores(X_f)
    scores = desc_f.copy()
    scores.insert(0, 'sequence', filtered)
    scores['p_active'] = p
    scores['threshold_used'] = ctx.thr_youden
    scores['passes_youden'] = p >= ctx.thr_youden
    scores['passes_proposal_%.2f' % PROB_CUTOFF_NOMINAL] = p >= PROB_CUTOFF_NOMINAL
    if not ctx.reference:
        scores['seen_in_curated_set'] = [s in set(ctx.reference) for s in filtered]

    sv, names = (None, None) if no_shap else gate_shap(ctx, X_f)
    if sv is not None:
        scores['mean_abs_shap'] = np.abs(sv).mean(axis=1)
        scores['shap_base_value'] = float(np.ravel(
            gate_shap_value(ctx, X_f)).mean()) if False else np.nan
    else:
        print('[2.5.4] master gate is not tree based -> SHAP agreement not scored')

    scores = scores.sort_values('p_active', ascending=False).reset_index(drop=True)
    scores.to_csv(ctx.path('02_5_4_candidate_scores.csv'), index=False)

    thr = max(ctx.thr_youden, PROB_CUTOFF_NOMINAL)
    shortlist = scores[scores['p_active'] >= PROB_CUTOFF_NOMINAL].copy()
    shortlist.to_csv(ctx.path('02_5_shortlist.csv'), index=False)

    print()
    print('=== 2.5.4 re-evaluation of %d candidates (frozen %s gate) ===' % (len(scores), ctx.block))
    if len(scores):
        print(scores[['sequence', 'p_active', 'amp_net_charge_pH7',
                      'amp_hydrophobic_moment_kd', 'amp_mean_hydropathy',
                      'theoretical_pi', 'aliphatic_index']].head(15).round(3).to_string(index=False))
    print()
    print('Youden threshold %.3f -> %d pass | proposal nominal %.2f -> %d pass'
          % (ctx.thr_youden, int(scores['passes_youden'].sum()),
             PROB_CUTOFF_NOMINAL, int(scores['passes_proposal_%.2f' % PROB_CUTOFF_NOMINAL].sum())))
    print('SHORTLIST for 2.6 / 2.7 / 2.8: %d candidates at p >= %.2f' % (len(shortlist), thr))
    if len(shortlist):
        print(shortlist[['sequence', 'p_active', 'amp_net_charge_pH7',
                         'amp_hydrophobic_moment_kd', 'theoretical_pi']].to_string(index=False))
    print()
    print('artefacts written to', ctx.workdir)
    for f in sorted(os.listdir(ctx.workdir)):
        if f.startswith('02_5'):
            print('  ', f)
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--workdir', default=None, help='directory holding the 2.3/2.4 artefacts')
    ap.add_argument('--templates', type=int, default=N_TEMPLATES)
    ap.add_argument('--per-template', type=int, default=SAMPLES_PER_TEMPLATE)
    ap.add_argument('--free-run', type=int, default=0, help='extra unconditional samples')
    ap.add_argument('--skip-generation', action='store_true',
                    help='reuse 02_5_1_generated_raw.csv instead of calling ProtGPT2')
    ap.add_argument('--rules', default=None,
                    help='comma-separated physicochemical descriptors to gate on')
    ap.add_argument('--no-shap', action='store_true')
    args = ap.parse_args()
    run(workdir=args.workdir, n_templates=args.templates, per_template=args.per_template,
        free_run=args.free_run, skip_generation=args.skip_generation,
        use_rules=[c.strip() for c in args.rules.split(',')] if args.rules else None,
        no_shap=args.no_shap)


if __name__ == '__main__':
    main()
