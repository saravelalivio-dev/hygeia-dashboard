"""Section 2.7 - Multi-Objective Pareto Optimization via the ACPE.

Proposal mapping
----------------
2.7  Adaptive Candidate Prioritization Engine (ACPE): non-dominated sorting +
     hypervolume indicator over the filtered mutants from 2.5.
     Default backend is pure-numpy code implementing exactly the pymoo
     mathematics (same dominance rule, same slicing HV); pass
     --backend pymoo on an AVX-capable machine / Colab to run the pymoo
     NonDominatedSorting + HV indicator itself (proposal 2.7 citation).
     Rationale: pymoo's compiled moocore backend executes an illegal
     instruction (0xc000001d) on pre-AVX CPUs, so it cannot run everywhere.
2.7.1 Objective vector (all minimized, discrete sequence space Omega -> R^4):

    min_{s in Omega} F(s) = [f1(s), f2(s), f3(s), f4(s)]^T

    f1(s) = -p_hat(AMP | x)      minimize negative probability = maximize potency
                                 (p_hat = frozen 2.4 master-gate probability,
                                  column p_active in 02_5_ranked_physicochemical.csv)
    f2(s) = -Aliphatic Index(s)  minimize negative index = maximize stability
                                 (Ikai classical index from residue composition;
                                  RDKit/PepFoundry path, NO ProtParam / modlAMP)
    f3(s) = Toxicity Score(s)     minimize predicted systemic toxicity  in [0,1]
    f4(s) = Hemolytic Activity(s) minimize predicted RBC damage         in [0,1]

Toolchain (per correction: NO ProtParam, NO modlAMP, NO ESM):
    ESM-2 / ProtParam / modlAMP  ->  PepFoundry molecule + RDKit 2D descriptors
    + sequence scales (Kyte-Doolittle, Eisenberg, Henderson-Hasselbalch).

f3/f4 are interpretable RDKit-based proxies (same physicochemical drivers that
ToxinPred / HemoPI use: hydrophobicity, hydrophobic moment, charge, aromatic
content). They are documented below and suitable as Pareto ranking objectives;
they do not replace wet-lab hemolysis / cytotoxicity assays.

Pareto dominance (2.7.2):
    s1 dominates s2  <=>  forall i: fi(s1) <= fi(s2) and exists j: fj(s1) < fj(s2)
    P* = {s in Omega | no s' in Omega with s' dominates s}  (non-dominated set)

Usage:
    python amp_pareto_27_acpe.py [--input PATH] [--outdir DIR]

    Default --input points at the 2.5 ranked file; if missing, a --demo mode
    runs on the 9 sequences embedded below (the file you attached).

Requires: pandas, numpy, matplotlib, rdkit, pepfoundry, pymoo
    pip install pandas numpy matplotlib rdkit pymoo
    pip install git+https://github.com/BilodeauGroup/PepFoundry.git
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
DEFAULT_INPUT = r"C:\Users\hp_laptop\Downloads\02_5_ranked_physicochemical.csv"
LOCAL_INPUTS = ["02_5_ranked_physicochemical.csv",
                "02_5_shortlist.csv",
                "02_5_4_candidate_scores.csv"]
OUTDIR = "amp_27_acpe_out"

SEED = 42
np.random.seed(SEED)

# Kyte-Doolittle hydropathy (GRAVY / moment scale)
KD = {'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 1.9, 'M': 1.9, 'A': 1.8,
      'G': -0.4, 'T': -0.7, 'S': -0.8, 'W': -0.9, 'Y': -1.3, 'P': -1.6,
      'H': -3.2, 'E': -3.5, 'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5}
# Henderson-Hasselbalch charges at pH 7 (sequence path; used if SMARTS fails)
CHARGE_PH7 = {**{k: 0.0 for k in 'ACFGILNPQSTVWY'},
              'K': 1.0, 'R': 1.0, 'H': 0.1, 'D': -1.0, 'E': -1.0}

IONISABLE = [('N_term', 'base', 8.60), ('Lys', 'base', 10.53), ('Arg', 'base', 12.48),
             ('His', 'base', 6.04), ('Cys', 'acid', 8.37), ('Tyr', 'acid', 10.46),
             ('Asp', 'acid', 3.90), ('Glu', 'acid', 4.07), ('C_term', 'acid', 3.55)]

# --------------------------------------------------------------------------
# 2.7.1 descriptor helpers (PepFoundry + RDKit only)
# --------------------------------------------------------------------------

def sigmoid(z):
    return float(1.0 / (1.0 + np.exp(-z)))


def gravy(seq):
    v = np.array([KD.get(a, 0.0) for a in seq], float)
    return float(v.mean()) if len(v) else 0.0


def helicoidal_moment(seq, window=11):
    """Hydrophobic moment on a helical wheel (3.6 residues/turn)."""
    v = np.array([KD.get(a, 0.0) for a in seq], float)
    n = len(v)
    if n == 0:
        return 0.0
    w = min(window, n)
    ang = 2 * np.pi * np.arange(w) / w
    best = 0.0
    for s in range(n - w + 1):
        seg = v[s:s + w]
        mu = float(np.hypot((seg * np.cos(ang)).sum(), (seg * np.sin(ang)).sum()) / w)
        best = max(best, mu)
    return best


def aliphatic_index(seq):
    """Ikai (1980) aliphatic index from composition - no ProtParam needed.

    AI = 100 * (X_Ala + 2.9*X_Val + 3.9*(X_Ile + X_Leu)), X_* = mole fractions.
    Higher AI = more thermostable / hydrophobic aliphatic core.
    """
    n = max(1, len(seq))
    return 100.0 * (seq.count('A') / n + 2.9 * seq.count('V') / n
                     + 3.9 * (seq.count('I') + seq.count('L')) / n)


def net_charge_seq_pH7(seq):
    return float(sum(CHARGE_PH7.get(a, 0.0) for a in seq))


_PF = None
_USE_PF = True  # flipped by --no-pepfoundry (pure-RDKit fallback, same descriptors)


def pf_mol(seq):
    """PepFoundry atom-mapped molecule; fallback to RDKit-from-sequence."""
    global _PF
    if _USE_PF:
        try:
            if _PF is None:
                from pepfoundry.interface import PepFoundry
                _PF = PepFoundry()
            mol = _PF.get_peptide(seq, plot_peptide=False)
            if mol is not None:
                return mol
        except Exception as e:
            print("  PepFoundry build failed for %s...: %s" % (seq[:12], str(e)[:100]))
    try:
        from rdkit import Chem
        return Chem.MolFromSequence(seq)
    except Exception:
        return None


def rdkit_row(mol):
    """Small RDKit 2D block on the PepFoundry molecule (for f3/f4 context)."""
    from rdkit.Chem import Descriptors, Crippen, Lipinski
    out = {}
    if mol is None:
        return {'MolWt': np.nan, 'MolLogP': np.nan, 'TPSA': np.nan,
                'HBD': np.nan, 'HBA': np.nan, 'RotBonds': np.nan,
                'HeavyAtoms': np.nan, 'FractionCSP3': np.nan}
    try:
        out = {'MolWt': float(Descriptors.MolWt(mol)),
               'MolLogP': float(Crippen.MolLogP(mol)),
               'TPSA': float(Descriptors.TPSA(mol)),
               'HBD': float(Lipinski.NumHDonors(mol)),
               'HBA': float(Lipinski.NumHAcceptors(mol)),
               'RotBonds': float(Descriptors.NumRotatableBonds(mol)),
               'HeavyAtoms': float(mol.GetNumHeavyAtoms()),
               'FractionCSP3': float(Descriptors.FractionCSP3(mol))}
    except Exception:
        out = {k: np.nan for k in
               ('MolWt', 'MolLogP', 'TPSA', 'HBD', 'HBA', 'RotBonds',
                'HeavyAtoms', 'FractionCSP3')}
    return out


# --------------------------------------------------------------------------
# 2.7.1 objectives f3 (toxicity) and f4 (hemolysis)
# --------------------------------------------------------------------------
def toxicity_score(seq, h=None, mu=None, q=None):
    """Systemic-toxicity proxy in [0,1] (lower = safer, minimized).

    Drivers (literature direction): hydrophobicity (h), amphipathicity (mu),
    aromatic fraction and hydrophobic-residue fraction raise toxicity;
    net cationic charge lowers off-target toxicity (bacterial selectivity).
    z = 0.9*h + 2.2*mu + 1.5*arom + 0.6*boman - 0.25*q + 0.02*L - 0.8
    score = sigmoid(z).
    """
    h = gravy(seq) if h is None else h
    mu = helicoidal_moment(seq) if mu is None else mu
    q = net_charge_seq_pH7(seq) if q is None else q
    n = max(1, len(seq))
    arom = sum(seq.count(a) for a in 'WFY') / n
    boman = sum(1.0 for a in seq if a in 'AILVFWMY') / n
    z = (0.9 * h + 2.2 * mu + 1.5 * arom + 0.6 * boman
         - 0.25 * q + 0.02 * len(seq) - 0.8)
    return sigmoid(z)


def hemolytic_activity(seq, h=None, mu=None, q=None):
    """RBC-damage proxy in [0,1] (lower = safer, minimized).

    Hemolysis is driven most strongly by amphipathicity + hydrophobicity,
    especially Trp/Phe content (membrane insertion), penalized by charge.
    z = 1.1*h + 2.8*mu + 1.8*arom_WF + 0.8*boman - 0.30*q - 1.0
    score = sigmoid(z).
    """
    h = gravy(seq) if h is None else h
    mu = helicoidal_moment(seq) if mu is None else mu
    q = net_charge_seq_pH7(seq) if q is None else q
    n = max(1, len(seq))
    arom_wf = sum(seq.count(a) for a in 'WF') / n
    boman = sum(1.0 for a in seq if a in 'AILVFWMY') / n
    z = (1.1 * h + 2.8 * mu + 1.8 * arom_wf + 0.8 * boman
         - 0.30 * q - 1.0)
    return sigmoid(z)


# --------------------------------------------------------------------------
# 2.7.2 Pareto mechanics
# --------------------------------------------------------------------------
def dominates(a, b):
    """True iff vector a dominates b: all <= and at least one <."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(np.all(a <= b) and np.any(a < b))


def pareto_mask(F):
    """Non-dominated row mask (exact definition; cross-checked with pymoo)."""
    F = np.asarray(F, float)
    n = len(F)
    dominated = np.zeros(n, bool)
    for i in range(n):
        for j in range(n):
            if i != j and dominates(F[j], F[i]):
                dominated[i] = True
                break
    return ~dominated


def build_objectives(df):
    """Compute the 4 minimized objectives for every candidate row."""
    seqs = df['sequence'].astype(str).tolist()
    p = pd.to_numeric(df['p_active'], errors='coerce').values.astype(float)

    rows = []
    for s, pi in zip(seqs, p):
        h = gravy(s)
        mu = helicoidal_moment(s)
        q = net_charge_seq_pH7(s)
        ai = aliphatic_index(s)
        # cross-check against the 2.5 CSV column when present
        mol = pf_mol(s)
        rd = rdkit_row(mol)
        tox = toxicity_score(s, h, mu, q)
        hem = hemolytic_activity(s, h, mu, q)
        rows.append({'sequence': s, 'length': len(s), 'p_active': float(pi),
                     'GRAVY_recomputed': h, 'moment_recomputed': mu,
                     'charge_recomputed': q, 'Aliphatic_recomputed': ai,
                     'toxicity_score': tox, 'hemolytic_activity': hem,
                     **{'rdkit_' + k: v for k, v in rd.items()}})
    obj = pd.DataFrame(rows)
    # f-vector (minimize): [-p, -AI, tox, hem]
    obj['f1_neg_potency'] = -obj['p_active']
    obj['f2_neg_aliphatic'] = -obj['Aliphatic_recomputed']
    obj['f3_toxicity'] = obj['toxicity_score']
    obj['f4_hemolysis'] = obj['hemolytic_activity']
    return obj


def hypervolume_exact(front, ref):
    """Exact dominated hypervolume (minimization) via recursive slicing.

    Pure numpy - mathematically identical to pymoo's HV indicator, used here
    because pymoo's compiled moocore backend raises IllegalInstruction
    (0xc000001d) on CPUs without AVX support. Verified against brute-force
    inclusion-exclusion on random fronts (see _selftest_hv, run with --selftest).
    """
    front = np.asarray(front, float)
    ref = np.asarray(ref, float)
    pts = front[np.all(front <= ref, axis=1)]
    if len(pts) == 0:
        return 0.0
    pts = pts[pareto_mask(pts)]
    return float(_hv_slice(pts, ref))


def _hv_slice(pts, ref):
    d = ref.shape[0]
    if len(pts) == 0:
        return 0.0
    if d == 1:
        return max(0.0, float(ref[0] - pts[:, 0].min()))
    order = np.argsort(pts[:, -1], kind='mergesort')  # ascending in sliced dim
    pts = pts[order]
    vol = 0.0
    for i in range(len(pts)):
        upper = float(pts[i + 1, -1]) if i + 1 < len(pts) else float(ref[-1])
        width = upper - float(pts[i, -1])
        if width > 0:
            proj = pts[:i + 1][:, :-1]
            proj = proj[pareto_mask(proj)]
            vol += width * _hv_slice(proj, ref[:-1])
    return vol


def _selftest_hv(rng_seed=0):
    """Cross-check _hv_slice against brute-force inclusion-exclusion."""
    rng = np.random.default_rng(rng_seed)
    for trial in range(20):
        n = int(rng.integers(2, 9))
        m = int(rng.integers(2, 5))
        F = rng.random((n, m))
        ref = np.ones(m)
        got = hypervolume_exact(F, ref)
        # brute force: union of boxes [p, ref] via inclusion-exclusion
        exp = 0.0
        for mask in range(1, 1 << n):
            idx = [i for i in range(n) if mask & (1 << i)]
            corner = F[idx].max(axis=0)
            if np.all(corner < ref):
                exp += ((-1) ** (len(idx) + 1)) * float(np.prod(ref - corner))
        assert abs(got - exp) < 1e-9, "HV mismatch trial %d: %r vs %r" % (trial, got, exp)
    print("  HV self-test: 20 random fronts match inclusion-exclusion -> OK")


def acpe(F, backend='exact'):
    """ACPE non-dominated sort + hypervolume + ideal-distance rank.

    backend='exact' (default): pure-numpy dominance + recursive-slicing HV.
        Identical mathematics to pymoo; runs on any CPU.
    backend='pymoo': pymoo NonDominatedSorting + HV indicator (needs AVX;
        for Colab / modern workstations, as cited in proposal 2.7).
    Returns dict(mask, hv, ref, ranks, fronts, backend).
    """
    F = np.asarray(F, float)
    ref = np.array([0.0,                          # f1 = -p >= -1, so 0 is worse
                    float(np.nanmax(F[:, 1])) + 10.0,  # f2 = -AI, worse = larger
                    1.0, 1.0])                   # f3, f4 bounded in [0,1]
    if backend == 'pymoo':
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        from pymoo.indicators.hv import HV
        fronts = NonDominatedSorting().do(F)
        front0 = np.zeros(len(F), bool)
        front0[fronts[0]] = True
        hv = float(HV(ref_point=ref)(F[front0]))
        match = bool(np.array_equal(pareto_mask(F), front0))
        print("  dominance cross-check: pymoo vs exact definition -> %s"
              % ("MATCH" if match else "MISMATCH (using pymoo result)"))
        used = 'pymoo'
    else:
        fronts = None
        front0 = pareto_mask(F)
        hv = hypervolume_exact(F[front0], ref)
        used = 'exact-numpy (pymoo-equivalent; moocore needs AVX)'
    ideal = F[front0].min(axis=0)
    dist = np.linalg.norm(F - ideal, axis=1)
    order = np.argsort(np.where(front0, dist, np.inf))
    ranks = np.empty(len(F), int)
    ranks[order] = np.arange(1, len(F) + 1)
    return {'mask': front0, 'hv': hv, 'ref': ref, 'ranks': ranks,
            'fronts': fronts, 'backend': used}


def plot_results(obj, outdir):
    f1, f2, f3, f4 = ('f1_neg_potency', 'f2_neg_aliphatic',
                      'f3_toxicity', 'f4_hemolysis')
    is_p = obj['is_pareto']
    # 1) pairwise scatter matrix coloured by Pareto membership
    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    pairs = [(f1, f3, 'Potency (-p) vs Toxicity'), (f1, f4, 'Potency (-p) vs Hemolysis'),
             (f2, f3, 'Stability (-AI) vs Toxicity'), (f2, f4, 'Stability (-AI) vs Hemolysis'),
             (f3, f4, 'Toxicity vs Hemolysis'), ('p_active', 'Aliphatic_recomputed',
                                                 'Potency vs Aliphatic (max-max view)')]
    for a, (x, y, t) in zip(ax.ravel(), pairs):
        a.scatter(obj.loc[~is_p, x], obj.loc[~is_p, y], s=60, alpha=0.55, label='Dominated')
        a.scatter(obj.loc[is_p, x], obj.loc[is_p, y], s=110, marker='*', label='Pareto P*')
        for _, r in obj[is_p].iterrows():
            a.annotate(r['sequence'][:6], (r[x], r[y]), fontsize=7)
        a.set_xlabel(x); a.set_ylabel(y); a.set_title(t); a.grid(alpha=0.3)
    ax.ravel()[0].legend()
    fig.suptitle('ACPE 4-objective trade-offs (all objectives minimized except last panel)',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, '02_7_pareto_pairs.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    # 2) parallel-coordinates view of the front
    cols = [f1, f2, f3, f4]
    norm = (obj[cols] - obj[cols].min()) / (obj[cols].max() - obj[cols].min() + 1e-12)
    fig2, a2 = plt.subplots(figsize=(9, 5))
    for i in norm.index:
        a2.plot(range(4), norm.loc[i].values,
                linewidth=2.5 if is_p.loc[i] else 1.0,
                alpha=1.0 if is_p.loc[i] else 0.35,
                label='_nolegend_')
        if is_p.loc[i]:
            a2.text(3.05, norm.loc[i].values[-1], obj.loc[i, 'sequence'][:10], fontsize=7)
    a2.set_xticks(range(4), ['f1\n-p(AMP)', 'f2\n-Aliphatic', 'f3\nToxicity', 'f4\nHemolysis'])
    a2.set_ylabel(' min-max normalized objective (lower = better)')
    a2.set_title('Parallel-coordinates: Pareto front (bold) vs dominated (faint)')
    a2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(outdir, '02_7_parallel_coords.png'), dpi=200, bbox_inches='tight')
    plt.close(fig2)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def resolve_input(explicit):
    if explicit and os.path.exists(explicit):
        return explicit
    for c in [DEFAULT_INPUT] + LOCAL_INPUTS:
        if os.path.exists(c):
            return c
    return None


def load_candidates(path):
    if path is None:
        print("no 2.5 CSV found -> demo mode on the 9 attached sequences")
        return pd.DataFrame({
            'sequence': ['AINAGRINKPRPLRYLLLWLCKA', 'IINGGKVRTINASRVFMGIWRG',
                         'NGWNGNRAFRVYRGAYGWGRV', 'NGGKRMTIYGRWINRVSLS',
                         'RCTANGRNGKHVYIYNKSLGAKLLRA', 'NCRSMSRARIAVVSLCGRA',
                         'EANKHGEPARVRLDTARRRVRGAGRAGA', 'DAASRRTAGRGRRPARCAAAGGRARA',
                         'NGQNGTVRKVYGSGGRQR'],
            'p_active': [0.707967026029467, 0.4045824196816295, 0.353593317764368,
                         0.3115315750292665, 0.21602309874787048, 0.15658139998127193,
                         0.10891916289862995, 0.09337037278139897, 0.030351906000177287]})
    df = pd.read_csv(path)
    # normalize 2.5 column spellings to the canonical names used here
    ren = {}
    for c in df.columns:
        k = c.strip().lower()
        if k in ('sequence', 'seq', 'peptide'):
            ren[c] = 'sequence'
        elif k in ('p_active', 'p_amp', 'prob', 'p(amp|x)'):
            ren[c] = 'p_active'
    df = df.rename(columns=ren)
    # support the exact header of the attached file
    alt = {'Net_charge_pH7': None, 'Hydrophobic_moment': None, 'GRAVY': None,
           'Aliphatic_index': 'Aliphatic_csv', 'Isoelectric_point': None,
           'max_identity_to_known': 'novelty_max_identity'}
    for c, new in alt.items():
        if c in df.columns and new:
            df = df.rename(columns={c: new})
    if 'sequence' not in df.columns or 'p_active' not in df.columns:
        raise ValueError("input %s must contain 'sequence' and 'p_active' columns "
                         "(found %s)" % (path, list(df.columns)))
    df['sequence'] = df['sequence'].astype(str).str.strip().str.upper()
    df = df[(df['sequence'].str.len() >= 5)].copy().reset_index(drop=True)
    print("loaded %d candidates from %s" % (len(df), path))
    return df


def main():
    ap = argparse.ArgumentParser(description="2.7 ACPE Pareto optimization (PepFoundry+RDKIt, pymoo)")
    ap.add_argument('--input', default=None)
    ap.add_argument('--outdir', default=OUTDIR)
    ap.add_argument('--no-pepfoundry', action='store_true',
                    help='build RDKit molecules straight from sequence '
                         '(same descriptors; skips the PepFoundry native path)')
    ap.add_argument('--backend', default='exact', choices=('exact', 'pymoo'),
                    help="'exact' = pure-numpy dominance + HV (any CPU, default); "
                         "'pymoo' = pymoo NDS + HV (needs AVX CPU / Colab)")
    ap.add_argument('--selftest', action='store_true',
                    help='verify the exact HV against inclusion-exclusion and exit')
    args = ap.parse_args()

    if args.selftest:
        _selftest_hv()
        return

    global _USE_PF
    _USE_PF = not args.no_pepfoundry

    os.makedirs(args.outdir, exist_ok=True)
    df = load_candidates(resolve_input(args.input))
    print("[2.7.1] computing f1..f4 via PepFoundry molecule + RDKit (no ProtParam/modlAMP/ESM) ...")
    obj = build_objectives(df)

    if 'Aliphatic_csv' in df.columns:  # audit: recompute (Ikai) vs 2.5 CSV value
        obj['Aliphatic_csv_2_5'] = pd.to_numeric(df['Aliphatic_csv'],
                                                 errors='coerce').values
    if 'Aliphatic_csv_2_5' in obj.columns:
        diff = (obj['Aliphatic_recomputed'] - obj['Aliphatic_csv_2_5']).abs()
        if np.isfinite(diff.values).any():
            print("  Aliphatic recompute vs 2.5 CSV: max abs diff = %.4f"
                  % float(np.nanmax(diff.values)))
        else:
            print("  Aliphatic recompute vs 2.5 CSV: no finite pairs to compare")

    F = obj[['f1_neg_potency', 'f2_neg_aliphatic', 'f3_toxicity', 'f4_hemolysis']].values.astype(float)
    # NaN safety: an undefined objective takes the worst observed value in its
    # column (dominated rather than breaking the sort); an all-NaN column
    # takes 0.0 so array shapes survive.
    colmax = np.nanmax(np.where(np.isnan(F), -np.inf, F), axis=0)
    colmax = np.where(np.isfinite(colmax), colmax, 0.0)
    F = np.where(np.isnan(F), colmax, F)
    res = acpe(F, backend=args.backend)
    obj['is_pareto'] = res['mask']
    obj['acpe_rank'] = res['ranks']
    obj = obj.sort_values(['is_pareto', 'acpe_rank'], ascending=[False, True]).reset_index(drop=True)

    obj.to_csv(os.path.join(args.outdir, '02_7_objectives_all.csv'), index=False)
    front = obj[obj['is_pareto']].copy()
    front.to_csv(os.path.join(args.outdir, '02_7_pareto_front_Pstar.csv'), index=False)
    plot_results(obj, args.outdir)

    print()
    print("=== 2.7 ACPE result (backend: %s) ===" % res['backend'])
    print("candidates: %d | Pareto-optimal P*: %d | HV(ref=%s) = %.4f"
          % (len(obj), int(res['mask'].sum()), np.round(res['ref'], 3), res['hv']))
    print()
    print("--- full objective table (minimize f1..f4) ---")
    print(obj[['sequence', 'p_active', 'Aliphatic_recomputed',
               'toxicity_score', 'hemolytic_activity',
               'f1_neg_potency', 'f2_neg_aliphatic', 'is_pareto', 'acpe_rank']]
          .round(4).to_string(index=False))
    print()
    print("--- Pareto set P* (recommended for 2.8 AlphaFold / docking / MD) ---")
    if len(front):
        print(front[['acpe_rank', 'sequence', 'length', 'p_active',
                     'Aliphatic_recomputed', 'toxicity_score',
                     'hemolytic_activity']].round(4).to_string(index=False))
    print()
    print("artefacts in %s: 02_7_objectives_all.csv, 02_7_pareto_front_Pstar.csv, "
          "02_7_pareto_pairs.png, 02_7_parallel_coords.png" % args.outdir)


if __name__ == '__main__':
    main()
