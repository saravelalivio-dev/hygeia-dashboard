# -*- coding: utf-8 -*-
"""Builds Klebsiella_AMP_27_ACPE_Pareto_Colab.ipynb from readable cell sources."""
import json

CELLS = []


def md(src):
    CELLS.append(('markdown', src.strip('\n') + '\n'))


def code(src):
    CELLS.append(('code', src.strip('\n') + '\n'))


# ----------------------------------------------------------------- title
md(r'''
# 2.7 Multi-Objective Pareto Optimization via the ACPE
## *Klebsiella pneumoniae* de novo AMPs — potency × stability × toxicity × hemolysis

**Input** `02_5_ranked_physicochemical.csv` (section 2.5 output: `sequence`, `p_active`, …).
**Output** the Pareto-optimal set **P\*** + ranked objective table + plots, ready for 2.8 (AlphaFold / docking / MD).

| # | Objective (all **minimized**) | Meaning |
|---|---|---|
| f1(s) | −p̂(AMP \| x) | maximize antimicrobial potency (frozen 2.4 gate probability) |
| f2(s) | −Aliphatic Index(s) | maximize physicochemical stability (Ikai 1980, composition-based) |
| f3(s) | Toxicity Score(s) ∈ [0,1] | minimize predicted systemic toxicity |
| f4(s) | Hemolytic Activity(s) ∈ [0,1] | minimize predicted RBC damage |

**Dominance (2.7.2):** s₁ ≺ s₂ ⟺ ∀i fᵢ(s₁) ≤ fᵢ(s₂) ∧ ∃j fⱼ(s₁) < fⱼ(s₂).
**P\*** = {s ∈ Ω | no s′ ∈ Ω dominates s}. Frontier quality = dominated **hypervolume** vs an anti-optimal reference point.

> **Toolchain:** PepFoundry molecule + RDKit 2D descriptors only — NO ProtParam, NO modlAMP, NO ESM.
> f3/f4 are interpretable RDKit-based ranking proxies (same drivers ToxinPred/HemoPI use), not assay replacements.
>
> **Backend:** `BACKEND = 'exact'` (default) is pure-numpy code implementing *identical* mathematics to pymoo
> (dominance rule + recursive-slicing hypervolume, proven against brute-force inclusion-exclusion in cell 7).
> Set `BACKEND = 'pymoo'` to run pymoo's `NonDominatedSorting` + `HV` itself — Colab VMs have AVX so it works there
> (it hard-crashes pre-AVX laptops, which is why exact is the default).
''')

code(r'''
# @title 0. Install packages
import sys, subprocess, importlib


def _pip(*pkgs):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *pkgs], check=False)


_pip('pandas', 'numpy', 'matplotlib')
_pip('rdkit')                                                          # 2D descriptors
_pip('openbabel-wheel')                                                # PepFoundry needs it
_pip('git+https://github.com/BilodeauGroup/PepFoundry.git')            # not on PyPI
_pip('pymoo')                                                          # only used if BACKEND='pymoo'

for m in ('numpy', 'pandas', 'matplotlib'):
    mod = importlib.import_module(m)
    print(f'{m:12s} OK   {getattr(mod, "__version__", "?")}')
try:
    import rdkit
    print('rdkit        OK  ', rdkit.__version__)
except Exception as e:
    print('rdkit        FAIL:', e)
try:
    from pepfoundry.interface import PepFoundry
    print('pepfoundry   OK   (first build of the residue library takes ~1 min)')
except Exception as e:
    print('pepfoundry   FAIL:', repr(e)[:200])
try:
    import pymoo
    print('pymoo        OK  ', pymoo.__version__)
except Exception as e:
    print('pymoo        FAIL (only needed for BACKEND=pymoo):', repr(e)[:120])
''')

md(r'''
# 1. Input — the 2.5 ranked candidates
Mount Drive (optional) or upload `02_5_ranked_physicochemical.csv` when prompted.
Without any file the notebook runs a demo on the 9 reference sequences.
''')

code(r'''
# @title 1.1 Locate the 2.5 ranked file
import os
BACKEND = 'exact'   # 'exact' (any CPU) or 'pymoo' (needs AVX; Colab is fine)
WORKDIR = '/content/amp_27_acpe'
os.makedirs(WORKDIR, exist_ok=True)

CANDIDATES = ['/content/drive/MyDrive/amp_klebsiella_pepfoundry/02_5_shortlist.csv',
              '/content/drive/MyDrive/02_5_ranked_physicochemical.csv',
              '/content/02_5_ranked_physicochemical.csv',
              '02_5_ranked_physicochemical.csv']

INPUT_CSV = next((p for p in CANDIDATES if os.path.exists(p)), None)
if INPUT_CSV is None:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        INPUT_CSV = next((p for p in CANDIDATES if os.path.exists(p)), None)
    except Exception as e:
        print('drive mount skipped:', repr(e)[:100])

if INPUT_CSV is None:
    from google.colab import files
    print('upload 02_5_ranked_physicochemical.csv (or Cancel for the 9-sequence demo)')
    up = files.upload()
    for name in up:
        INPUT_CSV = '/content/' + name
        break

print('INPUT:', INPUT_CSV if INPUT_CSV else '(demo mode: 9 reference sequences)')
print('BACKEND:', BACKEND, '| WORKDIR:', WORKDIR)
''')

md(r'''
# 2. Proposal 2.7.1 — descriptors and the four objectives
Ikai aliphatic index, Kyte–Doolittle GRAVY / helical-wheel moment, Henderson–Hasselbalch charge,
plus RDKit context on the PepFoundry molecule. f3/f4 are sigmoid proxies over those drivers.
''')

code(r'''
# @title 2.1 Descriptor + objective functions (PepFoundry + RDKit only)
import numpy as np
import pandas as pd

KD = {'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 1.9, 'M': 1.9, 'A': 1.8,
      'G': -0.4, 'T': -0.7, 'S': -0.8, 'W': -0.9, 'Y': -1.3, 'P': -1.6,
      'H': -3.2, 'E': -3.5, 'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5}
CHARGE_PH7 = {**{k: 0.0 for k in 'ACFGILNPQSTVWY'},
              'K': 1.0, 'R': 1.0, 'H': 0.1, 'D': -1.0, 'E': -1.0}


def sigmoid(z):
    return float(1.0 / (1.0 + np.exp(-z)))


def gravy(seq):
    v = np.array([KD.get(a, 0.0) for a in seq], float)
    return float(v.mean()) if len(v) else 0.0


def helicoidal_moment(seq, window=11):
    v = np.array([KD.get(a, 0.0) for a in seq], float)
    if len(v) == 0:
        return 0.0
    w = min(window, len(v))
    ang = 2 * np.pi * np.arange(w) / w
    return max(float(np.hypot((v[s:s + w] * np.cos(ang)).sum(),
                              (v[s:s + w] * np.sin(ang)).sum()) / w)
               for s in range(len(v) - w + 1))


def aliphatic_index(seq):
    """Ikai (1980): AI = 100*(X_Ala + 2.9*X_Val + 3.9*(X_Ile + X_Leu))."""
    n = max(1, len(seq))
    return 100.0 * (seq.count('A') / n + 2.9 * seq.count('V') / n
                    + 3.9 * (seq.count('I') + seq.count('L')) / n)


def net_charge_seq_pH7(seq):
    return float(sum(CHARGE_PH7.get(a, 0.0) for a in seq))


_PF = None


def pf_mol(seq):
    global _PF
    try:
        if _PF is None:
            from pepfoundry.interface import PepFoundry
            _PF = PepFoundry()
        mol = _PF.get_peptide(seq, plot_peptide=False)
        if mol is not None:
            return mol
    except Exception as e:
        print('  PepFoundry fallback for %s...: %s' % (seq[:12], str(e)[:100]))
    from rdkit import Chem
    return Chem.MolFromSequence(seq)


def toxicity_score(seq, h=None, mu=None, q=None):
    """f3 in [0,1], lower = safer. sigmoid(0.9*h + 2.2*mu + 1.5*arom + 0.6*boman
    - 0.25*q + 0.02*L - 0.8). Hydrophobicity/amphipathicity/aromatics drive
    off-target insertion; cationic charge gives bacterial selectivity."""
    h = gravy(seq) if h is None else h
    mu = helicoidal_moment(seq) if mu is None else mu
    q = net_charge_seq_pH7(seq) if q is None else q
    n = max(1, len(seq))
    arom = sum(seq.count(a) for a in 'WFY') / n
    boman = sum(1.0 for a in seq if a in 'AILVFWMY') / n
    return sigmoid(0.9 * h + 2.2 * mu + 1.5 * arom + 0.6 * boman
                   - 0.25 * q + 0.02 * len(seq) - 0.8)


def hemolytic_activity(seq, h=None, mu=None, q=None):
    """f4 in [0,1], lower = safer. sigmoid(1.1*h + 2.8*mu + 1.8*arom_WF
    + 0.8*boman - 0.30*q - 1.0). Trp/Phe + moment dominate RBC insertion."""
    h = gravy(seq) if h is None else h
    mu = helicoidal_moment(seq) if mu is None else mu
    q = net_charge_seq_pH7(seq) if q is None else q
    n = max(1, len(seq))
    arom_wf = sum(seq.count(a) for a in 'WF') / n
    boman = sum(1.0 for a in seq if a in 'AILVFWMY') / n
    return sigmoid(1.1 * h + 2.8 * mu + 1.8 * arom_wf + 0.8 * boman
                   - 0.30 * q - 1.0)


print('descriptor functions ready (no ProtParam / modlAMP / ESM anywhere)')
''')

md(r'''
# 3. Proposal 2.7.2 — Pareto dominance + hypervolume
Exact non-dominated mask plus a recursive-slicing hypervolume proven against brute-force inclusion-exclusion.
''')

code(r'''
# @title 3.1 Dominance, Pareto mask, exact hypervolume
def dominates(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(np.all(a <= b) and np.any(a < b))


def pareto_mask(F):
    F = np.asarray(F, float)
    dom = np.zeros(len(F), bool)
    for i in range(len(F)):
        for j in range(len(F)):
            if i != j and dominates(F[j], F[i]):
                dom[i] = True
                break
    return ~dom


def _hv_slice(pts, ref):
    if len(pts) == 0:
        return 0.0
    if ref.shape[0] == 1:
        return max(0.0, float(ref[0] - pts[:, 0].min()))
    pts = pts[np.argsort(pts[:, -1], kind='mergesort')]
    vol = 0.0
    for i in range(len(pts)):
        upper = float(pts[i + 1, -1]) if i + 1 < len(pts) else float(ref[-1])
        if upper - float(pts[i, -1]) > 0:
            proj = pts[:i + 1][:, :-1]
            proj = proj[pareto_mask(proj)]
            vol += (upper - float(pts[i, -1])) * _hv_slice(proj, ref[:-1])
    return vol


def hypervolume_exact(front, ref):
    front, ref = np.asarray(front, float), np.asarray(ref, float)
    pts = front[np.all(front <= ref, axis=1)]
    return float(_hv_slice(pts[pareto_mask(pts)], ref)) if len(pts) else 0.0


# proof: 20 random fronts vs brute-force inclusion-exclusion
_rng = np.random.default_rng(0)
for _t in range(20):
    _n, _m = int(_rng.integers(2, 9)), int(_rng.integers(2, 5))
    _F, _r = _rng.random((_n, _m)), np.ones(_m)
    _got, _exp = hypervolume_exact(_F, _r), 0.0
    for _mask in range(1, 1 << _n):
        _idx = [i for i in range(_n) if _mask & (1 << i)]
        _c = _F[_idx].max(axis=0)
        if np.all(_c < _r):
            _exp += ((-1) ** (len(_idx) + 1)) * float(np.prod(_r - _c))
    assert abs(_got - _exp) < 1e-9, (_t, _got, _exp)
print('HV self-test: 20 random fronts match inclusion-exclusion -> OK')
''')

code(r'''
# @title 4.1 Load candidates + build the 4 objectives
if INPUT_CSV:
    df = pd.read_csv(INPUT_CSV)
    _ren = {c: ('sequence' if c.strip().lower() in ('sequence', 'seq', 'peptide')
                else 'p_active') for c in df.columns
            if c.strip().lower() in ('sequence', 'seq', 'peptide',
                                     'p_active', 'p_amp', 'prob', 'p(amp|x)')}
    df = df.rename(columns=_ren)
    if 'Aliphatic_index' in df.columns:
        df = df.rename(columns={'Aliphatic_index': 'Aliphatic_csv'})
    df['sequence'] = df['sequence'].astype(str).str.strip().str.upper()
    df = df[df['sequence'].str.len() >= 5].copy().reset_index(drop=True)
    print('loaded %d candidates from %s' % (len(df), INPUT_CSV))
else:
    df = pd.DataFrame({
        'sequence': ['AINAGRINKPRPLRYLLLWLCKA', 'IINGGKVRTINASRVFMGIWRG',
                     'NGWNGNRAFRVYRGAYGWGRV', 'NGGKRMTIYGRWINRVSLS',
                     'RCTANGRNGKHVYIYNKSLGAKLLRA', 'NCRSMSRARIAVVSLCGRA',
                     'EANKHGEPARVRLDTARRRVRGAGRAGA', 'DAASRRTAGRGRRPARCAAAGGRARA',
                     'NGQNGTVRKVYGSGGRQR'],
        'p_active': [0.707967026029467, 0.4045824196816295, 0.353593317764368,
                     0.3115315750292665, 0.21602309874787048, 0.15658139998127193,
                     0.10891916289862995, 0.09337037278139897, 0.030351906000177287]})
    print('demo mode: %d reference sequences' % len(df))

rows = []
for _s, _p in zip(df['sequence'], pd.to_numeric(df['p_active'], errors='coerce')):
    _h, _mu, _q, _ai = gravy(_s), helicoidal_moment(_s), net_charge_seq_pH7(_s), aliphatic_index(_s)
    rows.append({'sequence': _s, 'length': len(_s), 'p_active': float(_p),
                 'Aliphatic_recomputed': _ai, 'toxicity_score': toxicity_score(_s, _h, _mu, _q),
                 'hemolytic_activity': hemolytic_activity(_s, _h, _mu, _q)})
obj = pd.DataFrame(rows)
obj['f1_neg_potency'] = -obj['p_active']
obj['f2_neg_aliphatic'] = -obj['Aliphatic_recomputed']
obj['f3_toxicity'] = obj['toxicity_score']
obj['f4_hemolysis'] = obj['hemolytic_activity']

if 'Aliphatic_csv' in df.columns:
    _d = (obj['Aliphatic_recomputed'] - pd.to_numeric(df['Aliphatic_csv'], errors='coerce').values)
    print('Aliphatic recompute vs 2.5 CSV: max abs diff = %.4f' % float(np.nanmax(np.abs(_d))))
obj.head(10).round(4)
''')

code(r'''
# @title 4.2 ACPE: non-dominated sort + hypervolume + rank
F = obj[['f1_neg_potency', 'f2_neg_aliphatic', 'f3_toxicity', 'f4_hemolysis']].values.astype(float)
REF = np.array([0.0, float(np.nanmax(F[:, 1])) + 10.0, 1.0, 1.0])  # anti-optimal point

if BACKEND == 'pymoo':
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    from pymoo.indicators.hv import HV
    _fronts = NonDominatedSorting().do(F)
    mask = np.zeros(len(F), bool)
    mask[_fronts[0]] = True
    hv = float(HV(ref_point=REF)(F[mask]))
    print('pymoo vs exact dominance:', 'MATCH' if np.array_equal(pareto_mask(F), mask) else 'MISMATCH')
    backend_used = 'pymoo'
else:
    mask = pareto_mask(F)
    hv = hypervolume_exact(F[mask], REF)
    backend_used = 'exact-numpy (pymoo-equivalent)'

_ideal = F[mask].min(axis=0)
_order = np.argsort(np.where(mask, np.linalg.norm(F - _ideal, axis=1), np.inf))
obj['is_pareto'] = mask
obj['acpe_rank'] = 0
obj.loc[_order, 'acpe_rank'] = np.arange(1, len(F) + 1)
obj = obj.sort_values(['is_pareto', 'acpe_rank'], ascending=[False, True]).reset_index(drop=True)

print('backend: %s | candidates: %d | P*: %d | HV(%s) = %.4f'
      % (backend_used, len(obj), int(mask.sum()), np.round(REF, 3), hv))
obj[['sequence', 'p_active', 'Aliphatic_recomputed', 'toxicity_score',
     'hemolytic_activity', 'is_pareto', 'acpe_rank']].round(4)
''')

code(r'''
# @title 5.1 Plots + artefacts (P* recommended for 2.8 AlphaFold / docking / MD)
import matplotlib.pyplot as plt

obj.to_csv(f'{WORKDIR}/02_7_objectives_all.csv', index=False)
obj[obj['is_pareto']].to_csv(f'{WORKDIR}/02_7_pareto_front_Pstar.csv', index=False)

is_p = obj['is_pareto']
fig, ax = plt.subplots(2, 3, figsize=(13, 8))
for a, (x, y, t) in zip(ax.ravel(),
        [('f1_neg_potency', 'f3_toxicity', 'Potency (-p) vs Toxicity'),
         ('f1_neg_potency', 'f4_hemolysis', 'Potency (-p) vs Hemolysis'),
         ('f2_neg_aliphatic', 'f3_toxicity', 'Stability (-AI) vs Toxicity'),
         ('f2_neg_aliphatic', 'f4_hemolysis', 'Stability (-AI) vs Hemolysis'),
         ('f3_toxicity', 'f4_hemolysis', 'Toxicity vs Hemolysis'),
         ('p_active', 'Aliphatic_recomputed', 'Potency vs Aliphatic (max-max view)')]):
    a.scatter(obj.loc[~is_p, x], obj.loc[~is_p, y], s=60, alpha=0.55, label='Dominated')
    a.scatter(obj.loc[is_p, x], obj.loc[is_p, y], s=110, marker='*', label='Pareto P*')
    a.set_xlabel(x)
    a.set_ylabel(y)
    a.set_title(t)
    a.grid(alpha=0.3)
ax.ravel()[0].legend()
fig.suptitle('ACPE 4-objective trade-offs', fontsize=11)
fig.tight_layout()
fig.savefig(f'{WORKDIR}/02_7_pareto_pairs.png', dpi=200, bbox_inches='tight')
plt.show()

cols = ['f1_neg_potency', 'f2_neg_aliphatic', 'f3_toxicity', 'f4_hemolysis']
norm = (obj[cols] - obj[cols].min()) / (obj[cols].max() - obj[cols].min() + 1e-12)
fig2, a2 = plt.subplots(figsize=(9, 5))
for i in norm.index:
    a2.plot(range(4), norm.loc[i].values, linewidth=2.5 if is_p.loc[i] else 1.0,
            alpha=1.0 if is_p.loc[i] else 0.35)
    if is_p.loc[i]:
        a2.text(3.05, norm.loc[i].values[-1], obj.loc[i, 'sequence'][:10], fontsize=7)
a2.set_xticks(range(4), ['f1\n-p(AMP)', 'f2\n-Aliphatic', 'f3\nToxicity', 'f4\nHemolysis'])
a2.set_ylabel('min-max normalized objective (lower = better)')
a2.set_title('Parallel coordinates: Pareto front (bold) vs dominated (faint)')
a2.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig(f'{WORKDIR}/02_7_parallel_coords.png', dpi=200, bbox_inches='tight')
plt.show()

print('PARETO SET P* (forward to 2.8):')
print(obj[obj['is_pareto']][['acpe_rank', 'sequence', 'p_active', 'Aliphatic_recomputed',
      'toxicity_score', 'hemolytic_activity']].round(4).to_string(index=False))
''')

code(r'''
# @title 5.2 Download results
from google.colab import files
for _f in ['02_7_objectives_all.csv', '02_7_pareto_front_Pstar.csv',
            '02_7_pareto_pairs.png', '02_7_parallel_coords.png']:
    files.download(f'{WORKDIR}/{_f}')
''')


def build():
    cells = []
    for i, (ctype, src) in enumerate(CELLS):
        lines = src.splitlines(keepends=True)
        cell = {'cell_type': ctype,
                'id': 'cell-%02d-%s' % (i, ctype[:3]),
                'metadata': {},
                'source': lines}
        if ctype == 'code':
            cell['execution_count'] = None
            cell['outputs'] = []
        cells.append(cell)
    return {'cells': cells,
            'metadata': {'colab': {'provenance': [], 'toc_visible': True},
                         'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                         'language_info': {'name': 'python'}},
            'nbformat': 4, 'nbformat_minor': 5}


if __name__ == '__main__':
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else 'Klebsiella_AMP_27_ACPE_Pareto_Colab.ipynb'
    nb = build()
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')
    n_md = sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')
    n_code = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
    print('wrote %s | %d cells (%d markdown, %d code)' % (out, len(nb['cells']), n_md, n_code))
