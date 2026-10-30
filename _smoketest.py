# -*- coding: utf-8 -*-
"""Offline smoke test for the notebook's logic (no Colab drive, small workload)."""
import json
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')

# the dataset holds mojibake units, and the console here is cp1252
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, 'Klebsiella_AMP_PepFoundry_RDKit_ML_Colab.ipynb')
CSV = r'C:\Users\hp_laptop\Downloads\Klebsiella pneumoniae\natural-modified-combined.csv'

nb = json.load(open(NB, encoding='utf-8'))
G = {'__name__': '__main__'}

# the notebook is the single source of truth: find cells by their @title so this test
# survives insertions/removals instead of drifting on hard-coded indices
CODE = [(i, ''.join(c['source'])) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']


def title_of(src):
    for ln in src.splitlines():
        if ln.startswith('# @title'):
            return ln.split('@title', 1)[1].strip()
    return src.splitlines()[0][:60] if src.splitlines() else ''


def pick(prefix):
    for i, src in CODE:
        if title_of(src).startswith(prefix):
            return i
    raise KeyError('no cell titled %r' % prefix)


def cell_src(i):
    return ''.join(nb['cells'][i]['source'])


def run(i):
    t0 = time.time()
    print('\n' + '=' * 90)
    print('CELL %d :: %s' % (i, title_of(cell_src(i))))
    print('=' * 90, flush=True)
    exec(compile(cell_src(i), 'cell%d' % i, 'exec'), G)
    print('--- cell %d finished in %.0f s' % (i, time.time() - t0), flush=True)


# --- config cell, then shrink the workload -----------------------------------
run(pick('2. Configuration'))
G['CSV_PATH'] = CSV
G['WORKDIR'] = r'C:\Users\HP_LAP~1\AppData\Local\Temp\opencode\amp_pf_test'
os.makedirs(G['WORKDIR'], exist_ok=True)
G['SCREEN_SEEDS'] = [42]
G['SEEDS'] = [42, 7]
G['TUNING_N_ITER'] = 2
G['N_BOOTSTRAP'] = 30
G['CV_REPEATS'] = 1
G['CV_REPORT_BLOCKS'] = ['RDKit', 'PepFoundry']
G['SPLIT_PAIRS'] = [(1.00, 0.00), (0.80, 0.20), (0.60, 0.40)]
G['DO_TUNING'] = False        # skip heavy Optuna in smoke test

# every code cell except the pip install cell (449 s, all packages present here),
# the Colab Drive mount (needs Colab -> replaced by CSV_PATH above) and the GIN cell,
# which is run last under an explicit flag
skip = {pick('0. Install packages'), pick('3.1 Mount Google Drive'), pick('15.1 GIN')}
for i, _ in CODE:
    if i not in skip:
        run(i)

G['SHAP_MAX_SAMPLES'] = 60
G['RUN_GNN'] = False
run(pick('15.1 GIN'))
G['RUN_GNN'] = True
G['GNN_SEEDS'] = [42]
G['GNN_EPOCHS'] = 2
run(pick('15.1 GIN'))

print('\nSMOKE TEST COMPLETED WITHOUT EXCEPTION')
