# -*- coding: utf-8 -*-
"""Run the optional GIN cell for real (the cell defines its own constants, so patch the source)."""
import json
import os
import time

import matplotlib
matplotlib.use('Agg')

NB = 'Klebsiella_AMP_PepFoundry_RDKit_ML_Colab.ipynb'
nb = json.load(open(NB, encoding='utf-8'))
G = {'__name__': '__main__'}


def run(i, patch=None):
    src = ''.join(nb['cells'][i]['source'])
    if patch:
        for a, b in patch:
            assert a in src, 'patch target missing: %s' % a
            src = src.replace(a, b)
    t0 = time.time()
    print('\n' + '=' * 90)
    print('CELL %d :: %s' % (i, src.splitlines()[0][:80]))
    print('=' * 90, flush=True)
    exec(compile(src, 'cell%d' % i, 'exec'), G)
    print('--- cell %d finished in %.0f s' % (i, time.time() - t0), flush=True)


# reuse the artefacts of the previous smoke run
G['CSV_PATH'] = r'C:\Users\hp_laptop\Downloads\Klebsiella pneumoniae\natural-modified-combined.csv'
run(3)                                                  # config (defines the PF_*/USE_* knobs)
G['WORKDIR'] = r'C:\Users\HP_LAP~1\AppData\Local\Temp\opencode\amp_pf_test'
y = __import__('numpy').load(os.path.join(G['WORKDIR'], 'y.npy'))
_df = __import__('pandas').read_csv(os.path.join(G['WORKDIR'], '02_dataset_final.csv'))
G['df'] = _df
G['seqs'] = _df['sequence'].tolist()

print('n=%d | actives=%d' % (len(y), int(y.sum())))
run(13)                                                # PepFoundry block, from cache
run(16)                                                # RDKit block, from cache
run(18)                                                # blocks + SafeSVD
run(20)                                                # models
run(21)                                                # metrics
run(22)                                                # splits
run(40, [('RUN_GNN = False', 'RUN_GNN = True'),
         ('GNN_SEEDS = [42, 7, 123]', 'GNN_SEEDS = [42]'),
         ('GNN_EPOCHS = 60', 'GNN_EPOCHS = 3')])
print('\nGIN TEST COMPLETED WITHOUT EXCEPTION')
