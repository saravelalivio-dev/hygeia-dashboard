# -*- coding: utf-8 -*-
"""Cold-run test: no caches on disk, feature extraction from scratch."""
import json
import os
import shutil
import time

import matplotlib
matplotlib.use('Agg')

NB = 'Klebsiella_AMP_PepFoundry_RDKit_ML_Colab.ipynb'
WD = r'C:\Users\HP_LAP~1\AppData\Local\Temp\opencode\amp_pf_cold'
if os.path.isdir(WD):
    shutil.rmtree(WD)
os.makedirs(WD)

nb = json.load(open(NB, encoding='utf-8'))
G = {'__name__': '__main__'}


def run(i):
    src = ''.join(nb['cells'][i]['source'])
    t0 = time.time()
    print('\n=== CELL %d :: %s' % (i, src.splitlines()[0][:70]), flush=True)
    exec(compile(src, 'cell%d' % i, 'exec'), G)
    print('--- cell %d in %.0f s' % (i, time.time() - t0), flush=True)


G['CSV_PATH'] = r'C:\Users\hp_laptop\Downloads\Klebsiella pneumoniae\natural-modified-combined.csv'
run(3)
G['WORKDIR'] = WD
for i in (6, 7, 9, 10, 11, 13, 16, 18):
    run(i)
print('\ncaches written:', sorted(f for f in os.listdir(WD) if f.endswith('.npz')))
print('COLD RUN COMPLETED WITHOUT EXCEPTION')
