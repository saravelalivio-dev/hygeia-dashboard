# -*- coding: utf-8 -*-
"""End-to-end test of the flattened single-file pipeline (no Colab, no Drive)."""
import os
import re
import shutil
import sys

import matplotlib
matplotlib.use('Agg')

SRC = 'Klebsiella_AMP_PepFoundry_RDKit.py'
WD = r'C:\Users\HP_LAP~1\AppData\Local\Temp\opencode\amp_pf_single'
if os.path.isdir(WD):
    shutil.rmtree(WD)
os.makedirs(WD)

src = open(SRC, encoding='utf-8').read()

# shrink the workload, point it at the local CSV and a temp workdir
src = src.replace("SEED = 42", "SEED = 42", 1)
src = src.replace("WORKDIR = (os.path.join(DRIVE_ROOT, 'amp_klebsiella_pepfoundry')\n"
                  "           if os.path.isdir(DRIVE_ROOT) else '/content/amp_klebsiella')",
                  "WORKDIR = r'%s'" % WD, 1)
src = src.replace("CSV_PATH = resolve_csv(CSV_CANDIDATES)",
                  r"CSV_PATH = r'C:\Users\hp_laptop\Downloads\Klebsiella pneumoniae\natural-modified-combined.csv'")
src = re.sub(r'^SCREEN_SEEDS = .*$', 'SCREEN_SEEDS = [42]', src, flags=re.M)
src = re.sub(r'^SEEDS = .*$', 'SEEDS = [42, 7]', src, flags=re.M)
src = re.sub(r'^TUNING_N_ITER = .*$', 'TUNING_N_ITER = 2', src, flags=re.M)
src = re.sub(r'^N_BOOTSTRAP = .*$', 'N_BOOTSTRAP = 30', src, flags=re.M)
src = re.sub(r'^CV_REPEATS = .*$', 'CV_REPEATS = 1', src, flags=re.M)
src = re.sub(r'SPLIT_PAIRS = \[.*?\]\n', 'SPLIT_PAIRS = [(1.00, 0.00), (0.80, 0.20), (0.60, 0.40)]\n',
             src, flags=re.S)
src = re.sub(r'^CV_REPORT_BLOCKS = .*$', "CV_REPORT_BLOCKS = ['RDKit', 'PepFoundry']", src, flags=re.M)
src = re.sub(r'^_pip\(', '_no_op(', src, flags=re.M)
src = ('_no_op = lambda *pkgs: None\n') + src

import ast
ast.parse(src)
print('patched copy parses OK, %d chars' % len(src), flush=True)
exec(compile(src, SRC, 'exec'), {'__name__': '__main__'})
print('\nSINGLE-FILE RUN COMPLETED WITHOUT EXCEPTION')
print('artefacts:', sorted(os.listdir(WD)))
