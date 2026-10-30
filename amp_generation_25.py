"""Section 2.5 - Tree-SHAP-Guided De Novo Peptide Generation.

2.5.1  ProtGPT2 candidate generation from top wild-type templates
2.5.2  Tree-SHAP guided structural filtering (length, charge, hydrophobicity,
       amphipathicity/hydrophobic moment, stability) using RDKit + modlAMP
2.5.3  Novelty assessment (dedupe, valid residues, similarity vs known AMPs)
2.5.4  Re-extraction (PepFoundry + RDKit + modlAMP) and RF re-evaluation

NOTE (2.5.4 substitution): ESM-2 -> PepFoundry, ProtParam -> RDKit, as before.
Run with:
    C:\\Users\\hp_laptop\\AppData\\Local\\Programs\\Python\\Python312\\python.exe amp_generation_25.py
For Colab: paste into cells (see end-of-file cell markers) and install with:
    !pip install -q git+https://github.com/BilodeauGroup/PepFoundry.git rdkit shap modlamp openbabel-wheel ipython transformers
"""

import re
import numpy as np
import pandas as pd
from difflib import SequenceMatcher

from rdkit import RDLogger
from rdkit.Chem import Descriptors, Crippen, Lipinski
from pepfoundry.interface import PepFoundry
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor

from sklearn.ensemble import RandomForestClassifier

import shap
import torch

RDLogger.DisableLog("rdApp.*")

# ------------------------------ configuration ---------------------------------
FEAT_CSV = "kpneumoniae_features_math.csv"   # output of the 2.3/2.4 pipeline
OUT_CSV = "generated_candidates.csv"
PROTGPT2_ID = "nferruz/ProtGPT2"

N_TEMPLATES = 5            # top wild-type templates to evolve
SAMPLES_PER_TEMPLATE = 20  # candidates generated per template
MIN_LEN, MAX_LEN = 10, 30  # 2.5.2 optimal length window
MIN_CHARGE = 2.0           # positive net charge
HYDRO_RATIO = (0.25, 0.70) # balanced hydrophobicity
MIN_MOMENT = 0.20          # amphipathicity (Eisenberg hydrophobic moment)
STAB_MAX = 40.0            # instability index < 40 (stable)
NOVELTY_CUTOFF = 0.75      # drop if similarity to known AMP > cutoff
PROB_CUTOFF = 0.80         # 2.5.4 keep candidates with p(AMP|x) >= cutoff
SEED = 42

D_ATOM, D_BOND = 24, 16
D_EMBED = (D_ATOM + D_BOND) * 4
RDK = ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotatableBonds",
       "FractionCSP3", "RingCount", "NumHeavyAtoms", "AromaticAtoms"]
VALID_AA = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")


def feature_columns():
    return [f"pf_{k}" for k in range(D_EMBED)] + RDK + ["net_charge", "hydrophobic_moment"]


# ---------------------- shared feature extraction (2.3) -----------------------
def candidate_vector(pf, seq):
    mol = pf.get_peptide(seq, plot_peptide=False)
    atom_t, bond_t = pf.get_peptide_tensor_atomic_features(mol)
    atom_t, bond_t = np.asarray(atom_t, float), np.asarray(bond_t, float)
    def agg(t):
        return np.concatenate([t.mean(0), t.std(0), t.max(0), t.sum(0)])
    emb = np.concatenate([agg(atom_t), agg(bond_t)])
    rdk = np.array([Descriptors.MolWt(mol), Crippen.MolLogP(mol), Descriptors.TPSA(mol),
                    Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol),
                    Descriptors.NumRotatableBonds(mol), Descriptors.FractionCSP3(mol),
                    Descriptors.RingCount(mol), mol.GetNumHeavyAtoms(),
                    sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())])
    gd = GlobalDescriptor(seq); gd.calculate_charge()
    pd_ = PeptideDescriptor(seq, "Eisenberg"); pd_.calculate_moment()
    mod = np.array([gd.descriptor.ravel()[0], pd_.descriptor.ravel()[0]])
    return np.concatenate([emb, rdk, mod])


# ------------------------- 2.5.2 / 2.5.4 filters -------------------------------
def structural_filters(seq):
    """Tree-SHAP guided filters: length, net charge, hydrophobicity,
    amphipathicity, instability index (replaces ProtParam via modlAMP)."""
    ok_len = MIN_LEN <= len(seq) <= MAX_LEN
    if not ok_len:
        return False, "length"
    gd = GlobalDescriptor(seq)
    gd.calculate_charge()
    gd.hydrophobic_ratio()
    gd.instability_index()
    vals = np.asarray(gd.descriptor).ravel()
    charge, hyr, instab = vals[0], vals[1], vals[2]
    pd_ = PeptideDescriptor(seq, "Eisenberg"); pd_.calculate_moment()
    moment = float(pd_.descriptor.ravel()[0])
    if charge < MIN_CHARGE: return False, "charge"
    if not (HYDRO_RATIO[0] <= hyr <= HYDRO_RATIO[1]): return False, "hydrophobicity"
    if moment < MIN_MOMENT: return False, "moment"
    if instab > STAB_MAX: return False, "instability"
    return True, "ok"


def max_similarity(seq, reference):
    """Max SequenceMatcher ratio vs. known AMPs (2.5.3 novelty)."""
    return max(SequenceMatcher(None, seq, ref).ratio() for ref in reference)


# ------------------------- 2.5.1 ProtGPT2 generation --------------------------
def generate_candidates(templates, per_template=20, seed=SEED):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[2.5.1] loading ProtGPT2 ({PROTGPT2_ID}) ...")
    tok = AutoTokenizer.from_pretrained(PROTGPT2_ID)
    lm = AutoModelForCausalLM.from_pretrained(PROTGPT2_ID)
    lm.eval()
    torch.manual_seed(seed)

    start = tok("<|endoftext|>", add_special_tokens=False).input_ids
    candidates = []
    for ti, template in enumerate(templates):
        prefix = template[:10]                      # seed with a WT fragment
        prompt = start + tok(prefix.lower(), add_special_tokens=False).input_ids
        with torch.no_grad():
            out = lm.generate(
                torch.tensor([prompt]), max_new_tokens=MAX_LEN + 10,
                do_sample=True, top_k=950, temperature=0.7,
                repetition_penalty=1.2, num_return_sequences=per_template)
        for s in out:
            text = tok.decode(s, skip_special_tokens=True)
            seq = text.upper().replace(" ", "")
            if seq.startswith(prefix.upper()):
                seq = seq[len(prefix):]
            if VALID_AA.match(seq):
                candidates.append((template, seq))
        print(f"  template {ti+1}/{len(templates)} -> {sum(1 for c in candidates if c[0]==template)} valid raw")
    return candidates, tok, lm


# ---------------------------------- main --------------------------------------
def main():
    print("[2.5] loading feature table from pipeline 2.3/2.4 ...")
    feats = pd.read_csv(FEAT_CSV)
    cols = feature_columns()
    X_wt = feats[cols].to_numpy()
    y_wt = feats["label"].to_numpy()
    seqs_wt = feats["sequence"].tolist()

    # trained Random Forest evaluator (2.5.4)
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=2,
                                class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf.fit(X_wt, y_wt)
    expl = shap.TreeExplainer(rf)

    # top-performing wild-type templates (2.5.1)
    p_wt = rf.predict_proba(X_wt)[:, 1]
    templates = [seqs_wt[i] for i in np.argsort(p_wt)[::-1]
                 if y_wt[i] == 1][:N_TEMPLATES]
    print("templates:", templates)

    # 2.5.1 generate
    raw, tok, lm = generate_candidates(templates, SAMPLES_PER_TEMPLATE)
    print(f"[2.5.1] generated {len(raw)} sequences (ProtGPT2)")

    # 2.5.3 dedupe + valid residues
    uniq = list(dict.fromkeys(s for _, s in raw))
    print(f"[2.5.3] after dedupe: {len(uniq)}")

    # 2.5.2 structural filters (Tree-SHAP guided) + 2.5.3 novelty
    known = list(dict.fromkeys(seqs_wt))            # dataset incl. DBAASP = known AMP pool
    pf = PepFoundry()
    kept = []
    for s in uniq:
        ok, why = structural_filters(s)
        if not ok:
            continue
        if max_similarity(s, known) > NOVELTY_CUTOFF:
            continue
        kept.append(s)
    print(f"[2.5.2/2.5.3] passed structural + novelty filters: {len(kept)}")
    if not kept:
        print("no candidates survived - relax thresholds or generate more samples")
        return

    # 2.5.4 re-extract + re-evaluate
    rows, failed = [], []
    for s in kept:
        try:
            v = candidate_vector(pf, s)
            prob = float(rf.predict_proba(v.reshape(1, -1))[:, 1][0])
            sv = np.asarray(expl.shap_values(v.reshape(1, -1)))
            sv = sv[:, :, 1] if sv.ndim == 3 else (sv[1] if isinstance(sv, list) else sv)
            rows.append({"sequence": s, "length": len(s),
                         "p_AMP": round(prob, 4),
                         "mean_abs_shap": round(float(np.abs(sv).mean()), 4),
                         "net_charge_shap": round(float(sv[0, cols.index("net_charge")]), 4)})
        except Exception as e:
            failed.append((s, str(e)))

    out = pd.DataFrame(rows)
    final = out[out["p_AMP"] >= PROB_CUTOFF]
    out.to_csv(OUT_CSV, index=False)

    print("\n[2.5.4] re-evaluation of", len(out), "candidates (RF + SHAP)")
    print(out.describe().round(4).loc[["mean", "min", "max"]])
    print(f"\nFINAL SELECTED ({PROB_CUTOFF} cutoff): {len(final)} peptides")
    print(final.to_string(index=False))
    print("\nsaved", OUT_CSV)

# =============================================================================
# Colab: copy everything above into cells; install with the one-liner in the
# docstring. generate_candidates() downloads ProtGPT2 (~500 MB) on first run.
# =============================================================================
if __name__ == "__main__":
    main()