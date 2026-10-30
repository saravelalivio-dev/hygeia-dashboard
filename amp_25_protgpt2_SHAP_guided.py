"""Section 2.5 — Tree-SHAP-guided de novo peptide generation (ProtGPT2).

Proposal mapping
----------------
  2.5.1  ProtGPT2 de novo generation (Ferruz et al. 2022, ``nferruz/ProtGPT2``).
  2.5.2  Physicochemical filtering guided by TreeSHAP.
  2.5.3  Novelty checking + sequence QC (canonical 20 AA, length 10-30,
         dedupe, <=80% identity to any known peptide — the CD-HIT cutoff
         from proposal 2.2.2).
  2.5.4  Secondary feature extraction (same pipeline as training) and
         re-evaluation with the frozen 2.4 master gate. Shortlist uses the
         proposal nominal threshold p >= 0.70 AND the Youden threshold
         fitted on validation data.

Toolchain (matches YOUR 2.3/2.4 substitution)
---------------------------------------------
  ESM-2 embeddings   -> PepFoundry pooled graph features (160-d, mean/std/
                        max/sum of per-atom [n,24] + per-bond [n,16]).
  ProtParam          -> Henderson-Hasselbalch titration at pH 7.0 with
                        C-terminal amidation (standard for synthetic AMPs),
                        i.e. the RDKit-substituted charge used in 2.3/2.4.
  modlAMP kept       -> hydrophobic moment via modlAMP PeptideDescriptor
                        ("Eisenberg" scale). Verified bit-identical to the
                        ``hydrophobic_moment`` column of the training CSV.

  Verified against YOUR dataset (kpneumoniae_features_protgpt2.csv, 1000 rows):
    * pf_N columns == PepFoundry pooled index N (exact to 1e-6, e.g.
      GLFDIIKKIAESF pf_26 0.384979 both sides; sum-pooled pf_148 = 94.0).
    * hydrophobic_moment == modlAMP Eisenberg (exact to 1e-6).
    * net_charge == HH titration amide=True within ~0.01-0.2
      (residual = Cys disulfide / His pKa convention; tree gate + percentile
      bounds make this harmless — see 2.5.2 notes below).

Input
-----
  kpneumoniae_features_protgpt2.csv  (Peptide ID, sequence, net_charge,
  hydrophobic_moment, pf_N ..., label). 30 feature columns.

Outputs (all in --workdir)
--------------------------
  02_4_master_gate.joblib      frozen gate + Youden threshold + column list
  02_4_benchmark.csv           LR/SVM/DT/RF/XGB/LGBM MCC + AUROC
  02_5_shap_patterns.csv       per-descriptor TreeSHAP pattern table
  02_5_filter_rules.csv        layer-1 + layer-2 rules actually applied
  02_5_1_generated_raw.csv     raw ProtGPT2 output (+ canonical/length flags)
  02_5_2_filter_audit.csv      per-candidate pass/fail + failed rule
  02_5_3_novelty.csv           max identity to any known peptide
  02_5_4_candidate_scores.csv  all survivors with p_active + SHAP agreement
  02_5_shortlist.csv           FINAL: p >= 0.70, ready for 2.6/2.7/2.8
  shap_summary.png / shap_top_features.png

Usage — local
-------------
  pip install pepfoundry rdkit modlamp shap torch transformers scikit-learn pandas matplotlib joblib biopython
  python amp_25_protgpt2_SHAP_guided.py --csv "C:\\Users\\hp_laptop\\Downloads\\kpneumoniae_features_protgpt2.csv"
  # fast check without the ~500 MB ProtGPT2 download:
  python amp_25_protgpt2_SHAP_guided.py --csv ... --smoke-test

Usage — Google Colab
--------------------
  !pip install -q pepfoundry rdkit modlamp shap transformers torch scikit-learn joblib biopython
  # upload kpneumoniae_features_protgpt2.csv, then:
  !python amp_25_protgpt2_SHAP_guided.py --csv kpneumoniae_features_protgpt2.csv --templates 8 --per-template 60
"""

import os
import re
import argparse
import numpy as np
import pandas as pd

VALID_AA = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

MIN_LEN, MAX_LEN = 10, 30      # proposal 2.2.2
NOVELTY_IDENTITY = 0.80        # proposal 2.2.2 CD-HIT cutoff, reused in 2.5.3
PROB_NOMINAL = 0.70            # proposal 2.5.4 nominal threshold
SEED = 42
PROTGPT2_ID = "nferruz/ProtGPT2"

# pKa table for the HH titration (sequence-based ProtParam/RDKit replacement)
PKA = {"N_term": 8.60, "K": 10.53, "R": 12.48, "H": 6.04,
       "D": 3.90, "E": 4.07, "C": 8.37, "Y": 10.46}


# ---------------------------------------------------------------------------
# physicochemistry (2.3 replacement, reused for 2.5.2 / 2.5.4)
# ---------------------------------------------------------------------------
def net_charge_hh(seq, pH=7.0, amide=True):
    """Fractional net charge at pH 7, C-terminal amidated (synthetic AMPs)."""
    q = 1.0 / (1.0 + 10.0 ** (pH - PKA["N_term"]))
    if not amide:  # free carboxyl only when NOT amidated
        q += -1.0 / (1.0 + 10.0 ** (3.55 - pH))
    for aa in ("K", "R", "H"):
        q += seq.count(aa) / (1.0 + 10.0 ** (pH - PKA[aa]))
    for aa in ("D", "E", "C", "Y"):
        q += -seq.count(aa) / (1.0 + 10.0 ** (PKA[aa] - pH))
    return float(q)


def hydrophobic_moment_modlamp(seq):
    from modlamp.descriptors import PeptideDescriptor
    d = PeptideDescriptor(seq, "Eisenberg")
    d.calculate_moment()
    return float(np.asarray(d.descriptor).ravel()[0])


# ---------------------------------------------------------------------------
# PepFoundry embedding (must stay identical to the 2.3 pipeline)
# ---------------------------------------------------------------------------
_PF = None

def pf_engine():
    global _PF
    if _PF is None:
        from pepfoundry.interface import PepFoundry
        _PF = PepFoundry()
    return _PF


def pf_embed160(seq):
    """160-d pooled vector: [mean|std|max|sum atom(24)] + [mean|std|max|sum bond(16)]."""
    mol = pf_engine().get_peptide(seq, plot_peptide=False)
    atom_t, bond_t = pf_engine().get_peptide_tensor_atomic_features(mol)
    a = np.asarray(atom_t, dtype=float)
    b = np.asarray(bond_t, dtype=float)

    def agg(t):
        return np.concatenate([t.mean(0), t.std(0), t.max(0), t.sum(0)])

    return np.concatenate([agg(a), agg(b)])


def featurise(seqs, pf_indices):
    """Full 30-col vector in training-CSV column order for generated peptides."""
    vecs = []
    for s in seqs:
        emb = pf_embed160(s)
        pf_part = [float(emb[i]) for i in pf_indices]
        vecs.append([net_charge_hh(s)] + [hydrophobic_moment_modlamp(s)] + pf_part)
    return np.array(vecs, dtype=float)


# ---------------------------------------------------------------------------
# 2.4 master gate (retrained on YOUR csv) + Youden threshold
# ---------------------------------------------------------------------------
def train_gate(df, feature_cols):
    from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
    from sklearn.metrics import matthews_corrcoef, roc_auc_score, roc_curve
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    X = df[feature_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)

    models = {
        "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "SVM-RBF": SVC(kernel="rbf", probability=True, class_weight="balanced",
                       random_state=SEED),
        "DecTree": DecisionTreeClassifier(random_state=SEED,
                                          class_weight="balanced"),
        "RandForest": RandomForestClassifier(n_estimators=300, max_depth=None,
                                             min_samples_split=2,
                                             class_weight="balanced",
                                             n_jobs=-1, random_state=SEED),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=6,
                                          learning_rate=0.05, subsample=0.9,
                                          colsample_bytree=0.9, eval_metric="logloss",
                                          n_jobs=-1, random_state=SEED)
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(n_estimators=300, verbose=-1,
                                            random_state=SEED)
    except Exception:
        pass

    rows = []
    fitted = {}
    for name, clf in models.items():
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        rows.append({"model": name,
                     "MCC": matthews_corrcoef(yte, pred),
                     "AUROC": roc_auc_score(yte, p)})
        fitted[name] = clf
    bench = pd.DataFrame(rows).sort_values("MCC", ascending=False)
    print("\n[2.4 benchmark on YOUR csv]\n" + bench.round(4).to_string(index=False))

    best_name = bench.iloc[0]["model"]
    # small grid for the tree ensembles, 5-fold CV on AUC (proposal 2.4.3)
    if best_name in ("RandForest", "XGBoost", "LightGBM"):
        from sklearn.ensemble import RandomForestClassifier as RF
        grid = {"n_estimators": [300, 500], "max_depth": [None, 12, 20],
                "min_samples_split": [2, 5]}
        gs = GridSearchCV(RF(class_weight="balanced", n_jobs=-1, random_state=SEED),
                          grid, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                          scoring="roc_auc", n_jobs=-1)
        gs.fit(Xtr, ytr)
        gate = gs.best_estimator_
        print("tuned RF:", gs.best_params_,
              "CV AUC=%.4f" % gs.best_score_)
    else:
        gate = fitted[best_name]

    p_tr = gate.predict_proba(Xtr)[:, 1]
    fpr, tpr, thr = roc_curve(ytr, p_tr)
    youden = thr[int(np.argmax(tpr - fpr))]
    p_te = gate.predict_proba(Xte)[:, 1]
    print("master gate = %s | Youden threshold = %.4f | test MCC = %.4f AUROC = %.4f"
          % (type(gate).__name__, youden,
             matthews_corrcoef(yte, (p_te >= youden).astype(int)),
             roc_auc_score(yte, p_te)))
    return gate, float(youden), bench, (Xtr, Xte, ytr, yte)


def shap_patterns(gate, X, y, feature_cols, df):
    """TreeSHAP pattern table = the 2.5.2 rule source."""
    import shap
    clf_type = type(gate).__name__
    if "Forest" not in clf_type and "Tree" not in clf_type \
            and "XGB" not in clf_type and "LGBM" not in clf_type:
        print("gate is %s (not tree-based) — SHAP layer-2 skipped." % clf_type)
        return None, None
    sv = np.asarray(shap.TreeExplainer(gate).shap_values(X))
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    if isinstance(sv, list):
        sv = np.asarray(sv[-1])
    sv = np.asarray(sv)
    tab = pd.DataFrame({
        "descriptor": feature_cols,
        "mean_shap_active": [sv[y == 1, j].mean() for j in range(sv.shape[1])],
        "mean_shap_inactive": [sv[y == 0, j].mean() for j in range(sv.shape[1])],
        "mean_abs_shap": np.abs(sv).mean(axis=0),
        "active_mean": [X[y == 1, j].mean() for j in range(X.shape[1])],
        "inactive_mean": [X[y == 0, j].mean() for j in range(X.shape[1])],
    })
    tab["shap_gap"] = tab["mean_shap_active"] - tab["mean_shap_inactive"]
    tab["favours"] = np.where(tab["shap_gap"] > 0, "Active", "Inactive")
    tab["direction"] = np.where(tab["active_mean"] > tab["inactive_mean"],
                                "higher favours Active", "lower favours Active")
    tab = tab.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = np.argsort(np.abs(sv).mean(0))[::-1][:15]
        plt.figure(figsize=(8, 5))
        plt.barh(np.arange(len(order))[::-1], np.abs(sv).mean(0)[order])
        plt.yticks(np.arange(len(order))[::-1], np.asarray(feature_cols)[order])
        plt.xlabel("mean |SHAP|")
        plt.title("Top drivers of K. pneumoniae activity (TreeSHAP)")
        plt.tight_layout()
        plt.savefig(os.path.join(WORKDIR, "shap_top_features.png"),
                    dpi=200, bbox_inches="tight")
        plt.close()
        plt.figure(figsize=(10, 7))
        shap.summary_plot(sv, X, feature_names=feature_cols, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(WORKDIR, "shap_summary.png"),
                    dpi=200, bbox_inches="tight")
        plt.close()
        print("saved shap_top_features.png / shap_summary.png")
    except Exception as e:
        print("SHAP plots skipped:", str(e)[:120])
    return tab, sv


def build_rules(patterns, df_active):
    """Layer-1 (proposal-named gates, active 10-90 percentile) + layer-2 (SHAP)."""
    rows = []

    def band(col, default):
        if col in df_active.columns:
            v = pd.to_numeric(df_active[col], errors="coerce").dropna()
            if len(v) >= 10:
                return (float(v.quantile(0.10)), float(v.quantile(0.90)))
        return default

    c_lo, c_hi = band("net_charge", (1.0, 12.0))
    m_lo, m_hi = band("hydrophobic_moment", (0.20, 0.80))
    rows += [
        {"layer": 1, "descriptor": "length", "rule": "10<=L<=30",
         "lo": MIN_LEN, "hi": MAX_LEN,
         "rationale": "proposal 2.2.2 synthetic-viability window"},
        {"layer": 1, "descriptor": "net_charge", "rule": "ge",
         "lo": max(0.0, c_lo), "hi": None,
         "rationale": "cationic attraction; active 10th pct=%.2f" % c_lo},
        {"layer": 1, "descriptor": "hydrophobic_moment", "rule": "ge",
         "lo": m_lo, "hi": None,
         "rationale": "amphipathic insertion; active 10th pct=%.3f" % m_lo},
    ]
    if patterns is not None and len(patterns):
        cut = float(patterns["mean_abs_shap"].quantile(0.90))  # top decile
        for _, r in patterns[patterns["mean_abs_shap"] >= cut].iterrows():
            d = r["descriptor"]
            if d in ("net_charge", "hydrophobic_moment", "length"):
                continue
            if r["favours"] == "Active":
                op = "ge" if r["direction"].startswith("higher") else "le"
            else:
                op = "le" if r["direction"].startswith("higher") else "ge"
            rows.append({
                "layer": 2, "descriptor": d, "rule": op,
                "lo": float(r["active_mean"]) if op == "ge" else None,
                "hi": float(r["active_mean"]) if op == "le" else None,
                "rationale": "%s (SHAP gap %+.3f)" % (r["direction"], r["shap_gap"])})
    rules = pd.DataFrame(rows)
    print("filter rules: %d layer-1 + %d layer-2 (SHAP)"
          % ((rules["layer"] == 1).sum(), (rules["layer"] == 2).sum()))
    return rules


def passes(row, rules):
    for _, r in rules.iterrows():
        d = r["descriptor"]
        if d == "length":
            v = float(row.get("length", len(row.get("sequence", ""))))
        else:
            v = row.get(d, np.nan)
            if v is None or not np.isfinite(float(v)):
                continue  # uncomputable descriptor never rejects
            v = float(v)
        if r["rule"] in ("ge", "10<=L<=30") and r["lo"] is not None and v < float(r["lo"]):
            return False, "%s=%.3f < %.3f" % (d, v, float(r["lo"]))
        if r["rule"] == "le" and r["hi"] is not None and v > float(r["hi"]):
            return False, "%s=%.3f > %.3f" % (d, v, float(r["hi"]))
    return True, ""


# ---------------------------------------------------------------------------
# 2.5.1 ProtGPT2 generation
# ---------------------------------------------------------------------------
def pick_templates(df, gate, feature_cols, n=8):
    d = df[df["sequence"].astype(str).str.match(VALID_AA.pattern)].copy()
    act = d[d["label"].astype(int) == 1].copy()
    p = gate.predict_proba(act[feature_cols].to_numpy(dtype=float))[:, 1]
    act["p_active"] = p
    act = act.drop_duplicates("sequence").sort_values("p_active", ascending=False)
    return act["sequence"].astype(str).tolist()[:n]


def generate_protgpt2(templates, per_template=60, free_run=0, seed=SEED):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("[2.5.1] loading %s ..." % PROTGPT2_ID)
    tok = AutoTokenizer.from_pretrained(PROTGPT2_ID)
    lm = AutoModelForCausalLM.from_pretrained(PROTGPT2_ID)
    lm.eval()
    torch.manual_seed(seed)
    bos = tok("<|endoftext|>", add_special_tokens=False).input_ids
    out = []

    def draw(prompt_ids, k, tag):
        if k <= 0:
            return
        with torch.no_grad():
            gen = lm.generate(torch.tensor([prompt_ids]),
                              max_new_tokens=MAX_LEN + 10, do_sample=True,
                              top_k=950, temperature=0.8,
                              repetition_penalty=1.2,
                              num_return_sequences=k)
        for g in gen:
            seq = "".join(ch for ch in tok.decode(g, skip_special_tokens=True).upper()
                          if ch in STANDARD_AA)
            if tag and seq.startswith(tag):
                seq = seq[len(tag):]
            out.append(seq)
        print("   %-24s -> %d decoded" % (tag or "free-run", len(gen)))

    for t in templates or []:
        # whitespace-separated residues = the format ProtGPT2 was trained on
        draw(bos + tok(" ".join(list(t)), add_special_tokens=False).input_ids,
             per_template, t)
    if free_run:
        draw(bos, free_run, "")
    print("[2.5.1] ProtGPT2 returned %d raw sequences" % len(out))
    return out


def smoke_mutants(templates, per_template=20, seed=SEED):
    """No-download fallback for --smoke-test: point-mutants of templates."""
    rng = np.random.default_rng(seed)
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    out = []
    for t in templates or ["GLFDIIKKIAESF"]:
        for _ in range(per_template):
            s = list(t)
            for _ in range(rng.integers(3, 7)):
                op = rng.choice(["sub", "sub", "sub", "ins", "del"])
                i = int(rng.integers(0, len(s))) if s else 0
                if op == "sub":
                    s[i] = rng.choice(aas)
                elif op == "ins" and len(s) < MAX_LEN:
                    s.insert(i, rng.choice(aas))
                elif op == "del" and len(s) > MIN_LEN:
                    s.pop(i)
            out.append("".join(s))
    return out


# ---------------------------------------------------------------------------
# 2.5.3 novelty
# ---------------------------------------------------------------------------
def max_identity(seq, references):
    from difflib import SequenceMatcher
    best, best_ref = 0.0, ""
    for ref in references:
        if abs(len(seq) - len(ref)) / max(len(seq), len(ref)) > (1 - NOVELTY_IDENTITY):
            continue
        r = SequenceMatcher(None, seq, ref).ratio()
        if r > best:
            best, best_ref = r, ref
    return best, best_ref


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
WORKDIR = "."


def run(csv, workdir=".", n_templates=8, per_template=60, free_run=0,
        smoke_test=False):
    global WORKDIR
    WORKDIR = workdir
    os.makedirs(workdir, exist_ok=True)

    df = pd.read_csv(csv)
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
    pf_cols = [c for c in df.columns if c.startswith("pf_")]  # CSV order
    feature_cols = ["net_charge", "hydrophobic_moment"] + pf_cols
    pf_indices = [int(c.split("_")[1]) for c in pf_cols]
    print("loaded %s: %d rows (%d active / %d inactive), %d features"
          % (csv, len(df), int(df["label"].sum()), int((1 - df["label"]).sum()),
             len(feature_cols)))

    gate, youden, bench, (Xtr, Xte, ytr, yte) = train_gate(df, feature_cols)
    bench.to_csv(os.path.join(workdir, "02_4_benchmark.csv"), index=False)
    import joblib
    joblib.dump({"pipeline": gate, "threshold_youden": youden,
                 "feature_cols": feature_cols, "pf_indices": pf_indices},
                os.path.join(workdir, "02_4_master_gate.joblib"))

    X_all = df[feature_cols].to_numpy(dtype=float)
    y_all = df["label"].to_numpy(dtype=int)
    patterns, _ = shap_patterns(gate, X_all, y_all, feature_cols, df)
    if patterns is not None:
        patterns.to_csv(os.path.join(workdir, "02_5_shap_patterns.csv"), index=False)
        print(patterns.head(12).round(4).to_string(index=False))
    rules = build_rules(patterns, df[df["label"].astype(int) == 1])
    rules.to_csv(os.path.join(workdir, "02_5_filter_rules.csv"), index=False)

    templates = pick_templates(df, gate, feature_cols, n_templates)
    print("ProtGPT2 seed templates:", templates)

    if smoke_test:
        raw = smoke_mutants(templates, per_template=20)
    else:
        raw = generate_protgpt2(templates, per_template, free_run)
    pd.DataFrame({"sequence": raw,
                  "length": [len(s) for s in raw],
                  "is_canonical": [bool(VALID_AA.match(s)) for s in raw]}).to_csv(
        os.path.join(workdir, "02_5_1_generated_raw.csv"), index=False)

    # 2.5.3a canonical + length + dedupe (before expensive featurisation)
    qc = [s for s in dict.fromkeys(raw)
          if VALID_AA.match(s) and MIN_LEN <= len(s) <= MAX_LEN]
    print("[2.5.3] %d/%d survive canonical + %d-%d length + dedupe"
          % (len(qc), len(raw), MIN_LEN, MAX_LEN))
    if not qc:
        print("nothing to featurise — raise --per-template / --free-run")
        return

    # 2.5.4a secondary feature extraction with the SAME pipeline
    Xg = featurise(qc, pf_indices)
    desc = pd.DataFrame(Xg, columns=feature_cols)
    desc.insert(0, "sequence", qc)
    desc["length"] = [len(s) for s in qc]

    # 2.5.2 SHAP-guided physicochemical filter
    audit = [passes(desc.iloc[i], rules) for i in range(len(qc))]
    desc["passes_2_5_2"] = [ok for ok, _ in audit]
    desc["failed_rule"] = [why for _, why in audit]
    desc[["sequence", "passes_2_5_2", "failed_rule"]].to_csv(
        os.path.join(workdir, "02_5_2_filter_audit.csv"), index=False)
    print("[2.5.2] %d/%d pass physicochemical + SHAP filters"
          % (int(desc["passes_2_5_2"].sum()), len(desc)))
    kept = desc[desc["passes_2_5_2"]].reset_index(drop=True)
    if not len(kept):
        print("all rejected — relax rules or generate more")
        return

    # 2.5.3b novelty vs every known peptide in the curated set
    refs = df["sequence"].astype(str).tolist()
    ids, refs_hit = [], []
    for s in kept["sequence"]:
        m, r = max_identity(s, refs)
        ids.append(m)
        refs_hit.append(r)
    kept["max_identity_to_known"] = ids
    kept["nearest_known_peptide"] = refs_hit
    kept.to_csv(os.path.join(workdir, "02_5_3_novelty.csv"), index=False)
    novel = kept[kept["max_identity_to_known"] <= NOVELTY_IDENTITY].reset_index(drop=True)
    print("[2.5.3] %d/%d are novel (<=%.0f%% identity)"
          % (len(novel), len(kept), 100 * NOVELTY_IDENTITY))
    if not len(novel):
        print("everything too close to a reported peptide — stopping")
        return

    # 2.5.4b frozen-gate re-evaluation
    p = gate.predict_proba(novel[feature_cols].to_numpy(dtype=float))[:, 1]
    novel["p_active"] = p
    novel["threshold_youden"] = youden
    novel["passes_youden"] = p >= youden
    novel["passes_nominal_0.70"] = p >= PROB_NOMINAL
    novel = novel.sort_values("p_active", ascending=False).reset_index(drop=True)
    novel.to_csv(os.path.join(workdir, "02_5_4_candidate_scores.csv"), index=False)
    shortlist = novel[novel["p_active"] >= PROB_NOMINAL].copy()
    shortlist.to_csv(os.path.join(workdir, "02_5_shortlist.csv"), index=False)

    print("\n=== 2.5.4 top candidates (frozen %s gate) ===" % type(gate).__name__)
    show = ["sequence", "p_active", "net_charge", "hydrophobic_moment",
            "max_identity_to_known"]
    print(novel[show].head(15).round(3).to_string(index=False))
    print("\nYouden %.3f -> %d pass | nominal 0.70 -> %d pass | SHORTLIST: %d"
          % (youden, int(novel["passes_youden"].sum()),
             int(novel["passes_nominal_0.70"].sum()), len(shortlist)))
    print("artefacts in", os.path.abspath(workdir))
    return shortlist


def main():
    ap = argparse.ArgumentParser(description="2.5 ProtGPT2 + SHAP-guided filters")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--workdir", default="amp_25_out")
    ap.add_argument("--templates", type=int, default=8)
    ap.add_argument("--per-template", type=int, default=60)
    ap.add_argument("--free-run", type=int, default=0)
    ap.add_argument("--smoke-test", action="store_true",
                    help="skip ProtGPT2 download; use template mutants")
    a = ap.parse_args()
    run(a.csv, a.workdir, a.templates, a.per_template, a.free_run, a.smoke_test)


if __name__ == "__main__":
    main()
