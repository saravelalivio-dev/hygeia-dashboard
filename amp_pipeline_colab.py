# =============================================================================
# AMP PIPELINE (Google Colab)
# PepFoundry + RDKit + modlAMP - Klebsiella pneumoniae only
# Copy each "--- CELL n ---" block as one notebook cell, run top to bottom.
# =============================================================================

# ------------------------------- CELL 1: installs ---------------------------
!pip install -q git+https://github.com/BilodeauGroup/PepFoundry.git rdkit shap modlamp openbabel-wheel ipython xgboost lightgbm

# ---------------------- CELL 2: load your CSV file --------------------------
from google.colab import files
uploaded = files.upload()
CSV_PATH = "FINAL_CD_HIT_DATASET_balanced.csv"

# ---------------------- CELL 3: imports + constants -------------------------
import numpy as np
import pandas as pd
from math import factorial
import itertools

from rdkit import RDLogger
from rdkit.Chem import Descriptors, Crippen, Lipinski
from pepfoundry.interface import PepFoundry
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, accuracy_score, matthews_corrcoef)

import shap
import matplotlib.pyplot as plt

RDLogger.DisableLog("rdApp.*")

OUT_CSV = "kpneumoniae_features_math.csv"
D_ATOM, D_BOND = 24, 16
D_EMBED = (D_ATOM + D_BOND) * 4
SEED = 42

# --------------------- CELL 4: data loading --------------------------------
def load_data(path):
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()].copy()

    def find(part):
        return next((c for c in df.columns if part.lower() in str(c).lower()), None)

    c_target = find("target")
    c_seq = find("sequence")
    c_id = find("peptide")
    c_act = find("activity") if find("activity") else find("label")
    print("cols -> target:", c_target, "| seq:", c_seq, "| id:", c_id, "| activity:", c_act)
    if not (c_seq and c_act):
        raise KeyError(f"could not find required columns. Available: {list(df.columns)}")

    df = df.map(lambda s: s.replace("Âµ", "µ").replace("Ã", "") if isinstance(s, str) else s)
    df = df[df[c_seq].notna()].copy()
    df[c_seq] = df[c_seq].astype(str).str.strip()

    if c_target:
        ts = df[c_target].fillna("").astype(str).str.strip()
        df = df[(ts.str.contains("pneumoniae", case=False, na=False) | (ts == "")) &
                ~ts.str.contains("aerogenes", case=False, na=False)]

    act = df[c_act]
    if pd.api.types.is_numeric_dtype(act):
        label = act.astype(int)
    else:
        label = act.astype(str).str.strip().str.lower().map({"active": 1, "inactive": 0})
    df = df[label.notna()].copy()
    df["label"] = label.astype(int)
    df = df.drop_duplicates(subset=[c_id, c_seq])
    df = df.rename(columns={c_seq: "sequence", c_id: "Peptide ID"})
    return df

# -------------------- CELL 5: feature extraction ---------------------------
def pepfoundry_embedding(pf, seq):
    mol = pf.get_peptide(seq, plot_peptide=False)
    atom_t, bond_t = pf.get_peptide_tensor_atomic_features(mol)
    atom_t = np.asarray(atom_t, dtype=float)
    bond_t = np.asarray(bond_t, dtype=float)
    def agg(t):
        return np.concatenate([t.mean(0), t.std(0), t.max(0), t.sum(0)])
    emb = np.concatenate([agg(atom_t), agg(bond_t)])
    assert len(emb) == D_EMBED
    return emb

def rdkit_descriptors(mol):
    return np.array([
        Descriptors.MolWt(mol), Crippen.MolLogP(mol), Descriptors.TPSA(mol),
        Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol), Descriptors.FractionCSP3(mol),
        Descriptors.RingCount(mol), mol.GetNumHeavyAtoms(),
        sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()),
    ])

def modlamp_descriptors(seq):
    gd = GlobalDescriptor(seq); gd.calculate_charge()
    pd_ = PeptideDescriptor(seq, "Eisenberg"); pd_.calculate_moment()
    return np.array([gd.descriptor.ravel()[0], pd_.descriptor.ravel()[0]])

def build_features(df):
    pf = PepFoundry()
    rows, failed = [], []
    RDK = ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotatableBonds",
           "FractionCSP3", "RingCount", "NumHeavyAtoms", "AromaticAtoms"]
    for _, row in df.iterrows():
        seq = row["sequence"]
        try:
            mol = pf.get_peptide(seq, plot_peptide=False)
            vec = np.concatenate([pepfoundry_embedding(pf, seq),
                                  rdkit_descriptors(mol),
                                  modlamp_descriptors(seq)])
            rows.append({"Peptide ID": row["Peptide ID"], "sequence": seq,
                         "label": row["label"], "v": vec})
        except Exception as e:
            failed.append((seq, str(e)))
    print(f"features: {len(rows)}  failed: {len(failed)}")
    out = pd.DataFrame(rows)
    M = np.vstack(out["v"].values)
    cols = [f"pf_{k}" for k in range(D_EMBED)] + RDK + ["net_charge", "hydrophobic_moment"]
    out = pd.concat([out[["Peptide ID", "sequence", "label"]].reset_index(drop=True),
                     pd.DataFrame(M, columns=cols)], axis=1)
    out.to_csv(OUT_CSV, index=False)
    return out, cols

# ------------- CELL 6: benchmarking suite + split tuning + SHAP -------------
def select_top_features(model, X, K):
    expl = shap.TreeExplainer(model)
    sv = np.asarray(expl.shap_values(X))
    vals = sv[:, :, 1] if sv.ndim == 3 else (np.asarray(sv[1]) if isinstance(sv, list) else sv)
    return np.argsort(np.abs(vals).mean(0))[::-1][:K]

def get_models_and_grids():
    """Return dict of model_name -> (base_estimator, param_grid) for GridSearchCV."""
    return {
        "LogisticRegression": (
            LogisticRegression(random_state=SEED, class_weight="balanced", max_iter=2000, n_jobs=-1),
            {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"]}
        ),
        "SVM": (
            SVC(random_state=SEED, class_weight="balanced", probability=True),
            {"C": [0.1, 1, 10], "kernel": ["rbf"], "gamma": ["scale"]}
        ),
        "DecisionTree": (
            DecisionTreeClassifier(random_state=SEED, class_weight="balanced"),
            {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]}
        ),
        "XGBoost": (
            XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=-1,
                          scale_pos_weight=1153/221),
            {"n_estimators": [100, 300], "max_depth": [3, 6, 10],
             "learning_rate": [0.05, 0.1], "min_child_weight": [1, 3]}
        ),
        "LightGBM": (
            LGBMClassifier(random_state=SEED, class_weight="balanced", n_jobs=-1, verbose=-1),
            {"n_estimators": [100, 300], "max_depth": [3, 6, 10],
             "learning_rate": [0.05, 0.1], "num_leaves": [15, 31]}
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=SEED, class_weight="balanced", n_jobs=-1),
            {"n_estimators": [100, 300], "max_depth": [10, 20, None],
             "min_samples_split": [2, 5]}
        ),
    }

def train_model(X, y, feature_names):
    print(f"matrix: {X.shape[0]} x {X.shape[1]} (embedding {D_EMBED} + physchem)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models_grids = get_models_and_grids()
    tuned = {}

    for name, (base, grid) in models_grids.items():
        gs = GridSearchCV(base, grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0)
        gs.fit(X, y)
        tuned[name] = gs.best_estimator_
        print(f"{name}: best params {gs.best_params_} | CV AUC {gs.best_score_:.4f}")

    best_K, best_score, sel_idx = 30, -1, None
    for K in [15, 30, 50]:
        idx = select_top_features(tuned["RandomForest"], X, K)
        s = cross_val_score(tuned["RandomForest"], X[:, idx], y, cv=cv, scoring="roc_auc").mean()
        print(f"top-{K} features CV AUC: {s:.4f}")
        if s > best_score:
            best_score, best_K, sel_idx = s, K, idx
    X = X[:, sel_idx]
    names = np.asarray(feature_names)[sel_idx]
    print(f"selected {len(sel_idx)} features (best K={best_K}, CV AUC {best_score:.4f})")

    summary = []
    best_mcc, best_model, best_name = -1, None, None
    train_sizes = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    for tr in train_sizes:
        ts = 1 - tr
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=ts, stratify=y, random_state=SEED)
        for name, base in tuned.items():
            model = clone(base)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            p = model.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            auroc = roc_auc_score(y_test, p)
            ratio = f"{int(tr*100)}/{int(ts*100)}"
            print(f"\n=== {name} | Split {ratio} ===")
            print(f"Accuracy: {acc:.4f}  MCC: {mcc:.4f}  AUROC: {auroc:.4f}")
            summary.append((ratio, name, round(acc, 4), round(mcc, 4), round(auroc, 4)))
            if mcc > best_mcc:
                best_mcc, best_model, best_name = mcc, model, name
                keep = (model, X_train, y_train, X_test, ratio)

    print("\n=== Summary (train/test, model) ===")
    print(f"{'split':<7}{'model':<18}{'accuracy':<10}{'MCC':<8}{'AUROC':<8}")
    for ratio, name, acc, mcc, auroc in summary:
        print(f"{ratio:<7}{name:<18}{acc:<10}{mcc:<8}{auroc:<8}")
    print(f"\n>>> Best model: {best_name} (MCC={best_mcc:.4f}) on split {keep[4]}")
    return keep[0], keep[1], keep[2], keep[3], names, best_name

# ------------- CELL 7: exact SHAP formula + TreeSHAP ------------------------
def coalition_predict(tree, x, S, background):
    x = x.copy()
    not_S = [i for i in range(len(x)) if i not in S]
    preds = []
    for b in background:
        xb = x.copy(); xb[not_S] = b[not_S]
        preds.append(tree.predict_proba(xb.reshape(1, -1))[0, 1])
    return float(np.mean(preds))

def exact_shapley(tree, x, K, background):
    F = list(range(K)); phis = np.zeros(K)
    for j in F:
        others = [i for i in F if i != j]
        for r in range(K):
            w = factorial(r) * factorial(K - r - 1) / factorial(K)
            for S in itertools.combinations(others, r):
                S = set(S)
                phis[j] += w * (coalition_predict(tree, x, set(S) | {j}, background) -
                                coalition_predict(tree, x, S, background))
    return phis

def verify_shap_formula(model, X_train, y_train, X_test, feature_names):
    K = 6
    imp = np.argsort(model.feature_importances_)[::-1][:K]
    Xk = X_train[:, imp]
    tree = DecisionTreeClassifier(max_depth=3, random_state=SEED)
    tree.fit(Xk, y_train)
    x = X_test[0, imp]
    exact = exact_shapley(tree, x, K, Xk[:50])
    expl = shap.TreeExplainer(tree, data=Xk[:50], feature_perturbation="interventional",
                              model_output="probability")
    sv = np.asarray(expl.shap_values(x.reshape(1, -1)))
    sv = sv[:, :, 1] if sv.ndim == 3 else (sv[1] if isinstance(sv, list) else sv)
    print("\nexact SHAP formula vs TreeSHAP (K={}, top: {})".format(K,
          [feature_names[i] for i in imp]))
    for j in range(K):
        print(f"  phi_{j}: exact={exact[j]:+.6f}  treeshap={sv.ravel()[j]:+.6f}")
    print("  max abs diff:", round(float(np.abs(exact - sv.ravel()[:K]).max()), 6))

# ----------------------- CELL 8: SHAP plots ---------------------------------
def explain_full(model, X_test, feature_names):
    expl = shap.TreeExplainer(model)
    sv = np.asarray(expl.shap_values(X_test))
    vals = sv[:, :, 1] if sv.ndim == 3 else (np.asarray(sv[1]) if isinstance(sv, list) else sv)
    shap.summary_plot(vals, X_test, feature_names=feature_names, show=True)
    mean_abs = np.abs(vals).mean(0)
    order = np.argsort(mean_abs)[::-1][:15]
    plt.figure(figsize=(8, 5))
    plt.barh(np.arange(len(order))[::-1], mean_abs[order], color="#4c72b0")
    plt.yticks(np.arange(len(order))[::-1], np.asarray(feature_names)[order])
    plt.xlabel("mean |phi_j| (SHAP value)")
    plt.title("Top dimensions driving AMP activity (TreeSHAP)")
    plt.show()

# -------------------- CELL 9: run the full pipeline -------------------------
df = load_data(CSV_PATH)
print(f"K. pneumoniae rows: {len(df)} (Active={df['label'].sum()}, Inactive={(1-df['label']).sum()})")

feats, feat_cols = build_features(df)
X = feats[feat_cols].to_numpy()
y = feats["label"].to_numpy()

model, X_train, y_train, X_test, sel_names, best_name = train_model(X, y, feat_cols)
verify_shap_formula(model, X_train, y_train, X_test, sel_names)
explain_full(model, X_test, sel_names)

# Save ProtGPT2-ready CSV (selected features only)
protgpt_csv = "kpneumoniae_features_protgpt2.csv"
sel_cols = list(sel_names) + ["label"]
out_df = feats[["Peptide ID", "sequence"] + sel_cols].copy()
out_df.to_csv(protgpt_csv, index=False)
print(f"Saved ProtGPT2-ready file: {protgpt_csv} ({out_df.shape[1]-2} features + label)")
files.download(protgpt_csv)
files.download(OUT_CSV)