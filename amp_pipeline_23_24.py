"""Sections 2.3 and 2.4 of the proposal, with ESM-2 -> PepFoundry
and ProtParam -> RDKit substitutions. Focus: Klebsiella pneumoniae only.

Implements the math verbatim:
  - 2.4.1: ensemble probability    p^(AMP|x) = (1/B) * sum_{b=1}^{B} p_b(AMP|x)
  - 2.4.2: TreeSHAP feature value  phi_j = sum_{S subset F\\{j}} [ |S|!(|F|-|S|-1)! / |F|! ]
                                           * [ f_x(S u {j}) - f_x(S) ]

A brute-force implementation of the 2.4.2 formula is included (on a single tree
and a small feature subset) and validated against TreeSHAP, then TreeSHAP is
applied to the full model for interpretability.
"""

import numpy as np
import pandas as pd
from math import factorial

from rdkit import RDLogger
from rdkit.Chem import Descriptors, Crippen, Lipinski
from pepfoundry.interface import PepFoundry
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RDLogger.DisableLog("rdApp.*")

CSV_PATH = r"C:\Users\hp_laptop\Downloads\natural-modified-combined.csv"
OUT_CSV = "kpneumoniae_features_math.csv"

# d_embed = number of PepFoundry embedding dimensions
D_ATOM = 24          # PepFoundry per-atom feature dimension
D_BOND = 16          # PepFoundry per-bond feature dimension
D_EMBED = (D_ATOM + D_BOND) * 4   # mean, std, max, sum -> 160

SEED = 42


# ---------------------------------------------------------------------------
# 1. Data loading (Klebsiella pneumoniae only; ESKAPE disregarded)
# ---------------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    df = df.map(lambda s: s.replace("Âµ", "µ").replace("Ã", "") if isinstance(s, str) else s)
    df = df[df["sequence"].notna()].copy()
    df["sequence"] = df["sequence"].astype(str).str.strip()

    df = df[df["Target Species"].str.contains("Klebsiella pneumoniae", case=False, na=False)]
    df = df.drop_duplicates(subset=["Peptide ID", "sequence"])

    label = df["Activity"].str.lower().map({"active": 1, "inactive": 0})
    df = df[label.notna()].copy()
    df["label"] = label.astype(int)
    return df


# ---------------------------------------------------------------------------
# 2. Feature extraction
# ---------------------------------------------------------------------------
def pepfoundry_embedding(pf, seq):
    """2.3.1 replacement - PepFoundry graph embedding (was: ESM-2 tokenizer + model).
    Builds the peptide 3D graph and aggregates the per-atom [n,24] and per-bond [n,16]
    tensor features into a fixed d_embed=160 dimensional vector."""
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
    """2.3.2 replacement - RDKit physicochemical descriptors (was: ProtParam).
    The Instability/Aliphatic/pI/GRAVY style properties are approximated by:
    MolWt, LogP, TPSA, HBD, HBA, rotatable bonds, FractionCSP3, rings."""
    return np.array([
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        Descriptors.TPSA(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.RingCount(mol),
        mol.GetNumHeavyAtoms(),
        sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()),
    ])


def modlamp_descriptors(seq):
    """2.3.2 additional AMP descriptors - modlAMP (net charge, hydrophobic moment)."""
    gd = GlobalDescriptor(seq)
    gd.calculate_charge()
    pd_ = PeptideDescriptor(seq, "Eisenberg")
    pd_.calculate_moment()
    return np.array([gd.descriptor.ravel()[0], pd_.descriptor.ravel()[0]])


def build_features(df):
    """2.3.3 Feature integration: X = [embedding E | RDKit P | modlAMP M]."""
    pf = PepFoundry()
    rows, failed = [], []
    RDKIT_NAMES = ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotatableBonds",
                   "FractionCSP3", "RingCount", "NumHeavyAtoms", "AromaticAtoms"]

    for i, (_, row) in enumerate(df.iterrows()):
        seq = row["sequence"]
        try:
            mol = pf.get_peptide(seq, plot_peptide=False)
            feat = np.concatenate([pepfoundry_embedding(pf, seq),
                                   rdkit_descriptors(mol),
                                   modlamp_descriptors(seq)])
            rows.append({"Peptide ID": row["Peptide ID"], "sequence": seq,
                         "Target Species": row["Target Species"],
                         "label": row["label"], "feature_vector": feat})
        except Exception as e:
            failed.append((seq, str(e)))
        if (i + 1) % 100 == 0:
            print(f"  processed {i+1}/{len(df)}")

    print(f"features computed: {len(rows)}  failed: {len(failed)}")
    for seq, err in failed[:10]:
        print(f"  FAIL {seq}: {err}")

    out = pd.DataFrame(rows)
    base = out[["Peptide ID", "sequence", "Target Species", "label"]]
    M = np.vstack(out["feature_vector"].values)
    cols = [f"pf_{k}" for k in range(D_EMBED)] + \
           RDKIT_NAMES + ["net_charge", "hydrophobic_moment"]
    out = pd.concat([base.reset_index(drop=True),
                     pd.DataFrame(M, columns=cols)], axis=1)
    out.to_csv(OUT_CSV, index=False)
    pd.Series(cols).to_csv("feature_columns_math.csv", index=False, header=False)
    return out, cols


# ---------------------------------------------------------------------------
# 3. 2.4.1 Random Forest + hyperparameter search
# ---------------------------------------------------------------------------
def ensemble_probability(model, x):
    """p^(AMP|x) = (1/B) * sum_{b=1}^{B} p_b(AMP|x)   [formula 2.4.1]
    Averaging each tree's predicted AMP probability over the B trees."""
    x = np.atleast_2d(x)
    p_all = np.column_stack([est.predict_proba(x)[:, 1] for est in model.estimators_])
    return p_all.mean(axis=1)


def train_model(X, y):
    print(f"\n[2.4.1] feature matrix: {X.shape[0]} samples x {X.shape[1]} features "
          f"(embedding d={D_EMBED} + physchem)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    param_grid = {"n_estimators": [100, 300],
                  "max_depth": [None, 10, 20],
                  "min_samples_split": [2, 5]}
    grid = GridSearchCV(RandomForestClassifier(random_state=SEED,
                                              class_weight="balanced", n_jobs=-1),
                        param_grid,
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
                        scoring="roc_auc", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    print("best_params:", grid.best_params_)
    print("best CV AUC:", round(grid.best_score_, 4))

    # --- verify 2.4.1 formula against sklearn's aggregation -----------------
    p_formula = ensemble_probability(model, X_test)          # (1/B) sum p_b
    p_sklearn = model.predict_proba(X_test)[:, 1]
    print("2.4.1 formula vs sklearn max abs diff:",
          round(float(np.abs(p_formula - p_sklearn).max()), 12))

    y_pred = model.predict(X_test)
    print("test ROC-AUC:", round(roc_auc_score(y_test, p_sklearn), 4))
    print(classification_report(y_test, y_pred, target_names=["Inactive", "Active"]))
    print("confusion matrix:\n", confusion_matrix(y_test, y_pred))
    return model, X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 4. 2.4.2 SHAP: exact formula (small demo) + full TreeSHAP
# ---------------------------------------------------------------------------
def coalition_predict(tree, x, S, background):
    """f_x(S): expected AMP probability given only the features in subset S,
    with the remaining features masked by a background distribution."""
    x = x.copy()
    not_S = [i for i in range(len(x)) if i not in S]
    preds = []
    for b in background:
        xb = x.copy()
        xb[not_S] = b[not_S]
        preds.append(tree.predict_proba(xb.reshape(1, -1))[0, 1])
    return float(np.mean(preds))


def exact_shapley(tree, x, K, background):
    """Brute force of formula 2.4.2 for K features and one tree:
    phi_j = sum_{S subset F\\{j}} |S|!(|F|-|S|-1)! / |F|! * [f_x(S u {j}) - f_x(S)]
    NOTE: expensive in |F| -> used here only as a verification of TreeSHAP."""
    F = list(range(K))
    phis = np.zeros(K)
    for j in F:
        others = [i for i in F if i != j]
        for r in range(K):                       # |S| = r
            weight = factorial(r) * factorial(K - r - 1) / factorial(K)
            for S in itertools_combinations(others, r):
                S = set(S)
                f_with = coalition_predict(tree, x, set(S) | {j}, background)
                f_without = coalition_predict(tree, x, S, background)
                phis[j] += weight * (f_with - f_without)
    return phis


import itertools
def itertools_combinations(items, r):
    return itertools.combinations(items, r)


def verify_shap_formula(model, X_train, y_train, X_test, feature_names):
    """Verify the exact SHAP formula (2.4.2) against TreeSHAP.

    The formula sums over ALL subsets of F, so the demonstration uses a closed
    game: a shallow decision tree trained only on the top-K features. Then the
    brute-force evaluation of phi_j and TreeSHAP must agree exactly."""
    K = 6
    imp = np.argsort(model.feature_importances_)[::-1][:K]
    top_cols = [feature_names[i] for i in imp]

    Xk = X_train[:, imp]
    xk_test = X_test[:, imp]
    tree = DecisionTreeClassifier(max_depth=3, random_state=SEED)
    tree.fit(Xk, y_train)

    background = Xk[:50]
    x = xk_test[0]
    exact = exact_shapley(tree, x, K, background)

    exp_treeshap = shap.TreeExplainer(tree, data=background,
                                      feature_perturbation="interventional",
                                      model_output="probability")
    sv = np.asarray(exp_treeshap.shap_values(x.reshape(1, -1)))
    sv = sv[:, :, 1] if sv.ndim == 3 else (sv[1] if isinstance(sv, list) else sv)
    treeshap = sv.ravel()[:K]

    print("\n[2.4.2] exact formula (closed game, K={}, top features {}) vs TreeSHAP:"
          .format(K, top_cols))
    for j in range(K):
        print(f"  phi_{j}: exact={exact[j]:+.6f}  treeshap={treeshap[j]:+.6f}")
    print("  max abs diff:", round(float(np.abs(exact - treeshap).max()), 6))
    return exact, treeshap


def explain_full(model, X_test, feature_names):
    """Full TreeSHAP on the ensemble + plots (2.4.2)."""
    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(explainer.shap_values(X_test))
    vals = shap_values[:, :, 1] if shap_values.ndim == 3 else \
           (np.asarray(shap_values[1]) if isinstance(shap_values, list) else shap_values)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(vals, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary_math.png", dpi=200, bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(vals).mean(0)
    order = np.argsort(mean_abs)[::-1][:15]
    plt.figure(figsize=(8, 5))
    plt.barh(np.arange(len(order))[::-1], mean_abs[order], color="#4c72b0")
    plt.yticks(np.arange(len(order))[::-1], np.asarray(feature_names)[order])
    plt.xlabel("mean |phi_j| (SHAP value)")
    plt.title("Top embedding/physiochemical dimensions driving activity (TreeSHAP)")
    plt.tight_layout()
    plt.savefig("shap_top_features_math.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("saved shap_summary_math.png / shap_top_features_math.png")

    print("\nTop dimensions mapped back to sequence context (phi_j):")
    for idx in order:
        lbl = feature_names[idx]
        print(f"  {lbl:20s} mean|phi|={mean_abs[idx]:.4f}")
    return vals


def main():
    print("[2.3] loading Klebsiella pneumoniae data...")
    df = load_data(CSV_PATH)
    print(f"rows: {len(df)}  (Active={df['label'].sum()}, Inactive={(1 - df['label']).sum()})")

    print("[2.3.1-2.3.3] computing PepFoundry + RDKit + modlAMP features...")
    feats, feat_cols = build_features(df)

    X = feats[feat_cols].to_numpy()
    y = feats["label"].to_numpy()

    model, Xtr, Xte, ytr, yte = train_model(X, y)
    verify_shap_formula(model, Xtr, ytr, Xte, feat_cols)
    explain_full(model, Xte, feat_cols)


if __name__ == "__main__":
    main()