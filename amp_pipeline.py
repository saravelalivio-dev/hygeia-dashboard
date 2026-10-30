"""AMP pipeline for Klebsiella pneumoniae (sections 2.3 / 2.4).

Replaces ESM-2 embeddings with PepFoundry graph-embedding summaries and
ProtParam with RDKit physicochemical descriptors. modlAMP descriptors, the
Random Forest classifier and TreeSHAP interpretability follow the proposal.
Run with the Python 3.12 interpreter:
    C:\\Users\\hp_laptop\\AppData\\Local\\Programs\\Python\\Python312\\python.exe amp_pipeline.py
"""

import numpy as np
import pandas as pd

from rdkit import RDLogger
from rdkit.Chem import Descriptors, Crippen, Lipinski
from pepfoundry.interface import PepFoundry
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RDLogger.DisableLog("rdApp.*")

CSV_PATH = r"C:\Users\hp_laptop\Downloads\natural-modified-combined.csv"
OUT_CSV = "kpneumoniae_features.csv"
SEED = 42


# ---------------------------------------------------------------------------
# 1. Data loading (handles mojibake, duplicate headers, ragged DBAASP rows)
# ---------------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    df = df.map(lambda s: s.replace("Âµ", "µ").replace("Ã", "") if isinstance(s, str) else s)
    df = df[df["sequence"].notna()].copy()                         # drop trailing blank rows
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
    """PepFoundry graph embedding (replaces ESM-2): aggregate per-atom [n,24]
    and per-bond [n,16] tensor features into a fixed-length vector."""
    mol = pf.get_peptide(seq, plot_peptide=False)
    atom_t, bond_t = pf.get_peptide_tensor_atomic_features(mol)
    atom_t = np.asarray(atom_t, dtype=float)
    bond_t = np.asarray(bond_t, dtype=float)

    def agg(t):
        return np.concatenate([t.mean(0), t.std(0), t.max(0), t.sum(0)])

    return np.concatenate([agg(atom_t), agg(bond_t)])


def rdkit_descriptors(mol):
    """RDKit physicochemical descriptors (replaces ProtParam)."""
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotatableBonds": Descriptors.NumRotatableBonds(mol),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
        "RingCount": Descriptors.RingCount(mol),
        "NumHeavyAtoms": mol.GetNumHeavyAtoms(),
        "AromaticAtoms": sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()),
    }


def modlamp_descriptors(seq):
    """modlAMP descriptors per proposal 2.3.2."""
    gd = GlobalDescriptor(seq)
    gd.calculate_charge()
    net_charge = gd.descriptor.ravel()[0]
    pd_ = PeptideDescriptor(seq, "Eisenberg")
    pd_.calculate_moment()
    hydro_moment = pd_.descriptor.ravel()[0]
    return {"net_charge": net_charge, "hydrophobic_moment": hydro_moment}


def build_features(df):
    pf = PepFoundry()
    rows, failed, names = [], [], []

    for i, (_, row) in enumerate(df.iterrows()):
        seq = row["sequence"]
        try:
            mol = pf.get_peptide(seq, plot_peptide=False)
            if mol is None:
                mol = Chem.MolFromSequence(seq)
            emb = pepfoundry_embedding(pf, seq)
            feat = {"Peptide ID": row["Peptide ID"],
                    "sequence": seq,
                    "Target Species": row["Target Species"],
                    "synthesis_type": row.get("synthesis type (for modified)", np.nan),
                    "label": row["label"]}
            feat.update(rdkit_descriptors(mol))
            feat.update(modlamp_descriptors(seq))
            for k, v in enumerate(emb):
                feat[f"pepfoundry_{k}"] = v
            rows.append(feat)
            names.append(seq)
        except Exception as e:  # noqa: BLE001 - keep going if one peptide fails
            failed.append((seq, str(e)))
        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(df)}")

    print(f"features computed for {len(rows)} peptides; {len(failed)} failed")
    for seq, err in failed[:10]:
        print(f"  FAIL {seq}: {err}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    return out


# ---------------------------------------------------------------------------
# 3. Training / optimisation (proposal 2.4.1)
# ---------------------------------------------------------------------------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=SEED, class_weight="balanced", n_jobs=-1),
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    print("best_params:", grid.best_params_)
    print("best CV AUC:", round(grid.best_score_, 4))
    y_prob = grid.predict_proba(X_test)[:, 1]
    y_pred = grid.predict(X_test)
    print("test ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))
    print(classification_report(y_test, y_pred, target_names=["Inactive", "Active"]))
    print("confusion matrix:\n", confusion_matrix(y_test, y_pred))
    return grid.best_estimator_, X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 4. SHAP interpretability (proposal 2.4.2)
# ---------------------------------------------------------------------------
def explain(model, X_train, X_test, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    sv = np.asarray(shap_values)
    if sv.ndim == 3:
        vals = sv[:, :, 1]                      # class 1 = Active
    elif isinstance(shap_values, list):
        vals = np.asarray(shap_values[1])
    else:
        vals = sv

    plt.figure()
    shap.summary_plot(vals, X_test, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(vals).mean(0)
    order = np.argsort(mean_abs)[::-1]
    top = 12
    plt.figure(figsize=(8, 5))
    plt.barh(np.arange(top)[::-1], mean_abs[order[:top]], color="#4c72b0")
    plt.yticks(np.arange(top)[::-1], np.asarray(feature_names)[order[:top]])
    plt.xlabel("mean |SHAP value| (impact on activity prediction)")
    plt.title("Top features driving AMP prediction (TreeSHAP)")
    plt.tight_layout()
    plt.savefig("shap_top_features.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("saved shap_summary.png and shap_top_features.png")


def main():
    df = load_data(CSV_PATH)
    print(f"Klebsiella pneumoniae rows: {len(df)}  (Active={df['label'].sum()}, "
          f"Inactive={(1 - df['label']).sum()})")

    print("computing features...")
    feats = build_features(df)

    feat_cols = [c for c in feats.columns if c.startswith("pepfoundry_") or c in
                 ("MolWt", "LogP", "TPSA", "HBD", "HBA", "RotatableBonds",
                  "FractionCSP3", "RingCount", "NumHeavyAtoms", "AromaticAtoms",
                  "net_charge", "hydrophobic_moment")]
    X = feats[feat_cols].to_numpy()
    y = feats["label"].to_numpy()
    print(f"feature matrix: {X.shape[0]} samples x {X.shape[1]} features")

    model, Xtr, Xte, ytr, yte = train_model(X, y)
    explain(model, Xtr, Xte, feat_cols)
    pd.Series(feat_cols).to_csv("feature_columns.csv", index=False)


if __name__ == "__main__":
    main()