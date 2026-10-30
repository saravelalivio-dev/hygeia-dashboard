"""
High-Dimensional, Low-Sample-Size (HDLSS) AMP Classification Pipeline
=====================================================================

Binary classification of antimicrobial peptides (AMP vs non-AMP) built on
ESM-2 mean-pooled embeddings (1280-dim) plus 6 physicochemical descriptors.

Data regime:
    N    = 123 peptide sequences
    D    = 1280 ESM-2 embedding dimensions (primary feature space)
    X    in R^(123 x 1280)  (raw representation)

Because N << D, an unrestricted model can overfit.  Two parallel modeling
conditions are evaluated to control overfitting:

    Model A  Full ESM-2      : X = [ESM-2 (1280) || physchem (6)]   in R^1286
    Model B  PCA-reduced ESM-2: X = [PCA_k (k in {10,20,30,50}) || physchem (6)]

Anti-leakage discipline (strict):
    1. Curation (invalid + duplicate removal) happens BEFORE any split.
    2. 70/15/15 train/validation/test split, stratified by class, BEFORE
       any learned preprocessing.
    3. PCA / scaling are fitted on the TRAINING set only.
       `.fit(X_train)` then `.transform(X_train | X_val | X_test)`
    4. The test set is never used for feature selection, scaling, PCA,
       hyperparameter tuning, or model/representation selection.
    5. Hyperparameters: 5-fold stratified CV inside the training set.
    6. Representation/model selection: VALIDATION set only.
    7. Test set: evaluated exactly once after the final model is chosen.
    8. Overfitting diagnostic: reported as (train - validation) gap.

Run (Colab / CLI):
    python amp_hdlss_pipeline.py --data data.csv --output results/

Colab quickstart:
    from amp_hdlss_pipeline import main
    main(data_csv="data.csv", output_dir="results")
"""

import os
import json
import argparse
import hashlib

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_predict, cross_val_score,
                                     RandomizedSearchCV)
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, matthews_corrcoef, roc_auc_score,
                             average_precision_score, confusion_matrix,
                             roc_curve, precision_recall_curve)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
SEED = 42
CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"   # 1280-dim embedding
PCA_DIMS = (10, 20, 30, 50)
MIN_SEQ_LEN = 10
MAX_SEQ_LEN = 30

PHYSCHEM_FEATURES = [
    "instability_index",
    "aliphatic_index",
    "theoretical_pI",
    "gravy",
    "net_charge",
    "hydrophobic_moment",
]

# Conservative RF hyperparameter search space.
# No unbounded/aggressive depth: candidates are bounded to avoid
# automatically selecting extremely deep trees on N = 123.
DEFAULT_PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced", None],
}


# ===========================================================================
# 1. LOAD DATA
# ===========================================================================
def load_data(csv_path, sequence_col="sequence", label_col="label"):
    """
    Load a cleaned CSV containing `sequence` and `label` (0/1) columns.

    The curated N=123 dataset is expected to have rows of the form:
        sequence,label
        SAVGRHGRRFGLRKHRKH,1

    Returns a DataFrame with the requested columns.
    """
    df = pd.read_csv(csv_path)
    if sequence_col not in df.columns:
        raise ValueError(f"Column '{sequence_col}' not found. "
                         f"Columns: {list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found. "
                         f"Columns: {list(df.columns)}")
    df = df[[sequence_col, label_col]].copy()
    df[label_col] = df[label_col].astype(int)
    df = df.dropna(subset=[sequence_col])
    df[sequence_col] = df[sequence_col].astype(str).str.strip().str.upper()
    print(f"[load_data] Loaded {len(df)} rows from {csv_path}")
    return df


# ===========================================================================
# 2. CLEAN SEQUENCES
# ===========================================================================
def clean_sequences(df, sequence_col="sequence", label_col="label"):
    """
    Apply established curation rules BEFORE any split.

    Rules (as per curation section):
    1. Canonical amino-acid restriction (20 standard residues).
    2. Length boundary filtering (10-30 residues).
    3. Exact duplicate removal.

    Returns (cleaned_df, stats): cleaned rows and a curation report dict.
    """
    n_before = len(df)
    stats = {}

    # Rule 1: canonical residues only
    canon_mask = df[sequence_col].apply(
        lambda s: all(aa in CANONICAL_AA for aa in s)
    )
    n_noncanon = int((~canon_mask).sum())
    df = df[canon_mask]
    stats["removed_non_canonical"] = n_noncanon
    print(f"[clean_sequences] Removed {n_noncanon} non-canonical sequences")

    # Rule 2: length boundaries
    len_mask = df[sequence_col].str.len().between(MIN_SEQ_LEN, MAX_SEQ_LEN)
    n_len = int((~len_mask).sum())
    df = df[len_mask]
    stats["removed_length_out_of_range"] = n_len
    print(f"[clean_sequences] Removed {n_len} sequences outside "
          f"{MIN_SEQ_LEN}-{MAX_SEQ_LEN} aa")

    # Rule 3: exact duplicates (keep first occurrence)
    n_dupes = int(df.duplicated(subset=[sequence_col]).sum())
    df = df.drop_duplicates(subset=[sequence_col], keep="first")
    stats["removed_exact_duplicates"] = n_dupes
    print(f"[clean_sequences] Removed {n_dupes} exact duplicates")

    df = df.reset_index(drop=True)
    stats["n_before"] = n_before
    stats["n_after"] = len(df)
    stats["class_counts"] = df[label_col].value_counts().sort_index().to_dict()
    print(f"[clean_sequences] Final cleaned dataset: {len(df)} sequences")
    return df, stats


# ===========================================================================
# 3. SPLIT DATA (70 / 15 / 15, stratified) -- BEFORE any preprocessing
# ===========================================================================
def split_data(df, sequence_col="sequence", label_col="label",
               train_frac=0.70, val_frac=0.15, test_frac=0.15, seed=SEED):
    """
    Stratified 70/15/15 split into train/validation/test.

    The test set is held out FIRST and is never touched until final
    evaluation.  Original row indices are preserved for feature alignment.
    """
    y = df[label_col].values

    # Hold out test first (15%)
    temp_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=test_frac, stratify=y,
        random_state=seed, shuffle=True)

    # Split remaining 85% into train (70) and validation (15),
    # stratified, with val fraction relative to the temp pool.
    y_temp = y[temp_idx]
    val_frac_of_temp = val_frac / (train_frac + val_frac)
    train_idx, val_idx = train_test_split(
        temp_idx, test_size=val_frac_of_temp, stratify=y_temp,
        random_state=seed + 1, shuffle=True)

    splits = {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "validation": df.iloc[val_idx].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
        "train_orig_idx": train_idx,
        "val_orig_idx": val_idx,
        "test_orig_idx": test_idx,
        "seed": seed,
    }
    for name in ("train", "validation", "test"):
        sub = splits[name]
        print(f"[split_data] {name:>10}: n={len(sub):3d}  "
              f"{(sub[label_col].value_counts().sort_index().to_dict())}")
    return splits


def _resolve_device(device):
    """Map 'auto' to 'cuda' when torch+GPU are available, else 'cpu'."""
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ===========================================================================
# 4. ESM-2 EMBEDDINGS (mean-pooled, 1280-dim)
# ===========================================================================
def extract_esm2_embeddings(sequences, model_name=DEFAULT_ESM2_MODEL,
                            device="auto", batch_size=8, extractor=None):
    """
    Compute mean-pooled ESM-2 embeddings e(s) = (1/L) * SUM h_i.

    A single extractor may be passed in and reused across splits so the
    model is loaded only once.  The pretrained model is frozen; no fitting
    ever touches validation or test data.
    """
    from feature_extraction import ESM2FeatureExtractor

    if extractor is None:
        extractor = ESM2FeatureExtractor(model_name=model_name,
                                         device=_resolve_device(device))
    emb = extractor.get_mean_pooled_embedding(sequences, batch_size=batch_size)
    print(f"[extract_esm2_embeddings] {emb.shape[0]} sequences x "
          f"{emb.shape[1]} dims")
    return emb, extractor


# ===========================================================================
# 5. PHYSICOCHEMICAL DESCRIPTORS (exactly 6 scalar features)
# ===========================================================================
def calculate_physicochemical_features(sequences):
    """
    Compute the six proposal-specified descriptors as scalar features:

        1. Instability Index     2. Aliphatic Index
        3. Theoretical pI        4. GRAVY
        5. Net Charge            6. Hydrophobic Moment

    Returns a DataFrame with exactly 6 columns (no expansion).
    """
    from feature_extraction import PhysicochemicalDescriptorExtractor

    ex = PhysicochemicalDescriptorExtractor()
    rows = []
    for seq in sequences:
        seq = seq.upper()
        rows.append({
            "instability_index": ex.compute_instability_index(seq),
            "aliphatic_index": ex.compute_aliphatic_index(seq),
            "theoretical_pI": ex.compute_pI(seq),
            "gravy": ex.compute_gravy(seq),
            "net_charge": ex.compute_net_charge(seq),
            "hydrophobic_moment": ex.compute_hydrophobic_moment(seq),
        })
    feat = pd.DataFrame(rows, columns=PHYSCHEM_FEATURES)
    print(f"[calculate_physicochemical_features] "
          f"{feat.shape[0]} sequences x {feat.shape[1]} features")
    return feat


# ===========================================================================
# 6. PCA (fitted on TRAINING set ONLY)
# ===========================================================================
def fit_pca(X_train, n_components, scale=True, seed=SEED):
    """
    Fit a PCA (optionally after z-scoring) on the TRAINING set only.

    Returns a bundle with `transform()` implementing:
        X_val  <- bundle.transform(X_val)   (no refit)
        X_test <- bundle.transform(X_test)  (no refit)

    Explained variance is reported for every configuration.
    """
    scaler = None
    X_fit = X_train
    if scale:
        scaler = StandardScaler().fit(X_train)
        X_fit = scaler.transform(X_train)

    pca = PCA(n_components=n_components, random_state=seed).fit(X_fit)
    evr = pca.explained_variance_ratio_
    cum_evr = np.cumsum(evr)

    def _transform(X):
        Xt = scaler.transform(X) if scale else X
        return pca.transform(Xt)

    bundle = {
        "n_components": n_components,
        "scaler": scaler,
        "pca": pca,
        "explained_variance_ratio": evr,
        "cumulative_explained_variance": cum_evr,
        "transform": _transform,
    }
    print(f"[fit_pca] n_components={n_components}  train-explained variance "
          f"cumulative={cum_evr[-1]:.3f}")
    return bundle


# ===========================================================================
# 7. RANDOM FOREST
# ===========================================================================
def train_random_forest(X, y, params=None, seed=SEED):
    """Fit a RandomForestClassifier with (optionally given) parameters."""
    rf = RandomForestClassifier(
        random_state=seed,
        n_jobs=-1,
        **(params or {}),
    )
    rf.fit(X, y)
    return rf


# ===========================================================================
# 8. CROSS-VALIDATION (5-fold stratified, training set ONLY)
# ===========================================================================
def cross_validate_model(X, y, param_grid=None, cv=5, seed=SEED,
                         scoring="roc_auc", n_iter=40):
    """
    Hyperparameter optimization with 5-fold stratified CV on the training
    set only.  Returns the best estimator, best parameters, and out-of-fold
    CV MCC / AUROC statistics.
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=seed, n_jobs=-1),
        param_grid,
        n_iter=n_iter,
        cv=folds,
        scoring=scoring,
        random_state=seed,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    search.fit(X, y)

    best_params = search.best_params_

    # Out-of-fold AUC (mean +/- std)
    cv_auc = cross_val_score(
        RandomForestClassifier(random_state=seed, n_jobs=-1, **best_params),
        X, y, cv=folds, scoring="roc_auc")

    # Out-of-fold MCC
    oof_preds = cross_val_predict(
        RandomForestClassifier(random_state=seed, n_jobs=-1, **best_params),
        X, y, cv=folds)
    cv_mcc = float(matthews_corrcoef(y, oof_preds))

    print(f"[cross_validate_model] best score ({scoring}) = {search.best_score_:.3f}")
    print(f"[cross_validate_model] best params = {json.dumps(best_params)}")
    print(f"[cross_validate_model] out-of-fold CV AUC = {cv_auc.mean():.3f} "
          f"+/- {cv_auc.std():.3f}   CV MCC = {cv_mcc:.3f}")

    return {
        "best_estimator": search.best_estimator_,
        "best_params": best_params,
        "cv_mean_auc": float(cv_auc.mean()),
        "cv_std_auc": float(cv_auc.std()),
        "cv_mcc": cv_mcc,
        "oob_preds": oof_preds,
        "cv_results": search.cv_results_,
    }


# ===========================================================================
# 9. EVALUATION
# ===========================================================================
def evaluate_model(model, X, y, set_name="", threshold=0.5):
    """
    Compute the full requested metric set for a given split.

    Returns a dict with scalar metrics plus y_pred/y_prob for plotting.
    """
    y = np.asarray(y)
    # Predict at default 0.5 threshold
    y_pred_default = model.predict(X)
    probs = model.predict_proba(X)
    pos_col = list(model.classes_).index(1)
    y_prob = probs[:, pos_col]
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "set": set_name,
        "n": int(len(y)),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "accuracy": float(accuracy_score(y, y_pred)),
        "sensitivity_recall": float(recall_score(y, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, y_pred)),
        "auc_roc": float(roc_auc_score(y, y_prob)) if len(set(y)) > 1 else float("nan"),
        "auc_pr": float(average_precision_score(y, y_prob)) if len(set(y)) > 1 else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
    print(f"[evaluate_model] {set_name or 'eval':>10}: MCC={metrics['mcc']:.3f} "
          f"AUROC={metrics['auc_roc']:.3f} AUPRC={metrics['auc_pr']:.3f} "
          f"ACC={metrics['accuracy']:.3f}")
    return metrics


# ===========================================================================
# 10. SHAP (TreeSHAP on the FINAL selected model)
# ===========================================================================
def run_shap_analysis(model, X_explain, feature_names, set_name="train",
                      representation_name="", output_dir=None):
    """
    TreeSHAP analysis for the final Random Forest.

    - Full ESM-2 model: SHAP values map 1:1 to ESM-2 feature dimensions
      (esm2_dim_0 ... esm2_dim_1279) followed by the 6 physchem features.
    - PCA model: SHAP values correspond to PCA *components*, NOT original
      ESM-2 dimensions.  This is stated explicitly, never implied otherwise.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_explain)

    # sklearn binary RF: shap_values may be a list of 2 arrays
    # (class 0, class 1); take class-1 (positive/AMP) values.
    if isinstance(raw, list):
        shap_values = np.asarray(raw[-1])
    else:
        shap_values = np.asarray(raw)
    if shap_values.ndim == 3:
        shap_values = shap_values[..., -1]

    if isinstance(feature_names, (list, tuple, np.ndarray)) and len(feature_names) == shap_values.shape[1]:
        names = list(feature_names)
    else:
        names = [str(i) for i in range(shap_values.shape[1])]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs_shap)[::-1]
    shap_df = pd.DataFrame({
        "feature": [names[i] for i in order],
        "mean_abs_shap": mean_abs_shap[order],
    }).reset_index(drop=True)
    shap_df.index.name = "rank"

    result = {
        "shap_values": shap_values,
        "feature_names": names,
        "summary_df": shap_df,
        "top_features": shap_df.head(30),
    }

    print(f"[run_shap_analysis] mean|SHAP| top-5: "
          f"{list(shap_df['feature'].head(5))}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        tag = representation_name.replace(" ", "_").replace("+", "plus")
        shap_df.to_csv(os.path.join(output_dir, f"shap_{tag}.csv"), index=True)

        fig, ax = plt.subplots(figsize=(8, 7))
        top = shap_df.head(30)
        ax.barh(np.arange(len(top))[::-1], top["mean_abs_shap"])
        ax.set_yticks(np.arange(len(top))[::-1])
        ax.set_yticklabels(top["feature"], fontsize=7)
        ax.set_xlabel("mean |SHAP value|")
        ax.set_title(f"Mean |SHAP| -- {representation_name} (set: {set_name})")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"shap_bar_{tag}.png"), dpi=150)
        plt.close(fig)
        print(f"[run_shap_analysis] Saved SHAP CSV + bar plot to {output_dir}")

    return result


# ===========================================================================
# 11. SAVE RESULTS
# ===========================================================================
def save_results(results, output_dir):
    """
    Persist every output artifact required by the proposal:
    comparison table, full metrics, split composition, PCA explained variance,
    selected hyperparameters, confusion matrix, ROC / PR curves, SHAP.
    """
    os.makedirs(output_dir, exist_ok=True)
    out = results

    # --- Comparison table (as specified) --------------------------------
    comp_rows = []
    for rep_name, rep in out["representations"].items():
        comp_rows.append({
            "Representation": rep_name,
            "Dimensions": rep["n_features"],
            "CV MCC": rep["cv"]["cv_mcc"],
            "Validation MCC": rep["validation"]["mcc"],
            "Test MCC": rep["test"]["mcc"],
            "Test AUROC": rep["test"]["auc_roc"],
            "Test AUPRC": rep["test"]["auc_pr"],
        })
    comp = pd.DataFrame(comp_rows)
    comp["SELECTED"] = comp["Representation"] == out["selected_representation"]
    comp_path = os.path.join(output_dir, "comparison_table.csv")
    comp.to_csv(comp_path, index=False)

    # --- Per-representation full metrics --------------------------------
    metric_rows = []
    for rep_name, rep in out["representations"].items():
        for split_name in ("cv", "train", "validation", "test"):
            m = rep[split_name]
            if split_name == "cv":
                row = {"representation": rep_name, "set": "cv_mean",
                       "mcc": m["cv_mcc"], "auc_roc": m["cv_mean_auc"],
                       "auc_pr": float("nan"), "accuracy": float("nan")}
            else:
                row = {"representation": rep_name, "set": split_name,
                       "mcc": m["mcc"], "auc_roc": m["auc_roc"],
                       "auc_pr": m["auc_pr"], "accuracy": m["accuracy"]}
            metric_rows.append(row)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(os.path.join(output_dir, "metrics_all.csv"), index=False)

    # --- Split composition ----------------------------------------------
    split_rows = []
    for name in ("train", "validation", "test"):
        sub = out["splits"][name]
        st = sub[out.get("label_col", "label")].value_counts().sort_index()
        split_rows.append({
            "split": name,
            "n": int(len(sub)),
            "n_AMP": int(st.get(1, 0)),
            "n_nonAMP": int(st.get(0, 0)),
        })
    pd.DataFrame(split_rows).to_csv(
        os.path.join(output_dir, "split_composition.csv"), index=False)

    # --- PCA explained variance -----------------------------------------
    pca_rows = []
    for rep_name, rep in out["representations"].items():
        pca = rep.get("pca_info")
        if pca is not None:
            pca_rows.append({
                "representation": rep_name,
                "n_components": pca["n_components"],
                "cumulative_explained_variance": round(
                    float(pca["cumulative_explained_variance"][-1]), 4),
                "explained_variance_ratio": [
                    float(v) for v in pca["explained_variance_ratio"]],
            })
    if pca_rows:
        pd.DataFrame(pca_rows).to_csv(
            os.path.join(output_dir, "pca_explained_variance.csv"), index=False)

    # --- Selected model hyperparameters ---------------------------------
    sel = out["representations"][out["selected_representation"]]
    with open(os.path.join(output_dir, "best_hyperparameters.json"), "w") as f:
        json.dump({
            "selected_representation": out["selected_representation"],
            "best_params": sel["cv"]["best_params"],
            "seed": out["seed"],
        }, f, indent=2)

    selected_tag = out["selected_representation"].replace(" ", "_").replace("+", "plus")

    # --- Confusion matrix (test, selected model) ------------------------
    cm = np.array([
        [sel["test"]["tn"], sel["test"]["fp"]],
        [sel["test"]["fn"], sel["test"]["tp"]],
    ])
    pd.DataFrame(cm, index=["true_0", "true_1"],
                 columns=["pred_0", "pred_1"]).to_csv(
        os.path.join(output_dir, f"confusion_matrix_{selected_tag}.csv"))

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred non-AMP", "pred AMP"])
    ax.set_yticklabels(["true non-AMP", "true AMP"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    ax.set_title(f"Confusion Matrix (test) -- {out['selected_representation']}")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"confusion_matrix_{selected_tag}.png"),
                dpi=150)
    plt.close(fig)

    # --- ROC + PR curves (test, selected model) -------------------------
    y_test = out["splits"]["test"][out.get("label_col", "label")].values
    y_prob = sel["test"]["y_prob"]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    pr, re, _ = precision_recall_curve(y_test, y_prob)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(fpr, tpr, label=f"AUC = {sel['test']['auc_roc']:.3f}")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC (test)"); ax1.legend(loc="lower right")

    ax2.plot(re, pr, label=f"AUPRC = {sel['test']['auc_pr']:.3f}")
    base = y_test.mean()
    ax2.axhline(base, color="k", linestyle="--", alpha=0.4, label=f"baseline {base:.2f}")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall (test)"); ax2.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"roc_pr_{selected_tag}.png"), dpi=150)
    plt.close(fig)

    # --- ESM-2 embedding matrix + sequence mapping ----------------------
    np.save(os.path.join(output_dir, "esm2_embeddings_full.npy"),
            out["esm2_embeddings"])
    pd.DataFrame({"sequence": out["sequences"],
                  "label": out["labels"]}).to_csv(
        os.path.join(output_dir, "sequence_labels.csv"), index=False)

    # --- Plain-text report -----------------------------------------------
    report = [f"AMP HDLSS pipeline report",
              f"=================================",
              f"Random seed: {out['seed']}",
              "",
              f"Cleaned dataset: {out['clean_stats']['n_after']} sequences "
              f"(removed {out['clean_stats']['n_before'] - out['clean_stats']['n_after']})",
              "",
              "Split composition:",
              *[f"  {r['split']}: n={r['n']} "
                f"(AMP={r['n_AMP']}, non-AMP={r['n_nonAMP']})"
                for r in split_rows],
              "",
              "Comparison:",
              comp.to_string(index=False),
              "",
              f"SELECTED representation: {out['selected_representation']} "
              f"(by validation MCC {sel['validation']['mcc']:.3f})",
              "",
              "Overfitting diagnostic (training vs validation):",
              *[f"  {name}: train MCC={rep['train']['mcc']:.3f} vs "
                f"val MCC={rep['validation']['mcc']:.3f} "
                f"(gap {rep['train']['mcc'] - rep['validation']['mcc']:+.3f})"
                for name, rep in out["representations"].items()],
              "",
              "SHAP note:",
              f"  SHAP for '{out['selected_representation']}' was computed with "
              f"TreeSHAP. For PCA-based models, SHAP values correspond to PCA "
              f"components, not original ESM-2 dimensions.",
              "",
              "Purpose (per proposal):",
              "  NOT to force a high score. The question is whether the "
              "ESM-2 representation generalizes beyond the curated peptides.",
              ]
    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report))

    print(f"[save_results] All results written to {output_dir}")
    return comp


# ===========================================================================
# MAIN ORCHESTRATION
# ===========================================================================
def build_representations(splits, seed=SEED, device="auto",
                          model_name=DEFAULT_ESM2_MODEL, pca_dims=PCA_DIMS):
    """
    Generate every candidate representation from the stratified splits.

    Returns a dict with:
        - seqs/labels aligned to the cleaned dataset
        - per-split vectorized features for:
            Full ESM-2  (1286 dims)
            PCA-k + physchem (k in {10,20,30,50} -> 16/26/36/56 dims)
    """
    label_col = "label"
    reps = {}

    # ---- ESM-2 embeddings (frozen pretrained model; split-before-embed) --
    extractor = None
    embed = {}
    for name in ("train", "validation", "test"):
        seqs = splits[name]["sequence"].tolist()
        emb, extractor = extract_esm2_embeddings(
            seqs, model_name=model_name, device=device, extractor=extractor)
        embed[name] = emb
        splits[name] = splits[name].copy()
        splits[name]["_emb"] = list(range(len(seqs)))  # alignment marker

    # ---- Physicochemical descriptors (deterministic, per sequence) -------
    phys = {}
    for name in ("train", "validation", "test"):
        seqs = splits[name]["sequence"].tolist()
        phys[name] = calculate_physicochemical_features(seqs)

    # Full cleaned alignment for the saved complete 1280-dim matrix
    all_seqs = pd.concat([splits[n] for n in ("train", "validation", "test")])
    all_seqs = all_seqs.drop(columns=["_emb"])
    all_seqs = all_seqs.reset_index(drop=True)

    # -------- Model A: FULL ESM-2 (1280 + 6) ------------------------------
    X = {}
    for name in ("train", "validation", "test"):
        X[name] = np.hstack([embed[name], phys[name].values])
    reps["Full ESM-2"] = {
        "n_features": X["train"].shape[1],
        "X": X, "y_train": splits["train"][label_col].values,
        "y_val": splits["validation"][label_col].values,
        "y_test": splits["test"][label_col].values,
        "pca_info": None,
        "esm2_block_dims": embed["train"].shape[1],
    }

    # -------- Model B: PCA-k + 6 physchem ---------------------------------
    for k in pca_dims:
        pca_bundle = fit_pca(embed["train"], n_components=k, scale=True)
        Xp = {}
        for name in ("train", "validation", "test"):
            Xp[name] = np.hstack(
                [pca_bundle["transform"](embed[name]), phys[name].values])
        rep_name = f"PCA-{k} + physicochemical"
        reps[rep_name] = {
            "n_features": Xp["train"].shape[1],
            "X": Xp, "y_train": splits["train"][label_col].values,
            "y_val": splits["validation"][label_col].values,
            "y_test": splits["test"][label_col].values,
            "pca_info": {k_: v for k_, v in pca_bundle.items()
                         if k_ != "transform"},
            "esm2_block_dims": k,
        }
        print(f"[build_representations] {rep_name}: "
              f"{Xp['train'].shape[1]} dims")

    # Full ESM-2 embedding matrix + labels (complete representation)
    esm2_full = np.vstack([embed["train"], embed["validation"], embed["test"]])
    labels_full = np.hstack([splits["train"][label_col].values,
                             splits["validation"][label_col].values,
                             splits["test"][label_col].values])

    return {
        "representations": reps,
        "sequences": all_seqs["sequence"].tolist(),
        "labels": labels_full,
        "esm2_embeddings": esm2_full,
        "splits": {n: splits[n].drop(columns=["_emb"])
                   for n in ("train", "validation", "test")},
    }


def run_pipeline(df, output_dir="results", seed=SEED, device="auto",
                 model_name=DEFAULT_ESM2_MODEL, pca_dims=PCA_DIMS,
                 param_grid=None, sequence_col="sequence", label_col="label",
                 n_iter=40):
    """
    End-to-end HDLSS pipeline given a cleaned DataFrame.

    Steps:
        1. clean, 2. split (stratif., 70/15/15), 3. embed (split-wise),
        4. physchem, 5. build representations (A + B), 6. tune via 5-fold CV,
        7. evaluate train/val/test per representation, 8. select by
        validation MCC, 9. TreeSHAP on the final model, 10. save everything.
    """
    np.random.seed(seed)

    # Step 1: curation BEFORE any split
    df_clean, clean_stats = clean_sequences(df, sequence_col, label_col)

    # Step 2: stratified 70 / 15 / 15
    splits = split_data(df_clean, sequence_col, label_col, seed=seed)

    # Steps 3-5: features + representations (split-before-preprocessing)
    built = build_representations(splits, seed=seed, device=device,
                                  model_name=model_name, pca_dims=pca_dims)
    reps = built["representations"]
    for rep_name in reps:
        rep = reps[rep_name]
        # Step 6: 5-fold stratified CV on TRAINING only
        rep["cv"] = cross_validate_model(
            rep["X"]["train"], rep["y_train"], param_grid=param_grid,
            cv=5, seed=seed, n_iter=n_iter)
        # Step 7: evaluate on train (diagnostic) / validation / test
        rep["train"] = evaluate_model(rep["cv"]["best_estimator"],
                                      rep["X"]["train"], rep["y_train"],
                                      set_name="train")
        rep["validation"] = evaluate_model(rep["cv"]["best_estimator"],
                                           rep["X"]["validation"],
                                           rep["y_val"], set_name="validation")
        rep["test"] = evaluate_model(rep["cv"]["best_estimator"],
                                     rep["X"]["test"], rep["y_test"],
                                     set_name="test")

    # Step 8: selection by VALIDATION MCC (never by test)
    def _val_key(name):
        m = reps[name]["validation"]
        return (m["mcc"] if not np.isnan(m["mcc"]) else -1,
                m["auc_roc"] if not np.isnan(m["auc_roc"]) else -1)
    selected = max(reps, key=_val_key)
    print(f"[select] Selected representation by validation MCC: {selected}")

    # Step 9: TreeSHAP on the FINAL model only
    final_data = reps[selected]
    feature_names_full = ([f"esm2_dim_{i}" for i in range(final_data["esm2_block_dims"])]
                          + PHYSCHEM_FEATURES)
    if selected.startswith("Full"):
        shap_names = feature_names_full
        shap_prefix = "esm2 (1280), then physchem (6)"
    else:
        shap_names = ([f"PC{i}" for i in range(final_data["esm2_block_dims"])]
                      + PHYSCHEM_FEATURES)
        shap_prefix = ("PCA components (NOT original ESM-2 dimensions), "
                       "then physchem (6)")
    shap_result = run_shap_analysis(
        final_data["cv"]["best_estimator"],
        final_data["X"]["train"], shap_names,
        set_name="train", representation_name=selected,
        output_dir=output_dir)

    # Step 10: assemble + save
    results = {
        "representations": reps,
        "splits": built["splits"],
        "sequences": built["sequences"],
        "labels": built["labels"],
        "esm2_embeddings": built["esm2_embeddings"],
        "selected_representation": selected,
        "selected_shap_prefix": shap_prefix,
        "shap_top_features": shap_result["top_features"],
        "clean_stats": clean_stats,
        "seed": seed,
        "label_col": label_col,
    }
    comparison = save_results(results, output_dir)

    # Print the required comparison table prominently
    print("\n" + "=" * 78)
    print("REQUIRED COMPARISON TABLE")
    print("=" * 78)
    print(comparison.to_string(index=False))
    print("=" * 78)
    print(f"\nSelected model: '{selected}' trained on {final_data['n_features']} "
          f"features.\nTreeSHAP feature semantics: {shap_prefix}\n")
    print("Train vs validation performance (overfitting diagnostic):")
    for name in reps:
        t = reps[name]["train"]["mcc"]; v = reps[name]["validation"]["mcc"]
        print(f"  {name:<28} train MCC={t:.3f}  val MCC={v:.3f}  gap={t-v:+.3f}")

    return results


def main(data_csv, output_dir="results", sequence_col="sequence",
         label_col="label", seed=SEED, device="auto",
         model_name=DEFAULT_ESM2_MODEL, pca_dims=PCA_DIMS,
         param_grid=None, n_iter=40):
    """CLI / notebook entry point."""
    df = load_data(data_csv, sequence_col, label_col)
    return run_pipeline(df, output_dir=output_dir, seed=seed, device=device,
                        model_name=model_name, pca_dims=pca_dims,
                        param_grid=param_grid,
                        sequence_col=sequence_col, label_col=label_col,
                        n_iter=n_iter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HDLSS AMP classification: ESM-2 embeddings + Random Forest")
    parser.add_argument("--data", required=True, help="Clean CSV (sequence,label)")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--sequence-col", default="sequence")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument("--esm2-model", default=DEFAULT_ESM2_MODEL,
                        help="HuggingFace ESM-2 model id (default 1280-dim)")
    parser.add_argument("--pca-dims", type=lambda s: [int(x) for x in s.split(",")],
                        default="10,20,30,50")
    parser.add_argument("--n-iter", type=int, default=40,
                        help="RandomizedSearchCV iterations per representation")
    args = parser.parse_args()

    main(data_csv=args.data, output_dir=args.output,
         sequence_col=args.sequence_col, label_col=args.label_col,
         seed=args.seed, device=args.device, model_name=args.esm2_model,
         pca_dims=tuple(args.pca_dims), n_iter=args.n_iter)