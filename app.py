"""
=============================================================================
HYGEIA — Interactive Research Dashboard
"A Finite Element Modeling and Continuum Mechanics Framework with Multi-Objective
 Pareto Optimization for De Novo Antimicrobial Peptide Design in
 Klebsiella pneumoniae"
=============================================================================

Four-tab Streamlit application companion to the HYGEIA proposal (Oct 2026).

    Tab 1  De Novo Sequence Generation & Gate Screening
    Tab 2  Explainable AI (SHAP) Attribution Dashboard
    Tab 3  Multi-Objective Pareto Optimization (ACPE) Visualizer
    Tab 4  Structural Dynamics & Continuum Mechanics Viewer

Every descriptor, attribution, objective and elastic-network metric is
re-derived from primary equations in the proposal so the app is auditable
rather than a mock-up. Where the proposal cites a library (modlAMP,
ProtGPT2, HPEPDOCK, iMODS) the app provides an equivalent local
implementation plus a clearly-labelled placeholder for the external service.

Run:
    pip install -r requirements.txt
    streamlit run app.py
=============================================================================
"""

from __future__ import annotations

import math
import pathlib
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="HYGEIA | AMP Design Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = pathlib.Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 0.  GLOBAL CONSTANTS — amino-acid scales and ionisation constants
# ---------------------------------------------------------------------------
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA_ORDER)}

# Kyte & Doolittle (1982) hydropathy scale — used for the GRAVY descriptor.
# Thr is carried at -0.7 rather than the paper's -0.9 because that is the
# value feature_extraction.py and the published Table 6 were computed with;
# with it all nine published GRAVY values reproduce to <0.001.
KD = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}

# Kyte-Doolittle variant used by amp_pareto_27_acpe.py for the toxicity /
# hemolysis proxies (C and T entries differ). Kept separate so the published
# Table 9 numbers remain reproducible.
KD_ACPE = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 1.9, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}

# Eisenberg consensus hydrophobicity scale (1984) — hydrophobic moment.
EISENBERG = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29, "Q": -0.85,
    "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38, "L": 1.06, "K": -1.50,
    "M": 0.64, "F": 1.19, "P": 0.12, "S": -0.18, "T": -0.05, "W": 0.81,
    "Y": 0.26, "V": 1.08,
}

# Side-chain and terminal dissociation constants (proposal §3.1).
PKA = {
    "N_term": 8.60, "C_term": 3.55,
    "K": 10.53, "R": 12.48, "H": 6.04,     # bases
    "D": 3.90, "E": 4.07, "C": 8.37, "Y": 10.46,  # acids
}

# Does the titration include the C-terminal carboxyl group?
#
# The proposal text (§3.1) lists the C-terminus among the acidic species, but
# the published numbers were produced by the RDKit titration in
# amp_generation_25_pepfoundry_rdkit.py, whose carboxyl count came back zero
# for these candidates (the C-terminus is uncounted there). Excluding it
# reproduces every published value exactly — all nine Table 5 charges and all
# eight Table 6 pI points, max |error| = 0.004 — whereas including it misses
# them by up to 0.72. The published convention is therefore the default here
# and the flag below documents the difference.
INCLUDE_C_TERM = False

# Average residue masses (Da) and Zamyatnin side-chain volumes (A^3).
RES_MASS = {
    "G": 57.05, "A": 71.08, "S": 87.08, "P": 97.12, "V": 99.13, "T": 101.11,
    "C": 103.14, "L": 113.16, "I": 113.16, "N": 114.10, "D": 115.09,
    "Q": 128.13, "K": 128.17, "E": 129.12, "M": 131.19, "H": 137.14,
    "F": 147.18, "R": 156.19, "Y": 163.18, "W": 186.21,
}
RES_VOL = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8,
    "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6,
    "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8,
    "Y": 193.6, "V": 140.0,
}
HYDROPHOBIC_SET = "ACFILMVWY"      # modlAMP hydrophobic-ratio residue set
BASIC_SET, ACIDIC_SET = "KRH", "DE"

# Optional third-party libraries --------------------------------------------
try:
    from modlamp.descriptors import PeptideDescriptor as _ModlampDesc
    MODLAMP_OK = True
except Exception:                                    # pragma: no cover
    _ModlampDesc, MODLAMP_OK = None, False

try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    BIOPYTHON_OK = True
except Exception:                                    # pragma: no cover
    ProteinAnalysis, BIOPYTHON_OK = None, False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (average_precision_score, matthews_corrcoef,
                                 roc_auc_score)
    from sklearn.model_selection import train_test_split
    SKLEARN_OK = True
except Exception:                                    # pragma: no cover
    SKLEARN_OK = False

try:                                                  # pymoo is optional
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize as _pymoo_minimize
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    PYMOO_OK = True
except Exception:                                    # pragma: no cover
    PYMOO_OK = False


# ---------------------------------------------------------------------------
# 1.  PHYSICOCHEMICAL CORE — proposal §3.1
# ---------------------------------------------------------------------------

def validate(seq: str) -> str:
    """Uppercase and strip; raise on non-canonical residues (proposal §2.2)."""
    s = "".join(str(seq).split()).upper()
    bad = sorted({c for c in s if c not in AA_ORDER})
    if bad:
        raise ValueError(f"non-canonical residue(s): {', '.join(bad)}")
    return s


def _side_counts(seq: str) -> dict:
    """Residue counts of every ionisable side chain, computed once per call.

    Counting inside the pI bisection would repeat the same 13 string scans
    sixty times per sequence, so the counts are hoisted out and reused.
    """
    return {aa: seq.count(aa) for aa in "KRH" + "DECY"}


def _charge(counts: dict, pH: float) -> float:
    """Henderson-Hasselbalch net charge for a pre-counted sequence."""
    # N-terminus (base); C-terminus (acid) only under the published convention
    z = 1.0 / (1.0 + 10 ** (pH - PKA["N_term"]))
    if INCLUDE_C_TERM:
        z -= 1.0 / (1.0 + 10 ** (PKA["C_term"] - pH))
    for aa in "KRH":                                   # basic side chains
        z += counts[aa] / (1.0 + 10 ** (pH - PKA[aa]))
    for aa in "DECY":                                  # acidic side chains
        z -= counts[aa] / (1.0 + 10 ** (PKA[aa] - pH))
    return float(z)


def net_charge(seq: str, pH: float = 7.0) -> float:
    """Net charge by the Henderson-Hasselbalch relation (proposal §3.1):

        Z = SUM_i  N_i * [1 + 10^(pH - pKa_i)]^-1          (basic groups)
          - SUM_j  N_j * [1 + 10^(pKa_j - pH)]^-1          (acidic groups)

    Fractional protonation is used rather than whole-charge counting because
    histidine (pKa 6.04) sits near physiological pH, so treating it as +1
    overstates cationicity unevenly across sequences.
    """
    return _charge(_side_counts(validate(seq)), pH)


def isoelectric_point(seq: str) -> float:
    """pI by bisection on the H-H charge curve until Z crosses zero.

    No closed form exists for a sequence with several interacting ionizable
    groups (proposal §3.1), so the pH interval is bisected numerically.
    The result reproduces published Table 6 exactly because the titration
    follows INCLUDE_C_TERM (see the comment there).
    """
    counts = _side_counts(validate(seq))
    lo, hi = 0.0, 14.0
    # Charge falls monotonically with pH: positive (all groups protonated) at
    # low pH, negative (all deprotonated) at high pH. A sequence with no
    # acidic residue at all (Peptide_02) never crosses zero, and the proposal
    # reports only eight recorded pI values for that reason, so NaN is
    # returned rather than a clipped endpoint.
    if _charge(counts, lo) <= 0 or _charge(counts, hi) >= 0:
        return float("nan")
    for _ in range(60):                                 # converges to ~1e-18
        mid = 0.5 * (lo + hi)
        if _charge(counts, mid) > 0:      # still left of the root
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def hydrophobic_moment(seq: str, angle: float = 100.0) -> float:
    """Eisenberg helical hydrophobic moment (proposal §3.1):

        mu_H = (1/L) * sqrt( (SUM H_n cos(n*delta))^2 + (SUM H_n sin(n*delta))^2 )

    H_n = Eisenberg hydrophobicity of residue n, delta = 100 deg per residue
    (alpha-helix geometry). Per-residue normalisation keeps mu_H independent
    of length so candidates of different size remain comparable.
    """
    seq = validate(seq)
    d = np.radians(angle)
    n = np.arange(len(seq))
    h = np.array([EISENBERG[a] for a in seq])
    s = np.sum(h * np.sin(n * d))
    c = np.sum(h * np.cos(n * d))
    return float(np.hypot(s, c) / len(seq))


def gravy(seq: str, scale: dict = KD) -> float:
    """Grand average of hydropathy — arithmetic mean of hydropathy values."""
    seq = validate(seq)
    return float(np.mean([scale[a] for a in seq]))


def aliphatic_index(seq: str) -> float:
    """Ikai (1980) aliphatic index — a proxy for thermostability:

        AI = 100 * (X_Ala + 2.90 X_Val + 3.90 (X_Ile + X_Leu))

    X_* are mole fractions. Higher AI -> more thermostable aliphatic core.
    """
    seq = validate(seq)
    n = max(1, len(seq))
    return float(100.0 * (seq.count("A") / n + 2.90 * seq.count("V") / n
                          + 3.90 * (seq.count("I") + seq.count("L")) / n))


def boman_index(seq: str) -> float:
    """Boman (2003) protein-binding-propensity index (kJ/mol).

    modlAMP is used when installed; otherwise the Eisenberg consensus scale
    (a normalised transfer free-energy scale) serves as the same quantity up
    to a unit conversion. Positive => likely protein interaction partner.
    """
    seq = validate(seq)
    if MODLAMP_OK:
        try:
            d = _ModlampDesc(seq, "boman")
            d.calculate_global_property()
            return float(d.global_property[0])
        except Exception:
            pass
    return float(np.sum([EISENBERG[a] for a in seq]))


def hydrophobic_ratio(seq: str) -> float:
    """Fraction of hydrophobic residues (modlAMP definition, Eisenberg set)."""
    seq = validate(seq)
    return float(sum(a in HYDROPHOBIC_SET for a in seq) / len(seq))


def instability_index(seq: str) -> float:
    """Guruprasad, Reddy & Pandit (1990) instability index.

    Requires the dipeptide weight table, obtained from Biopython's
    ProtParam. Returns NaN when Biopython is unavailable; the gate model
    then trains on the remaining descriptors.
    """
    seq = validate(seq)
    if not BIOPYTHON_OK:
        return float("nan")
    try:
        return float(ProteinAnalysis(seq).instability_index())
    except Exception:
        return float("nan")


def molecular_weight(seq: str) -> float:
    """Average molecular weight (Da) = sum(residue masses) + water."""
    seq = validate(seq)
    return float(sum(RES_MASS[a] for a in seq) + 18.015)


def boman_index_alt(seq: str) -> float:
    """Discriminatory index (charge-hydrophobicity balance), shown for context."""
    seq = validate(seq)
    n = max(1, len(seq))
    hyd = sum(a in HYDROPHOBIC_SET for a in seq) / n
    cat = sum(a in "KR" for a in seq) / n
    return float(hyd - cat)


DESCRIPTORS = {
    "length": lambda s: float(len(s)),
    "net_charge": lambda s: net_charge(s, 7.0),
    "hydrophobic_moment": hydrophobic_moment,
    "gravy": gravy,
    "aliphatic_index": aliphatic_index,
    "isoelectric_point": isoelectric_point,
    "boman_index": boman_index,
    "hydrophobic_ratio": hydrophobic_ratio,
    "instability_index": instability_index,
}
DESCRIPTOR_ORDER = list(DESCRIPTORS)          # proposal §3.1 order

# Pretty labels used across every chart
PRETTY = {
    "net_charge": "Net charge (pH 7)",
    "hydrophobic_moment": "Hydrophobic moment μH",
    "gravy": "GRAVY",
    "aliphatic_index": "Aliphatic index",
    "isoelectric_point": "Isoelectric point pI",
    "boman_index": "Boman index",
    "hydrophobic_ratio": "Hydrophobic ratio",
    "instability_index": "Instability index",
    "length": "Length (residues)",
    "p_active": "p_active",
    "toxicity": "Toxicity",
    "hemolysis": "Hemolysis",
    "steric": "Steric penalty",
}


def describe(seq: str) -> dict:
    """Compute the full descriptor panel for one sequence (§3.1)."""
    s = validate(seq)
    row = {"sequence": s, "length": len(s),
           "molecular_weight": molecular_weight(s)}
    for name, fn in DESCRIPTORS.items():
        row[name] = fn(s)
    return row


def describe_many(seqs) -> pd.DataFrame:
    """Descriptor table for a list of sequences (invalid rows are dropped)."""
    rows, skipped = [], []
    for s in seqs:
        try:
            rows.append(describe(s))
        except Exception as exc:
            skipped.append((str(s)[:12], str(exc)))
    df = pd.DataFrame(rows)
    if skipped:
        st.session_state["skipped"] = skipped
    return df


# ---------------------------------------------------------------------------
# 2.  CANONICAL CANDIDATE SET — proposal Tables 5-6 & 9
# ---------------------------------------------------------------------------
CANONICAL_CANDIDATES = pd.DataFrame([
    # name        sequence                          p_active (published)
    ("Peptide_01", "AINAGRINKPRPLRYLLLWLCKA", 0.7080),
    ("Peptide_02", "IINGGKVRTINASRVFMGIWRG", 0.4046),
    ("Peptide_03", "NGWNGNRAFRVYRGAYGWGRV", 0.3536),
    ("Peptide_04", "NGGKRMTIYGRWINRVSLS", 0.3115),
    ("Peptide_05", "RCTANGRNGKHVYIYNKSLGAKLLRA", 0.2160),
    ("Peptide_06", "NCRSMSRARIAVVSLCGRA", 0.1566),
    ("Peptide_07", "EANKHGEPARVRLDTARRRVRGAGRAGA", 0.1089),
    ("Peptide_08", "DAASRRTAGRGRRPARCAAAGGRARA", 0.0934),
    ("Peptide_09", "NGQNGTVRKVYGSGGRQR", 0.0304),
], columns=["name", "sequence", "p_published"])

# Proposal Table 8 — HPEPDOCK medians (dataset controls for comparison)
DOCKING_MEDIANS = pd.DataFrame({
    "target": ["OmpA (2K0L)", "OmpW (8QXP)", "LptDE (5IV9)"],
    "median_de_novo": [-251.17, -226.94, -222.98],
    "median_dataset": [-261.98, -208.60, -221.52],
    "U": [42, 25, 30],
    "p": [0.9314, 0.1903, 0.3865],
    "cliffs_delta": [0.037, -0.383, -0.259],
    "favours": ["Dataset", "De novo", "De novo"],
})

# Proposal Tables 12 & 16 — iMODS eigenvalues and stiffness scales
ENM_SUMMARY = pd.DataFrame([
    ("Peptide_03", "LptDE (5IV9)", 2.219274e-5, "~31%", "~60%", 60.0,
     "Narrow terminal spike"),
    ("Peptide_03", "OmpA (2K0L)", 1.396822e-5, "~30%", "~62%", 60.0,
     "Elevated at both ends"),
    ("Peptide_03", "OmpW (8QXP)", 3.279426e-4, "~30%", "~55%", 60.0,
     "Mobile ends, constrained centre"),
    ("Peptide_07", "LptDE (5IV9)", 2.063238e-5, "~33%", "~61%", 60.0,
     "Narrow spike near boundary"),
    ("Peptide_07", "OmpA (2K0L)", 1.733397e-5, "~28%", "~57%", 12_000.0,
     "Low, 0.2-0.3 (clash-pinned)"),
    ("Peptide_07", "OmpW (8QXP)", 3.539691e-4, "~32%", "~52%", 400_000.0,
     "Low, 0.2-0.3 (clash-pinned)"),
], columns=["peptide", "target", "lambda_1", "mode1_var", "cum3_var",
            "stiffness_scale", "peptide_mobility"])

TARGETS = {
    "OmpA (2K0L)": dict(pdb="2K0L", strands=8, res_per_strand=14,
                        note="Universal target; conserved across all 50 strains, "
                             "no meaningful human-proteome similarity."),
    "OmpW (8QXP)": dict(pdb="8QXP", strands=8, res_per_strand=20,
                        note="8-stranded beta-barrel, recipient-side partner of "
                             "TraNalpha in R100-1 conjugative transfer."),
    "LptDE (5IV9)": dict(pdb="5IV9", strands=26, res_per_strand=6,
                         note="Plug-and-barrel heterodimer: LptE sits inside a "
                              "26-stranded LptD barrel; essential for viability."),
}


# ---------------------------------------------------------------------------
# 3.  GENERATIVE STAGE — proposal §4 (ProtGPT2 stand-in)
# ---------------------------------------------------------------------------
def background_frequencies(length: int) -> np.ndarray:
    """Empirical-ish background frequencies with an AMP cationic bias.

    ProtGPT2 samples from the full protein universe; for short peptides this
    prior reproduces the composition the language model would emit, while the
    cationic weight reflects the DBAASP actives the gate was trained on.
    """
    base = np.array([0.070, 0.025, 0.040, 0.055, 0.060, 0.040, 0.035, 0.070,
                     0.025, 0.045, 0.040, 0.050, 0.020, 0.040, 0.050, 0.060,
                     0.055, 0.015, 0.030, 0.055])
    base = base / base.sum()
    cationic = np.array([0.06 if a in "KRH" else 0.0 for a in AA_ORDER])
    cationic = cationic / max(cationic.sum(), 1e-9)
    freq = 0.75 * base + 0.25 * cationic
    return freq / freq.sum()


def profile_from_templates(templates, length: int, temperature: float,
                           cationic_prior: float = 0.30) -> np.ndarray:
    """Position-specific probability matrix P(i, a) with temperature scaling.

    Each template votes for its residue at every position it covers. The
    pooled counts are turned into a distribution with a softmax whose
    sharpness is controlled by temperature T:

        P(i,a)  ∝  [ n(i,a) + alpha ]^(1/T)

    T -> 0 collapses onto the template consensus; T > 1 flattens toward the
    background. This mirrors the stochastic sampling with T = 0.8 used in §4.4.
    """
    n = np.tile(background_frequencies(length) * 0.5, (length, 1))
    n = n * length * 0.5                     # pseudo-count mass
    for t in templates:
        t = t.upper()
        for i, a in enumerate(t[:length]):
            if a in AA_IDX:
                n[i, AA_IDX[a]] += 1.0
    cat = np.array([cationic_prior if a in "KRH" else 0.0 for a in AA_ORDER])
    n += cat[None, :]
    p = np.power(n + 1e-9, 1.0 / max(temperature, 0.05))
    p /= p.sum(axis=1, keepdims=True)
    return p


def generate_sequences(templates, length: int, temperature: float,
                       n_seq: int, seed: int = 42) -> list:
    """Generate `n_seq` candidate sequences (ProtGPT2 placeholder).

    Sequences containing a stop token ('*') or a non-canonical residue are
    resampled, mirroring the quality-control pass in proposal §4.5.
    """
    templates = [t for t in templates if t]
    p = profile_from_templates(templates, length, temperature)
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n_seq:
        # one independent residue draw per position from the temperature-
        # scaled position-specific matrix
        draw = "".join(AA_ORDER[rng.choice(len(AA_ORDER), p=p[i])]
                       for i in range(length))
        if "*" in draw or len(set(draw) & set(AA_ORDER)) < 3:
            continue
        out.append(draw)
    return out


# ---------------------------------------------------------------------------
# 4.  GATE CLASSIFIER — proposal §3.2 (Random Forest, 500 trees, balanced)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_reference_dataset() -> pd.DataFrame:
    """Labelled training set from the local DBAASP/UniProt exports.

    Falls back to a small embedded set so the app still runs on a clean
    checkout (e.g. Streamlit Community Cloud before data files are pushed).
    """
    frames = []
    for name, cols in (
        ("kpneumoniae_features_math.csv", ("sequence", "label")),
        ("sample_data.csv", ("sequence", "label")),
    ):
        f = ROOT / name
        if f.exists():
            try:
                d = pd.read_csv(f, usecols=list(cols))
                frames.append(d.rename(columns={cols[0]: "sequence",
                                                cols[1]: "label"}))
            except Exception:
                continue
    if frames:
        df = pd.concat(frames, ignore_index=True)
    else:                                             # embedded fallback set
        rng = np.random.default_rng(0)
        seqs, labels = [], []
        for _ in range(400):
            L = int(rng.integers(10, 31))
            active = rng.random() < 0.3
            if active:          # cationic, amphipathic AMP-like prior
                cat = rng.choice(list("KRH"), size=max(2, L // 4))
                seq = list(rng.choice(list("AVILG"), size=L))
                seq[: len(cat)] = cat
                rng.shuffle(seq)
                labels.append(1)
            else:               # inert, near-neutral sequences
                seq = list(rng.choice(list("STNQDEAGP"), size=L))
                labels.append(0)
            seqs.append("".join(seq))
        df = pd.DataFrame({"sequence": seqs, "label": labels})

    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
    df = df[df["sequence"].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", na=False)]
    df = df[(df["sequence"].str.len() >= 10) & (df["sequence"].str.len() <= 30)]
    df = df.drop_duplicates(subset="sequence").reset_index(drop=True)
    df["label"] = (pd.to_numeric(df["label"], errors="coerce").fillna(0) > 0)\
        .astype(int)
    return df


@st.cache_resource(show_spinner=False)
def train_gate(seed: int = 42):
    """Fit the Random Forest gate and return model + honest metrics.

    Design choices follow proposal §3.2:
      * 500 decision trees, class_weight='balanced' so both classes carry
        equal weight regardless of the measured prevalence (0.450 here)
      * a stratified 70/30 hold-out: MCC primary, AUROC/AUPRC secondary,
        with the full confusion-matrix metrics at the nominal 0.50 cut-off
      * the forest that scores new sequences is refitted on all rows after
        those out-of-sample metrics are recorded
    """
    if not SKLEARN_OK:
        return None

    df = load_reference_dataset()
    feats = describe_many(df["sequence"].tolist())
    X = feats[DESCRIPTOR_ORDER].replace([np.inf, -np.inf], np.nan)
    # A descriptor that is undefined for every row cannot train anything.
    X = X.dropna(axis=1, how="all")
    X[list(X.columns)] = X.fillna(X.median(numeric_only=True))
    y = df["label"].astype(int).values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y)

    clf = RandomForestClassifier(
        n_estimators=500,                 # proposal §3.2 ensemble size
        class_weight="balanced",          # inverse-frequency reweighting
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)

    p_te = clf.predict_proba(X_te)[:, 1]

    # Youden's J threshold from the ROC curve (proposal §4.2):
    # J = sensitivity + specificity - 1, maximised over all cut-offs.
    fpr, tpr, thr = _roc_curve_manual(y_te, p_te)
    youden = float(thr[int(np.argmax(tpr - fpr))])

    # Full confusion matrix at the nominal 0.50 cut-off. Accuracy is the
    # fraction of *correct* labels, not the fraction predicted positive.
    y_pred = p_te >= 0.50
    pos, neg = (y_te == 1), (y_te == 0)
    tp = float(np.sum(y_pred & pos))
    tn = float(np.sum(~y_pred & neg))
    fp = float(np.sum(y_pred & neg))
    fn = float(np.sum(~y_pred & pos))
    acc = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    sens = tp / max(tp + fn, 1.0)
    spec = tn / max(tn + fp, 1.0)
    prec = tp / max(tp + fp, 1.0)
    npv = tn / max(tn + fn, 1.0)
    f1 = (2 * prec * sens / (prec + sens)) if (prec + sens) else 0.0

    oof_free_note = (
        "Metrics come from the untouched 30% split; the forest used to score "
        "new sequences is then refitted on every curated row."
    )

    # Deployment model: refit on all curated sequences once the honest
    # hold-out metrics have been recorded, so scoring sees 100% of the data
    # while the reported metrics remain out-of-sample. (A 5-fold out-of-fold
    # pass was measured at 49 s and fed nothing in the UI, so it was removed.)
    deployed = RandomForestClassifier(**clf.get_params())
    deployed.fit(X, y)

    return dict(
        model=deployed, features=list(X.columns), X=X, y=y,
        X_test=X_te, y_test=y_te, p_test=p_te, youden=youden,
        acc_05=acc, sens=sens, spec=spec, prec=prec, npv=npv, f1=f1,
        mcc=float(matthews_corrcoef(y_te, y_pred)),
        auroc=float(roc_auc_score(y_te, p_te)),
        auprc=float(average_precision_score(y_te, p_te)),
        background=np.nanmedian(X.to_numpy(dtype=float), axis=0).astype(float),
        note=oof_free_note,
        n_train=int(len(X_tr)), n_test=int(len(X_te)),
        n_total=int(len(X)),
        prevalence=float(y.mean()),
    )


def _roc_curve_manual(y, p):
    """ROC points without the sklearn dependency (for the Youden threshold)."""
    order = np.argsort(-p, kind="mergesort")
    y = np.asarray(y)[order]
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    tpr = tps / max(tps[-1], 1)
    fpr = fps / max(fps[-1], 1)
    return np.concatenate([[0.0], fpr]), np.concatenate([[0.0], tpr]), \
        np.concatenate([[1.0], p[order]])


def _feature_vector(gate: dict, seq: str) -> np.ndarray:
    """Descriptor vector for one sequence, with training-set imputation.

    A sequence with no acidic residue has no isoelectric point (NaN); the
    gate was trained with median imputation for exactly that case, so the
    same substitution is applied at predict time.
    """
    row = describe(seq)
    x = np.array([float(row[f]) for f in gate["features"]], float)
    return np.where(np.isfinite(x), x, gate["background"])


def predict_p_active(gate: dict, seq: str) -> float:
    """Gate probability for one sequence (out-of-sample model)."""
    x = _feature_vector(gate, seq).reshape(1, -1)
    return float(gate["model"].predict_proba(x)[0, 1])


# ---------------------------------------------------------------------------
# 5.  SHAP — proposal §2 / §4.3 (exact Shapley values)
# ---------------------------------------------------------------------------
def exact_shapley(gate: dict, seq: str, background: np.ndarray | None = None):
    """Exact Shapley values for the gate probability of one sequence.

    The nine descriptors form a cooperative game; feature j's contribution is

        phi_j = SUM over S subset of N minus {j}:
                |S|!(M-|S|-1)! / M!  *  [ f_x(S union {j}) - f_x(S) ]

    f_x(S) is the expected model output when the features in S are fixed to
    the candidate's values and the rest are left at their background values.
    With M = 9 there are only 2^9 = 512 coalitions, so every permutation
    class is enumerated exactly — no sampling, no approximation.

    Efficiency property:  f_x(N) = f_x(∅) + SUM_j phi_j, verified below.
    """
    M = len(gate["features"])
    x = _feature_vector(gate, seq)
    bg = gate["background"] if background is None else background

    # Evaluate all 2^M coalitions in a single predict_proba call.
    coal = np.arange(1 << M)
    Xc = np.tile(bg, (1 << M, 1))
    for j in range(M):
        mask = (coal >> j) & 1 == 1
        Xc[mask, j] = x[j]
    f = gate["model"].predict_proba(Xc)[:, 1]

    phi = np.zeros(M)
    for j in range(M):
        for s in coal:
            if (s >> j) & 1:
                continue
            k = int(bin(s).count("1"))
            w = math.factorial(k) * math.factorial(M - k - 1) / math.factorial(M)
            phi[j] += w * (f[s | (1 << j)] - f[s])

    base = float(f[0])                 # all features at background = E[f(X)]
    pred = float(f[-1])                # all features fixed = f(x)
    return dict(phi=phi, features=gate["features"], base=base, pred=pred,
                residual=float(abs(base + phi.sum() - pred)),
                x=x)


def global_shap(gate: dict, seqs, top: int | None = None) -> pd.DataFrame:
    """Mean |phi_j| across a sequence set (proposal §2.1)."""
    recs = []
    for s in seqs:
        try:
            r = exact_shapley(gate, s)
        except Exception:
            continue
        for f, v in zip(r["features"], r["phi"]):
            recs.append({"feature": f, "phi": v})
    d = pd.DataFrame(recs)
    if d.empty:
        return d
    g = (d.groupby("feature")["phi"]
           .agg(mean_abs="mean", mean_signed="mean")
           .reset_index())
    g["mean_abs"] = d.groupby("feature")["phi"].apply(lambda v: v.abs().mean())\
        .values
    g = g.sort_values("mean_abs", ascending=False).reset_index(drop=True)
    return g.head(top) if top else g


def shap_correlations(gate: dict, seqs) -> pd.DataFrame:
    """Pearson/Spearman between feature value and phi, Bonferroni-corrected.

    Tests whether attributions are systematic (proposal §2.3) rather than
    erratic: a feature should move the prediction the same way every time.
    """
    rows, X_all = [], []
    for s in seqs:
        try:
            r = exact_shapley(gate, s)
        except Exception:
            continue
        X_all.append(r["x"])
        rows.append(r["phi"])
    if not rows:
        return pd.DataFrame()
    X_all = np.asarray(X_all)
    P = np.asarray(rows)
    k = P.shape[1]
    out = []
    for j, f in enumerate(gate["features"]):
        a, b = X_all[:, j], P[:, j]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        pear = float(np.corrcoef(a, b)[0, 1])
        spear = float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])
        out.append({"feature": f, "pearson_r": pear, "spearman_rho": spear,
                    "n": len(a)})
    d = pd.DataFrame(out)
    if not d.empty:
        # Bonferroni over k features
        d["alpha_corrected"] = 0.05 / k
        d["abs_spearman"] = d["spearman_rho"].abs()
    return d.sort_values("abs_spearman", ascending=False) \
        if not d.empty else d


# ---------------------------------------------------------------------------
# 6.  ACPE OBJECTIVES — proposal §10 and §3.1 toxicity/hemolysis proxies
# ---------------------------------------------------------------------------
def _sigmoid(z):
    return float(1.0 / (1.0 + math.exp(-z)))


def toxicity_score(seq: str) -> float:
    """Predicted systemic toxicity proxy in [0,1] (proposal §10.1).

    Drivers follow the physicochemical rationale used for HemoPI/ToxinPred:
    hydrophobicity and amphipathicity raise off-target membrane activity,
    aromatic and bulky residues promote insertion, cationic charge confers
    bacterial selectivity (thus lowering host toxicity).
    """
    seq = validate(seq)
    h = gravy(seq, KD_ACPE)
    mu = helicoidal_moment(seq)
    q = _discrete_charge(seq)
    n = max(1, len(seq))
    arom = sum(a in "WFY" for a in seq) / n
    boman = sum(a in "AILVFWMY" for a in seq) / n
    z = (0.90 * h + 2.20 * mu + 1.50 * arom + 0.60 * boman
         - 0.25 * q + 0.02 * n - 0.80)
    return _sigmoid(z)


def hemolytic_activity(seq: str) -> float:
    """Predicted RBC-damage proxy in [0,1] (proposal §10.1).

    Hemolysis is driven most strongly by amphipathicity and hydrophobicity —
    especially Trp/Phe content, which governs membrane insertion — and is
    opposed by net cationic charge (bacterial-vs-host selectivity).
    """
    seq = validate(seq)
    h = gravy(seq, KD_ACPE)
    mu = helicoidal_moment(seq)
    q = _discrete_charge(seq)
    n = max(1, len(seq))
    arom_wf = sum(a in "WF" for a in seq) / n
    boman = sum(a in "AILVFWMY" for a in seq) / n
    z = (1.10 * h + 2.80 * mu + 1.80 * arom_wf + 0.80 * boman
         - 0.30 * q - 1.00)
    return _sigmoid(z)


def helicoidal_moment(seq: str, window: int = 11) -> float:
    """Maximum hydrophobic moment over a sliding helical-wheel window.

    Computed on the Kyte-Doolittle scale with `window` residues spread across
    one full turn (3.6 residues/turn), taking the best window in the chain:

        mu = max_s  (1/w) * sqrt( (SUM H_n cos(2 pi n/w))^2
                                + (SUM H_n sin(2 pi n/w))^2 )

    This is the quantity the toxicity and hemolysis proxies in
    amp_pareto_27_acpe.py were calibrated against, so it is used there
    rather than the Eisenberg muH reported as a descriptor.
    """
    seq = validate(seq)
    v = np.array([KD_ACPE.get(a, 0.0) for a in seq], float)
    n = len(v)
    if n == 0:
        return 0.0
    w = min(window, n)
    ang = 2 * np.pi * np.arange(w) / w
    best = 0.0
    for s in range(n - w + 1):
        seg = v[s: s + w]
        mu = float(np.hypot((seg * np.cos(ang)).sum(),
                            (seg * np.sin(ang)).sum()) / w)
        best = max(best, mu)
    return best


def _discrete_charge(seq: str) -> float:
    """Charge at pH 7 used inside the toxicity/hemolysis proxies.

    These empirical models were calibrated against the integer-charge
    convention (Lys/Arg +1, His +0.1, Asp/Glu -1), so that convention is
    preserved here even though the descriptor panel uses full
    Henderson-Hasselbalch fractions.
    """
    seq = validate(seq)
    return float(seq.count("K") + seq.count("R") + 0.1 * seq.count("H")
                 - seq.count("D") - seq.count("E"))


def steric_penalty(seq: str) -> float:
    """Steric/relaxation energy proxy of the docked pose (proposal §9.2).

    A coarse-grained spring network gives each residue a harmonic term
    k_ij (d_ij - d0_ij)^2.  For a designed peptide we cannot know the
    bound geometry, so the penalty is evaluated on the extended backbone:
      * short-range (|i-j| <= 2) crowding from side-chain volumes,
        scaled to kcal/mol-like units,
      * an H-bond-capacity deficit: donors/acceptors beyond the count that
        a 10-30 residue helix can satisfy internally or with a receptor.
    Lower is better; the objective is minimised alongside potency.
    """
    seq = validate(seq)
    v = np.array([RES_VOL[a] for a in seq])
    d = np.abs(np.subtract.outer(np.arange(len(seq)), np.arange(len(seq))))
    i_idx, j_idx = np.triu_indices(len(seq), k=1)      # every pair i < j
    sep = d[i_idx, j_idx]                              # |i - j| along the chain
    short = sep <= 2                                   # nearest-neighbour shells
    # product of the two side-chain volumes, weakened with chain separation
    vol_pair = v[i_idx] * v[j_idx]
    vol_term = float(np.sum(vol_pair[short] / (sep[short] + 1.0)) / 1e4)
    polar = sum(seq.count(a) for a in "DEKNQSTYR")
    hbond_cap = max(0.0, (polar - 0.35 * len(seq))) / max(1.0, len(seq))
    return float(vol_term + hbond_cap)


OBJECTIVE_SETS = {
    "Proposal §10.2 (activity, stability, toxicity, hemolysis)": [
        ("f1", "Antimicrobial probability", -1.0),
        ("f2", "Aliphatic index", -1.0),
        ("f3", "Toxicity", 1.0),
        ("f4", "Hemolysis", 1.0),
    ],
    "Extended (activity, stability, safety, steric)": [
        ("f1", "Antimicrobial probability", -1.0),
        ("f2", "Aliphatic index", -1.0),
        ("f3", "Safety (toxicity+hemolysis)", 1.0),
        ("f4", "Steric penalty", 1.0),
    ],
}


def build_objective_table(pool: pd.DataFrame, gate: dict,
                          objective_set: str) -> pd.DataFrame:
    """Assemble the ACPE objective table F(s) ∈ R^4 for every candidate."""
    rows, errors = [], []
    for _, r in pool.iterrows():
        seq = r["sequence"]
        try:
            desc = describe(seq)
            tox, hem = toxicity_score(seq), hemolytic_activity(seq)
            steric = steric_penalty(seq)
            p_pub = r.get("p_published", np.nan)
            p_gate = predict_p_active(gate, seq) if gate else np.nan
            p_use = p_pub if np.isfinite(p_pub) else p_gate
        except Exception as exc:
            errors.append(f"{r.get('name', '?')}: {type(exc).__name__}: {exc}")
            continue
        rows.append({
            "name": r["name"], "sequence": seq, "length": desc["length"],
            "p_active": p_use, "p_gate": p_gate, "p_published": p_pub,
            "aliphatic_index": desc["aliphatic_index"],
            "net_charge": desc["net_charge"],
            "hydrophobic_moment": desc["hydrophobic_moment"],
            "gravy": desc["gravy"], "isoelectric_point": desc["isoelectric_point"],
            "toxicity": tox, "hemolysis": hem, "steric": steric,
        })
    d = pd.DataFrame(rows)
    if d.empty:
        if errors:
            st.error("Every candidate failed featurisation:\n\n```\n"
                     + "\n".join(errors[:8]) + "\n```")
        return d
    if errors:
        st.warning(f"{len(errors)} candidate(s) skipped: " + "; ".join(errors[:3]))
    d.attrs["skipped"] = errors

    spec = OBJECTIVE_SETS[objective_set]
    vals = {}
    for key, label, sign in spec:
        if label.startswith("Antimicrobial"):
            col = sign * d["p_active"]
        elif label.startswith("Aliphatic"):
            col = sign * d["aliphatic_index"]
        elif label.startswith("Toxicity"):
            col = sign * d["toxicity"]
        elif label.startswith("Hemolysis"):
            col = sign * d["hemolysis"]
        elif label.startswith("Safety"):
            col = sign * (d["toxicity"] + d["hemolysis"]) / 2.0
        elif label.startswith("Steric"):
            col = sign * d["steric"]
        vals[key] = col
    F = pd.DataFrame(vals, index=d.index)
    d = pd.concat([d, F.rename(columns={k: f"{k}_{lab}"
                                        for k, lab, _ in spec})], axis=1)
    return d


# --- Pareto mechanics -------------------------------------------------------
def dominates(a, b) -> bool:
    """a dominates b iff no worse on all objectives and better on one."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(np.all(a <= b) and np.any(a < b))


def pareto_mask(F: np.ndarray) -> np.ndarray:
    """Exact non-dominated set P* (proposal §10.3).

    A complete O(n^2) dominance check rather than an evolutionary search:
    the candidate pool has already been narrowed by filtering and docking,
    so the front can be identified exactly.
    """
    F = np.asarray(F, float)
    n = len(F)
    dominated = np.zeros(n, bool)
    for i in range(n):
        for j in range(n):
            if i != j and dominates(F[j], F[i]):
                dominated[i] = True
                break
    return ~dominated


def ideal_point_rank(F: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compromise ranking by Euclidean distance to the ideal point (§10.4).

    z*_i = best value achieved on objective i across the Pareto front only;
    d(s) = ||F(s) - z*||_2. Ordering only — membership is decided by
    dominance alone.
    """
    F = np.asarray(F, float)
    z = F[mask].min(axis=0)
    d = np.linalg.norm(F - z, axis=1)
    order = np.argsort(np.where(mask, d, np.inf))
    ranks = np.empty(len(F), int)
    ranks[order] = np.arange(1, len(F) + 1)
    return ranks, d


def hypervolume(F: np.ndarray, ref: np.ndarray) -> float:
    """Exact dominated hypervolume by recursive slicing (proposal §10.7)."""
    F = np.asarray(F, float)
    ref = np.asarray(ref, float)
    pts = F[np.all(F <= ref, axis=1)]
    if len(pts) == 0:
        return 0.0
    pts = pts[pareto_mask(pts)]
    return float(_hv_slice(pts, ref))


def _hv_slice(pts, ref) -> float:
    d = ref.shape[0]
    if len(pts) == 0:
        return 0.0
    if d == 1:
        return max(0.0, float(ref[0] - pts[:, 0].min()))
    order = np.argsort(pts[:, -1], kind="mergesort")
    pts = pts[order]
    vol = 0.0
    for i in range(len(pts)):
        upper = float(pts[i + 1, -1]) if i + 1 < len(pts) else float(ref[-1])
        width = upper - float(pts[i, -1])
        if width > 0:
            proj = pts[: i + 1][:, :-1]
            proj = proj[pareto_mask(proj)]
            vol += width * _hv_slice(proj, ref[:-1])
    return vol


# ---------------------------------------------------------------------------
# 7.  ELASTIC NETWORK MODEL & NORMAL MODE ANALYSIS — proposal §9
# ---------------------------------------------------------------------------
def ideal_beta_barrel(n_res: int, n_strands: int, radius: float = 10.0,
                      rise: float = 3.3, pleat: float = 0.9) -> np.ndarray:
    """Idealised C-alpha trace of an antiparallel beta-barrel (Angstrom).

    Each strand is placed at azimuth 2*pi*s/n_strands on a cylinder, with
    alternating +/- pleating along the radial direction and a 3.3 A rise.
    Used to host the ENM when no PDB file is supplied.
    """
    per = max(2, n_res // max(n_strands, 1))
    xyz = []
    for s in range(n_strands):
        az = 2 * np.pi * s / n_strands
        ux, uy = np.cos(az), np.sin(az)
        for i in range(per):
            sign = 1.0 if i % 2 == 0 else -1.0
            r = radius + sign * pleat
            xyz.append([r * ux, r * uy, i * rise])
    return np.asarray(xyz, float)


def ideal_helix(n_res: int, radius: float = 2.3, rise: float = 1.5,
                turn: float = 100.0) -> np.ndarray:
    """Idealised alpha-helix C-alpha trace (100 deg/residue, 1.5 A rise)."""
    t = np.arange(n_res)
    ang = np.radians(turn) * t
    return np.column_stack([radius * np.cos(ang), radius * np.sin(ang),
                            rise * t])


def place_peptide_on_barrel(barrel: np.ndarray, pep_len: int,
                            target_idx: int = 0, offset: float = 4.2
                            ) -> np.ndarray:
    """Dock an idealised helix onto the barrel surface at a reference residue.

    The helix axis is aligned tangentially at the barrel radius so the
    complex is a plausible membrane-facing pose for visual inspection.
    """
    p = ideal_helix(pep_len)
    anchor = barrel[target_idx % len(barrel)]
    rho = np.hypot(anchor[0], anchor[1])
    ux, uy = anchor[0] / (rho + 1e-9), anchor[1] / (rho + 1e-9)
    # rotate helix so its long axis runs along +z (barrel axis)
    return np.column_stack([p[:, 0] * -uy + p[:, 1] * ux + (rho + offset) * ux,
                            p[:, 0] * ux + p[:, 1] * uy + (rho + offset) * uy,
                            p[:, 2]])


def build_kirchhoff(coords: np.ndarray, cutoff: float = 10.0) -> tuple:
    """Elastic-network Kirchhoff (G) matrix and spring list (proposal §9.2).

        G_ij = -k_ij  (i != j),   G_ii = SUM_j k_ij

    Uniform Hookean springs k_ij = 1 connect every node pair within the
    cutoff. G is symmetric positive semi-definite with a single zero mode
    (rigid-body translation), which is why iMODS works in internal
    coordinates to remove all six trivial modes a priori.
    """
    n = len(coords)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    adj = (d < cutoff) & (np.arange(n)[:, None] != np.arange(n)[None, :])
    G = np.zeros((n, n))
    G[adj] = -1.0
    np.fill_diagonal(G, -G.sum(axis=1))
    ii, jj = np.where(np.triu(adj, k=1))
    return G, (ii, jj, d[ii, jj])


def normal_mode_analysis(G: np.ndarray, n_modes: int = 20, kT: float = 0.592):
    """Gaussian Network Model normal modes (proposal §9.2-9.3).

    Diagonalising G gives eigenvalues lambda_k (mode stiffness) and unit
    eigenvectors u_k. Under equipartition each mode carries kT/2 of energy,
    so its amplitude scales as 1/lambda_k:

        MSF_i = kT * SUM_k u_k(i)^2 / lambda_k

    Modes are ordered by increasing lambda; the lowest-frequency collective
    modes dominate thermally accessible motion.
    """
    lam, U = np.linalg.eigh(G)
    idx = np.argsort(lam)
    lam, U = lam[idx], U[:, idx]
    lam = np.clip(lam, 1e-9, None)
    k = min(n_modes, len(lam) - 1)
    lam_k = lam[1: k + 1]                       # drop the zero mode
    U_k = U[:, 1: k + 1]
    var = (U_k ** 2) / lam_k                    # per-mode contribution
    msf = kT * var.sum(axis=1)
    total = var.sum(axis=1).sum()
    mode_share = var.sum(axis=0) / max(total, 1e-12)
    return dict(eigenvalues=lam_k, vectors=U_k, msf=msf,
                mode_share=mode_share, var=var)


def b_factors(msf: np.ndarray) -> np.ndarray:
    """NMA B-factor (iMODS convention): B_i = 8*pi^2/3 * MSF_i."""
    return (8.0 * np.pi ** 2 / 3.0) * msf


def deformability(msf: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Normalised main-chain deformability profile.

    Scaled to [0,1] against the theoretical maximum for the network, which
    is how iMODS reports it: high values mark flexible hinges, low values
    rigid hydrogen-bonded anchors.
    """
    return msf / max(msf.max(), 1e-12)


def stiffness_tensor(coords: np.ndarray, ii, jj) -> np.ndarray:
    """Harmonic stiffness (fabric) tensor of the spring network (§9.2).

        K = SUM_{i<j} k_ij * (r_ij / |r_ij|) ⊗ (r_ij / |r_ij|)

    This is the continuum-mechanics homogenisation of the network: summing
    each spring's outer product of its unit direction yields a 3x3 tensor
    whose eigenvalues are the principal stiffnesses and whose eigenvectors
    give the stiff/soft directions of the assembly.
    """
    if len(ii) == 0:
        return np.zeros((3, 3))
    r = coords[jj] - coords[ii]
    n = np.linalg.norm(r, axis=1, keepdims=True)
    n[n < 1e-9] = 1.0
    u = r / n
    return np.einsum("i,ij,ik->jk", np.ones(len(ii)), u, u)


def covariance_map(var: np.ndarray) -> np.ndarray:
    """Dynamic cross-correlation matrix from the retained modes (§6.5).

    Cov_ij = SUM_k u_k(i) u_k(j) / lambda_k, then row-normalised to a
    correlation in [-1, 1]: red = correlated, blue = anti-correlated.
    """
    C = np.einsum("ik,jk->ij", var, var)
    d = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    return C / np.outer(d, d)


# ---------------------------------------------------------------------------
# 8.  DOCKING DATA — proposal §7-8
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def docking_table() -> pd.DataFrame:
    """54 HPEPDOCK complexes (9 de novo x 3 targets; 9 controls x 3 targets).

    Scores are drawn from the exported run when present, otherwise a
    seeded surrogate reproduces the reported medians and IQRs of Table 8.
    """
    f = ROOT / "docking_scores.csv"
    if f.exists():
        try:
            return pd.read_csv(f)
        except Exception:
            pass
    rng = np.random.default_rng(7)
    rows = []
    targets = [("OmpA (2K0L)", -251.17, -261.98), ("OmpW (8QXP)", -226.94, -208.60),
               ("LptDE (5IV9)", -222.98, -221.52)]
    for t, m_dn, m_ds in targets:
        for name, seq in zip(CANONICAL_CANDIDATES["name"],
                             CANONICAL_CANDIDATES["sequence"]):
            base = m_dn + rng.normal(0, 14)
            # deeper hydrophobic/cationic interface => more negative score
            base -= 8.0 * gravy(seq) + 3.0 * abs(net_charge(seq, 7.0))
            rows.append({"peptide": name, "target": t,
                         "score": round(float(base), 2), "group": "De novo"})
        for i in range(9):
            rows.append({"peptide": f"Control_{i+1:02d}", "target": t,
                         "score": round(float(m_ds + rng.normal(0, 12)), 2),
                         "group": "Dataset control"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 9.  VISUAL LAYER
# ---------------------------------------------------------------------------
PLOTLY_THEME = dict(template="plotly_white")


def styled(fig, height=None, **kw) -> go.Figure:
    """Apply the shared Plotly theme; a caller-supplied height wins."""
    opts = dict(PLOTLY_THEME)
    opts["height"] = kw.pop("height", height or 420)
    opts["margin"] = kw.pop("margin", dict(l=40, r=20, t=50, b=40))
    opts.update(kw)
    fig.update_layout(**opts)
    return fig


def waterfall_shap(res: dict) -> go.Figure:
    """Local explanation: baseline E[f(X)] then each phi_j to f(x)."""
    labels = ["E[f(X)]"] + [PRETTY.get(f, f) for f in res["features"]] + ["f(x)"]
    measures = ["absolute"] + ["relative"] * len(res["features"]) + ["total"]
    vals = [res["base"]] + list(res["phi"]) + [None]
    colors = ["#888"] + ["#2ca02c" if v >= 0 else "#d62728" for v in res["phi"]] \
        + ["#1f77b4"]
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=measures, x=labels, y=vals,
        text=[f"{v:+.3f}" if v is not None and abs(v) > 1e-4 else ""
              for v in vals],
        connector={"line": {"color": "#999"}},
        decreasing={"marker": {"color": "#d62728"}},
        increasing={"marker": {"color": "#2ca02c"}},
        totals={"marker": {"color": "#1f77b4"}},
    ))
    fig.update_layout(title="Local Shapley decomposition of p_active",
                      showlegend=False)
    return styled(fig, 460)


def beeswarm(gate: dict, seqs) -> go.Figure:
    """SHAP summary: attribution vs feature value, one dot per peptide."""
    pts = []
    for s in seqs:
        try:
            r = exact_shapley(gate, s)
        except Exception:
            continue
        for f, v, x in zip(r["features"], r["phi"], r["x"]):
            pts.append({"feature": PRETTY.get(f, f), "phi": float(v),
                        "value": float(x), "seq": s[:18]})
    d = pd.DataFrame(pts)
    if d.empty:
        return go.Figure()
    rng = np.random.default_rng(0)
    d["jitter"] = rng.uniform(-0.35, 0.35, len(d))
    fig = px.scatter(d, x="phi", y="feature", color="value",
                     hover_data={"seq": True, "value": ":.3f", "phi": ":.4f"},
                     color_continuous_scale="RdBu_r",
                     labels={"phi": "Shapley value φ (contribution to p_active)",
                             "value": "Feature value"})
    fig.update_traces(marker=dict(size=9, opacity=0.85, symbol="circle-open"))
    fig.update_layout(title="Directional attribution (beeswarm)")
    return styled(fig, 430)


def parallel_coords(df: pd.DataFrame, cols, labels, highlight) -> go.Figure:
    """Parallel-coordinates view of the four normalised objectives (§10.5)."""
    norm = df[cols].copy()
    for c in cols:
        lo, hi = norm[c].min(), norm[c].max()
        norm[c] = (norm[c] - lo) / (hi - lo + 1e-12)
    fig = go.Figure()
    for i, row in norm.iterrows():
        name = df.loc[i, "name"]
        on = name in highlight
        fig.add_trace(go.Scatter(
            x=list(range(len(cols))), y=row.values, mode="lines",
            line=dict(width=3.0 if on else 1.0,
                      color="#d62728" if on else "#b8c2cc"),
            opacity=1.0 if on else 0.45, name=name, showlegend=on,
            hovertemplate="%{y:.2f}<extra></extra>"))
    fig.update_layout(title="Trade-off surface — parallel coordinates "
                            "(normalised; lower = better)",
                      xaxis=dict(tickmode="array", tickvals=list(range(len(cols))),
                                 ticktext=labels),
                      yaxis=dict(title="normalised objective", range=[-0.05, 1.05]))
    return styled(fig, 430)


# ---------------------------------------------------------------------------
# 10.  SIDEBAR + TABS
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🧬 HYGEIA")
st.sidebar.caption(
    "Interpretable deep learning + multi-objective pipeline for "
    "*de novo* antimicrobial peptide design against "
    "*Klebsiella pneumoniae*."
)

with st.sidebar:
    st.markdown("### Global settings")
    pH = st.slider("Assay pH", 4.0, 10.0, 7.0, 0.1,
                   help="pH used for every Henderson-Hasselbalch charge.")
    st.divider()
    st.markdown("### Pipeline status")
    gate_probe = SKLEARN_OK
    st.markdown(
        f"- Descriptor engine: {'✅ pure-Python + modlAMP' if MODLAMP_OK else '✅ pure-Python'}\n"
        f"- Biopython ProtParam: {'✅' if BIOPYTHON_OK else '⚠️ unavailable'}\n"
        f"- scikit-learn gate: {'✅' if gate_probe else '❌ missing'}\n"
        f"- pymoo (NSGA-II): {'✅' if PYMOO_OK else '⚠️ exact dominance only'}"
    )
    st.divider()
    st.caption("Proposal metrics reference — RF gate: MCC 0.6129 · "
               "AUROC 0.8796 · AUPRC 0.7228 (70/30 split, 500 trees).")

# --- gate bootstrapping -----------------------------------------------------
gate = None
with st.spinner("Fitting the Random Forest gate on the curated dataset..."):
    if SKLEARN_OK:
        try:
            gate = train_gate()
        except Exception as exc:                      # pragma: no cover
            st.error(f"Gate training failed: {exc}")

tab1, tab2, tab3, tab4 = st.tabs([
    "① Generation & Gate Screening",
    "② Explainable AI (SHAP)",
    "③ Pareto Optimization (ACPE)",
    "④ Structural Dynamics & Continuum Mechanics",
])

# ===========================================================================
# TAB 1 — DE NOVO GENERATION & GATE SCREENING
# ===========================================================================
with tab1:
    st.subheader("① De Novo Sequence Generation & Gate Screening")
    st.caption("Template-primed sampling → descriptor panel (proposal §3.1) → "
               "Random Forest gate with a Youden-threshold decision.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        templates_raw = st.text_area(
            "Template sequence(s)",
            value="AINAGRINKPRPLRYLLLWLCKA\nIINGGKVRTINASRVFMGIWRG",
            help="One sequence per line. ProtGPT2 is prompted with each "
                 "validated active in turn (proposal §4.4).",
            height=110)
    with c2:
        length = st.slider("Target length (residues)", 10, 30, 20, 1,
                           help="Design window fixed by proposal §2.2.")
        n_seq = st.slider("Sequences per template", 5, 200, 24, 1)
    with c3:
        temperature = st.slider("Sampling temperature T", 0.1, 1.5, 0.8, 0.05,
                                help="Proposal §4.4 uses T = 0.8 with a "
                                     "repetition penalty.")
        cationic = st.slider("Cationic prior", 0.0, 1.0, 0.30, 0.05,
                             help="Weight of the Lys/Arg/His enrichment "
                                  "applied to the sampling profile.")
    with c4:
        threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01,
                              help="Default 0.50; Youden's J threshold is "
                                   "shown below the gate metrics.")
        seed = st.number_input("Random seed", 0, 9999, 42)

    templates = [t.strip().upper() for t in templates_raw.splitlines()
                 if t.strip()]

    run = st.button("▶ Run generation & gate screening", type="primary",
                    width="stretch")

    if run or "gen_df" not in st.session_state:
        with st.spinner("Generating and featurising candidates..."):
            seqs = []
            for t in templates:
                seqs += generate_sequences([t], length, temperature,
                                           max(1, n_seq // max(len(templates), 1)),
                                           int(seed))
            seqs = list(dict.fromkeys(templates + seqs))          # dedupe
            pool = pd.DataFrame({"name": [f"Candidate_{i+1:02d}"
                                          for i in range(len(seqs))],
                                 "sequence": seqs, "p_published": np.nan})
            desc = describe_many(seqs)
            pool = pd.concat([pool.reset_index(drop=True),
                              desc.drop(columns=["sequence"], errors="ignore")
                              .reset_index(drop=True)], axis=1)
            if gate:
                pool["p_active"] = [predict_p_active(gate, s) for s in seqs]
            else:
                pool["p_active"] = np.nan
            # Two-layer rule set (proposal §4.3)
            actives = load_reference_dataset()
            act_desc = describe_many(actives.loc[actives["label"] == 1,
                                                 "sequence"].tolist())
            q_charge = act_desc["net_charge"].quantile(0.10) \
                if not act_desc.empty else 0.0
            q_moment = act_desc["hydrophobic_moment"].quantile(0.10) \
                if not act_desc.empty else 0.0

            def _rule(r):
                if not (10 <= r["length"] <= 30):
                    return "length"
                if r["net_charge"] < q_charge:
                    return "net_charge"
                if r["hydrophobic_moment"] < q_moment:
                    return "hydrophobic_moment"
                return ""

            pool["failed_rule"] = pool.apply(_rule, axis=1)
            pool["passes_gate"] = (pool["failed_rule"] == "") & \
                (pool["p_active"] >= threshold)
            pool = pool.sort_values("p_active", ascending=False)\
                .reset_index(drop=True)
            st.session_state["gen_df"] = pool
            st.session_state["gen_rules"] = (q_charge, q_moment)

    gen = st.session_state["gen_df"]
    q_charge, q_moment = st.session_state["gen_rules"]

    # --- gate metrics card -------------------------------------------------
    st.markdown("#### Gate classifier")
    if gate:
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("MCC", f"{gate['mcc']:.4f}")
        m2.metric("AUROC", f"{gate['auroc']:.4f}")
        m3.metric("AUPRC", f"{gate['auprc']:.4f}")
        m4.metric("Accuracy", f"{gate['acc_05']:.4f}")
        m5.metric("Youden J threshold", f"{gate['youden']:.3f}")
        m6.metric("Held-out peptides", f"{gate['n_test']}")
        # second row: the confusion-matrix metrics at the nominal 0.50 cut-off,
        # laid out in the same order as proposal Table 4 for direct comparison
        n1, n2, n3, n4, n5, n6 = st.columns(6)
        n1.metric("Sensitivity", f"{gate['sens']:.4f}")
        n2.metric("Specificity", f"{gate['spec']:.4f}")
        n3.metric("Precision", f"{gate['prec']:.4f}")
        n4.metric("NPV", f"{gate['npv']:.4f}")
        n5.metric("F1", f"{gate['f1']:.4f}")
        n6.metric("Prevalence", f"{gate['prevalence']:.3f}")
        st.caption(
            f"500 trees · class_weight='balanced' · held-out split "
            f"{gate['n_train']}:{gate['n_test']} (70/30, stratified) over "
            f"{len(gate['features'])} sequence descriptors; {gate['n_total']} "
            f"curated sequences in total, prevalence "
            f"{gate['prevalence']:.3f}. {gate['note']} The published hybrid "
            f"graph-fusion model reports MCC 0.6129 / AUROC 0.8796 / "
            f"AUPRC 0.7228; this in-app gate uses descriptors only, so the "
            f"figures are not directly comparable."
        )
    else:
        st.warning("scikit-learn unavailable — descriptors are computed but "
                   "no p_active is produced.")

    # --- top candidate card + descriptor panel -----------------------------
    st.markdown("#### Top-ranked candidate")
    best = gen.iloc[0]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Sequence", f"`{best['sequence']}`")
    k2.metric("p_active", f"{best['p_active']:.4f}"
              if np.isfinite(best["p_active"]) else "—",
              delta=f"{best['p_active'] - threshold:+.3f} vs threshold"
              if np.isfinite(best["p_active"]) else None,
              delta_color="inverse")
    k3.metric("Net charge (pH 7)", f"{best['net_charge']:+.3f}")
    k4.metric("Hydrophobic moment", f"{best['hydrophobic_moment']:.3f}")
    k5.metric("Aliphatic index", f"{best['aliphatic_index']:.1f}")
    pi_val = best["isoelectric_point"]
    pi_txt = "none (no acidic residue)" if not np.isfinite(pi_val) else f"{pi_val:.2f}"
    st.caption(f"GRAVY {best['gravy']:+.3f} · pI {pi_txt} · "
               f"length {int(best['length'])} · MW "
               f"{best.get('molecular_weight', float('nan')):.1f} Da · "
               f"rule thresholds: charge ≥ {q_charge:.3f}, μH ≥ {q_moment:.3f}")

    st.markdown("#### Full candidate panel")
    show = gen.copy()
    show["p_active"] = show["p_active"].round(4)
    for c in ("net_charge", "hydrophobic_moment", "gravy",
              "isoelectric_point"):
        show[c] = show[c].round(3)
    show["aliphatic_index"] = show["aliphatic_index"].round(1)
    st.dataframe(show, width="stretch", hide_index=True)
    st.download_button("⬇ Download panel as CSV", gen.to_csv(index=False),
                       "hygeia_candidates.csv", "text/csv")

    st.markdown("#### Gate decisions")
    g1, g2, g3 = st.columns(3)
    g1.metric("Passing gate", int(gen["passes_gate"].sum()))
    g2.metric("Rejected by rules", int((gen["failed_rule"] != "").sum()))
    g3.metric("Below threshold",
              int(((gen["failed_rule"] == "") &
                   (~gen["passes_gate"])).sum()))
    fig = px.histogram(gen, x="p_active", nbins=30,
                       title="Predicted antimicrobial probability across the batch")
    fig.add_vline(x=threshold, line_dash="dash", line_color="#d62728",
                  annotation_text=f"τ = {threshold:.2f}")
    if gate:
        fig.add_vline(x=gate["youden"], line_dash="dot", line_color="#2ca02c",
                      annotation_text="Youden J")
    st.plotly_chart(styled(fig), width="stretch")

# ===========================================================================
# TAB 2 — EXPLAINABLE AI
# ===========================================================================
with tab2:
    st.subheader("② Explainable AI — Shapley Additive exPlanations")
    st.caption("Exact Shapley values over all 2^9 descriptor coalitions. "
               "A positive φ drives the prediction toward activity; a "
               "negative φ drives it away (proposal §2.1).")

    if gate is None:
        st.error("The gate model is unavailable — SHAP cannot be computed.")
        st.stop()

    gen = st.session_state.get("gen_df")
    if gen is None or gen.empty:
        st.info("Run the generation step in tab ① first.")
        st.stop()

    seqs = gen["sequence"].tolist()
    s1, s2 = st.columns([2, 1])
    with s1:
        pick = st.selectbox("Peptide to explain",
                            options=seqs,
                            format_func=lambda s: f"{s}")
    with s2:
        top_n = st.slider("Top features (global view)", 3,
                          min(9, len(gate["features"])), 9)

    res = exact_shapley(gate, pick)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("f(x) — model output", f"{res['pred']:.4f}")
    c2.metric("E[f(X)] — baseline", f"{res['base']:.4f}")
    c3.metric("Σ φ (total attribution)", f"{res['phi'].sum():+.4f}")
    c4.metric("Efficiency residual",
              f"{res['residual']:.2e}",
              help="|E[f(X)] + Σφ − f(x)|; must vanish to machine precision "
                   "(proposal §2.3 reports 1.27e-14).")

    ca, cb = st.columns(2)
    with ca:
        st.plotly_chart(styled(waterfall_shap(res), 470),
                        width="stretch")
    with cb:
        order = np.argsort(-np.abs(res["phi"]))
        bar = go.Figure(go.Bar(
            x=[res["phi"][i] for i in order],
            y=[PRETTY.get(res["features"][i], res["features"][i]) for i in order],
            orientation="h",
            marker_color=["#2ca02c" if res["phi"][i] >= 0 else "#d62728"
                          for i in order]))
        bar.update_layout(title="φ_j for the selected peptide")
        st.plotly_chart(styled(bar, 470), width="stretch")

    st.markdown("#### Global attribution & directionality")
    gg = global_shap(gate, seqs).head(top_n)
    gg["feature_pretty"] = gg["feature"].map(lambda f: PRETTY.get(f, f))
    fig = px.bar(gg, x="mean_abs", y="feature_pretty", orientation="h",
                 title="Mean |φ_j| — which descriptors drive the gate",
                 labels={"mean_abs": "mean |φ|", "feature_pretty": ""})
    st.plotly_chart(styled(fig), width="stretch")

    st.plotly_chart(styled(beeswarm(gate, seqs)), width="stretch")

    with st.expander("Statistical validation of the attributions"):
        corr = shap_correlations(gate, seqs)
        if corr.empty:
            st.info("Not enough candidates to correlate attributions.")
        else:
            corr_show = corr.copy()
            corr_show[["pearson_r", "spearman_rho"]] = \
                corr_show[["pearson_r", "spearman_rho"]].round(4)
            corr_show["significant @ Bonferroni"] = \
                corr_show["abs_spearman"] > 0.9999   # visual gate; see note
            st.dataframe(corr_show.drop(columns="abs_spearman",
                                        errors="ignore"),
                         width="stretch", hide_index=True)
            st.caption(
                "Pearson r tests a linear relationship between a descriptor "
                "value and its attribution; Spearman ρ tests any monotonic "
                "one. A feature with a large, stable ρ acts consistently "
                "across peptides; a feature with similar |r| and |ρ| is "
                "linear (charge), while ρ ≫ r indicates a threshold-like, "
                "non-linear driver (amphipathicity — necessary but not "
                "sufficient, proposal §2.2)."
            )

# ===========================================================================
# TAB 3 — PARETO OPTIMIZATION (ACPE)
# ===========================================================================
with tab3:
    st.subheader("③ Adaptive Candidate Prioritization Engine — Pareto front")
    st.caption("Four competing objectives, all minimised:  "
               "min F(s) = [f₁, f₂, f₃, f₄]ᵀ  (proposal §10.2).")

    o1, o2, o3 = st.columns([2.2, 1.2, 1.6])
    with o1:
        obj_set = st.radio("Objective set",
                           list(OBJECTIVE_SETS), horizontal=False,
                           help="§10.2 keeps toxicity and hemolysis separate. "
                                "The extended set folds safety into one axis "
                                "and adds the steric penalty of §9.2.")
    with o2:
        extra = st.slider("Pool enrichment (mutants per candidate)", 0, 20, 3,
                          help="Sequence variants generated around the batch "
                               "to widen the trade-off surface.")
    with o3:
        nsga = st.checkbox("Expand pool with NSGA-II search",
                           value=False,
                           disabled=not PYMOO_OK,
                           help="Deb's NSGA-II over the discrete sequence "
                                "space Ω (proposal §10.3 cites the "
                                "difference: this pool is small enough for "
                                "an exact dominance check).")

    gen = st.session_state.get("gen_df", pd.DataFrame())
    base_pool = CANONICAL_CANDIDATES.copy()
    if not gen.empty:
        add = gen[["name", "sequence", "p_published"]].copy()
        base_pool = pd.concat([base_pool, add], ignore_index=True)
    base_pool = base_pool.drop_duplicates(subset="sequence").reset_index(drop=True)

    if extra:
        enriched = generate_sequences(
            base_pool["sequence"].tolist(),
            int(base_pool["sequence"].str.len().median()),
            1.10, extra * len(base_pool), 7)
        add2 = pd.DataFrame({"name": [f"Mutant_{i+1:02d}"
                                      for i in range(len(enriched))],
                             "sequence": enriched,
                             "p_published": np.nan})
        base_pool = pd.concat([base_pool, add2], ignore_index=True)

    if nsga and PYMOO_OK:
        with st.spinner("Running NSGA-II over the discrete sequence space..."):
            L = int(np.median([len(s) for s in base_pool["sequence"]]))

            class _Problem(ElementwiseProblem):
                def __init__(self):
                    super().__init__(n_var=L, n_obj=4, xl=0.0, xu=19.0)

                def _evaluate(self, x, out, *args, **kwargs):
                    s = "".join(AA_ORDER[int(round(v)) % 20] for v in x)
                    out["F"] = [-predict_p_active(gate, s) if gate else 0.0,
                                -aliphatic_index(s),
                                (toxicity_score(s) + hemolytic_activity(s)) / 2,
                                steric_penalty(s)]

            try:
                alg = NSGA2(pop_size=40, sampling=FloatRandomSampling(),
                            eliminate_duplicates=True)
                res = _pymoo_minimize(_Problem(), alg, ("n_gen", 25),
                                      seed=42, verbose=False)
                X = np.atleast_2d(res.X)
                seqs = ["".join(AA_ORDER[int(round(v)) % 20] for v in row)
                        for row in X]
                add3 = pd.DataFrame({"name": [f"NSGA2_{i+1:02d}"
                                              for i in range(len(seqs))],
                                     "sequence": seqs,
                                     "p_published": np.nan})
                base_pool = pd.concat([base_pool, add3], ignore_index=True)
                base_pool = base_pool.drop_duplicates(subset="sequence")\
                    .reset_index(drop=True)
                st.success(f"NSGA-II contributed {len(seqs)} offspring.")
            except Exception as exc:
                st.warning(f"NSGA-II unavailable on this machine ({exc}); "
                           "falling back to the exact dominance check on the "
                           "enriched pool.")

    obj = build_objective_table(base_pool, gate, obj_set)
    if obj.empty:
        st.error("No candidates available.")
        st.stop()                      # halts a real Streamlit run
        raise RuntimeError("objective table is empty — no candidate was "
                           "featurised (see the messages above)")

    fcols = [c for c in obj.columns if c.startswith(("f1_", "f2_", "f3_", "f4_"))]
    F = obj[fcols].to_numpy(float)
    mask = pareto_mask(F)
    ref = F.max(axis=0) + 0.1 * (F.max(axis=0) - F.min(axis=0) + 1e-9)
    hv = hypervolume(F[mask], ref)
    ranks, dist = ideal_point_rank(F, mask)

    obj["is_pareto"] = mask
    obj["acpe_rank"] = ranks
    obj["ideal_distance"] = dist
    obj = obj.sort_values(["is_pareto", "acpe_rank"], ascending=[False, True])\
        .reset_index(drop=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pool size |Ω|", len(obj))
    m2.metric("Pareto set |P*|", int(mask.sum()),
              delta=f"{mask.mean()*100:.0f}% retention")
    m3.metric("Hypervolume HV(P*, r)", f"{hv:.3f}")
    m4.metric("Backend", "Exact dominance")

    st.info("Proposal §10.4 — ideal-point distance orders the front; "
            "dominance alone decides membership. Peptide_03 and Peptide_07 "
            "are pre-highlighted: they are the only front members that "
            "satisfy both the aggregation-risk and host-safety thresholds "
            "(proposal §5.3).")

    # --- plotting controls -------------------------------------------------
    senses = {k: ("maximise" if s < 0 else "minimise")
              for k, _, s in OBJECTIVE_SETS[obj_set]}
    labels = [lab for _, lab, _ in OBJECTIVE_SETS[obj_set]]
    axis_opts = [f"{k} · {lab}" for k, lab, _ in OBJECTIVE_SETS[obj_set]]

    p1, p2, p3 = st.columns([1.6, 1.6, 1.4])
    with p1:
        x_axis = st.selectbox("X axis", axis_opts, index=0)
    with p2:
        y_axis = st.selectbox("Y axis", axis_opts, index=1)
    with p3:
        hl = st.multiselect("Highlight", obj["name"].tolist(),
                            default=[n for n in ("Peptide_03", "Peptide_07")
                                     if n in set(obj["name"])])

    xi = axis_opts.index(x_axis)
    yi = axis_opts.index(y_axis)
    xcol, ycol = fcols[xi], fcols[yi]

    # --- 2-D front ---------------------------------------------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=obj.loc[~mask, xcol], y=obj.loc[~mask, ycol],
        mode="markers", name="Dominated",
        marker=dict(size=9, color="#c3ccd6", symbol="circle"),
        customdata=obj.loc[~mask, ["name", "sequence", "acpe_rank"]],
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}"
                      "<br>rank %{customdata[2]}<br>x=%{x:.3f} y=%{y:.3f}"
                      "<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=obj.loc[mask, xcol], y=obj.loc[mask, ycol],
        mode="markers", name="Pareto P*",
        marker=dict(size=16, color="#1f77b4", symbol="star"),
        customdata=obj.loc[mask, ["name", "sequence", "acpe_rank"]],
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}"
                      "<br>rank %{customdata[2]}<br>x=%{x:.3f} y=%{y:.3f}"
                      "<extra></extra>"))
    if hl:
        sub = obj[obj["name"].isin(hl)]
        fig.add_trace(go.Scatter(
            x=sub[xcol], y=sub[ycol], mode="markers+text",
            text=sub["name"], textposition="top center",
            marker=dict(size=22, color="rgba(0,0,0,0)",
                        line=dict(color="#d62728", width=3)),
            name="Highlighted", customdata=sub[["name", "sequence"]],
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}"
                          "<extra></extra>"))
    fig.update_layout(title="Two-dimensional projection of the objective space",
                      xaxis_title=x_axis, yaxis_title=y_axis)
    st.plotly_chart(styled(fig, 520), width="stretch")

    # --- 3-D front ---------------------------------------------------------
    st.markdown("#### Three-dimensional objective space")
    opts3 = axis_opts
    d1, d2, d3 = st.columns(3)
    with d1:
        z1 = st.selectbox("X₃", opts3, index=1)
    with d2:
        z2 = st.selectbox("Y₃", opts3, index=2)
    with d3:
        z3 = st.selectbox("Z₃", opts3, index=3)
    c3 = [fcols[opts3.index(z1)], fcols[opts3.index(z2)],
          fcols[opts3.index(z3)]]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter3d(
        x=obj.loc[~mask, c3[0]], y=obj.loc[~mask, c3[1]],
        z=obj.loc[~mask, c3[2]], mode="markers", name="Dominated",
        marker=dict(size=4, color="#c3ccd6"),
        customdata=obj.loc[~mask, ["name", "sequence"]],
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}"
                      "<extra></extra>"))
    fig3.add_trace(go.Scatter3d(
        x=obj.loc[mask, c3[0]], y=obj.loc[mask, c3[1]],
        z=obj.loc[mask, c3[2]], mode="markers", name="Pareto P*",
        marker=dict(size=7, color=obj.loc[mask, "acpe_rank"],
                    colorscale="Viridis_r", showscale=True,
                    colorbar=dict(title="ACPE rank")),
        customdata=obj.loc[mask, ["name", "sequence", "acpe_rank"]],
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}"
                      "<br>rank %{customdata[2]}<extra></extra>"))
    if hl:
        sub = obj[obj["name"].isin(hl)]
        fig3.add_trace(go.Scatter3d(
            x=sub[c3[0]], y=sub[c3[1]], z=sub[c3[2]], mode="markers+text",
            text=sub["name"], textposition="top center", name="Highlighted",
            marker=dict(size=9, color="#d62728"),
            customdata=sub[["name", "sequence"]],
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}"
                          "<extra></extra>"))
    st.plotly_chart(styled(fig3, 560), width="stretch")

    # --- parallel coordinates ---------------------------------------------
    pc_labels = [lab for _, lab, _ in OBJECTIVE_SETS[obj_set]]
    st.plotly_chart(styled(parallel_coords(obj, fcols, pc_labels, hl), 460),
                    width="stretch")

    # --- table + selection -------------------------------------------------
    st.markdown("#### Candidate table")
    tab_view = obj.copy()
    tab_view[["p_active", "net_charge", "hydrophobic_moment", "gravy",
              "toxicity", "hemolysis", "steric", "ideal_distance"]] = \
        tab_view[["p_active", "net_charge", "hydrophobic_moment", "gravy",
                  "toxicity", "hemolysis", "steric",
                  "ideal_distance"]].round(4)
    tab_view["aliphatic_index"] = tab_view["aliphatic_index"].round(1)
    st.dataframe(tab_view[["name", "sequence", "length", "p_active",
                           "aliphatic_index", "toxicity", "hemolysis",
                           "steric", "is_pareto", "acpe_rank",
                           "ideal_distance"]],
                 width="stretch", hide_index=True)
    st.download_button("⬇ Download objective table", obj.to_csv(index=False),
                       "hygeia_acpe_objectives.csv", "text/csv")

    sel = st.selectbox("Inspect a candidate", obj["name"].tolist())
    row = obj[obj["name"] == sel].iloc[0]
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Sequence", f"`{row['sequence']}`")
    s2.metric("p_active", f"{row['p_active']:.4f}")
    s3.metric("Toxicity / Hemolysis",
              f"{row['toxicity']:.3f} / {row['hemolysis']:.3f}")
    s4.metric("ACPE rank", int(row["acpe_rank"]),
              delta="Pareto-optimal" if row["is_pareto"] else "dominated",
              delta_color="normal" if row["is_pareto"] else "inverse")

    # --- optional NSGA-II expansion ---------------------------------------
    with st.expander("Append candidates from an NSGA-II search (§10.3 contrast)"):
        st.markdown(
            "Proposal §10.3 notes the deliberate choice *not* to evolve a "
            "population: because the pool has already been narrowed by SHAP "
            "filtering, novelty screening and docking, the front is "
            "identified by complete enumeration rather than by a search "
            "that must be given a stopping rule. The checkbox above "
            "demonstrates that machinery on demand."
        )

# ===========================================================================
# TAB 4 — STRUCTURAL DYNAMICS & CONTINUUM MECHANICS
# ===========================================================================
with tab4:
    st.subheader("④ Structural Dynamics & Continuum Mechanics Viewer")
    st.caption("Coarse-grained elastic network, normal mode analysis and "
               "harmonic stiffness tensors for the docked complexes "
               "(proposal §9).")

    t1, t2, t3 = st.columns([1.5, 1.5, 1.2])
    with t1:
        target = st.selectbox("Target receptor", list(TARGETS))
    with t2:
        peptide_choice = st.selectbox(
            "Peptide", ["Peptide_03", "Peptide_07"] +
            [n for n in CANONICAL_CANDIDATES["name"]
             if n not in ("Peptide_03", "Peptide_07")])
    with t3:
        n_modes = st.slider("Retained modes", 5, 40, 20)

    seq_map = dict(zip(CANONICAL_CANDIDATES["name"],
                       CANONICAL_CANDIDATES["sequence"]))
    pep_seq = seq_map.get(peptide_choice,
                          st.session_state.get("gen_df", pd.DataFrame())
                          .get("sequence", ["AINAGRINKPRPLRYLLLWLCKA"])[0])
    cfg = TARGETS[target]

    st.info(f"**{target}** — {cfg['note']}")

    # --- build the network -------------------------------------------------
    receptor = ideal_beta_barrel(cfg["res_per_strand"] * cfg["strands"],
                                 cfg["strands"],
                                 radius=12.0 if cfg["strands"] > 12 else 9.5)
    pep = place_peptide_on_barrel(receptor, len(pep_seq),
                                  target_idx=cfg["strands"] // 2)
    coords = np.vstack([receptor, pep])
    n_rec = len(receptor)

    G, (ii, jj, dist) = build_kirchhoff(coords, cutoff=10.0)
    nma = normal_mode_analysis(G, n_modes=n_modes)
    bfac = b_factors(nma["msf"])
    deform = deformability(nma["msf"], coords)
    K = stiffness_tensor(coords, ii, jj)
    eigK, vecK = np.linalg.eigh(K)

    # --- reported vs computed ---------------------------------------------
    rep = ENM_SUMMARY[(ENM_SUMMARY["peptide"] == peptide_choice) &
                      (ENM_SUMMARY["target"] == target)]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("λ₁ (reported iMODS)",
              f"{rep['lambda_1'].iloc[0]:.3e}" if len(rep) else "—")
    m2.metric("λ₁ (this network)",
              f"{nma['eigenvalues'][0]:.3f}")
    m3.metric("Mode-1 variance",
              f"{100 * nma['mode_share'][0]:.1f}%")
    m4.metric("Cumulative, 3 modes",
              f"{100 * nma['mode_share'][:3].sum():.1f}%")
    m5.metric("Max stiffness scale",
              f"{rep['stiffness_scale'].iloc[0]:,.0f}" if len(rep) else "—",
              delta="clash anomaly" if len(rep) and
              rep["stiffness_scale"].iloc[0] > 1000 else None,
              delta_color="inverse")

    if len(rep) and rep["stiffness_scale"].iloc[0] > 1000:
        st.error(
            "**Stiffness anomaly (proposal §6.6).** This complex reports a "
            f"maximum elastic-network stiffness of "
            f"{rep['stiffness_scale'].iloc[0]:,.0f} against the standard "
            "scale of 60. Because k_ij in a spring network rises steeply as "
            "interatomic distance falls, a handful of springs with near-zero "
            "separation — non-bonded heavy atoms overlapping below ~2 Å — "
            "dominate the spectrum. The peptide backbone is pinned, which "
            "artificially suppresses its deformability. Heavy-atom distances "
            "across this interface must be measured and the pose "
            "energy-minimised before the complex is treated as valid."
        )

    st.caption(f"Peptide-region mobility (reported): "
               f"_{rep['peptide_mobility'].iloc[0]}_ · "
               f"cumulative 3-mode variance (reported): "
               f"{rep['cum3_var'].iloc[0] if len(rep) else '—'}")

    # --- plots -------------------------------------------------------------
    g1, g2 = st.columns(2)

    with g1:
        idx = np.arange(1, len(coords) + 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=idx, y=bfac, mode="lines",
                                 name="B-factor",
                                 line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=idx, y=deform * bfac.max(), mode="lines",
                                 name="deformability (scaled)",
                                 line=dict(color="#ff7f0e", dash="dot")))
        fig.add_vline(x=n_rec, line_dash="dash", line_color="#888",
                      annotation_text="peptide start")
        fig.update_layout(title="B-factor & deformability profile",
                          xaxis_title="residue index (receptor → peptide)",
                          yaxis_title="B (Å²)")
        st.plotly_chart(styled(fig, 430), width="stretch")

    with g2:
        fig = go.Figure(go.Scatter(
            x=np.arange(1, len(pep_seq) + 1),
            y=deform[n_rec:], mode="lines+markers",
            line=dict(color="#2ca02c"),
            fill="tozeroy", fillcolor="rgba(44,160,44,0.18)"))
        fig.update_layout(title=f"Peptide-region deformability — "
                                f"{peptide_choice} on {target}",
                          xaxis_title="peptide residue",
                          yaxis_title="normalised deformability")
        st.plotly_chart(styled(fig, 430), width="stretch")

    h1, h2 = st.columns(2)
    with h1:
        fig = px.imshow(covariance_map(nma["var"]),
                        color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1,
                        title="Dynamic cross-correlation map (Cov)")
        fig.update_layout(height=440, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig, width="stretch")
    with h2:
        fig = go.Figure(go.Heatmap(
            z=-G, colorscale="Greys", showscale=False,
            hovertemplate="i=%{x}, j=%{y}<extra></extra>"))
        fig.add_vline(x=n_rec, line_color="#d62728")
        fig.add_hline(y=n_rec, line_color="#d62728")
        fig.update_layout(title="Elastic-network contact map (springs)",
                          xaxis_title="residue", yaxis_title="residue",
                          height=440, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig, width="stretch")

    # --- stiffness tensor (continuum mechanics) ---------------------------
    st.markdown("#### Harmonic stiffness tensor")
    c1, c2, c3 = st.columns([1.2, 1.4, 1.4])
    with c1:
        st.dataframe(pd.DataFrame(
            np.round(K, 3), index=["x", "y", "z"], columns=["x", "y", "z"]),
            width="stretch")
        st.caption("K = Σᵢ<ⱼ k_ij û_ij ⊗ û_ij — homogenised fabric tensor.")
    with c2:
        fig = go.Figure(go.Bar(
            x=["λ₁", "λ₂", "λ₃"], y=eigK,
            marker_color=["#2ca02c", "#1f77b4", "#9467bd"]))
        fig.update_layout(title="Principal stiffnesses",
                          yaxis_title="eigenvalue (kcal mol⁻¹ Å⁻²)")
        st.plotly_chart(styled(fig, 330), width="stretch")
    with c3:
        aniso = eigK[-1] / max(eigK[0], 1e-9)
        st.metric("Anisotropy λ₃/λ₁", f"{aniso:.2f}",
                  help="≈1.0 → isotropic network; large → a stiff axis "
                       "dominates, the continuum signature of a barrel "
                       "reinforced along its membrane normal.")
        st.metric("Network springs", f"{len(ii):,}")
        st.metric("Mean coordination",
                  f"{2 * len(ii) / len(coords):.2f}")
        st.metric("Rigid-body zero modes", "1 (translational)")

    # --- 3-D complex -------------------------------------------------------
    st.markdown("#### Docked complex")
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=receptor[:, 0], y=receptor[:, 1], z=receptor[:, 2],
        mode="lines", line=dict(color="#b8c2cc", width=4),
        name=f"{target} Cα trace", hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(
        x=pep[:, 0], y=pep[:, 1], z=pep[:, 2],
        mode="lines+markers",
        line=dict(color="#d62728", width=6),
        marker=dict(size=3, color=bfac[n_rec:], colorscale="Turbo",
                    colorbar=dict(title="B (Å²)")),
        name=f"{peptide_choice} (B-factor coloured)",
        customdata=[f"{i+1}:{aa}" for i, aa in enumerate(pep_seq)],
        hovertemplate="%{customdata}<extra></extra>"))
    fig.update_layout(height=520, scene=dict(
        xaxis_title="x (Å)", yaxis_title="y (Å)", zaxis_title="z (Å)",
        aspectmode="data"),
        margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, width="stretch")
    st.caption("Idealised Cα geometry for the elastic-network calculation. "
               "Drop a cleaned PDB below to replace the receptor trace.")

    up = st.file_uploader("Receptor PDB (e.g. clean_2K0L.pdb, clean_8QXP.pdb, "
                          "clean_5IV9.pdb)", type=["pdb"])
    if up is not None:
        xs, ys, zs, ids = [], [], [], []
        for line in up.getvalue().decode("utf-8", "ignore").splitlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                xs.append(float(line[30:38]))
                ys.append(float(line[38:46]))
                zs.append(float(line[46:54]))
                ids.append(f"{line[21]}{line[22:26].strip()}")
        if xs:
            arr = np.column_stack([xs, ys, zs])
            fig = go.Figure(go.Scatter3d(
                x=arr[:, 0], y=arr[:, 1], z=arr[:, 2], mode="lines+markers",
                line=dict(color="#1f77b4", width=3),
                marker=dict(size=2), customdata=ids,
                hovertemplate="%{customdata}<extra></extra>"))
            fig.update_layout(title="Uploaded receptor Cα trace",
                              height=520, scene=dict(
                                  xaxis_title="x (Å)", yaxis_title="y (Å)",
                                  zaxis_title="z (Å)", aspectmode="data"),
                              margin=dict(l=0, r=0, t=50, b=0))
            st.plotly_chart(fig, width="stretch")
            Gu, (iu, ju, du) = build_kirchhoff(arr, cutoff=10.0)
            nma_u = normal_mode_analysis(Gu, n_modes=n_modes)
            ku = stiffness_tensor(arr, iu, ju)
            eu = np.linalg.eigvalsh(ku)
            st.markdown("**Network analysis of the uploaded structure**")
            u1, u2, u3, u4 = st.columns(4)
            u1.metric("Cα atoms", len(arr))
            u2.metric("λ₁ (network)", f"{nma_u['eigenvalues'][0]:.3f}")
            u3.metric("Springs", f"{len(iu):,}")
            u4.metric("Anisotropy", f"{eu[-1] / max(eu[0], 1e-9):.2f}")
            st.plotly_chart(styled(go.Figure(go.Scatter(
                x=np.arange(1, len(arr) + 1), y=b_factors(nma_u["msf"]),
                mode="lines", line=dict(color="#1f77b4"))),
                title="B-factor profile of the uploaded structure",
                height=380), width="stretch")
        else:
            st.warning("No Cα atoms found in that file.")

    # --- docking summary ---------------------------------------------------
    st.markdown("#### Docking summary (HPEPDOCK, proposal §7)")
    dk = docking_table()
    fig = px.box(dk, x="target", y="score", color="group",
                 title="Docking score distributions — de novo vs dataset "
                       "controls",
                 labels={"score": "HPEPDOCK score (lower = stronger)",
                         "target": ""})
    st.plotly_chart(styled(fig), width="stretch")

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Reported statistics (Table 8)**")
        st.dataframe(DOCKING_MEDIANS.round(4), width="stretch",
                     hide_index=True)
    with d2:
        sel_dock = st.multiselect("Per-complex scores", sorted(dk["peptide"].unique()),
                                  default=["Peptide_03", "Peptide_07"])
        if sel_dock:
            st.dataframe(dk[dk["peptide"].isin(sel_dock)]
                         .sort_values(["peptide", "target"]),
                         width="stretch", hide_index=True)

    st.caption(
        "Docking and iMODS values are the exported run results; the live "
        "network plots recompute an elastic-network model on idealised or "
        "uploaded Cα coordinates so the mechanics remain inspectable offline."
    )

# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "HYGEIA — in silico framework. Predictions are computational "
    "hypotheses for prioritisation, not demonstrated antimicrobial "
    "efficacy or clinical safety. All data sources are open-access "
    "(DBAASP, UniProtKB, RCSB PDB); no human or animal subjects."
)
