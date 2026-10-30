"""
Section 2.3 - Feature Extraction via ESM-2 Protein Language Model
High-Dimensional Feature Extraction Pipeline

Computes:
1. ESM-2 evolutionary semantic embeddings (1280-dim per sequence)
2. Physicochemical descriptors (RDKit / ProtParam)
3. Hybridized feature fusion matrix X = [e_ESM2 (1280) || p_RDKit (20)] in R^1300
"""

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import molecular_weight as bio_mw
import re

# Optional imports for ESM-2 (torch/transformers) - not required for physchem-only
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARN: torch/transformers not installed. ESM-2 embeddings unavailable.")
    print("      Install with: pip install torch transformers")


# ESM-2 model configuration from proposal Section 2.3.2
# Default: proposal-specified model. Can be overridden at init time.
ESM2_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
ESM2_EMBEDDING_DIM = 1280  # d = 1280 per proposal
CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


class ESM2FeatureExtractor:
    """
    Contextual sequence representations using pretrained ESM-2 transformer.

    Implements Section 2.3.2:
    - Tokenization and Forward Pass: sequence s converted to token IDs
      bounded by <cls> and <eos> tokens, routed through 33-layer network
    - Per-Residue Extraction: final hidden layer tensor shape (L+2) x 1280
    - Mean Pooling Aggregation: e(s) = (1/L) * sum_{i=1}^{L} h_i
      outputs fixed 1280-dimensional continuous embedding vector
    """

    def __init__(self, model_name=None, device="cpu"):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "torch/transformers not installed. Install with:\n"
                "  pip install torch transformers"
            )
        if model_name is None:
            # Resolve at runtime so callers can override the module global
            model_name = ESM2_MODEL_NAME
        self.device = device
        self.model_name = model_name
        print(f"Loading ESM-2 model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        # Derive embedding dimension from the actual model config
        self.embedding_dim = self.model.config.hidden_size

    def tokenize_sequences(self, sequences):
        """Convert sequences to token IDs with <cls> and <eos> tokens."""
        tokenized = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32  # L+2 for max length 30
        )
        return tokenized

    def get_mean_pooled_embedding(self, sequences, batch_size=8):
        """
        Compute mean-pooled ESM-2 embeddings.

        Formula (Section 2.3.2):
            e(s) = (1/L) * SUM_{i=1}^{L} h_i

        Where:
            L = sequence length (number of residue tokens)
            h_i = hidden representation of residue i from final layer
            <cls> and <eos> boundary tokens are EXCLUDED from pooling

        Returns:
            Array of shape (N, 1280)
        """
        all_embeddings = []

        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=32
            ).to(self.device)

            # Forward pass through ESM-2 transformer (no gradient tracking)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Hidden states of final layer: shape (batch, L+2, 1280)
                hidden = outputs.last_hidden_state

            # Exclude <cls> (index 0) and <eos> (last) tokens
            # Mean pooling over L residue tokens only
            attention_mask = inputs["attention_mask"]
            # Zero out boundary tokens for all sequences
            lengths = mask_lengths = attention_mask.sum(dim=1)
            embeddings = []
            for j in range(hidden.size(0)):
                seq_len = lengths[j].item()
                # Residue tokens are indices 1 to seq_len-2 for ESM-2 tokenizer
                # (since there's no padding within batch after truncation)
                # For ESM-2, format is: <cls> residue1 ... residueL <eos> <pad>...
                # Sequence content is between position 1 and (index of last non-pad - 1)
                # Last non-pad token is <eos>, so residue tokens = positions 1..(len-2)
                residue_tokens = hidden[j, 1:seq_len - 1, :]  # L x 1280
                mean_pooled = residue_tokens.mean(dim=0)  # 1280-dim
                embeddings.append(mean_pooled)

            all_embeddings.extend(
                [e.cpu().numpy() for e in embeddings]
            )

        return np.array(all_embeddings)


class PhysicochemicalDescriptorExtractor:
    """
    Deterministic physicochemical descriptor calculation.

    Implements Section 2.4.1:
    - RDKit/ProtParam descriptors for 20 standard canonical amino acids
    - Net charge, hydrophobic moment, amphipathicity, molecular weight,
      aliphatic index, instability index, pI, GRAVY
    """

    # Kyte-Doolittle hydrophobicity scale (kcal/mol)
    KYTE_DOOLITTLE = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }

    # Eisenberg consensus hydrophobicity scale
    EISENBERG = {
        'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
        'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
        'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
        'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
    }

    # Amino acid properties for net charge calc (pKa values)
    PKA_AMINO = {'NT': 9.69, 'CT': 2.34}
    PKA_SIDECHAIN = {
        'R': 12.48, 'D': 3.65, 'E': 4.25,
        'H': 6.00, 'C': 8.18, 'Y': 10.07, 'K': 10.53
    }

    # Eisenberg hydrophobic moment periodicity angle (100 degrees for amphipathic)
    ANGLE = 100.0

    def __init__(self):
        self.canonical_aa = CANONICAL_AA

    def _validate_sequence(self, sequence):
        if not all(aa in CANONICAL_AA for aa in sequence.upper()):
            raise ValueError(
                f"Non-canonical amino acid in sequence: {sequence}"
            )
        return sequence.upper()

    def compute_net_charge(self, sequence, pH=7.0):
        """
        Compute net charge using Henderson-Hasselbalch equation.

        Q = SUM(N_i * z_i) where z_i is the charge state at given pH
        """
        seq = self._validate_sequence(sequence)
        n_term = (1 / (1 + 10**(self.PKA_AMINO['NT'] - pH)))  # +1 when protonated
        c_term = -(1 / (1 + 10**(pH - self.PKA_AMINO['CT'])))  # -1 when deprotonated

        charge = n_term + c_term
        for aa in seq:
            if aa in self.PKA_SIDECHAIN:
                pka = self.PKA_SIDECHAIN[aa]
                if aa in ['R', 'K']:  # Basic residues
                    charge += 1 / (1 + 10**(pka - pH))
                elif aa in ['D', 'E', 'C', 'Y']:  # Acidic residues
                    charge += -1 / (1 + 10**(pH - pka))
                elif aa == 'H':  # Histidine (can be +/-)
                    charge += 1 / (1 + 10**(pka - pH))
        return charge

    def compute_hydrophobic_moment(self, sequence):
        """
        Compute Eisenberg hydrophobic moment (mu_H).

        mu_H = sqrt( (SUM H_n * sin(n*delta))^2 + (SUM H_n * cos(n*delta))^2 )

        Where:
            H_n = hydrophobicity of residue n
            delta = angle between successive residues (100 deg for alpha-helix)
        """
        seq = self._validate_sequence(sequence)
        delta = np.radians(self.ANGLE)

        sum_sin = sum(
            self.EISENBERG[aa] * np.sin(n * delta)
            for n, aa in enumerate(seq)
        )
        sum_cos = sum(
            self.EISENBERG[aa] * np.cos(n * delta)
            for n, aa in enumerate(seq)
        )
        return float(np.sqrt(sum_sin**2 + sum_cos**2))

    def compute_gravy(self, sequence):
        """Grand Average of Hydropathy (GRAVY) index using Kyte-Doolittle."""
        seq = self._validate_sequence(sequence)
        return sum(self.KYTE_DOOLITTLE[aa] for aa in seq) / len(seq)

    def compute_amphipathicity(self, sequence):
        """Amphipathicity score from hydrophobic moment / mean hydrophobicity."""
        seq = self._validate_sequence(sequence)
        mu = self.compute_hydrophobic_moment(seq)
        mean_hydro = sum(self.EISENBERG[aa] for aa in seq) / len(seq)
        return mu / (mean_hydro + 1e-10)

    def compute_aliphatic_index(self, sequence):
        """Compute aliphatic index using Ikai's formula."""
        seq = self._validate_sequence(sequence)
        n = len(seq)
        ala = seq.count('A')
        val = seq.count('V')
        leu = seq.count('L')
        ile = seq.count('I')
        return float((ala + 2.9 * val + 3.9 * (leu + ile)) * 100 / n)

    def compute_instability_index(self, sequence):
        """
        Compute instability index using ProtParam algorithm.
        (Guruprasad, Reddy, & Pandit 1990)
        """
        seq = self._validate_sequence(sequence)

        # Dipeptide instability weight table (subset of standard table)
        # Full table available in ProtParam
        try:
            analysis = ProteinAnalysis(seq)
            return float(analysis.instability_index())
        except Exception:
            return 0.0

    def compute_pI(self, sequence):
        """Theoretical isoelectric point using ProtParam."""
        seq = self._validate_sequence(sequence)
        try:
            analysis = ProteinAnalysis(seq)
            return float(analysis.isoelectric_point())
        except Exception:
            return 7.0

    def compute_molecular_weight(self, sequence):
        """Molecular weight using Biopython (protein sequence)."""
        seq = self._validate_sequence(sequence)
        return float(bio_mw(seq, seq_type="protein"))

    def compute_all_descriptors(self, sequence):
        """Compute all physicochemical descriptors for a sequence."""
        seq = self._validate_sequence(sequence)

        return {
            "length": len(seq),
            "net_charge_pH7": self.compute_net_charge(seq),
            "hydrophobic_moment": self.compute_hydrophobic_moment(seq),
            "gravy_index": self.compute_gravy(seq),
            "amphipathicity": self.compute_amphipathicity(seq),
            "aliphatic_index": self.compute_aliphatic_index(seq),
            "instability_index": self.compute_instability_index(seq),
            "isoelectric_point": self.compute_pI(seq),
            "molecular_weight": self.compute_molecular_weight(seq),

            # Additional composition descriptors
            "aa_composition_fraction": self._composition_fraction(seq),
            "aromaticity": self._aromaticity(seq),
            "secondary_structure": self._secondary_structure_fraction(seq)
        }

    def _composition_fraction(self, sequence):
        """Fraction of amino acid composition for each class."""
        seq = self._validate_sequence(sequence)
        n = len(seq)
        nonpolar = sum(seq.count(aa) for aa in 'AVILMFWPG') / n
        polar = sum(seq.count(aa) for aa in 'SNQTYC') / n
        basic = sum(seq.count(aa) for aa in 'KRH') / n
        acidic = sum(seq.count(aa) for aa in 'DE') / n
        return {
            'nonpolar_frac': nonpolar,
            'polar_frac': polar,
            'basic_frac': basic,
            'acidic_frac': acidic
        }

    def _aromaticity(self, sequence):
        """Aromaticity (F, W, Y content)."""
        seq = self._validate_sequence(sequence)
        return sum(seq.count(aa) for aa in 'FYW') / len(seq)

    def _secondary_structure_fraction(self, sequence):
        """Secondary structure fractions using Chou-Fasman."""
        try:
            analysis = ProteinAnalysis(self._validate_sequence(sequence))
            return analysis.secondary_structure_fraction()
        except Exception:
            return (0.0, 0.0, 0.0)


def build_feature_fusion_matrix(sequences, use_esm2=True,
                                esm2_extractor=None, device="cpu"):
    """
    Build the hybridized feature fusion matrix X.

    Implements Section 2.4.1:
        X_i = [e_ESM2, p_RDKit] in R^1300

    Where:
        e_ESM2: 1280-dim mean-pooled ESM-2 embedding
        p_RDKit: 20-dim physicochemical descriptors

    Returns:
        DataFrame with columns:
            - 1280 ESM-2 columns
            - 20 RDKit descriptor columns
    """
    descriptors = PhysicochemicalDescriptorExtractor()

    # Compute ESM-2 embeddings
    if use_esm2:
        if esm2_extractor is None:
            esm2_extractor = ESM2FeatureExtractor(device=device)
        print("Computing ESM-2 embeddings...")
        esm2_embeddings = esm2_extractor.get_mean_pooled_embedding(sequences)

    # Compute physicochemical descriptors
    print("Computing physicochemical descriptors...")
    descriptor_rows = []
    for seq in sequences:
        desc = descriptors.compute_all_descriptors(seq)
        row = {
            "net_charge": desc["net_charge_pH7"],
            "hydrophobic_moment": desc["hydrophobic_moment"],
            "gravy_index": desc["gravy_index"],
            "amphipathicity": desc["amphipathicity"],
            "aliphatic_index": desc["aliphatic_index"],
            "instability_index": desc["instability_index"],
            "isoelectric_point": desc["isoelectric_point"],
            "molecular_weight": desc["molecular_weight"],
        }
        # Flatten composition and secondary structure dicts
        comp = desc["aa_composition_fraction"]
        row.update({
            "nonpolar_frac": comp["nonpolar_frac"],
            "polar_frac": comp["polar_frac"],
            "basic_frac": comp["basic_frac"],
            "acidic_frac": comp["acidic_frac"],
        })
        ss = desc["secondary_structure"]
        row.update({
            "ss_helix": ss[0],
            "ss_sheet": ss[1],
            "ss_turn": ss[2],
            "aromaticity": desc["aromaticity"],
            "length": desc["length"],
        })
        descriptor_rows.append(row)

    physchem_df = pd.DataFrame(descriptor_rows)
    print(f"Physicochemical descriptors computed: {physchem_df.shape}")

    # Add additional physchem features to reach exactly 20 dimensions
    additional_features = pd.DataFrame({
        "charge_density": [
            desc["net_charge"] / len(s) for desc, s in zip(descriptor_rows, sequences)
        ],
        "hydrophobic_balance": [desc["gravy_index"] / (desc["hydrophobic_moment"] + 1e-10)
                                for desc in descriptor_rows],
        "cationic_ratio": [desc["basic_frac"] / (desc["acidic_frac"] + 1e-10)
                           for desc in descriptor_rows],
    })
    physchem_df = pd.concat([physchem_df, additional_features], axis=1)

    print(f"Final physicochemical matrix: {physchem_df.shape[1]} features")
    assert physchem_df.shape[1] == 20, (
        f"Expected 20 physicochemical features, got {physchem_df.shape[1]}"
    )

    if use_esm2:
        embedding_dim = esm2_embeddings.shape[1]
        esm2_df = pd.DataFrame(
            esm2_embeddings,
            columns=[f"esm2_dim_{i}" for i in range(embedding_dim)]
        )
        # Horizontal concatenation: X = [e_ESM2 || p_RDKit]
        feature_matrix = pd.concat([esm2_df, physchem_df], axis=1)
        print(f"Feature fusion matrix: {feature_matrix.shape} = "
              f"[{embedding_dim} ESM-2 + {physchem_df.shape[1]} physchem]")
    else:
        feature_matrix = physchem_df
        print(f"Physchem-only matrix: {feature_matrix.shape}")

    return feature_matrix


def validate_tokenizer_compatibility(sequences):
    """
    Validate that all sequences are compatible with ESM-2 tokenizer.

    Section 2.3.1: All sequences with non-canonical, ambiguous characters
    are explicitly filtered out during dataset quality control.
    """
    valid = []
    invalid = []
    for seq in sequences:
        if all(aa in CANONICAL_AA for aa in seq.upper()):
            valid.append(seq)
        else:
            invalid.append(seq)
    return valid, invalid
