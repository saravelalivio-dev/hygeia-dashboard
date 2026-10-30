"""
De Novo Antimicrobial Peptide Design Pipeline
Sections implemented:
- 2.2.2: Curation, Activity Thresholding, and Quality Control (CD-HIT)
- 2.3:   Feature Extraction via ESM-2 Protein Language Model (Embedding + Descriptors)

Usage:
    python main_pipeline.py --data data.csv --output results/
"""

import os
import argparse
import pandas as pd
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from cd_hit_clustering import cd_hit_quality_control, validate_cd_hit_installation
from feature_extraction import (
    ESM2FeatureExtractor,
    PhysicochemicalDescriptorExtractor,
    build_feature_fusion_matrix,
    validate_tokenizer_compatibility
)


CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='AMP design pipeline: CD-HIT curation + ESM-2 feature extraction'
    )
    parser.add_argument(
        '--data', required=True,
        help='Path to input CSV with columns: sequence, label (or mic_value)'
    )
    parser.add_argument(
        '--output', default='results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--sequence-col', default='sequence',
        help='Name of sequence column'
    )
    parser.add_argument(
        '--label-col', default='label',
        help='Name of label column (1=active, 0=inactive)'
    )
    parser.add_argument(
        '--device', default='auto',
        help='Device for ESM-2 (auto, cpu, cuda)'
    )
    parser.add_argument(
        '--skip-esm2', action='store_true',
        help='Skip ESM-2 embedding (use physchem features only)'
    )
    parser.add_argument(
        '--skip-cdhit', action='store_true',
        help='Skip CD-HIT deduplication'
    )
    parser.add_argument(
        '--identity-threshold', type=float, default=0.8,
        help='CD-HIT sequence identity threshold (default 0.8)'
    )
    return parser.parse_args()


def load_and_curate_data(csv_path, sequence_col, label_col):
    """Load dataset and apply initial curation with MIC thresholds."""
    df = pd.read_csv(csv_path)

    print("\n" + "=" * 60)
    print("SECTION 2.2.2: DATASET CURATION & QUALITY CONTROL")
    print("=" * 60)

    # Handle MIC values if present instead of binary labels
    if 'mic' in df.columns and label_col not in df.columns:
        # Section 2.2.2: Activity Thresholding
        # Active (AMP): MIC <= 60 ug/mL
        # Inactive (Non-AMP): MIC >= 32 ug/mL
        # Intermediate: 16 < MIC < 32 excluded
        print("\nApplying MIC activity thresholds:")
        print("  Active (AMP): MIC <= 60 ug/mL")
        print("  Inactive (Non-AMP): MIC >= 32 ug/mL")
        print("  Intermediate excluded: 16 < MIC < 32 ug/mL")

        active_mask = df['mic'] <= 60
        inactive_mask = df['mic'] >= 32
        df.loc[active_mask, label_col] = 1
        df.loc[inactive_mask, label_col] = 0

        excluded = ~(active_mask | inactive_mask)
        if excluded.sum() > 0:
            print(f"  Excluding {excluded.sum()} intermediate entries")
            df = df[~excluded]

    # Validate required columns
    if sequence_col not in df.columns:
        raise ValueError(f"Sequence column '{sequence_col}' not found")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found")

    print(f"\nLoaded {len(df)} sequences")
    print(f"Class distribution: {np.bincount(df[label_col].astype(int))}")

    return df


def main():
    args = parse_arguments()
    os.makedirs(args.output, exist_ok=True)

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu'
    else:
        device = args.device
    print(f"\nUsing device: {device}")

    # 1. Load and curate data
    df = load_and_curate_data(args.data, args.sequence_col, args.label_col)

    # 2. Run CD-HIT quality control (Section 2.2.2)
    if not args.skip_cdhit:
        print("\nChecking CD-HIT availability...")
        cd_hit_available = validate_cd_hit_installation()
        if not cd_hit_available:
            print("CD-HIT binary not found - will use built-in in-memory "
                  "fallback (same identity clustering logic).")
            print("To use native CD-HIT, install via:")
            print("  conda install -c bioconda cd-hit")

        df_filtered, cdhit_stats, removed = cd_hit_quality_control(
            df, args.sequence_col, identity_threshold=args.identity_threshold
        )

        # Save CD-HIT QC report
        report_path = os.path.join(args.output, 'cdhit_qc_report.csv')
        pd.DataFrame([{
            'identity_threshold': args.identity_threshold,
            'input_sequences': cdhit_stats['input_sequences'],
            'output_sequences': cdhit_stats['output_sequences'],
            'redundancy_removed': cdhit_stats['redundancy_removed'],
            'reduction_percentage': cdhit_stats['reduction_percentage'],
            'num_clusters': cdhit_stats['num_clusters'],
            'method': cdhit_stats['method']
        }]).to_csv(report_path, index=False)
        print(f"\nSaved CD-HIT QC report to {report_path}")

        df = df_filtered
    else:
        print("\nSkipping CD-HIT (--skip-cdhit)")

    # 3. Print final dataset summary
    print(f"\nFinal curated dataset: {len(df)} sequences")
    print(f"Class distribution: {np.bincount(df[args.label_col].astype(int))}")

    sequences = df[args.sequence_col].tolist()
    labels = df[args.label_col].astype(int).values

    # 4. Feature extraction (Section 2.3)
    print("\n" + "=" * 60)
    print("SECTION 2.3: FEATURE EXTRACTION VIA ESM-2")
    print("=" * 60)

    # Validate tokenizer compatibility (Section 2.3.1)
    valid_seqs, invalid_seqs = validate_tokenizer_compatibility(sequences)
    print(f"\nESM-2 compatible sequences: {len(valid_seqs)}")
    if invalid_seqs:
        print(f"  {len(invalid_seqs)} sequences incompatible with ESM-2 tokenizer")

    if args.skip_esm2:
        feature_matrix = build_feature_fusion_matrix(
            valid_seqs, use_esm2=False, device=device
        )
    else:
        esm2 = ESM2FeatureExtractor(device=device)
        feature_matrix = build_feature_fusion_matrix(
            valid_seqs, use_esm2=True,
            esm2_extractor=esm2, device=device
        )

    # Align labels with valid sequences
    valid_mask = [s in valid_seqs for s in sequences]
    labels = labels[valid_mask]

    print(f"\nFinal feature matrix: {feature_matrix.shape}")
    print(f"Final label vector: {labels.shape}")

    # Save feature matrix with labels
    feature_matrix['label'] = labels
    feature_matrix_out = os.path.join(args.output, 'feature_matrix.csv')
    feature_matrix.to_csv(feature_matrix_out, index=False)
    print(f"Saved feature matrix to {feature_matrix_out}")

    # Save ESM-2 embeddings separately for downstream use
    esm2_cols = [c for c in feature_matrix.columns if c.startswith('esm2_')]
    physchem_cols = [
        c for c in feature_matrix.columns
        if not c.startswith('esm2_') and c != 'label'
    ]
    print(f"\n  ESM-2 embedding dimensions: {len(esm2_cols)}")
    print(f"  Physicochemical descriptors: {len(physchem_cols)}")
    total_dim = len(esm2_cols) + len(physchem_cols)
    if esm2_cols:
        print(f"  Feature fusion: X = [e_ESM2 ({len(esm2_cols)}) || "
              f"p_RDKit ({len(physchem_cols)})] in R^{total_dim}")
    else:
        print(f"  Physchem-only feature matrix: X in R^{total_dim}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE (Sections 2.2.2 + 2.3)")
    print("=" * 60)
    print("\nResults saved to:")
    print(f"  {args.output}/cdhit_qc_report.csv")
    print(f"  {args.output}/feature_matrix.csv")


if __name__ == "__main__":
    main()