"""
Data Quality Validation: Spectral Integrity Check
==================================================

PURPOSE:
--------
An optional utility to verify that audio embeddings contain full-spectrum information and are not
affected by telephony bandwidth limitations (e.g., 8kHz PSTN cutoff).

BACKGROUND:
-----------
Some speech datasets are labeled as "telephone recordings," which could mean:
1. Modern smartphones (full bandwidth, 16kHz+ Nyquist) ✓
2. Legacy PSTN lines (8kHz bandwidth, missing high frequencies) ✗

This script checks whether embeddings show signs of 8kHz lowpass filtering
by measuring high-frequency energy (HeAR dimensions 32+).

WHEN TO USE:
------------
- After extracting embeddings from a new dataset
- When dataset metadata mentions "telephone" or "phone" recordings
- To validate that training data has consistent spectral coverage

PAPER REFERENCE:
----------------
Quality Control
This validation ensures that recording device differences (smartphone vs studio)
do not introduce systematic spectral artifacts that could confound the model.

AUTHORS: Jim McCormack
CREATED: 02/07/2026
"""

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Spectral integrity threshold
# HeAR embeddings with high-frequency energy below this value indicate
# possible 8kHz bandwidth limitation (telephony codec artifacts)
HIGH_FREQ_ENERGY_THRESHOLD = 1e-6

# HeAR embedding dimensions
# Dimensions 0-31: Lower frequency bands
# Dimensions 32-511: Higher frequency bands (tested here)
HIGH_FREQ_DIM_START = 32

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def check_spectral_integrity(embedding_path):
    """
    Check if a single embedding has full-spectrum content.
    
    Args:
        embedding_path (str): Path to .npy embedding file
    
    Returns:
        dict: {
            'file': filename,
            'high_freq_energy': float,
            'status': 'PASS' or 'FAIL',
            'suspected_issue': str (if failed)
        }
    """
    try:
        embedding = np.load(embedding_path)
        
        # Handle both 1D and 2D embeddings
        # (1D = single segment, 2D = multiple segments stacked)
        if embedding.ndim == 1:
            high_freq_energy = np.mean(np.abs(embedding[HIGH_FREQ_DIM_START:]))
        else:
            high_freq_energy = np.mean(np.abs(embedding[:, HIGH_FREQ_DIM_START:]))
        
        # Check against threshold
        is_pass = high_freq_energy > HIGH_FREQ_ENERGY_THRESHOLD
        
        return {
            'file': os.path.basename(embedding_path),
            'high_freq_energy': high_freq_energy,
            'status': 'PASS' if is_pass else 'FAIL',
            'suspected_issue': '8kHz telephony cutoff' if not is_pass else None
        }
    
    except Exception as e:
        return {
            'file': os.path.basename(embedding_path),
            'high_freq_energy': None,
            'status': 'ERROR',
            'suspected_issue': str(e)
        }

def scan_dataset(dataset_path, dataset_name="Dataset"):
    """
    Scan all embeddings in a dataset directory for spectral integrity.
    
    Args:
        dataset_path (str): Path to directory containing .npy embedding files
        dataset_name (str): Human-readable name for reporting
    
    Returns:
        pd.DataFrame: Validation results for all files
    """
    print("=" * 70)
    print(f"🔍 SPECTRAL INTEGRITY SCAN: {dataset_name}")
    print("=" * 70)
    
    # Find all .npy files recursively
    dataset_path = Path(dataset_path)
    embedding_files = list(dataset_path.rglob("*.npy"))
    
    if not embedding_files:
        print(f"⚠️  No .npy files found in {dataset_path}")
        print(f"    Check path or run embedding extraction first.")
        return None
    
    print(f"📂 Found {len(embedding_files)} embedding files")
    print(f"🧪 Testing for 8kHz bandwidth limitation...\n")
    
    # Check each file
    results = []
    for emb_path in tqdm(embedding_files, desc="Validating"):
        result = check_spectral_integrity(emb_path)
        results.append(result)
    
    # Create results dataframe
    df = pd.DataFrame(results)
    
    # Compute statistics
    pass_count = (df['status'] == 'PASS').sum()
    fail_count = (df['status'] == 'FAIL').sum()
    error_count = (df['status'] == 'ERROR').sum()
    pass_rate = pass_count / len(df) if len(df) > 0 else 0
    
    # Report
    print("\n" + "=" * 70)
    print(f"📊 {dataset_name.upper()} VALIDATION REPORT")
    print("=" * 70)
    print(f"Total files scanned:        {len(df)}")
    print(f"✅ PASS (full spectrum):    {pass_count} ({pass_rate:.1%})")
    print(f"❌ FAIL (8kHz suspected):   {fail_count}")
    print(f"⚠️  ERROR (read failure):    {error_count}")
    
    if fail_count > 0:
        print(f"\n⚠️  WARNING: Detected {fail_count} files with low high-frequency energy")
        print(f"   This suggests possible 8kHz telephony bandwidth limitation.")
        print(f"\n   Failed files (showing first 10):")
        failed_files = df[df['status'] == 'FAIL'].head(10)
        for idx, row in failed_files.iterrows():
            print(f"   - {row['file']}")
        print(f"\n   ⚡ ACTION REQUIRED:")
        print(f"   1. Verify recording device specifications")
        print(f"   2. Check if dataset was PSTN-captured (8kHz limit)")
        print(f"   3. Consider excluding these files from training")
    else:
        print(f"\n✅ EXCELLENT: All files show full-spectrum content")
        print(f"   No 8kHz telephony artifacts detected")
        print(f"   Dataset is suitable for multi-device training")
    
    print("=" * 70 + "\n")
    
    return df

# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate spectral integrity of HeAR embeddings"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to directory containing .npy embedding files"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="Dataset",
        help="Human-readable name for the dataset (for reporting)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: Save detailed results to CSV"
    )
    
    args = parser.parse_args()
    
    # Run validation
    results_df = scan_dataset(args.dataset_path, args.dataset_name)
    
    # Save results if requested
    if results_df is not None and args.output:
        results_df.to_csv(args.output, index=False)
        print(f"💾 Detailed results saved to: {args.output}")
