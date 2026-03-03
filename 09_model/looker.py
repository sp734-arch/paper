#!/usr/bin/env python3
"""
Inspect Purified V2 Model Weights 

PAPER CONTEXT: This script provides detailed inspection of the Purified V2 model
weights for interpretability analysis. It verifies model integrity
against the paper's published hash and displays feature importance.

⚠️ SECURITY WARNING: This script loads a pickle file. Only run on models you
generated yourself using Genesis_purified_v2_PKL.py. Never load .pkl files
from untrusted sources.

USAGE:
    python inspect_purified_v2.py

PAPER REFERENCE: Section 8 
CREATED: 2024-02-07
LOCATION: /app/scripts/pkl/inspect_purified_v2.py
"""

import joblib
import pandas as pd
import numpy as np
import hashlib
import json
import sys
import os
from pathlib import Path

# Constants
MODEL_FILE = "pdhear_PURIFIED_V2_HASHED.pkl"
PAPER_HASH = "a50d941b6209f186a74d100e45cfee7a6a3c0ee70cdb17833fbd5e293b841dc0"
SCRIPT_DIR = Path(__file__).parent

def compute_bundle_hash(bundle):
    """
    Compute reproducible hash of model bundle for verification.
    
    PAPER CONTEXT: This matches the hash computation method used in the
    original model creation script (purified_v2_hashed_pkl.py).
    """
    # Hash the critical components (must match original computation)
    components = {
        'coefficients': [float(x) for x in bundle['model'].coef_.flatten()],
        'intercept': [float(x) for x in bundle['model'].intercept_.flatten()],
        'indices': sorted([int(x) for x in bundle['indices']]),
        'scaler_mean': [float(x) for x in bundle['scaler'].mean_.flatten()],
        'scaler_scale': [float(x) for x in bundle['scaler'].scale_.flatten()]
    }
    # Ensure deterministic ordering for hash consistency
    json_str = json.dumps(components, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(json_str.encode()).hexdigest()

def get_interpretation(weight, threshold_strong=0.5, threshold_moderate=0.2):
    """Get human-readable interpretation of weight magnitude."""
    abs_weight = abs(weight)
    if weight > 0:
        if abs_weight > threshold_strong:
            return "Strong PD biomarker"
        elif abs_weight > threshold_moderate:
            return "Moderate PD biomarker"
        else:
            return "Weak PD biomarker"
    else:
        if abs_weight > threshold_strong:
            return "Strong healthy biomarker"
        elif abs_weight > threshold_moderate:
            return "Moderate healthy biomarker"
        else:
            return "Weak healthy biomarker"

def inspect_model():
    """Load and inspect model weights from the local directory."""
    
    # Construct full path to model
    model_path = SCRIPT_DIR / MODEL_FILE
    
    print("\n" + "="*80)
    print("🔬 PURIFIED V2 MODEL INSPECTION - PAPER VERSION")
    print("="*80)
    print(f"Script: {Path(__file__).name}")
    print(f"Model:  {MODEL_FILE}")
    print(f"Path:   {model_path}")
    print("="*80)
    
    # Check if file exists
    if not model_path.exists():
        print(f"\n❌ ERROR: Model file not found: {model_path}")
        print(f"   Please ensure {MODEL_FILE} exists in the same directory.")
        print(f"   Current directory: {SCRIPT_DIR}")
        print(f"   Files in directory:")
        for f in SCRIPT_DIR.iterdir():
            print(f"     - {f.name}")
        sys.exit(1)
    
    # Load bundle
    print(f"\n📂 Loading model from: {model_path}")
    bundle = joblib.load(model_path)
    
    # Compute hash
    bundle_hash = compute_bundle_hash(bundle)
    
    print(f"\n🔐 MODEL VERIFICATION:")
    print(f"   Computed hash: {bundle_hash[:16]}...")
    print(f"   Paper hash:    {PAPER_HASH[:16]}...")
    
    if bundle_hash == PAPER_HASH:
        print("   ✅ HASH MATCHES PAPER VERSION")
        print("      This model is identical to the one used in the paper.")
    else:
        print("   ⚠️  HASH DIFFERS FROM PAPER VERSION")
        print("      This may be a regenerated model or modified version.")
    
    # Extract components
    model = bundle['model']
    indices = bundle['indices']
    weights = model.coef_[0]
    
    # Create analysis table
    analysis = pd.DataFrame({
        'HeAR_Dimension': indices,
        'Weight': weights,
        'Abs_Weight': np.abs(weights),
        'Direction': ['PD' if w > 0 else 'Healthy' for w in weights]
    })
    
    analysis = analysis.sort_values(by='Abs_Weight', ascending=False)
    
    # Add interpretation
    analysis['Interpretation'] = analysis['Weight'].apply(get_interpretation)
    
    print("\n" + "="*80)
    print("📊 TABLE X: PURIFIED V2 MODEL FEATURE WEIGHTS")
    print("="*80)
    print("\n| HeAR Dimension | Weight | Interpretation |")
    print("|:--------------|:-------|:---------------|")
    
    for _, row in analysis.iterrows():
        dim = int(row['HeAR_Dimension'])
        weight = row['Weight']
        interpretation = row['Interpretation']
        print(f"| {dim:14} | {weight:+7.3f} | {interpretation:16} |")
    
    print("\n*Note: Positive weights indicate Parkinson's disease, negative weights indicate healthy state.*")
    
    # Statistics
    print("\n" + "-" * 45)
    print("📈 WEIGHT STATISTICS:")
    print("-" * 45)
    print(f"  Mean absolute weight: {np.abs(weights).mean():.3f}")
    print(f"  Weight std deviation: {weights.std():.3f}")
    print(f"  Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
    print(f"  PD indicators (positive): {sum(weights > 0)}/7")
    print(f"  HC indicators (negative): {sum(weights < 0)}/7")
    print(f"  Intercept: {model.intercept_[0]:.3f}")
    
    # Bundle metadata
    print("\n" + "-" * 45)
    print("📋 MODEL METADATA:")
    print("-" * 45)
    
    if 'paper_reference' in bundle:
        print(f"  Paper reference: {bundle['paper_reference']}")
    
    if 'training_data_summary' in bundle:
        summary = bundle['training_data_summary']
        print(f"  Training samples: {summary.get('n_samples', 'N/A'):,}")
        print(f"  Subjects: {summary.get('n_subjects', 'N/A')}")
        print(f"  Healthy/PD: {summary.get('n_healthy', 'N/A'):,}/{summary.get('n_parkinsons', 'N/A'):,}")
    
    if 'loso_performance' in bundle:
        loso = bundle['loso_performance']
        print(f"  LOSO accuracy: {loso.get('mean_accuracy', 0):.3f} ± {loso.get('std_accuracy', 0):.3f}")
        print(f"  Folds: {loso.get('n_folds', 0)}")
    
    print("\n" + "="*80)
    print("✅ INSPECTION COMPLETE")
    print("="*80)
    
    return bundle, analysis

def export_weights_json(bundle, filename='model_weights.json'):
    """Export model weights to JSON for external analysis."""
    output_path = SCRIPT_DIR / filename
    
    weights_dict = {
        'paper_reference': 'Section 7.2.4 - Purified V2 Model',
        'model_file': MODEL_FILE,
        'inspection_script': Path(__file__).name,
        'dimensions': [int(x) for x in bundle['indices']],
        'weights': [float(x) for x in bundle['model'].coef_[0]],
        'intercept': float(bundle['model'].intercept_[0]),
        'scaler_mean': [float(x) for x in bundle['scaler'].mean_],
        'scaler_scale': [float(x) for x in bundle['scaler'].scale_],
        'hash': compute_bundle_hash(bundle),
        'hash_algorithm': 'SHA-256',
        'export_date': pd.Timestamp.now().isoformat(),
        'classes': {
            '0': 'healthy',
            '1': 'parkinsons'
        },
        'interpretation': {
            'positive_weights': 'Indicate Parkinson\'s disease',
            'negative_weights': 'Indicate healthy state'
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(weights_dict, f, indent=2)
    
    print(f"✅ Exported to {output_path}")
    return weights_dict

def show_directory_info():
    """Show information about files in the current directory."""
    print("\n📁 DIRECTORY INFORMATION:")
    print("-" * 45)
    print(f"Path: {SCRIPT_DIR}")
    print(f"Files:")
    
    files = list(SCRIPT_DIR.iterdir())
    if not files:
        print("  (empty)")
    else:
        for f in sorted(files):
            size = f.stat().st_size / 1024  # KB
            if f.is_file():
                print(f"  📄 {f.name:35} ({size:.1f} KB)")
            else:
                print(f"  📁 {f.name}")
    
    # Check for model file specifically
    model_path = SCRIPT_DIR / MODEL_FILE
    if model_path.exists():
        model_size = model_path.stat().st_size / 1024
        print(f"\n✅ Model file found: {MODEL_FILE} ({model_size:.1f} KB)")
    else:
        print(f"\n❌ Model file NOT found: {MODEL_FILE}")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print(f"\nUsage: python {Path(__file__).name} [options]")
        print("\nOptions:")
        print("  (no arguments)  Inspect the model in current directory")
        print("  -h, --help      Show this help message")
        print("  -l, --list      List files in directory")
        print("\nModel file expected: {MODEL_FILE}")
        sys.exit(0)
    
    if len(sys.argv) > 1 and sys.argv[1] in ['-l', '--list']:
        show_directory_info()
        sys.exit(0)
    
    try:
        # Show directory info first
        show_directory_info()
        
        # Run inspection
        bundle, analysis = inspect_model()
        
        # Ask about export
        print("\n" + "-" * 45)
        export = input("💾 Export weights to JSON for external analysis? (y/n): ").strip().lower()
        if export == 'y':
            weights_dict = export_weights_json(bundle)
            print(f"   Hash: {weights_dict['hash'][:16]}...")
            print(f"   File saved in: {SCRIPT_DIR}")
            
    except Exception as e:
        print(f"\n❌ Error during inspection: {e}")
        print("\nTroubleshooting:")
        print("  1. Check that you're in the correct directory")
        print(f"  2. Ensure {MODEL_FILE} exists")
        print("  3. Verify you have read permissions")
        print(f"\nCurrent directory: {SCRIPT_DIR}")
        sys.exit(1)
