#!/usr/bin/env python3
"""
Identity Leakage test
PIPELINE ALIGNMENT TEST - Step 1 & Step 2
================================================================================
PURPOSE:
    Verifies that Step 1 outputs are correctly aligned and Step 2 can load them
    without errors. Tests both KCL 30s mode and full cross-lingual mode.

📋 HOW TO USE:
    Quick test (no cleanup):
        # Test both modes
        python test_pipeline_alignment.py
        
        # Test only KCL mode
        python test_pipeline_alignment.py --kcl_30s
    
    Clean test (delete existing files):
        # Full clean test
        python test_pipeline_alignment.py --clean
        
        # Clean test of KCL only
        python test_pipeline_alignment.py --clean --kcl_30s

WHAT THIS TESTS:
    ✓ Step 1 execution (both KCL and full modes)
    ✓ File existence and completeness
    ✓ Metadata/embedding alignment (row counts match)
    ✓ Placeholder detection (subjects with 0 slices)
    ✓ Subject counts (HC=17, PD=12 for KCL mode)
    ✓ Step 2 execution and data loading
    ✓ Results file integrity via hash verification

RETURNS:
    0 if all tests pass
    1 if any test fails
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import subprocess
import json
import hashlib
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
FEATURES_DIR = Path(r'C:\Projects\hear_italian\features_certified')
RESULTS_DIR = Path(r'C:\Projects\hear_italian\step2_results')
TEST_RUN_NAME = "pipeline_test"

def run_step1_kcl():
    """Run Step 1 in KCL 30s mode"""
    print("\n" + "="*70)
    print("🔧 TEST 1: Step 1 (KCL 30s Mode)")
    print("="*70)
    
    cmd = ["python", "11_certified_extractor_pytorch.py", 
           "--kcl_30s", "--device", "cuda", "--silence_thresh", "-55"]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Step 1 FAILED")
        print(result.stderr)
        return False
    
    print("✅ Step 1 completed")
    return True

def run_step1_full():
    """Run Step 1 in full cross-lingual mode"""
    print("\n" + "="*70)
    print("🔧 TEST 2: Step 1 (Full Cross-Lingual Mode)")
    print("="*70)
    
    cmd = ["python", "11_certified_extractor_pytorch.py", 
           "--device", "cuda", "--silence_thresh", "-55"]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Step 1 FAILED")
        print(result.stderr)
        return False
    
    print("✅ Step 1 completed")
    return True

def verify_step1_outputs(prefix):
    """Verify Step 1 outputs are consistent"""
    print(f"\n📊 Verifying {prefix} outputs...")
    
    # Check files exist
    required_files = [
        f"{prefix}_metadata.csv",
        f"{prefix}_embeddings_512_raw.npy",
        f"{prefix}_embeddings_7d_raw.npy",
        f"{prefix}_embeddings_7d_scaled.npy"
    ]
    
    missing = []
    for f in required_files:
        if not (FEATURES_DIR / f).exists():
            missing.append(f)
    
    if missing:
        print(f"❌ Missing files: {missing}")
        return False
    
    print("✅ All required files present")
    
    # Load and verify alignment
    metadata = pd.read_csv(FEATURES_DIR / f"{prefix}_metadata.csv")
    emb_512 = np.load(FEATURES_DIR / f"{prefix}_embeddings_512_raw.npy")
    emb_7d_scaled = np.load(FEATURES_DIR / f"{prefix}_embeddings_7d_scaled.npy")
    
    print(f"\n   Metadata: {len(metadata)} rows")
    print(f"   512D embeddings: {emb_512.shape}")
    print(f"   7D scaled embeddings: {emb_7d_scaled.shape}")
    
    # Check alignment
    if len(metadata) != len(emb_512) != len(emb_7d_scaled):
        print(f"❌ MISALIGNMENT: Meta={len(metadata)}, 512={len(emb_512)}, 7d={len(emb_7d_scaled)}")
        return False
    
    print("✅ Metadata and embeddings aligned")
    
    # Check for placeholder rows
    if 'slice_index' in metadata.columns:
        placeholders = (metadata['slice_index'] == -1).sum()
        if placeholders > 0:
            print(f"⚠️  Found {placeholders} placeholder rows (slice_index = -1)")
            subjects_with_placeholders = metadata[metadata['slice_index'] == -1]['subject_id'].unique()
            print(f"   Subjects with placeholders: {list(subjects_with_placeholders)}")
            
            # Test removal
            metadata_clean = metadata[metadata['slice_index'] != -1]
            print(f"   After removal: {len(metadata_clean)} rows")
        else:
            print("✅ No placeholder rows found")
    
    # Subject counts
    n_subjects = metadata['subject_id'].nunique()
    n_hc = metadata[metadata['disease']=='HC']['subject_id'].nunique()
    n_pd = metadata[metadata['disease']=='PD']['subject_id'].nunique()
    
    print(f"\n   Subjects: {n_subjects} total (HC={n_hc}, PD={n_pd})")
    
    if prefix == 'kcl_condition_d_30s':
        expected = (17, 12, 29)
        if (n_hc, n_pd, n_subjects) != expected:
            print(f"❌ Expected HC=17, PD=12, Total=29, got HC={n_hc}, PD={n_pd}, Total={n_subjects}")
            return False
        print("✅ KCL Condition D subject counts match expected (HC=17, PD=12)")
    
    return True

def run_step2(prefix, kcl_mode=False):
    """Run Step 2 and verify results"""
    print("\n" + "="*70)
    print("🔧 TEST 3: Step 2 Experiments")
    print("="*70)
    
    cmd = ["python", "11_step2_experiments.py", 
           "--run_name", f"{prefix}_{TEST_RUN_NAME}"]
    
    if kcl_mode:
        cmd.append("--kcl_30s")
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Step 2 FAILED")
        print(result.stderr)
        return False
    
    print("✅ Step 2 completed")
    
    # Load and verify results
    results_file = RESULTS_DIR / f"{prefix}_{TEST_RUN_NAME}_results.json"
    hash_file = RESULTS_DIR / f"{prefix}_{TEST_RUN_NAME}_hash.txt"
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return False
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Verify hash
    with open(hash_file, 'r') as f:
        stored_hash = f.readline().split(': ')[1].strip()
    
    content = json.dumps(results, sort_keys=True).encode()
    computed_hash = hashlib.sha256(content).hexdigest()[:16]
    
    if stored_hash != computed_hash:
        print(f"❌ Hash mismatch: stored={stored_hash}, computed={computed_hash}")
        return False
    
    print("✅ Results hash verified")
    
    # Print summary
    print("\n📊 Results Summary:")
    for r in results['pd_results']:
        cohort = r['cohort'] if r['cohort'] else "ALL"
        if cohort != "ALL" or len(results['pd_results']) == 1:
            print(f"   {cohort}: AUROC={r['auroc']:.3f}, N={r['n_subjects_evaluated']}")
    
    return True

def clean_features():
    """Delete all feature files for clean test"""
    print("\n🧹 Cleaning existing feature files...")
    count = 0
    for f in FEATURES_DIR.glob("kcl_condition_d_30s_*"):
        f.unlink()
        count += 1
    for f in FEATURES_DIR.glob("cross_lingual_real_*"):
        f.unlink()
        count += 1
    print(f"   Deleted {count} files")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test Step 1 and Step 2 alignment')
    parser.add_argument('--clean', action='store_true', help='Delete existing feature files before testing')
    parser.add_argument('--kcl_30s', action='store_true', help='Test KCL mode only (skip full cross-lingual)')
    args = parser.parse_args()
    
    if args.clean:
        clean_features()
    
    tests_passed = 0
    tests_failed = 0
    
    # Test KCL 30s mode
    print("\n" + "="*80)
    print("🧪 TESTING KCL CONDITION D MODE")
    print("="*80)
    
    if run_step1_kcl():
        if verify_step1_outputs('kcl_condition_d_30s'):
            tests_passed += 1
            if run_step2('kcl_condition_d_30s', kcl_mode=True):
                tests_passed += 1
            else:
                tests_failed += 1
        else:
            tests_failed += 1
    else:
        tests_failed += 1
    
    # Test full cross-lingual mode (if not just KCL mode)
    if not args.kcl_30s:
        print("\n" + "="*80)
        print("🧪 TESTING FULL CROSS-LINGUAL MODE")
        print("="*80)
        
        if run_step1_full():
            if verify_step1_outputs('cross_lingual_real'):
                tests_passed += 1
                if run_step2('cross_lingual_real', kcl_mode=False):
                    tests_passed += 1
                else:
                    tests_failed += 1
            else:
                tests_failed += 1
        else:
            tests_failed += 1
    
    # Summary
    print("\n" + "="*80)
    print("📋 TEST SUMMARY")
    print("="*80)
    print(f"Tests passed: {tests_passed}")
    print(f"Tests failed: {tests_failed}")
    
    if tests_failed == 0:
        print("\n✅✅✅ PIPELINE VERIFIED - All tests passed!")
        print("   Step 1 and Step 2 are correctly aligned.")
        return 0
    else:
        print("\n❌❌❌ PIPELINE FAILED - Check errors above.")
        return 1

if __name__ == "__main__":

    sys.exit(main())
