#!/usr/bin/env python3
"""
STEP 6G: YOUDEN'S J THRESHOLD SWEEP - CONDITION D (PRIMARY)
================================================================================
PRIMARY DATASET: 30s Center Window + Speech Boundary Alignment
QC: PASS only, Subject-level aggregation (1 vote per subject)
CLINICAL DECISION DATE: 2026-02-12 (END-TO-END TEST CONFIRMATION)
================================================================================

╔════════════════════════════════════════════════════════════════════════════════════════╗
║  VALIDATION STATUS - END-TO-END TEST 2026-02-12                                       ║
║  ✓ Using latest audited datasets (2026-02-12)                                         ║
║  ✓ Dataset integrity verified via SHA256 hashes                                       ║
║  ✓ Subject-level aggregation (no pseudoreplication)                                   ║
║  ✓ QC PASS only (verified measurements)                                               ║
║  ✓ Primary dataset only (Condition D, not legacy DurMatch)                           ║
║  ✓ Boundary thresholds use IEEE 754 nextafter (not arbitrary ±0.001)                 ║
║  ✓ Unique thresholds enforced via np.unique()                                        ║
║  ✓ Ties at max Youden's J handled via EXACT INTEGER MATH (no float fragility)       ║
║  ✓ Full ID tracking for TN, FP, TP, FN in summary CSV                               ║
║  ✓ Clinical policy threshold computed dynamically from data                          ║
║  ✓ Consistent ceiling rule applied to all reported thresholds                        ║
║  ⚠ External validation required before clinical deployment                            ║
╚════════════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Compute Youden's J statistic across all possible thresholds on the
    PRIMARY Condition D dataset to identify the optimal operating point.
    This is a DESCRIPTIVE STATISTIC only — not a clinical policy decision.
    
DATASETS (LOCKED - 2026-02-12 END-TO-END TEST):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HC: HC_NorthWind_PD-Likeness_30sCenter_20260212_100148.csv
  - SHA256: 33db8c2855a6bb66b1e6d2fa7da839613e5042a0d8b5cb552da5ae133a8be86d
  - 17 SUCCESS subjects (QC_Pass=True)
  - Top scores: 0.6195 (ID14), 0.4951 (ID09), 0.4658 (ID22)

PD: PD_NorthWind_PD-Likeness_30sCenter_20260212_101240.csv
  - SHA256: de8f6756efb38ed1ad4c3ac301511b1eed425095c718afde5797098475683419
  - 12 SUCCESS subjects (QC_Pass=True)
  - Top scores: 0.9842 (ID32), 0.9487 (ID13), 0.8765 (ID20)

CLINICAL POLICY REFERENCE (from Step 6F - LOCKED):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Operational threshold: T = 0.495101 (used in code, ε = 1e-6)
  • Reported threshold:    T = 0.4952 (conservative CEILING to 4dp)
  • Policy basis:          ≤1 False Positive in HC (risk tolerance)
  • False Positives:       1 (ID14: 0.6195)
  • True Negatives:        16
  • True Positives:        6 (ID06, ID13, ID20, ID27, ID29, ID32)
  • False Negatives:       6 (ID02, ID07, ID17, ID24, ID33, ID34)
  • Specificity:          94.12% (16/17)
  • Sensitivity:          50.00% (6/12)
  • Youden's J:           0.4412

NOTE ON YOUDEN OPTIMAL:
    The exact optimal threshold range is DETERMINED DYNAMICALLY from the sweep
    using EXACT INTEGER ARITHMETIC on the Youden numerator. No float equality
    is used for tie detection. Expected confusion matrix for the locked dataset:
    FP=4, TP=9, FN=3, Specificity=76.47%, Sensitivity=75.00%, Youden's J=0.5147.
================================================================================
"""

import numpy as np
import pandas as pd
import hashlib
import math
from pathlib import Path
from datetime import datetime

# =============================================================================
# PRIMARY DATA - CONDITION D (LOCKED - 2026-02-12 END-TO-END TEST)
# =============================================================================
HC_CSV = Path(r"C:\Projects\hear_italian\audit_results\HC_NorthWind_PD-Likeness_30sCenter_20260212_100148.csv")
PD_CSV = Path(r"C:\Projects\hear_italian\audit_results\PD_NorthWind_PD-Likeness_30sCenter_20260212_101240.csv")

# =============================================================================
# DATASET INTEGRITY - LOCKED REFERENCE HASHES (2026-02-12)
# =============================================================================
EXPECTED_HC_HASH = "33db8c2855a6bb66b1e6d2fa7da839613e5042a0d8b5cb552da5ae133a8be86d"
EXPECTED_PD_HASH = "de8f6756efb38ed1ad4c3ac301511b1eed425095c718afde5797098475683419"

# =============================================================================
# CLINICAL POLICY REFERENCE - FROM STEP 6F (LOCKED)
# =============================================================================
CLINICAL_EPSILON = 1e-6
# CLINICAL_TARGET_FP retained for documentation - used in Step 6F policy
CLINICAL_TARGET_FP = 1  # Documented policy constraint
CLINICAL_HC_EXPECTED = 17
CLINICAL_PD_EXPECTED = 12

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================
OUTPUT_DIR = Path(r"C:\Projects\hear_italian\audit_results\threshold_sweep")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# COLUMN MAPPINGS - CONDITION D
# =============================================================================
SCORE_COL = "PD_Likeness_Score"
QC_COL = "QC_Pass"
ID_COL = "SubjectID"

# =============================================================================
# UTILITY: CONSERVATIVE CEILING TO 4 DECIMAL PLACES
# =============================================================================
def ceiling_to_4dp(x: float) -> float:
    """Apply conservative ceiling rounding to 4 decimal places.
       Used for all reported thresholds to ensure documented threshold
       is never lower than operational threshold."""
    return math.ceil(x * 10_000) / 10_000

# =============================================================================
# DATASET INTEGRITY VERIFICATION - TAMPER-PROOF LOCK
# =============================================================================
def verify_dataset_integrity():
    """CRITICAL: Ensures we're using the exact locked datasets."""
    
    def file_hash(path):
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    print("\n" + "-" * 80)
    print("🔐 DATASET INTEGRITY VERIFICATION")
    print("-" * 80)
    
    # Verify HC dataset
    hc_hash = file_hash(HC_CSV)
    hc_valid = (hc_hash == EXPECTED_HC_HASH)
    print(f"\nHC Dataset: {HC_CSV.name}")
    print(f"  Expected: {EXPECTED_HC_HASH}")
    print(f"  Actual:   {hc_hash}")
    print(f"  Status:   {'✅ VERIFIED' if hc_valid else '❌ MODIFIED'}")
    
    # Verify PD dataset
    pd_hash = file_hash(PD_CSV)
    pd_valid = (pd_hash == EXPECTED_PD_HASH)
    print(f"\nPD Dataset: {PD_CSV.name}")
    print(f"  Expected: {EXPECTED_PD_HASH}")
    print(f"  Actual:   {pd_hash}")
    print(f"  Status:   {'✅ VERIFIED' if pd_valid else '❌ MODIFIED'}")
    
    # Assert both datasets are unmodified
    assert hc_valid, f"\n❌ HC DATASET MODIFIED!\nFile: {HC_CSV.name}"
    assert pd_valid, f"\n❌ PD DATASET MODIFIED!\nFile: {PD_CSV.name}"
    
    print("\n✅ Dataset integrity verified - using locked clinical reference files")
    return True

# =============================================================================
# LOAD AND AGGREGATE - SUBJECT LEVEL (CRITICAL)
# =============================================================================
def load_subject_level_scores(path: Path, cohort_name: str):
    """Load QC PASS, subject-level aggregated scores."""
    
    df = pd.read_csv(path)
    
    # QC PASS only
    assert QC_COL in df.columns, f"QC_COL '{QC_COL}' not found in {cohort_name}"
    df = df[df[QC_COL] == True].copy()
    
    # Verify ID column exists
    assert ID_COL in df.columns, f"ID_COL '{ID_COL}' not found in {cohort_name}"
    
    # CRITICAL: Subject-level aggregation (one vote per subject)
    # Take mean of multiple windows/files per subject
    subject_df = df.groupby(ID_COL, as_index=False).agg({
        SCORE_COL: "mean"
    }).rename(columns={SCORE_COL: "score"})
    
    # Drop NaN
    subject_df = subject_df.dropna(subset=["score"])
    
    # Sort by score descending for audit trail
    subject_df = subject_df.sort_values("score", ascending=False).reset_index(drop=True)
    
    print(f"    {cohort_name}: {len(subject_df)} subjects (from {len(df)} measurements)")
    
    # Return both scores and full dataframe for ID tracking
    return subject_df["score"].to_numpy(dtype=float), subject_df

# =============================================================================
# COMPUTE CLINICAL POLICY THRESHOLD DIRECTLY FROM DATA
# =============================================================================
def compute_clinical_policy_threshold(hc_df: pd.DataFrame):
    """
    Compute the clinical policy threshold directly from HC data.
    Rule: T = 2nd highest HC score + ε (ε = 1e-6)
    This matches Step 6F exactly and is reviewer-proof.
    """
    assert ID_COL in hc_df.columns, "ID_COL missing from HC dataframe"
    assert "score" in hc_df.columns, "score column missing from HC dataframe"
    
    hc_sorted_df = hc_df.sort_values("score", ascending=False).reset_index(drop=True)
    
    # Second highest score (index 1 when TARGET_FP=1)
    threshold_idx = 1
    threshold_row = hc_sorted_df.loc[threshold_idx]
    
    raw_score = threshold_row["score"]
    subject_id = threshold_row[ID_COL]
    
    # Operational threshold (used in code)
    T_operational = raw_score + CLINICAL_EPSILON
    
    # Reported threshold (conservative ceiling to 4dp)
    T_reported = ceiling_to_4dp(T_operational)
    
    # Verify we got the expected values from locked dataset
    assert subject_id == "ID09", f"Expected ID09 at 2nd highest score, got {subject_id}"
    assert np.isclose(raw_score, 0.4951, rtol=0, atol=1e-6), f"Expected 0.4951, got {raw_score}"
    
    return {
        "raw_score": raw_score,
        "subject_id": subject_id,
        "T_operational": T_operational,
        "T_reported": T_reported,
        "threshold_idx": threshold_idx
    }

# =============================================================================
# EVALUATE THRESHOLD WITH FULL ID TRACKING
# =============================================================================
def evaluate_threshold_with_ids(hc_df: pd.DataFrame, pd_df: pd.DataFrame, threshold: float):
    """Evaluate a threshold and return counts plus subject IDs."""
    
    hc_positive = hc_df[hc_df["score"] >= threshold].copy()
    hc_negative = hc_df[hc_df["score"] < threshold].copy()
    
    pd_positive = pd_df[pd_df["score"] >= threshold].copy()
    pd_negative = pd_df[pd_df["score"] < threshold].copy()
    
    return {
        "threshold": threshold,
        "hc_fp": len(hc_positive),
        "hc_tn": len(hc_negative),
        "hc_fp_ids": sorted(hc_positive[ID_COL].tolist()),
        "hc_tn_ids": sorted(hc_negative[ID_COL].tolist()),
        "pd_tp": len(pd_positive),
        "pd_fn": len(pd_negative),
        "pd_tp_ids": sorted(pd_positive[ID_COL].tolist()),
        "pd_fn_ids": sorted(pd_negative[ID_COL].tolist()),
        "specificity": len(hc_negative) / (len(hc_positive) + len(hc_negative)),
        "sensitivity": len(pd_positive) / (len(pd_positive) + len(pd_negative))
    }

# =============================================================================
# YOUDEN SWEEP - AUDIT-GRADE BOUNDARIES WITH EXACT INTEGER MATH
# =============================================================================
def youden_sweep(hc_scores: np.ndarray, pd_scores: np.ndarray, 
                 hc_df: pd.DataFrame, pd_df: pd.DataFrame):
    """
    Compute Youden's J for all unique thresholds.
    Uses EXACT INTEGER numerator for tie detection - no float equality issues.
    
    Youden's J = sensitivity + specificity - 1
               = tp/N_pd + tn/N_hc - 1
               = (tp * N_hc + tn * N_pd - N_hc * N_pd) / (N_hc * N_pd)
    
    j_num = tp * N_hc + tn * N_pd - N_hc * N_pd  (integer, exact)
    j_den = N_hc * N_pd                          (integer, constant)
    """
    N_hc = len(hc_df)
    N_pd = len(pd_df)
    j_den = N_hc * N_pd
    
    # Combine all unique scores for threshold candidates
    all_scores = np.concatenate([hc_scores, pd_scores])
    thresholds = np.unique(all_scores)
    
    # AUDIT-GRADE BOUNDARIES: Use nextafter for exact edge thresholds
    min_score = thresholds[0]
    max_score = thresholds[-1]
    
    # Create thresholds just below min and just above max
    boundary_below = np.nextafter(min_score, -np.inf)  # Smallest float < min_score
    boundary_above = np.nextafter(max_score, np.inf)   # Largest float > max_score
    
    # Add midpoints between consecutive scores for complete sweep
    midpoints = (thresholds[:-1] + thresholds[1:]) / 2
    
    # Complete threshold set
    all_thresholds = np.concatenate([
        [boundary_below],           # Below minimum: all scores ≥ T → FP=N, TP=N
        thresholds,                 # All observed scores
        midpoints,                 # Between-score thresholds
        [boundary_above]           # Above maximum: no scores ≥ T → FP=0, TP=0
    ])
    
    # CRITICAL: Remove duplicates and sort
    all_thresholds = np.unique(all_thresholds)
    all_thresholds = np.sort(all_thresholds)
    
    results = []
    
    for t in all_thresholds:
        eval_result = evaluate_threshold_with_ids(hc_df, pd_df, t)
        
        # EXACT INTEGER YOUDEN NUMERATOR - NO FLOAT TIE ISSUES
        tp = eval_result["pd_tp"]
        tn = eval_result["hc_tn"]
        j_num = tp * N_hc + tn * N_pd - j_den
        youden_j = j_num / j_den if j_den != 0 else np.nan
        
        results.append({
            "threshold": t,
            "specificity_hc": eval_result["specificity"],
            "sensitivity_pd": eval_result["sensitivity"],
            "youden_j": youden_j,
            "j_num": j_num,           # EXACT INTEGER - use for max/tie detection
            "j_den": j_den,           # Constant denominator (included for completeness)
            "hc_fp": eval_result["hc_fp"],
            "hc_tn": eval_result["hc_tn"],
            "pd_tp": eval_result["pd_tp"],
            "pd_fn": eval_result["pd_fn"],
        })
    
    return pd.DataFrame(results)

# =============================================================================
# PROCESS YOUDEN OPTIMAL - EXACT INTEGER TIE DETECTION
# =============================================================================
def process_youden_optimal(df: pd.DataFrame, hc_df: pd.DataFrame, pd_df: pd.DataFrame):
    """
    Find max Youden's J using EXACT INTEGER numerator.
    Ties are detected via integer equality - no float precision issues.
    """
    N_hc = len(hc_df)
    N_pd = len(pd_df)
    j_den = N_hc * N_pd
    
    # Find maximum integer numerator (EXACT)
    max_j_num = df["j_num"].max()
    
    # All thresholds achieving the exact maximum numerator
    tied_df = df[df["j_num"] == max_j_num].copy()
    
    t_low = tied_df["threshold"].min()
    t_high = tied_df["threshold"].max()
    
    # Deterministic choice: use the LOWEST threshold in the tied set
    # This yields the most sensitive option within the optimal range
    representative_threshold = t_low
    
    # Get full evaluation with IDs for the representative threshold
    representative_eval = evaluate_threshold_with_ids(hc_df, pd_df, representative_threshold)
    
    # Calculate Youden J from integer components
    tp = representative_eval["pd_tp"]
    tn = representative_eval["hc_tn"]
    j_num_rep = tp * N_hc + tn * N_pd - j_den
    youden_j_rep = j_num_rep / j_den
    
    return {
        "max_j_num": max_j_num,
        "j_den": j_den,
        "max_j": max_j_num / j_den,
        "tied_count": len(tied_df),
        "t_low": t_low,
        "t_high": t_high,
        "representative_threshold": representative_threshold,
        "representative_reported": ceiling_to_4dp(representative_threshold),
        "representative_eval": representative_eval,
        "representative_youden_j": youden_j_rep,
        "tied_df": tied_df
    }

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "=" * 80)
    print("📊 YOUDEN'S J THRESHOLD SWEEP - CONDITION D (PRIMARY)")
    print("=" * 80)
    print("📅 Clinical Decision Date: 2026-02-12 (END-TO-END TEST)")
    print("🔬 Purpose: Descriptive statistics only - not a policy decision")
    print("🧮 Tie Detection: EXACT INTEGER MATH (no float equality)")
    
    # =========================================================================
    # VERIFY DATASET INTEGRITY
    # =========================================================================
    verify_dataset_integrity()
    
    # =========================================================================
    # LOAD DATA WITH FULL DATAFRAMES FOR ID TRACKING
    # =========================================================================
    print("\n" + "-" * 80)
    print("📂 LOADING CONDITION D DATASETS - 2026-02-12 END-TO-END TEST")
    print("-" * 80)
    
    hc_scores, hc_df = load_subject_level_scores(HC_CSV, "HC")
    pd_scores, pd_df = load_subject_level_scores(PD_CSV, "PD")
    
    N_hc = len(hc_df)
    N_pd = len(pd_df)
    
    # Verify expected subject counts
    assert N_hc == CLINICAL_HC_EXPECTED, f"Expected {CLINICAL_HC_EXPECTED} HC subjects, got {N_hc}"
    assert N_pd == CLINICAL_PD_EXPECTED, f"Expected {CLINICAL_PD_EXPECTED} PD subjects, got {N_pd}"
    
    print(f"\n✅ Subject counts verified: HC={N_hc}, PD={N_pd}")
    print(f"\n📊 HC score range: [{hc_scores.min():.4f}, {hc_scores.max():.4f}]")
    print(f"📊 PD score range: [{pd_scores.min():.4f}, {pd_scores.max():.4f}]")
    
    # Print top HC subjects for audit trail
    print(f"\n📋 HC TOP SUBJECTS (for audit):")
    for i in range(min(3, len(hc_df))):
        row = hc_df.iloc[i]
        print(f"   {i+1}. {row[ID_COL]}: {row['score']:.4f}")
    
    # Print top PD subjects for audit trail
    print(f"\n📋 PD TOP SUBJECTS (for audit):")
    for i in range(min(3, len(pd_df))):
        row = pd_df.iloc[i]
        print(f"   {i+1}. {row[ID_COL]}: {row['score']:.4f}")
    
    # =========================================================================
    # COMPUTE CLINICAL POLICY THRESHOLD DIRECTLY FROM DATA
    # =========================================================================
    print("\n" + "-" * 80)
    print("🔐 COMPUTING CLINICAL POLICY THRESHOLD (STEP 6F RULE)")
    print("-" * 80)
    
    clinical_threshold_info = compute_clinical_policy_threshold(hc_df)
    
    print(f"\n   Mathematical Rule: T = HC_sorted[1] + ε")
    print(f"   HC_sorted[1] = {clinical_threshold_info['raw_score']:.4f} (ID: {clinical_threshold_info['subject_id']})")
    print(f"   ε = {CLINICAL_EPSILON}")
    print(f"\n✅ OPERATIONAL THRESHOLD: T = {clinical_threshold_info['T_operational']:.6f}")
    print(f"✅ REPORTED THRESHOLD:    T = {clinical_threshold_info['T_reported']:.4f} (conservative CEILING to 4dp)")
    
    # =========================================================================
    # EVALUATE CLINICAL POLICY PERFORMANCE
    # =========================================================================
    clinical_eval = evaluate_threshold_with_ids(
        hc_df, pd_df, clinical_threshold_info['T_operational']
    )
    
    # Calculate Youden's J using integer math
    tp_clin = clinical_eval["pd_tp"]
    tn_clin = clinical_eval["hc_tn"]
    j_num_clin = tp_clin * N_hc + tn_clin * N_pd - N_hc * N_pd
    j_den = N_hc * N_pd
    clinical_youden = j_num_clin / j_den
    
    print(f"\n📊 CLINICAL POLICY PERFORMANCE:")
    print(f"   HC: FP={clinical_eval['hc_fp']} ({clinical_eval['hc_fp_ids']})")
    print(f"   HC: TN={clinical_eval['hc_tn']} ({clinical_eval['hc_tn_ids']})")
    print(f"   PD: TP={clinical_eval['pd_tp']} ({clinical_eval['pd_tp_ids']})")
    print(f"   PD: FN={clinical_eval['pd_fn']} ({clinical_eval['pd_fn_ids']})")
    print(f"   Specificity: {clinical_eval['specificity']:.2%}")
    print(f"   Sensitivity: {clinical_eval['sensitivity']:.2%}")
    print(f"   Youden's J:  {clinical_youden:.4f} (exact integer numerator: {j_num_clin})")
    
    # Verify against expected locked values
    assert clinical_eval['hc_fp'] == 1, f"Expected 1 FP, got {clinical_eval['hc_fp']}"
    assert clinical_eval['hc_fp_ids'] == ["ID14"], f"Expected FP ID14, got {clinical_eval['hc_fp_ids']}"
    assert clinical_eval['pd_tp'] == 6, f"Expected 6 TP, got {clinical_eval['pd_tp']}"
    assert clinical_eval['pd_fn'] == 6, f"Expected 6 FN, got {clinical_eval['pd_fn']}"
    
    # =========================================================================
    # COMPUTE YOUDEN SWEEP
    # =========================================================================
    print("\n" + "-" * 80)
    print("📊 COMPUTING YOUDEN'S J ACROSS ALL THRESHOLDS")
    print("-" * 80)
    
    df = youden_sweep(hc_scores, pd_scores, hc_df, pd_df)
    print(f"\n✅ Sweep complete: {len(df)} UNIQUE thresholds evaluated")
    print(f"   Youden denominator (constant): {j_den}")
    
    # =========================================================================
    # PROCESS YOUDEN OPTIMAL WITH EXACT INTEGER TIE HANDLING
    # =========================================================================
    print("\n" + "-" * 80)
    print("🎯 YOUDEN'S J OPTIMAL - EXACT INTEGER TIE ANALYSIS")
    print("-" * 80)
    
    optimal_info = process_youden_optimal(df, hc_df, pd_df)
    
    print(f"\n   Maximum Youden's J:        {optimal_info['max_j']:.4f}")
    print(f"   Max integer numerator:     {optimal_info['max_j_num']}")
    print(f"   Number of tied thresholds: {optimal_info['tied_count']} (detected via exact integer match)")
    print(f"\n   TIED THRESHOLD RANGE:")
    print(f"     • Lowest:  {optimal_info['t_low']:.6f} (reported: {ceiling_to_4dp(optimal_info['t_low']):.4f})")
    print(f"     • Highest: {optimal_info['t_high']:.6f} (reported: {ceiling_to_4dp(optimal_info['t_high']):.4f})")
    
    print(f"\n   REPRESENTATIVE THRESHOLD (lowest in tied set):")
    print(f"     • Operational: {optimal_info['representative_threshold']:.6f}")
    print(f"     • Reported:     {optimal_info['representative_reported']:.4f} (conservative CEILING)")
    
    rep_eval = optimal_info['representative_eval']
    print(f"\n   PERFORMANCE AT REPRESENTATIVE THRESHOLD:")
    print(f"     HC: FP={rep_eval['hc_fp']} ({rep_eval['hc_fp_ids']})")
    print(f"     HC: TN={rep_eval['hc_tn']} ({rep_eval['hc_tn_ids']})")
    print(f"     PD: TP={rep_eval['pd_tp']} ({rep_eval['pd_tp_ids']})")
    print(f"     PD: FN={rep_eval['pd_fn']} ({rep_eval['pd_fn_ids']})")
    print(f"     Specificity: {rep_eval['specificity']:.2%}")
    print(f"     Sensitivity: {rep_eval['sensitivity']:.2%}")
    print(f"     Youden's J:  {optimal_info['representative_youden_j']:.4f}")
    
    # Verify against expected confusion matrix for locked dataset
    # This is what clinically matters - the exact threshold values may vary at 1e-15
    assert rep_eval['hc_fp'] == 4, f"Expected 4 FP at optimal, got {rep_eval['hc_fp']}"
    assert rep_eval['pd_tp'] == 9, f"Expected 9 TP at optimal, got {rep_eval['pd_tp']}"
    assert rep_eval['pd_fn'] == 3, f"Expected 3 FN at optimal, got {rep_eval['pd_fn']}"
    assert np.isclose(rep_eval['specificity'], 13/17, rtol=0, atol=1e-10), "Specificity mismatch"
    assert np.isclose(rep_eval['sensitivity'], 9/12, rtol=0, atol=1e-10), "Sensitivity mismatch"
    
    # =========================================================================
    # COMPARISON TABLE
    # =========================================================================
    print("\n" + "-" * 80)
    print("🔐 COMPARISON: CLINICAL POLICY vs YOUDEN OPTIMAL")
    print("-" * 80)
    
    comparison_data = []
    
    # Clinical policy row
    comparison_data.append({
        "Threshold (Reported)": f"{clinical_threshold_info['T_reported']:.4f}",
        "Threshold (Operational)": f"{clinical_threshold_info['T_operational']:.6f}",
        "Specificity": f"{clinical_eval['specificity']:.2%}",
        "Sensitivity": f"{clinical_eval['sensitivity']:.2%}",
        "Youden's J": f"{clinical_youden:.4f}",
        "HC FP": str(clinical_eval['hc_fp']),
        "PD TP": str(clinical_eval['pd_tp']),
        "Notes": "Clinical policy (≤1 FP)"
    })
    
    # Youden optimal row (representative threshold)
    comparison_data.append({
        "Threshold (Reported)": f"{optimal_info['representative_reported']:.4f}",
        "Threshold (Operational)": f"{optimal_info['representative_threshold']:.6f}",
        "Specificity": f"{rep_eval['specificity']:.2%}",
        "Sensitivity": f"{rep_eval['sensitivity']:.2%}",
        "Youden's J": f"{optimal_info['representative_youden_j']:.4f}",
        "HC FP": str(rep_eval['hc_fp']),
        "PD TP": str(rep_eval['pd_tp']),
        "Notes": f"Youden optimal (tied range, n={optimal_info['tied_count']})"
    })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # =========================================================================
    # INTERPRETATION NOTE
    # =========================================================================
    print("\n" + "-" * 80)
    print("⚠️  INTERPRETATION NOTE - READ FOR AUDIT")
    print("-" * 80)
    print(f"""
    Youden's J identifies the statistically optimal trade-off between
    sensitivity and specificity. For the locked 2026-02-12 dataset,
    the maximum Youden's J = {optimal_info['max_j']:.4f} is achieved by
    {optimal_info['tied_count']} tied thresholds (detected via EXACT INTEGER 
    numerator matching, not float equality).
    
    The clinical policy threshold (T=0.4952) was deliberately chosen to 
    achieve ≤1 False Positive in the HC sample, prioritizing specificity 
    over statistical optimization. This is a POLICY DECISION, not an error.
    
    ┌─────────────────────┬─────────────────────────────────────────────┐
    │ Clinical Policy     │ Priority: Minimize false positives        │
    │                     │ Use case: Population screening           │
    │                     │ Constraint: ≤1 FP in 17 HC subjects      │
    ├─────────────────────┼─────────────────────────────────────────────┤
    │ Youden Optimal      │ Priority: Balance sensitivity/specificity │
    │                     │ Use case: Research / exploratory         │
    │                     │ Constraint: Maximize J statistic         │
    └─────────────────────┴─────────────────────────────────────────────┘
    
    The Youden-optimal threshold range is provided for descriptive purposes only.
    For operational reproducibility, the LOWEST threshold in the tied set
    ({optimal_info['representative_threshold']:.6f}) is used as the representative.
    This does not supersede the locked clinical policy decision.
    """)
    
    # =========================================================================
    # SAVE TO CSV
    # =========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Full sweep results (includes exact integer numerator for audit)
    out_csv = OUTPUT_DIR / f"06g_youden_j_threshold_sweep_30sCenter_{timestamp}.csv"
    df.to_csv(out_csv, index=False)
    
    # Summary with BOTH thresholds and COMPLETE ID tracking (audit trail)
    summary_rows = []
    
    # Clinical policy summary with FULL ID lists (TN, FP, TP, FN)
    summary_rows.append({
        "analysis": "clinical_policy",
        "threshold_operational": clinical_threshold_info['T_operational'],
        "threshold_reported": clinical_threshold_info['T_reported'],
        "youden_j": clinical_youden,
        "youden_j_num": j_num_clin,
        "youden_j_den": j_den,
        "specificity": clinical_eval['specificity'],
        "sensitivity": clinical_eval['sensitivity'],
        "hc_fp": clinical_eval['hc_fp'],
        "hc_tn": clinical_eval['hc_tn'],
        "pd_tp": clinical_eval['pd_tp'],
        "pd_fn": clinical_eval['pd_fn'],
        "hc_fp_ids": str(clinical_eval['hc_fp_ids']),
        "hc_tn_ids": str(clinical_eval['hc_tn_ids']),  # COMPLETE: 16 IDs
        "pd_tp_ids": str(clinical_eval['pd_tp_ids']),
        "pd_fn_ids": str(clinical_eval['pd_fn_ids'])
    })
    
    # Youden optimal summary with FULL ID lists (representative threshold)
    summary_rows.append({
        "analysis": "youden_optimal",
        "threshold_operational": optimal_info['representative_threshold'],
        "threshold_reported": optimal_info['representative_reported'],
        "youden_j": optimal_info['representative_youden_j'],
        "youden_j_num": optimal_info['max_j_num'],
        "youden_j_den": optimal_info['j_den'],
        "specificity": rep_eval['specificity'],
        "sensitivity": rep_eval['sensitivity'],
        "hc_fp": rep_eval['hc_fp'],
        "hc_tn": rep_eval['hc_tn'],
        "pd_tp": rep_eval['pd_tp'],
        "pd_fn": rep_eval['pd_fn'],
        "hc_fp_ids": str(rep_eval['hc_fp_ids']),
        "hc_tn_ids": str(rep_eval['hc_tn_ids']),  # COMPLETE: 13 IDs
        "pd_tp_ids": str(rep_eval['pd_tp_ids']),
        "pd_fn_ids": str(rep_eval['pd_fn_ids']),
        "tied_range_low": optimal_info['t_low'],
        "tied_range_high": optimal_info['t_high'],
        "tied_count": optimal_info['tied_count']
    })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = OUTPUT_DIR / f"06g_youden_j_summary_30sCenter_{timestamp}.csv"
    summary_df.to_csv(summary_csv, index=False)
    
    print(f"\n💾 Sweep results saved to:  {out_csv.name}")
    print(f"💾 Summary saved to:        {summary_csv.name}")
    print(f"\n📁 Output directory:        {OUTPUT_DIR}")
    
    # =========================================================================
    # AUDIT CERTIFICATE
    # =========================================================================
    print("\n" + "=" * 80)
    print("🔐 AUDIT CERTIFICATE - YOUDEN'S J SWEEP")
    print("=" * 80)
    print(f"""
✓ Analysis Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
✓ Clinical Reference:  2026-02-12 End-to-End Test

✓ DATASET INTEGRITY:
  • HC: {HC_CSV.name}
    SHA256: {EXPECTED_HC_HASH}
    Subjects: {N_hc} (✓ verified)
    Score range: [{hc_scores.min():.4f}, {hc_scores.max():.4f}]
  
  • PD: {PD_CSV.name}
    SHA256: {EXPECTED_PD_HASH}
    Subjects: {N_pd} (✓ verified)
    Score range: [{pd_scores.min():.4f}, {pd_scores.max():.4f}]

✓ THRESHOLD GRID:
  • Unique thresholds evaluated: {len(df)}
  • Boundary method: IEEE 754 nextafter (exact)
  • Duplicate removal: ✓ enforced via np.unique()
  • Midpoints: ✓ included for complete sweep

✓ YOUDEN OPTIMAL - EXACT INTEGER MATH:
  • Maximum J numerator: {optimal_info['max_j_num']} (constant denominator: {j_den})
  • Maximum J: {optimal_info['max_j']:.4f}
  • Tied thresholds: {optimal_info['tied_count']} (detected via EXACT integer equality)
  • Tied range: [{optimal_info['t_low']:.6f}, {optimal_info['t_high']:.6f}]
  • Representative: {optimal_info['representative_threshold']:.6f} (lowest)
  • Representative reported: {optimal_info['representative_reported']:.4f} (CEILING)
  • Performance: FP={rep_eval['hc_fp']}, TP={rep_eval['pd_tp']}, FN={rep_eval['pd_fn']} ✓ verified

✓ CLINICAL POLICY (LOCKED - STEP 6F):
  • Rule: T = 2nd highest HC + ε
  • 2nd highest: {clinical_threshold_info['raw_score']:.4f} (ID: {clinical_threshold_info['subject_id']})
  • Operational T: {clinical_threshold_info['T_operational']:.6f}
  • Reported T: {clinical_threshold_info['T_reported']:.4f} (CEILING)
  • Performance: FP=1 (ID14), TP=6, FN=6, J={clinical_youden:.4f}
  • Full ID tracking: ✓ saved to summary CSV

✓ STATUS: AUDIT PASSED - Descriptive statistics complete
  • Tie detection: EXACT INTEGER MATH (no float equality)
  • Threshold reporting: CONSISTENT CEILING rule throughout
  • ID tracking: COMPLETE (TN/FP/TP/FN) for both thresholds
  • Clinical policy: UNCHANGED, not superseded by this analysis
    """)
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()