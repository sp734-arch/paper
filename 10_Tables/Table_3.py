#!/usr/bin/env python3
"""
TABLE 3: Clinical Threshold Policy
================================================================================
PURPOSE:
    Generates Table 3 for the manuscript showing the selected clinical threshold
    (T=0.4952) compared to Youden optimum and zero-FP alternative.
    Output format matches the LaTeX table in the manuscript.

DEPENDENCY CHAIN (CRITICAL FOR REPRODUCIBILITY):
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║  THIS SCRIPT FORMATS RESULTS FROM TWO SOURCES:                          ║
    ║  1. Youden optimum from threshold sweep (09_youden_j_peak_plot)         ║
    ║  2. Clinical policy from calibration script (06d_calibrate_tier1)       ║
    ║                                                                          ║
    ║  The actual threshold calibration is performed by:                      ║
    ║    06d_calibrate_tier1_thresholds_30sCenter.py                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝

DATA FLOW:
    Audit CSVs ────────────────────────────────┐
        ↓                                      │
    Step 6F (calibration script) ──────────────┼───┐
        ↓                                      │   │
    Console output with FP/TP counts ──────────┘   │
        ↓                                          │
    THIS SCRIPT ───────────────────────────────────┘
        ↓
    Console output matching LaTeX table format

SOURCE CODE REFERENCE:
    - Script: 06d_calibrate_tier1_thresholds_30sCenter.py (v1.4)
    - Script: 09_youden_j_peak_plot_30sCenter.py
================================================================================
"""

# =============================================================================
# LOCKED VALUES FROM CALIBRATION (DO NOT MODIFY)
# =============================================================================
# Source: 06d_calibrate_tier1_thresholds_30sCenter.py v1.4 (2026-02-12)

YOUden = {
    'threshold': 0.4395,
    'specificity': 76.47,  # 13/17
    'hc_fp': 4,
    'sensitivity': 75.00,  # 9/12
    'pd_tp': 9
}

CLINICAL = {
    'threshold': 0.4952,  # Reported (operational=0.495101)
    'specificity': 94.12,  # 16/17
    'hc_fp': 1,
    'sensitivity': 50.00,  # 6/12
    'pd_tp': 6
}

ZERO_FP = {
    'threshold': 0.6195,  # Max HC score (ID14)
    'specificity': 100.00,  # 17/17
    'hc_fp': 0,
    'sensitivity': 25.00,  # 3/12
    'pd_tp': 3
}

HC_N = 17
PD_N = 12

# =============================================================================
# GENERATE TABLE EXACTLY MATCHING LATEX FORMAT
# =============================================================================
print("\n" + "="*80)
print("TABLE 3: Clinical Threshold Policy (LaTeX Format)")
print("="*80)
print("Source: 06d_calibrate_tier1_thresholds_30sCenter.py v1.4 (2026-02-12)")
print("="*80)

print("\nPolicy & Threshold & Specificity & HC FP & Sensitivity & PD TP \\\\")
print("-"*80)
print(f"Youden Optimum & {YOUden['threshold']:.4f} & {YOUden['specificity']:.2f}\\% ({HC_N-YOUden['hc_fp']}/{HC_N}) & {YOUden['hc_fp']} & {YOUden['sensitivity']:.2f}\\% ({YOUden['pd_tp']}/{PD_N}) & {YOUden['pd_tp']} \\\\")
print(f"\\textbf{{Clinical Policy (Selected)}} & \\textbf{{{CLINICAL['threshold']:.4f}}} & \\textbf{{{CLINICAL['specificity']:.2f}\\% ({HC_N-CLINICAL['hc_fp']}/{HC_N})}} & \\textbf{{{CLINICAL['hc_fp']}}} & \\textbf{{{CLINICAL['sensitivity']:.2f}\\% ({CLINICAL['pd_tp']}/{PD_N})}} & \\textbf{{{CLINICAL['pd_tp']}}} \\\\")
print(f"Zero-FP Alternative & {ZERO_FP['threshold']:.4f} & {ZERO_FP['specificity']:.2f}\\% ({HC_N-ZERO_FP['hc_fp']}/{HC_N}) & {ZERO_FP['hc_fp']} & {ZERO_FP['sensitivity']:.2f}\\% ({ZERO_FP['pd_tp']}/{PD_N}) & {ZERO_FP['pd_tp']} \\\\")