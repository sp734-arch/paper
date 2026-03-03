#!/usr/bin/env python3
"""
TABLE 2: Primary Validation Results - PD vs HC Classification
================================================================================
PURPOSE:
    Generates Table 2 for the manuscript showing cohort statistics and 
    classification metrics for Condition D with -55 dBFS threshold.
    Output format matches the LaTeX table in the manuscript.

DEPENDENCY CHAIN (CRITICAL FOR REPRODUCIBILITY):
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║  THIS SCRIPT COMBINES TWO DATA SOURCES:                                 ║
    ║  1. Cohort statistics from raw audit CSVs (Table 2a)                    ║
    ║  2. Classification metrics from Step 2 LOSO (Table 2b)                  ║
    ║                                                                          ║
    ║  The actual PD classification experiments are run by:                   ║
    ║    11_step2_experiments.py --kcl_30s --run_name kcl_30s_analysis        ║
    ╚══════════════════════════════════════════════════════════════════════════╝

DATA FLOW:
    Audit CSVs (HC/PD) ──────────────────────┐
        ↓                                    │
    Cohort statistics (this script)          │
        ↓                                    │
    Step 2 (11_step2_experiments.py) ────────┼───┐
        ↓                                    │   │
    kcl_30s_analysis_results.json ───────────┘   │
        ↓                                        │
    THIS SCRIPT ─────────────────────────────────┘
        ↓
    Console output matching LaTeX table format

INPUTS:
    1. Audit CSVs: HC_NorthWind_PD-Likeness_30sCenter_20260212_100148.csv
                    PD_NorthWind_PD-Likeness_30sCenter_20260212_101240.csv
    2. Step 2 results: kcl_30s_analysis_results.json

OUTPUT:
    Console output formatted exactly like the LaTeX table in the manuscript

SOURCE CODE REFERENCE:
    - Class:      Step2Experiments (in 11_step2_experiments.py)
    - Method:     run_pd_loso_subject_level(cohort='KCL')
    - Task label: 'pd_loso_subject_level' in results JSON
================================================================================
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
AUDIT_DIR = Path(r'C:\Projects\hear_italian\audit_results')
HC_AUDIT = AUDIT_DIR / 'HC_NorthWind_PD-Likeness_30sCenter_20260212_100148.csv'
PD_AUDIT = AUDIT_DIR / 'PD_NorthWind_PD-Likeness_30sCenter_20260212_101240.csv'
RESULTS_FILE = Path(r'C:\Projects\hear_italian\step2_results\kcl_30s_analysis_results.json')

# =============================================================================
# LOAD AND COMPUTE COHORT STATISTICS
# =============================================================================
hc_df = pd.read_csv(HC_AUDIT)
pd_df = pd.read_csv(PD_AUDIT)

# Filter to SUCCESS only
hc_df = hc_df[hc_df['QC_Pass'] == True]
pd_df = pd_df[pd_df['QC_Pass'] == True]

# Subject-level scores (mean across slices)
hc_scores = hc_df.groupby('SubjectID')['PD_Likeness_Score'].mean().values
pd_scores = pd_df.groupby('SubjectID')['PD_Likeness_Score'].mean().values

def calc_stats(scores):
    return {
        'n': len(scores),
        'mean': np.mean(scores),
        'std': np.std(scores, ddof=1),
        'median': np.median(scores),
        'q1': np.percentile(scores, 25),
        'q3': np.percentile(scores, 75),
        'min': np.min(scores),
        'max': np.max(scores)
    }

hc_stats = calc_stats(hc_scores)
pd_stats = calc_stats(pd_scores)

# =============================================================================
# LOAD STEP 2 RESULTS
# =============================================================================
with open(RESULTS_FILE, 'r') as f:
    results = json.load(f)

# Find KCL results
for r in results['pd_results']:
    if r['cohort'] == 'KCL':
        kcl = r
        break

# =============================================================================
# GENERATE TABLE EXACTLY MATCHING LATEX FORMAT
# =============================================================================
print("\n" + "="*80)
print("TABLE 2: Primary Validation Results (LaTeX Format)")
print("="*80)

print("\nCohort & N & Mean PD-Likeness $\\pm$ SD & Median [Q1-Q3] & Range \\\\")
print("-"*80)
print(f"Healthy Control (HC) & {hc_stats['n']} & {hc_stats['mean']:.4f} $\\pm$ {hc_stats['std']:.4f} & {hc_stats['median']:.4f} [{hc_stats['q1']:.4f}, {hc_stats['q3']:.4f}] & {hc_stats['min']:.4f}--{hc_stats['max']:.4f} \\\\")
print(f"Parkinson's Disease (PD) & {pd_stats['n']} & {pd_stats['mean']:.4f} $\\pm$ {pd_stats['std']:.4f} & {pd_stats['median']:.4f} [{pd_stats['q1']:.4f}, {pd_stats['q3']:.4f}] & {pd_stats['min']:.4f}--{pd_stats['max']:.4f} \\\\")

print("\nMetric & Estimate & 95\\% CI & Interpretation \\\\")
print("-"*60)
print(f"AUROC & {kcl['auroc']:.3f} & — & Moderate discrimination \\\\")
print(f"Sensitivity & {kcl['sensitivity']:.3f} & — & {kcl['confusion_matrix']['tp']}/{pd_stats['n']} detected \\\\")
print(f"Specificity & {kcl['specificity']:.3f} & — & {kcl['confusion_matrix']['tn']}/{hc_stats['n']} correctly identified \\\\")
