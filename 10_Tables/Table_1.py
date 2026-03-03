#!/usr/bin/env python3
"""
TABLE 1: Identity Leakage Audit - Speaker Identification Reduction
================================================================================
PURPOSE:
    Generates Table 1 for the manuscript showing reduction in speaker identifiability
    after projecting to the audited 7-dimensional subspace.

DEPENDENCY CHAIN (CRITICAL FOR REPRODUCIBILITY):
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║  THIS SCRIPT DOES NOT PERFORM INFERENCE                                 ║
    ║  It only formats pre-computed results from Step 2.                      ║
    ║  The actual speaker identification experiments are run by:              ║
    ║    11_step2_experiments.py → run_speaker_id_closed_set()                ║
    ╚══════════════════════════════════════════════════════════════════════════╝

DATA FLOW:
    Step 1 (Extractor)
        ↓
    Raw embeddings + metadata CSVs
        ↓
    Step 2 (11_step2_experiments.py) ←───┐
        ↓                                 │
    complete_analysis_results.json  ←─────┘
        ↓                                 │
    THIS SCRIPT ──────────────────────────┘
        ↓
    Console Table (for reference)

INPUT:
    JSON file from Step 2 containing speaker_id_closed_set results
    Default: C:\Projects\hear_italian\step2_results\complete_analysis_results.json

OUTPUT:
    Formatted console table with 512D/7D accuracies, Δ, and % reduction

SOURCE CODE REFERENCE:
    - Class:      Step2Experiments (in 11_step2_experiments.py)
    - Method:     run_speaker_id_closed_set()
    - Task label: 'speaker_id_closed_set' in results JSON

NOTE ON INFERENCE:
    The speaker identification experiments (training logistic regression models,
    computing macro accuracies) are performed by Step 2. This script only:
        1. Loads the pre-computed results JSON
        2. Extracts macro accuracy values for Italian healthy cohorts
        3. Calculates Δ (7D-512D) and percentage reduction
        4. Formats for console display

    To regenerate the underlying data, run:
        python 11_step2_experiments.py --run_name complete_analysis
================================================================================
"""

import json
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
# Use your complete_analysis results file
RESULTS_FILE = Path(r'C:\Projects\hear_italian\step2_results\complete_analysis_results.json')
# If you want the -75 threshold specific results, use:
# RESULTS_FILE = Path(r'C:\Projects\hear_italian\step2_results\complete_analysis_-75_results.json')

# =============================================================================
# LOAD RESULTS
# =============================================================================
with open(RESULTS_FILE, 'r') as f:
    results = json.load(f)

print("\n" + "="*80)
print("TABLE 1: Identity Leakage Audit - Speaker Identification Reduction")
print("="*80)
print("Source: Step2Experiments.run_speaker_id_closed_set()")
print("Task: 'speaker_id_closed_set'")
print("="*80)

# =============================================================================
# EXTRACT SPEAKER ID RESULTS FOR ITALIAN COHORTS
# =============================================================================
table_data = []

for item in results['speaker_id_results']:
    if item['task'] == 'speaker_id_closed_set':
        cohort = item['cohort']
        emb_type = item['embedding_type']
        acc = item['macro_accuracy']
        
        # We only want Italian healthy cohorts
        if 'Italian' in cohort and 'HC' in cohort:
            table_data.append({
                'cohort': cohort,
                'emb_type': emb_type,
                'acc': acc
            })

# =============================================================================
# ORGANIZE INTO COHORT ROWS
# =============================================================================
cohorts = {}
for item in table_data:
    cohort = item['cohort']
    if cohort not in cohorts:
        cohorts[cohort] = {}
    cohorts[cohort][item['emb_type']] = item['acc']

# =============================================================================
# GENERATE TABLE
# =============================================================================
print("\n{:<25} {:>10} {:>10} {:>15} {:>15}".format(
    "Cohort", "512D Acc", "7D Acc", "Δ (7D-512D)", "% Reduction"))
print("-"*80)

for cohort in sorted(cohorts.keys()):
    acc_512 = cohorts[cohort]['512_raw']
    acc_7d = cohorts[cohort]['7d_scaled']
    delta = acc_7d - acc_512
    pct_reduction = ((acc_512 - acc_7d) / acc_512) * 100
    
    display_name = cohort.replace('Italian_', '').replace('_HC', ' HC')
    print("{:<25} {:>10.3f} {:>10.3f} {:>+14.3f} {:>14.1f}%".format(
        display_name, acc_512, acc_7d, delta, pct_reduction))

print("-"*80)
