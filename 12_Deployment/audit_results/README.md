# Audit Results Directory
## HEAR-Italian Study - Condition D Analysis

This directory contains all derived data necessary to reproduce the tables and figures in the manuscript. 
The files are organized by their role in the analysis pipeline.

---

## 📁 **Directory Contents**

### Primary Subject-Level Data (from Step 1)

These CSV files contain the final PD-likeness scores after applying the -55 dBFS silence threshold and 
30s center window extraction (Condition D). Each row represents one subject with their mean score across all slices.

| File | Description | Used By |
|------|-------------|---------|
| `HC_NorthWind_PD-Likeness_30sCenter_20260212_100148.csv` | Subject-level scores for 17 healthy controls | `table_2_primary_results.py`, `08_calculate_cohens_d.py`, `09_drop_top_robustness.py`, `06d_calibrate_tier1_thresholds_30sCenter.py` |
| `PD_NorthWind_PD-Likeness_30sCenter_20260212_101240.csv` | Subject-level scores for 12 Parkinson's disease subjects | Same scripts as above |

### Step 2 Results (Classification Experiments)

These JSON files contain the complete output from the identity vs disease experiments, including 
subject-level predictions, confusion matrices, and all derived metrics.

| File | Description | Used By |
|------|-------------|---------|
| `kcl_30s_analysis_results.json` | Subject-level LOSO classification results for KCL Condition D cohort (AUROC=0.706) | `table_2_primary_results.py` (Table 2 metrics) |
| `complete_analysis_results.json` | Complete cross-lingual analysis including speaker identification experiments | `table_1_identity_leakage.py` (Table 1 speaker ID results) |

---

## 📊 **Reproducing Manuscript Tables**

To regenerate any table, run the corresponding script from the `wavstudy/` directory:

| Table | Script | Required Files |
|-------|--------|----------------|
| **Table 1** (Identity Leakage) | `python table_1_identity_leakage.py` | `complete_analysis_results.json` |
| **Table 2** (Primary Validation) | `python table_2_primary_results.py` | Both CSV files + `kcl_30s_analysis_results.json` |
| **Table 3** (Clinical Threshold) | `python 06d_calibrate_tier1_thresholds_30sCenter.py` | Both CSV files |
| **Table 4** (Drop-Top Robustness) | `python 09_drop_top_robustness.py` | Both CSV files |

---

## 🔬 **Data Provenance**

All files in this directory were generated from raw audio using the following pipeline:

1. **Step 1 Extraction**: `11_certified_extractor_pytorch.py --kcl_30s --silence_thresh -55`
   - Extracts 2s sliding windows from 30s center segments
   - Projects 512D HeAR embeddings to 7D audited subspace
   - Averages across all slices to produce subject-level scores

2. **Step 2 Experiments**: `11_step2_experiments.py --kcl_30s` and `--run_name complete_analysis`
   - Performs subject-level LOSO classification
   - Runs closed-set speaker identification experiments
   - Saves complete results with cryptographic hashes

---

## 🔐 **File Integrity**

The following SHA256 hashes verify file integrity (generated 2026-02-12):

- HC_NorthWind_PD-Likeness_30sCenter_20260212_100148.csv
- 33db8c2855a6bb66b1e6d2fa7da839613e5042a0d8b5cb552da5ae133a8be86d

- PD_NorthWind_PD-Likeness_30sCenter_20260212_101240.csv
- de8f6756efb38ed1ad4c3ac301511b1eed425095c718afde5797098475683419
