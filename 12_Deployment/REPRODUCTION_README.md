# 🎙️ HeAR V2: Stability-First Laryngeal Auditing Pipeline
## Complete Reproduction Guide (Numerical Order)

---

## 📋 Complete Script Inventory (Numerical Order)

| # | Script | Description |
|---|--------|-------------|
| 00 | `00_task_control.py` | Task coordination helper for managing parallel processes (not required for reproduction). |
| 01 | `01_northwind_30s_center_window_distributions.py` | Legacy distribution analysis script (not required for primary results). |
| 06 | `06d_calibrate_tier1_thresholds_30sCenter.py` | **Clinical threshold calibration.** Calculates T = HC_sorted[1] + ε = 0.495101 (reported as 0.4952) to enforce ≤1 false positive in HC. |
| 06 | `06g_youden_j_threshold_sweep_30sCenter.py` | **Youden's J optimization.** Sweeps thresholds to identify statistical optimum (0.4395) for comparison with clinical policy. |
| 06 | `06h_extract_hear_embeddings_30sCenter_20260212.py` | Legacy extractor — use `11_certified_extractor_pytorch.py` instead. |
| 07 | `07_Perf_Severity_correlation_audit_KCL_bootstrap.py` | **Severity correlation.** Correlates PD-likeness scores with clinical severity measures (H&Y, UPDRS) in KCL cohort. |
| 08 | `08_calculate_cohens_d.py` | **Effect size audit.** Computes Cohen's d and 95% confidence intervals via bootstrap (10,000 iterations) from subject-level scores. |
| 09 | `09_drop_top_robustness.py` | **Outlier analysis.** Removes top 1 and top 2 PD subjects to test if results are driven by extreme cases. |
| 09 | `09_threshold_validation.py` | Supplementary threshold validation on held-out data. |
| 10 | `10_all_Extrator.py` | Legacy batch processing script (not required). |
| 10 | `10a_german_hc_audit.py` | **German validation.** Tests frozen model on German healthy cohort (n=101) using KCL aggregation method. |
| 10 | `10b_swedish_hc_audit.py` | **Swedish validation.** Tests frozen model on Swedish healthy cohort (n=102). |
| 10 | `10c_VCTK_hc_audit.py` | **VCTK English validation.** Tests frozen model on VCTK English healthy cohort (n=77). |
| 10 | `10d_Napali_hc_audit.py` | **Nepali validation.** Tests frozen model on Nepali healthy cohort (n=339). |
| 11 | `11_Identity_Leakage_Test.py` | **Pipeline integrity test.** Verifies Step 1 and Step 2 alignment, checks for placeholder rows, validates subject counts. |
| 11 | `11_certified_extractor_pytorch.py` | **Primary feature extractor (Step 1).** Loads HeAR model, extracts 2s sliding windows from 30s center segments, projects 512D embeddings to 7D audited subspace, scales, and saves with cryptographic hashes. |
| 11 | `11_step2_experiments.py` | **Core experiments (Step 2).** Runs PD classification (subject-level LOSO) and speaker identification (closed-set) on both 512D and 7D embeddings. Saves results with hash verification. |

---

## 🚀 Recommended Execution Order for Reproduction

For full reproduction of all manuscript results, run in this order:

```bash
# STEP 1: Feature Extraction
python 11_certified_extractor_pytorch.py --kcl_30s --device cuda

# STEP 2: Core Experiments
python 11_step2_experiments.py --kcl_30s --run_name kcl_30s_analysis

# STEP 3: Effect Size and Robustness
python 08_calculate_cohens_d.py
python 09_drop_top_robustness.py

# STEP 4: Threshold Calibration and Optimization
python 06d_calibrate_tier1_thresholds_30sCenter.py
python 06g_youden_j_threshold_sweep_30sCenter.py

# STEP 5: Severity Correlation
python 07_Perf_Severity_correlation_audit_KCL_bootstrap.py

# STEP 6: Cross-Linguistic Validation
python 10a_german_hc_audit.py
python 10b_swedish_hc_audit.py
python 10c_VCTK_hc_audit.py
python 10d_Napali_hc_audit.py

# STEP 7: Pipeline Integrity Verification (Optional)
python 11_Identity_Leakage_Test.py
