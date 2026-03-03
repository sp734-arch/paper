# 🛡️ Reviewer FAQ: Methodological Clarifications and Validation Analyses

This document provides evidence-based clarifications regarding the 7-dimensional audited subspace and the operational threshold (T = 0.4952) derived under the final −55 dBFS analysis. For full technical detail, please refer to the manuscript and CODEX documentation.

The goal of this FAQ is to clarify methodological choices and interpretive boundaries, not to extend claims beyond what is reported in the paper.

---

## 🎯 Q1: Statistical Power and External Cohort Size (KCL N=12 PD)

**The Inquiry:**  
Is the external validation cohort (N=12 PD, N=17 HC) sufficient to support claims of generalization?

**Clarification:**  
The KCL cohort serves as an external robustness test rather than a discovery dataset. The manifold and linear probe were frozen prior to KCL evaluation. The KCL dataset therefore functions as an independent perturbation test under new recording conditions.

While subject-level N is modest, evaluation includes 841 slices prior to thresholding (757 after −55 dBFS processing). Robustness is further examined through cross-linguistic healthy cohorts:

- **KCL HC (Condition D):** μ = 0.4044 (N = 17)
- **German HC:** μ = 0.2478 (N = 101) — 98.02% specificity  
- **Swedish HC:** μ = 0.2391 (N = 102) — 98.04% specificity  
- **Nepali Parliamentary:** μ = 0.3070 (N = 339) — 98.53% specificity  
- **VCTK English:** μ = 0.1620 (N = 77) — 100.00% specificity  

**Interpretation:**  
The KCL cohort demonstrates observable separation (AUROC = 0.706; 95% CI [0.55, 0.94]). Cross-linguistic evaluation across 619 healthy speakers indicates stable specificity behavior (98.0–100%) at the fixed KCL-derived threshold. These results support robustness of the safety-constrained threshold across recording conditions and languages, while not constituting population-level diagnostic validation.

---

## 🎯 Q2: Dimensionality Reduction and Overfitting Risk

**The Inquiry:**  
Does selecting 7 dimensions from a 512-dimensional embedding introduce overfitting or selection bias?

**Clarification:**  
Dimension selection was performed using a label-blind invariance protocol. No disease labels were used during pruning. Selection criteria required:

- High intra-class correlation (ICC > 0.5)
- Low task sensitivity (|d| < 0.4)
- Low redundancy within the retained subset

**Identity Leakage Audit:**

- Italian Young HC: 38.2% → 15.7% (58.9% reduction)
- Italian Elderly HC: 27.1% → 7.7% (71.6% reduction)

These reductions indicate suppression of speaker-identifiable structure under a fixed linear probe.

**Interpretation:**  
The 7-dimensional subspace represents a constrained feature space derived under adversarial, label-blind criteria. The retained axes demonstrate stability and low task sensitivity, supporting the interpretation that they encode trait-like structure rather than speaker-specific artifacts. This does not prove physiological encoding directly but supports compatibility with screening applications.

---

## 🎯 Q3: Threshold Selection and Methodological Circularity

**The Inquiry:**  
Is using the KCL cohort for threshold calibration (T = 0.4952) and performance reporting circular?

**Clarification:**  
A distinction is maintained between:

- **Feature Representation** (audited subspace + frozen linear head)
- **Operational Threshold**

The manifold and classifier were frozen prior to threshold selection. The AUROC (0.706) reflects representation-level discriminative capacity independent of threshold.

The operational threshold is defined as:
-T_operational = HC_sorted[1] + ε = 0.495101

Reported as T = 0.4952 (conservative ceiling to 4 decimal places).

This enforces ≤1 false positive in the KCL HC cohort (94.12% specificity).

Cross-cohort validation:

- Nepali HC (N = 339): 98.53% pass rate  
- German HC (N = 101): 98.02%  
- Swedish HC (N = 102): 98.04%  
- VCTK English (N = 77): 100.00%  

**Interpretation:**  
The threshold represents a safety-constrained policy decision. Because representation and weights were frozen prior to threshold selection, AUROC remains an unbiased measure of sorting capacity. Cross-linguistic specificity stability suggests the threshold remains conservative across populations.

---

## 🎯 Q4: Sensitivity–Specificity Trade-Off

**The Inquiry:**  
Is 50% sensitivity sufficient for screening?

**Clarification:**  
The selected operating point (T = 0.4952) prioritizes specificity to reduce false-positive burden.

| Policy | Threshold | Specificity | Sensitivity | HC FP | PD TP |
|--------|-----------|-------------|-------------|-------|-------|
| Youden Optimum | 0.4395 | 76.5% | 75.0% | 4 | 9 |
| **Clinical Policy (Selected)** | **0.4952** | **94.1%** | **50.0%** | **1** | **6** |
| Zero-FP Alternative | 0.6195 | 100% | 25.0% | 0 | 3 |

The selected policy reduces false positives by 75% relative to the Youden optimum (4 → 1).

This framework is designed for longitudinal monitoring, where repeated independent assessments may increase cumulative detection probability. Threshold selection is tunable per deployment context.

**Interpretation:**  
The selected threshold reflects a safety-constrained screening policy rather than a statistical optimum. It is explicitly adjustable based on clinical risk tolerance.

---

## 🎯 Q5: Rationale for Foundation Model Embeddings

**The Inquiry:**  
Why use foundation embeddings rather than traditional acoustic features (e.g., jitter, shimmer)?

**Clarification:**  
Foundation embeddings provide robustness to environmental and task variation:

- Stable under connected speech (30s passages)
- Robust across recording conditions (mobile, studio, multilingual corpora)
- Identity leakage reduced by 71.6% in audited subspace

The VCTK cohort (studio recordings) exhibits expected baseline shift (μ = 0.1620) while maintaining 100% specificity at the fixed threshold.

**Interpretation:**  
The auditing framework transforms a 512-dimensional embedding into a stability-qualified 7-dimensional subspace. The protocol does not replace classical acoustics but provides a structured method for auditing high-dimensional representations prior to screening use.

---

## ✅ Cross-Linguistic Specificity Matrix

| Metric | KCL HC | German HC | Swedish HC | Nepali HC | VCTK English |
|:-------|:------:|:---------:|:----------:|:---------:|:------------:|
| **N-Size** | 17 | 101 | 102 | 339 | 77 |
| **Mean Score** | 0.4044 | 0.2478 | 0.2391 | 0.3070 | 0.1620 |
| **Specificity @ T=0.4952** | 94.12% | 98.02% | 98.04% | 98.53% | 100.00% |

**Aggregate Healthy Validation:**

- Total HC subjects: 619  
- Weighted mean specificity: 98.3% (610/619)  
- Observed range: 98.0–100.0%

---

## Summary

The KCL-derived threshold (T = 0.4952) demonstrates stable specificity behavior across five independent language cohorts comprising 619 healthy speakers. The audited 7-dimensional subspace exhibits:

- Stability under perturbation  
- Reduced identity leakage  
- Cross-cohort specificity robustness  

These findings support the interpretation that the audited subspace captures language-robust, task-invariant structure compatible with screening workflows. They do not constitute diagnostic validation or regulatory approval.

---

*Last updated: 2026-02-25*  
*Based on −55 dBFS analysis (AUROC = 0.706; 95% CI [0.55, 0.94])*
