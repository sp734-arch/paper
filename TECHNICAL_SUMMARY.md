# Robustness Before Diagnosis: A Technical Summary

## Core Contribution

This work establishes a **measurement-first framework** for speech-based health screening, reframing the problem from classification to proxy measurement construction.

Screening outputs must be treated as **proxy measurements**, not predictive labels — constructed scalar quantities with explicitly characterized stability, invariance, and calibration properties.

---

## 1. Problem Statement: Misuse of Foundation Audio Models

Foundation audio models encode high-resolution acoustic structure.

High apparent screening accuracy often arises from confounds (task, language, device, recording context), not physiology. Standard cross-validation can systematically overstate validity.

Treating embeddings as screening-ready features is a category error.

Confounds can dominate signal even when disease labels are held out, creating a false impression of robustness.

---

## 2. Central Claim: Screening Requires Constructed Proxies

Screening outputs must be treated as **proxy measurements**, not predictions.

Proxy characteristics must be explicitly constructed and qualified.

**Representation quality ≠ measurement validity.**

---

## 3. Key Failure Mode Identified

Naïve pipelines exploit shortcut signals.

Disease separation collapses toward chance under properly controlled evaluation:

- Task-controlled evaluation  
- Subject-disjoint splits  

This represents a **representation-level failure for screening use**, not a failure of model expressivity.

---

## 4. Proposed Solution: Stability-First Auditing Framework

The paper inverts the conventional pipeline.

**Standard pipeline (insufficient for screening):**

```
representation → classifier → validation
```

**Proposed pipeline:**
```
representation → audit → proxy construction → constrained screening
```

Representation qualification precedes discrimination.

---

## 5. Auditing Criteria (Minimum Validity Conditions)

Candidate proxy characteristics must satisfy:

1. Representational auditability  
2. Proxy constructability (derivable without disease labels)  
3. Within-subject stability  
4. Invariance to non-target factors  
5. Person-specific structure under dominant non-pathological confounds (e.g., language, age), ensuring the proxy reflects trait-like structure rather than contextual variation  
6. Reference-based calibration  
7. Non-circular alignment to independent clinical standards  

Failure at any step disqualifies the proxy from screening use.

---

## 6. Operationalization

- Audit embedding dimensions **without disease labels**
- Quantify:
  - Stability (ICC threshold)
  - Invariance (effect-size thresholds)
- Reject dimensions failing criteria
- Retain only a low-dimensional audited subspace

**Result:** 512 dimensions → 7 retained (~1.4%)

The retained subspace is fixed prior to disease modeling.

---

## 7. Calibration and Interpretation

Calibration is performed **prior to disease modeling**.

**Purpose:**

- Establish scale and reference frame  
- Prevent proxy meaning from being defined by downstream optimization  

Disease effects are treated as **perturbations within a qualified measurement space**, not defining features of the representation.

---

## 8. Empirical Demonstration (Parkinson's Disease)

Parkinson's disease is used as a **worked example**, not as a disease-specific claim.

The framework is disease-agnostic; PD is selected due to availability of independent clinical scales.

Demonstrates:

- Collapse of naïve screening under control  
- Recovery of interpretable separation after auditing  
- Robustness under task control  
- Alignment with independent clinician-administered scales (used for interpretation, not model training)

---

## 9. Explicit Non-Claims

This work does **not** claim:

- Diagnostic validity  
- Mechanistic physiological modeling  
- Turnkey clinical deployment  
- Disease universality  
- Replacement of clinician judgment  

The framework defines eligibility conditions for screening proxies — not clinical sufficiency.

---

## 10. Contribution

Reframes speech-based screening as a **measurement qualification problem**.

Introduces a **minimum validity standard** for proxy measurements.

Separates:

1. Representation learning  
2. Proxy construction  
3. Screening evaluation  

Shifts evaluation from **performance-first** to **measurement-first**.

---

## Interpretation Note

Failures under age, language, or protocol shift should not be interpreted as model breakdown.

They represent expected behavior of a calibrated measurement instrument operating outside its validated domain.

Such behavior reflects domain specificity rather than instability.

---

## 📊 The 7-Step Validity Standard (Quick Reference)

| Step | Criterion | Pass/Fail |
|------|-----------|-----------|
| 1 | Representational auditability | ✓ |
| 2 | Proxy constructability (no disease labels) | ✓ |
| 3 | Within-subject stability (ICC > 0.5) | ✓ |
| 4 | Invariance to non-target factors | ✓ |
| 5 | Person-specific structure under confound shift | ✓ |
| 6 | Reference-based calibration | ✓ |
| 7 | Alignment to independent clinical standards | ✓ |

**Result:** 512D → 7D retained (1.4%)
