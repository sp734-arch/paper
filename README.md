# 🎯 Stability-First Auditing for Safe Speech-Based Health Screening
## Measurement Before Prediction

### Official code repository for the paper:  
*"Stability-First Auditing for Safe Speech-Based Health Screening"*

**Author:** Jim McCormack | **Year:** 2026 | **Paper:** [link forthcoming]

---
⚠️ Research Use Only

This repository contains research code and audited model artifacts
intended for methodological study. It is not a medical device and
is not approved for clinical use.

---

## 📖 Technical Navigation

This repository is built for **methodological reproducibility** — not model optimization.

| Document | Purpose |
| :--- | :--- |
| **[Model Summary](./09_model/README.md)** | Technical specification of the audited 7-D manifold and frozen linear head |

---

## 💡 Core Contribution

Foundation audio models (e.g., Google HeAR) expose high-resolution embedding spaces (512 dimensions). Such models can achieve high validation accuracy by exploiting stable but non-target structure — background acoustics, language, recording protocol, duration — leading to false confidence under distribution shift.

Conventional pipeline:

```text
representation → classifier → validation
```

We introduce a stability-first framework that inverts this ordering:

```text
representation → audit → proxy construction → constrained screening
```

Representation suitability is established **before** disease labels are introduced.

Embedding dimensions are audited label-blind for:

- **Within-subject stability**
- **Invariance** to task, language, and recording context
- **Independence / redundancy** structure

Only dimensions satisfying these criteria are retained, yielding a constrained **7-dimensional audited manifold**. Disease labels are introduced **only after the measurement space is fixed**.

Crucially:

> Discriminative behavior should be interpreted as a property of the complete measurement system (audited manifold + decision rule + sampling protocol + calibration), rather than of the embedding alone.

---

## ⚠️ Important Clarification

This is **not a Parkinson’s disease classifier repository.**

Parkinson’s disease is used solely as a demonstration context because it provides a well-characterized speech perturbation setting.

The contribution is methodological:

- Label-blind stability auditing  
- Confound sensitivity (invariance) testing  
- Identity leakage sanity checks  
- Frozen low-capacity linear probing  
- Reference-relative threshold anchoring  
- A minimum validity standard for speech-based screening proxies  

The objective is to qualify representations as measurement instruments — not to maximize predictive performance.

---

## 🔬 What This Repository Demonstrates

1. **Audit** foundation embeddings using healthy, non-disease data.  
2. **Constrain** the representation from 512 dimensions to a stability-qualified 7-D manifold.  
3. **Freeze** the subspace and linear probe.  
4. **Test** whether pathology-associated perturbation is observable under strict environmental, linguistic, and duration controls.  
5. **Calibrate** scores relative to a healthy reference distribution.

Performance metrics are interpreted as evidence of pathology-associated perturbation within a qualified measurement space — not as clinical validation.

---
## 🔁 Reproducing Published Results

All commands required to regenerate the manuscript’s figures, tables, and auditing analyses are documented in the dedicated reproduction guide:

👉 **[12_Deployment/REPRODUCTION_README.md](./12_Deployment/REPRODUCTION_README.md)**

The guide includes:

- Environment setup  
- Stability and invariance auditing scripts  
- Collapse vs. stability comparison (Table 1)  
- Auditing gates (Figure 3)  
- Frozen manifold loading and verification  
- External validation procedures  

This separation keeps the root README focused on the methodological framework, while the reproduction guide provides the full experimental workflow.

---

## 📂 Repository Registry (Epistemic Pipeline)

The repository mirrors the staged sequence described in the manuscript:

- **01_extract_embeddings**  
  Foundation embedding extraction and preprocessing.

- **02_extract_language_embeddings**  
  Cross-linguistic embedding generation.

- **03_loso_pipelines**  
  Leave-one-subject-out evaluation pipelines (naïve transfer behavior).

- **04_centroid_analysis**  
  Cohort-level structure and shift analysis.

- **05_dimension_selection**  
  Initial candidate dimension filtering.

- **06_dimension_independence**  
  Redundancy and correlation structure analysis.

- **07_dimension_stability**  
  Within-subject stability auditing.

- **08_final_dimension**  
  Final audited manifold definition (7-D subspace).

- **09_model**  
  Frozen linear head (SafeTensors, hashed bundle).

- **10_Tables / 11_figures**  
  Reproduction scripts for all manuscript tables and figures.

- **12_Deployment**  
  Reproduction guide and external validation procedures.

- **13_Prototype_App**  
  Research prototype demonstrating measurement-style screening workflow.

Each stage enforces the central principle:

> Representation qualification precedes discrimination.
---

## 🛡️ Model Distribution (SafeTensors)

The audited “Purified V2” head is distributed in SafeTensors format for:

- Bit-for-bit reproducibility  
- No arbitrary code execution  
- Transparent inspection  

**Model Hash:**  
`a50d941b6209f186a74d100e45cfee7a6a3c0ee70cdb17833fbd5e293b841dc0`

Example load:

```python
from 09_model.safetensor_loader import load_purified_v2_model

model_bundle = load_purified_v2_model(
    "09_model/pdhear_PURIFIED_V2_HASHED.safetensors"
)
```

The released bundle contains:

- Audited dimension indices  
- Scaling parameters  
- Frozen logistic regression head  

No retraining is required to reproduce published results.

---

## 📐 Minimum Validity Standard

This repository operationalizes a proposed set of seven necessary (not sufficient) conditions for speech-based screening proxies:

1. Representational auditability  
2. Label-independent proxy constructability  
3. Within-subject stability  
4. Invariance to non-target factors  
5. Person-specific structure under confound shift  
6. Reference-based calibration  
7. Non-circular alignment to independent standards  

Satisfying these conditions does **not** establish diagnostic validity. It establishes eligibility for interpretable screening use.

---

## 🎓 Citation

If you use this methodology, please cite:

```bibtex
@article{mccormack2026robustness,
  title={Stability-First Auditing for Foundation Audio Models in Speech-Based Health Screening},
  author={McCormack, Jim},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026},
  note={Purified V2 Model Hash: a50d941b6209f186...}
}

```
## 🙏 Acknowledgments

This work builds upon publicly available datasets and foundation models developed by the broader research community.

We gratefully acknowledge:

- **Health Acoustic Representations (HeAR)**  
  Shor et al. (2024) — provides the 512-dimensional foundation embedding used in this study.

- **Italian Parkinson’s Disease Corpus**  
  Dimauro et al. (2017), *IEEE Access* — used for representation auditing and probe training.

- **English Parkinson’s Disease Corpus**  
  Prior et al. (2023), *Figshare* — used for cross-linguistic evaluation.

- **Mobile Device Voice Recordings at King’s College London (MDVR-KCL)**  
  Rusz et al. (2019), *Zenodo* — used as the independent external validation cohort.

- **EMOVO Corpus (Italian Emotional Speech Database)**  
  Costantini et al. (2014), *LREC* — used for structured perturbation auditing of stability.

- **SES-ED Spanish Emotional Speech Dataset**  
  Roberto Ángel Meléndez-Armenta — used for invariance auditing during label-blind representation qualification.  
  Available at: https://www.kaggle.com/datasets/angeluxarmenta/ses-ed

- **SES-SD English Emotional Speech Dataset**  
  Roberto Ángel Meléndez-Armenta — paired dataset used for cross-linguistic invariance auditing.  
  Available at: https://www.kaggle.com/datasets/angeluxarmenta/ses-sd

- **VCTK Corpus**  
  Yamagishi et al. (2019), University of Edinburgh — used for clean, studio-quality healthy baseline evaluation.

- **Nepali Parliamentary Speech Dataset**  
  Subedi (2022), *GitHub* — used for naturalistic high-reverberance stress testing.

## All datasets were used in accordance with their respective licenses and terms of use.  
This repository distributes only derived model artifacts and auditing procedures, not original dataset content.
---

## ⚠️ Important Notes

### Data Availability

Due to privacy restrictions, we cannot provide:
- Raw audio files
- Patient identifiers
- Clinical metadata

We provide:
- ✅ Complete code pipeline
- ✅ Trained model weights
- ✅ Dimension selection methodology
- ✅ Validation framework

### Cross-Lingual Robustness

**Important:** Cross-lingual results in the paper are preliminary. Our primary validated claim is **task-invariance** (reading vs vowel tasks). Cross-lingual generalization across distant language families may require population-specific calibration.

For deployment, we recommend:
1. Task-controlled validation on target language
2. Population-specific healthy reference baselines
3. Language-family-appropriate calibration

See paper Discussion section for details.

---

**Questions? Open an issue or contact the author**

**Paper:** [link forthcoming]

**Last updated:** February 2026

---

## 📝 License

Apache License 2.0
