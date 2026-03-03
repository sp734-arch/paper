# 🎯 Robustness Before Diagnosis

**Official code repository for the paper:**  
**Robustness Before Diagnosis: Auditing Foundation Audio Models for Safe Speech-Based Health Screening**

**Author:** Jim McCormack  
**Year:** 2026  
**Paper:** arXiv link forthcoming

---

## 🎯 What This Repository Provides

This repository implements the full experimental pipeline described in the accompanying paper. It enables readers to:

- Reproduce the auditing and evaluation framework used in the study  
- Understand how robustness failures arise in naïve foundation-model transfer  
- Apply the proposed stability-first auditing methodology to new speech datasets  

The focus of this repository is **methodological reproducibility**, not deployment-ready software.

---

## 💡 Core Contribution

### Problem
Foundation audio models paired with disease classifiers often achieve high internal accuracy while exploiting non-physiological structure related to task, language, or recording conditions. Under task-controlled evaluation, this apparent disease signal frequently collapses.

### Approach
We introduce a **stability-first auditing framework** that inverts the conventional pipeline. Instead of optimizing diagnostic performance first, we:

- Identify candidate physiological dimensions using non-disease data  
- Audit stability, independence, and invariance at the representation level  
- Restrict downstream screening to a validated low-dimensional subspace  

### Outcome
This approach isolates vocal structure that remains stable across task, language, and device variation, enabling screening that is robust under external validation.


**Paper sections:** See Section 7.2.4 (dimension selection), Figure 3 (auditing), Table 1 (results).

---

## 📊 Reproducing Paper Results

### Table 1: Main Results

Table 1 is a **synthesis of multiple experiments** (not a single script). See [`06_table1_provenance/Table_1_Provenance.md`](06_table1_provenance/Table_1_Provenance.md) for exact mapping of each row to source code.

**Quick summary:**
- **Naïve rows (mixed-task):** Use Italian + English data, standard cross-validation
- **Task-controlled rows:** Use KCL held-out cohort, subject-disjoint evaluation
- **Full embedding:** All 512 HeAR dimensions (collapses under task control)
- **Audited subspace:** Our 7 validated dimensions (maintains performance)

Scripts: `02_dimension_auditing/` (Steps 5-8) + `03_train_purified_head/` + `04_evaluate_kcl/`

---

### Figure 2: Task Collapse Demonstration

Shows how naïve training fails under task control.

**Generate:**
```bash
python 05_generate_figures/Fig_2_abc_mixed_task.py
```

**Paper reference:** Figure 2 (page 3), Section 7.2.5

**Note:** Uses demonstration data. See script comments for adapting to your evaluation outputs.

---

### Figure 3: Auditing Process (a, b, c)

Shows the dimension selection and validation process.

**Generate all panels:**
```bash
python 05_generate_figures/Fig_3_a_stability_barchart.py      # Stability spectrum
python 05_generate_figures/fig_3_b_confound_sensitivity_audit.py  # Invariance
python 05_generate_figures/fig_3_c__corrolation_Structure.py     # Independence
```

**Paper reference:** Figure 3 (page 4), Section 5 (auditing protocol)

**What they show:**
- **3a:** Most of 512 dimensions are unstable (only small subset passes)
- **3b:** Most dimensions confounded by task/language/device (Cohen's d > 0.4)
- **3c:** Full space is redundant; purified space is independent

---

### Figure 4: Reference Calibration

Shows population-specific baseline calibration.

**Generate:**
```bash
python 05_generate_figures/Fig_4_abc_callibrated.py
```

**Paper reference:** Figure 4 (page 5), Section 6 (calibration)

---

## 🔬 The Complete Pipeline (512 → 10 → 7 Dimensions)

**See [`02_dimension_auditing/DIMENSION_SELECTION_PIPELINE.md`](02_dimension_auditing/DIMENSION_SELECTION_PIPELINE.md) for complete methodology.**

### Quick Overview:

**Stage 1: Feature Selection (512 → 10)**
```bash
python 02_dimension_auditing/unmask_hear_drivers_STEP_5.py
```
- Uses **SES emotion data** (`esen_ovlap_75`) - NOT Parkinson's data!
- Trains LinearSVC on English vs Spanish classification
- Selects top 10 dimensions by weight: `[419, 227, 43, 346, 204, 317, 38, 98, 267, 146]`
- **Why emotion data?** Language discrimination finds stable vocal physiology, not disease artifacts
- **Paper reference:** Section 7.2.4

---

**Stage 2: Independence Validation (10 → 10)**
```bash
python 02_dimension_auditing/Independence_Audit_STEP_6.py
```
- Computes Pearson correlation matrix (10×10)
- Average |r| = 0.24 (threshold < 0.3)
- All 10 dimensions pass → No redundancy
- **Paper reference:** Section 5, Figure 3c

---

**Stage 3: Stability Validation (10 → 10)**
```bash
python 02_dimension_auditing/stability_50_Step_7.py
```
- Computes between-subject / within-subject variance ratio
- Ratio ≈ 9.8 (threshold > 1.5, proxy for ICC > 0.5)
- All 10 dimensions pass → Trait-level stability
- **Paper reference:** Section 5, Figure 3a

---

**Stage 4: Individual Ranking (10 → 7)**
```bash
python 02_dimension_auditing/Marker_leaderboard_STEP_8.py
```
- Computes individual stability ratio for each dimension
- Drops 3 lowest: 227 (0.60), 317 (0.49), 98 (0.46)
- **Final 7 dimensions:** `[38, 43, 146, 204, 267, 346, 419]`
- **Paper reference:** Section 7.2.4 (lists exact indices)

---

**Stage 5: Train Purified Model**
```bash
python 03_train_purified_head/Genesis_purified_v2_PKL__DNE_IP.py
```
- Trains LogisticRegression (C=0.5, balanced) on 7 dimensions
- Data: Italian + English (NOT KCL - that's held-out test set)
- LOSO validation: 66% accuracy (honest estimate, not inflated)
- Outputs: `pdhear_PURIFIED_V2.pkl` (or `.safetensors`)
- **Paper reference:** Section 7.2.4

---

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/robustness-before-diagnosis.git
cd robustness-before-diagnosis

# Install dependencies
pip install -r requirements.txt
```

### Minimal Reproduction (Without Full Dataset)

**If you just want to see the model and understand the pipeline:**
```bash
# 1. Inspect the final model
python models/inspect_safetensors.py models/pdhear_PURIFIED_V2.safetensors

# 2. See dimension selection methodology
cat 02_dimension_auditing/DIMENSION_SELECTION_PIPELINE.md

# 3. See Table 1 provenance
cat 06_table1_provenance/Table_1_Provenance.md

# 4. Generate figures (uses demo data)
python 05_generate_figures/Fig_3_a_stability_barchart.py  # (requires embeddings)
```

### Full Reproduction (With Dataset)

**Requirements:**
- HeAR foundation model embeddings (512-D)
- Italian, English, KCL cohort data
- SES emotion dataset (`esen_ovlap_75`)

**See:** `01_extract_embeddings/` for embedding extraction instructions.

**Then run complete pipeline:**
```bash
# Step 5-8: Dimension auditing
python 02_dimension_auditing/unmask_hear_drivers_STEP_5.py
python 02_dimension_auditing/Independence_Audit_STEP_6.py
python 02_dimension_auditing/stability_50_Step_7.py
python 02_dimension_auditing/Marker_leaderboard_STEP_8.py

# Train model
python 03_train_purified_head/Genesis_purified_v2_PKL__DNE_IP.py

# Evaluate on KCL
python 04_evaluate_kcl/stress_test.py
python 04_evaluate_kcl/LINGUISTIC_FAMILY_AUDIT.py

# Generate figures
python 05_generate_figures/Fig_3_*.py
```

---

## 📁 Repository Structure
```
.
├── 01_extract_embeddings/          # Step 1: HeAR embedding extraction
├── 02_dimension_auditing/          # Steps 5-8: Find and validate 7 dimensions
│   ├── DIMENSION_SELECTION_PIPELINE.md  ← KEY DOCUMENT (explains 512→10→7)
│   └── [Step 5, 6, 7, 8 scripts]
├── 03_train_purified_head/         # Train logistic regression on 7-D
├── 04_evaluate_kcl/                # External validation (Table 1 bottom rows)
├── 05_generate_figures/            # Reproduce Figures 2, 3, 4
├── 06_table1_provenance/           # Table 1 row-by-row documentation
│   └── Table_1_Provenance.md       ← KEY DOCUMENT (explains Table 1)
└── models/                         # Model weights (SafeTensors format)
```

---

## 🔑 Key Documentation

**Must-read for understanding the paper:**

1. **[DIMENSION_SELECTION_PIPELINE.md](02_dimension_auditing/DIMENSION_SELECTION_PIPELINE.md)**  
   Complete explanation of how 512 dimensions → 7 final dimensions  
   → Answers: "Where do the dimensions come from?"

2. **[Table_1_Provenance.md](06_table1_provenance/Table_1_Provenance.md)**  
   Maps each Table 1 row to specific scripts and datasets  
   → Answers: "How do I reproduce Table 1?"

3. **[Genesis Script Comments](03_train_purified_head/Genesis_purified_v2_PKL__DNE_IP.py)**  
   See top of file for dimension provenance block  
   → Answers: "Why these 7 dimensions?"

---

## 🛡️ Model Distribution

### SafeTensors Format (Recommended)

We provide the trained model in **SafeTensors format** for security and reproducibility.

**Why SafeTensors?**
- ✅ No arbitrary code execution (safe to load)
- ✅ Fast, memory-efficient
- ✅ Industry standard (HuggingFace, Stability AI)
- ✅ Inspectable without loading

**Load the model:**
```python
from models.load_from_safetensors import load_from_safetensors

bundle = load_from_safetensors("models/pdhear_PURIFIED_V2.safetensors")
model = bundle['model']        # LogisticRegression
scaler = bundle['scaler']      # StandardScaler
indices = bundle['indices']    # [267, 346, 43, 146, 204, 38, 419]
```

**Or inspect without loading:**
```bash
python models/inspect_safetensors.py models/pdhear_PURIFIED_V2.safetensors
```

**Generate yourself (if you prefer):**
```bash
python 03_train_purified_head/Genesis_purified_v2_PKL__DNE_IP.py
python models/convert_to_safetensors.py pdhear_PURIFIED_V2.pkl
```

---

## 🎓 Citation

If you use this code or methodology, please cite:
```bibtex
@article{yourname2025robustness,
  title={Robustness Before Diagnosis: Auditing Foundation Audio Models for Safe Speech-Based Health Screening},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

## 📝 License

[MIT License / Apache 2.0 / Your Choice]

---

## 🤝 Contributing

This repository is provided for reproducibility of our published work. For questions or issues:

1. Check the documentation first:
   - [DIMENSION_SELECTION_PIPELINE.md](02_dimension_auditing/DIMENSION_SELECTION_PIPELINE.md)
   - [Table_1_Provenance.md](06_table1_provenance/Table_1_Provenance.md)

2. Open a GitHub issue with:
   - Which paper result you're trying to reproduce
   - What you tried
   - What didn't work

3. For general questions about the methodology, see the paper or contact: [your email]

---

## 🙏 Acknowledgments

- HeAR foundation model from Google Research
- Italian Parkinson's dataset from [institution]
- KCL cohort from [institution]
- SES emotion dataset from [source]

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

**Questions? Open an issue or contact [your email]**

**Paper:** [arXiv link after submission]

**Last updated:** February 2026

🎯 KEY ELEMENTS THIS README PROVIDES

✅ Direct paper connection - Links every section to paper sections/figures
✅ Reproduces Table 1 - Clear provenance documentation
✅ Explains innovation - Healthy data first (the money move)
✅ Complete pipeline - 512→10→7 with script names
✅ Quick start - Multiple entry points (inspect, partial, full)
✅ Key documents - Points to DIMENSION_SELECTION_PIPELINE.md and Table_1_Provenance.md
✅ Model distribution - SafeTensors with security explanation
✅ Honest caveats - Notes about cross-lingual being preliminary
