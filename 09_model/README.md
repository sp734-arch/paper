
# 🎯 Purified V2 Model - Stable Physiological Signal Extraction

**Paper Reference**: "The Purified V2 Head"  
**Model Hash**: `a50d941b6209f186a74d100e45cfee7a6a3c0ee70cdb17833fbd5e293b841dc0`  
**Created**: 2024-02-07  
**Status**: ✅ Production-ready, hash-validated

## 📁 File Overview

| File | Purpose | Paper Connection |
|:-----|:--------|:-----------------|
| **`model_build.py`** | Creates the final Purified V2 model bundle | model creation |
| **`looker.py`** | Inspects model weights, generates paper tables |  |
| **`model_weights.json`** | Human-readable weight representation | Safe format for paper review/supplement |
| **`pdhear_PURIFIED_V2_HASHED.safetensors`** | Primary model file (secure format) | Production model  |
| **`safetensor_converter.py`** | Converts `.pkl` → `.safetensors` | Security protocol for model sharing |
| **`safetensor_hash_validator.py`** | Validates model integrity via hash | Ensures reproducibility |
| **`safetensor_loader.py`** | Simple loader + prediction function | Deployment utility |
| **`loader_function_example.py`** | Example integration code | Reference implementation for other studies |
| **`model_readme.png`** | Visual overview of model architecture | Supplementary material  |
| **`readme.md`** | This documentation file | Ties all components to paper narrative |

## 🔄 Workflow: From Audit to Deployment

This repository encapsulates the **final stage** of the paper's stability-first auditing framework:

🏗️ Core Model Components
1. The Purified Subspace 

    Dimensions: 7/512 retained (98.6% reduction)

    Indices: [267, 346, 43, 146, 204, 38, 419]

2. Model Architecture
python

# Simplified architecture from model_build.py
Input: 512-dim HeAR embedding
↓
Mask: Select 7 stable dimensions [267, 346, 43, 146, 204, 38, 419]
↓
Scaling: sklearn StandardScaler (LOSO-compatible)
↓
Classification: LogisticRegression(C=0.5, class_weight='balanced')
↓
Output: PD probability [0, 1]

3. Training & Validation

    Data: Italian + English PD corpora (multi-lingual)

    Samples: 14,161 (6,416 healthy, 7,745 PD)

    Subjects: 62 unique speakers

    Validation: Leave-One-Subject-Out (LOSO)

    Performance: 66.1% (Single shot screening calibration)

🚀 Quick Start Guide
Option A: Load Pre-built Model (Recommended)
python

# Using the simple loader
from safetensor_loader import load_purified_v2_model, predict_pd

# Load model (validates hash automatically)
model_bundle = load_purified_v2_model('pdhear_PURIFIED_V2_HASHED.safetensors')

# Make prediction on HeAR embeddings
# embeddings: numpy array shape (n_samples, 512)
predictions = predict_pd(model_bundle, embeddings)

Option B: Build from Scratch
bash

# 1. Create the model (requires training data)
python model_build.py  # outputs pdhear_PURIFIED_V2_HASHED.pkl

# 2. Convert to secure format
python safetensor_converter.py pdhear_PURIFIED_V2_HASHED.pkl model.safetensors

# 3. Validate hash
python safetensor_hash_validator.py model.safetensors

# 4. Inspect model weights (generates paper Table X)
python looker.py

Option C: Integration Example
python

# See loader_function_example.py for complete example
from loader_function_example import load_safetensors_model, recreate_sklearn_model

# Full recreation of sklearn pipeline
bundle = load_safetensors_model('pdhear_PURIFIED_V2_HASHED.safetensors')
model, scaler, metadata = recreate_sklearn_model(bundle)

🔐 Security & Reproducibility
Hash Validation
python

# Always validate model integrity
from safetensor_hash_validator import validate_safetensors_hash

expected_hash = "a50d941b6209f186a74d100e45cfee7a6a3c0ee70cdb17833fbd5e293b841dc0"
is_valid = validate_safetensors_hash('model.safetensors', expected_hash)

Safe Formats
Format	Security	Use Case
.safetensors	✅ Safe	Production deployment
.json	✅ Safe	Paper review, analysis
.pkl	⚠️ Risky	Internal use only

⚠️ Warning: Never load .pkl files from untrusted sources. Use .safetensors or .json for sharing.
📊 Model Inspection & Analysis
Generate Paper Tables
bash

# See Weights
python looker.py

# Output includes:
# - Dimension weights and interpretations
# - Stability metrics for each dimension
# - Clinical correlation notes

Examine Weights Safely
python

# No code execution required
import json
with open('model_weights.json') as f:
    weights = json.load(f)
    
print(f"Dimensions: {weights['dimensions']}")
print(f"Paper reference: {weights['paper_reference']}")

🧩 Dependencies
bash

# Core dependencies
pip install scikit-learn numpy pandas joblib

# For safetensors support
pip install safetensors torch

# For hash validation
pip install hashlib  # Standard library

❓ Frequently Asked Questions
Q: Why 7 dimensions?

A: Through the auditing framework (Steps 1-9), these 7 dimensions showed:

    High subject-level stability (ICC > 0.5)

    Language invariance (Spanish/English test)

    Low inter-correlation (< 0.3 average)

    Clinical correlation with PD severity

Q: Can I use this model directly?

A: Yes, but note:

    Input must be 512-dim HeAR embeddings (from heAR foundation model)

    Requires 2-second audio clips at 16kHz

    Use safetensor_loader.py for easiest integration

Q: How to validate on new data?

A: Follow the paper's LOSO protocol:

    Extract HeAR embeddings with 50% overlap

    Apply the 7-dimension mask

    Use predict_pd() from safetensor_loader.py

    Report subject-level accuracy

Q: Where's the training data?

A: Due to privacy constraints, only the model weights are shared. 
📚 Citation & References

When using this model, please cite:
bibtex

@article{mccormack2026stability,
  title={Stability-first auditing of foundation audio representations for equitable disease screening},
  author={McCormack, James and others},
  journal={arvix},
  year={2026},
  note={Purified V2 Model Hash: a50d941b6209f186a74d100e45cfee7a6a3c0ee70cdb17833fbd5e293b841dc0}
}

🆘 Support & Issues

    Model issues: Open GitHub issue with hash validation results

    Integration help: Refer to loader_function_example.py

    Hash mismatch: Re-run model_build.py with exact same random seeds

Maintainer: Jim McCormack
Last Updated: February 2026

