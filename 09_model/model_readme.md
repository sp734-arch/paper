# Purified V2 Model - Production Scripts

**Paper Reference**: Section 7.2.4 - The Purified V2 Head  
**Model Hash**: `a50d941b6209f186a74d100e45cfee7a6a3c0ee70cdb17833fbd5e293b841dc0`  
**Created**: 2024-02-07

## 📋 Core Scripts

This directory contains the essential scripts for creating, converting, and inspecting the Purified V2 model:

### 1. 🏗️ `model_build.py`
**Purpose**: Create the final Purified V2 model bundle with cryptographic hash  
**Output**: `pdhear_PURIFIED_V2_HASHED.pkl`  
**Key Features**:
- Applies 7-dimension stability mask [267, 346, 43, 146, 204, 38, 419]
- Multi-lingual training (Italian + English)
- Leave-One-Subject-Out validation
- SHA-256 reproducibility hash


## Run to create model
python model_build.py

### 2. 🔍 looker.py

Purpose: Inspect model weights and generate paper-ready tables
Output: Table X format with interpretations
bash

# Generate Table X from paper
python looker.py

# Output example:
 | HeAR Dimension | Weight | Interpretation |
 |:--------------|:-------|:---------------|
 |            38  | +0.802 | Strong PD biomarker |
 |           419  | +0.356 | Moderate PD biomarker |
 |            43  | +0.056 | Weak PD biomarker |

### 3. 🔄 safetensor_converter.py

Purpose: Convert .pkl to secure .safetensors format
Security: Avoids pickle security risks
bash

Convert model
python safetensor_converter.py pdhear_PURIFIED_V2_HASHED.pkl model.safetensors

Features:
- Preserves all model parameters
- Embeds metadata as JSON
- Maintains hash compatibility

### 4. ✅ safetensor_hash_validator.py

Purpose: Validate hash integrity of safetensors files
Usage: Ensure converted models match original hash
python

Validate a safetensors file
from safetensor_hash_validator import validate_safetensors_hash

is_valid = validate_safetensors_hash(
    "model.safetensors", 
    "a50d941b6209f186a74d100e45cfee7a6a3c0ee70cdb17833fbd5e293b841dc0"
)

### 5. 📝 loader_function_example.py

Purpose: Example implementation for loading models from safetensors
Use Case: Reference for integrating into other projects
python

from loader_function_example import load_safetensors_model

load and recreate sklearn model
bundle = load_safetensors_model("model.safetensors")
model = bundle['model']  # sklearn LogisticRegression
scaler = bundle['scaler']  # sklearn StandardScaler

### 6. 📊 model_weights.json

Purpose: Safe, human-readable representation of model weights
Use Case: Reviewers can inspect without security risks
json

{
  "dimensions": [267, 346, 43, 146, 204, 38, 419],
  "weights": [0.802, 0.356, 0.056, -0.139, -0.248, -0.260, -0.280],
  "paper_reference": "Section 7.2.4 - Purified V2 Model",
  "hash": "a50d941b6209f186a74d100e45cfee7a6a3c0ee70cdb17833fbd5e293b841dc0"
}

🔄 Workflow
📈 Model Specifications
Specification	Value
Dimensions	7/512 (98.6% reduction)
Stable Indices	[267, 346, 43, 146, 204, 38, 419]
Classifier	LogisticRegression(C=0.5, class_weight='balanced')
Training Data	Italian + English PD corpora
Validation	LOSO with 62 subjects
Performance	66.1% ± 19.0% LOSO accuracy
Samples	14,161 (6,416 healthy, 7,745 PD)
🔐 Security Notes

⚠️ Warning: .pkl files can execute arbitrary code.

Safe Alternatives:

    model_weights.json - Human-readable, no execution risk

    .safetensors - Secure tensor format

    Always verify hash before using any model file

🚀 Quick Start

 1. Create model (if needed)
python model_build.py

 2. Inspect weights (generates Table X)
python looker.py

 3. Convert to safe format
python safetensor_converter.py pdhear_PURIFIED_V2_HASHED.pkl model.safetensors

# 4. Validate conversion
python safetensor_hash_validator.py model.safetensors

# 5. Use in your project (example)
python loader_function_example.py

📝 Dependencies

# Required packages
pip install scikit-learn numpy pandas joblib
pip install safetensors torch  # For safetensors support
