#!/usr/bin/env python3
"""
Convert pdhear_PURIFIED_V2.pkl to SafeTensors Format

Converts the Purified V2 model from pickle format to SafeTensors for:
- Security (no arbitrary code execution)
- Inspectability (can view without loading)
- Compatibility (cross-platform, fast loading)
"""

# simple_converter.py
import joblib
import torch
import numpy as np
import json
from safetensors.torch import save_file
from pathlib import Path

def convert_pkl_to_safetensors_simple(pkl_path):
    """Simple converter without Unicode issues"""
    print(f"Converting {pkl_path} to safetensors...")
    
    # Load with joblib
    data = joblib.load(pkl_path)
    
    # Extract model components
    model = data['model']
    scaler = data['scaler']
    indices = data['indices']
    
    # Create metadata
    metadata = {
        'paper_reference': data.get('paper_reference', 'pdhear_PURIFIED_V2'),
        'dimensions': model.coef_.shape[1],
        'indices': indices,
        'model_parameters': data.get('model_parameters', {}),
        'performance': data.get('loso_performance', {})
    }
    
    # Create tensors
    tensors = {
        'weights': torch.tensor(model.coef_, dtype=torch.float32),
        'bias': torch.tensor(model.intercept_, dtype=torch.float32),
        'scaler_mean': torch.tensor(scaler.mean_, dtype=torch.float32),
        'scaler_std': torch.tensor(scaler.scale_, dtype=torch.float32),
        'indices': torch.tensor(indices, dtype=torch.int32),
    }
    
    # Add metadata
    tensors['metadata'] = torch.tensor(
        bytearray(json.dumps(metadata), 'utf-8'),
        dtype=torch.uint8
    )
    
    # Save
    output_path = pkl_path.replace('.pkl', '.safetensors')
    save_file(tensors, output_path)
    
    print(f"Saved to {output_path}")
    print(f"Model has {model.coef_.shape[1]} features")
    print(f"Feature indices: {indices}")
    
    return output_path

if __name__ == "__main__":
    convert_pkl_to_safetensors_simple("pdhear_PURIFIED_V2_HASHED.pkl")
