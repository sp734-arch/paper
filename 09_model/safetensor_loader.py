"""
Simple SafeTensors Loader for pdhear_PURIFIED_V2

Loads model from SafeTensors format without pickle dependencies.
Includes manual prediction implementation for maximum compatibility.

Usage:
    python load_safetensors_simple.py pdhear_PURIFIED_V2.safetensors
    
Returns:
    Bundle with model, scaler, indices, and raw weights for manual prediction
"""

import torch
import numpy as np
import json
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def load_pdhear_model_simple(safetensors_path):
    """
    Simple loader that avoids parameter compatibility issues.
    """
    print(f"Loading model from: {safetensors_path}")
    
    # Load tensors
    tensors = load_file(safetensors_path)
    
    # Extract metadata
    metadata_bytes = tensors['metadata'].numpy().tobytes()
    metadata = json.loads(metadata_bytes.decode('utf-8').strip('\x00'))
    
    print(f"Model Info:")
    print(f"  Paper: {metadata.get('paper_reference')}")
    print(f"  Dimensions: {metadata.get('dimensions')}")
    print(f"  Indices: {metadata.get('indices')}")
    
    # Extract numpy arrays
    weights_np = tensors['weights'].numpy()
    bias_np = tensors['bias'].numpy()
    scaler_mean_np = tensors['scaler_mean'].numpy()
    scaler_std_np = tensors['scaler_std'].numpy()
    indices_np = tensors['indices'].numpy()
    
    # Create a simple logistic regression model with correct parameters
    model = LogisticRegression(
        C=0.5,
        class_weight='balanced',
        max_iter=10000,
        solver='lbfgs',
        penalty='l2'
    )
    
    # Initialize model with loaded weights
    model.classes_ = np.array([0, 1])
    model.coef_ = weights_np
    model.intercept_ = bias_np
    model.n_features_in_ = weights_np.shape[1]
    model.n_iter_ = [10000]
    
    # Create scaler
    scaler = StandardScaler()
    scaler.mean_ = scaler_mean_np
    scaler.scale_ = scaler_std_np
    scaler.n_features_in_ = weights_np.shape[1]
    scaler.var_ = scaler.scale_ ** 2
    
    # Create bundle
    bundle = {
        'model': model,
        'scaler': scaler,
        'indices': indices_np.tolist(),
        'weights': weights_np,
        'bias': bias_np,
        'scaler_mean': scaler_mean_np,
        'scaler_std': scaler_std_np
    }
    
    print(f"Model loaded successfully!")
    print(f"  Features: {weights_np.shape[1]}")
    print(f"  Indices: {len(indices_np)}")
    
    return bundle

def predict(bundle, X):
    """
    Make predictions with the model.
    X should have shape (n_samples, n_features_total)
    where n_features_total >= max(indices) + 1
    """
    # Select features
    X_selected = X[:, bundle['indices']]
    
    # Scale
    X_scaled = (X_selected - bundle['scaler_mean']) / bundle['scaler_std']
    
    # Predict using logistic function manually
    logits = X_scaled @ bundle['weights'].T + bundle['bias']
    probabilities = 1 / (1 + np.exp(-logits)).flatten()
    predictions = (probabilities > 0.5).astype(int)
    
    return predictions, probabilities

# Quick test
if __name__ == "__main__":
    # Load model
    bundle = load_pdhear_model_simple("pdhear_PURIFIED_V2_HASHED.safetensors")
    
    # Test with random data
    max_idx = max(bundle['indices'])
    n_features = max_idx + 1
    
    print(f"\nRequired total features: {n_features}")
    print(f"Selected features (indices): {bundle['indices']}")
    
    # Generate test data
    X_test = np.random.randn(3, n_features)
    
    # Make predictions
    preds, probs = predict(bundle, X_test)
    
    print(f"\nTest predictions:")
    for i in range(len(preds)):
        print(f"  Sample {i+1}: prob={probs[i]:.4f}, pred={preds[i]} ({'PARKINSONS' if preds[i]==1 else 'HEALTHY'})")
    
    # Also test with sklearn model
    X_selected = X_test[:, bundle['indices']]
    X_scaled = bundle['scaler'].transform(X_selected)
    sklearn_probs = bundle['model'].predict_proba(X_scaled)[:, 1]
    
    print(f"\nVerification (sklearn vs manual):")
    print(f"  Probabilities match: {np.allclose(probs, sklearn_probs, rtol=1e-10)}")
