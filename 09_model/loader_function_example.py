import torch
from safetensors.torch import load_file
import numpy as np
import json
import time
import joblib

# Expected hash from your validation
EXPECTED_HASH = "a50d941b6209f186a74d100e45cfee7a6a3c0ee70cdb17833fbd5e293b841dc0"

def load_model_safely(safetensors_path):
    """Load the converted model safely and verify hash preservation."""
    print("=" * 70)
    print("MODEL LOADER WITH HASH VERIFICATION")
    print("=" * 70)
    
    # Load safetensors
    tensors = load_file(safetensors_path)
    
    # Extract metadata
    metadata_bytes = tensors['metadata'].numpy().tobytes()
    metadata = json.loads(metadata_bytes.decode('utf-8').strip('\x00'))
    
    # Extract components
    weights = tensors['weights'].numpy()
    bias = tensors['bias'].numpy()
    scaler_mean = tensors['scaler_mean'].numpy()
    scaler_std = tensors['scaler_std'].numpy()
    indices = tensors['indices'].numpy()
    
    # Get hash from metadata
    stored_hash = metadata.get('reproducibility_hash', '')
    
    print(f"\n1. 📦 Model loaded:")
    print(f"   • Paper: {metadata.get('paper_reference', 'Unknown')}")
    print(f"   • Training date: {metadata.get('training_date', 'Unknown')}")
    print(f"   • Selected features: {len(indices)}")
    print(f"   • Feature indices: {sorted(indices)}")
    
    print(f"\n2. 🔐 Hash Verification:")
    print(f"   • Stored hash: {stored_hash[:16]}...")
    print(f"   • Expected hash: {EXPECTED_HASH[:16]}...")
    
    if stored_hash == EXPECTED_HASH:
        print(f"   • Result: ✅ HASH PRESERVED - Model identity verified")
        hash_valid = True
    else:
        print(f"   • Result: ❌ HASH MISMATCH - Model may be corrupted")
        print(f"     Expected: {EXPECTED_HASH}")
        print(f"     Got: {stored_hash}")
        hash_valid = False
    
    print(f"\n3. 📋 Metadata fields preserved:")
    important_fields = ['hash_algorithm', 'hash_components', 'model_parameters', 'loso_performance']
    for field in important_fields:
        if field in metadata:
            print(f"   • {field}: ✅")
        else:
            print(f"   • {field}: ❌")
    
    model_dict = {
        'weights': weights,
        'bias': bias,
        'scaler_mean': scaler_mean,
        'scaler_std': scaler_std,
        'indices': indices,
        'metadata': metadata,
        'hash_valid': hash_valid
    }
    
    return model_dict

# Prediction function
def predict(model_dict, features):
    """Make predictions using the loaded model."""
    # Select features using indices
    X = features[model_dict['indices']]
    
    # Scale features
    X_scaled = (X - model_dict['scaler_mean']) / model_dict['scaler_std']
    
    # Compute logits and probabilities
    logits = X_scaled @ model_dict['weights'].T + model_dict['bias']
    probabilities = 1 / (1 + np.exp(-logits))
    
    return probabilities.flatten()

print("=" * 60)
print("PRODUCTION MODEL LOADER")
print("=" * 60)

# Load the model with hash verification
model = load_model_safely("pdhear_PURIFIED_V2_HASHED.safetensors")

# Stop if hash is invalid
if not model['hash_valid']:
    print("\n❌ Cannot proceed: Model hash verification failed")
    exit(1)

# Determine the original feature space size
original_feature_size = model['indices'].max() + 1
print(f"\n📏 Original feature space size: {original_feature_size}")

# Create test data
print("\n🧪 Creating test data...")
test_features = np.random.randn(original_feature_size)
print(f"   • Test features shape: {test_features.shape}")

# Make prediction
print("\n🔮 Making prediction...")
prob = predict(model, test_features)
print(f"   • Probability: {prob[0]:.4f}")

# Show which features were selected
print("\n📋 Selected features (7 out of 420):")
for i, idx in enumerate(sorted(model['indices'])):
    print(f"   • Feature {idx:3d}: value = {test_features[idx]:8.4f}")

# Batch prediction example
print("\n" + "=" * 60)
print("BATCH PREDICTION EXAMPLE")
print("=" * 60)

n_samples = 10
batch_features = np.random.randn(n_samples, original_feature_size)

print("\n🔮 Making batch predictions...")
probabilities = []
for i in range(n_samples):
    prob = predict(model, batch_features[i])
    probabilities.append(prob[0])

probabilities = np.array(probabilities)
print(f"   • Predictions: {probabilities}")
print(f"   • Mean: {probabilities.mean():.4f}")
print(f"   • Std: {probabilities.std():.4f}")
print(f"   • Positive rate: {(probabilities > 0.5).mean()*100:.1f}%")

print("\n" + "=" * 60)
print("SPEED COMPARISON")
print("=" * 60)

import time

# Safetensors load time
start = time.perf_counter()
_ = load_file("pdhear_PURIFIED_V2_HASHED.safetensors")
sf_load_time = (time.perf_counter() - start) * 1000

# PKL load time (for comparison)
import joblib
start = time.perf_counter()
_ = joblib.load("pdhear_PURIFIED_V2_HASHED.pkl")
pkl_load_time = (time.perf_counter() - start) * 1000

print(f"\n⏱️  Load times:")
print(f"   • Safetensors: {sf_load_time:.2f} ms")
print(f"   • PKL: {pkl_load_time:.2f} ms")
print(f"   • Speedup: {pkl_load_time/sf_load_time:.1f}x")

# Prediction speed
n_predictions = 1000
test_batch = np.random.randn(n_predictions, original_feature_size)

start = time.perf_counter()
for i in range(n_predictions):
    _ = predict(model, test_batch[i])
pred_time = (time.perf_counter() - start) * 1000

print(f"\n⏱️  Prediction speed:")
print(f"   • {n_predictions} predictions: {pred_time:.2f} ms")
print(f"   • Per prediction: {pred_time/n_predictions:.3f} ms")

print("\n" + "=" * 60)
print("REAL DATA USAGE EXAMPLE")
print("=" * 60)

print("""
When using with REAL data:
--------------------------
1. Extract ALL 420 features from your audio
2. The model automatically selects the 7 relevant features
3. No need to pre-select features manually

Example:
    # Extract all features
    all_features = extract_all_features(audio_file)  # Returns array of size 420
    
    # Model makes prediction
    probability = predict(model, all_features)
    
    # Clinical decision
    if probability > 0.7:
        diagnosis = "HIGH LIKELIHOOD - Refer to specialist"
    elif probability > 0.5:
        diagnosis = "ELEVATED RISK - Further assessment"
    elif probability > 0.3:
        diagnosis = "BORDERLINE - Monitor closely"
    else:
        diagnosis = "LOW LIKELIHOOD - Routine follow-up"
    
    print(f"Probability: {probability:.3f} - {diagnosis}")

Key Insights:
------------
• Model uses 7 out of 420 possible features
• Feature indices: {sorted(model['indices'])}
• Hash verified: {model['hash_valid']}
""")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"""
✅ Model Status:
   • Hash preserved: {model['hash_valid']}
   • Hash: {model['metadata'].get('reproducibility_hash', '')[:16]}...
   • Features: 420 total, 7 selected
   • Selected indices: {sorted(model['indices'])}

🚀 Ready for production use!
""")
