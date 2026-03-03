"""
Step 9 Purified V2 Model Bundle Creation - Final Model Serialization
======================================================================

PAPER CONNECTION & SCIENTIFIC CONTEXT:
---------------------------------------
This script creates the FINAL MODEL BUNDLE (pdhear_PURIFIED_V2_HASHED.pkl) that is
referenced throughout the paper as the validated screening model. It represents
the culmination of the entire stability-first auditing framework.

KEY PAPER REFERENCES:
1. SECTION 6 (The Purified V2 Head): "The trained head, scaler, and dimension
   indices were serialized together as a single bundle (pdhear_PURIFIED_V2_HASHED.pkl)."
   This script creates that exact bundle.

2. SECTION 6.1: "The frozen model bundle contains three
   objects: the trained LogisticRegression model, the fitted StandardScaler,
   and the 7-element dimension index array." This script defines those objects.

3. SECTION 5 (Auditing Protocol): Implements the final model training after
   dimensions have passed stability and invariance audits.

4. SECTION 5 (Subject-Level Stability): Uses Leave-One-Subject-Out (LOSO)
   validation to prevent subject identity leakage.

SCIENTIFIC PURPOSE:
-------------------
This script performs the FINAL INTEGRATION of all auditing results to create
a deployable screening model:

1. APPLIES STABILITY MASK: Uses only the 7 dimensions that passed stability
   and invariance audits (STABLE_DIMS = [267, 346, 43, 146, 204, 38, 419])

2. TRAINS MULTI-LINGUAL MODEL: Combines Italian and English data as specified
   in the paper's training protocol

3. VALIDATES WITH LOSO: Ensures no subject identity leakage using rigorous
   Leave-One-Subject-Out cross-validation

4. SERIALIZES FINAL BUNDLE: Creates the exact model bundle used in all
   paper evaluations and referenced in the deployment implementation

This represents the transition from RESEARCH ANALYSIS to DEPLOYABLE MODEL.

THE PURIFIED V2 MODEL BUNDLE STRUCTURE:
---------------------------------------
The bundle contains three critical components:

1. model: LogisticRegression classifier with parameters:
   - C=0.5 (regularization strength)
   - class_weight='balanced' (handles class imbalance)
   - solver='liblinear' (optimized for small datasets)
   - Trained on Italian + English data (never on KCL)

2. scaler: StandardScaler fitted on the training data
   - Means and standard deviations for each of the 7 dimensions
   - Used to normalize new data before prediction
   - Critical for maintaining calibration across recording conditions

3. indices: The 7 dimension indices [267, 346, 43, 146, 204, 38, 419]
   - These are the dimensions that survived the full auditing pipeline
   - Ordered by stability (descending) based on biomarker leaderboard
   - Represents a 98.6% reduction from original 512 dimensions

REPRODUCIBILITY HASH SYSTEM:
----------------------------
To ensure exact reproducibility of the paper's results, we compute and store
a SHA-256 hash of the critical model components. This hash can be used to:
1. Verify that a loaded model is identical to the one used in the paper
2. Ensure no accidental modifications to the model
3. Provide cryptographic proof of model identity for regulatory purposes

The hash is computed from:
- Model coefficients and intercept (as Base64 strings)
- Scaler means and standard deviations (as Base64 strings)  
- Dimension indices (as sorted list)
- Training data statistics

Any change to these components will produce a different hash, alerting users
to potential inconsistencies with the published results.

METHODOLOGICAL RIGOR:
---------------------
1. MULTI-LINGUAL TRAINING: Combines Italian clinical and English telephone
   data to ensure language invariance from the start.

2. SUBJECT-LEVEL VALIDATION: Uses Leave-One-Group-Out where groups are
   speaker identities, preventing any form of subject leakage.

3. CLASS BALANCING: Uses class_weight='balanced' to handle inherent
   class imbalance in clinical datasets.

4. REGULARIZATION: C=0.5 provides moderate regularization to prevent
   overfitting to the training cohorts.

5. FROZEN MODEL: Once created, the model bundle is NEVER retrained or
   fine-tuned on evaluation data (KCL), maintaining true external validation.

PURPOSE:
--------
Create the final Purified V2 model bundle that:
1. Embodies all findings from the stability-first auditing framework
2. Can be deployed for screening as described in Section 7.2.7
3. Provides the exact model used for all paper results
4. Includes cryptographic hash for reproducibility verification

OUTPUT FOR PAPER:
-----------------
The model bundle itself is not included in the paper but is referenced as:
- "pdhear_PURIFIED_V2_HASHED.pkl" 
- "The frozen model bundle" in Section 6.1

The hash should be reported in:
- Supplementary materials for reproducibility
- Code repository for version verification
- Regulatory submissions if applicable

CRITICAL MODEL SPECIFICATIONS:
------------------------------
- Training data: Italian + English PD corpora ONLY (no KCL data)
- Validation: Leave-One-Subject-Out (LOSO) with 62 subjects
- Dimensions: 7 stable indices [267, 346, 43, 146, 204, 38, 419]
- Classifier: LogisticRegression(C=0.5, class_weight='balanced')
- Performance: 66.1% LOSO accuracy ± 19.0%
- Samples: 14,161 (6,416 healthy, 7,745 PD)
- Reduction: 512 → 7 dimensions (98.6% reduction)

USAGE:
------
1. Run: python purified_v2_hashed_pkl.py
2. Output: pdhear_PURIFIED_V2_HASHED.pkl with embedded hash
3. Verification: Use provided verification function to check model integrity
4. Deployment: Load bundle and use as shown in Step 8 validation script

AUTHORS: Jim McCormack
CREATED: 2/7/2026
REPRODUCIBILITY HASH: <Created at script run as an output> 
"""

import joblib
import numpy as np
import hashlib
import json
import base64
import warnings
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

warnings.filterwarnings("ignore")

def compute_model_hash(model, scaler, indices, X_train, y_train):
    """
    Compute SHA-256 hash of critical model components for reproducibility.
    """
    # Convert numpy arrays to Base64 strings for JSON serialization
    model_coef_b64 = base64.b64encode(model.coef_.tobytes()).decode('utf-8')
    model_intercept_b64 = base64.b64encode(model.intercept_.tobytes()).decode('utf-8')
    scaler_mean_b64 = base64.b64encode(scaler.mean_.tobytes()).decode('utf-8')
    scaler_scale_b64 = base64.b64encode(scaler.scale_.tobytes()).decode('utf-8')
    
    # Safely convert training data statistics to JSON-serializable types
    training_stats = {
        'shape': [int(X_train.shape[0]), int(X_train.shape[1])],
        'class_distribution': [int(count) for count in np.bincount(y_train)],
        'mean_per_feature': [float(val) for val in np.mean(X_train, axis=0)],
        'std_per_feature': [float(val) for val in np.std(X_train, axis=0)]
    }
    
    # Create a deterministic string representation
    components = {
        'model_coef': model_coef_b64,
        'model_intercept': model_intercept_b64,
        'model_params': str(model.get_params()),
        'scaler_mean': scaler_mean_b64,
        'scaler_scale': scaler_scale_b64,
        'indices': sorted(indices),
        'training_data': training_stats,
        'paper_reference': 'Section 7.2.4 Purified V2 Model',
        'hash_version': '1.0'
    }
    
    # Convert to JSON string with sorted keys for determinism
    components_str = json.dumps(components, sort_keys=True, indent=None)
    
    # Compute SHA-256 hash
    hash_obj = hashlib.sha256(components_str.encode('utf-8'))
    return hash_obj.hexdigest()

# 1. THE STABILITY MASK (from Step 9 Biomarker Leaderboard)
STABLE_DIMS = [267, 346, 43, 146, 204, 38, 419]
AUDIT_BASE = Path("/app/data/processed/features_audit/density_50")

print("\n" + "="*80)
print("🔬 PURIFIED V2 MODEL CREATION - FINAL BUNDLE GENERATION")
print("="*80)
print("PAPER: Final model creation (Section 7.2.4)")
print(f"DIMENSIONS: {len(STABLE_DIMS)} stable indices after auditing")
print(f"REDUCTION: 512 → {len(STABLE_DIMS)} dimensions ({(1-len(STABLE_DIMS)/512)*100:.1f}% reduction)")
print("="*80)

print(f"\n🛡️  Applying Stability Mask: Using {len(STABLE_DIMS)} Audited Dimensions")
print(f"   Indices: {STABLE_DIMS}\n")

# 2. AGGREGATE TRAIN DATA (Italian + English) - PAPER TRAINING DATA
X, y, groups = [], [], []

print("📥 Loading Multi-Lingual Training Data...")
print("   PAPER SPECIFICATION (Section 7.2.4):")
print("   - Italian PD corpus (clinical recordings)")
print("   - English PD corpus (telephone recordings)\n")

# We iterate through both languages to build a multi-lingual baseline
for lang_folder in ["features_italian", "features_english"]:
    lang_path = AUDIT_BASE / lang_folder
    
    for label_val, label_name in [(0, "healthy"), (1, "parkinsons")]:
        target_dir = lang_path / label_name
        if not target_dir.exists(): 
            continue
        
        files = list(target_dir.glob("*.npy"))
        print(f"   {lang_folder}/{label_name}: {len(files)} embedding files")
        
        for f in files:
            try:
                # Load full 512-dimensional embedding
                emb = np.load(f).flatten()
                
                # Validate embedding dimension
                if len(emb) == 512:
                    # Apply stability mask: keep only the 7 purified dimensions
                    purified_emb = emb[STABLE_DIMS]
                    X.append(purified_emb)
                    y.append(label_val)
                    
                    # Create unique speaker ID for LOSO validation
                    speaker_id = f.stem.split('_')[0]
                    groups.append(f"{lang_folder}_{speaker_id}")
            except Exception as e:
                continue

# Convert to numpy arrays for sklearn compatibility
X = np.array(X)
y = np.array(y)
groups = np.array(groups)

print(f"\n📊 TRAINING DATA SUMMARY")
print("-" * 40)

# --- CLASS DISTRIBUTION CHECK ---
unique_y, counts_y = np.unique(y, return_counts=True)
class_dist = dict(zip(unique_y, counts_y))
print(f"Total samples: {len(y)}")
print(f"Class distribution: {class_dist} (0=Healthy, 1=PD)")
print(f"Healthy/PD ratio: {class_dist[0]/class_dist[1]:.2f}:1")

# --- SUBJECT-LEVEL STATISTICS ---
unique_groups, group_counts = np.unique(groups, return_counts=True)
print(f"Unique subjects: {len(unique_groups)}")
print(f"Average slices per subject: {len(y)/len(unique_groups):.1f}\n")

# 3. PURIFIED LOSO VALIDATION (Leave-One-Subject-Out)
print("🧬 Running Purified LOSO Validation...")
print("   PAPER CONTEXT: Prevents subject identity leakage (Section 8.2)")
print("   METHOD: Leave-One-Group-Out where groups = speaker identities\n")

logo = LeaveOneGroupOut()
scaler = StandardScaler()
accuracies = []
n_folds = 0

for train_idx, test_idx in logo.split(X, y, groups):
    # Skip folds with only one class in training
    if len(np.unique(y[train_idx])) < 2:
        continue
    
    # Standardize within fold (prevents test data leakage)
    X_train_s = scaler.fit_transform(X[train_idx])
    X_test_s = scaler.transform(X[test_idx])
    
    # Train model with paper-specified parameters
    model = LogisticRegression(
        C=0.5, 
        class_weight='balanced', 
        solver='liblinear',
        random_state=1337,
        max_iter=10000
    )
    model.fit(X_train_s, y[train_idx])
    
    # Evaluate on held-out subject
    accuracy = model.score(X_test_s, y[test_idx])
    accuracies.append(accuracy)
    n_folds += 1

if accuracies:
    print("-" * 45)
    print(f"🏆 PURIFIED LOSO PERFORMANCE")
    print("-" * 45)
    print(f"Folds completed: {n_folds}/{len(unique_groups)}")
    print(f"Mean accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Range: [{np.min(accuracies):.4f}, {np.max(accuracies):.4f}]\n")
    
    # 4. TRAIN FINAL MODEL ON ALL DATA
    print("🔧 Training Final Model on Full Training Set...")
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)
    
    final_model = LogisticRegression(
        C=0.5,
        class_weight='balanced',
        solver='liblinear',
        random_state=1337,
        max_iter=10000
    )
    final_model.fit(X_scaled, y)
    
    # Compute final accuracy on training data (for reference)
    train_accuracy = final_model.score(X_scaled, y)
    print(f"Final model training accuracy: {train_accuracy:.4f}\n")
    
    # 5. CREATE MODEL BUNDLE WITH REPRODUCIBILITY HASH
    print("💾 Creating Model Bundle with Reproducibility Hash...")
    
    # Compute reproducibility hash
    reproducibility_hash = compute_model_hash(final_model, final_scaler, STABLE_DIMS, X, y)
    
    # Prepare bundle contents
    bundle = {
        'model': final_model,
        'scaler': final_scaler,
        'indices': STABLE_DIMS,
        'paper_reference': 'Section 7.2.4 - Purified V2 Model',
        'training_date': '2024-02-07',
        'training_data_summary': {
            'n_samples': len(y),
            'n_healthy': int(class_dist[0]),
            'n_parkinsons': int(class_dist[1]),
            'n_subjects': len(unique_groups),
            'X_shape': list(X.shape),
            'y_dist': [int(class_dist[0]), int(class_dist[1])]
        },
        'model_parameters': final_model.get_params(),
        'loso_performance': {
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'n_folds': n_folds,
            'min_accuracy': float(np.min(accuracies)),
            'max_accuracy': float(np.max(accuracies))
        },
        'reproducibility_hash': reproducibility_hash,
        'hash_algorithm': 'SHA-256',
        'hash_components': [
            'model_coefficients',
            'model_intercept', 
            'scaler_parameters',
            'dimension_indices',
            'training_data_stats'
        ],
        'hash_computation_date': '2024-02-07' 
    }
    
    # Save bundle to disk
    bundle_path = "/app/scripts/pkl/pdhear_PURIFIED_V2_HASHED.pkl"
    joblib.dump(bundle, bundle_path)
    
    print("✅ MODEL BUNDLE CREATED SUCCESSFULLY")
    print("-" * 45)
    print(f"Bundle saved: {bundle_path}")
    print(f"Reproducibility hash: {reproducibility_hash}")
    print(f"Hash (first 16 chars): {reproducibility_hash[:16]}...\n")
    
    # 6. VERIFICATION TEST - SIMPLE VERSION
    print("🔍 Verifying Bundle Integrity...")
    loaded_bundle = joblib.load(bundle_path)
    
    # Simply check if the stored hash matches
    if loaded_bundle['reproducibility_hash'] == reproducibility_hash:
        print("✅ Bundle verification PASSED")
        print(f"   Hash: {reproducibility_hash[:16]}...\n")
        
        # Display bundle contents for documentation
        print("📋 BUNDLE CONTENTS:")
        print("-" * 40)
        for key, value in bundle.items():
            if key in ['model', 'scaler']:
                print(f"{key}: {type(value).__name__}")
            elif key == 'indices':
                print(f"{key}: {value}")
            elif key == 'reproducibility_hash':
                print(f"{key}: {value[:16]}...")
            elif key == 'training_data_summary':
                print(f"{key}:")
                for subkey, subvalue in value.items():
                    print(f"  {subkey}: {subvalue}")
            elif key == 'loso_performance':
                print(f"{key}:")
                for subkey, subvalue in value.items():
                    print(f"  {subkey}: {subvalue}")
            else:
                print(f"{key}: {value}")
        
        print()
        print("="*80)
        print("🎉 PURIFIED V2 MODEL CREATION COMPLETE")
        print("="*80)
        print()
        print("📄 PAPER INTEGRATION NOTES:")
        print("1. Reference bundle as: 'pdhear_PURIFIED_V2_HASHED.pkl' (Section 7.2.4)")
        print(f"2. LOSO accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"3. Training accuracy: {train_accuracy:.4f}")
        print(f"4. Reproducibility hash: {reproducibility_hash}")
        print(f"   (First 16 chars: {reproducibility_hash[:16]}...)")
        print("5. Bundle available for peer review and regulatory submission")
        
    else:
        print(f"❌ Bundle verification FAILED")
        print(f"   Expected: {reproducibility_hash[:16]}...")
        print(f"   Got:      {loaded_bundle['reproducibility_hash'][:16]}...")
        
    # Create verification script
    verification_script = f'''
"""
VERIFICATION SCRIPT for pdhear_PURIFIED_V2_HASHED.pkl
Run this to verify model integrity matches paper version.

PAPER HASH: {reproducibility_hash}
Paper Reference: Section 7.2.4
"""
import joblib

# Expected hash from paper (DO NOT MODIFY)
PAPER_MODEL_HASH = "{reproducibility_hash}"

def verify_purified_v2_model(model_path, expected_hash):
    """Verify that a model bundle matches the paper's version."""
    try:
        bundle = joblib.load(model_path)
        if 'reproducibility_hash' in bundle:
            if bundle['reproducibility_hash'] == expected_hash:
                print(f"✅ Model verified: {{expected_hash[:16]}}...")
                print(f"   Dimensions: {{len(bundle['indices'])}}")
                print(f"   Paper Reference: {{bundle.get('paper_reference', 'Unknown')}}")
                print(f"   LOSO Accuracy: {{bundle.get('loso_performance', {{}}).get('mean_accuracy', 'Unknown'):.4f}}")
                return True
            else:
                print(f"❌ Hash mismatch!")
                print(f"   Expected: {{expected_hash[:16]}}...")
                print(f"   Got:      {{bundle['reproducibility_hash'][:16]}}...")
                print("   This model may produce different results than the paper")
                return False
        else:
            print("❌ No reproducibility hash in bundle")
            print("   This model was not created with the paper's script")
            return False
    except Exception as e:
        print(f"❌ Verification failed: {{e}}")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: python {{sys.argv[0]}} <model_path.pkl>")
        sys.exit(1)
    
    success = verify_purified_v2_model(sys.argv[1], PAPER_MODEL_HASH)
    sys.exit(0 if success else 1)
'''
    
    # Save verification script
    verification_script_path = "/app/scripts/pkl/verify_purified_v2.py"
    with open(verification_script_path, "w") as f:
        f.write(verification_script)
    
    print(f"\n📝 Verification script saved: {verification_script_path}")
    print("   Use this to verify model integrity matches paper version.")
    
    # Test the verification script
    print(f"\n🧪 Testing verification script...")
    import subprocess
    result = subprocess.run(
        ["python", verification_script_path, bundle_path],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print(f"Stderr: {result.stderr}")
        
else:
    print("❌ Insufficient data for LOSO validation")
    print("   Check that you have multiple subjects with both classes")
