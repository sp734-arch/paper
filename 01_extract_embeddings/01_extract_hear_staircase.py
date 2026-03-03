"""
Step 1: Multi-Density HeAR Embedding Extraction
================================================

PURPOSE:
--------
Extracts HeAR embeddings at three temporal sampling densities (10%, 50%, 75% overlap)
to test whether physiological signal remains stable under different windowing strategies.

The "staircase" design tests a core hypothesis: if a dimension encodes stable physiology,
it should remain consistent whether we sample densely or sparsely. If a dimension only
appears under dense sampling, it may be capturing transient acoustic artifacts rather
than underlying vocal motor structure.

PAPER REFERENCE:
----------------
Section 7.1.3 (Embedding Extraction)
Section 7.1.2 (Quality Control - silence gate)

KEY SCIENTIFIC DECISIONS:
-------------------------
1. Pre-emphasis: Enhances high-frequency content (vocal tract resonances)
2. Silence gate (-50 dBFS): Removes non-speech segments without per-speaker normalization
3. Fixed 2-second windows: HeAR's native input size
4. Subject-aware naming: Enables Leave-One-Subject-Out validation later

OUTPUT STRUCTURE:
-----------------
features_audit/
    density_10/  ← 10% overlap (hop=28,800 samples) - Baseline
    density_50/  ← 50% overlap (hop=16,000 samples) - Production
    density_75/  ← 75% overlap (hop=8,000 samples)  - High-density test
        features_kcl/
            healthy/
            parkinsons/
        features_italian/
            healthy/
            parkinsons/
        features_english/
            healthy/
            parkinsons/

USAGE:
------
1. Download HeAR model to ./models/hear (from Hugging Face: google/hear)
2. Organize data into ./data/[cohort]/[HC|PD]/*.wav
3. Run: python extract_embeddings_multidensity.py
4. Wait: ~2-4 hours for full extraction (depends on GPU)

AUTHORS: Jim McCormack
CREATED: 02/07/2026
"""

import os
import glob
import numpy as np
import tensorflow as tf
import librosa
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to HeAR foundation model (download from HuggingFace: google/hear)
MODEL_PATH = "./models/hear"

# Audio preprocessing constants
SAMPLE_RATE = 16000      # HeAR expects 16 kHz mono
CLIP_LENGTH = 32000      # 2 seconds at 16 kHz (HeAR's native window)
SILENCE_THRESHOLD = -50.0  # RMS loudness threshold in dBFS (absolute, not relative)

# The "staircase": Test three temporal sampling densities
# Format: (hop_size_in_samples, output_folder_name)
STAIRCASE = [
    (28800, "density_10"),  # 10% overlap  → 1.8s hop (sparse baseline)
    (16000, "density_50"),  # 50% overlap  → 1.0s hop (production setting)
    (8000,  "density_75")   # 75% overlap  → 0.5s hop (dense robustness test)
]

# Root output directory for all extracted features
ROOT_OUT = "./features_audit"

# ============================================================================
# LOAD HEAR MODEL
# ============================================================================

print("⏳ Loading HeAR foundation model...")
print(f"   Model path: {MODEL_PATH}")
print(f"   Expected input: 16 kHz mono, 2-second clips")

model = tf.saved_model.load(MODEL_PATH)
infer_fn = model.signatures["serving_default"]

print("✅ HeAR model loaded successfully\n")

# ============================================================================
# SUBJECT IDENTITY EXTRACTION (FOR LOSO VALIDATION)
# ============================================================================

def get_subject_id(file_path, cohort_key):
    """
    Extract subject ID from filename using cohort-specific naming conventions.
    
    This is critical for Leave-One-Subject-Out (LOSO) cross-validation later.
    Each cohort has a different filename structure, so we parse accordingly.
    
    Args:
        file_path (str): Full path to audio file
        cohort_key (str): One of ["KCL", "English", "UCI", "Italian"]
    
    Returns:
        str: Subject ID (stable across all recordings from same person)
    
    Examples:
        KCL:     "ID02_pd_1_2_1.wav"           → "ID02"
        English: "AH_123_M65_sustained_a.wav"  → "AH_123_M65"
        Italian: "/data/Subject_001/trial1.wav" → "Subject_001"
    """
    file_base = os.path.basename(file_path)
    
    if cohort_key == "KCL":
        # KCL format: IDNN_diagnosis_hy_updrs2_updrs3.wav
        return file_base.split('_')[0]
    
    elif cohort_key in ["English", "UCI"]:
        # English format: ID_AgeSex_task.wav
        # We need both ID and AgeSex to uniquely identify subjects
        return "_".join(file_base.split('_')[:2])
    
    elif cohort_key == "Italian":
        # Italian format: recordings are organized in subject-specific folders
        # Subject ID is the parent directory name
        return os.path.basename(os.path.dirname(file_path))
    
    # Fallback for unknown cohorts: use first underscore-delimited token
    return file_base.split('_')[0]

# ============================================================================
# MAIN EXTRACTION PIPELINE
# ============================================================================

def run_staircase_extraction():
    """
    Extract HeAR embeddings at multiple temporal densities for all cohorts.
    
    This function:
    1. Iterates through three overlap densities (10%, 50%, 75%)
    2. Processes all cohorts (Italian, English, KCL)
    3. Applies silence gating to remove non-speech segments
    4. Saves one .npy file per 2-second segment
    
    The staircase design allows us to test whether stable physiological
    dimensions remain consistent across different temporal sampling rates.
    """
    
    # Define all data sources
    # Format: (file_pattern, label, cohort_key, output_subfolder)
    sources = [
        # KCL (mobile smartphone recordings, external validation set)
        ("./data/KCL/HC/*.wav", "healthy", "KCL", "features_kcl"),
        ("./data/KCL/PD/*.wav", "parkinsons", "KCL", "features_kcl"),
        
        # English (sustained vowel /a/ phonations from telephones)
        ("./data/HC_AH/**/*.wav", "healthy", "English", "features_english"),
        ("./data/PD_AH/**/*.wav", "parkinsons", "English", "features_english"),
        
        # Italian (clinical recordings: text reading + vowels)
        ("./data/15 Young Healthy Control/**/*.wav", "healthy", "Italian", "features_italian"),
        ("./data/22 Elderly Healthy Control/**/*.wav", "healthy", "Italian", "features_italian"),
        ("./data/28 People with Parkinson's disease/**/*.wav", "parkinsons", "Italian", "features_italian")
    ]
    
    # STAIRCASE LOOP: Extract at each density level
    for hop_size, density_folder in STAIRCASE:
        overlap_pct = int((1 - hop_size / CLIP_LENGTH) * 100)
        print(f"\n{'='*70}")
        print(f"🚀 STAIRCASE LAYER: {density_folder}")
        print(f"   Hop size: {hop_size} samples ({hop_size/SAMPLE_RATE:.2f}s)")
        print(f"   Overlap:  {overlap_pct}%")
        print(f"{'='*70}\n")
        
        # COHORT LOOP: Process each dataset
        for pattern, label, cohort_key, output_name in sources:
            files = glob.glob(pattern, recursive=True)
            
            if not files:
                print(f"⚠️  No files found for {cohort_key}-{label} (pattern: {pattern})")
                continue
            
            # Create output directory
            # Structure: features_audit / density_X / features_cohort / label
            save_dir = os.path.join(ROOT_OUT, density_folder, output_name, label)
            os.makedirs(save_dir, exist_ok=True)
            
            print(f"📂 Processing {cohort_key}-{label}: {len(files)} files")
            
            # FILE LOOP: Extract embeddings from each audio file
            for file_path in tqdm(files, desc=f"   🧬 {cohort_key}-{label}"):
                try:
                    # Extract subject ID (for LOSO later)
                    subject_id = get_subject_id(file_path, cohort_key)
                    file_base = os.path.basename(file_path)
                    
                    # =========================================================
                    # AUDIO PREPROCESSING
                    # =========================================================
                    
                    # Load audio at 16 kHz mono
                    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
                    
                    # Apply pre-emphasis filter (boosts high frequencies)
                    # This enhances formants and vocal tract resonances
                    audio = librosa.effects.preemphasis(audio)
                    
                    # Pad if shorter than 2 seconds
                    if len(audio) < CLIP_LENGTH:
                        audio = np.pad(audio, (0, CLIP_LENGTH - len(audio)))
                    
                    # =========================================================
                    # SLIDING WINDOW EXTRACTION
                    # =========================================================
                    
                    slice_idx = 0  # Counter for segments from this file
                    
                    for start_sample in range(0, len(audio) - CLIP_LENGTH + 1, hop_size):
                        # Extract 2-second segment
                        segment = audio[start_sample : start_sample + CLIP_LENGTH]
                        
                        # =====================================================
                        # SILENCE GATE (Quality Control)
                        # =====================================================
                        # Compute RMS loudness in dBFS
                        # -50 dBFS is an absolute threshold (not speaker-normalized)
                        # This removes silence/near-silence without biasing per-speaker
                        rms_energy = np.sqrt(np.mean(segment**2))
                        rms_loudness = 20 * np.log10(rms_energy + 1e-9)
                        
                        if rms_loudness < SILENCE_THRESHOLD:
                            # Skip silent segments (no speech content)
                            continue
                        
                        # =====================================================
                        # HEAR EMBEDDING EXTRACTION
                        # =====================================================
                        # Convert to TensorFlow tensor (batch size 1)
                        input_tensor = tf.constant(segment[np.newaxis, ...], dtype=tf.float32)
                        
                        # Run inference (returns 512-D embedding)
                        embedding = infer_fn(x=input_tensor)['output_0'].numpy().flatten()
                        
                        # =====================================================
                        # SAVE TO DISK
                        # =====================================================
                        # Filename format: {subjectID}_{originalFile}_s{sliceNumber}.npy
                        # This allows us to:
                        # 1. Group by subject for LOSO
                        # 2. Trace back to original file
                        # 3. Track slice index within file
                        output_filename = f"{subject_id}_{file_base.replace('.wav', '')}_s{slice_idx}.npy"
                        output_path = os.path.join(save_dir, output_filename)
                        
                        np.save(output_path, embedding)
                        slice_idx += 1
                
                except Exception as e:
                    print(f"❌ Error processing {file_path}: {e}")
                    continue
    
    print(f"\n{'='*70}")
    print(f"✅ STAIRCASE EXTRACTION COMPLETE")
    print(f"   Output directory: {ROOT_OUT}")
    print(f"   Next step: Stability audit (Step 7) and Invariance audit (Step 6)")
    print(f"{'='*70}\n")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_staircase_extraction()
