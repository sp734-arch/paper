"""
Step 2: Master Overlap Sweep - Physiological Unit Extraction
================================================================

PURPOSE:
--------
Extracts HeAR embeddings at three overlap densities (10%, 50%, 75%) for emotional
speech data (Spanish/English) to generate the data for Figure 1 in the paper.

Figure 1 ("Stability-first auditing of foundation audio representations") requires
demonstrating that different temporal sampling strategies yield consistent
physiological units when the underlying signal is stable. This script implements
the "Master Sweep" concept: systematically varying window overlap to test whether
physiological structure persists across sampling densities.

PAPER REFERENCE:
----------------
Section 2: Principles for Physiological Measurement Validity

RELATION TO 01_extract_hear_staircase.py:
------------------------------------------
- COMPLEMENTARY DESIGN: 01 tests stability across temporal densities for PD speech
- THIS SCRIPT: Tests stability across temporal densities for emotional speech (ES/EN)
- SHARED CORE HYPOTHESIS: If a dimension encodes stable physiology, it should
  remain consistent whether we sample densely (75% overlap) or sparsely (10% overlap)
- DIFFERENT CONTEXT: This uses emotional speech data to test generalizability
  of the stability principle beyond pathological speech

KEY SCIENTIFIC DECISIONS:
-------------------------
1. Three overlap densities mirror staircase design:
   - 75% overlap: "Deep Mining" - captures subtle transitions, respiratory tremors
   - 50% overlap: Balanced production setting
   - 10% overlap: "Clean" set - ensures unique 2-second clips for pure LOSO validation
2. Language-aware naming (ES/EN): Enables testing invariance to language confounds
3. Voice Activity Detection (VAD): Removes non-speech segments using amplitude threshold
   (top_db=25) rather than RMS-based silence gate
4. Subject-aware extraction: Maintains subject identity for later LOSO analysis

OUTPUT STRUCTURE:
-----------------
/app/data/processed/
    ovlap_10/  ← 10% overlap (hop=28,800 samples) - Clean, unique samples
    ovlap_50/  ← 50% overlap (hop=16,000 samples) - Balanced production setting  
    ovlap_75/  ← 75% overlap (hop=8,000 samples)  - High redundancy, deep mining

    File naming: {SUBJECT_ID}_{LANGUAGE}_{SEQUENCE_ID:05d}.npy
    Example: S001_EN_00042.npy

    This naming convention enables:
    1. Subject-level grouping for LOSO validation
    2. Language-based analysis for invariance testing
    3. Sequence tracking for temporal analysis

USAGE:
------
1. Ensure HeAR model is available at /app/models
2. Place raw emotional speech files in /app/data/raw/
   - Spanish files should contain "ses-es" in path
   - English files should contain "ses-ed" in path
3. Run: python Spanish_English_Execution.py
4. Output will be used for Figure 3 Identifying a physiological subspace

AUTHORS: Jim McCormack
CREATED: 02/07/2026

"""

import os
import numpy as np
import tensorflow as tf
import librosa
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# BLACKWELL GPU CLUSTER OPTIMIZATION
# ============================================================================
# Disable XLA and reduce TensorFlow verbosity for Blackwell cluster compatibility
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================================
# MODEL LOADING
# ============================================================================
MODEL_PATH = "/app/models"
print("⏳ Loading HeAR foundation model...")
print(f"   Model path: {MODEL_PATH}")
print(f"   Expected input: 16 kHz mono, 2-second clips")

model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serving_default"]
print("✅ HeAR model loaded successfully\n")

# ============================================================================
# EXTRACTION FUNCTION
# ============================================================================

def extract_with_overlap(f_path, overlap_pct):
    # Extract metadata for LOSO and invariance testing
    sub_id = f_path.stem.split('_')[0]  # Subject ID from filename
    lang = "EN" if "ses-ed" in str(f_path) else "ES"  # Language from path
    
    # Load audio at HeAR's native sample rate
    audio, _ = librosa.load(f_path, sr=16000)
    
    # Voice Activity Detection (VAD) - remove non-speech regions
    # Uses amplitude-based detection (top_db=25) rather than RMS-based gate
    intervals = librosa.effects.split(audio, top_db=25)
    
    # Calculate hop size based on overlap percentage
    # Formula: hop = window_size * (1 - overlap/100)
    hop_length = int(32000 * (1 - (overlap_pct / 100)))
    
    chunks = []
    
    # Extract overlapping windows from each speech interval
    for start, end in intervals:
        seg = audio[start:end]
        
        # Slide window through speech segment
        for i in range(0, len(seg) - 32000 + 1, hop_length):
            chunk = seg[i:i+32000]
            chunks.append(chunk)
            
    if not chunks:
        return 0  # No speech segments found
    
    # Batch inference for efficiency
    # Convert all chunks to TensorFlow tensor in single batch
    tensor = tf.convert_to_tensor(np.array(chunks), dtype=tf.float32)
    output = infer(x=tensor)
    
    # Extract embeddings (handles different output key names)
    embeddings = output[list(output.keys())[0]].numpy()
    
    # Save embeddings with structured naming
    out_dir = f"/app/data/processed/ovlap_{overlap_pct}"
    os.makedirs(out_dir, exist_ok=True)
    
    for idx, emb in enumerate(embeddings):
        # Naming: SUBJECT_LANGUAGE_SEQUENCE.npy
        # Enables tracking for LOSO and temporal analysis
        fn = f"{sub_id}_{lang}_{idx:05d}.npy"
        np.save(os.path.join(out_dir, fn), emb)
    
    return len(chunks)

# ============================================================================
# MAIN EXECUTION LOOP
# ============================================================================

def run_master_sweep():  
    # Find all raw audio files
    raw_files = list(Path("/app/data/raw").rglob("*.wav"))
    
    if not raw_files:
        print("❌ No raw audio files found in /app/data/raw/")
        return
    
    print(f"📊 Found {len(raw_files)} raw audio files for processing")
    print(f"🎯 Testing three overlap densities: [10%, 50%, 75%]")
    print()
    
    # Three overlap densities mirror staircase design from paper
    overlaps = [10, 50, 75]
    
    # Process each overlap density
    for ov in overlaps:
        print(f"\n{'='*60}")
        print(f"🔥 STAIRCASE LAYER: {ov}% Overlap")
        print(f"   Hop size: {int(32000 * (1 - (ov / 100)))} samples")
        print(f"   Window overlap: {ov}%")
        
        if ov == 75:
            print("   MODE: Deep Mining (high redundancy for subtle transitions)")
        elif ov == 10:
            print("   MODE: Clean Set (unique samples for pure LOSO validation)")
        else:
            print("   MODE: Balanced Production Setting")
        print(f"{'='*60}")
        
        total_slices = 0
        
        # Process each file with current overlap setting
        for f in tqdm(raw_files, desc=f"   Processing {ov}% overlap"):
            total_slices += extract_with_overlap(f, ov)
        
        print(f"✅ Finished {ov}%: Created {total_slices} physiological units.")
    
    print(f"\n{'='*70}")
    print(f"🎉 MASTER SWEEP COMPLETE")
    print(f"   Output directory: /app/data/processed/")
    print(f"   Next step: Use ovlap_* directories for Figure 1 analysis")
    print(f"{'='*70}\n")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_master_sweep()
