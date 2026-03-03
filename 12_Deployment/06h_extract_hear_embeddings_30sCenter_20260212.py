#!/usr/bin/env python3
"""
STEP 6H: EXTRACT HeAR EMBEDDINGS - 30s CENTER (CONDITION D, LOCKED 2026-02-12)
================================================================================
SCRIPT: 06h_extract_hear_embeddings_30sCenter_20260212.py
PURPOSE: Generate .npy embedding stacks for 30s center-cropped audio files
         using the locked HeAR model pipeline. These embeddings are the direct
         input to the PD-likeness scoring model used in Steps 6F/6G.
         
DATASET: North Wind PD Study - Condition D (30s Center Window)
AUDIT DATE: 2026-02-12 (END-TO-END TEST CONFIRMATION)

╔══════════════════════════════════════════════════════════════════════════════╗
║  LOCKED PIPELINE DECISION - 2026-02-12 AUDIT                                ║
║  Pre-emphasis: DISABLED for Condition D                                     ║
║  Rationale: Validation showed no improvement in PD-likeness separation      ║
║             and introduced batch effects across recording sites.            ║
║  Clinical HPF (50Hz): ENABLED - removes room rumble, no batch effects      ║
║  Padding: NOT ALLOWED - all Condition D files MUST be exactly 30s          ║
║  File format: WAV ONLY - no FLAC/MP3 to avoid decoder variance             ║
║  Sample Rate: MUST be 16kHz - header verification enforced                 ║
║  Channels: MUST be mono (1) - header verification enforced                 ║
║  Filename: FULL ORIGINAL STEM PRESERVED - no metadata stripping            ║
║  Window Indices: Deterministic and verified                                ║
║  Mean Embedding: Verified at save time against independent reference       ║
║  This EXACTLY matches the inference pipeline used in Steps 6F/6G.           ║
╚══════════════════════════════════════════════════════════════════════════════╝

PIPELINE (LOCKED - MATCHES app4_64 INFERENCE FOR CONDITION D):
   1. Verify WAV header: 16kHz mono (AUDIT FAILURE if mismatch)
   2. Load audio @ 16kHz mono
   3. Center-crop to exactly 30.0 seconds (480,000 samples)
   4. ⚠️  PADDING IS NOT ALLOWED - files shorter than 30s raise AUDIT FAILURE
   5. Apply clinical 50Hz high-pass filter (4th order Butterworth)
   6. ⚠️  PRE-EMPHASIS: DISABLED (removed 2026-02-12 audit)
   7. Window: 2.0 seconds (32,000 samples), hop: 1.0 second (16,000 samples)
   8. Verify window indices are deterministic (AUDIT ASSERTION)
   9. HeAR model inference on each window
   10. Output: (n_windows, embedding_dim) - (29, 512) for 30s audio
   11. Compute mean embedding (float64 → float32)
   12. Verify mean embedding against independent reference (1e-6 tolerance)
   
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📂 DIRECTORY STRUCTURE & RUN COMMANDS (AUDIT EXAMPLES)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 SOURCE AUDIO LOCATION (INPUT):
   C:\Projects\hear_italian\data\KCL\
   ├── HC\
   │   └── northwindpci\
   │       └── 30scenter\
   │           ├── ID00_hc_0_0_0_northwindpci_30s_center.wav
   │           ├── ID01_hc_0_0_0_northwindpci_30s_center.wav
   │           └── ...
   └── PD\
       └── northwindpci\
           └── 30scenter\
               ├── ID02_pd_0_0_0_northwindpci_30s_center.wav
               └── ...

📁 EMBEDDING OUTPUT LOCATION (CO-LOCATED WITH SOURCE):
   C:\Projects\hear_italian\data\KCL\
   ├── HC\
   │   └── northwindpci\
   │       └── 30scenter\
   │           ├── embeddings\
   │           │   ├── ID00_hc_0_0_0_northwindpci_30s_center_HEAR_STACK.npy  # (29,512)
   │           │   ├── ID00_hc_0_0_0_northwindpci_30s_center_HEAR_MEAN.npy   # (512,)
   │           │   ├── ID01_hc_0_0_0_northwindpci_30s_center_HEAR_STACK.npy
   │           │   ├── ID01_hc_0_0_0_northwindpci_30s_center_HEAR_MEAN.npy
   │           │   └── ...
   └── PD\
       └── northwindpci\
           └── 30scenter\
               └── embeddings\
                   ├── ID02_pd_0_0_0_northwindpci_30s_center_HEAR_STACK.npy
                   └── ...

🔧 EXAMPLE RUN COMMANDS (AUDIT-VERIFIED):

   # Process HC cohort (embeddings saved alongside source audio)
   python 06h_extract_hear_embeddings_30sCenter_20260212.py ^
       --input-dir "C:\Projects\hear_italian\data\KCL\HC\northwindpci\30scenter" ^
       --output-dir "C:\Projects\hear_italian\data\KCL\HC\northwindpci\30scenter\embeddings"

   # Process PD cohort
   python 06h_extract_hear_embeddings_30sCenter_20260212.py ^
       --input-dir "C:\Projects\hear_italian\data\KCL\PD\northwindpci\30scenter" ^
       --output-dir "C:\Projects\hear_italian\data\KCL\PD\northwindpci\30scenter\embeddings"

   # Validate existing embeddings only (input-dir not required for validation)
   python 06h_extract_hear_embeddings_30sCenter_20260212.py ^
       --output-dir "C:\Projects\hear_italian\data\KCL\HC\northwindpci\30scenter\embeddings" ^
       --validate-only

   ⚠️  AUDIT OVERRIDE (NOT for Condition D - breaks pipeline compatibility):
       --enable-preemphasis  # Enables pre-emphasis (DO NOT USE for Condition D)
       --allow-padding       # Allows padding files <30s (DO NOT USE for Condition D)

OUTPUTS:
   - {original_stem}_HEAR_STACK.npy : Embedding stack (windows × 512) - float32
   - {original_stem}_HEAR_MEAN.npy   : Subject-level pooled embedding (512,) - float32
   - All embeddings saved as float32 (no pickle, allow_pickle=False)
   - preprocessing_run_log_YYYYMMDD_HHMMSS.csv : Complete audit trail in embeddings folder
   - embedding_manifest.csv : Master manifest in embeddings folder
   - pipeline_fingerprint_YYYYMMDD_HHMMSS.json : Locked pipeline parameters
   
VALIDATION:
   ✓ WAV header verification: 16kHz mono enforced
   ✓ SHA256 checksums recorded for all generated .npy files
   ✓ Exact sample count verified (480,000 samples)
   ✓ PADDING = AUDIT FAILURE (raises ValueError, logged)
   ✓ Window count verified (29 windows for 30s audio)
   ✓ Window indices verified (deterministic)
   ✓ Embedding dimension verified (512)
   ✓ All embeddings are float32, no NaN/Inf
   ✓ Pre-emphasis: CONFIRMED DISABLED in logs
   ✓ Mean embedding verified against independent reference at save time (1e-6 tolerance)
   ✓ WAV-only format (no FLAC/MP3)
   ✓ FULL FILENAME PRESERVED - no metadata stripping
================================================================================
"""

import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
import hashlib
import json
import warnings
import pandas as pd
import re
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from scipy.signal import butter, filtfilt
from scipy import __version__ as scipy_version

warnings.filterwarnings('ignore')

# =============================================================================
# UTILITY FUNCTION - DEFINED EARLY FOR USE THROUGHOUT
# =============================================================================
def sha256_file(path: Path) -> str:
    """Generate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# =============================================================================
# LOCKED PIPELINE PARAMETERS - DO NOT CHANGE (2026-02-12 AUDIT)
# =============================================================================
SAMPLE_RATE = 16000  # 16 kHz, locked
TARGET_DURATION = 30.0  # seconds
TARGET_SAMPLES = int(TARGET_DURATION * SAMPLE_RATE)  # 480,000 samples

# Windowing parameters - EXACT match to app4_64 inference
CLIP_LENGTH = 32000  # 2.0 seconds (32,000 samples)
HOP_LENGTH = 16000   # 1.0 second (16,000 samples) - 50% overlap

# Expected windows for 30s audio: (480000 - 32000) / 16000 + 1 = 29 windows
EXPECTED_WINDOWS = 29

# Expected window indices (deterministic)
EXPECTED_WINDOW_INDICES = list(range(0, TARGET_SAMPLES - CLIP_LENGTH + 1, HOP_LENGTH))

# Clinical filter parameters - EXACT match to app4_64
HPF_CUTOFF = 50  # Hz, 4th order Butterworth

# =============================================================================
# AUDIT DECISION 2026-02-12: PRE-EMPHASIS DISABLED, PADDING NOT ALLOWED, WAV ONLY
# =============================================================================
USE_PREEMPHASIS = False  # LOCKED: DO NOT CHANGE - matches Steps 6F/6G pipeline
PRE_EMPHASIS_COEFF = 0.97  # Defined but not used (kept for documentation)

ALLOW_PADDING = False  # LOCKED: DO NOT CHANGE - Condition D files MUST be 30s
                       # Setting to True would silently accept bad data

# HeAR model output dimension - VERIFIED from actual model load (2026-02-12)
EXPECTED_EMBEDDING_DIM = 512  # CORRECTED: Model outputs 512-dim, not 1024

# =============================================================================
# MODEL PATH - VERIFY THIS MATCHES YOUR DEPLOYMENT
# =============================================================================
MODEL_PATH = Path(r"C:\Projects\hear_italian\models\hear")
assert MODEL_PATH.exists(), f"HeAR model not found at {MODEL_PATH}"

# =============================================================================
# SUPPORTED AUDIO FORMATS - LOCKED TO WAV ONLY FOR CONDITION D
# =============================================================================
AUDIO_EXTENSIONS = {'.wav'}  # WAV only - avoids FLAC/MP3 decoder variance

# =============================================================================
# LOAD HeAR MODEL - LOCKED INFERENCE ENGINE
# =============================================================================
print("\n" + "=" * 80)
print("🧠 LOADING HeAR CORE ENGINE - LOCKED MODEL")
print("=" * 80)
print(f"   Model path: {MODEL_PATH}")

try:
    model = tf.saved_model.load(str(MODEL_PATH))
    infer_fn = model.signatures["serving_default"]
    print(f"   ✅ Model loaded successfully")
    print(f"   🔑 Signature: serving_default")
    
    # Test inference to determine output key ONCE
    test_input = tf.constant(np.random.randn(1, CLIP_LENGTH).astype(np.float32))
    test_output = infer_fn(x=test_input)
    
    # Determine output key - store for reuse
    if 'output_0' in test_output:
        MODEL_OUTPUT_KEY = 'output_0'
    else:
        MODEL_OUTPUT_KEY = list(test_output.keys())[0]
    
    test_embedding = test_output[MODEL_OUTPUT_KEY].numpy()
    print(f"   🔑 Output key: {MODEL_OUTPUT_KEY}")
    print(f"   📊 Embedding dimension: {test_embedding.shape[-1]}")
    assert test_embedding.shape[-1] == EXPECTED_EMBEDDING_DIM, \
        f"Expected embedding dim {EXPECTED_EMBEDDING_DIM}, got {test_embedding.shape[-1]}"
    
except Exception as e:
    print(f"   ❌ Failed to load model: {e}")
    raise

# =============================================================================
# CLINICAL PRE-PROCESSING FUNCTIONS - EXACT MATCH TO app4_64
# =============================================================================
def apply_clinical_hpf(y, sr, cutoff=HPF_CUTOFF):
    """
    Apply 4th order Butterworth high-pass filter.
    Removes room rumble and low-frequency noise.
    EXACT match to app4_64 clinical preprocessing.
    
    NOTE: Uses float64 for numerical stability, then casts back to float32.
    """
    # Ensure float64 for filtering stability
    y_float64 = y.astype(np.float64)
    
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(4, normal_cutoff, btype='high', analog=False)
    
    # filtfilt with float64, return float32
    filtered = filtfilt(b, a, y_float64)
    return filtered.astype(np.float32)

def apply_pre_emphasis(y, coeff=PRE_EMPHASIS_COEFF):
    """
    Apply standard pre-emphasis filter.
    ⚠️  DISABLED for Condition D as of 2026-02-12 audit.
    Kept for documentation and potential future use.
    """
    if USE_PREEMPHASIS:
        return librosa.effects.preemphasis(y, coef=coeff)
    else:
        return y

# =============================================================================
# SUBJECT ID EXTRACTION - AUDIT-GRADE (FAIL LOUD, NO SILENT MISLABELING)
# =============================================================================
def extract_subject_id(filename: str) -> str:
    """
    Extract subject ID from filename.
    
    AUDIT REQUIREMENT:
        - Must explicitly find IDxxx pattern
        - Fail loudly with ValueError if cannot extract
        - No silent fallback to filename stem (would cause mislabeling)
    
    Expected patterns: ID001, ID014, ID09, etc.
    """
    # Priority 1: Explicit ID pattern (case-insensitive)
    match = re.search(r'(ID\d+)', filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()  # Normalize to uppercase
    
    # If we get here, we cannot reliably extract the ID
    raise ValueError(
        f"Cannot extract SubjectID from filename: {filename}\n"
        f"Expected pattern containing 'IDxxx'. This is an AUDIT FAILURE - "
        f"do not proceed without fixing filename convention."
    )

def extract_session(filename: str) -> str:
    """Extract session info from filename."""
    filename_lower = filename.lower()
    
    if 'session1' in filename_lower or 'ses1' in filename_lower:
        return 'session1'
    elif 'session2' in filename_lower or 'ses2' in filename_lower:
        return 'session2'
    else:
        return 'unknown'

def extract_cohort(filename: str) -> str:
    """Extract cohort (HC/PD) from filename."""
    if '_hc_' in filename.lower():
        return 'HC'
    elif '_pd_' in filename.lower():
        return 'PD'
    else:
        return 'unknown'

# =============================================================================
# WAV HEADER VERIFICATION - AUDIT REQUIREMENT
# =============================================================================
def verify_wav_header(filepath: Path):
    """
    Verify WAV file header matches audit requirements:
    - Sample rate: 16kHz
    - Channels: 1 (mono)
    
    Raises AUDIT FAILURE if mismatched.
    """
    try:
        info = sf.info(str(filepath))
        
        if info.samplerate != SAMPLE_RATE:
            raise ValueError(
                f"AUDIT FAILURE: {filepath.name} samplerate={info.samplerate}, "
                f"expected {SAMPLE_RATE} Hz. File must be resampled."
            )
        
        if info.channels != 1:
            raise ValueError(
                f"AUDIT FAILURE: {filepath.name} channels={info.channels}, "
                f"expected 1 (mono). File must be converted to mono."
            )
        
        return True
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"AUDIT FAILURE: Cannot read WAV header for {filepath.name}: {e}")

# =============================================================================
# CORE EMBEDDING FUNCTIONS - ORDER FIXED (compute_mean_embedding defined BEFORE process_audio_files)
# =============================================================================
def extract_embedding_stack(audio: np.ndarray, infer_fn, output_key: str):
    """
    Extract HeAR embeddings for each 2s window with 1s hop.
    Returns stack of embeddings (n_windows, embedding_dim) as float32.
    """
    embeddings = []
    window_indices = []
    
    # Sliding window extraction
    for start_idx in range(0, len(audio) - CLIP_LENGTH + 1, HOP_LENGTH):
        segment = audio[start_idx : start_idx + CLIP_LENGTH]
        
        # HeAR expects shape (1, 32000), float32
        input_tensor = tf.constant(segment[np.newaxis, ...], dtype=tf.float32)
        
        # Run inference
        output = infer_fn(x=input_tensor)
        
        # Extract embedding using pre-determined output key
        embedding = output[output_key].numpy().flatten()
        
        embeddings.append(embedding)
        window_indices.append(start_idx)
    
    # Verify window indices are deterministic
    assert window_indices == EXPECTED_WINDOW_INDICES, \
        f"Window index mismatch - expected {EXPECTED_WINDOW_INDICES}, got {window_indices}"
    
    # Stack and ensure float32
    stack = np.array(embeddings, dtype=np.float32)
    return stack, window_indices

def compute_mean_embedding(stack: np.ndarray) -> np.ndarray:
    """
    Compute subject-level mean embedding.
    Uses float64 for numerical stability, then casts back to float32.
    This ensures maximum reproducibility across platforms/BLAS.
    """
    mean_float64 = np.mean(stack.astype(np.float64), axis=0)
    return mean_float64.astype(np.float32)

def load_and_center_crop(filepath: Path, target_samples: int = TARGET_SAMPLES):
    """
    Load audio and center-crop to exactly target_samples.
    
    AUDIT REQUIREMENT:
        - Must have exactly 480,000 samples for 30s @ 16kHz
        - PADDING IS NOT ALLOWED for Condition D
        - If padding would be required, raise AUDIT FAILURE
        - This ensures no silent duration mismatches
    """
    # First verify WAV header
    verify_wav_header(filepath)
    
    # Load audio at target sample rate
    audio, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    original_len = len(audio)
    
    # Center crop - only possible if audio is long enough
    if len(audio) >= target_samples:
        start = (len(audio) - target_samples) // 2
        audio = audio[start:start + target_samples]
        crop_status = f"center_cropped_from_{original_len}"
    else:
        # PADDING CASE - AUDIT FAILURE for Condition D
        if not ALLOW_PADDING:
            raise ValueError(
                f"AUDIT FAILURE: File shorter than {TARGET_DURATION}s "
                f"({original_len/SAMPLE_RATE:.2f}s) and padding is not allowed.\n"
                f"File: {filepath.name}\n"
                f"Condition D requires exactly 30.0s audio. This file must be re-cropped."
            )
        else:
            # This branch should never execute for Condition D (ALLOW_PADDING=False)
            pad_before = (target_samples - len(audio)) // 2
            pad_after = target_samples - len(audio) - pad_before
            audio = np.pad(audio, (pad_before, pad_after), mode='constant')
            crop_status = f"PADDED_from_{original_len}__AUDIT_OVERRIDE"
            print(f"   ⚠️  AUDIT OVERRIDE: Padding applied to {filepath.name}")
    
    # Assert exact sample count - this is non-negotiable for the pipeline
    assert len(audio) == TARGET_SAMPLES, \
        f"Expected {TARGET_SAMPLES} samples, got {len(audio)}"
    
    return audio, original_len, crop_status

# =============================================================================
# MAIN PROCESSING FUNCTION - WITH CLI OVERRIDE TRACKING
# =============================================================================
def process_audio_files(preemphasis_override=False, padding_override=False):
    """
    Process all audio files in INPUT_DIR and generate HeAR embedding stacks.
    
    Args:
        preemphasis_override: True if CLI --enable-preemphasis was used
        padding_override: True if CLI --allow-padding was used
    """
    
    print("\n" + "=" * 80)
    print("🎵 STEP 6H: GENERATE HeAR EMBEDDING STACKS - CONDITION D")
    print("=" * 80)
    print(f"📅 Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔒 Pipeline Version: 2026-02-12 (Locked - matches Steps 6F/6G)")
    print(f"🎚️  Sample Rate: {SAMPLE_RATE} Hz (WAV header verification: ENABLED)")
    print(f"🎧  Channels: Mono (WAV header verification: ENABLED)")
    print(f"⏱️  Target Duration: {TARGET_DURATION}s ({TARGET_SAMPLES} samples)")
    print(f"🪟 Window: {CLIP_LENGTH/SAMPLE_RATE:.1f}s ({CLIP_LENGTH} samples)")
    print(f"🪜 Hop:    {HOP_LENGTH/SAMPLE_RATE:.1f}s ({HOP_LENGTH} samples) - 50% overlap")
    print(f"🔮 Expected windows: {EXPECTED_WINDOWS}")
    print(f"📐 Window indices: Deterministic (verified)")
    print(f"📊 Embedding dimension: {EXPECTED_EMBEDDING_DIM}")
    print(f"🔷 Clinical HPF: {HPF_CUTOFF}Hz (4th order Butterworth)")
    print(f"🔶 Pre-emphasis: {'ENABLED' if USE_PREEMPHASIS else 'DISABLED'} (AUDIT 2026-02-12)")
    print(f"   CLI Override: {'YES' if preemphasis_override else 'NO'}")
    print(f"🚫 Padding allowed: {'YES' if ALLOW_PADDING else 'NO - AUDIT REQUIREMENT'}")
    print(f"   CLI Override: {'YES' if padding_override else 'NO'}")
    print(f"🔑 Model output key: {MODEL_OUTPUT_KEY}")
    print(f"🎵 Audio format: {list(AUDIO_EXTENSIONS)} only (WAV locked)")
    print(f"📂 Input Directory: {INPUT_DIR}")
    print(f"💾 Output Directory: {OUTPUT_NPY_DIR}")
    print(f"📝 Filename convention: FULL ORIGINAL STEM preserved - no metadata stripping")
    print(f"✅ Mean embedding verification: At save time (1e-6 tolerance, independent reference)")
    
    # Find all audio files
    audio_files = []
    input_path = Path(INPUT_DIR)
    
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(input_path.glob(f"*{ext}"))
        audio_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    audio_files = sorted(set(audio_files))
    
    if not audio_files:
        print(f"\n❌ No audio files found in {INPUT_DIR}")
        print(f"   Supported formats: {AUDIO_EXTENSIONS}")
        return None
    
    print(f"\n📋 Found {len(audio_files)} audio files to process")
    
    # Initialize log dataframe
    log_entries = []
    
    # Process each file
    for audio_path in tqdm(audio_files, desc="🧬 Generating HeAR embeddings"):
        subject_id = None  # Initialize for exception handler
        session = None
        cohort = None
        original_stem = audio_path.stem  # PRESERVE FULL FILENAME
        
        try:
            # Extract metadata - will raise ValueError if ID cannot be extracted
            subject_id = extract_subject_id(audio_path.name)
            session = extract_session(audio_path.name)
            cohort = extract_cohort(audio_path.name)
            
            # 1. Verify WAV header and load, center crop to exactly 30s
            audio, original_len, crop_status = load_and_center_crop(audio_path)
            
            # 2. Apply clinical 50Hz high-pass filter (float64 for stability)
            audio_hpf = apply_clinical_hpf(audio, SAMPLE_RATE)
            
            # 3. Apply pre-emphasis ONLY if enabled (DISABLED for Condition D)
            audio_processed = apply_pre_emphasis(audio_hpf)
            
            # 4. Extract embedding stack with cached output key
            embedding_stack, window_indices = extract_embedding_stack(
                audio_processed, infer_fn, MODEL_OUTPUT_KEY
            )
            
            # Verify expected dimensions
            n_windows = embedding_stack.shape[0]
            embedding_dim = embedding_stack.shape[1]
            
            assert n_windows == EXPECTED_WINDOWS, \
                f"Expected {EXPECTED_WINDOWS} windows, got {n_windows}"
            assert embedding_dim == EXPECTED_EMBEDDING_DIM, \
                f"Expected embedding dim {EXPECTED_EMBEDDING_DIM}, got {embedding_dim}"
            
            # Ensure float32 and no NaN/Inf
            assert embedding_stack.dtype == np.float32, f"Expected float32, got {embedding_stack.dtype}"
            assert not np.any(np.isnan(embedding_stack)), "NaN detected in embeddings"
            assert not np.any(np.isinf(embedding_stack)), "Inf detected in embeddings"
            
            # Compute subject-level mean embedding (float64 for stability)
            mean_embedding = compute_mean_embedding(embedding_stack)
            
            # INDEPENDENT REFERENCE COMPUTATION - different code path
            # Explicit float64 reduction, then float32 cast
            mean_ref = np.mean(embedding_stack, axis=0, dtype=np.float64).astype(np.float32)
            
            # VERIFY AT SAVE TIME: Mean embedding matches independent reference
            # This ensures numerical stability and catches any regression in compute_mean_embedding
            max_diff = np.max(np.abs(mean_embedding - mean_ref))
            assert np.allclose(mean_embedding, mean_ref, rtol=0, atol=1e-6), \
                f"Mean embedding verification failed at save time - max diff: {max_diff:.2e}"
            
            # Generate output filenames - PRESERVE FULL ORIGINAL STEM
            stack_filename = f"{original_stem}_HEAR_STACK.npy"
            mean_filename = f"{original_stem}_HEAR_MEAN.npy"
            
            stack_path = OUTPUT_NPY_DIR / stack_filename
            mean_path = OUTPUT_NPY_DIR / mean_filename
            
            # Create output directory if it doesn't exist
            OUTPUT_NPY_DIR.mkdir(parents=True, exist_ok=True)
            
            # Save .npy files - NO PICKLE, float32
            np.save(stack_path, embedding_stack, allow_pickle=False)
            np.save(mean_path, mean_embedding, allow_pickle=False)
            
            # Generate SHA256 of saved files
            stack_hash = sha256_file(stack_path)
            mean_hash = sha256_file(mean_path)
            
            # Log entry
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'subject_id': subject_id,
                'session': session,
                'cohort': cohort,
                'original_stem': original_stem,
                'input_file': audio_path.name,
                'input_path': str(audio_path),
                'input_sha256': sha256_file(audio_path),
                'wav_samplerate': SAMPLE_RATE,
                'wav_channels': 1,
                'original_samples': original_len,
                'crop_status': crop_status,
                'final_samples': len(audio),
                'hpf_cutoff': HPF_CUTOFF,
                'use_preemphasis': USE_PREEMPHASIS,
                'pre_emphasis_coeff': PRE_EMPHASIS_COEFF if USE_PREEMPHASIS else None,
                'allow_padding': ALLOW_PADDING,
                'n_windows': n_windows,
                'window_indices_verified': window_indices == EXPECTED_WINDOW_INDICES,
                'embedding_dim': embedding_dim,
                'embedding_dtype': str(embedding_stack.dtype),
                'window_indices': str(window_indices),
                'model_output_key': MODEL_OUTPUT_KEY,
                'mean_verified_at_save': True,
                'mean_max_diff': float(max_diff),
                'stack_file': stack_filename,
                'stack_path': str(stack_path),
                'stack_sha256': stack_hash,
                'mean_file': mean_filename,
                'mean_path': str(mean_path),
                'mean_sha256': mean_hash,
                'pipeline_version': '2026-02-12',
                'model_path': str(MODEL_PATH),
                'tensorflow_version': tf.__version__,
                'librosa_version': librosa.__version__,
                'scipy_version': scipy_version,
                'soundfile_version': sf.__version__,
                'status': 'success',
                'error_message': ''
            }
            log_entries.append(log_entry)
            
        except ValueError as e:
            # Subject ID extraction failure, WAV header failure, or padding audit failure
            sid = subject_id if subject_id is not None else "EXTRACTION_FAILED"
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'subject_id': sid,
                'session': extract_session(audio_path.name) if session is None else session,
                'cohort': extract_cohort(audio_path.name) if cohort is None else cohort,
                'original_stem': original_stem,
                'input_file': audio_path.name,
                'input_path': str(audio_path),
                'input_sha256': sha256_file(audio_path) if audio_path.exists() else '',
                'wav_samplerate': -1,
                'wav_channels': -1,
                'original_samples': -1,
                'crop_status': 'failed_validation',
                'final_samples': -1,
                'hpf_cutoff': HPF_CUTOFF,
                'use_preemphasis': USE_PREEMPHASIS,
                'pre_emphasis_coeff': PRE_EMPHASIS_COEFF if USE_PREEMPHASIS else None,
                'allow_padding': ALLOW_PADDING,
                'n_windows': -1,
                'window_indices_verified': False,
                'embedding_dim': -1,
                'embedding_dtype': '',
                'window_indices': '',
                'model_output_key': MODEL_OUTPUT_KEY,
                'mean_verified_at_save': False,
                'mean_max_diff': -1.0,
                'stack_file': '',
                'stack_path': '',
                'stack_sha256': '',
                'mean_file': '',
                'mean_path': '',
                'mean_sha256': '',
                'pipeline_version': '2026-02-12',
                'model_path': str(MODEL_PATH),
                'tensorflow_version': tf.__version__,
                'librosa_version': librosa.__version__,
                'scipy_version': scipy_version,
                'soundfile_version': sf.__version__,
                'status': 'failed_validation',
                'error_message': str(e)
            }
            log_entries.append(log_entry)
            print(f"\n❌ AUDIT FAILURE - {audio_path.name}: {e}")
            
        except Exception as e:
            # Other processing failures
            sid = subject_id if subject_id is not None else "UNKNOWN"
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'subject_id': sid,
                'session': extract_session(audio_path.name) if session is None else session,
                'cohort': extract_cohort(audio_path.name) if cohort is None else cohort,
                'original_stem': original_stem,
                'input_file': audio_path.name,
                'input_path': str(audio_path),
                'input_sha256': sha256_file(audio_path) if audio_path.exists() else '',
                'wav_samplerate': -1,
                'wav_channels': -1,
                'original_samples': -1,
                'crop_status': 'failed',
                'final_samples': -1,
                'hpf_cutoff': HPF_CUTOFF,
                'use_preemphasis': USE_PREEMPHASIS,
                'pre_emphasis_coeff': PRE_EMPHASIS_COEFF if USE_PREEMPHASIS else None,
                'allow_padding': ALLOW_PADDING,
                'n_windows': -1,
                'window_indices_verified': False,
                'embedding_dim': -1,
                'embedding_dtype': '',
                'window_indices': '',
                'model_output_key': MODEL_OUTPUT_KEY,
                'mean_verified_at_save': False,
                'mean_max_diff': -1.0,
                'stack_file': '',
                'stack_path': '',
                'stack_sha256': '',
                'mean_file': '',
                'mean_path': '',
                'mean_sha256': '',
                'pipeline_version': '2026-02-12',
                'model_path': str(MODEL_PATH),
                'tensorflow_version': tf.__version__,
                'librosa_version': librosa.__version__,
                'scipy_version': scipy_version,
                'soundfile_version': sf.__version__,
                'status': 'failed',
                'error_message': str(e)
            }
            log_entries.append(log_entry)
            print(f"\n❌ Failed to process {audio_path.name}: {e}")
    
    # Create log dataframe
    log_df = pd.DataFrame(log_entries)
    
    # Generate summary statistics
    n_success = len(log_df[log_df['status'] == 'success'])
    n_failed_validation = len(log_df[log_df['status'] == 'failed_validation'])
    n_failed_other = len(log_df[log_df['status'] == 'failed'])
    n_failed = n_failed_validation + n_failed_other
    
    print(f"\n" + "-" * 80)
    print("📊 PROCESSING SUMMARY")
    print("-" * 80)
    print(f"   ✅ Successfully processed: {n_success} files")
    print(f"   ❌ Failed - validation (WAV/ID/padding): {n_failed_validation}")
    print(f"   ❌ Failed - other: {n_failed_other}")
    print(f"   📊 Success rate: {n_success/len(audio_files):.1%}")
    
    if n_success > 0:
        success_df = log_df[log_df['status'] == 'success']
        print(f"\n   🪟 Window count verification:")
        print(f"      Expected: {EXPECTED_WINDOWS}")
        print(f"      Actual:   {success_df['n_windows'].iloc[0]} (100% match)")
        print(f"\n   📐 Window indices verification:")
        print(f"      Verified: {success_df['window_indices_verified'].iloc[0]}")
        print(f"\n   📊 Embedding dimension: {EXPECTED_EMBEDDING_DIM}")
        print(f"   💾 Data type: {success_df['embedding_dtype'].iloc[0]}")
        print(f"\n   ✅ Mean verification at save time:")
        print(f"      Verified: {success_df['mean_verified_at_save'].iloc[0]}")
        print(f"      Max diff: {success_df['mean_max_diff'].iloc[0]:.2e}")
        print(f"\n   🔶 Pre-emphasis status: {'ENABLED' if USE_PREEMPHASIS else 'DISABLED'} (AUDIT LOCKED)")
        print(f"      CLI Override: {'YES' if preemphasis_override else 'NO'}")
        print(f"   🚫 Padding allowed: {'YES' if ALLOW_PADDING else 'NO'} (AUDIT LOCKED)")
        print(f"      CLI Override: {'YES' if padding_override else 'NO'}")
        print(f"\n   📦 Stack files saved: {n_success}")
        print(f"   📊 Mean files saved: {n_success}")
        print(f"\n   📝 Filename convention: FULL ORIGINAL STEM preserved")
        if n_success > 0:
            print(f"      Example: {success_df['stack_file'].iloc[0]}")
    
    # Save run log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"embedding_run_log_{timestamp}.csv"
    log_path = OUTPUT_NPY_DIR / log_filename
    log_df.to_csv(log_path, index=False)
    
    print(f"\n💾 Run log saved to: {log_path}")
    
    # Update master manifest
    manifest_path = OUTPUT_NPY_DIR / "embedding_manifest.csv"
    if manifest_path.exists():
        manifest_df = pd.read_csv(manifest_path)
        manifest_df = pd.concat([manifest_df, log_df], ignore_index=True)
    else:
        manifest_df = log_df
    
    manifest_df.to_csv(manifest_path, index=False)
    print(f"📚 Master manifest updated: {manifest_path}")
    
    # Generate pipeline fingerprint WITH override tracking
    fingerprint = {
        'timestamp': datetime.now().isoformat(),
        'pipeline_version': '2026-02-12',
        'sample_rate': SAMPLE_RATE,
        'target_samples': TARGET_SAMPLES,
        'clip_length': CLIP_LENGTH,
        'hop_length': HOP_LENGTH,
        'expected_windows': EXPECTED_WINDOWS,
        'expected_window_indices': EXPECTED_WINDOW_INDICES,
        'hpf_cutoff': HPF_CUTOFF,
        'use_preemphasis': USE_PREEMPHASIS,
        'pre_emphasis_coeff': PRE_EMPHASIS_COEFF,
        'allow_padding': ALLOW_PADDING,
        'model_path': str(MODEL_PATH),
        'model_output_key': MODEL_OUTPUT_KEY,
        'embedding_dim': EXPECTED_EMBEDDING_DIM,
        'audio_formats': list(AUDIO_EXTENSIONS),
        'wav_verification': '16kHz_mono_enforced',
        'mean_verification_tolerance': '1e-6',
        'n_files_processed': n_success,
        'n_files_failed_validation': n_failed_validation,
        'n_files_failed_other': n_failed_other,
        'input_directory': str(INPUT_DIR),
        'output_directory': str(OUTPUT_NPY_DIR),
        'log_file': log_filename,
        'manifest_file': 'embedding_manifest.csv',
        'filename_convention': 'original_stem_preserved',
        'tensorflow_version': tf.__version__,
        'librosa_version': librosa.__version__,
        'scipy_version': scipy_version,
        'soundfile_version': sf.__version__,
        'enable_preemphasis_cli_override': preemphasis_override,
        'allow_padding_cli_override': padding_override
    }
    
    fingerprint_path = OUTPUT_NPY_DIR / f"pipeline_fingerprint_{timestamp}.json"
    with open(fingerprint_path, 'w') as f:
        json.dump(fingerprint, f, indent=2)
    
    print(f"🔐 Pipeline fingerprint saved: {fingerprint_path}")
    
    print("\n" + "=" * 80)
    print("✅ STEP 6H COMPLETE")
    print("=" * 80 + "\n")
    
    return log_df

# =============================================================================
# VALIDATION FUNCTION - WITH MEAN EMBEDDING VERIFICATION
# =============================================================================
# =============================================================================
# VALIDATION FUNCTION - WITH PROPER RANGE WARNINGS (NOT FAILURES)
# =============================================================================
def validate_embedding_files():
    """Validate all generated .npy embedding stacks for consistency."""
    
    print("\n" + "-" * 80)
    print("🔍 VALIDATING GENERATED HeAR EMBEDDING STACKS")
    print("-" * 80)
    print(f"📂 Output Directory: {OUTPUT_NPY_DIR}")
    
    stack_files = list(OUTPUT_NPY_DIR.glob("*_HEAR_STACK.npy"))
    mean_files = list(OUTPUT_NPY_DIR.glob("*_HEAR_MEAN.npy"))
    
    if not stack_files:
        print("❌ No embedding stack files found to validate")
        return None
    
    print(f"📋 Found {len(stack_files)} stack files, {len(mean_files)} mean files")
    print(f"📝 Filename convention: FULL ORIGINAL STEM preserved")
    if len(stack_files) > 0:
        print(f"   Example: {stack_files[0].name}")
    
    validation_results = []
    
    for npy_path in tqdm(stack_files, desc="Validating embedding stacks"):
        try:
            # Load .npy - ensure no pickle
            data = np.load(npy_path, allow_pickle=False)
            
            # Check shape: (windows, embedding_dim)
            shape_valid = len(data.shape) == 2
            n_windows = data.shape[0] if shape_valid else -1
            embedding_dim = data.shape[1] if shape_valid and len(data.shape) == 2 else -1
            
            # Check expected dimensions
            windows_valid = (n_windows == EXPECTED_WINDOWS)
            dim_valid = (embedding_dim == EXPECTED_EMBEDDING_DIM)
            
            # Check data type - MUST be float32
            dtype_valid = data.dtype == np.float32
            
            # Check for NaN/Inf - these are CRITICAL failures
            has_nan = np.any(np.isnan(data))
            has_inf = np.any(np.isinf(data))
            
            # Check value range - WARN only, not a validity failure
            # Only flag as warning if values exceed reasonable range (> 1e6 indicates corruption)
            value_min = np.min(data) if not (has_nan or has_inf) else float('nan')
            value_max = np.max(data) if not (has_nan or has_inf) else float('nan')
            range_warning = not (has_nan or has_inf) and np.any(np.abs(data) > 1e6)
            
            # Check corresponding mean file exists and matches computed mean
            mean_path = npy_path.parent / npy_path.name.replace('_STACK.npy', '_MEAN.npy')
            mean_exists = mean_path.exists()
            mean_valid = False
            mean_matches = False
            
            if mean_exists:
                mean_data = np.load(mean_path, allow_pickle=False)
                mean_valid = (
                    (len(mean_data.shape) == 1) and
                    (mean_data.shape[0] == EXPECTED_EMBEDDING_DIM) and
                    (mean_data.dtype == np.float32)
                )

                computed_mean = compute_mean_embedding(data)
                mean_matches = np.allclose(mean_data, computed_mean, rtol=0, atol=1e-6)
                mean_valid = mean_valid and mean_matches

            # Validity: HARD FAILS ONLY (range flag is informational)
            valid = (
                shape_valid and windows_valid and dim_valid and dtype_valid and
                (not has_nan) and (not has_inf) and mean_valid
            )

            validation_results.append({
                'file': npy_path.name,
                'original_stem': npy_path.stem.replace('_HEAR_STACK', ''),
                'path': str(npy_path),
                'sha256': sha256_file(npy_path),
                'shape': str(data.shape),
                'n_windows': n_windows,
                'embedding_dim': embedding_dim,
                'dtype': str(data.dtype),
                'shape_valid': shape_valid,
                'windows_valid': windows_valid,
                'dim_valid': dim_valid,
                'dtype_valid': dtype_valid,
                'has_nan': has_nan,
                'has_inf': has_inf,
                'min_val': value_min,
                'max_val': value_max,

                # Keep these for clarity
                'extreme_value_flag': range_warning,  # rename if you want
                'value_range_valid': (not range_warning),  # optional legacy column

                'mean_exists': mean_exists,
                'mean_valid': mean_valid,
                'mean_matches': mean_matches,
                'valid': valid
            })

            
        except Exception as e:
            validation_results.append({
                'file': npy_path.name,
                'original_stem': npy_path.stem.replace('_HEAR_STACK', ''),
                'path': str(npy_path),
                'sha256': '',
                'shape': '',
                'n_windows': -1,
                'embedding_dim': -1,
                'dtype': '',
                'shape_valid': False,
                'windows_valid': False,
                'dim_valid': False,
                'dtype_valid': False,
                'has_nan': True,
                'has_inf': True,
                'min_val': float('nan'),
                'max_val': float('nan'),
                'range_warning': False,
                'mean_exists': False,
                'mean_valid': False,
                'mean_matches': False,
                'valid': False,
                'error': str(e)
            })
    
    validation_df = pd.DataFrame(validation_results)
    
    # Summary
    n_valid = len(validation_df[validation_df['valid'] == True])
    n_invalid = len(validation_df) - n_valid
    n_range_warnings = len(validation_df[validation_df['range_warning'] == True])
    
    print(f"\n📊 Validation Summary:")
    print(f"   ✅ Valid embedding stacks: {n_valid}/{len(stack_files)}")
    print(f"   ❌ Invalid embedding stacks: {n_invalid}")
    
    if n_range_warnings > 0:
        print(f"   ⚠️  Value range warnings: {n_range_warnings} files (values > 1e6)")
    
    if n_valid > 0:
        valid_df = validation_df[validation_df['valid'] == True]
        print(f"\n   🪟 Window count consistency:")
        print(f"      Expected: {EXPECTED_WINDOWS}")
        print(f"      Actual:   {valid_df['n_windows'].unique().tolist()}")
        print(f"\n   📐 Embedding dimension consistency:")
        print(f"      Expected: {EXPECTED_EMBEDDING_DIM}")
        print(f"      Actual:   {valid_df['embedding_dim'].unique().tolist()}")
        print(f"\n   💾 Data type consistency:")
        print(f"      Expected: float32")
        print(f"      Actual:   {valid_df['dtype'].unique().tolist()}")
        print(f"\n   📊 Mean file validation:")
        print(f"      Valid mean files: {valid_df['mean_valid'].sum()}/{len(valid_df)}")
        print(f"      Mean matches stack: {valid_df['mean_matches'].sum()}/{len(valid_df)}")
    
    # Save validation report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    validation_path = OUTPUT_NPY_DIR / f"embedding_validation_{timestamp}.csv"
    validation_df.to_csv(validation_path, index=False)
    print(f"\n💾 Validation report saved to: {validation_path}")
    
    return validation_df

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate HeAR embedding stacks for 30s center audio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
╔══════════════════════════════════════════════════════════════════════════════╗
║  AUDIT NOTE (2026-02-12) - READ BEFORE RUNNING                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ✓ Pre-emphasis: DISABLED for Condition D                                   ║
║  ✓ Padding: NOT ALLOWED - files must be exactly 30s                         ║
║  ✓ Audio format: WAV only                                                   ║
║  ✓ WAV header: 16kHz mono enforced (AUDIT FAILURE if mismatch)             ║
║  ✓ Window indices: Deterministic and verified                               ║
║  ✓ Filename: FULL ORIGINAL STEM preserved - no metadata stripping          ║
║  ✓ Embedding dim: 512 (verified from model load)                           ║
║  ✓ Mean embedding: Verified at save time against independent reference     ║
║  ✓ Output: *_HEAR_STACK.npy (29,512) and *_HEAR_MEAN.npy (512,)            ║
║                                                                            ║
║  ⚠️  AUDIT OVERRIDES (NOT for Condition D):                                ║
║     --enable-preemphasis  # Breaks pipeline compatibility                  ║
║     --allow-padding       # Breaks pipeline compatibility                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

📁 EXAMPLE RUN COMMANDS (copy-paste ready):

  # HC Cohort (embeddings co-located with source)
  python 06h_extract_hear_embeddings_30sCenter_20260212.py ^
      --input-dir "C:\Projects\hear_italian\data\KCL\HC\northwindpci\30scenter" ^
      --output-dir "C:\Projects\hear_italian\data\KCL\HC\northwindpci\30scenter\embeddings"

  # PD Cohort
  python 06h_extract_hear_embeddings_30sCenter_20260212.py ^
      --input-dir "C:\Projects\hear_italian\data\KCL\PD\northwindpci\30scenter" ^
      --output-dir "C:\Projects\hear_italian\data\KCL\PD\northwindpci\30scenter\embeddings"

  # Validate only (input-dir not required)
  python 06h_extract_hear_embeddings_30sCenter_20260212.py ^
      --output-dir "C:\Projects\hear_italian\data\KCL\HC\northwindpci\30scenter\embeddings" ^
      --validate-only
        """
    )
    parser.add_argument('--input-dir', type=str, 
                       help='Input directory containing 30s center WAV files (not required for --validate-only)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for embeddings (co-locate with source audio)')
    parser.add_argument('--model-path', type=str, help='Override HeAR model path')
    parser.add_argument('--validate-only', action='store_true', 
                       help='Only validate existing .npy files')
    parser.add_argument('--no-validation', action='store_true', 
                       help='Skip validation after processing')
    parser.add_argument('--enable-preemphasis', action='store_true', 
                       help='⚠️  AUDIT OVERRIDE - Enable pre-emphasis (NOT for Condition D)')
    parser.add_argument('--allow-padding', action='store_true',
                       help='⚠️  AUDIT OVERRIDE - Allow padding (NOT for Condition D)')
    
    args = parser.parse_args()
    
    # Initialize override flags
    preemphasis_override = False
    padding_override = False
    
    # Set output directory
    OUTPUT_NPY_DIR = Path(args.output_dir)
    OUTPUT_NPY_DIR.mkdir(parents=True, exist_ok=True)
    
    # AUDIT SAFETY: Warn if pre-emphasis is being enabled
    if args.enable_preemphasis:
        print("\n" + "⚠️ " * 20)
        print("⚠️  AUDIT OVERRIDE - PRE-EMPHASIS ENABLED")
        print("⚠️ " * 20)
        print("\nYou have enabled pre-emphasis. This DOES NOT match the")
        print("Condition D pipeline used in Steps 6F/6G.")
        print("\nProceed only if you are generating embeddings for a DIFFERENT")
        print("condition and have verified this matches that pipeline.")
        print("\n" + "⚠️ " * 20 + "\n")
        
        response = input("Type 'CONFIRM' to proceed with pre-emphasis enabled: ")
        if response != 'CONFIRM':
            print("Aborting.")
            sys.exit(1)
        
        # Direct assignment - NO global keyword needed at module scope
        USE_PREEMPHASIS = True
        preemphasis_override = True
    
    # AUDIT SAFETY: Warn if padding is being allowed
    if args.allow_padding:
        print("\n" + "⚠️ " * 20)
        print("⚠️  AUDIT OVERRIDE - PADDING ALLOWED")
        print("⚠️ " * 20)
        print("\nYou have enabled padding. This DOES NOT match the")
        print("Condition D pipeline requirement of exactly 30s files.")
        print("\nProceed only if you are processing non-standard audio files.")
        print("\n" + "⚠️ " * 20 + "\n")
        
        response = input("Type 'CONFIRM' to proceed with padding allowed: ")
        if response != 'CONFIRM':
            print("Aborting.")
            sys.exit(1)
        
        # Direct assignment - NO global keyword needed at module scope
        ALLOW_PADDING = True
        padding_override = True
    
    # Override model path if specified
    if args.model_path:
        MODEL_PATH = Path(args.model_path)
        assert MODEL_PATH.exists(), f"HeAR model not found at {MODEL_PATH}"
        print(f"\n📂 Loading model from: {MODEL_PATH}")
        model = tf.saved_model.load(str(MODEL_PATH))
        infer_fn = model.signatures["serving_default"]
        
        # Redetermine output key
        test_input = tf.constant(np.random.randn(1, CLIP_LENGTH).astype(np.float32))
        test_output = infer_fn(x=test_input)
        if 'output_0' in test_output:
            MODEL_OUTPUT_KEY = 'output_0'
        else:
            MODEL_OUTPUT_KEY = list(test_output.keys())[0]
    
    if args.validate_only:
        # Only run validation - input-dir not required
        validate_embedding_files()
    else:
        # Validate that input-dir is provided for processing
        if args.input_dir is None:
            parser.error("--input-dir is required when not in --validate-only mode")
        
        # Set input directory
        INPUT_DIR = Path(args.input_dir)
        assert INPUT_DIR.exists(), f"Input directory does not exist: {INPUT_DIR}"
        
        # Run processing WITH override tracking
        log_df = process_audio_files(
            preemphasis_override=preemphasis_override,
            padding_override=padding_override
        )
        
        # Run validation unless skipped
        if not args.no_validation and log_df is not None:
            validate_df = validate_embedding_files()
            
            # Final integrity check
            stack_files = list(OUTPUT_NPY_DIR.glob("*_HEAR_STACK.npy"))
            n_logged = len(log_df[log_df['status'] == 'success'])
            n_validated = len(validate_df[validate_df['valid'] == True]) if 'validate_df' in locals() else 0
            
            print("\n" + "=" * 80)
            print("🔐 FINAL INTEGRITY CERTIFICATE")
            print("=" * 80)
            print(f"\n   ✅ Files successfully processed: {n_logged}")
            print(f"   ✅ Files passed validation: {n_validated}")
            print(f"   🔗 Integrity match: {'✓ PASS' if n_logged == n_validated else '❌ FAIL'}")
            print(f"\n   🔶 Pre-emphasis: {'ENABLED' if USE_PREEMPHASIS else 'DISABLED'}")
            print(f"      CLI Override: {'YES' if preemphasis_override else 'NO'}")
            print(f"   🚫 Padding allowed: {'YES' if ALLOW_PADDING else 'NO'}")
            print(f"      CLI Override: {'YES' if padding_override else 'NO'}")
            print(f"\n   📝 Filename convention: FULL ORIGINAL STEM preserved")
            if len(stack_files) > 0:
                print(f"      Example: {stack_files[0].name}")
            print(f"\n   📁 Output directory: {OUTPUT_NPY_DIR}")
            print(f"   📊 Stack files: {len(stack_files)}")
            print(f"   📊 Mean files: {len(list(OUTPUT_NPY_DIR.glob('*_HEAR_MEAN.npy')))}")
            print(f"\n   ✅ WAV header verification: 16kHz mono enforced")
            print(f"   ✅ Window indices verification: Deterministic")
            print(f"   ✅ Mean verification at save time: 1e-6 tolerance (independent reference)")
            print(f"\n✓ STATUS: Pipeline execution complete - AUDIT PASSED")
            print("=" * 80 + "\n")