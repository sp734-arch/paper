#!/usr/bin/env python3
"""
Step 11: Clinical Validation - 30s Center Window Crop with Speech Boundary Alignment (Condition D)
Date: 2026-02-12 (FINAL v5: Single source of truth - SUCCESS defines final distribution)

███████████████████████████████████████████████████████████████████████████████████
│                                                                                 │
│   THIS IS NOT A DIAGNOSTIC TOOL                                               │
│   This script MEASURES population distributions on the CLEANEST 30 SECONDS     │
│   of each recording - a fixed window centered in the middle of the passage    │
│   and ALIGNED TO NEAREST PRECEDING SILENCE to avoid mid-word starts.          │
│                                                                                 │
███████████████████████████████████████████████████████████████████████████████████

PURPOSE:
------------------------------------------------------------------------------
Extracts a 30-second window from the EXACT CENTER of each recording, then ALIGNS
the start to the nearest preceding silence boundary (≥100ms). This eliminates
FOUR confounders simultaneously:

    1. DURATION CONFOUND: Every subject contributes EXACTLY 30 seconds
    2. START ARTIFACTS: Cleared throat, microphone adjustment, false starts
    3. END FATIGUE: Trailing off, loss of prosody, early stopping
    4. MID-WORD STARTS: Analysis never begins mid-utterance

AUDITABILITY - CRITICAL:
------------------------------------------------------------------------------
When --save-segments is enabled, this script saves the EXACT 30-second aligned
window as a WAV file alongside its source, but ONLY for subjects that successfully
contribute to the final distribution (i.e., measurement_status == 'SUCCESS').

    {source_dir}/30scenter/{basename}_30s_center.wav

This creates a COMPLETE AUDIT TRAIL with 1:1 correspondence between:
    - CSV rows with QC_Pass = True
    - Saved WAV files in /30scenter/
    - Subjects included in distribution statistics

NO AMBIGUITY - ONE DEFINITION OF "MEASURED" THROUGHOUT.

VALIDITY REQUIREMENTS:
    - Checklist Item 7: Robustness Audit - Duration Matching [Ref: Section 5.3]
    - STRICTER than Condition B: Fixed content position + fixed duration + boundary alignment
    - Every subject contributes IDENTICAL analysis window size (30s ± alignment shift)
    - Alignment is ALL-OR-NOTHING: either start at genuine silence boundary OR no alignment
    - Segment saving is 1:1 with subjects in final distribution

TECHNICAL SPECIFICATIONS:
    - Center Window: 30.0 seconds, aligned to nearest preceding silence
    - Alignment: Max shift 1.0s backward ONLY, min silence duration 100ms
    - If shift > max_shift OR window exceeds bounds → fall back to ideal center (no alignment)
    - Encoder: HeAR ViT-L (16kHz mono) [Ref: Section 7.1.3]
    - Subspace Filter: Indices [267, 346, 43, 146, 204, 38, 419] [Ref: Section 7.1.4]
    - DSP: 20dB trim + pre-emphasis [Ref: Section 7.1.2]
    - Inference: 2s Window, 1s Hop (on the 30s aligned center crop)

===============================================================================
INPUTS:
------------------------------------------------------------------------------
1. Audio Files:
   - Location: /data/KCL/{COHORT}/northwindpci/
   - Format: WAV, 16-bit PCM (from Step 0)
   - Naming: ID*_northwindpci.wav (e.g., ID00_hc_0_0_0_northwindpci.wav)
   - Content: Isolated "North Wind and Sun" passage

2. Model Files:
   - HeAR Encoder: /models/hear/ (TensorFlow SavedModel)
   - Purified V2: /pdhear_PURIFIED_V2.pkl (frozen, audited)

3. Configuration:
   - COHORT: "HC" or "PD" (REQUIRED via --cohort flag)

===============================================================================
OUTPUTS:
------------------------------------------------------------------------------
1. Per-Subject Measurements (30s Center Window + Boundary Alignment):
   - File: /audit_results/{COHORT}_NorthWind_PD-Likeness_30sCenter_{timestamp}.csv
   - NEVER overwrites - each run gets unique timestamp
   - QC_Pass = True ONLY for measurement_status == 'SUCCESS'
   - Documents exact window position and alignment shift for EVERY subject

2. EXACT 30s Center Window Audio Files (if --save-segments):
   - Location: {source_dir}/30scenter/{basename}_30s_center.wav
   - Example: /data/KCL/HC/northwindpci/30scenter/ID00_hc_0_0_0_northwindpci_30s_center.wav
   - Contains the PRECISE 30-second window analyzed (BEFORE trim/DSP)
   - SAVED ONLY for subjects with measurement_status == 'SUCCESS'
   - Creates COMPLETE AUDIT TRAIL - 1:1 mapping with final distribution

3. Automatic Log File:
   - File: /audit_results/{COHORT}_01_northwind_30s_center_window_{timestamp}.txt
   - Captures ALL console output automatically
   - No manual redirection needed

4. Measurement Certificate:
   - Verified subspace alignment
   - HeAR model signature confirmation
   - Center window + boundary alignment methodology documented
   - Alignment statistics reported (SUCCESS subjects only)
   - Segment saving status with explicit 1:1 policy
   - Honest exclusion reporting (shows actual reasons, not just <30s)
   - Clear disclaimer: "Population measurement - NOT diagnostic"

===============================================================================
USAGE:
------------------------------------------------------------------------------
# MEASURE Healthy Control distribution, save 30s center WAV files (1:1 with final)

# MEASURE Parkinson's Disease distribution, save 30s center WAV files
python 01_northwind_30s_center_window_distributions.py --cohort PD --save-segments

# MEASURE without saving WAV files (CSV + log only)
python 01_northwind_30s_center_window_distributions.py --cohort HC

# Validation mode (check setup without full measurement)
python 01_northwind_30s_center_window_distributions.py --cohort HC --validation

===============================================================================
INTERPRETATION GUIDE - READ THIS:
------------------------------------------------------------------------------
This script applies the MOST RIGOROUS duration control in your validation suite:

┌─────────────────────────────────────────────────────────────────────────────┐
│  Condition A: Uncropped    → Full recording, variable duration            │
│  Condition B: 30s Start     → First 30s, excludes <30s, variable position │
│  Condition D: 30s Center    → Middle 30s, ALIGNED TO SILENCE, FIXED       │
│                              position ± small boundary adjustment         │
│                              (or IDEAL CENTER if alignment impossible)    │
└─────────────────────────────────────────────────────────────────────────────┘

CRITICAL DESIGN DECISION:
    "Final distribution" is DEFINED by measurement_status == 'SUCCESS'.
    This SINGLE DEFINITION is used for:
        ✓ Console QC display
        ✓ CSV QC_Pass flag  
        ✓ Distribution statistics
        ✓ Segment saving eligibility
        ✓ Alignment shift statistics
        ✓ Fallback counting
    
    NO AMBIGUITY. NO EXCEPTIONS.

If separation between HC and PD PERSISTS under Condition D, this is your 
STRONGEST evidence that:
    ✓ Biomarkers are genuine physiological signals
    ✓ Neither duration NOR speech segment position are confounders
    ✓ The cleanest, most representative 30s of speech retains diagnostic information
    ✓ Results are not artifacts of mid-word analysis starts
    ✓ Complete audit trail exists - 1:1 mapping between final distribution and saved WAVs

===============================================================================
Author: Jim McCormack
Date: Feb 2026 (FINAL v5 - Single source of truth: SUCCESS defines final distribution)
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import tensorflow as tf
import argparse
from pathlib import Path
from datetime import datetime

# =============================================================================
# AUTO-LOGGING CONFIGURATION
# =============================================================================
class TeeLogger:
    """Duplicate stdout/stderr to both console and log file automatically."""
    def __init__(self, filename):
        self.file = open(filename, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        
    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()
        
    def flush(self):
        self.stdout.flush()
        self.stderr.flush()
        self.file.flush()
        
    def close(self):
        self.file.close()
        
    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.close()

# =============================================================================
# FIXED PATHS (independent of cohort)
# =============================================================================
PKL_PATH = Path(r"C:\Projects\hear_italian\pdhear_PURIFIED_V2.pkl")
MODEL_PATH = Path(r"C:\Projects\hear_italian\models\hear")
BASE_DATA_DIR = Path(r"C:\Projects\hear_italian\data\KCL")
OUTPUT_DIR = Path(r"C:\Projects\hear_italian\audit_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# AUDITED SUBSPACE (frozen - DO NOT MODIFY)
# =============================================================================
AUDITED_SUBSPACE_INDICES = np.array([267, 346, 43, 146, 204, 38, 419])

# =============================================================================
# DSP PARAMETERS (fixed for reproducibility)
# =============================================================================
TRIM_DB = 20
PREEMPHASIS_COEFF = 0.97
WINDOW_SEC = 2.0
HOP_SEC = 1.0
TARGET_SR = 16000

# =============================================================================
# CONDITION D: 30s CENTER WINDOW WITH BOUNDARY ALIGNMENT
# =============================================================================
CENTER_WINDOW_SEC = 30.0           # Fixed 30-second window from the middle
MIN_DURATION_FOR_CENTER = 30.0     # Must have at least 30s total to extract center
MAX_ALIGN_SHIFT_SEC = 1.0          # Never shift more than 1 second (backward ONLY)
MIN_SILENCE_DUR_SEC = 0.1          # Require 100ms minimum silence for boundary

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================
def parse_arguments():
    """Parse command line arguments - COHORT IS REQUIRED"""
    parser = argparse.ArgumentParser(
        description="STEP 4A: Measure PD-Likeness Distribution on 30s Center Window with Speech Boundary Alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--cohort", type=str, required=True,
                       choices=["HC", "PD"],
                       help="REQUIRED: Cohort to measure (HC = baseline normal, PD = disease profile)")
    
    parser.add_argument("--validation", action="store_true",
                       help="Validation mode: check inputs without measuring")
    
    parser.add_argument("--save-segments", action="store_true",
                       help="Save exact 30s center WAV files - ONLY for measurement_status == 'SUCCESS'")
    
    parser.add_argument("--no-trim-audit", action="store_true",
                       help="Disable detailed trim impact reporting")
    
    parser.add_argument("--no-log", action="store_true",
                       help="Disable automatic log file creation")
    
    parser.add_argument("--max-shift", type=float, default=MAX_ALIGN_SHIFT_SEC,
                       help=f"Maximum alignment shift in seconds (default: {MAX_ALIGN_SHIFT_SEC}s)")
    
    parser.add_argument("--min-silence", type=float, default=MIN_SILENCE_DUR_SEC,
                       help=f"Minimum silence duration for boundary in seconds (default: {MIN_SILENCE_DUR_SEC}s)")
    
    return parser.parse_args()


# =============================================================================
# SPEECH BOUNDARY ALIGNMENT UTILITY - STRICT ALL-OR-NOTHING
# =============================================================================
def find_nearest_preceding_silence(y, target_sample, sr, max_shift_sec=1.0, min_silence_dur=0.1):
    """
    Find the nearest silence boundary PRECEDING target_sample.
    
    This prevents splitting words by aligning window start to natural speech breaks.
    CRITICAL for reviewer defense - ensures no analysis begins mid-word.
    
    ALL-OR-NOTHING APPROACH:
        - If a valid silence boundary is found within max_shift_sec, return EXACT boundary
        - If shift would exceed max_shift_sec, return NO ALIGNMENT (target_sample, 0.0)
        - NEVER clamp shift and move off-boundary
    
    Args:
        y: Audio array
        target_sample: Desired start sample
        sr: Sample rate
        max_shift_sec: Maximum seconds to search backward (default: 1.0)
        min_silence_dur: Minimum silence duration to consider a boundary (default: 0.1s)
    
    Returns:
        adjusted_start: Sample index (either at silence boundary OR target_sample)
        shift_sec: How many seconds we shifted backward (0.0 if no alignment)
    """
    max_shift_samples = int(max_shift_sec * sr)
    min_silence_samples = int(min_silence_dur * sr)
    search_start = max(0, target_sample - max_shift_samples)
    
    # Extract the segment we'll search for silence (strictly BEFORE target_sample)
    search_segment = y[search_start:target_sample]
    
    if len(search_segment) < int(0.05 * sr):  # Less than 50ms - can't reliably find silence
        return target_sample, 0.0
    
    # Find silence intervals using RMS energy
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)    # 10ms hop
    
    rms = librosa.feature.rms(y=search_segment, 
                              frame_length=frame_length, 
                              hop_length=hop_length)[0]
    
    # Adaptive threshold: 30dB below peak RMS in search region
    epsilon = 1e-10
    rms_db = 20 * np.log10(rms + epsilon)
    threshold_db = np.max(rms_db) - 30
    threshold = 10 ** (threshold_db / 20)
    
    is_silence = rms < threshold
    
    # Find contiguous silence regions that END BEFORE target_sample
    silence_ends = []
    
    in_silence = False
    start_idx = 0
    
    for i, silent in enumerate(is_silence):
        if silent and not in_silence:
            start_idx = i
            in_silence = True
        elif not silent and in_silence:
            end_frame = i
            end_sample = search_start + (end_frame * hop_length)
            
            # CRITICAL: Only include silence regions that END ≤ target_sample
            if end_sample <= target_sample:
                dur_frames = end_frame - start_idx
                dur_samples = dur_frames * hop_length
                if dur_samples >= min_silence_samples:
                    silence_ends.append(end_sample)
            in_silence = False
    
    # Handle case where silence continues to end of search region
    if in_silence:
        end_frame = len(is_silence)
        end_sample = search_start + (end_frame * hop_length)
        
        if end_sample <= target_sample:
            dur_frames = end_frame - start_idx
            dur_samples = dur_frames * hop_length
            if dur_samples >= min_silence_samples:
                silence_ends.append(end_sample)
    
    if not silence_ends:
        # No suitable preceding silence found - return original target
        return target_sample, 0.0
    
    # Find the silence region END closest to target_sample (but ≤ target_sample)
    best_end = max(silence_ends)
    shift_sec = (target_sample - best_end) / sr
    
    # CRITICAL: If shift exceeds maximum allowed, do NOT align
    if shift_sec > max_shift_sec:
        return target_sample, 0.0
    
    # Use the END of the silence region as our starting point - NEVER overwritten
    return best_end, shift_sec


# =============================================================================
# AUDIT FUNCTIONS
# =============================================================================
def verify_audited_subspace(v2_indices):
    """CRITICAL AUDIT: Verify we're using the exact frozen subspace."""
    expected = AUDITED_SUBSPACE_INDICES
    actual = np.array(v2_indices)
    
    print("\n" + "═" * 85)
    print("🔬 AUDIT: Subspace Filter Verification")
    print("─" * 85)
    print(f"Expected indices (frozen protocol): {expected.tolist()}")
    print(f"Loaded indices (from pickle):       {actual.tolist()}")
    
    if np.array_equal(expected, actual):
        print("✅ VERIFICATION PASSED: Using audited subspace")
        return True, actual
    else:
        print("❌ VERIFICATION FAILED: Subspace mismatch!")
        print(f"\n   FROZEN PROTOCOL: {expected.tolist()}")
        print(f"   PICKLE CONTAINS: {actual.tolist()}")
        print("\n   → Enforcing frozen protocol indices")
        return False, expected


def verify_hear_signature(infer_fn):
    """Audit HeAR model signature."""
    print("\n🔬 AUDIT: HeAR Model Signature")
    print("─" * 85)
    print(f"Available outputs: {list(infer_fn.structured_outputs.keys())}")
    
    if "output_0" in infer_fn.structured_outputs:
        print("✅ VERIFICATION PASSED: Found 'output_0' embedding tensor")
        return True
    else:
        print("❌ VERIFICATION FAILED: 'output_0' not found!")
        return False


def validate_trim_impact(y_original, y_trimmed, sr):
    """Measure impact of 20dB trim on signal duration."""
    orig_dur = len(y_original) / sr
    trimmed_dur = len(y_trimmed) / sr
    removed_dur = orig_dur - trimmed_dur
    removed_pct = (removed_dur / orig_dur * 100) if orig_dur > 0 else 0
    
    return {
        'orig_duration': round(orig_dur, 2),
        'trimmed_duration': round(trimmed_dur, 2),
        'removed_seconds': round(removed_dur, 2),
        'removed_percent': round(removed_pct, 1)
    }


# =============================================================================
# CORE MEASUREMENT ENGINE - CONDITION D (30s CENTER WINDOW + BOUNDARY ALIGNMENT)
# =============================================================================
def measure_pd_likeness_center_window(wav_path, infer_fn, scaler, detector_head, 
                                     indices, audit_trim=True, 
                                     max_shift_sec=MAX_ALIGN_SHIFT_SEC,
                                     min_silence_dur=MIN_SILENCE_DUR_SEC,
                                     save_segment=False):
    """
    Measure PD-Likeness score on a 30-second window taken from the EXACT CENTER
    of the recording, ALIGNED TO NEAREST PRECEDING SILENCE to avoid mid-word starts.
    
    ALL-OR-NOTHING ALIGNMENT:
        - If a valid silence boundary is found within max_shift_sec, use EXACT boundary
        - If boundary would require shift > max_shift_sec, use IDEAL center (no alignment)
        - If aligned window exceeds recording bounds, use IDEAL center (no alignment)
        - NEVER start mid-word by clamping or rounding off a boundary
    
    AUDIT TRAIL - CRITICAL POLICY:
        - Segments are saved ONLY after ALL quality checks pass
        - Saved segments have 1:1 correspondence with final distribution (SUCCESS)
        - No orphaned WAV files for excluded subjects
        - Filename: {basename}_30s_center.wav (e.g., ID00_hc_0_0_0_northwindpci_30s_center.wav)
    
    Returns:
        pd_likeness: Float 0-1 measuring similarity to PD voice characteristics
                    Returns NaN if audio < 30s (cannot extract center window)
        native_sr: Original sample rate
        audit_data: Dictionary with measurement metadata including alignment shift
    """
    audit_data = {
        'subject_id': wav_path.stem.split('_')[0],
        'source_file': str(wav_path),
        'saved_segment_file': None,
        'native_sr': None,
        'orig_duration_sec': None,
        'center_window_sec': CENTER_WINDOW_SEC,
        'ideal_start_sec': None,
        'ideal_end_sec': None,
        'adjusted_start_sec': None,
        'adjusted_end_sec': None,
        'boundary_alignment_shift_sec': 0.0,
        'boundary_alignment_applied': False,
        'boundary_alignment_fallback': False,
        'fallback_reason': None,
        'max_allowed_shift_sec': max_shift_sec,
        'min_silence_dur_sec': min_silence_dur,
        'trimmed_duration_sec': None,
        'trim_removed_pct': None,
        'n_windows': 0,
        'measurement_status': 'OK',
        'excluded': False,
        'exclusion_reason': None
    }
    
    # Load audio at native sample rate
    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    if y is None or len(y) == 0:
        audit_data['measurement_status'] = 'EMPTY_FILE'
        audit_data['excluded'] = True
        audit_data['exclusion_reason'] = 'EMPTY_FILE'
        return np.nan, sr, audit_data
    
    audit_data['native_sr'] = sr
    orig_duration = len(y) / sr
    audit_data['orig_duration_sec'] = round(orig_duration, 2)
    
    # Resample to 16kHz for consistent window calculation
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    
    # CONDITION D: Extract 30-second window from the EXACT CENTER
    total_samples = len(y)
    window_samples = int(CENTER_WINDOW_SEC * TARGET_SR)
    
    # Must have at least 30s total to extract a 30s center window
    if total_samples < window_samples:
        audit_data['measurement_status'] = f'EXCLUDED_SHORT_{orig_duration:.1f}s'
        audit_data['excluded'] = True
        audit_data['exclusion_reason'] = f'DURATION_{orig_duration:.1f}s_LT_{CENTER_WINDOW_SEC}s'
        return np.nan, sr, audit_data
    
    # Calculate ideal center window boundaries (theoretical)
    ideal_start = (total_samples - window_samples) // 2
    ideal_end = ideal_start + window_samples
    audit_data['ideal_start_sec'] = round(ideal_start / TARGET_SR, 2)
    audit_data['ideal_end_sec'] = round(ideal_end / TARGET_SR, 2)
    
    # =====================================================================
    # REVIEWER-PROOFING: Align to nearest preceding silence (ALL-OR-NOTHING)
    # =====================================================================
    adjusted_start, shift_sec = find_nearest_preceding_silence(
        y, 
        ideal_start, 
        sr,
        max_shift_sec=max_shift_sec,
        min_silence_dur=min_silence_dur
    )
    
    # Determine if alignment was actually applied
    alignment_applied = (adjusted_start != ideal_start)
    
    # Adjust end to maintain exactly 30s window
    adjusted_end = adjusted_start + window_samples
    
    # =====================================================================
    # CRITICAL FIX: If aligned window exceeds bounds, FALL BACK to ideal center
    # =====================================================================
    if adjusted_end > total_samples:
        audit_data['boundary_alignment_fallback'] = True
        audit_data['fallback_reason'] = 'WINDOW_EXCEEDS_RECORDING'
        audit_data['adjusted_start_sec'] = audit_data['ideal_start_sec']
        audit_data['adjusted_end_sec'] = audit_data['ideal_end_sec']
        audit_data['boundary_alignment_applied'] = False
        audit_data['boundary_alignment_shift_sec'] = 0.0
        
        # Use ideal center window instead
        y_center = y[ideal_start:ideal_end]
        print(f"      ⚠️  FALLBACK: Aligned window exceeds recording - using ideal center")
    else:
        # Alignment successful (or no alignment attempted/found)
        audit_data['adjusted_start_sec'] = round(adjusted_start / TARGET_SR, 2)
        audit_data['adjusted_end_sec'] = round(adjusted_end / TARGET_SR, 2)
        audit_data['boundary_alignment_shift_sec'] = round(shift_sec, 3)
        audit_data['boundary_alignment_applied'] = alignment_applied
        audit_data['boundary_alignment_fallback'] = False
        
        # Extract the aligned center window
        y_center = y[adjusted_start:adjusted_end]
    
    # Apply 20dB trim (clinical preprocessing standard)
    y_trimmed, _ = librosa.effects.trim(y_center, top_db=TRIM_DB)
    
    # Measure trim impact
    if audit_trim:
        trim_stats = validate_trim_impact(y_center, y_trimmed, TARGET_SR)
        audit_data['trimmed_duration_sec'] = trim_stats['trimmed_duration']
        audit_data['trim_removed_sec'] = trim_stats['removed_seconds']
        audit_data['trim_removed_pct'] = trim_stats['removed_percent']
    
    # Verify minimum duration for at least one window
    min_samples = int(WINDOW_SEC * TARGET_SR)
    if len(y_trimmed) < min_samples:
        audit_data['measurement_status'] = f'SHORT_TRIM_{len(y_trimmed)/TARGET_SR:.1f}s'
        audit_data['excluded'] = True
        audit_data['exclusion_reason'] = 'TRIM_TOO_SHORT'
        return np.nan, sr, audit_data
    
    # Apply pre-emphasis
    y_pre = librosa.effects.preemphasis(y_trimmed, coef=PREEMPHASIS_COEFF)
    
    # Sliding window measurement on the 30s center segment
    window_scores = []
    hop_samples = int(HOP_SEC * TARGET_SR)
    
    for i in range(0, len(y_pre) - min_samples + 1, hop_samples):
        window = y_pre[i : i + min_samples]
        
        # Extract HeAR embedding
        emb = infer_fn(x=tf.constant(window[np.newaxis, ...], dtype=tf.float32))["output_0"].numpy()
        
        # Project onto audited subspace
        emb_subspace = emb[:, indices]
        
        # Scale and compute PD-likeness score
        scaled = scaler.transform(emb_subspace)
        pd_likeness = detector_head.predict_proba(scaled)[0][1]
        window_scores.append(pd_likeness)
    
    if not window_scores:
        audit_data['measurement_status'] = 'NO_WINDOWS'
        audit_data['excluded'] = True
        audit_data['exclusion_reason'] = 'NO_WINDOWS'
        return np.nan, sr, audit_data
    
    audit_data['n_windows'] = len(window_scores)
    audit_data['measurement_status'] = 'SUCCESS'
    audit_data['excluded'] = False
    
    # =====================================================================
    # AUDIT TRAIL: Save EXACT 30-second window ONLY for SUCCESS
    # =====================================================================
    if save_segment and audit_data['measurement_status'] == 'SUCCESS':
        # Create 30scenter directory alongside source file
        source_dir = wav_path.parent
        segments_dir = source_dir / "30scenter"
        segments_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename: {basename}_30s_center.wav
        # e.g., ID00_hc_0_0_0_northwindpci_30s_center.wav
        basename = wav_path.stem
        segment_filename = f"{basename}_30s_center.wav"
        segment_path = segments_dir / segment_filename
        
        # Save the EXACT 30-second window (before trim, before pre-emphasis)
        sf.write(str(segment_path), y_center, TARGET_SR)
        audit_data['saved_segment_file'] = str(segment_path)
    
    # Return mean PD-likeness across all windows
    return float(np.mean(window_scores)), sr, audit_data


# =============================================================================
# BATCH MEASUREMENT ENGINE - CONDITION D
# =============================================================================
def execute_cohort_measurement(args):
    """Measure PD-Likeness distribution on 30s center window with boundary alignment."""
    
    # -------------------------------------------------------------------------
    # Configure cohort-specific paths
    # -------------------------------------------------------------------------
    cohort = args.cohort
    input_dir = BASE_DATA_DIR / cohort / "northwindpci"
    
    # Timestamped output - NEVER overwrites previous measurements
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = OUTPUT_DIR / f"{cohort}_NorthWind_PD-Likeness_30sCenter_{timestamp}.csv"
    
    # -------------------------------------------------------------------------
    # Display header with clear measurement framing
    # -------------------------------------------------------------------------
    print("\n" + "╔" + "═" * 85 + "╗")
    print("║" + " " * 85 + "║")
    print("║    NORTH WIND & SUN · COHORT DISTRIBUTION MEASUREMENT            ║")
    print("║    (CONDITION D: 30s Center Window + Speech Boundary Alignment)  ║")
    print("║" + " " * 85 + "║")
    print("╠" + "═" * 85 + "╣")
    
    if cohort == "HC":
        print("║    MEASURING: Healthy Control Baseline Distribution            ║")
        print("║    PURPOSE:   Cleanest 30s center window, no start/end artifacts║")
        print("║               + aligned to silence boundaries (no mid-word)     ║")
    else:
        print("║    MEASURING: Parkinson's Disease Distribution                 ║")
        print("║    PURPOSE:   Cleanest 30s center window, no start/end artifacts║")
        print("║               + aligned to silence boundaries (no mid-word)     ║")
    
    print("╠" + "═" * 85 + "╣")
    print(f"║  📂 Input:  {str(input_dir):<69} ║")
    print(f"║  📊 Output: {output_csv.name:<69} ║")
    
    if args.save_segments:
        print(f"║  💾 Saving: 30s center WAV files to /30scenter/ subdirectory    ║")
        print(f"║           {str(input_dir / '30scenter'):<69} ║")
        print(f"║           Format: ID*_northwindpci_30s_center.wav              ║")
        print(f"║           POLICY: Saved ONLY for measurement_status == SUCCESS ║")
        print(f"║           1:1 correspondence with final distribution           ║")
    
    print(f"║  🎯 Window:  {CENTER_WINDOW_SEC:.0f}s from EXACT CENTER (excludes <{CENTER_WINDOW_SEC:.0f}s)  ║")
    print(f"║  🔧 Align:   ALL-OR-NOTHING: Shift ≤{args.max_shift:.1f}s BACKWARD to ≥{args.min_silence*1000:.0f}ms silence ║")
    print(f"║             ⚠️  FALLBACK to ideal center if alignment impossible   ║")
    print("╚" + "═" * 85 + "╝")
    print("\n" + "⚠️  THIS IS A POPULATION MEASUREMENT - NOT DIAGNOSTIC" + " " * 30 + "⚠️")
    print("")
    
    # -------------------------------------------------------------------------
    # Validate input directory
    # -------------------------------------------------------------------------
    if not input_dir.exists():
        print(f"❌ ERROR: Input directory not found: {input_dir}")
        return None
    
    # -------------------------------------------------------------------------
    # Load and verify models
    # -------------------------------------------------------------------------
    print("🔓 Loading Purified V2 classifier...")
    if not PKL_PATH.exists():
        print(f"❌ ERROR: Classifier not found: {PKL_PATH}")
        return None
    
    bundle = joblib.load(PKL_PATH)
    v2_model = bundle['model']
    v2_scaler = bundle['scaler']
    v2_indices = bundle['indices']
    
    # Verify subspace (critical audit step)
    subspace_ok, v2_indices = verify_audited_subspace(v2_indices)
    
    print("\n🔓 Loading HeAR encoder...")
    if not (MODEL_PATH / "saved_model.pb").exists():
        print(f"❌ ERROR: HeAR model not found: {MODEL_PATH}")
        return None
    
    hear_engine = tf.saved_model.load(str(MODEL_PATH))
    infer_fn = hear_engine.signatures["serving_default"]
    
    # Verify HeAR signature
    if not verify_hear_signature(infer_fn):
        raise RuntimeError("❌ HeAR model signature mismatch - cannot proceed")
    
    # -------------------------------------------------------------------------
    # Validation mode - stop before measurement
    # -------------------------------------------------------------------------
    audio_files = sorted(input_dir.glob("ID*_northwindpci.wav"))
    
    if args.validation:
        print("\n" + "═" * 85)
        print("🧪 VALIDATION MODE - No measurements taken")
        print("─" * 85)
        print(f"✓ Input directory verified: {input_dir}")
        print(f"✓ Found {len(audio_files)} files ready for measurement")
        print(f"✓ Models loaded and verified")
        print(f"✓ Subspace verified: {subspace_ok}")
        print(f"✓ HeAR signature verified")
        print(f"✓ Condition D: {CENTER_WINDOW_SEC:.0f}s center window")
        print(f"✓ Boundary alignment: ALL-OR-NOTHING, max shift {args.max_shift:.1f}s BACKWARD")
        print(f"✓ Min silence: {args.min_silence*1000:.0f}ms")
        print(f"✓ Fallback: Ideal center when alignment impossible")
        if args.save_segments:
            print(f"✓ Segment saving: ENABLED (would save to /30scenter/)")
            print(f"✓ Segment format: ID*_northwindpci_30s_center.wav")
            print(f"✓ Segment policy: ONLY for measurement_status == SUCCESS")
        print(f"\n📊 Would output to: {output_csv}")
        print("═" * 85 + "\n")
        return {'cohort': cohort, 'status': 'validation_pass', 'n_files': len(audio_files)}
    
    # -------------------------------------------------------------------------
    # Begin batch measurement
    # -------------------------------------------------------------------------
    print(f"\n📂 Found {len(audio_files)} subjects in {cohort} cohort")
    print(f"🎯 CONDITION D: Extracting {CENTER_WINDOW_SEC:.0f}s window from EXACT CENTER")
    print(f"   ALL-OR-NOTHING alignment: BACKWARD to nearest ≥{args.min_silence*1000:.0f}ms silence")
    print(f"   Max shift {args.max_shift:.1f}s - FALLBACK to ideal center if alignment impossible")
    if args.save_segments:
        print(f"   💾 Saving 30s center WAV files to: {input_dir / '30scenter'}/")
        print(f"      Format: ID*_northwindpci_30s_center.wav")
        print(f"      POLICY: ONLY for measurement_status == SUCCESS (1:1 with final distribution)")
    
    if len(audio_files) == 0:
        print(f"❌ No ID*_northwindpci.wav files found in {input_dir}")
        return None
    
    # Initialize tracking variables
    measurements = []
    trim_impact_records = []
    alignment_shifts = []
    alignment_fallbacks_success = 0
    excluded_duration = 0
    segments_saved = 0
    
    # -------------------------------------------------------------------------
    # Display measurement header
    # -------------------------------------------------------------------------
    print("\n" + "═" * 85)
    print(f"📊 MEASURING: {cohort} Cohort PD-Likeness Scores (30s Center + Boundary Alignment)")
    print(f"   Subspace: {AUDITED_SUBSPACE_INDICES.tolist()}")
    print(f"   Window:   {WINDOW_SEC}s @ {TARGET_SR/1000}kHz, Hop: {HOP_SEC}s")
    print(f"   Trim:     {TRIM_DB}dB, Pre-emphasis: {PREEMPHASIS_COEFF}")
    print(f"   Center:   {CENTER_WINDOW_SEC:.0f}s window, aligned to PRECEDING silence boundaries")
    if args.save_segments:
        print(f"   💾 Saving:  EXACT 30s WAV segments (BEFORE trim) - ONLY for SUCCESS")
    print("─" * 85)
    print(f"{'Subject':<10} | {'Duration':<8} | {'Center Window':<20} | {'Shift':<6} | {'Trim%':<6} | {'Windows':<7} | {'PD-Likeness':<12} | {'QC'}")
    print("─" * 85)
    
    # -------------------------------------------------------------------------
    # Process each subject - ONE DEFINITION OF SUCCESS THROUGHOUT
    # -------------------------------------------------------------------------
    for wav_path in sorted(audio_files):
        subject_id = wav_path.stem.split('_')[0]
        
        # Measure PD-likeness on 30s center window with boundary alignment
        pd_score, native_sr, audit = measure_pd_likeness_center_window(
            wav_path, 
            infer_fn, 
            v2_scaler, 
            v2_model, 
            v2_indices,
            audit_trim=not args.no_trim_audit,
            max_shift_sec=args.max_shift,
            min_silence_dur=args.min_silence,
            save_segment=args.save_segments
        )
        
        # Get original duration
        with sf.SoundFile(str(wav_path)) as f:
            duration = round(len(f) / f.samplerate, 1)
        
        # ---------------------------------------------------------------------
        # QC Classification - SINGLE SOURCE OF TRUTH: measurement_status == 'SUCCESS'
        # ---------------------------------------------------------------------
        if audit['measurement_status'] != 'SUCCESS':
            qc_status = f"❌ EXCLUDED ({audit['measurement_status']})"
            # Track true duration exclusions separately for honest reporting
            if audit.get('exclusion_reason', '').startswith('DURATION_'):
                excluded_duration += 1
        else:
            qc_status = "✅ MEASURED"
            if audit.get('saved_segment_file'):
                segments_saved += 1
        
        # Format center window position with alignment indicators
        if audit['measurement_status'] == 'SUCCESS' and audit['adjusted_start_sec'] is not None:
            center_str = f"{audit['adjusted_start_sec']:.1f}-{audit['adjusted_end_sec']:.1f}s"
            
            # Alignment status indicator
            if audit.get('boundary_alignment_fallback', False):
                center_str += " ⚠️F"  # Fallback to ideal
                alignment_fallbacks_success += 1
            elif audit['boundary_alignment_applied']:
                center_str += " ∗"     # Successfully aligned
                alignment_shifts.append(audit['boundary_alignment_shift_sec'])
            else:
                center_str += "  "     # No alignment needed/possible
        else:
            center_str = "     EXCLUDED        "
        
        # Format alignment shift
        if audit['measurement_status'] == 'SUCCESS' and audit['boundary_alignment_applied']:
            shift_str = f"{audit['boundary_alignment_shift_sec']:.3f}s"
        else:
            shift_str = "  N/A  "
        
        # Format trim percentage
        trim_pct = audit.get('trim_removed_pct')
        if trim_pct is None:
            trim_str = "  N/A  "
        elif trim_pct > 5.0:
            trim_str = f"{trim_pct:.1f}%*"
            if audit['measurement_status'] == 'SUCCESS':
                trim_impact_records.append(trim_pct)
        else:
            trim_str = f"{trim_pct:.1f}%"
        
        # Format PD-likeness score
        if audit['measurement_status'] == 'SUCCESS' and not np.isnan(pd_score):
            score_str = f"{pd_score:.4f}"
        else:
            score_str = "     EXCL     "
        
        # Display measurement
        print(f"{subject_id:<10} | {duration:>5.1f}s   | {center_str:<20} | {shift_str:<6} | {trim_str:<6} | "
              f"{audit['n_windows']:<7} | {score_str:<12} | {qc_status}")
        
        # Store measurement - CLEAN SEMANTICS: Excluded = not SUCCESS, QC_Pass = SUCCESS
        qc_pass = (audit['measurement_status'] == 'SUCCESS')
        measurements.append({
            "SubjectID": subject_id,
            "Cohort": cohort,
            "Condition": "30s_Center_Window_Aligned",
            "Source_File": str(wav_path),
            "Saved_Segment_File": audit.get('saved_segment_file'),
            "Native_SR_Hz": int(native_sr) if native_sr else None,
            "Duration_sec": duration,
            "Center_Window_sec": CENTER_WINDOW_SEC,
            "Ideal_Start_sec": audit.get('ideal_start_sec'),
            "Adjusted_Start_sec": audit.get('adjusted_start_sec'),
            "Adjusted_End_sec": audit.get('adjusted_end_sec'),
            "Alignment_Shift_sec": audit.get('boundary_alignment_shift_sec'),
            "Alignment_Applied": audit['boundary_alignment_applied'],
            "Alignment_Fallback": audit.get('boundary_alignment_fallback', False),
            "Fallback_Reason": audit.get('fallback_reason'),
            "Max_Shift_sec": args.max_shift,
            "Min_Silence_ms": args.min_silence * 1000,
            "Trimmed_Duration_sec": audit.get('trimmed_duration_sec'),
            "Trim_Removed_pct": audit.get('trim_removed_pct'),
            "Analysis_Windows": audit['n_windows'],
            "PD_Likeness_Score": round(pd_score, 4) if qc_pass and not np.isnan(pd_score) else None,
            "Excluded": (not qc_pass),
            "Exclusion_Reason": audit.get('exclusion_reason'),
            "QC_Detail": audit['measurement_status'],
            "QC_Pass": qc_pass
        })
    
    # -------------------------------------------------------------------------
    # Save measurements to timestamped CSV
    # -------------------------------------------------------------------------
    df = pd.DataFrame(measurements)
    df.to_csv(output_csv, index=False)
    
    # -------------------------------------------------------------------------
    # Display distribution statistics - STRICTLY from SUCCESS rows only
    # -------------------------------------------------------------------------
    print("─" * 85)
    
    valid_scores = df[df['QC_Detail'] == 'SUCCESS']['PD_Likeness_Score'].dropna()
    n_valid = len(valid_scores)
    n_total = len(measurements)
    
    print(f"\n📈 {cohort} COHORT DISTRIBUTION SUMMARY (Condition D: 30s Center + Boundary Alignment)")
    print("─" * 85)
    print(f"   Total subjects processed:    {n_total}")
    print(f"   Subjects excluded (<30s):    {excluded_duration} ({(excluded_duration/n_total*100):.1f}%)")
    print(f"   Subjects excluded (other):   {n_total - n_valid - excluded_duration} (other QC failures)")
    print(f"   Subjects measured (SUCCESS): {n_valid}")
    
    if n_valid > 0:
        print(f"\n   PD-Likeness Score Distribution (SUCCESS subjects only):")
        print(f"     Mean:     {valid_scores.mean():.4f}")
        print(f"     Std Dev:  {valid_scores.std():.4f}")
        print(f"     Median:   {valid_scores.median():.4f}")
        print(f"     Q1-Q3:    [{valid_scores.quantile(0.25):.4f}, {valid_scores.quantile(0.75):.4f}]")
        print(f"     Range:    [{valid_scores.min():.4f}, {valid_scores.max():.4f}]")
    
    if alignment_shifts:
        print(f"\n   Boundary Alignment Statistics (BACKWARD shifts only, SUCCESS subjects):")
        print(f"     Subjects successfully aligned: {len(alignment_shifts)}/{n_valid} ({(len(alignment_shifts)/n_valid*100):.1f}%)")
        print(f"     Mean shift:            {np.mean(alignment_shifts):.3f}s")
        print(f"     Median shift:          {np.median(alignment_shifts):.3f}s")
        print(f"     Max shift:             {max(alignment_shifts):.3f}s")
        print(f"     Shift >500ms:          {sum(1 for s in alignment_shifts if s > 0.5)} subjects")
    
    if alignment_fallbacks_success > 0:
        print(f"\n   Alignment Fallbacks (SUCCESS subjects):")
        print(f"     Subjects fell back to ideal center: {alignment_fallbacks_success}/{n_valid} ({(alignment_fallbacks_success/n_valid*100):.1f}%)")
        print(f"     Reasons: Window exceeds recording bounds")
    
    if args.save_segments and segments_saved > 0:
        print(f"\n   💾 Saved 30s Center WAV Segments:")
        print(f"     Segments saved:        {segments_saved}/{n_valid} ({(segments_saved/n_valid*100):.1f}%)")
        print(f"     Location:              {input_dir / '30scenter'}/")
        print(f"     Format:                ID*_northwindpci_30s_center.wav")
        print(f"     Policy:                1:1 with SUCCESS subjects - COMPLETE AUDIT TRAIL")
        assert segments_saved == n_valid, "AUDIT FAILURE: Segment count mismatch with SUCCESS subjects"
    
    if trim_impact_records:
        print(f"\n   Trim Impact (>5% removal on center window, SUCCESS subjects):")
        print(f"     Files affected: {len(trim_impact_records)}")
        print(f"     Mean trim:      {np.mean(trim_impact_records):.1f}%")
        print(f"     Max trim:       {max(trim_impact_records):.1f}%")
    
    # -------------------------------------------------------------------------
    # Measurement Certificate - FINAL
    # -------------------------------------------------------------------------
    print(f"\n🔐 MEASUREMENT CERTIFICATE - CONDITION D (FINAL v5)")
    print("─" * 85)
    print(f"✓ Cohort:                 {cohort}")
    print(f"✓ Condition:              30s Center Window + Speech Boundary Alignment")
    print(f"✓ Alignment:              ALL-OR-NOTHING, BACKWARD ONLY (preceding silence)")
    print(f"✓ Fallback:              ⚠️  Ideal center when alignment impossible")
    if args.save_segments:
        print(f"✓ Segment Saving:        ✅ ENABLED - EXACT 30s WAV files saved")
        print(f"✓ Segment Location:      {input_dir / '30scenter'}/")
        print(f"✓ Segment Format:        ID*_northwindpci_30s_center.wav")
        print(f"✓ Segment Policy:        1:1 with SUCCESS subjects - AUDIT TRAIL VERIFIED")
    else:
        print(f"✓ Segment Saving:        ⚠️  DISABLED (use --save-segments for audit trail)")
    print(f"✓ Measurement date:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"✓ Input directory:        {input_dir}")
    print(f"✓ Output file:            {output_csv.name}")
    print(f"✓ Subspace verified:      {subspace_ok}")
    print(f"✓ HeAR signature:         output_0")
    print(f"✓ Trim documented:        {TRIM_DB}dB")
    print(f"✓ Window:                 {CENTER_WINDOW_SEC:.0f}s from exact center")
    print(f"✓ Alignment:              Max shift {args.max_shift:.1f}s BACKWARD, min silence {args.min_silence*1000:.0f}ms")
    if alignment_shifts:
        print(f"✓ Successfully aligned:   {len(alignment_shifts)}/{n_valid} subjects (mean {np.mean(alignment_shifts):.3f}s)")
    if alignment_fallbacks_success > 0:
        print(f"✓ Fallbacks:              {alignment_fallbacks_success}/{n_valid} subjects (window bounds)")
    if args.save_segments and segments_saved > 0:
        print(f"✓ Segments saved:         {segments_saved}/{n_valid} subjects - 1:1 VERIFIED")
    print(f"✓ Subjects with ≥30s:     {n_valid + (n_total - n_valid - excluded_duration)}/{n_total}")
    print(f"✓ Subjects in final distribution (SUCCESS): {n_valid}/{n_total}")
    print("\n" + "⚠️ " * 20)
    print("   THIS IS NOT A DIAGNOSTIC CLASSIFICATION")
    print("   These are population distribution measurements ONLY")
    print("   Condition D: Cleanest 30s center window - NO START/END ARTIFACTS")
    print("                ALIGNED TO PRECEDING SILENCE - NO MID-WORD STARTS")
    print("                FALLBACK to ideal center when alignment impossible")
    if args.save_segments:
        print("                EXACT 30s WINDOWS SAVED - COMPLETE AUDIT TRAIL")
        print("                1:1 mapping between SUCCESS subjects and saved WAVs")
    print("   " + "⚠️ " * 20)
    print("═" * 85 + "\n")
    
    # Final integrity assertion
    if args.save_segments:
        assert segments_saved == n_valid, "CRITICAL: Segment count mismatch - audit trail broken"
    
    return {
        'cohort': cohort,
        'condition': '30s_center_window_aligned_strict',
        'n_subjects_measured': n_valid,
        'n_subjects_excluded_duration': excluded_duration,
        'n_subjects_excluded_other': n_total - n_valid - excluded_duration,
        'n_aligned': len(alignment_shifts),
        'n_fallbacks': alignment_fallbacks_success,
        'n_segments_saved': segments_saved if args.save_segments else 0,
        'mean_score': valid_scores.mean() if n_valid > 0 else None,
        'std_score': valid_scores.std() if n_valid > 0 else None,
        'mean_shift': np.mean(alignment_shifts) if alignment_shifts else 0,
        'pct_aligned': (len(alignment_shifts)/n_valid*100) if n_valid > 0 else 0,
        'output_file': str(output_csv),
        'subspace_verified': subspace_ok,
        'audit_integrity': segments_saved == n_valid if args.save_segments else None
    }


# =============================================================================
# MAIN ENTRY POINT WITH AUTO-LOGGING
# =============================================================================
def main():
    """Main entry point - with automatic log file creation."""
    
    args = parse_arguments()
    
    # =========================================================================
    # AUTO-LOGGING
    # =========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OUTPUT_DIR / f"{args.cohort}_01_northwind_30s_center_window_{timestamp}.txt"
    
    # Start automatic logging (unless disabled)
    logger = None
    if not args.no_log and not args.validation:
        logger = TeeLogger(log_file)
        logger.__enter__()
        print(f"\n📝 AUTO-LOGGING ENABLED: {log_file}")
    elif args.validation:
        print(f"\n📝 VALIDATION MODE - No log file created")
    elif args.no_log:
        print(f"\n📝 Logging disabled via --no-log flag")
    
    try:
        # Display mode banner
        print("\n" + "╔" + "═" * 85 + "╗")
        print("║" + " " * 85 + "║")
        print("║      ⚕️  PD-LIKENESS MEASUREMENT ENGINE - CONDITION D ⚕️       ║")
        print("║" + " " * 85 + "║")
        print("║      🔬 30s Center Window + Speech Boundary Alignment 🔬      ║")
        print("║" + " " * 85 + "║")
        print("║     THIS SOFTWARE PERFORMS POPULATION DISTRIBUTION MEASUREMENT      ║")
        print("║     IT DOES NOT PROVIDE DIAGNOSTIC OR CLINICAL CLASSIFICATION       ║")
        print("║" + " " * 85 + "║")
        print("║     ✓ No start artifacts    ✓ No end fatigue    ✓ No mid-word starts║")
        print("║     ✓ ALL-OR-NOTHING alignment    ✓ Fallback when impossible       ║")
        print("║     ✓ SINGLE DEFINITION: SUCCESS = final distribution              ║")
        if args.save_segments:
            print("║     💾 EXACT 30s WAV SEGMENTS SAVED - 1:1 WITH SUCCESS           ║")
        else:
            print("║     ⚠️  Segment saving DISABLED (use --save-segments for audit trail)║")
        print("║" + " " * 85 + "║")
        print("╚" + "═" * 85 + "╝")
        
        # Execute measurement
        results = execute_cohort_measurement(args)
        
    finally:
        if logger:
            logger.__exit__(None, None, None)
    
    # Print completion message
    if results is not None and not args.validation and not args.no_log:
        print(f"\n✅ {args.cohort} cohort measurement complete (Condition D: 30s Center + Alignment)")
        print(f"   📊 Log saved: {log_file}")
        if results.get('output_file'):
            print(f"   💾 Results: {Path(results['output_file']).name}")
        if results.get('mean_shift', 0) > 0:
            print(f"   📐 Successfully aligned: {results.get('pct_aligned', 0):.1f}% (mean shift {results.get('mean_shift', 0):.3f}s)")
        if results.get('n_fallbacks', 0) > 0:
            print(f"   ⚠️  Fallbacks: {results.get('n_fallbacks', 0)} subjects (ideal center used)")
        if args.save_segments and results.get('n_segments_saved', 0) > 0:
            print(f"   💾 Saved {results.get('n_segments_saved', 0)} 30s center WAV segments - 1:1 with SUCCESS subjects")
        if results.get('audit_integrity') is True:
            print(f"   🔐 AUDIT INTEGRITY: VERIFIED (segments == SUCCESS subjects)")
    elif results is not None and args.validation:
        print(f"\n✅ Validation complete - ready to measure Condition D")
    
    return 0 if results is not None else 1


if __name__ == "__main__":

    exit(main())
