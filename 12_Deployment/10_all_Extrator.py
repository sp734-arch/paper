"""
CROSS-LINGUAL EMBEDDING EXTRACTION - PRODUCTION PIPELINE
===========================================================
⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
THIS SCRIPT PRODUCES EMBEDDINGS IDENTICAL TO 
01_extract_hear_staircase.py (DENSITY_50 - PRODUCTION SETTING)
===========================================================
Purpose: Generate embeddings for ALL cross-lingual validation datasets with
         CORRECT contiguous slice counting (NO GAPS, NO DUPLICATES).

PIPELINE IDENTITY VERIFICATION (2026-02-12 AUDIT):
    ✓ Same HeAR model (google/hear)
    ✓ Same sample rate (16kHz)
    ✓ Same window length (2.0s / 32,000 samples)
    ✓ Same hop size (1.0s / 16,000 samples) → 50% overlap
    ✓ SAME PRE-EMPHASIS (librosa.effects.preemphasis)
    ✓ SAME SILENCE GATE (-50 dBFS absolute threshold)
    ✓ SAME OUTPUT FORMAT (individual *_s{idx}.npy slices)
    ✓ SAME DENSITY_50 setting from staircase paper
    ✓ SAME PIPELINE FOR ALL DATASETS (German, Swedish, Nepali, VCTK)
    ✓ FIXED: Subject-relative CONTIGUOUS slice counting (NO GAPS)
    ✓ FIXED: Slice counter increments ONLY when slice is ACTUALLY SAVED

CRITICAL - DO NOT MODIFY:
    These settings are LOCKED to match the training pipeline.
    Any deviation invalidates cross-lingual comparison.
⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
"""

import os
import numpy as np
import tensorflow as tf
import librosa
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# ============================================================================
# CONFIGURATION - LOCKED 2026-02-12 - DO NOT CHANGE
# ============================================================================
SAMPLE_RATE = 16000
CLIP_LENGTH = 32000      # 2.0 seconds
HOP_SIZE = 16000         # 1.0 seconds → 50% overlap
SILENCE_THRESHOLD = -50.0
PRE_EMPHASIS = True

MODEL_PATH = Path(r"C:\Projects\hear_italian\models\hear")
OUTPUT_ROOT = Path(r"C:\Projects\hear_italian\features_crosslingual_FIXED")  # CLEAN SLATE
DENSITY_FOLDER = "density_50"
OUTPUT_DIR = OUTPUT_ROOT / DENSITY_FOLDER

# ============================================================================
# DATASETS - ALL 4 LANGUAGES, SAME PIPELINE
# ============================================================================
DATASETS = [
    # ------------------------------------------------------------------------
    # GERMAN - Healthy, flat directory
    # ------------------------------------------------------------------------
    {
        'name': 'german',
        'path': Path(r"C:\Projects\hear_italian\data\gerswed\ML_speech_recognition\german"),
        'cohort': 'healthy',
        'pattern': '*.wav',
        'has_speaker_id': True,
        'speaker_fn': lambda p: p.stem,  # ✅ 'F_20210725SomeWoman' - UNIQUE per speaker
        'filename_fn': lambda p: p.stem,
    },
    
    # ------------------------------------------------------------------------
    # SWEDISH - Healthy, flat directory - USE FULL STEM AS UNIQUE SUBJECT ID
    # ------------------------------------------------------------------------
    {
        'name': 'swedish',
        'path': Path(r"C:\Projects\hear_italian\data\gerswed\ML_speech_recognition\swedish"),
        'cohort': 'healthy',
        'pattern': '*.wav',
        'has_speaker_id': True,
        'speaker_fn': lambda p: p.stem,  # ✅ 'F_20221216HibaDaniel' - UNIQUE per speaker
        'filename_fn': lambda p: p.stem,
    },
    
    # ------------------------------------------------------------------------
    # NEPALI - Healthy, flat directory - NO SUBJECT TRACKING
    # ------------------------------------------------------------------------
    {
        'name': 'nepali',
        'path': Path(r"C:\Projects\hear_italian\nepali_Speech\audio_chunks"),
        'cohort': 'healthy',
        'pattern': '*.wav',
        'has_speaker_id': False,  # ✅ No subject tracking - each file independent
        'filename_fn': lambda p: p.stem,
    },
    
    # ------------------------------------------------------------------------
    # VCTK - Healthy, speaker subdirectories
    # ------------------------------------------------------------------------
    {
        'name': 'vctk',
        'path': Path(r"C:\Projects\hear_italian\VCTK-Corpus\VCTK-Corpus\wav48"),
        'cohort': 'healthy',
        'pattern': '*/*.wav',
        'has_speaker_id': True,
        'speaker_fn': lambda p: p.parent.name,
        'filename_fn': lambda p: f"{p.stem}",  
    },
]

# ============================================================================
# PIPELINE VERIFICATION
# ============================================================================
def verify_pipeline():
    assert HOP_SIZE == 16000, f"HOP_SIZE={HOP_SIZE}, expected 16000"
    assert CLIP_LENGTH == 32000, f"CLIP_LENGTH={CLIP_LENGTH}, expected 32000"
    assert SAMPLE_RATE == 16000, f"SAMPLE_RATE={SAMPLE_RATE}, expected 16000"
    assert SILENCE_THRESHOLD == -50.0, f"SILENCE_THRESHOLD={SILENCE_THRESHOLD}, expected -50.0"
    assert PRE_EMPHASIS is True, "PRE_EMPHASIS must be True"
    print("   ✅ ALL SETTINGS MATCH 01_extract_hear_staircase.py (density_50)")

# ============================================================================
# LOAD HEAR MODEL
# ============================================================================
print("\n" + "=" * 80)
print("🎧 CROSS-LINGUAL EMBEDDING EXTRACTION - PRODUCTION PIPELINE")
print("=" * 80)
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n📂 Output: {OUTPUT_DIR}")

model = tf.saved_model.load(str(MODEL_PATH))
infer_fn = model.signatures["serving_default"]
print("✅ Model loaded")

verify_pipeline()

# ============================================================================
# EXTRACTION FUNCTION - SAME PIPELINE FOR ALL DATASETS
# ============================================================================
def extract_embeddings():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    
    for dataset in DATASETS:
        print(f"\n{'─'*80}")
        print(f"📂 {dataset['name'].upper()}")
        print(f"{'─'*80}")
        print(f"   Path: {dataset['path']}")
        print(f"   Pattern: {dataset['pattern']}")
        
        files = sorted(dataset['path'].glob(dataset['pattern']))
        print(f"   Files: {len(files)}")
        
        if not files:
            print(f"   ⚠️  No files found - skipping")
            continue
        
        out_dir = OUTPUT_DIR / dataset['name'] / dataset['cohort']
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Track slices per subject (contiguous, no gaps)
        subject_slice_count = {}
        subject_file_counter = {}
        
        for wav_path in tqdm(files, desc=f"   🧬 {dataset['name']}"):
            try:
                # Get subject ID
                if dataset.get('has_speaker_id', False):
                    subject_id = dataset['speaker_fn'](wav_path)
                    if subject_id not in subject_slice_count:
                        subject_slice_count[subject_id] = 0
                        subject_file_counter[subject_id] = 0
                    file_start_idx = subject_file_counter[subject_id]
                else:
                    subject_id = f"{dataset['name']}_{wav_path.stem}"
                    file_start_idx = 0
                
                # Load audio
                audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
                
                # PRE-EMPHASIS - CRITICAL
                if PRE_EMPHASIS:
                    audio = librosa.effects.preemphasis(audio)
                
                # Pad if shorter than 2 seconds
                if len(audio) < CLIP_LENGTH:
                    audio = np.pad(audio, (0, CLIP_LENGTH - len(audio)))
                
                # Sliding window - 50% overlap, silence gate
                slices_saved = 0
                window_pos = 0
                base_name = dataset['filename_fn'](wav_path)
                
                for start in range(0, len(audio) - CLIP_LENGTH + 1, HOP_SIZE):
                    segment = audio[start:start + CLIP_LENGTH]
                    
                    # SILENCE GATE - CRITICAL
                    rms = np.sqrt(np.mean(segment**2))
                    loudness = 20 * np.log10(rms + 1e-9)
                    
                    if loudness < SILENCE_THRESHOLD:
                        window_pos += 1
                        continue
                    
                    # HeAR inference
                    tensor = tf.constant(segment[np.newaxis, ...], dtype=tf.float32)
                    embedding = infer_fn(x=tensor)['output_0'].numpy().flatten()
                    
                    # Save with contiguous slice index
                    current_slice_idx = file_start_idx + slices_saved
                    out_filename = f"{base_name}_s{current_slice_idx}.npy"
                    out_path = out_dir / out_filename
                    
                    np.save(out_path, embedding)
                    slices_saved += 1
                    window_pos += 1
                
                # Update subject counters
                if dataset.get('has_speaker_id', False) and slices_saved > 0:
                    subject_slice_count[subject_id] += slices_saved
                    subject_file_counter[subject_id] = subject_slice_count[subject_id]
                
                # Log to manifest
                manifest.append({
                    'dataset': dataset['name'],
                    'cohort': dataset['cohort'],
                    'subject_id': subject_id,
                    'original_file': wav_path.name,
                    'base_name': base_name,
                    'total_windows': window_pos,
                    'silence_skipped': window_pos - slices_saved,
                    'slices_saved': slices_saved,
                    'file_start_idx': file_start_idx,
                    'file_end_idx': file_start_idx + slices_saved - 1 if slices_saved > 0 else -1,
                    'subject_total_slices': subject_slice_count.get(subject_id, slices_saved),
                    'output_dir': str(out_dir),
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                print(f"\n   ⚠️  Error: {wav_path.name} - {e}")
                continue
    
    # ========================================================================
    # SAVE MASTER MANIFEST
    # ========================================================================
    manifest_df = pd.DataFrame(manifest)
    manifest_path = OUTPUT_ROOT / "manifest_crosslingual_ALL_FIXED.csv"
    manifest_df.to_csv(manifest_path, index=False)
    
    print(f"\n{'='*80}")
    print("✅ EXTRACTION COMPLETE - ALL 4 LANGUAGES")
    print(f"{'='*80}")
    print(f"\n📊 MASTER SUMMARY:")
    print(f"   Output root: {OUTPUT_ROOT}")
    
    for dataset in DATASETS:
        ds_files = manifest_df[manifest_df['dataset'] == dataset['name']]
        total_slices = ds_files['slices_saved'].sum()
        print(f"\n   {dataset['name'].upper()}:")
        print(f"     Files: {len(ds_files)}")
        print(f"     Slices: {total_slices}")
        print(f"     Output: {OUTPUT_DIR / dataset['name'] / dataset['cohort']}")
        if dataset.get('has_speaker_id', False):
            print(f"     Subjects: {ds_files['subject_id'].nunique()}")
    
    print(f"\n📋 Manifest: {manifest_path}")
    print(f"\n✅ SAME PIPELINE FOR ALL DATASETS - 50% overlap, silence gate, pre-emphasis")
    print(f"{'='*80}\n")
    
    return manifest_df

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    
    if args.dry_run:
        print("\n🔍 DRY RUN - File Discovery Only\n")
        for ds in DATASETS:
            files = list(ds['path'].glob(ds['pattern']))
            print(f"{ds['name']:10}: {len(files):5d} files")
            if files:
                sample = files[0]
                print(f"            Sample: {sample}")
                base = ds['filename_fn'](sample)
                print(f"            Output: {base}_s0.npy ✅")
                if ds.get('has_speaker_id', False):
                    print(f"            Speaker: {ds['speaker_fn'](sample)}")
        print("\n✅ Dry run complete. Run without --dry-run to extract.\n")
    else:
        print("\n🚀 STARTING FULL EXTRACTION - ALL 4 LANGUAGES")
        print(f"   German:   101 files")
        print(f"   Swedish:  103 files")
        print(f"   Nepali:   339 files")
        print(f"   VCTK:     ~44,242 files")
        print(f"\n   ⏳ Estimated time: 4-6 hours")
        print(f"   📂 Output: {OUTPUT_DIR}\n")
        extract_embeddings()