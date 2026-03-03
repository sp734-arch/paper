#!/usr/bin/env python
"""
Identity Leakage test
CERTIFIED SLICE-LEVEL FEATURE EXTRACTOR - STEP 1
================================================================================
PURPOSE:
    Extracts 2-second sliding windows from audio files, generates 512D HeAR embeddings,
    projects to a 7D audited subspace, and scales for downstream analysis.
    Produces slice-level embeddings with full provenance tracking and cryptographic hashes.

DEPENDENCY CHAIN (CRITICAL FOR REPRODUCIBILITY):
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║  THIS SCRIPT PERFORMS THE ACTUAL FEATURE EXTRACTION                     ║
    ║  Outputs are used by Step 2 experiments and all downstream analyses.    ║
    ║                                                                          ║
    ║  Required inputs:                                                        ║
    ║    - HeAR model: C:\Projects\hear_italian\models\hear\pytorch_model.bin ║
    ║    - PD bundle: C:\Projects\hear_italian\WAVstudy\pdhear_PURIFIED_V2_HASHED.pkl ║
    ║    - WAV files: Organized by cohort in C:\Projects\hear_italian\data    ║
    ╚══════════════════════════════════════════════════════════════════════════╝

DATA FLOW:
    Input WAV files ─────────────────────────────────┐
        ↓                                            │
    HeAR model (HuggingFace ViT) ────────────────────┤
        ↓                                            │
    512D raw embeddings ─────────────────────────────┤
        ↓                                            │
    Project to 7D using audited indices [266,345,42,145,203,37,418] ──┤
        ↓                                            │
    Scale 7D subspace ───────────────────────────────┤
        ↓                                            │
    Save embeddings + metadata with SHA256 hashes ───┘
        ↓
    Output files in: C:\Projects\hear_italian\features_certified

USAGE:
    # Process all data (cross-lingual cohorts)
    python 11_certified_extractor_pytorch.py --device cuda
    
    # Process only KCL Condition D 30s center segments
    python 11_certified_extractor_pytorch.py --kcl_30s --device cuda
    
    # Process with custom silence threshold (default -55 dBFS)
    python 11_certified_extractor_pytorch.py --kcl_30s --device cuda --silence_thresh -60
    
    # Limit to first N files (for testing)
    python 11_certified_extractor_pytorch.py --kcl_30s --device cuda --max_files 10

INPUT FILES:
    - WAV files: Any .wav files in the data directory (recursive search)
    - For KCL 30s mode: Specifically looks in:
        HC: C:\Projects\hear_italian\data\KCL\HC\northwindpci\30scenter\
        PD: C:\Projects\hear_italian\data\KCL\PD\northwindpci\30scenter\

OUTPUT FILES (prefix depends on mode):
    With --kcl_30s:    kcl_condition_d_30s_*
    Without --kcl_30s: cross_lingual_real_*
    
    Files created:
    - {prefix}_embeddings_512_raw.npy      # Raw 512D embeddings
    - {prefix}_embeddings_7d_raw.npy       # Raw 7D projections (unscaled)
    - {prefix}_embeddings_7d_scaled.npy    # Scaled 7D (for PD classification)
    - {prefix}_metadata.csv                 # Slice-level metadata
    - {prefix}_manifest.csv                  # Slice manifest
    - {prefix}_hashes.json                   # Cryptographic hashes
    - {prefix}_certification.txt              # Certification report

PROCESSING PARAMETERS (FROZEN):
    • Window:        2.0 seconds (32000 samples @ 16kHz)
    • Hop:           1.0 seconds (16000 samples)
    • Silence threshold: -55 dBFS (default, adjustable via --silence_thresh)
    • Pre-emphasis:  0.97
    • Spectrogram:   192 mel bands × 128 time bins
    • Indices (1-based): [267, 346, 43, 146, 204, 38, 419]
    • Indices (0-based): [266, 345, 42, 145, 203, 37, 418]
    • Deterministic algorithms: ENABLED
    • CPU threads:    1

NOTES:
    - Subjects with 0 slices after thresholding get placeholder rows (slice_index = -1)
    - Placeholders must be removed before Step 2 (handled by downstream scripts)
    - All outputs include SHA256 hashes for integrity verification
    - The -55 dBFS threshold was selected via grid search as optimal

VERSION HISTORY:
    v1.0 (2026-02-12): Initial release
    v1.1 (2026-02-14): Added placeholder rows for 0-slice subjects
    v1.2 (2026-02-16): Fixed Unicode encoding issues
    v1.3 (2026-02-24): Set default threshold to -55 dBFS (optimal from grid search)
================================================================================
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import joblib
import hashlib
import json
import warnings
import soundfile as sf
import torch
import torchaudio.transforms as T
from datetime import datetime
warnings.filterwarnings('ignore')


import sys
import traceback

def global_excepthook(exctype, value, tb):
    print("\n" + "="*70)
    print("  UNCAUGHT EXCEPTION:")
    print("="*70)
    traceback.print_exception(exctype, value, tb)
    print("="*70)
    sys.__excepthook__(exctype, value, tb)

sys.excepthook = global_excepthook


# ===== REPRODUCIBILITY SETTINGS =====
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Add hear repository to path
sys.path.append(r'C:\Projects\hear_italian')

# ===== SIMPLE PREPROCESSING FUNCTIONS =====
def load_audio_mono(wav_path, target_sr=16000):
    """Load audio as mono, resample to target_sr"""
    audio, sr = sf.read(wav_path)
    
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    if sr != target_sr:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            print("      Warning: librosa not available, using original sr")
    
    return audio, target_sr

def apply_pre_emphasis_librosa(audio, coeff=0.97):
    """Apply pre-emphasis filter"""
    return np.append(audio[0], audio[1:] - coeff * audio[:-1])

def compute_window_rms_db(window, eps=1e-12):
    """Compute RMS in dBFS"""
    rms = np.sqrt(np.mean(window**2))
    return 20 * np.log10(rms + eps)

STEP1_AVAILABLE = True
print("Preprocessing functions loaded")


class CertifiedSliceExtractor:
    """
    Produces slice-level embeddings matching Step 1 pipeline EXACTLY.
    Uses joblib for PD bundle with 1-based indices.
    Projects to 7D THEN scales (scaler was fit on 7D subspace).
    """
    
    def __init__(self, 
                 hear_model_path=r'C:\Projects\hear_italian\models\hear\pytorch_model.bin',
                 pd_model_path=r'C:\Projects\hear_italian\WAVstudy\pdhear_PURIFIED_V2_HASHED.pkl',
                 output_root=r'C:\Projects\hear_italian\features_certified',
                 indices_1based=[267, 346, 43, 146, 204, 38, 419],  # From hashed_looker.py
                 window_sec=2.0,
                 hop_sec=1.0,
                 target_sr=16000,
                 silence_thresh_db=-55,
                 pre_emphasis_coeff=0.97,
                 device='cpu',
                 allow_partial_model=False,
                 kcl_30s_mode=False):  # NEW FLAG
        
        self.hear_model_path = Path(hear_model_path)
        self.pd_model_path = Path(pd_model_path)
        self.output_root = Path(output_root)
        self.kcl_30s_mode = kcl_30s_mode  # NEW
        
        # Convert 1-based indices to 0-based for Python indexing
        self.indices_1based = indices_1based
        self.indices = [i - 1 for i in indices_1based]  # Convert to 0-based
        print(f"\n     Index conversion:")
        print(f"   1-based indices: {indices_1based}")
        print(f"   0-based indices: {self.indices}")
        
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.target_sr = target_sr
        self.silence_thresh_db = silence_thresh_db
        self.pre_emphasis_coeff = pre_emphasis_coeff
        self.device = device
        self.allow_partial_model = allow_partial_model
        
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Compute window/hop samples
        self.window_samples = int(window_sec * target_sr)
        self.hop_samples = int(hop_sec * target_sr)
        
        # Spectrogram parameters (from config)
        self.n_mels = 192
        self.target_width = 128
        self.n_fft = 400
        self.hop_length = 160
        
        # ===== LOAD HeAR MODEL (HuggingFace) =====
        print(f"\n Loading HeAR HuggingFace model from: {hear_model_path}")
        model_dir = Path(hear_model_path).parent
        print(f"   Model directory: {model_dir}")
        print(f"   Config file: {model_dir / 'config.json'}")
        print(f"   Allow partial model: {allow_partial_model}")
        
        try:
            from transformers import AutoModel
            
            self.hear_model = AutoModel.from_pretrained(
                str(model_dir), 
                trust_remote_code=True
            )
            self.hear_model.to(device)
            self.hear_model.eval()
            
            # ===== TEST INFERENCE =====
            print("\n     Testing model output...")
            dummy_audio = np.random.randn(self.window_samples).astype(np.float32)
            dummy_spec = self.audio_to_spectrogram(dummy_audio)
            dummy_spec = dummy_spec.to(device)
            
            with torch.no_grad():
                outputs = self.hear_model(pixel_values=dummy_spec)
            
            # Inspect outputs
            print(f"   has pooler_output: {hasattr(outputs, 'pooler_output')}")
            
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                test_embedding = outputs.pooler_output
                print(f"         Using pooler_output")
                print(f"   pooler_output shape: {test_embedding.shape}")
                self.use_pooler = True
            else:
                raise ValueError("No pooler_output found - model may need trust_remote_code=True")
            
            hidden_size = test_embedding.shape[-1]
            print(f"   Embedding dimension: {hidden_size}")
            
            if hidden_size != 512:
                error_msg = f"Expected 512-dim, got {hidden_size}"
                if not allow_partial_model:
                    raise ValueError(error_msg)
                else:
                    print(f"         Warning: {error_msg}")
            
            self.model_load_warning = (hidden_size != 512)
            self.hidden_size = hidden_size
            self.output_dtype = str(test_embedding.dtype)
            
            print(f"\n         HeAR model loaded successfully on {device}")
            
        except Exception as e:
            print(f"\n  EXCEPTION CAUGHT: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            print("  Extraction failed")
            return None
        
        # ===== LOAD PD CLASSIFIER BUNDLE (with joblib) =====
        print(f"\n     Loading PD classifier bundle from: {pd_model_path}")
        try:
            self.pd_bundle = joblib.load(pd_model_path)
            print(f"         Loaded with joblib")
            print(f"   Bundle type: {type(self.pd_bundle)}")
            
            if isinstance(self.pd_bundle, dict):
                print(f"   Bundle keys: {list(self.pd_bundle.keys())}")
            
            # Extract scaler (fitted on 7D subspace)
            if 'scaler' in self.pd_bundle:
                self.scaler = self.pd_bundle['scaler']
                print(f"         Found 'scaler' in bundle (expects 7D input)")
                
                # Verify scaler expects 7 features
                if hasattr(self.scaler, 'n_features_in_'):
                    print(f"         Scaler expects {self.scaler.n_features_in_} features")
                    if self.scaler.n_features_in_ != 7:
                        print(f"         Warning: Scaler expects {self.scaler.n_features_in_} features, not 7")
            else:
                # Look for scaler-like object
                for key in ['scaler', 'normalizer', 'standardizer']:
                    if key in self.pd_bundle:
                        self.scaler = self.pd_bundle[key]
                        print(f"         Found '{key}' in bundle")
                        break
                else:
                    raise ValueError("No scaler found in PD bundle")
            
            # Verify indices match
            if 'indices' in self.pd_bundle:
                bundle_indices = self.pd_bundle['indices']
                print(f"   Bundle indices (1-based): {bundle_indices}")
                if bundle_indices != indices_1based:
                    print(f"         Warning: Bundle indices differ from provided indices")
            
            # Bundle hash
            with open(pd_model_path, 'rb') as f:
                self.pd_bundle_hash = hashlib.sha256(f.read()).hexdigest()[:16]
            
            # Provenance
            self.provenance = self.pd_bundle.get('provenance', None) if isinstance(self.pd_bundle, dict) else None
            self.provenance_missing = self.provenance is None
            
            if self.provenance_missing:
                print(f"         No provenance in PD bundle")
            else:
                print(f"         Scaler trained on: {self.provenance.get('training_cohorts', 'UNKNOWN')}")
            
        except Exception as e:
            print(f"     Failed to load PD bundle: {e}")
            sys.exit(1)
        
        # Verify indices
        print(f"\n using audited indices (0-based): {self.indices}")
        if 37 in self.indices:
            print(f"         Dim38 (index 37) included")
        
        print(f"\n     Extraction parameters:")
        print(f"   Window: {window_sec}s ({self.window_samples} samples)")
        print(f"   Hop: {hop_sec}s ({self.hop_samples} samples)")
        print(f"   Silence threshold: {silence_thresh_db} dBFS")
        print(f"   Spectrogram: {self.n_mels}×{self.target_width}")
        print(f"   Projection: 512D to 7D (indices), THEN scale")
        print(f"   Device: {device}")
        print(f"   Deterministic algorithms: ENABLED")
        print(f"   CPU threads: 1")
        print(f"   KCL 30s Mode: {kcl_30s_mode}")  # NEW
    
    def audio_to_spectrogram(self, audio):
        """Convert audio to mel spectrogram [1,1,192,128]"""
        audio_tensor = torch.from_numpy(audio).float()
        
        mel_spec = T.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0
        )(audio_tensor)
        
        mel_spec_db = T.AmplitudeToDB()(mel_spec)
        
        if mel_spec_db.shape[-1] > self.target_width:
            mel_spec_db = mel_spec_db[..., :self.target_width]
        elif mel_spec_db.shape[-1] < self.target_width:
            pad = self.target_width - mel_spec_db.shape[-1]
            mel_spec_db = torch.nn.functional.pad(mel_spec_db, (0, pad))
        
        return mel_spec_db.unsqueeze(0).unsqueeze(0)
    
    def extract_embedding_from_window(self, window):
        """Extract HeAR embedding using pooler_output"""
        spectrogram = self.audio_to_spectrogram(window)
        spectrogram = spectrogram.to(self.device)
        
        with torch.no_grad():
            outputs = self.hear_model(pixel_values=spectrogram)
        
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embedding = outputs.pooler_output.squeeze(0).cpu().numpy()
        else:
            raise ValueError("No pooler_output found")
        
        if embedding.shape != (512,):
            raise ValueError(f"Expected 512-dim, got {embedding.shape}")
        
        return embedding
    
    def extract_slices_from_file(self, wav_path):
        """
        Extract slices matching Step 1 exactly.
        Project to 7D using indices, THEN scale (scaler was fit on 7D subspace).
        """
        try:
            audio, _ = load_audio_mono(wav_path, target_sr=self.target_sr)
            
            # ===== CRITICAL DEBUG FOR ID00 AND ID14 =====
            subject = Path(wav_path).stem.split('_')[0]
            if subject in ['ID00', 'ID14']:
                print(f"\n     DEBUG - Processing {subject}")
                print(f"   Audio shape: {audio.shape}")
                print(f"   Audio dtype: {audio.dtype}")
                print(f"   Audio min/max: {audio.min():.3f}/{audio.max():.3f}")
                print(f"   Audio RMS: {20*np.log10(np.sqrt(np.mean(audio**2)) + 1e-12):.1f} dBFS")
                print(f"   Threshold: {self.silence_thresh_db} dBFS")
                print(f"   Window samples: {self.window_samples}")
                print(f"   Hop samples: {self.hop_samples}")
            # ===== END DEBUG =====
            
            audio = apply_pre_emphasis_librosa(audio, coeff=self.pre_emphasis_coeff)
            
            n_samples = len(audio)
            print(f"      DEBUG - File: {Path(wav_path).name}")
            print(f"      DEBUG - n_samples: {n_samples}")
            print(f"      DEBUG - window_samples: {self.window_samples}")
            print(f"      DEBUG - hop_samples: {self.hop_samples}")
            print(f"      DEBUG - Expected windows: {(n_samples - self.window_samples) // self.hop_samples + 1}")
            
            slices = []
            window_count = 0
            
            for start in range(0, n_samples - self.window_samples + 1, self.hop_samples):
                window_count += 1
                end = start + self.window_samples
                window = audio[start:end]
                
                rms_db = compute_window_rms_db(window)
                print(f"      DEBUG - Window {window_count}: start={start}, end={end}, RMS={rms_db:.1f} dBFS")
                
                if rms_db < self.silence_thresh_db:
                    print(f"      DEBUG - Window {window_count}: BELOW THRESHOLD (skipping)")
                    continue
                
                # Extract 512D embedding
                embedding_512 = self.extract_embedding_from_window(window)
                
                # Project to 7D using indices (0-based)
                embedding_7d_raw = embedding_512[self.indices]  # Shape: (7,)
                
                # Scale the 7D vector (scaler was fit on 7D subspace)
                embedding_7d_scaled = self.scaler.transform(embedding_7d_raw.reshape(1, -1)).flatten()
                
                slices.append({
                    'embedding_512_raw': embedding_512.copy(),
                    'embedding_7d_raw': embedding_7d_raw.copy(),
                    'embedding_7d_scaled': embedding_7d_scaled.copy(),
                    'slice_start_sample': start,
                    'slice_end_sample': end,
                    'slice_start_time': start / self.target_sr,
                    'slice_end_time': end / self.target_sr,
                    'slice_duration': self.window_sec,
                    'slice_rms_db': rms_db,
                    'slice_passed_gate': True
                })
            
            print(f"      DEBUG - Total slices extracted: {len(slices)}")
            return slices
            
        except Exception as e:
            print(f"     Error processing {Path(wav_path).name}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def test_similarity_with_step1(self, test_files, step1_ref_dir, n_files=3):
        """Cross-backend similarity check against Step-1 reference."""
        print("\n" + "="*70)
        print("     CROSS-BACKEND SIMILARITY CHECK WITH STEP-1")
        print("="*70)
        
        print("      NOTE: Step-1 may use different backend")
        print("   High similarity expected, exact equality not required\n")
        
        # Load Step-1 reference outputs
        ref_embeddings = np.load(Path(step1_ref_dir) / 'ref_embeddings_512_raw.npy')
        ref_metadata = pd.read_csv(Path(step1_ref_dir) / 'ref_metadata.csv')
        
        # Verify reference has required columns
        assert 'row_idx' in ref_metadata.columns, "Reference missing 'row_idx' column"
        assert 'slice_start_sample' in ref_metadata.columns, "Reference missing 'slice_start_sample'"
        assert 'filename' in ref_metadata.columns, "Reference missing 'filename'"
        
        print(f"Loaded Step-1 reference: {len(ref_metadata)} slices")
        
        results = []
        
        for wav_path in test_files[:n_files]:
            print(f"\nTesting: {Path(wav_path).name}")
            
            # Get filename for matching
            fname = Path(wav_path).name
            
            # Extract slices with this extractor
            slices = self.extract_slices_from_file(wav_path)
            
            if not slices:
                print(f"     No slices extracted")
                
                # ===== ADDED: Debug why no slices =====
                try:
                    audio, sr = load_audio_mono(wav_path, target_sr=self.target_sr)
                    print(f"   DEBUG - Audio length: {len(audio)} samples ({len(audio)/sr:.1f}s)")
                    print(f"   DEBUG - Required: {self.window_samples} samples ({self.window_sec}s)")
                    
                    if len(audio) < self.window_samples:
                        print(f"   DEBUG -   Audio too short")
                    else:
                        # Test first window
                        test_window = audio[:self.window_samples]
                        test_rms = compute_window_rms_db(test_window)
                        print(f"   DEBUG - First window RMS: {test_rms:.1f} dBFS")
                        print(f"   DEBUG - Threshold: {self.silence_thresh_db} dBFS")
                        
                        if test_rms < self.silence_thresh_db:
                            print(f"   DEBUG -   Below silence threshold")
                        else:
                            # Try embedding extraction
                            try:
                                test_emb = self.extract_embedding_from_window(test_window)
                                print(f"   DEBUG -       Embedding extraction succeeded")
                                print(f"   DEBUG -   But slices list is empty - check loop logic")
                            except Exception as e:
                                print(f"   DEBUG -   Embedding extraction failed: {e}")
                except Exception as e:
                    print(f"   DEBUG -   Error loading audio: {e}")
                # ===== END DEBUG =====
                
                continue
            
            # Get reference slices for this file
            ref_mask = ref_metadata['filename'] == fname
            ref_file = ref_metadata[ref_mask]
            
            print(f"   Our slices: {len(slices)}")
            print(f"   Step-1 slices: {len(ref_file)}")
            
            # Check slice count
            if len(slices) != len(ref_file):
                print(f"         Slice count mismatch!")
                results.append({
                    'file': fname,
                    'match': False,
                    'reason': f'slice_count: {len(slices)} vs {len(ref_file)}'
                })
                continue
            
            # Compare first slice numerically using sample index matching
            if len(slices) > 0 and len(ref_file) > 0:
                our_slice = slices[0]
                our_start_samp = our_slice['slice_start_sample']
                
                # Match by sample index (integer) - bulletproof
                ref_match = ref_file[ref_file['slice_start_sample'] == our_start_samp]
                
                if len(ref_match) == 0:
                    print(f"     No matching slice at sample {our_start_samp}")
                    results.append({
                        'file': fname,
                        'match': False,
                        'reason': f'sample_mismatch: {our_start_samp}'
                    })
                    continue
                
                # Get row index into embedding array
                ref_row_idx = int(ref_match.iloc[0]['row_idx'])
                ref_emb = ref_embeddings[ref_row_idx]
                
                # Our embedding
                our_emb = our_slice['embedding_512_raw']
                
                # Compute metrics
                max_diff = np.max(np.abs(our_emb - ref_emb))
                l2_diff = np.linalg.norm(our_emb - ref_emb)
                cos_sim = np.dot(our_emb, ref_emb) / (np.linalg.norm(our_emb) * np.linalg.norm(ref_emb))
                
                print(f"   Max absolute diff: {max_diff:.2e}")
                print(f"   L2 diff: {l2_diff:.2e}")
                print(f"   Cosine similarity: {cos_sim:.6f}")
                
                # Determine similarity quality
                if cos_sim > 0.999:
                    quality = "HIGH"
                elif cos_sim > 0.99:
                    quality = "GOOD"
                elif cos_sim > 0.95:
                    quality = "MODERATE"
                else:
                    quality = "POOR"
                
                print(f"   Similarity: {quality}")
                
                results.append({
                    'file': fname,
                    'cos_sim': cos_sim,
                    'max_diff': max_diff,
                    'quality': quality
                })
        
        # Summary
        print("\n" + "="*70)
        print("     SIMILARITY CHECK SUMMARY")
        print("="*70)
        
        for r in results:
            if 'quality' in r:
                print(f"   {Path(r['file']).name}: {r['quality']} (cos={r['cos_sim']:.6f})")
            else:
                print(f"   {Path(r['file']).name}:      {r.get('reason', 'unknown')}")
        
        return results
    
    def get_kcl_30s_files(self):
        """Get KCL Condition D 30s center WAV files."""
        hc_dir = Path(r"C:\Projects\hear_italian\data\KCL\HC\northwindpci\30scenter")
        pd_dir = Path(r"C:\Projects\hear_italian\data\KCL\PD\northwindpci\30scenter")
        
        hc_files = list(hc_dir.glob("*.wav")) if hc_dir.exists() else []
        pd_files = list(pd_dir.glob("*.wav")) if pd_dir.exists() else []
        
        all_files = hc_files + pd_files
        
        print(f"\n     KCL 30s Mode:")
        print(f"   HC 30s files: {len(hc_files)}")
        print(f"   PD 30s files: {len(pd_files)}")
        print(f"   Total: {len(all_files)}")
        
        return [str(f) for f in all_files]
    
    def extract_all(self, 
                    root_dirs=None,
                    output_name="cross_lingual_slices",
                    max_files=None,
                    step1_ref_dir=None):
        """Extract slice-level features with full provenance."""
        
        try:
            # ===== BUILD FILE METADATA =====
            file_metadata = None
            missing_files = []
            
            # ----- KCL 30s Mode: Use locked CSVs -----
            if self.kcl_30s_mode:
                # FORCE the output name to kcl_condition_d_30s
                output_name = "kcl_condition_d_30s"
                
                print(f"   Output name forced to: {output_name}")
                print("   Creating metadata from locked Condition D CSVs...")
                
                # Load the locked CSVs
                hc_csv = Path(r"C:\Projects\hear_italian\audit_results\HC_NorthWind_PD-Likeness_30sCenter_20260212_100148.csv")
                pd_csv = Path(r"C:\Projects\hear_italian\audit_results\PD_NorthWind_PD-Likeness_30sCenter_20260212_101240.csv")
                
                hc_meta = pd.read_csv(hc_csv)
                pd_meta = pd.read_csv(pd_csv)
                
                # Filter to SUCCESS only
                hc_meta = hc_meta[hc_meta['QC_Pass'] == True]
                pd_meta = pd_meta[pd_meta['QC_Pass'] == True]
                
                # Build file metadata for KCL
                file_metadata = []
                
                # Process HC files
                print("\n        Matching HC files...")
                hc_dir = Path(r"C:\Projects\hear_italian\data\KCL\HC\northwindpci\30scenter")
                all_hc_wavs = list(hc_dir.glob("*.wav"))
                print(f"   Found {len(all_hc_wavs)} total WAV files in HC directory")

                for _, row in hc_meta.iterrows():
                    subj_id = row['SubjectID']
                    
                    # METHOD 1: Direct pattern with wildcard
                    pattern = f"{subj_id}_hc_*_northwindpci_30s_center.wav"
                    possible_files = list(hc_dir.glob(pattern))
                    
                    # METHOD 2: If not found, search by subject ID anywhere in filename
                    if not possible_files:
                        possible_files = [f for f in all_hc_wavs if subj_id in f.name]
                    
                    if possible_files:
                        wav_path = possible_files[0]
                        print(f"           {subj_id} -> {wav_path.name}")
                        
                        file_metadata.append({
                            'full_path': str(wav_path),
                            'filename': wav_path.name,
                            'subject_id': subj_id,
                            'subject_key': f"KCL::{subj_id}",
                            'recording_key': f"KCL::{subj_id}_30s",
                            'cohort': 'KCL',
                            'disease': 'HC'
                        })
                    else:
                        print(f"           MISSING: {subj_id} (no file found in {hc_dir})")
                        missing_files.append(f"HC:{subj_id}")
                
                # Process PD files
                print("\n        Matching PD files...")
                pd_dir = Path(r"C:\Projects\hear_italian\data\KCL\PD\northwindpci\30scenter")
                
                for _, row in pd_meta.iterrows():
                    subj_id = row['SubjectID']
                    possible_files = list(pd_dir.glob(f"{subj_id}_pd_*_northwindpci_30s_center.wav"))
                    
                    if possible_files:
                        wav_path = possible_files[0]
                        file_metadata.append({
                            'full_path': str(wav_path),
                            'filename': wav_path.name,
                            'subject_id': subj_id,
                            'subject_key': f"KCL::{subj_id}",
                            'recording_key': f"KCL::{subj_id}_30s",
                            'cohort': 'KCL',
                            'disease': 'PD'
                        })
                        print(f"           {subj_id} -> {wav_path.name}")
                    else:
                        print(f"           MISSING: {subj_id} (no file found)")
                        missing_files.append(f"PD:{subj_id}")
                
                file_metadata = pd.DataFrame(file_metadata)
                
                print(f"\n        Summary:")
                print(f"      Expected HC: {len(hc_meta)}")
                print(f"      Found HC: {sum(file_metadata['disease'] == 'HC')}")
                print(f"      Expected PD: {len(pd_meta)}")
                print(f"      Found PD: {sum(file_metadata['disease'] == 'PD')}")
                
                if missing_files:
                    print(f"\n         Missing files: {missing_files}")
                    print(f"      These subjects will be skipped")
            
            # ----- Regular Mode: Use metadata parser -----
            else:
                if root_dirs is None:
                    root_dirs = [r'C:\Projects\hear_italian\data']
                
                # Collect all WAV files
                print("\n" + "="*70)
                print("     SCANNING FOR WAV FILES")
                print("="*70)
                
                all_wavs = []
                for root in root_dirs:
                    wavs = glob.glob(os.path.join(root, '**', '*.wav'), recursive=True)
                    all_wavs.extend(wavs)
                    print(f"   {root}: {len(wavs)} files")
                
                if max_files and max_files < len(all_wavs):
                    all_wavs = all_wavs[:max_files]
                    print(f"\n   Limiting to {max_files} files")
                
                print(f"\n   Total WAV files: {len(all_wavs)}")
                
                # Parse metadata using reviewer-safe parser
                print("\n" + "="*70)
                print("     PARSING FILE-LEVEL METADATA")
                print("="*70)
                
                metadata_parser_path = Path(__file__).parent / 'metadata_parser.py'
                if metadata_parser_path.exists():
                    sys.path.append(str(Path(__file__).parent))
                    from metadata_parser import ReviewerSafeParser
                    parser = ReviewerSafeParser(require_disease=True, parse_timestamps=False)
                    file_metadata = parser.build_dataframe(all_wavs, f"{output_name}_files")
                else:
                    # Create minimal metadata for smoke test
                    print("      metadata_parser.py not found - creating minimal metadata")
                    file_metadata = pd.DataFrame({
                        'full_path': all_wavs,
                        'filename': [Path(f).name for f in all_wavs],
                        'subject_id': [f"SUBJ_{i}" for i in range(len(all_wavs))],
                        'subject_key': [f"COHORT::SUBJ_{i}" for i in range(len(all_wavs))],
                        'recording_key': [f"COHORT::REC_{i}" for i in range(len(all_wavs))],
                        'cohort': ['SMOKE_TEST'] * len(all_wavs),
                        'disease': ['HC'] * len(all_wavs)
                    })
            
            # ===== PRE-FLIGHT VALIDATION =====
            print("\n     Pre-flight validation:")
            
            required_fields = [
                'subject_id', 'subject_key', 'recording_key', 'cohort', 'disease'
            ]
            
            for field in required_fields:
                assert field in file_metadata.columns, f"metadata missing {field}"
                assert file_metadata[field].notna().all(), f"{field} missing for some files"
                print(f"         {field}: present")
            
            print(f"\n     Valid files for extraction: {len(file_metadata)}")
            
            # ===== EXTRACT SLICES =====
            print("\n" + "="*70)
            print(" EXTRACTING SLICE-LEVEL EMBEDDINGS")
            print("="*70)
            print("   Pipeline: 512D to project to 7D (indices) then scale 7D")

            # Initialize counters
            files_processed = 0
            total_slices = 0

            all_slices = []
            slice_metadata = []

            for idx, row in tqdm(file_metadata.iterrows(), total=len(file_metadata), desc="Files"):
                wav_path = row['full_path']
                
                # Check if file exists
                if not os.path.exists(wav_path):
                    print(f"\n         File not found: {wav_path}")
                    continue
                
                # Extract slices
                slices = self.extract_slices_from_file(wav_path)
                
                # Get subject ID (handle both naming conventions)
                subject_id = row.get('subject_id', row.get('SubjectID', 'UNKNOWN'))
                
                if len(slices) == 0:
                    print(f"\n         CRITICAL: No slices extracted for {subject_id}!")
                    print(f"      File: {wav_path}")
                else:
                    print(f"         {subject_id}: {len(slices)} slices")
                
                files_processed += 1
                total_slices += len(slices)
                
                # Handle subjects with no slices
                if len(slices) == 0:
                    # Add a placeholder row to metadata only (no embeddings)
                    placeholder_row = row.to_dict()
                    placeholder_row.update({
                        'slice_index': -1,
                        'slice_start_sample': 0,
                        'slice_end_sample': 0,
                        'slice_start_time': 0,
                        'slice_end_time': 0,
                        'slice_duration': 0,
                        'slice_rms_db': -100,
                        'slice_passed_gate': False,
                        'slice_key': f"{row['recording_key']}_noslices",
                        'speaker_id': row['subject_id']
                    })
                    slice_metadata.append(placeholder_row)
                else:
                    # Record each slice with metadata and embeddings
                    for i, slice_data in enumerate(slices):
                        slice_row = row.to_dict()
                        
                        slice_row.update({
                            'slice_index': i,
                            'slice_start_sample': slice_data['slice_start_sample'],
                            'slice_end_sample': slice_data['slice_end_sample'],
                            'slice_start_time': slice_data['slice_start_time'],
                            'slice_end_time': slice_data['slice_end_time'],
                            'slice_duration': slice_data['slice_duration'],
                            'slice_rms_db': slice_data['slice_rms_db'],
                            'slice_passed_gate': slice_data['slice_passed_gate'],
                            'slice_key': f"{row['recording_key']}_slice{i:04d}",
                            'speaker_id': row['subject_id']
                        })
                        
                        all_slices.append({
                            'embedding_512_raw': slice_data['embedding_512_raw'],
                            'embedding_7d_raw': slice_data['embedding_7d_raw'],
                            'embedding_7d_scaled': slice_data['embedding_7d_scaled']
                        })
                        slice_metadata.append(slice_row)
                
                if (idx + 1) % 10 == 0:
                    print(f"   Progress: {idx+1}/{len(file_metadata)} files, {total_slices} slices")
            
            # ===== CONVERT TO ARRAYS =====
            n_slices = len(all_slices)
            if n_slices == 0:
                print("\n  No slices extracted!")
                return None
            
            embeddings_512_raw = np.array([s['embedding_512_raw'] for s in all_slices])
            embeddings_7d_raw = np.array([s['embedding_7d_raw'] for s in all_slices])
            embeddings_7d_scaled = np.array([s['embedding_7d_scaled'] for s in all_slices])
            metadata_df = pd.DataFrame(slice_metadata)
            
            print(f"\n     Extraction Results:")
            print(f"   • Files processed: {files_processed}/{len(file_metadata)}")
            print(f"   • Total slices: {n_slices}")
            print(f"   • Mean slices/file: {n_slices/files_processed:.1f}")
            print(f"   • Raw 512D shape: {embeddings_512_raw.shape}")
            print(f"   • Raw 7D shape: {embeddings_7d_raw.shape}")
            print(f"   • Scaled 7D shape: {embeddings_7d_scaled.shape}")
            
            # ===== SAVE OUTPUTS =====
            self._save_outputs(embeddings_512_raw, embeddings_7d_raw, embeddings_7d_scaled, 
                              metadata_df, output_name)
            
            self._generate_certification_report(embeddings_512_raw, embeddings_7d_raw, embeddings_7d_scaled, 
                                               metadata_df, output_name)
            
            return {
                'embeddings_512_raw': embeddings_512_raw,
                'embeddings_7d_raw': embeddings_7d_raw,
                'embeddings_7d_scaled': embeddings_7d_scaled,
                'metadata': metadata_df,
                'output_dir': self.output_root
            }
            
        except Exception as e:
            print(f"\n  CRITICAL ERROR in extract_all: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            print("\n  Extraction failed - see error above")
            return None

    def _save_outputs(self, emb_512_raw, emb_7d_raw, emb_7d_scaled, metadata, output_name):
        """Save all outputs with cryptographic hashes and pipeline config"""
        
        # Save embeddings
        np.save(self.output_root / f'{output_name}_embeddings_512_raw.npy', emb_512_raw)
        np.save(self.output_root / f'{output_name}_embeddings_7d_raw.npy', emb_7d_raw)
        np.save(self.output_root / f'{output_name}_embeddings_7d_scaled.npy', emb_7d_scaled)
        print(f"\n    Saved embeddings to: {self.output_root}")
        
        # Save metadata
        metadata.to_csv(self.output_root / f'{output_name}_metadata.csv', index=False)
        print(f"   Saved metadata: {output_name}_metadata.csv ({len(metadata)} slices)")
        
        # Save manifest
        manifest = metadata[['slice_key', 'recording_key', 'subject_key', 'speaker_id',
                            'cohort', 'disease', 'filename', 'slice_start_time', 'slice_start_sample']]
        manifest.to_csv(self.output_root / f'{output_name}_manifest.csv', index=False)
        
        # Compute hashes
        hash_data = {
            'embeddings_512_raw': hashlib.sha256(emb_512_raw.tobytes()).hexdigest(),
            'embeddings_7d_raw': hashlib.sha256(emb_7d_raw.tobytes()).hexdigest(),
            'embeddings_7d_scaled': hashlib.sha256(emb_7d_scaled.tobytes()).hexdigest(),
            'metadata': hashlib.sha256(metadata.to_csv(index=False).encode()).hexdigest(),
            'pipeline': {
                'window_sec': self.window_sec,
                'hop_sec': self.hop_sec,
                'window_samples': self.window_samples,
                'hop_samples': self.hop_samples,
                'silence_thresh_db': self.silence_thresh_db,
                'pre_emphasis_coeff': self.pre_emphasis_coeff,
                'target_sr': self.target_sr,
                'spectrogram_mels': self.n_mels,
                'spectrogram_width': self.target_width,
                'indices_1based': self.indices_1based,
                'indices_0based': self.indices,
                'hear_model_path': str(self.hear_model_path),
                'hear_model_type': 'HuggingFace ViT with pooler_output',
                'using_pooler': getattr(self, 'use_pooler', False),
                'hidden_size': getattr(self, 'hidden_size', 'unknown'),
                'pd_bundle_path': str(self.pd_model_path),
                'pd_bundle_hash': self.pd_bundle_hash,
                'scaler_features': getattr(self.scaler, 'n_features_in_', 'unknown'),
                'model_load_warning': getattr(self, 'model_load_warning', False),
                'allow_partial_model': self.allow_partial_model,
                'deterministic_algorithms': True,
                'cpu_threads': 1,
                'device': self.device,
                'kcl_30s_mode': self.kcl_30s_mode  # NEW
            },
            'created': datetime.now().isoformat()
        }
        
        with open(self.output_root / f'{output_name}_hashes.json', 'w') as f:
            json.dump(hash_data, f, indent=2)
    
    def _generate_certification_report(self, emb_512_raw, emb_7d_raw, emb_7d_scaled, metadata, output_name):
        """Generate comprehensive validation report"""
        
        report = []
        report.append("="*70)
        report.append("     SLICE-LEVEL EXTRACTION CERTIFICATION REPORT")
        report.append("="*70)
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Output: {output_name}")
        report.append("")
        
        # Pipeline parameters
        report.append("     EXTRACTION PARAMETERS:")
        report.append(f"   Window: {self.window_sec}s ({self.window_samples} samples)")
        report.append(f"   Hop: {self.hop_sec}s ({self.hop_samples} samples)")
        report.append(f"   Silence threshold: {self.silence_thresh_db} dBFS")
        report.append(f"   Pre-emphasis coeff: {self.pre_emphasis_coeff}")
        report.append(f"   Target SR: {self.target_sr} Hz")
        report.append(f"   Spectrogram: {self.n_mels}×{self.target_width}")
        report.append(f"   Pipeline: 512D to project to 7D (indices) then scale 7D")
        report.append(f"   Indices (1-based): {self.indices_1based}")
        report.append(f"   Indices (0-based): {self.indices}")
        report.append(f"   Device: {self.device}")
        report.append(f"   Deterministic algorithms: ENABLED")
        report.append(f"   CPU threads: 1")
        report.append(f"   KCL 30s Mode: {self.kcl_30s_mode}")
        report.append("")
        
        # Model provenance
        report.append("     MODEL PROVENANCE:")
        report.append(f"   HeAR model: {self.hear_model_path}")
        report.append(f"   Model type: HuggingFace ViT with pooler_output")
        report.append(f"   Using pooler_output: {getattr(self, 'use_pooler', False)}")
        report.append(f"   Hidden size: {getattr(self, 'hidden_size', 'unknown')}")
        
        if hasattr(self, 'model_load_warning') and self.model_load_warning:
            report.append("         CRITICAL: Model dimension mismatch")
            report.append("         Results may be INVALID for scientific use")
        else:
            report.append("         Model loaded with expected 512 dimensions")
        
        report.append(f"   Allow partial model: {self.allow_partial_model}")
        report.append(f"   PD bundle: {self.pd_model_path.name}")
        report.append(f"   PD bundle hash: {self.pd_bundle_hash}")
        report.append(f"   Scaler expects {getattr(self.scaler, 'n_features_in_', 'unknown')} features")
        
        if self.provenance_missing:
            report.append("         Scaler training data: UNKNOWN (frozen scaling - for deployment only)")
        else:
            report.append(f"   Scaler training cohorts: {self.provenance.get('training_cohorts', 'UNKNOWN')}")
        
        report.append("")
        
        # Data statistics
        report.append("     DATA STATISTICS:")
        report.append(f"   Total slices: {len(emb_512_raw)}")
        report.append(f"   Total files: {metadata['recording_key'].nunique()}")
        report.append(f"   Total subjects: {metadata['subject_key'].nunique()}")
        report.append("")
        
        # Cohort breakdown
        report.append("     COHORT BREAKDOWN:")
        for cohort in metadata['cohort'].unique():
            cohort_df = metadata[metadata['cohort'] == cohort]
            hc = len(cohort_df[cohort_df['disease'] == 'HC'])
            pd = len(cohort_df[cohort_df['disease'] == 'PD'])
            subjects = cohort_df['subject_key'].nunique()
            slices = len(cohort_df)
            
            report.append(f"   {cohort}:")
            report.append(f"      Slices: {slices} (HC:{hc} PD:{pd})")
            report.append(f"      Subjects: {subjects}")
            report.append(f"      Mean slices/subject: {slices/subjects:.1f}")
        
        report.append("")
        
        # Validation checks
        report.append("     VALIDATION CHECKS:")
        
        # No NaNs
        if np.isfinite(emb_512_raw).all() and np.isfinite(emb_7d_scaled).all():
            report.append("         No NaN/inf values in embeddings")
        
        # Shapes
        if emb_512_raw.shape[1] == 512:
            report.append("         Raw 512D correct dimension")
        if emb_7d_raw.shape[1] == 7:
            report.append("         Raw 7D correct dimension")
        if emb_7d_scaled.shape[1] == 7:
            report.append("         Scaled 7D correct dimension")
        
        # Alignment
        if len(emb_512_raw) == len(metadata):
            report.append("         Embeddings and metadata aligned")
        
        # Subject keys
        if all('::' in str(k) for k in metadata['subject_key']):
            report.append("         Subject keys have cohort prefix")
        
        # Disease labels
        if metadata['disease'].notna().all():
            report.append("         All slices have disease labels")
        
        # Speaker-ID safety
        if metadata.groupby('cohort')['speaker_id'].nunique().sum() == metadata['speaker_id'].nunique():
            report.append("         Speaker IDs are cohort-unique (safe for within-cohort tasks)")
        
        report.append("")
        report.append("="*70)
        report.append("     USAGE NOTES:")
        report.append("• For speaker-ID: split by recording_key, use within single cohort")
        report.append("• For PD classification: LOSO by subject_key, use embeddings_7d_scaled")
        report.append("• For identity vs disease: compare Δ accuracy (512D vs 7D)")
        report.append("• For raw 7D (unscaled): use embeddings_7d_raw")
        report.append("• For scaled 7D (deployment): use embeddings_7d_scaled")
        report.append("="*70)
        
        # Save report
        report_path = self.output_root / f'{output_name}_certification.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"\nCertification report saved to: {report_path}")


def main():
    try:
        parser = argparse.ArgumentParser(description='Certified slice-level extraction')
        
        parser.add_argument('--silence_thresh', type=int, default=-55,
                   help='Silence threshold in dBFS')
        
        parser.add_argument('--output', type=str, default='smoke_test',
                           help='Output name prefix')
        parser.add_argument('--hear_model', type=str, 
                           default=r'C:\Projects\hear_italian\models\hear\pytorch_model.bin',
                           help='Path to HeAR model')
        parser.add_argument('--pd_model', type=str,
                           default=r'C:\Projects\hear_italian\WAVstudy\pdhear_PURIFIED_V2_HASHED.pkl',
                           help='Path to PD bundle')
        parser.add_argument('--data_root', type=str, 
                           default=r'C:\Projects\hear_italian\data',
                           help='Root data directory')
        parser.add_argument('--max_files', type=int, default=None,
                           help='Max files to process')
        parser.add_argument('--step1_ref_dir', type=str, default=None,
                           help='Directory with Step-1 reference outputs')
        parser.add_argument('--device', type=str, default='cpu',
                           help='Device to use')
        parser.add_argument('--allow_partial_model', action='store_true',
                           help='Allow dimension mismatch')
        parser.add_argument('--kcl_30s', action='store_true',
                           help='Process KCL Condition D 30s center segments')
        
        args = parser.parse_args()
        
        output_dir_display = r'C:\Projects\hear_italian\features_certified'
        print("\n" + "="*70)
        print("     CERTIFIED SLICE-LEVEL EXTRACTOR - HuggingFace ViT")
        print("="*70)
        print(f"Output directory: {output_dir_display}")
        print(f"Device: {args.device}")
        print(f"Max files: {args.max_files}")
        print(f"Deterministic algorithms: ENABLED")
        print(f"Allow partial model: {args.allow_partial_model}")
        print(f"KCL 30s Mode: {args.kcl_30s}")
        
        if args.allow_partial_model:
            print("\n DANGER: Partial model loading enabled")
            print("   Results will be INVALID for scientific use")
            print("   Use ONLY for debugging\n")
        
        # Initialize extractor with new flag
        extractor = CertifiedSliceExtractor(
            hear_model_path=args.hear_model,
            pd_model_path=args.pd_model,
            output_root=r'C:\Projects\hear_italian\features_certified',
            indices_1based=[267, 346, 43, 146, 204, 38, 419],
            window_sec=2.0,
            hop_sec=1.0,
            silence_thresh_db=args.silence_thresh,
            device=args.device,
            allow_partial_model=args.allow_partial_model,
            kcl_30s_mode=args.kcl_30s
        )
        
        # Run extraction
        results = extractor.extract_all(
            root_dirs=[args.data_root],
            output_name=args.output,
            max_files=args.max_files,
            step1_ref_dir=args.step1_ref_dir
        )
        
        if results:
            print("\n" + "="*70)
            print(" EXTRACTION COMPLETE")
            print("="*70)
            print("\nOutput files in:", extractor.output_root)
            print(f"   {args.output if not args.kcl_30s else 'kcl_condition_d_30s'}_embeddings_512_raw.npy")
            print(f"   {args.output if not args.kcl_30s else 'kcl_condition_d_30s'}_embeddings_7d_raw.npy")
            print(f"   {args.output if not args.kcl_30s else 'kcl_condition_d_30s'}_embeddings_7d_scaled.npy")
            print(f"   {args.output if not args.kcl_30s else 'kcl_condition_d_30s'}_metadata.csv")
            print(f"   {args.output if not args.kcl_30s else 'kcl_condition_d_30s'}_manifest.csv")
            print(f"   {args.output if not args.kcl_30s else 'kcl_condition_d_30s'}_hashes.json")
            print(f"   {args.output if not args.kcl_30s else 'kcl_condition_d_30s'}_certification.txt")
        else:
            print("\n Extraction failed")
            
    except Exception as e:
        print(f"\n CRITICAL ERROR in main: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("\n Extraction failed - see error above")
        sys.exit(1)


if __name__ == "__main__":
    main()