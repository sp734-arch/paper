# Step 1: Multi-Density Embedding Extraction

## Purpose

This script extracts HeAR embeddings at three temporal sampling densities to test representation stability under different windowing strategies.  It us used to prepare the data sets for analysis and use by the model pipeline.  Multiple source data sets are required to validate task control for minimum validity standard 8.1 

## The "Staircase" Design

| Density | Hop Size | Overlap | Purpose |
|---------|----------|---------|---------|
| **10%** | 28,800 samples (1.8s) | 10% | Sparse baseline |
| **50%** | 16,000 samples (1.0s) | 50% | Production setting (used in paper) |
| **75%** | 8,000 samples (0.5s) | 75% | Dense robustness test |

**Scientific Rationale:** If a dimension encodes stable physiology, it should remain consistent across all three densities. If it only appears under dense sampling, it may be an artifact.

## Key Processing Steps

1. **Load audio** at 16 kHz mono
2. **Apply pre-emphasis** (enhances high frequencies / formants)
3. **Sliding window** at three overlap densities
4. **Silence gate** at -50 dBFS (removes non-speech segments)
5. **HeAR inference** → 512-D embedding per 2-second window
6. **Save** one .npy file per segment with subject-aware naming

## Quality Control

**Silence Threshold:** -50 dBFS (absolute RMS)
- This is **NOT** speaker-normalized
- Segments below this threshold are excluded
- This prevents empty/silence embeddings without introducing per-speaker bias

**Why absolute, not relative?**
- Relative thresholds (e.g., "bottom 10% per file") would normalize out genuine loudness differences between PD and HC
- Absolute threshold preserves physiological signal while removing technical artifacts

## Output Structure
```
features_audit/
├── density_10/
│   ├── features_kcl/
│   │   ├── healthy/
│   │   │   └── ID01_...._s0.npy
│   │   └── parkinsons/
│   ├── features_italian/
│   └── features_english/
├── density_50/  (same structure)
└── density_75/  (same structure)
```

## Subject ID Extraction

Each cohort has different filename conventions:

- **KCL:** `ID02_pd_1_2_1.wav` → Subject ID = `ID02`
- **English:** `AH_123_M65_vowel_a.wav` → Subject ID = `AH_123_M65`
- **Italian:** Folder-based → Subject ID = parent directory name

This enables Leave-One-Subject-Out (LOSO) cross-validation in later steps.

## Runtime

- **CPU only:** ~12-16 hours (not recommended)
- **GPU (NVIDIA T4):** ~2-3 hours
- **GPU (A100/H100):** ~30-45 minutes

## Dependencies
```bash
tensorflow>=2.10.0
librosa>=0.10.0
numpy>=1.21.0
tqdm>=4.65.0
```
