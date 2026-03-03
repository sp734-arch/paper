# Step 2: Master Overlap Sweep - Physiological Unit Extraction

## PURPOSE
Extracts HeAR embeddings at three overlap densities (10%, 50%, 75%) for emotional speech data (Spanish/English)

Figure 1 ("Stability-first auditing of foundation audio representations") requires demonstrating that different temporal sampling strategies yield consistent physiological units when the underlying signal is stable. This script implements the "Master Sweep" concept: systematically varying window overlap to test whether physiological structure persists across sampling densities.

## PAPER REFERENCE
- **Figure 2**: Identifying a physiological subspace

## RELATION TO 01_extract_hear_staircase.py
- **COMPLEMENTARY DESIGN**: 01 tests stability across temporal densities for PD speech
- **THIS SCRIPT**: Tests stability across temporal densities for emotional speech (ES/EN)
- **SHARED CORE HYPOTHESIS**: If a dimension encodes stable physiology, it should remain consistent whether we sample densely (75% overlap) or sparsely (10% overlap)
- **DIFFERENT CONTEXT**: This uses emotional speech data to test generalizability of the stability principle beyond pathological speech

## KEY SCIENTIFIC DECISIONS
1. **Three overlap densities mirror staircase design**:
   - **75% overlap**: "Deep Mining" - captures subtle transitions, respiratory tremors
   - **50% overlap**: Balanced production setting
   - **10% overlap**: "Clean" set - ensures unique 2-second clips for pure LOSO validation
2. **Language-aware naming (ES/EN)**: Enables testing invariance to language confounds
3. **Voice Activity Detection (VAD)**: Removes non-speech segments using amplitude threshold (`top_db=25`) rather than RMS-based silence gate
4. **Subject-aware extraction**: Maintains subject identity for later LOSO analysis

## OUTPUT STRUCTURE
**File naming**: `{SUBJECT_ID}_{LANGUAGE}_{SEQUENCE_ID:05d}.npy`  
**Example**: `S001_EN_00042.npy`

This naming convention enables:
1. Subject-level grouping for LOSO validation
2. Language-based analysis for invariance testing
3. Sequence tracking for temporal analysis

## USAGE
1. Ensure HeAR model is available at `/app/models`
2. Place raw emotional speech files in `/app/data/raw/`
   - Spanish files should contain "ses-es" in path
   - English files should contain "ses-ed" in path
3. Run: `python mastersweepoverlapping_STEP_2.py`
4. Output will be used for Figure 1 generation in analysis pipeline

**AUTHORS**: Jim McCormack
**CREATED**: 0207/2026  
