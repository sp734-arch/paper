# Step 7: Short-File Stability Audit - Subject-Level Signal Consistency

## PAPER CONNECTION & SCIENTIFIC CONTEXT
This script implements the **SUBJECT-LEVEL STABILITY** requirement from the paper's auditing framework. It tests whether candidate physiological dimensions remain consistent within individuals across short audio segments.

### KEY PAPER REFERENCES
1. **SECTION 4 (Subject-Level Stability)**: "Are relevant dimensions stable within individuals?" This script provides the quantitative test for that requirement.

2. **SECTION 5 (Auditing Protocol)**: Operationalizes stability testing with the variance ratio metric (between-subject / within-subject variance).

3. **SECTION 9 (Stability-First Framework)**: Demonstrates the core principle that physiological signals must be repeatable within individuals before being used for diagnosis.

4. **ABSTRACT (Stable Physiological Signal)**: Supports the claim that stable vocal structure can be isolated from foundation embeddings.

## SCIENTIFIC PURPOSE
Tests the fundamental property of valid physiological measurements: **REPEATABILITY**. A true physiological trait should:

1. **Vary MORE** between different people (high inter-subject variance)
2. **Vary LESS** within the same person across time (low intra-subject variance)

This addresses the question: *"Is the signal a stable trait of the individual, or just random fluctuation that happens to correlate with labels?"*

## THE STABILITY RATIO METRIC (Between/Within Variance)
**Formula**: Ratio = Variance(Between Subjects) / Variance(Within Subjects)

**Interpretation**:
- **Ratio < 1**: Signal is **NOISY** - varies more within subjects than between
- **Ratio ~1-3**: **Weak trait** - some consistency but substantial noise
- **Ratio > 3-5**: **Moderate trait** - reasonably stable
- **Ratio > 8-10**: **STRONG TRAIT** - highly stable biological fingerprint
- **Ratio > 15+**: Possibly overfitted or capturing invariant confounds

The code comment insight: **"9.80 is a High-Resolution Biological Fingerprint"** indicates exceptionally stable signal. In psychometrics/psychophysiology, ratios > 5 are considered excellent for trait measurement.

## BIOLOGICAL SIGNIFICANCE OF HIGH STABILITY
A high stability ratio suggests the signal is:

1. **"STRUCTURALLY ANCHORED"** (code comment): Baked into the physiological hardware rather than dependent on specific content
2. **RESISTANT TO CONTEXT**: Doesn't drift with specific words, emotions, or speaking tasks
3. **INTRINSIC TO INDIVIDUAL**: Reflects stable biological properties rather than transient states

This supports the paper's distinction between:
- **STABLE PHYSIOLOGY**: Hardware-level vocal apparatus properties
- **TRANSIENT ARTIFACTS**: Software-level linguistic/content effects

## METHODOLOGICAL INNOVATION - "SHORT-FILE MODE"
Uses **50% overlap embeddings** (1-second hop) rather than full recordings. This is a **STRICTER test** because:

1. Short segments (2 seconds) have less signal averaging
2. Higher risk of capturing transient artifacts
3. If signal remains stable across short segments, it's robust evidence of true physiological trait

**Contrast** with "long-file" stability which could simply reflect consistent recording conditions or speaking style across a longer session.

## DATA SOURCE RATIONALE
Uses **50% overlap embeddings** because:
1. Balanced temporal resolution (not too sparse, not too redundant)
2. Provides multiple independent segments per subject for variance estimation
3. Consistent with production setting in paper's extraction pipeline

The **top 10 dimensions** analyzed are the same candidates from previous steps, ensuring consistency across the auditing pipeline.

## PURPOSE
Calculate the stability ratio (between-subject / within-subject variance) for the top 10 candidate physiological dimensions to determine whether they:

1. **Represent stable individual traits** (high ratio → desirable)
2. **Are too noisy for reliable measurement** (low ratio → problematic)
3. **Fall in the biologically plausible range** of "loose consistency"

This test is **CRITICAL** for the paper's claim that foundation models can extract stable physiological signals suitable for clinical measurement.

## OUTPUT INTERPRETATION FOR PAPER
The stability ratio should be reported in the paper as:
- Quantitative evidence supporting subject-level stability (Section 4)
- Demonstration that signals are traits rather than states
- Validation of the "structurally anchored" physiological baseline claim

**Threshold interpretation** (from code):
- `ratio > 1.5`: "Trait" signal - subjects remain unique
- `ratio ≤ 1.5`: "Noise" - too jittery for stable baseline

## RELATION TO OTHER AUDITS
- **PREREQUISITE FOR**: Disease screening validity - unstable signals cannot support reliable diagnosis
- **COMPLEMENTS**: Invariance audit - a signal can be stable but confounded (e.g., stable recording artifact)
- **FEEDS INTO**: Final dimension selection - only stable dimensions should be retained for the Purified V2 manifold
- **CONTRASTS WITH**: Language discrimination analysis - stability tests within-subject consistency regardless of content

## USAGE
1. **Requires**: 50% overlap embeddings and pre-identified top 10 dimensions
2. **Run**: `python stability_audit.py`
3. **Output**: Stability ratio and interpretation
4. **Decision**: Retain dimensions with ratio > threshold (paper uses ICC > 0.5, which corresponds to ratio > 1.0 in this formulation)

---

**AUTHORS**: Jim McCormack  
**CREATED**: Feb 2026  
**PAPER REFERENCE**: Sections 4 (Subject-Level Stability), 5 (Auditing Protocol), Abstract (stable vocal structure)
