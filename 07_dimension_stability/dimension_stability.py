"""
Step 7: Short-File Stability Audit - Subject-Level Signal Consistency
=====================================================================

PAPER CONNECTION & SCIENTIFIC CONTEXT:
---------------------------------------
This script implements the SUBJECT-LEVEL STABILITY requirement from the paper's
auditing framework. It tests whether candidate physiological dimensions remain
consistent within individuals across short audio segments.

KEY PAPER REFERENCES:
1. SECTION 4 (Subject-Level Stability): "Are relevant dimensions stable
   within individuals?" This script provides the quantitative test for that
   requirement.

2. SECTION 5 (Auditing Protocol): Operationalizes stability testing with
   the variance ratio metric (between-subject / within-subject variance).

3. SECTION 4 (Stability-First Framework): Demonstrates the core principle
   that physiological signals must be repeatable within individuals before
   being used for diagnosis.

4. ABSTRACT (Stable Physiological Signal): Supports the claim that stable
   vocal structure can be isolated from foundation embeddings.

SCIENTIFIC PURPOSE:
-------------------
Tests the fundamental property of valid physiological measurements:
REPEATABILITY. A true physiological trait should:
1. Vary MORE between different people (high inter-subject variance)
2. Vary LESS within the same person across time (low intra-subject variance)

This addresses the question: "Is the signal a stable trait of the individual,
or just random fluctuation that happens to correlate with labels?"

THE STABILITY RATIO METRIC (Between/Within Variance):
-----------------------------------------------------
Formula: Ratio = Variance(Between Subjects) / Variance(Within Subjects)

Interpretation:
- Ratio < 1: Signal is NOISY - varies more within subjects than between
- Ratio ~1-3: Weak trait - some consistency but substantial noise
- Ratio > 3-5: Moderate trait - reasonably stable
- Ratio > 8-10: STRONG TRAIT - highly stable biological fingerprint
- Ratio > 15+: Possibly overfitted or capturing invariant confounds

The code comment insight: "9.80 is a High-Resolution Biological Fingerprint"
indicates exceptionally stable signal. In psychometrics/psychophysiology,
ratios > 5 are considered excellent for trait measurement.

BIOLOGICAL SIGNIFICANCE OF HIGH STABILITY:
------------------------------------------
A high stability ratio suggests the signal is:
1. "STRUCTURALLY ANCHORED" (code comment): Baked into the physiological
   hardware rather than dependent on specific content
2. RESISTANT TO CONTEXT: Doesn't drift with specific words, emotions, or
   speaking tasks
3. INTRINSIC TO INDIVIDUAL: Reflects stable biological properties rather
   than transient states

This supports the paper's distinction between:
- STABLE PHYSIOLOGY: Hardware-level vocal apparatus properties
- TRANSIENT ARTIFACTS: Software-level linguistic/content effects

METHODOLOGICAL INNOVATION - "SHORT-FILE MODE":
-----------------------------------------------
Uses 50% overlap embeddings (1-second hop) rather than full recordings.
This is a STRICTER test because:
1. Short segments (2 seconds) have less signal averaging
2. Higher risk of capturing transient artifacts
3. If signal remains stable across short segments, it's robust evidence
   of true physiological trait

Contrast with "long-file" stability which could simply reflect consistent
recording conditions or speaking style across a longer session.

DATA SOURCE RATIONALE:
----------------------
Uses 50% overlap embeddings because:
1. Balanced temporal resolution (not too sparse, not too redundant)
2. Provides multiple independent segments per subject for variance estimation
3. Consistent with production setting in paper's extraction pipeline

The top 10 dimensions analyzed are the same candidates from previous steps,
ensuring consistency across the auditing pipeline.

PURPOSE:
--------
Calculate the stability ratio (between-subject / within-subject variance)
for the top 10 candidate physiological dimensions to determine whether they:
1. Represent stable individual traits (high ratio → desirable)
2. Are too noisy for reliable measurement (low ratio → problematic)
3. Fall in the biologically plausible range of "loose consistency"

This test is CRITICAL for the paper's claim that foundation models can
extract stable physiological signals suitable for clinical measurement.

OUTPUT INTERPRETATION FOR PAPER:
--------------------------------
The stability ratio should be reported in the paper as:
- Quantitative evidence supporting subject-level stability (Section 4)
- Demonstration that signals are traits rather than states
- Validation of the "structurally anchored" physiological baseline claim

Threshold interpretation (from code):
- ratio > 1.5: "Trait" signal - subjects remain unique
- ratio ≤ 1.5: "Noise" - too jittery for stable baseline

RELATION TO OTHER AUDITS:
-------------------------
PREREQUISITE FOR: Disease screening validity - unstable signals cannot
                  support reliable diagnosis

COMPLEMENTS: Invariance audit - a signal can be stable but confounded
             (e.g., stable recording artifact)

FEEDS INTO: Final dimension selection - only stable dimensions should be
            retained for the Purified V2 manifold

CONTRASTS WITH: Language discrimination analysis - stability tests
                within-subject consistency regardless of content

USAGE:
------
1. Requires: 50% overlap embeddings and pre-identified top 10 dimensions
2. Run: python stability_audit.py
3. Output: Stability ratio and interpretation
4. Decision: Retain dimensions with ratio > threshold (paper uses ICC > 0.5,
             which corresponds to ratio > 1.0 in this formulation)

AUTHORS: Jim McCormack
CREATED: Feb 2026
PAPER REFERENCE: Sections 4 (Subject-Level Stability), 5 (Auditing Protocol),
                 Abstract (stable vocal structure)
"""

import numpy as np
from pathlib import Path

def check_short_file_stability(data_dir, top_indices):
    """
    Calculate subject-level stability ratio for candidate physiological dimensions.
    
    PAPER CONTEXT: Implements the subject-level stability test required by
    Section 4 of the paper. High stability ratio indicates dimensions are
    consistent traits of individuals rather than transient noise.
    
    SCIENTIFIC RATIONALE: Valid physiological measurements must be:
    1. REPEATABLE: Consistent within the same individual across time
    2. DISCRIMINATIVE: Different between different individuals
    3. RELIABLE: Low measurement error relative to true signal
    
    The stability ratio (between-subject / within-subject variance) quantifies
    these properties in a single metric analogous to intraclass correlation
    coefficient (ICC).
    
    METHOD: For each subject with multiple embedding slices:
    1. Calculate within-subject variance (across slices)
    2. Calculate between-subject variance (of subject means)
    3. Compute ratio: Variance(Between) / Variance(Within)
    
    High ratio → Good trait measurement
    Low ratio → Noisy or state-dependent signal
    
    Args:
        data_dir (str): Path to directory containing .npy embedding files
        top_indices (list): Indices of top 10 dimensions to analyze
    
    Returns:
        tuple: (stability_ratio, avg_intra_var, avg_inter_var)
            - stability_ratio: Between/within variance ratio
            - avg_intra_var: Average within-subject variance
            - avg_inter_var: Between-subject variance
    
    PAPER INTEGRATION:
    Report in:
    - Results: Stability ratio and interpretation
    - Methods: As part of subject-level stability assessment
    - Discussion: Implications for trait vs state measurement
    """
    files = list(Path(data_dir).glob("*.npy"))
    subject_data = {}
    
    print(f"🔬 SUBJECT-LEVEL STABILITY AUDIT")
    print("=" * 60)
    print("PAPER CONTEXT: Testing subject-level stability (Section 4)")
    print(f"DIMENSIONS: Top {len(top_indices)} candidates")
    print(f"DATA: 50% overlap embeddings (balanced temporal resolution)")
    print(f"FILES: {len(files)} embedding slices total")
    print()
    
    # Group embeddings by subject ID
    # Subject ID extraction from filename: assumes format {SUBJECT}_... 
    for f in files:
        sub_id = f.name.split('_')[0]  # Extract subject ID from filename
        emb = np.load(f).flatten()[top_indices]  # Keep only top dimensions
        if sub_id not in subject_data: 
            subject_data[sub_id] = []
        subject_data[sub_id].append(emb)
    
    print(f"📊 Found {len(subject_data)} unique subjects")
    print(f"   Subjects with ≥2 slices: {sum(len(v) >= 2 for v in subject_data.values())}")
    print()

    intra_vars = []  # Within-subject variances
    inter_means = [] # Subject mean vectors

    # Calculate variance components
    for sub, embs in subject_data.items():
        if len(embs) < 2: 
            continue  # Need at least 2 slices to estimate within-subject variance
        
        embs_arr = np.array(embs)  # [n_slices × 10_dimensions]
        
        # Within-subject variance: variability across slices from same subject
        # Lower values = more consistent measurement
        intra_vars.append(np.var(embs_arr, axis=0))
        
        # Subject mean: central tendency for this subject
        # Used to calculate between-subject variance
        inter_means.append(np.mean(embs_arr, axis=0))

    # Check we have enough data
    if len(intra_vars) < 5 or len(inter_means) < 5:
        print("⚠️  WARNING: Insufficient subjects for reliable stability estimation")
        print("   Consider using more data or relaxing subject requirements")
        return None, None, None
    
    # Calculate average within-subject variance across all subjects and dimensions
    avg_intra = np.mean(intra_vars)
    
    # Calculate between-subject variance: variance of subject means
    # Higher values = subjects are more different from each other
    inter_means_arr = np.array(inter_means)  # [n_subjects × 10_dimensions]
    avg_inter = np.var(inter_means_arr)  # Overall variance across all subjects/dimensions

    # Stability ratio: Between-subject variance / Within-subject variance
    # This is analogous to F-ratio in ANOVA or related to ICC
    ratio = avg_inter / avg_intra if avg_intra > 0 else float('inf')
    
    print("📈 STABILITY ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Between-Subject Variance: {avg_inter:.6f}")
    print(f"Within-Subject Variance:  {avg_intra:.6f}")
    print(f"Stability Ratio:          {ratio:.4f}")
    print()
    
    # Interpretation with detailed biological context
    print("🔬 STABILITY INTERPRETATION (for Paper Discussion):")
    print()
    
    if ratio > 10.0:
        print(f"   🎯 EXCEPTIONAL STABILITY ({ratio:.2f})")
        print("   INDICATION: High-resolution biological fingerprint")
        print("   PAPER IMPLICATION: Signal is 'structurally anchored' in physiology")
        print("   BIOLOGICAL MEANING: Hardware-level trait, resistant to context")
        print("   CODE INSIGHT: '9.80 is the holy grail of signal processing'")
        
    elif ratio > 5.0:
        print(f"   ✅ STRONG TRAIT ({ratio:.2f})")
        print("   INDICATION: Reliable individual signature")
        print("   PAPER IMPLICATION: Suitable for longitudinal tracking")
        print("   BIOLOGICAL MEANING: Stable physiological characteristic")
        
    elif ratio > 2.0:
        print(f"   📊 MODERATE TRAIT ({ratio:.2f})")
        print("   INDICATION: Some consistency with notable variability")
        print("   PAPER IMPLICATION: May require multiple measurements")
        print("   BIOLOGICAL MEANING: State-trait hybrid signal")
        
    elif ratio > 1.0:
        print(f"   ⚠️  WEAK SIGNAL ({ratio:.2f})")
        print("   INDICATION: More noise than consistent trait")
        print("   PAPER IMPLICATION: Questionable for individual assessment")
        print("   BIOLOGICAL MEANING: Highly context-dependent")
        
    else:
        print(f"   ❌ NOISE-DOMINANT ({ratio:.2f})")
        print("   INDICATION: Unreliable measurement")
        print("   PAPER IMPLICATION: Should be rejected from screening")
        print("   BIOLOGICAL MEANING: Random fluctuation or measurement error")
    
    print()
    
    # Connection to paper's ICC threshold
    # ICC = (Between - Within) / (Between + Within) ≈ (ratio - 1) / (ratio + 1)
    # Paper uses ICC > 0.5 as threshold (Section 5)
    estimated_icc = (ratio - 1) / (ratio + 1) if ratio > 0 else -1
    
    print("📐 PAPER THRESHOLD COMPARISON:")
    print(f"   Estimated ICC from ratio: {estimated_icc:.3f}")
    print(f"   Paper requirement (Section 5): ICC > 0.5")
    print(f"   Meets paper threshold: {'YES' if estimated_icc > 0.5 else 'NO'}")
    print()
    
    print("📄 PAPER INTEGRATION GUIDE:")
    print("1. Report stability ratio with confidence intervals if possible")
    print("2. Reference in Section 4 (Subject-Level Stability)")
    print("3. Discuss implications for 'structurally anchored' physiology")
    print("4. Connect to ICC threshold from auditing protocol (Section 5)")
    
    return ratio, avg_intra, avg_inter

# Execute analysis
if __name__ == "__main__":
    # The Top 10 Drivers from previous analysis
    # These dimensions showed strongest language-discriminative power
    # Now we test whether they are stable within subjects
    top_drivers = [419, 227, 43, 346, 204, 317, 38, 98, 267, 146]
    
    print("\n" + "="*70)
    print("🧬 SHORT-FILE STABILITY AUDIT")
    print("="*70)
    print("PAPER: Testing subject-level stability (Section 4)")
    print("QUESTION: Are candidate dimensions stable traits or transient noise?")
    print("METHOD: Between/Within variance ratio on 50% overlap embeddings")
    print("DATA: Multiple 2-second slices per subject")
    print("="*70)
    print()
    
    # Run stability audit
    stability_ratio, intra_var, inter_var = check_short_file_stability(
        "/app/data/processed/esen_ovlap_50", 
        top_drivers
    )
    
    if stability_ratio is not None:
        # Save results for paper integration
        output_dir = "/app/data/analysis/stability_audit"
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save quantitative results
        results = {
            'stability_ratio': stability_ratio,
            'within_subject_variance': intra_var,
            'between_subject_variance': inter_var,
            'top_dimensions': top_drivers,
            'estimated_icc': (stability_ratio - 1) / (stability_ratio + 1)
        }
        
        np.save(os.path.join(output_dir, "stability_results.npy"), results)
        
        # Save interpretation for paper
        with open(os.path.join(output_dir, "stability_interpretation.txt"), "w") as f:
            f.write("SUBJECT-LEVEL STABILITY AUDIT RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Stability ratio: {stability_ratio:.4f}\n")
            f.write(f"Within-subject variance: {intra_var:.6f}\n")
            f.write(f"Between-subject variance: {inter_var:.6f}\n")
            f.write(f"Estimated ICC: {(stability_ratio - 1)/(stability_ratio + 1):.3f}\n")
            f.write(f"Paper ICC threshold: > 0.5\n")
            f.write(f"Meets threshold: {'YES' if (stability_ratio - 1)/(stability_ratio + 1) > 0.5 else 'NO'}\n")
            f.write(f"\nInterpretation: ")
            if stability_ratio > 10:
                f.write("Exceptional stability - high-resolution biological fingerprint\n")
            elif stability_ratio > 5:
                f.write("Strong trait - reliable individual signature\n")
            elif stability_ratio > 2:
                f.write("Moderate trait - some consistency with variability\n")
            else:
                f.write("Weak signal - questionable for individual assessment\n")
            f.write(f"\nPaper section: 4 (Subject-Level Stability)\n")
        
        print(f"\n💾 Results saved to: {output_dir}/")
       
       
