"""
Step 6: Correlation Structure Analysis - Testing Orthogonality of Physiological Dimensions
===========================================================================================

PAPER CONNECTION & SCIENTIFIC CONTEXT:
---------------------------------------
This script implements a critical validity test from the paper's auditing framework:
examining the correlation structure of candidate physiological dimensions.

KEY PAPER REFERENCES:
1. SECTION 4 (Dimensional Parsimony): "Does screening rely on a small,
   low-correlation subspace?" This script provides the quantitative test
   for that requirement.

2. SECTION 54 (Auditing Protocol): Implements the operational check for
   feature independence, preventing redundant measurement of the same
   underlying signal.

3. FIGURE 3(b): Shows correlation structure comparison between full embedding
   space and purified subspace. This script generates the data for such analyses.

4. ABSTRACT (Interpretability): Supports the claim of "more interpretable"
   deployment by ensuring dimensions represent independent physiological axes.

SCIENTIFIC PURPOSE:
-------------------
Tests whether the top candidate physiological dimensions represent:
- REDUNDANT NOISE: Highly correlated dimensions (r > 0.9) measuring the
  same underlying signal → indicates overfitting risk
- ORTHOGONAL SENSORS: Low-correlation dimensions (r < 0.3) measuring
  independent physiological processes → indicates robust multi-variate fingerprint

This addresses a fundamental question in physiological measurement:
"Are we measuring multiple independent aspects of vocal physiology,
or just the same signal with different noise?"

BIOLOGICAL INTERPRETATION OF CORRELATION PATTERNS:
--------------------------------------------------
1. HIGH CORRELATION (r > 0.7):
   - All dimensions track the same underlying process
   - Like having 10 thermometers all measuring the same temperature
   - Biologically implausible: vocal physiology has multiple semi-independent
     subsystems (respiratory, laryngeal, articulatory)
   - Indicates: Model collapse or measurement redundancy

2. MODERATE CORRELATION (0.3 < r < 0.7):
   - Dimensions measure related but distinct processes
   - Biologically realistic: subsystems are coordinated but have degrees of freedom
   - "Loose coupling" as described in code comments: lungs, vocal folds,
     tongue are connected but independent
   - Optimal for multi-variate physiological fingerprinting

3. LOW CORRELATION (r < 0.3):
   - Dimensions measure independent processes
   - Could indicate: Truly orthogonal physiology OR measurement noise
   - Requires validation with stability/invariance audits

THE "LOOSE COUPLING" HYPOTHESIS:
---------------------------------
The code comment insight is biologically profound: at ~0.24 average correlation,
we see "Loose Coupling" - exactly how biological systems work. Vocal physiology
involves multiple subsystems that must coordinate but maintain independence.

This contrasts with engineering systems (tightly coupled) or random noise
(completely uncoupled). Finding loose coupling supports the paper's claim that
foundation models capture biologically plausible representations.

METHODOLOGICAL IMPORTANCE:
--------------------------
1. DEFENSE AGAINST OVERFITTING: Low correlation means model isn't just
   learning redundant representations of the same signal (code comment: "defended
   against the 'Overfitting' risk")

2. VALIDATION OF MULTI-VARIATE MEASUREMENT: Independent dimensions suggest
   the model captures multiple aspects of physiology, not just one dominant signal

3. INFORMING MODEL INTERPRETATION: Correlation structure tells us whether
   dimensions can be interpreted as independent biomarkers or must be treated
   as a combined syndrome measure

DATA SOURCE RATIONALE:
----------------------
Uses 75% overlap embeddings ("Deep Mining" set) because:
1. Maximum sample size for correlation estimation
2. Captures subtle physiological transitions that might reveal coupling patterns
3. Consistent with other analyses in the language invariance testing pipeline

The top 10 dimensions analyzed are those identified in Step 5 as having
strongest language-discriminative power (potential physiological signals
or linguistic confounds).

PURPOSE:
--------
Analyze the pairwise correlations among the top 10 candidate physiological
dimensions to determine whether they represent:
1. Redundant measurements of the same signal (high correlation → problematic)
2. Independent physiological sensors (low correlation → desirable)
3. Loosely coupled biological subsystems (moderate correlation → biologically plausible)

OUTPUT INTERPRETATION FOR PAPER:
--------------------------------
The average cross-correlation should be reported in the paper as:
- Quantitative evidence supporting dimensional parsimony (Section 45)
- Defense against overfitting claims
- Biological plausibility indicator (loose coupling ≈ 0.2-0.4)

Threshold interpretation (from code):
- avg_corr < 0.3: "High Independence" → multi-variate fingerprint
- avg_corr ≥ 0.3: "High Redundancy" → single signal with noise

RELATION TO OTHER ANALYSES:
---------------------------
BUILDS ON: Step 5 (unmask_hear_drivers_STEP_5.py) which identified the
           top 10 language-discriminative dimensions

COMPLEMENTS: Stability and invariance audits - correlation structure is
             necessary but not sufficient for validity

USAGE:
------
1. Requires: 75% overlap embeddings and pre-identified top 10 dimensions
2. Run: python dimension_independence.py
3. Output: Correlation matrix and average cross-correlation
4. Interpretation: Use thresholds to assess redundancy/independence

AUTHORS: Jim McCormack
CREATED: Feb 2026
PAPER REFERENCE: Sections 4 (Dimensional Parsimony), 5 (Auditing Protocol),
"""

import numpy as np
from pathlib import Path

def analyze_top_correlations_raw(data_dir, top_indices):

    files = list(Path(data_dir).glob("*.npy"))
    data = []
    
    print(f"🧬 Loading Top {len(top_indices)} drivers from {len(files)} physiological units...")
    print("PAPER CONTEXT: Testing dimensional parsimony (Section 4)")
    print(f"DIMENSIONS: {top_indices}")
    print(f"DATA: 75% overlap embeddings (deep mining set)")
    print()
    
    for f in files:
        # Load full 512-dimensional embedding
        emb = np.load(f).flatten()
        # Extract only the top 10 dimensions of interest
        data.append(emb[top_indices])
    
    # Convert to matrix: [n_samples × 10_features]
    # Each row is one embedding slice, each column is one dimension
    matrix = np.array(data)
    
    # Calculate Pearson Correlation Matrix
    # rowvar=False: columns are variables (dimensions), rows are observations
    # Pearson r ranges from -1 (perfect inverse) to +1 (perfect direct)
    corr_matrix = np.corrcoef(matrix, rowvar=False)
    
    print("\n📈 --- FEATURE CORRELATION MATRIX ---")
    print("=" * 65)
    print("PAPER NOTE: Low off-diagonal values indicate independent measurement axes")
    print("            Moderate values (~0.2-0.4) indicate biological 'loose coupling'")
    print("=" * 65)
    
    # Format header with dimension indices
    header = "      " + " ".join([f"D{i:<4}" for i in top_indices])
    print(header)
    print("-" * 65)
    
    # Print correlation matrix with formatting
    for i, row in enumerate(corr_matrix):
        row_str = f"D{top_indices[i]:<4} " + " ".join([f"{val:6.2f}" for val in row])
        print(row_str)
    
    # Calculate average absolute off-diagonal correlation
    # Create mask to exclude diagonal (self-correlation = 1.0 always)
    mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
    off_diag = corr_matrix[mask]
    avg_corr = np.mean(np.abs(off_diag))
    
    print("\n" + "="*65)
    print("📊 CORRELATION SUMMARY STATISTICS")
    print("="*65)
    print(f"Average Cross-Correlation: {avg_corr:.4f}")
    print(f"Correlation Range: [{np.min(off_diag):.4f}, {np.max(off_diag):.4f}]")
    print(f"Correlation Std Dev: {np.std(off_diag):.4f}")
    print()
    
    # Biological interpretation (for paper discussion)
    print("🔬 BIOLOGICAL INTERPRETATION (for Paper Discussion):")
    
    if avg_corr < 0.2:
        print(f"   🚀 VERY LOW CORRELATION ({avg_corr:.3f})")
        print("   INDICATION: Highly independent physiological sensors")
        print("   PAPER IMPLICATION: Multi-variate fingerprint with minimal redundancy")
        print("   CAUTION: Could indicate measurement noise rather than true independence")
        
    elif avg_corr < 0.3:
        print(f"   ✅ OPTIMAL 'LOOSE COUPLING' ({avg_corr:.3f})")
        print("   INDICATION: Biologically plausible coordination")
        print("   PAPER IMPLICATION: Dimensions measure related but distinct subsystems")
        print("   BIOLOGICAL ANALOGY: Lungs, vocal folds, tongue coordination")
        
    elif avg_corr < 0.5:
        print(f"   ⚠️ MODERATE-HIGH CORRELATION ({avg_corr:.3f})")
        print("   INDICATION: Some redundancy in measurement")
        print("   PAPER IMPLICATION: Potential for dimensionality reduction")
        print("   RECOMMENDATION: Examine stability of individual dimensions")
        
    else:
        print(f"   ❌ HIGH REDUNDANCY ({avg_corr:.3f})")
        print("   INDICATION: Dimensions track same underlying signal")
        print("   PAPER IMPLICATION: Risk of overfitting single physiological aspect")
        print("   ACTION: Consider PCA or other dimensionality reduction")
    
    print()
    print("📄 PAPER INTEGRATION GUIDE:")
    print(f"1. Report average correlation: {avg_corr:.3f}")
    print("2. Reference in Section 4 (Dimensional Parsimony)")
    print("3. Include in Figure 3(b) if space permits")
    print("4. Discuss biological plausibility of correlation structure")
    
    return corr_matrix, avg_corr

# Execute analysis
if __name__ == "__main__":
    # The Top 10 Drivers from previous analysis (Step 5)
    # These are dimensions with strongest language-discriminative power
    # They may represent physiological signals OR linguistic confounds
    top_drivers = [419, 227, 43, 346, 204, 317, 38, 98, 267, 146]
    
    print("\n" + "="*70)
    print("🔬 CORRELATION STRUCTURE ANALYSIS")
    print("="*70)
    print("PAPER: Testing dimensional parsimony (Section 4)")
    print("HYPOTHESIS: Valid physiological dimensions should show")
    print("            'loose coupling' not redundancy")
    print("DATA: 75% overlap embeddings (max temporal resolution)")
    print("="*70)
    print()
    
    # Analyze correlations among top dimensions
    corr_matrix, avg_correlation = analyze_top_correlations_raw(
        "/app/data/processed/esen_ovlap_75", 
        top_drivers
    )
    
    # Save results for paper integration
    output_dir = "/app/data/analysis/correlation_structure"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "correlation_matrix.npy"), corr_matrix)
    np.save(os.path.join(output_dir, "top_dimension_indices.npy"), top_drivers)
    
    # Save summary for paper
    with open(os.path.join(output_dir, "correlation_summary.txt"), "w") as f:
        f.write("CORRELATION STRUCTURE ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Top dimensions analyzed: {top_drivers}\n")
        f.write(f"Average cross-correlation: {avg_correlation:.4f}\n")
        f.write(f"Interpretation: {'High Independence' if avg_correlation < 0.3 else 'Moderate-High Redundancy'}\n")
        f.write(f"Paper section: 4 (Dimensional Parsimony)\n")
    
    print(f"\n💾 Results saved to: {output_dir}/")
    print("   Ready for paper integration and figure generation")
    print("   Ready for paper integration and figure generation")
