"""
Step 5: Physiological Driver Discovery - Language-Invariant Trait Identification
===============================================================================

PAPER CONNECTION & SCIENTIFIC CONTEXT:
---------------------------------------
This script implements a critical diagnostic test in the paper's stability-first
auditing framework. It answers the question: "Which HeAR embedding dimensions
capture systematic differences between Spanish and English speakers?"

WHY THIS MATTERS FOR THE PAPER:
1. TESTS LANGUAGE INVARIANCE PRINCIPLE (Section 4): Dimensions that strongly
   distinguish languages are likely capturing linguistic/prosodic confounds
   rather than pure physiology. These dimensions become candidates for REJECTION
   in the invariance audit.

2. IMPLEMENTS LABEL-FREE AUDITING (Section 4): Uses language labels (not
   disease labels) to discover potential confounds BEFORE any disease prediction
   is attempted. This prevents shortcut learning.

3. SUPPORTS EQUITY CLAIMS (Abstract): By identifying language-sensitive
   dimensions, we can explicitly control for or remove them, ensuring models
   work equitably across linguistic populations.

4. INFORMES CALIBRATION STRATEGY (Section 7): If strong language-specific
   dimensions exist, language-specific calibration may be needed. If not,
   single calibration may suffice.

SCIENTIFIC HYPOTHESES BEING TESTED:
------------------------------------
1. THE "LINGUISTIC RESPIRATORY BASELINE" HYPOTHESIS: Do different languages
   induce systematic physiological differences in respiratory/vocal patterns?
   This would manifest as dimensions that strongly separate Spanish vs English.

2. THE "CONFOUND VS SIGNAL" DICHOTOMY: Are language-distinguishing dimensions:
   - CONFOUNDS: Capturing linguistic content, prosody, or recording artifacts?
   - SIGNALS: Capturing true physiological differences between populations?
   The paper's auditing framework provides the test: stable, invariant dimensions
   may be signals; unstable, sensitive dimensions are confounds.

3. THE "CLIFF VS DISTRIBUTED" PATTERN (from code comments):
   - CLIFF PATTERN: Top 10 dimensions >> rest → Specific physiological parameters
   - DISTRIBUTED PATTERN: Weights spread across space → Holistic integration
   This informs whether foundation models extract specific biomarkers or
   integrated physiological patterns.

METHODOLOGICAL APPROACH:
------------------------
We train a linear SVM to distinguish Spanish vs English embeddings. The key insight:
"Disease prediction should be language-invariant; language prediction reveals
potential confounds."

Counter-intuitive logic: Dimensions that are GOOD at language discrimination
are BAD candidates for disease screening (unless they also show stability and
invariance to non-biological factors).

DATA SOURCE EXPLANATION:
------------------------
Uses 75% overlap embeddings ("Deep Mining" set) from emotional speech data.
Why 75% overlap? Maximum temporal resolution to capture subtle physiological
transitions and respiratory patterns that might differ between languages.

File naming convention: {SUBJECT}_{LANGUAGE}_{SEQ}.npy
- "_EN_": English emotional speech
- "_ES_": Spanish emotional speech

This provides clean language labels for the diagnostic task.

PURPOSE:
--------
Identify the top 10 HeAR embedding dimensions that most strongly differentiate
Spanish vs English emotional speech. These dimensions become:
1. CANDIDATES FOR REJECTION in invariance audit (if sensitive to language)
2. SUBJECTS FOR EXAMINATION in stability audit (to see if stable within subjects)
3. INFORMANTS for calibration strategy (language-specific vs universal)

OUTPUT INTERPRETATION FOR PAPER:
--------------------------------
High-weight dimensions should be reported:
- Evidence of language-specific signal in foundation model embeddings
- Demonstration of the need for language invariance testing
- Quantitative basis for equity considerations in deployment

The magnitude of weights indicates strength of language association.
The "cliff ratio" (top 10 mean / rest mean) indicates concentration vs distribution.

RELATION TO OTHER STEPS:
------------------------
CONTRASTS: That script finds disease-predictive dimensions; this finds
           language-predictive dimensions. Together they map the full
           confound landscape described in the paper.

FEEDS INTO: Subsequent stability and invariance audits that test whether
            language-sensitive dimensions are stable within subjects and
            invariant to other confounds.

USAGE:
------
1. Requires: 75% overlap embeddings from emotional speech data
   Directory: /app/data/processed/esen_ovlap_75/
2. Run: python <Script name>
3. Output: Top 10 language-discriminative dimensions
4. Next: Test these dimensions in stability and invariance audits

AUTHORS: Jim McCormack
CREATED: Feb, 2016
PAPER REFERENCE: Sections 8.3 (Invariance), 8.4 (Label-Free Auditing), Abstract (Equity)
"""

import numpy as np
import os
from pathlib import Path
from sklearn.svm import LinearSVC

def extract_signal_drivers(data_dir):
    """
    Extract HeAR embedding dimensions that distinguish Spanish vs English speech.
    
    PAPER CONTEXT: Implements diagnostic test for language invariance principle
    (Section 8.3). Language-discriminative dimensions are potential confounds
    that must be examined in the full auditing pipeline.
    
    SCIENTIFIC RATIONALE: If a dimension strongly signals language difference,
    it likely captures:
    1. Linguistic content differences (confound - should be rejected)
    2. Prosodic patterns (confound - should be rejected)  
    3. Recording environment correlations (confound - should be rejected)
    4. True physiological population differences (rare, requires validation)
    
    METHOD: Linear SVM with strong regularization (C=0.1) to find sparse,
    interpretable dimensions that discriminate languages.
    
    Args:
        data_dir (str): Path to directory containing .npy embedding files
                       Files must contain "_EN_" or "_ES_" in names for labeling
    
    Returns:
        tuple: (top_indices, all_weights)
            - top_indices: Top 10 dimension indices (descending by weight)
            - all_weights: Absolute weights for all 512 dimensions
    
    PAPER INTEGRATION:
    Report results in:
    - Methods: As part of label-free auditing protocol
    - Results: Quantitative evidence of language-sensitive dimensions
    - Discussion: Implications for cross-linguistic generalizability
    """
    files = list(Path(data_dir).glob("*.npy"))
    X, y = [], []
    
    print("🧠 Training Diagnostic Model for Language Signal Discovery...")
    print("PAPER CONTEXT: Label-free auditing (Section 8.4)")
    
    for f in files:
        # Load 512-dimensional HeAR embedding
        X.append(np.load(f).flatten())
        # Create binary labels from filename: 1=English, 0=Spanish
        y.append(1 if "_EN_" in f.name else 0)
    
    # Linear SVM with strong regularization
    # Strong regularization (C=0.1) encourages sparse solutions
    # This helps identify concentrated language signals rather than distributed noise
    clf = LinearSVC(dual=False, C=0.1, random_state=42, max_iter=10000)
    clf.fit(X, y)
    
    # Get absolute weights - magnitude indicates importance for language discrimination
    weights = np.abs(clf.coef_[0])
    
    # Identify top 10 dimensions with highest weights
    top_indices = np.argsort(weights)[-10:][::-1]  # The Top 10 Drivers
    
    print("\n" + "="*60)
    print("🚀 TOP 10 LANGUAGE-DISCRIMINATIVE DIMENSIONS")
    print("="*60)
    print("NOTE: High weight = Strong language association")
    print("      May indicate linguistic confound (examine in audits)")
    print("-" * 60)
    
    # Calculate statistics for paper reporting
    avg_weight = np.mean(weights)
    top_mean = np.mean(weights[top_indices])
    rest_mean = np.mean(np.delete(weights, top_indices))
    cliff_ratio = top_mean / rest_mean
    
    for i, idx in enumerate(top_indices):
        weight = weights[idx]
        rel_importance = weight / avg_weight
        print(f"{i+1:2d}. Dimension {idx:3}: Weight {weight:.6f} ({rel_importance:.2f}x avg)")
    
    print("=" * 60)
    print()
    
    # Pattern analysis for paper discussion
    print("📊 PATTERN ANALYSIS (for Paper Discussion):")
    print(f"   Cliff ratio (top 10 mean / rest mean): {cliff_ratio:.2f}")
    
    if cliff_ratio > 3.0:
        print("   🎯 STRONG CLIFF: Concentrated language signals")
        print("   PAPER IMPLICATION: Specific dimensions capture linguistic differences")
    elif cliff_ratio > 1.5:
        print("   📊 MODERATE CONCENTRATION: Some specialization")
        print("   PAPER IMPLICATION: Mixed language encoding")
    else:
        print("   🌊 DISTRIBUTED: Holistic language encoding")
        print("   PAPER IMPLICATION: Language differences spread across embedding")
    
    print()
    
    # Paper integration guidance
    print("🔬 PAPER INTEGRATION NOTES:")
    print("1. These dimensions are LANGUAGE-SENSITIVE → potential confounds")
    print("2. Next: Test in stability audit (Section 4)")
    print("   - Are they stable within subjects across tasks?")
    print("3. Then: Test in invariance audit (Section 4)") 
    print("   - Are they sensitive to other confounds (task, recording)?")
    print("4. Decision: Reject if unstable or confound-sensitive")
    print()
    print("📈 EQUITY CONSIDERATION (Paper Abstract):")
    print("   Language-sensitive dimensions could cause performance disparities")
    print("   across linguistic groups. Auditing them addresses equity concerns.")
    
    return top_indices, weights

# Execute analysis
if __name__ == "__main__":
    # Use 75% overlap embeddings ("Deep Mining" set)
    # Maximum temporal resolution for capturing subtle physiological patterns
    # that might differ between Spanish and English speakers
    top_dims, all_weights = extract_signal_drivers("/app/data/processed/esen_ovlap_75")
    
    # Save results for paper integration and subsequent audits
    output_dir = "/app/data/analysis/language_drivers"
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "top_10_language_dimensions.npy"), top_dims)
    np.save(os.path.join(output_dir, "all_language_weights.npy"), all_weights)
    
    print(f"\n💾 Results saved to: {output_dir}/")
    print("   Files prepared for paper integration and auditing pipeline")
    
    
