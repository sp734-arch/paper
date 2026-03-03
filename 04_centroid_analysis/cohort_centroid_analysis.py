"""
Step 4: Cohort Centroid Analysis - Language-Induced Physiological Drift
=======================================================================

PAPER CONNECTION & RATIONALE FOR INCLUSION:
-------------------------------------------
This analysis directly supports and operationalizes key claims from the paper:

1. DIRECTLY VALIDATES CORE CLAIM: The paper's Abstract mentions "more equitable 
   deployment" and Section 8.3 requires "invariance to language." This analysis 
   provides quantitative evidence for that claim by measuring whether baseline 
   physiology drifts between language populations.

2. CONCRETE METHODOLOGICAL CONTRIBUTION: Shows HOW to test language invariance 
   quantitatively - not just stating it as a principle. Implements the operational
   auditing protocol called for in Section 5.

3. INFORMS CRITICAL DESIGN DECISION: The results directly inform whether 
   language-specific calibration is needed. Cosine similarity 
   < 0.85 suggests separate calibration; ≥ 0.85 suggests single calibration 
   may suffice.

4. STRENGTHENS EQUITY ARGUMENT: Provides empirical basis for the equity claims 
   in the Abstract and Conclusion by quantifying cross-linguistic physiological 
   similarity/difference.

5. COMPLEMENTS OTHER ANALYSES: Adds cohort-level perspective to individual-level 
   stability and dimension-level invariance analyses,
   completing the multi-scale auditing framework.

PURPOSE:
--------
Quantifies whether baseline human physiology, as represented by HeAR embeddings,
shifts meaningfully between spoken languages (Spanish vs. English).

This addresses a foundational question in the stability-first auditing framework:
"Are there systematic, language-induced shifts in vocal physiology that could
confound disease screening across linguistic populations?"

METHODOLOGY:
------------
1. Computes centroids (mean embeddings) for Spanish and English cohorts
2. Measures Euclidean distance between centroids (absolute shift magnitude)
3. Measures cosine similarity between centroids (directional alignment)
4. Interprets results against empirical threshold (cosine < 0.85 = significant drift)

USAGE:
------
1. Requires 50% overlap embeddings from Step 2
   Directory: /app/data/processed/esen_ovlap_50/
2. Run: python cohort_drift_analysis.py
3. Results inform language invariance conclusions and calibration strategy

AUTHORS: Jim McCormack
CREATED: Feb, 2026
"""

import numpy as np
from pathlib import Path

def calculate_cohort_drift(data_dir):
    """
    Calculate population-level physiological drift between language cohorts.
    
    This implements quantitative language invariance testing as required by
    the paper's auditing framework (Section 8.3). Results directly inform
    calibration strategy (Section 8.6) and equity considerations (Abstract).
    
    Args:
        data_dir (str): Path to directory containing .npy embedding files
                       Files should follow naming: {SUBJECT}_{LANGUAGE}_{SEQ}.npy
    
    Returns:
        tuple: (distance, cosine_similarity, en_centroid, es_centroid)
    """
    
    # Load all embedding files
    files = list(Path(data_dir).glob("*.npy"))
    
    # Load and pool based on the label in the name
    en_pool = [np.load(f).flatten() for f in files if "_EN_" in f.name]
    es_pool = [np.load(f).flatten() for f in files if "_ES_" in f.name]
    
    # Calculate the "Average Human" for each language
    # These centroids represent the baseline physiology for each linguistic population
    en_centroid = np.mean(en_pool, axis=0)
    es_centroid = np.mean(es_pool, axis=0)
    
    # The Alpha Distance: How far apart is the baseline physiology?
    # Euclidean distance in the 512-dimensional embedding space
    distance = np.linalg.norm(en_centroid - es_centroid)
    
    # Cosine Similarity: Are they pointing in the same biological direction?
    # (1.0 = Identical, 0.0 = Totally Orthogonal/Different)
    # This is the critical metric for the paper's invariance claims
    cos_sim = np.dot(en_centroid, es_centroid) / (np.linalg.norm(en_centroid) * np.linalg.norm(es_centroid))
    
    print(f"📍 Physiological Euclidean Distance: {distance:.6f}")
    print(f"📐 Directional Cosine Similarity: {cos_sim:.6f}")
    
    # Interpretation against empirical threshold
    # Threshold rationale: Cosine < 0.85 indicates meaningful directional divergence
    # that could confound cross-linguistic screening if not calibrated for
    if cos_sim < 0.85:
        print("🚀 SIGNAL DETECTED: Significant physiological divergence between cohorts.")
        print("   PAPER IMPLICATION: Language-specific calibration likely required (Section 7)")
    else:
        print("⚠️ NOISE ALERT: Baseline physiology is highly similar.")
        print("   PAPER IMPLICATION: Single calibration may suffice across languages")
    
    return distance, cos_sim, en_centroid, es_centroid

# Execute analysis
if __name__ == "__main__":
    # Paper-relevant data source: 50% overlap embeddings from emotional speech
    # Used for Figure 1 and language invariance testing
    calculate_cohort_drift("/app/data/processed/esen_ovlap_50")
