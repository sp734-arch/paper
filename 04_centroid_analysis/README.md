# Step 4: Cohort Centroid Analysis - Language-Induced Physiological Drift

## PAPER CONNECTION & RATIONALE FOR INCLUSION
This analysis directly supports and operationalizes key claims from the paper:

### 1. CONCRETE METHODOLOGICAL CONTRIBUTION
Shows **HOW** to test language invariance quantitatively - not just stating it as a principle. Implements the operational auditing protocol called for in Section 5.

### 2. INFORMS CRITICAL DESIGN DECISION
The results directly inform whether language-specific calibration is needed

**Interpretation thresholds:**
- Cosine similarity **< 0.85**: suggests separate calibration needed
- Cosine similarity **≥ 0.85**: suggests single calibration may suffice

### 3. STRENGTHENS EQUITY ARGUMENT
Provides empirical basis for the equity claims in the Abstract and Conclusion by quantifying cross-linguistic physiological similarity/difference.

Completes the **multi-scale auditing framework**.

## PURPOSE
Quantifies whether baseline human physiology, as represented by HeAR embeddings, shifts meaningfully between spoken languages (Spanish vs. English).

This addresses a foundational question in the stability-first auditing framework: *"Are there systematic, language-induced shifts in vocal physiology that could confound disease screening across linguistic populations?"*

## METHODOLOGY
1. **Computes centroids** (mean embeddings) for Spanish and English cohorts
2. **Measures Euclidean distance** between centroids (absolute shift magnitude)
3. **Measures cosine similarity** between centroids (directional alignment)
4. **Interprets results** against empirical threshold (cosine < 0.85 = significant drift)

## USAGE
1. Requires 50% overlap embeddings from Step 2:
   - **Directory**: `/app/data/processed/esen_ovlap_50/`
2. Run: `python cohort_drift_analysis.py`
3. Results inform language invariance conclusions and calibration strategy

---

**AUTHORS**: Jim McCormack  
**CREATED**: Feb, 2026  
**PAPER REFERENCE**: Sections 8.3 (Invariance), 8.6 (Calibration), Abstract (Equity)
