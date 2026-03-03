# Step 6: Correlation Structure Analysis - Testing Orthogonality of Physiological Dimensions

## PAPER CONNECTION & SCIENTIFIC CONTEXT
This script implements a critical validity test from the paper's auditing framework: examining the correlation structure of candidate physiological dimensions.

### KEY PAPER REFERENCES
1. **SECTION 4 (Dimensional Parsimony)**: "Does screening rely on a small, low-correlation subspace?" This script provides the quantitative test for that requirement.

2. **SECTION 4 (Auditing Protocol)**: Implements the operational check for feature independence, preventing redundant measurement of the same underlying signal.

3. **FIGURE 3b**: Shows correlation structure comparison between full embedding space and purified subspace. This script generates the data for such analyses.

4. **ABSTRACT (Interpretability)**: Supports the claim of "more interpretable" deployment by ensuring dimensions represent independent physiological axes.

## SCIENTIFIC PURPOSE
Tests whether the top candidate physiological dimensions represent:

- **REDUNDANT NOISE**: Highly correlated dimensions (r > 0.9) measuring the same underlying signal → indicates overfitting risk
- **ORTHOGONAL SENSORS**: Low-correlation dimensions (r < 0.3) measuring independent physiological processes → indicates robust multi-variate fingerprint

This addresses a fundamental question in physiological measurement: *"Are we measuring multiple independent aspects of vocal physiology, or just the same signal with different noise?"*

## BIOLOGICAL INTERPRETATION OF CORRELATION PATTERNS

### 1. HIGH CORRELATION (r > 0.7)
- All dimensions track the same underlying process
- Like having 10 thermometers all measuring the same temperature
- **Biologically implausible**: vocal physiology has multiple semi-independent subsystems (respiratory, laryngeal, articulatory)
- **Indicates**: Model collapse or measurement redundancy

### 2. MODERATE CORRELATION (0.3 < r < 0.7)
- Dimensions measure related but distinct processes
- **Biologically realistic**: subsystems are coordinated but have degrees of freedom
- **"Loose coupling"** as described in code comments: lungs, vocal folds, tongue are connected but independent
- **Optimal** for multi-variate physiological fingerprinting

### 3. LOW CORRELATION (r < 0.3)
- Dimensions measure independent processes
- **Could indicate**: Truly orthogonal physiology OR measurement noise
- **Requires validation** with stability/invariance audits

## THE "LOOSE COUPLING" HYPOTHESIS
The code comment insight is biologically profound: at ~0.24 average correlation, we see **"Loose Coupling"** - exactly how biological systems work. Vocal physiology involves multiple subsystems that must coordinate but maintain independence.

This contrasts with:
- **Engineering systems**: tightly coupled
- **Random noise**: completely uncoupled

Finding loose coupling supports the paper's claim that foundation models capture biologically plausible representations.

## METHODOLOGICAL IMPORTANCE

### 1. DEFENSE AGAINST OVERFITTING
Low correlation means model isn't just learning redundant representations of the same signal (code comment: "defended against the 'Overfitting'")

### 2. VALIDATION OF MULTI-VARIATE MEASUREMENT
Independent dimensions suggest the model captures multiple aspects of physiology, not just one dominant signal

### 3. INFORMING MODEL INTERPRETATION
Correlation structure tells us whether dimensions can be interpreted as independent biomarkers or must be treated as a combined syndrome measure

## DATA SOURCE RATIONALE
Uses **75% overlap embeddings** ("Deep Mining" set) because:
1. Maximum sample size for correlation estimation
2. Captures subtle physiological transitions that might reveal coupling patterns
3. Consistent with other analyses in the language invariance testing pipeline

The **top 10 dimensions** analyzed are those identified in Step 5 as having strongest language-discriminative power (potential physiological signals or linguistic confounds).

## PURPOSE
Analyze the pairwise correlations among the top 10 candidate physiological dimensions to determine whether they represent:

1. **Redundant measurements** of the same signal (high correlation → problematic)
2. **Independent physiological sensors** (low correlation → desirable)
3. **Loosely coupled biological subsystems** (moderate correlation → biologically plausible)

## OUTPUT INTERPRETATION FOR PAPER
The average cross-correlation should be reported in the paper as:
- Quantitative evidence supporting dimensional parsimony (Section 8.5)
- Defense against overfitting claims
- Biological plausibility indicator (loose coupling ≈ 0.2-0.4)

**Threshold interpretation** (from code):
- `avg_corr < 0.3`: "High Independence" → multi-variate fingerprint
- `avg_corr ≥ 0.3`: "High Redundancy" → single signal with noise

## RELATION TO OTHER ANALYSES
- **BUILDS ON**: Step 5 which identified the top 10 language-discriminative dimensions
- **COMPLEMENTS**: Stability and invariance audits - correlation structure is necessary but not sufficient for validity
- **FEEDS INTO**: Figure 3(b) in paper showing correlation structure of purified subspace vs full embedding space

## USAGE
1. **Requires**: 75% overlap ES EN embeddings and pre-identified top 10 dimensions
   - **Directory**: `/app/data/processed/esen_ovlap_75/`
2. **Run**: `python correlation_analysis.py`
3. **Output**: Correlation matrix and average cross-correlation
4. **Interpretation**: Use thresholds to assess redundancy/independence

---

**AUTHORS**: Jim McCormack  
**CREATED**: Feb 2026  
**PAPER REFERENCE**: Sections 8.5 (Dimensional Parsimony), 5 (Auditing Protocol), Figure 3(c) (Correlation Structure)
