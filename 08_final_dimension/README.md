# Step 9: Biomarker Stability Leaderboard - Individual Dimension Stability Ranking

## PAPER CONNECTION & SCIENTIFIC CONTEXT
This script implements the **FINAL RANKING STAGE** of the paper's stability-first auditing framework. It identifies which individual dimensions demonstrate the highest subject-level stability, providing a quantitative ranking of candidate biomarkers by their reliability as physiological measurements.

### KEY PAPER REFERENCES
1. **SECTION 5 (Auditing Protocol)**: "Dimensions were required to exceed a minimum stability threshold of ICC 0.5." This script provides the exact stability metrics for that filtering.

2. **SECTION 8 (Construct Alignment with Clinical Severity)**: Mentions "Dimension 38 - one of the seven dimensions retained by the stability-and-invariance audit" as having clinical correlation. This script shows **HOW** dimensions are ranked by stability before clinical validation.

3. **FIGURE 2(a) (Stability Spectrum)**: Shows embedding dimensions sorted by within-subject stability. This script generates the quantitative data for such visualizations.

4. **ABSTRACT (Stable Vocal Structure)**: "Our results show that stable vocal structure can be isolated from foundation embeddings." This script identifies the **MOST stable dimensions**.

## SCIENTIFIC PURPOSE
While previous steps tested stability of the **TOP 10 dimensions as a GROUP**, this script examines **EACH DIMENSION INDIVIDUALLY** to answer:

*"Which specific HeAR embedding dimensions demonstrate the highest subject-level stability, and therefore represent the strongest candidate biomarkers for physiological measurement?"*

## PURPOSE
Rank the top candidate dimensions by their individual stability ratios to:

1. **ID:** IDENTIFY the most reliable dimensions for physiological measurement
2. **FILTER:** Apply the ICC > 0.5 threshold from Section 5
3. **SELECT:** Choose the final dimensions for the Purified V2 manifold
4. **REPORT:** Provide quantitative stability rankings for publication

This is the **FINAL analytical step** before dimension selection for the published model.

## RELATION TO FINAL MODEL
The final Purified V2 model uses **7 dimensions**: `[38, 43, 146, 204, 267, 346, 419]`

These dimensions were selected through:
- **Stability filtering** (this script's leaderboard)
- **Invariance validation** (rejecting confounded dimensions)
- **Clinical correlation** (Section 8 construct alignment)

---

**AUTHORS**: Jim McCormack  
**CREATED**: Feb 2026  
**PAPER REFERENCE**: Sections 5 (stability threshold), 8 (dimension 38), Figure 2(a) (stability spectrum), Abstract (stable structure)
