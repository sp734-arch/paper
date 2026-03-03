# Step 5: Physiological Driver Discovery - Language-Invariant Trait Identification

## PAPER CONNECTION & SCIENTIFIC CONTEXT
This script implements a critical diagnostic test in the paper's stability-first auditing framework. It answers the question: *"Which HeAR embedding dimensions capture systematic differences between Spanish and English speakers?"*

### WHY THIS MATTERS FOR THE PAPER
1. **TESTS LANGUAGE INVARIANCE PRINCIPLE** (Section 4): Dimensions that strongly distinguish languages are likely capturing linguistic/prosodic confounds rather than pure physiology. These dimensions become candidates for **REJECTION** in the invariance audit.

2. **IMPLEMENTS LABEL-FREE AUDITING** (Section 4): Uses language labels (not disease labels) to discover potential confounds **BEFORE** any disease prediction is attempted. This prevents shortcut learning.

3. **SUPPORTS EQUITY CLAIMS** (Abstract): By identifying language-sensitive dimensions, we can explicitly control for or remove them, ensuring models work equitably across linguistic populations.

4. **INFORMS CALIBRATION STRATEGY** (Section 7): If strong language-specific dimensions exist, language-specific calibration may be needed. If not, single calibration may suffice.

## SCIENTIFIC HYPOTHESES BEING TESTED

### 1. THE "LINGUISTIC RESPIRATORY BASELINE" HYPOTHESIS
Do different languages induce systematic physiological differences in respiratory/vocal patterns? This would manifest as dimensions that strongly separate Spanish vs English.

### 2. THE "CONFOUND VS SIGNAL" DICHOTOMY
Are language-distinguishing dimensions:
- **CONFOUNDS**: Capturing linguistic content, prosody, or recording artifacts?
- **SIGNALS**: Capturing true physiological differences between populations?

The paper's auditing framework provides the test: stable, invariant dimensions may be signals; unstable, sensitive dimensions are confounds.

### 3. THE "CLIFF VS DISTRIBUTED" PATTERN
- **CLIFF PATTERN**: Top 10 dimensions >> rest → Specific physiological parameters
- **DISTRIBUTED PATTERN**: Weights spread across space → Holistic integration

This informs whether foundation models extract specific biomarkers or integrated physiological patterns.

## METHODOLOGICAL APPROACH
We train a linear SVM to distinguish Spanish vs English embeddings. The key insight: *"Disease prediction should be language-invariant; language prediction reveals potential confounds."*

**Counter-intuitive logic**: Dimensions that are **GOOD** at language discrimination are **BAD** candidates for disease screening (unless they also show stability and invariance to non-biological factors).

## DATA SOURCE EXPLANATION
Uses **75% overlap embeddings** ("Deep Mining" set) from emotional speech data.

**Why 75% overlap?** Maximum temporal resolution to capture subtle physiological transitions and respiratory patterns that might differ between languages.

**File naming convention**: `{SUBJECT}_{LANGUAGE}_{SEQ}.npy`
- `_EN_`: English emotional speech
- `_ES_`: Spanish emotional speech

This provides clean language labels for the diagnostic task.

## PURPOSE
Identify the top 10 HeAR embedding dimensions that most strongly differentiate Spanish vs English emotional speech. These dimensions become:

1. **CANDIDATES FOR REJECTION** in invariance audit (if sensitive to language)
2. **SUBJECTS FOR EXAMINATION** in stability audit (to see if stable within subjects)
3. **INFORMANTS for calibration strategy** (language-specific vs universal)

## OUTPUT INTERPRETATION FOR PAPER
High-weight dimensions should be reported as:
- Evidence of language-specific signal in foundation model embeddings
- Demonstration of the need for language invariance testing
- Quantitative basis for equity considerations in deployment

The **magnitude of weights** indicates strength of language association.  
The **"cliff ratio"** (top 10 mean / rest mean) indicates concentration vs distribution.

## RELATION TO OTHER STEPS
- **CONTRASTS**: Step 3 finds disease-predictive dimensions; this finds language-predictive dimensions. Together they map the full confound landscape described in the paper.
- **FEEDS INTO**: Subsequent stability and invariance audits that test whether language-sensitive dimensions are stable within subjects and invariant to other confounds.

## USAGE
1. **Requires**: 75% overlap embeddings from emotional speech data
   - **Directory**: `/app/data/processed/esen_ovlap_75/`
2. **Run**: `python <Script name>` (replace with actual script name)
3. **Output**: Top 10 language-discriminative dimensions
4. **Next**: Test these dimensions in stability and invariance audits

---

**AUTHORS**: Jim McCormack  
**CREATED**: Feb, 2026  
**PAPER REFERENCE**: Sections 8.3 (Invariance), 8.4 (Label-Free Auditing), Abstract (Equity)
