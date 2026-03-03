"""
Step 8: Biomarker Stability Leaderboard - Individual Dimension Stability Ranking
===============================================================================

PAPER CONNECTION & SCIENTIFIC CONTEXT:
---------------------------------------
This script implements the FINAL RANKING STAGE of the paper's stability-first
auditing framework. It identifies which individual dimensions demonstrate the
highest subject-level stability, providing a quantitative ranking of candidate
biomarkers by their reliability as physiological measurements.

KEY PAPER REFERENCES:
1. SECTION 5 (Auditing Protocol): "Dimensions were required to exceed a
   minimum stability threshold of ICC 0.5." This script provides the exact
   stability metrics for that filtering.

2. SECTION 9 (Construct Alignment with Clinical Severity): Mentions
   "Dimension 38 - one of the seven dimensions retained by the stability-and-
   invariance audit" as having clinical correlation. This script shows HOW
   dimensions are ranked by stability before clinical validation.

3. FIGURE 2(a) (Stability Spectrum): Shows embedding dimensions sorted by
   within-subject stability. This script generates the quantitative data for
   such visualizations.

4. ABSTRACT (Stable Vocal Structure): "Our results show that stable vocal
   structure can be isolated from foundation embeddings." This script
   identifies the MOST stable dimensions.

SCIENTIFIC PURPOSE:
-------------------
While previous steps tested stability of the TOP 10 dimensions as a GROUP,
this script examines EACH DIMENSION INDIVIDUALLY to answer:

"Which specific HeAR embedding dimensions demonstrate the highest
subject-level stability, and therefore represent the strongest candidate
biomarkers for physiological measurement?"

PURPOSE:
--------
Rank the top candidate dimensions by their individual stability ratios to:

This is the FINAL analytical step before dimension selection for the
published model.

RELATION TO FINAL MODEL:
------------------------
The final Purified V2 model uses 7 dimensions: [38, 43, 146, 204, 267, 346, 419]

AUTHORS: Jim McCormack
CREATED: Feb 2026
"""

import numpy as np
from pathlib import Path

def biomarker_leaderboard(data_dir, top_indices):
    files = list(Path(data_dir).glob("*.npy"))
    subject_data = {sub: [] for sub in set(f.name.split('_')[0] for f in files)}
    for f in files:
        subject_data[f.name.split('_')[0]].append(np.load(f).flatten()[top_indices])

    print(f"🏆 --- MARKER STABILITY LEADERBOARD ---")
    print(f"{'Dimension':<12} | {'Stability Ratio':<15}")
    print("-" * 30)

    for i, idx in enumerate(top_indices):
        intra_vars = []
        inter_means = []
        for sub, embs in subject_data.items():
            if len(embs) < 2: continue
            arr = np.array(embs)[:, i]
            intra_vars.append(np.var(arr))
            inter_means.append(np.mean(arr))
        
        ratio = np.var(inter_means) / np.mean(intra_vars)
        print(f"Dim {idx:<9} | {ratio:<15.4f}")

top_drivers = [419, 43, 346, 204, 38, 267, 146]
# dropped lower stability 227, 98,317 
biomarker_leaderboard("/app/data/processed/esen_ovlap_50", top_drivers)
