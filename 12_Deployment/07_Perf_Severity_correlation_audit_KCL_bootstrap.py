#!/usr/bin/env python3
"""
AUDIT STEP 07: Clinical Severity Correlation & Bootstrap Stability

Purpose:
Validates that the 7-dimensional audited manifold (specifically Dimension 38) 
tracks the physiological progression of Parkinson's Disease. It correlates model 
embeddings against clinical 'Gold Standards': Hoehn & Yahr (H&Y) and 
UPDRS-III (Motor) scores [Ref: Section 8.5].

Validity Requirement:
- Checklist Item 4: Clinical Validity [Ref: Section 8.4].
- Physiological Grounding: Ensures the model is measuring disease severity 
  rather than binary acoustic artifacts [Ref: Section 8.5].

Inputs:
- Feature Source: C:\Projects\hear_italian\features_kcl\parkinsons (Stored .npy)
- Metadata: Encoded in filenames (SubjectID_H&Y_UPDRS_...) [Ref: Filename Structure].

Outputs:
- Terminal Report: Spearman’s rho (ρ) for H&Y and UPDRS-III.
- Bootstrap Evidence: 95% Confidence Intervals (N=10,000) for correlation stability.

Technical Specs:
- Target Dimension: 38 (Stability-Audited Index) [Ref: Figure 3a].
- Aggregation: Subject-level mean (one state per human).
- Statistics: Non-parametric Spearman Rank Correlation [Ref: Section 5.4].

Author: Jim McCormack
Date: Feb 2026
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import sys

# ======================== CLINICAL CONFIG ========================
# Points directly to your extracted PD features
BASE = Path(r"C:\Projects\hear_italian\features_kcl\parkinsons")

DIM_38 = 38
N_BOOT = 10_000
RNG = np.random.default_rng(42)
# =================================================================

if not BASE.exists():
    print(f"❌ ERROR: Directory not found: {BASE}")
    sys.exit()

all_npy = list(BASE.glob("*.npy"))
print(f"📂 Found {len(all_npy)} .npy files.")

records = []

# Verify indices using your example: ID02_ID02_pd_2_0_0_s9
# Index: 0(ID02) 1(ID02) 2(pd) 3(2) 4(0)
sample_parts = all_npy[0].stem.split("_")
print(f"🔬 Filename structure: {sample_parts}")
print(f"   Using Index 3 for H&Y: '{sample_parts[3]}'")
print(f"   Using Index 4 for UPDRS: '{sample_parts[4]}'")

for f in all_npy:
    try:
        parts = f.stem.split("_")
        
        subject = parts[0]
        hy = int(parts[3])      
        updrs_iii = int(parts[4]) 

        emb = np.load(f).flatten()
        val = emb[DIM_38]

        records.append({
            "subject": subject,
            "hy": hy,
            "updrs_iii": updrs_iii,
            "dim38": val
        })
    except Exception:
        continue

if not records:
    print(f"❌ ERROR: Parsing failed. Ensure indices 3 and 4 are the numeric scores.")
    sys.exit()

df = pd.DataFrame(records)
print(f"✅ Processed {len(df)} samples from {df['subject'].nunique()} subjects.")

# Aggregation: One clinical state per human (The Subject Mean)
subj = df.groupby("subject").agg({
    "dim38": "mean", 
    "hy": "first", 
    "updrs_iii": "first"
}).reset_index()

X, Y_hy, Y_up = subj["dim38"].values, subj["hy"].values, subj["updrs_iii"].values
N = len(subj)

# Point estimates
rho_hy, _ = spearmanr(X, Y_hy)
rho_up, _ = spearmanr(X, Y_up)

# Bootstrap for robust 95% Confidence Intervals

# --- DIAGNOSTIC BLOCK ---
print("\n📊 Clinical Score Distribution Check:")
print(f"Unique H&Y values:   {subj['hy'].unique()}")
print(f"Unique UPDRS values: {subj['updrs_iii'].unique()}")

if subj['updrs_iii'].std() == 0:
    print("🛑 ALERT: Your UPDRS values are all the same (Zero Variance). Correlation is impossible.")
elif (subj['updrs_iii'] == 0).sum() > (len(subj) / 2):
    print("⚠️ ALERT: Most UPDRS scores are '0'. This 'Floor Effect' is killing your correlation.")
# ------------------------



print(f"🔄 Running {N_BOOT} bootstrap iterations...")
boot_hy = [spearmanr(X[idx], Y_hy[idx])[0] for idx in (RNG.integers(0, N, N) for _ in range(N_BOOT))]
boot_up = [spearmanr(X[idx], Y_up[idx])[0] for idx in (RNG.integers(0, N, N) for _ in range(N_BOOT))]

print("\n" + "="*50)
print(f"📊 CORRELATION RESULTS (N={N} Subjects)")
print("="*50)
print(f"Stage (H&Y) vs Dim 38:   ρ = {rho_hy:.3f} | 95% CI: [{np.percentile(boot_hy, 2.5):.3f}, {np.percentile(boot_hy, 97.5):.3f}]")
print(f"Motor (UPDRS) vs Dim 38: ρ = {rho_up:.3f} | 95% CI: [{np.percentile(boot_up, 2.5):.3f}, {np.percentile(boot_up, 97.5):.3f}]")
print("="*50)