#!/usr/bin/env python3
"""
Generate Figure 2: Stability Spectrum of HeAR Embedding Dimensions

Visualizes the stability ratio for all 512 HeAR dimensions, demonstrating
that only a small subset exhibits trait-like stability.

Stability Metric:
- Ratio = Between-subject variance / Within-subject variance
- High ratio (>1.5): Dimension is stable within subjects, varies between subjects
- Low ratio (<1.0): Dimension is noisy, inconsistent

Data Requirements:
- HeAR embeddings from Italian + English cohorts
- Multiple recordings per subject
- Path: ./features_audit/density_50/

Expected Output:
- Clear "cliff" showing separation between stable and unstable dimensions
- Top ~10-15 dimensions show high stability (orange bars)
- Justifies dimensionality reduction (512 → 7)

Paper Reference: Figure 3a
Author: Jim McCormack
Date: Feb 2026

Generate Figure 3a: Stability Spectrum of HeAR Embedding Dimensions (1:1 Ratio)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
AUDIT_ROOT = "../features_audit/density_50" 
FINAL_7 = [38, 43, 146, 204, 267, 346, 419]

def compute_stability_metric(X, subject_labels):
    """
    Quantifies within-subject stability across tasks. 
    High values indicate the dimension captures speaker-specific physiology.
    """
    unique_subjects = np.unique(subject_labels)
    n_dims = X.shape[1]
    stability_ratios = np.zeros(n_dims)
    
    global_mean = np.mean(X, axis=0)
    
    for dim in range(n_dims):
        between_var = 0
        within_var_list = []
        
        for sub in unique_subjects:
            sub_data = X[subject_labels == sub, dim]
            if len(sub_data) < 2: continue
            
            within_var_list.append(np.var(sub_data, ddof=1))
            between_var += (np.mean(sub_data) - global_mean[dim])**2
            
        avg_within = np.mean(within_var_list)
        stability_ratios[dim] = (between_var / len(unique_subjects)) / avg_within if avg_within != 0 else 0
        
    return stability_ratios

# 1. LOAD LOCAL DATA
records = []
print(f"📂 Processing HeAR embeddings from: {AUDIT_ROOT}")

for root, dirs, files in os.walk(AUDIT_ROOT):
    for f in files:
        if f.endswith('.npy'):
            path = os.path.join(root, f)
            speaker_id = f.split('_')[0] 
            try:
                emb = np.load(path).flatten()
                if len(emb) == 512:
                    records.append({'emb': emb, 'subject': speaker_id})
            except: continue

df = pd.DataFrame(records)
X_all = np.stack(df['emb'].values)
subjects = df['subject'].values

# 2. CALCULATE AND SORT STABILITY SPECTRUM
stability_scores = compute_stability_metric(X_all, subjects)
sorted_idx = np.argsort(stability_scores)[::-1]
sorted_scores = stability_scores[sorted_idx]

# 3. GENERATE AND SAVE FIGURE (1:1 Aspect Ratio)
# Using 8x8 for a clear square plot
plt.figure(figsize=(8, 8)) 
plt.style.use('default') 

# Match the colors from your reference image
top_n = 15 
colors = ['#f5a64a'] * top_n + ['#1f77b4'] * (512 - top_n)

plt.bar(range(512), sorted_scores, color=colors, width=1.0)

# Add the specific annotation from the image
# Adjusted text coordinates for the square layout
plt.annotate('Physiological candidates', 
             xy=(5, sorted_scores[0]), xytext=(80, sorted_scores[0] + 0.4),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=12, fontweight='bold')

plt.title("Stability spectrum of embedding dimensions", fontsize=16, pad=20)
plt.xlabel("Embedding Dimension Index (sorted by stability)", fontsize=12)
plt.ylabel("Score (Within-Subject Variance Ratio)", fontsize=12)

# Y-axis limit matching your audit framework
plt.ylim(0, 3.0)

# Final Polish: remove top/right spines for that clean "Nature" look
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# --- SAVE TO DISK ---
output_filename = "Figure_3a_Stability_1to1.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"✅ Success! Square figure saved to: {os.path.abspath(output_filename)}")
