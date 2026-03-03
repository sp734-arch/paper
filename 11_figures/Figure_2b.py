#!/usr/bin/env python3
"""
Generate Figure 2b: Correlation Structure Comparison for Italian, English, KCL Data Sets 50% overlap

Demonstrates dimensional parsimony achieved through the auditing pipeline by
comparing correlation structure between:
- Full 512-D HeAR embedding (redundant)
- Purified 7-D audited subset (independent)

Method:
- Compute absolute Pearson correlation matrix for all dimensions

Key Finding:
- Full 512×512: Dense correlation structure (many redundant dimensions)
- Purified 7×7: Sparse structure (low off-diagonal correlation)

This validates Step 6 (Independence Audit) which filtered for low correlation on ES and EN dataset.

Expected Output:
- figure_2b.png
- Side-by-side heatmaps showing parsimony

Paper Reference: Figure 2
Author: Jim McCormack
Date: Feb 2026
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
AUDIT_ROOT = "../features_audit/density_50" 
# The Purified V2 Subset indices verified in your draft
FINAL_7 = [38, 43, 146, 204, 267, 346, 419] 

# 1. DATA LOADING (Reused from your bar chart/stability script)
records = []
print(f"📂 Loading embeddings for Correlation Audit from: {AUDIT_ROOT}")

for root, dirs, files in os.walk(AUDIT_ROOT):
    for f in files:
        if f.endswith('.npy'):
            path = os.path.join(root, f)
            try:
                emb = np.load(path).flatten()
                if len(emb) == 512:
                    records.append({'emb': emb})
            except: continue

if not records:
    print("❌ Error: No .npy files found. Check your AUDIT_ROOT path.")
    exit()

# Define X_all globally so the plot function can see it
X_all = np.stack([r['emb'] for r in records])
print(f"✅ Loaded matrix shape: {X_all.shape}")

# 2. PLOTTING FUNCTION
def plot_correlation_structure(embeddings, indices):
    """
    Generates Figure 3c: Correlation Structure.
    Compares full 512D redundancy vs 7D independence.
    """
    # Calculate Pearson's r
    corr_full = pd.DataFrame(embeddings).corr()
    corr_purified = pd.DataFrame(embeddings[:, indices]).corr()
    
    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])
    
    # Left: Full Space
    ax1 = fig.add_subplot(gs[0])
    sns.heatmap(corr_full, cmap='RdBu_r', center=0, vmin=-1, vmax=1, 
                ax=ax1, cbar=False, xticklabels=False, yticklabels=False)
    ax1.set_title("Full HeAR Embedding Space (512D)\n(High Inter-dimensional Redundancy)", fontsize=13)
    
    # Right: Purified Subset
    ax2 = fig.add_subplot(gs[1])
    # Annotation proves the 'low inter-dimensional correlation' claim
    sns.heatmap(corr_purified, cmap='RdBu_r', center=0, vmin=-1, vmax=1, 
                ax=ax2, annot=True, fmt=".2f", square=True,
                xticklabels=[f"D{i}" for i in indices],
                yticklabels=[f"D{i}" for i in indices],
                cbar_kws={'label': "Pearson's r"})
    
    ax2.set_title("Purified Stable Subset (7D)\n(Independent Measurement Axes)", fontsize=13)
    
    plt.tight_layout()
    output_path = "Figure_2b.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"🚀 Figure 2b successfully saved to: {os.path.abspath(output_path)}")

# 3. EXECUTION
if __name__ == "__main__":
    plot_correlation_structure(X_all, FINAL_7)
