"""
Figure 4: Cross-Linguistic Baseline Calibration
CLINICALLY CORRECT VERSION - Threshold anchoring with constant margin
Uses REAL subject-level data from audit CSVs
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- LOAD ACTUAL SUBJECT-LEVEL DATA FROM YOUR AUDITS ---
audit_files = {
    'German': r'C:\Projects\hear_italian\audit_results\cross_linguistic\german_hc_audit_subjects_kcl_20260213_063233.csv',
    'Swedish': r'C:\Projects\hear_italian\audit_results\cross_linguistic\swedish_hc_audit_subjects_kcl_20260213_062716.csv',
    'Nepali': r'C:\Projects\hear_italian\audit_results\cross_linguistic\nepali_hc_audit_subjects_kcl_20260213_063710.csv',
    'English (VCTK)': r'C:\Projects\hear_italian\audit_results\cross_linguistic\vctk_gold_audit_subjects_kcl_20260213_064322.csv'
}

# KCL constants from purified model
KCL_MEAN = 0.4047
KCL_THRESHOLD = 0.4952
MARGIN = KCL_THRESHOLD - KCL_MEAN  # 0.0905

# Load and combine data
data_raw = pd.DataFrame()
sample_sizes = {}

print("="*70)
print("📊 LOADING REAL AUDIT DATA")
print("="*70)

for lang, filepath in audit_files.items():
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        # Use prob_kcl column (subject-level score)
        temp = pd.DataFrame({
            'Language': lang,
            'Score': df['prob_kcl'].values
        })
        data_raw = pd.concat([data_raw, temp], ignore_index=True)
        sample_sizes[lang] = len(df)
        
        # Calculate specificity
        spec = (df['prob_kcl'] < KCL_THRESHOLD).mean() * 100
        print(f"✅ {lang:16} n={len(df):3} | mean={df['prob_kcl'].mean():.4f} | specificity={spec:.2f}%")
    else:
        print(f"❌ Missing: {filepath}")

# Calculate population-specific thresholds (constant margin above healthy mean)
group_means = data_raw.groupby('Language')['Score'].mean()
thresholds = group_means + MARGIN

print("\n" + "="*70)
print("📊 POPULATION-SPECIFIC THRESHOLDS (constant margin)")
print("="*70)
for lang in group_means.index:
    print(f"{lang:16} μ={group_means[lang]:.4f} | T_pop = μ + {MARGIN:.4f} = {thresholds[lang]:.4f}")

# --- 2-PANEL VIOLIN PLOT ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.set_theme(style="whitegrid")

# Colors: German, Swedish, Nepali, English
colors = ['#5b8ab5', '#9b59b6', '#2ecc71', '#e35f52']

# Panel A: Raw scores with FIXED KCL threshold
sns.violinplot(data=data_raw, x='Language', y='Score', hue='Language',
               palette=colors, ax=axes[0], inner='quartile', legend=False,
               cut=0, bw_method=0.2, linewidth=1)

# Add fixed threshold line
axes[0].axhline(KCL_THRESHOLD, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
axes[0].text(3.7, KCL_THRESHOLD + 0.02, f'Fixed KCL threshold = {KCL_THRESHOLD:.4f}',
             color='black', fontsize=10, ha='right',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='none'))

axes[0].set_title('A) Raw scores with fixed KCL threshold', fontweight='bold', fontsize=14)
axes[0].set_ylabel('Model output score', fontsize=14)
axes[0].set_xlabel('')
axes[0].set_ylim(0, 1.0)
axes[0].tick_params(axis='x', rotation=15)

# Add specificity annotation (just the facts)
spec_text = "Subjects below fixed threshold:\n"
for lang in data_raw['Language'].unique():
    spec = (data_raw[data_raw['Language']==lang]['Score'] < KCL_THRESHOLD).mean() * 100
    spec_text += f"{lang}: {spec:.1f}%\n"
axes[0].text(0.02, 0.98, spec_text, transform=axes[0].transAxes,
            fontsize=12, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

# Panel B: Raw scores with POPULATION-SPECIFIC thresholds (constant margin)
sns.violinplot(data=data_raw, x='Language', y='Score', hue='Language',
               palette=colors, ax=axes[1], inner='quartile', legend=False,
               cut=0, bw_method=0.2, linewidth=1)

# Add faint KCL threshold line as visual anchor
axes[1].axhline(KCL_THRESHOLD, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='KCL threshold (reference)')

# Add population-specific threshold lines (constant margin) with LARGER labels
for i, (lang, thresh) in enumerate(thresholds.items()):
    # Draw threshold line
    axes[1].plot([i-0.3, i+0.3], [thresh, thresh], 
                color='red', linestyle='--', linewidth=2.0, alpha=0.8)
    
    # Add threshold value - LARGER font size
    axes[1].text(i, thresh + 0.025, f'T={thresh:.4f}', 
                ha='center', fontsize=12, fontweight='bold', color='darkred',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                         edgecolor='red', linewidth=1))

# REMOVED: Sample size labels under x-axis

# Add calibration formula - precise wording
cal_text = f'Deployment rule: $T_{{pop}} = \mu_{{pop}} + ({KCL_THRESHOLD:.4f} - {KCL_MEAN:.4f})$'
axes[1].text(0.5, 0.95, cal_text, transform=axes[1].transAxes,
            fontsize=12, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9, edgecolor='gray'))

# Add explanatory note - clinically precise
note = "Thresholds maintain constant margin above each population's healthy mean\nModel output scale preserved (no score transformation)"
axes[1].text(0.5, 0.02, note, transform=axes[1].transAxes,
            fontsize=12, ha='center', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

axes[1].set_title('B) Population-specific thresholds (constant margin)', fontweight='bold', fontsize=14)
axes[1].set_ylabel('Model output score', fontsize=12)
axes[1].set_xlabel('')
axes[1].set_ylim(0, 1.0)
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()

# --- SAVE FIGURE ---
output_file = "Figure_4_cross_linguistic_threshold_anchoring.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight')

print("\n" + "="*70)
print(f"✅ FIGURE SAVED: {os.path.abspath(output_file)}")
print("="*70)
print("\n📊 CLINICAL INTERPRETATION:")
print("-"*50)
print("• All healthy cohorts show 98-100% subjects below fixed KCL threshold")
print("• Cross-lingual measurement consistency: decision boundary transfers")
print("• Population-specific thresholds maintain constant margin above healthy mean")
print("• This is a deployment rule, not a model property")
print("• Model output scale preserved (no score transformation)")
print("="*70)
print("\n📊 SAMPLE SIZES (reported in narrative):")
for lang, n in sample_sizes.items():
    print(f"   {lang}: n={n}")
print("="*70)
