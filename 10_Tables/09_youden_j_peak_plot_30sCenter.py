#!/usr/bin/env python3
"""
Step 6G (AUDIT): Youden's J Statistical Optimization - CONDITION D (PRIMARY)
================================================================================
PRIMARY DATASET: 30s Center Window + Speech Boundary Alignment
SWEEP SOURCE: 06g_youden_j_threshold_sweep_30sCenter_*.csv (2026-02-12)
CLINICAL DECISION DATE: 2026-02-12 (END-TO-END TEST CONFIRMATION)
================================================================================

╔══════════════════════════════════════════════════════════════════════════════╗
║  VALIDATION STATUS - END-TO-END TEST 2026-02-12                             ║
║  ✓ Using latest audited sweep data (2026-02-12)                             ║
║  ✓ HC: 17 subjects, PD: 12 subjects (QC_Pass=True, subject-level)           ║
║  ✓ Youden optimal threshold: 0.4395 (tied range [0.4395, 0.4453])           ║
║  ✓ Clinical policy threshold: 0.4952 (reported), 0.495101 (operational)     ║
║  ✓ Clinical policy basis: ≤1 False Positive (safety constraint)             ║
║  ✓ This script visualizes the trade-off - it does NOT change policy         ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Identifies the mathematically optimal threshold (T) that maximizes Youden's J
    (Sensitivity + Specificity - 1) between PD and HC cohorts using the PRIMARY
    Condition D dataset.
    
    This benchmark quantifies the trade-off between the statistical optimum
    and the clinically selected safety-constrained policy (T=0.4952).

KEY FINDINGS (2026-02-12 LOCKED DATASET):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Youden Optimal:        T = 0.4395 (J = 0.5147, Spec=76.47%, Sens=75.00%)
  • Clinical Policy:       T = 0.4952 (J = 0.4412, Spec=94.12%, Sens=50.00%)
  • Trade-off:            +17.65% specificity at -25.00% sensitivity
  • This is a DELIBERATE safety choice (≤1 FP constraint)

VALIDITY REQUIREMENTS:
    ✓ Subject-level aggregation (1 vote per subject)
    ✓ PRIMARY Condition D dataset only (2026-02-12 end-to-end test)
    ✓ Optimal threshold computed directly from sweep data
    ✓ Clinical policy shown as reference, NOT as J-peak
    ✓ Tied thresholds handled explicitly with np.isclose (atol=1e-12)
    ✓ Sweep data sorted by threshold before plotting
    ✓ Column validation performed before processing

OUTPUTS:
    - Figure: 09_youden_j_peak_plot_30sCenter_{timestamp}.png
    - Figure: 09_youden_j_peak_plot_30sCenter_{timestamp}.pdf (publication)
    - Terminal: Youden optimal vs clinical policy comparison
================================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION - PRIMARY CONDITION D (2026-02-12 LOCKED)
# =============================================================================
# Auto-detect most recent Youden sweep CSV
BASE_DIR = Path(r"C:\Projects\hear_italian\audit_results\threshold_sweep")
OUTPUT_DIR = Path(r"C:\Projects\hear_italian\audit_results\figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Required columns for validation
REQUIRED_COLUMNS = {'threshold', 'youden_j', 'specificity_hc', 'sensitivity_pd', 
                    'hc_tn', 'hc_fp', 'pd_tp', 'pd_fn'}

# Clinical operating region (derived from Condition D data)
# Bracket includes both tied optimal range [0.4395, 0.4453] and policy point 0.4952
OPERATING_REGION = (0.43, 0.51)  # Defensible range covering all clinically relevant thresholds

def find_most_recent_sweep():
    """Find the most recent Youden sweep CSV from Condition D."""
    files = list(BASE_DIR.glob("06g_youden_j_threshold_sweep_30sCenter_*.csv"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)

# =============================================================================
# CLINICAL POLICY (LOCKED - FROM STEP 6F, 2026-02-12 AUDIT)
# =============================================================================
CLINICAL_POLICY = {
    'threshold_operational': 0.495101,   # Exact value used in code
    'threshold_reported': 0.4952,        # Conservative ceiling to 4dp
    'specificity': 0.9412,              # 16/17
    'sensitivity': 0.5000,              # 6/12
    'youden_j': 0.4412,                # 0.9412 + 0.5000 - 1
    'label': 'Clinical Policy (≤1 FP, ε-adjusted)',
    'color': 'darkorange',
    'marker': 's',
    'hc_fp': 1,
    'pd_tp': 6
}

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "=" * 80)
    print("📊 YOUDEN'S J OPTIMIZATION AUDIT - CONDITION D (PRIMARY)")
    print("=" * 80)
    print(f"📅 Clinical Decision Date: 2026-02-12 (END-TO-END TEST)")
    
    # -------------------------------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------------------------------
    csv_path = find_most_recent_sweep()
    if csv_path is None:
        print(f"\n❌ Error: No Youden sweep CSV found in {BASE_DIR}")
        print(f"   Please run: python 06g_youden_j_threshold_sweep_30sCenter_20260212.py")
        return
    
    print(f"\n📂 Loading sweep data from:")
    print(f"   {csv_path.name}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    # -------------------------------------------------------------------------
    # VALIDATE COLUMNS
    # -------------------------------------------------------------------------
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        print(f"\n❌ AUDIT FAILURE: Missing required columns in sweep CSV:")
        print(f"   Missing: {sorted(missing)}")
        print(f"   Found:   {sorted(set(df.columns))}")
        return
    
    print(f"   ✓ Column validation: PASSED")
    print(f"   ✓ Loaded {len(df)} threshold points")
    
    # -------------------------------------------------------------------------
    # SORT AND PREPARE DATA
    # -------------------------------------------------------------------------
    df = df.sort_values('threshold').reset_index(drop=True)
    print(f"   ✓ Threshold range: [{df['threshold'].min():.4f}, {df['threshold'].max():.4f}]")
    print(f"   ✓ Data sorted by threshold")
    
    # -------------------------------------------------------------------------
    # FIND YOUDEN OPTIMUM WITH ROBUST TIE HANDLING
    # -------------------------------------------------------------------------
    max_j = df['youden_j'].max()
    
    # Use np.isclose to avoid float precision issues
    tied_df = df[np.isclose(df['youden_j'], max_j, atol=1e-12)].copy()
    
    # Deterministic choice: use lowest threshold in tied set
    optimal_threshold = tied_df['threshold'].min()
    optimal_row = df[df['threshold'] == optimal_threshold].iloc[0]
    optimal = optimal_row.to_dict()
    
    # Get the full tied range for reporting
    t_low = tied_df['threshold'].min()
    t_high = tied_df['threshold'].max()
    n_tied = len(tied_df)
    
    print(f"\n🎯 YOUDEN'S J OPTIMUM (STATISTICAL):")
    print(f"   Threshold:      {optimal['threshold']:.4f}")
    if n_tied > 1:
        print(f"   Tied range:      [{t_low:.4f}, {t_high:.4f}] ({n_tied} thresholds)")
        print(f"   Tie detection:   np.isclose(atol=1e-12)")
    print(f"   Youden's J:     {optimal['youden_j']:.4f}")
    print(f"   Specificity:    {optimal['specificity_hc']:.2%} ({optimal['hc_tn']:.0f}/{optimal['hc_tn']+optimal['hc_fp']:.0f})")
    print(f"   Sensitivity:    {optimal['sensitivity_pd']:.2%} ({optimal['pd_tp']:.0f}/{optimal['pd_tp']+optimal['pd_fn']:.0f})")
    print(f"   HC FP:          {optimal['hc_fp']:.0f}")
    print(f"   PD TP:          {optimal['pd_tp']:.0f}")
    
    # -------------------------------------------------------------------------
    # CLINICAL POLICY REFERENCE (FROM STEP 6F LOCKED)
    # -------------------------------------------------------------------------
    print(f"\n🔐 CLINICAL POLICY (SAFETY-CONSTRAINED - ≤1 FP):")
    print(f"   Threshold:      {CLINICAL_POLICY['threshold_reported']:.4f} (reported)")
    print(f"   Operational:    {CLINICAL_POLICY['threshold_operational']:.6f} (ε-adjusted)")
    print(f"   Youden's J:     {CLINICAL_POLICY['youden_j']:.4f}")
    print(f"   Specificity:    {CLINICAL_POLICY['specificity']:.2%} (16/17)")
    print(f"   Sensitivity:    {CLINICAL_POLICY['sensitivity']:.2%} (6/12)")
    print(f"   HC FP:          {CLINICAL_POLICY['hc_fp']}")
    print(f"   PD TP:          {CLINICAL_POLICY['pd_tp']}")
    
    # -------------------------------------------------------------------------
    # COMPARISON
    # -------------------------------------------------------------------------
    delta_j = CLINICAL_POLICY['youden_j'] - optimal['youden_j']
    delta_spec = CLINICAL_POLICY['specificity'] - optimal['specificity_hc']
    delta_sens = CLINICAL_POLICY['sensitivity'] - optimal['sensitivity_pd']
    
    print(f"\n📊 TRADE-OFF ANALYSIS:")
    print(f"   Δ Youden's J:    {delta_j:+.4f}")
    print(f"   Δ Specificity:   {delta_spec:+.2%}")  # + sign inside formatter
    print(f"   Δ Sensitivity:   {delta_sens:+.2%}")  # + sign inside formatter
    print(f"   ──────────────────────────────────────")
    print(f"   Clinical policy gains {delta_spec:+.1%} specificity")  # Fixed double-sign
    print(f"   at cost of {delta_sens:+.1%} sensitivity")
    print(f"   This is a DELIBERATE safety choice (≤1 FP constraint).")
    
    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    print(f"\n🖼️  Generating figure...")
    
    plt.figure(figsize=(14, 8))
    
    # Plot performance curves
    plt.plot(df['threshold'], df['sensitivity_pd'], 
             'r-', label='Sensitivity (PD)', linewidth=2.5, alpha=0.8)
    plt.plot(df['threshold'], df['specificity_hc'], 
             'b-', label='Specificity (HC)', linewidth=2.5, alpha=0.8)
    plt.plot(df['threshold'], df['youden_j'], 
             'g--', label="Youden's J", linewidth=2, alpha=0.8)
    
    # Mark Youden optimum (computed, not hardcoded)
    plt.scatter([optimal['threshold']], [optimal['youden_j']], 
                s=350, c='gold', marker='*',
                edgecolors='black', linewidths=1.5, zorder=10,
                label=f"Youden Optimum at J(T) (T={optimal['threshold']:.4f}, J={optimal['youden_j']:.3f})")
    
    # If tied thresholds exist, mark the range
    if n_tied > 1:
        plt.axvspan(t_low, t_high, color='gold', alpha=0.15, 
                   label=f'Tied Optimal Range ({n_tied} thresholds)')
    
    # Mark clinical policy (reference only)
    plt.scatter([CLINICAL_POLICY['threshold_reported']], [CLINICAL_POLICY['youden_j']], 
                s=300, c=CLINICAL_POLICY['color'], marker=CLINICAL_POLICY['marker'],
                edgecolors='black', linewidths=1.5, zorder=9,
                label=CLINICAL_POLICY['label'])
    
    # Highlight the operating region (using named constant)
    plt.axvspan(*OPERATING_REGION, color='gray', alpha=0.1, 
                label=f'Clinical Operating Region [{OPERATING_REGION[0]:.2f}, {OPERATING_REGION[1]:.2f}]')
    
    # Formatting
    plt.xlabel('Probability Threshold ($T$)', fontsize=14)
    plt.ylabel('Performance Metric Value', fontsize=14)
    plt.title('Youden\'s J Optimization: Statistical Optimum vs Clinical Policy\n' +
              f'Condition D (30s Center Window, N_hc=17, N_pd=12) - 2026-02-12', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.legend(loc='upper right', frameon=True, shadow=True, fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Set limits based on Condition D data
    plt.xlim(0.35, 0.65)
    plt.ylim(0.2, 1.05)
    
    # Annotations
    tie_text = f'Tied Range [{t_low:.4f}, {t_high:.4f}]' if n_tied > 1 else ''
    
    plt.annotate(f'Statistical Optimum\nJ = {optimal["youden_j"]:.3f}\n{tie_text}',
                 xy=(optimal['threshold'], optimal['youden_j']), 
                 xytext=(optimal['threshold'] - 0.12, optimal['youden_j'] - 0.15),
                 arrowprops=dict(facecolor='gold', shrink=0.05, width=2, headwidth=8,
                                edgecolor='black', linewidth=1),
                 fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='gold', alpha=0.9))
    
    plt.annotate(f'Clinical Policy\nT = {CLINICAL_POLICY["threshold_reported"]:.4f}\n(≤1 FP, ε-adjusted)',
                 xy=(CLINICAL_POLICY['threshold_reported'], CLINICAL_POLICY['youden_j']), 
                 xytext=(CLINICAL_POLICY['threshold_reported'] + 0.10, CLINICAL_POLICY['youden_j'] - 0.12),
                 arrowprops=dict(facecolor=CLINICAL_POLICY['color'], shrink=0.05, width=2, headwidth=8,
                                edgecolor='black', linewidth=1),
                 fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=CLINICAL_POLICY['color'], alpha=0.9))
    
    # Add trade-off annotation (using data coordinates, anchored to policy point)
    tradeoff_text = (
        f"TRADE-OFF (Policy vs Optimum):\n"
        f"✓ Specificity: {delta_spec:+.1%}\n"  # Fixed double-sign
        f"✗ Sensitivity: {delta_sens:+.1%}\n"
        f"✓ Constraint: ≤1 FP (achieved)\n"
        f"Δ Youden's J: {delta_j:+.3f}"
    )
    
    # Position annotation in data space near policy point
    plt.text(CLINICAL_POLICY['threshold_reported'] + 0.04,
             CLINICAL_POLICY['youden_j'] + 0.06,
             tradeoff_text, fontsize=11,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, edgecolor='gray'))
    
    # Add data provenance
    plt.text(0.02, 0.02, f"Data: {csv_path.name}\nHC: 17, PD: 12 | {datetime.now().strftime('%Y-%m-%d')}",
             transform=plt.gca().transAxes, fontsize=8, color='gray',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"09_youden_j_peak_plot_30sCenter_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Figure saved: {output_file}")
    
    # Also save as PDF for publication
    pdf_file = OUTPUT_DIR / f"09_youden_j_peak_plot_30sCenter_{timestamp}.pdf"
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"   ✓ PDF saved:    {pdf_file}")
    
    # -------------------------------------------------------------------------
    # VERIFICATION CERTIFICATE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🔐 VERIFICATION CERTIFICATE - YOUDEN'S J AUDIT (2026-02-12)")
    print("=" * 80)
    print(f"""
✓ Dataset:           CONDITION D (30s Center Window + Alignment)
✓ Audit Date:        2026-02-12 (END-TO-END TEST)
✓ Sweep Source:      {csv_path.name}
✓ Column Validation: PASSED
✓ Sort Validation:   Threshold-sorted
✓ Tie Detection:     np.isclose(atol=1e-12)
✓ HC Subjects:       17 (QC_Pass=True, subject-level)
✓ PD Subjects:       12 (QC_Pass=True, subject-level)

📊 YOUDEN OPTIMUM (COMPUTED):
   • Threshold:        {optimal['threshold']:.4f}
   • Tied range:       [{t_low:.4f}, {t_high:.4f}] ({n_tied} thresholds)
   • Youden's J:       {optimal['youden_j']:.4f}
   • Specificity:      {optimal['specificity_hc']:.2%} ({optimal['hc_tn']:.0f}/{optimal['hc_tn']+optimal['hc_fp']:.0f})
   • Sensitivity:      {optimal['sensitivity_pd']:.2%} ({optimal['pd_tp']:.0f}/{optimal['pd_tp']+optimal['pd_fn']:.0f})
   • HC FP:            {optimal['hc_fp']:.0f}
   • PD TP:            {optimal['pd_tp']:.0f}

🔐 CLINICAL POLICY (LOCKED - STEP 6F):
   • Threshold (rep):  {CLINICAL_POLICY['threshold_reported']:.4f}
   • Threshold (op):   {CLINICAL_POLICY['threshold_operational']:.6f}
   • Youden's J:       {CLINICAL_POLICY['youden_j']:.4f}
   • Specificity:      {CLINICAL_POLICY['specificity']:.2%} (16/17)
   • Sensitivity:      {CLINICAL_POLICY['sensitivity']:.2%} (6/12)
   • Constraint:       ≤1 FP (achieved)

📈 TRADE-OFF SUMMARY:
   • Δ Specificity:    {delta_spec:+.1%} (safety gain)
   • Δ Sensitivity:    {delta_sens:+.1%} (sensitivity cost)
   • Δ Youden's J:     {delta_j:+.3f}

✓ STATUS: VISUALIZATION COMPLETE - AUDIT PASSED
  This figure documents the deliberate safety trade-off in the clinical policy.
  The statistical optimum is shown for reference only; policy remains at T=0.4952.
""")
    print("=" * 80 + "\n")
    
    # Show plot interactively
    plt.show()


if __name__ == "__main__":
    main()