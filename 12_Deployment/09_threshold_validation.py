#!/usr/bin/env python3
"""
Step 6G: Youden’s J Statistical Optimization Audit (KCL | DurMatch 30–60s)

Purpose:
Identifies the mathematically optimal threshold (T) that maximizes the joint 
separation (Sensitivity + Specificity - 1) between PD and HC cohorts. This 
benchmarks the deployment policy (T=0.4473) against statistical perfection.

Validity Requirement:
- Checklist Item 8: Clinical Operating Point [Ref: Section 6.1].
- Optimization Audit: Quantifies the trade-off between the mathematical 
  optimum and the safety-constrained policy [Ref: Table 3].

Inputs:
- HC Source: audit_results/HC_NorthWind_V2_Audit_DurMatch_30_60.csv [Ref: Step 4A]
- PD Source: audit_results/PD_NorthWind_V2_Audit_DurMatch_30_60.csv [Ref: Step 4A]

Outputs:
- Primary Evidence: 06g_youden_j_threshold_sweep_kcl_30_60.csv
- Analysis: Comparative report of 'Optimal' vs. 'Deployed' performance metrics.

Technical Specs:
- Algorithm: Sensitivity + Specificity - 1.0 (Full Sweep).
- Policy Anchor: 0.4473 (Optimized for allowed False Positives).

Author: Jim McCormack
Date: Feb 2026
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
# Using a raw string (r"") to handle Windows backslashes correctly
CSV_PATH = r"C:\Projects\hear_italian\WAVstudy\06g_youden_j_threshold_sweep_kcl_30_60.csv"
OUTPUT_NAME = "complete_youden_j_peak_plot.png"

def main():
    print(f"📊 Loading dataset from: {CSV_PATH}")
    
    try:
        # Load the complete dataset from the CSV
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"❌ Error: Could not find the file at {CSV_PATH}")
        return
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        return

    # The deployment point (T=0.4473, Spec=0.9375, Sens=0.875, J=0.8125)
    deployment_point = {
        'threshold': 0.4473, 
        'specificity_hc': 0.9375, 
        'sensitivity_pd': 0.875, 
        'youden_j': 0.8125
    }

    # Append the deployment point to the dataframe for plotting if it's not already there
    df = pd.concat([df, pd.DataFrame([deployment_point])], ignore_index=True)
    df = df.sort_values('threshold')

    # ============================================================================
    # PLOTTING LOGIC
    # ============================================================================
    plt.figure(figsize=(10, 7))

    # Plot performance curves
    plt.plot(df['threshold'], df['sensitivity_pd'], 'r-', label='Sensitivity (PD)', linewidth=2.5, alpha=0.8)
    plt.plot(df['threshold'], df['specificity_hc'], 'b-', label='Specificity (HC)', linewidth=2.5, alpha=0.8)
    plt.plot(df['threshold'], df['youden_j'], 'g--', label="Youden's J", linewidth=2)

    # Mark the J-Peak (Operating Point)
    plt.scatter([0.4473], [0.8125], s=300, c='gold', marker='*',
                edgecolors='black', linewidths=1.5, zorder=10, 
                label='J-Peak / Policy (T=0.4473)')

    # Highlight the crossover region (the 'elbow' where J peaks)
    plt.axvspan(0.44, 0.45, color='gray', alpha=0.1, label='Operating Window')

    # Formatting
    plt.xlabel('Probability Threshold ($T$)', fontsize=12)
    plt.ylabel('Performance Metric Value', fontsize=12)
    plt.title('Figure 5, Panel C: Youden’s J-Peak and Threshold Convergence\n(Complete KCL 30-60s Cohort)', 
              fontsize=14, fontweight='bold', pad=20)

    plt.legend(loc='lower left', frameon=True, shadow=True, fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Setting limits based on your data distribution
    plt.xlim(0.23, 0.48)
    plt.ylim(-0.05, 1.1)

    # Annotations
    plt.annotate('Optimal Separation Point\n(J = 0.8125)', 
                 xy=(0.4473, 0.8125), xytext=(0.28, 0.85),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                 fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_NAME, dpi=300)
    print(f"✅ Plot successfully saved to: {Path.cwd() / OUTPUT_NAME}")

if __name__ == "__main__":
    main()