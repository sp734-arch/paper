#!/usr/bin/env python
"""
Identity Leakage test
STEP 2: IDENTITY VS DISEASE EXPERIMENTS
================================================================================
PURPOSE:
    Runs two core experiments on certified slice-level features from Step 1:
    1. PD Classification - Leave-One-Subject-Out (LOSO) logistic regression
    2. Speaker Identification - Closed-set per-subject train/test split
    
    Validates that the 7D audited subspace preserves disease signal while
    suppressing speaker identity.

DEPENDENCY CHAIN (CRITICAL FOR REPRODUCIBILITY):
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║  THIS SCRIPT DEPENDS ON STEP 1 OUTPUTS                                   ║
    ║  Requires embeddings and metadata from:                                  ║
    ║    11_certified_extractor_pytorch.py                                    ║
    ║                                                                          ║
    ║  Outputs are used by:                                                    ║
    ║    - table_1_identity_leakage.py (Speaker ID results)                   ║
    ║    - table_2_primary_results.py (PD classification results)             ║
    ║    - 08_calculate_cohens_d.py (Effect size)                             ║
    ║    - 09_drop_top_robustness.py (Outlier analysis)                       ║
    ╚══════════════════════════════════════════════════════════════════════════╝

DATA FLOW:
    Step 1 outputs (features_certified/) ─────────────────────┐
        ↓                                                     │
    Load metadata and embeddings ─────────────────────────────┤
        ↓                                                     │
    PD LOSO (run_pd_loso_subject_level) ──────────────────────┼───┐
        ↓                                                     │   │
    Speaker ID (run_speaker_id_closed_set) ───────────────────┼───┤
        ↓                                                     │   │
    Save results JSON with cryptographic hash ────────────────┘   │
        ↓                                                         │
    Downstream analysis scripts ──────────────────────────────────┘

USAGE:
    # Full cross-lingual analysis (all cohorts)
    python 11_step2_experiments.py --run_name complete_analysis
    
    # KCL Condition D only (30s center segments, -55 dBFS threshold)
    python 11_step2_experiments.py --kcl_30s --run_name kcl_30s_analysis
    
    # Custom features directory or output location
    python 11_step2_experiments.py --features_dir /path/to/features --output_dir /path/to/output

INPUT FILES (from Step 1):
    Format: {prefix}_embeddings_512_raw.npy
           {prefix}_embeddings_7d_scaled.npy
           {prefix}_metadata.csv
    
    Default prefixes:
    - Without --kcl_30s: cross_lingual_real
    - With --kcl_30s:    kcl_condition_d_30s

OUTPUT FILES:
    {run_name}_results.json  # Full results with all metrics
    {run_name}_hash.txt      # SHA256 hash for verification

EXPERIMENT DETAILS:

    1. PD CLASSIFICATION (Subject-Level LOSO):
       - Uses 7D scaled embeddings
       - Leave-One-Subject-Out cross-validation
       - Logistic regression with class_weight='balanced'
       - Aggregates slice predictions to subject level (mean)
       - Reports: AUROC, accuracy, balanced accuracy, confusion matrix
       
    2. SPEAKER IDENTIFICATION (Closed Set):
       - Tests both 512D raw and 7D scaled embeddings
       - Each subject contributes to both train and test
       - 70/30 split by recording (ensures >=1 recording in each)
       - Logistic regression (multinomial)
       - Reports: macro accuracy (per-subject average)

NOTES:
    - In KCL 30s mode (--kcl_30s), Speaker ID is automatically skipped
      because Condition D has only one recording per subject
    - Results include cryptographic hash for integrity verification
    - All random processes use fixed seed (42) for reproducibility
    - Placeholder rows (slice_index = -1) are automatically removed
      from metadata only (embeddings are already correct length)

VERSION HISTORY:
    v1.0 (2026-02-12): Initial release
    v1.1 (2026-02-14): Added placeholder removal
    v1.2 (2026-02-16): Fixed Unicode encoding issues
    v1.3 (2026-02-24): Added Speaker ID skip for KCL 30s mode
    v1.4 (2026-02-24): Final audit version with full documentation
    v1.5 (2026-02-25): Fixed placeholder removal logic - metadata only
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
import argparse
import hashlib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ===== HELPER FUNCTION FOR JSON SERIALIZATION =====
def convert_numpy(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy(item) for item in obj)
    else:
        return obj


class Step2Experiments:
    """
    Run identity vs disease experiments on certified slice-level features.
    - PD LOSO: subject-level metrics (correct)
    - Speaker ID: closed-set with per-subject train/test split
    """
    
    def __init__(self, 
                 features_dir=r'C:\Projects\hear_italian\features_certified',
                 output_dir=r'C:\Projects\hear_italian\step2_results',
                 run_name='step2_results',
                 kcl_30s_mode=False):
        
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.kcl_30s_mode = kcl_30s_mode
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*70)
        print("STEP 2: IDENTITY VS DISEASE EXPERIMENTS")
        print("="*70)
        print(f"Features directory: {self.features_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Run name: {run_name}")
        print(f"KCL 30s Mode: {kcl_30s_mode}")
    
    def load_data(self, prefix='cross_lingual_real'):
        """
        Load embeddings and metadata from Step 1.
        In KCL 30s mode, uses the Condition D dataset.
        Handles placeholder rows (slice_index = -1) automatically.
        """
        
        # Override prefix if in KCL 30s mode
        if self.kcl_30s_mode:
            prefix = 'kcl_condition_d_30s'
            print(f"\n     KCL 30s MODE ACTIVE - Using Condition D dataset")
        
        print(f"\n     Loading data with prefix: {prefix}")
        
        # Load metadata
        metadata_path = self.features_dir / f'{prefix}_metadata.csv'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        self.metadata = pd.read_csv(metadata_path)
        print(f"   Metadata: {len(self.metadata)} slices")
        
        # Load embeddings
        self.embeddings_512_raw = np.load(self.features_dir / f'{prefix}_embeddings_512_raw.npy')
        self.embeddings_7d_scaled = np.load(self.features_dir / f'{prefix}_embeddings_7d_scaled.npy')
        
        print(f"   512D embeddings: {self.embeddings_512_raw.shape}")
        print(f"   7D scaled embeddings: {self.embeddings_7d_scaled.shape}")
        
        # Check for and handle misalignment due to placeholder rows
        if len(self.metadata) != len(self.embeddings_512_raw):
            print(f"\n   WARNING: Metadata/embedding mismatch detected:")
            print(f"      Metadata: {len(self.metadata)} rows")
            print(f"      Embeddings: {len(self.embeddings_512_raw)} rows")
            
            if 'slice_index' in self.metadata.columns:
                placeholder_mask = self.metadata['slice_index'] == -1
                n_placeholders = placeholder_mask.sum()
                
                if n_placeholders > 0:
                    print(f"   Found {n_placeholders} placeholder rows (slice_index = -1)")
                    
                    # Get subject IDs with placeholders for logging
                    placeholder_subjects = self.metadata.loc[placeholder_mask, 'subject_id'].unique()
                    if len(placeholder_subjects) <= 10:
                        print(f"   Subjects with placeholders: {list(placeholder_subjects)}")
                    else:
                        print(f"   Subjects with placeholders: {len(placeholder_subjects)} total")
                    
                    # CRITICAL FIX: Only apply to metadata, not to embeddings
                    # Embeddings are already the correct length (Step 1 saved without placeholders)
                    valid_indices = ~placeholder_mask
                    self.metadata = self.metadata[valid_indices].reset_index(drop=True)
                    
                    print(f"   After metadata cleanup: {len(self.metadata)} rows")
                else:
                    raise AssertionError(
                        f"Metadata/embedding mismatch but no placeholder rows found. "
                        f"Meta={len(self.metadata)}, Embed={len(self.embeddings_512_raw)}"
                    )
        
        # Verify alignment
        assert len(self.metadata) == len(self.embeddings_512_raw) == len(self.embeddings_7d_scaled), \
            f"Metadata and embeddings misaligned: Meta={len(self.metadata)}, 512={len(self.embeddings_512_raw)}, 7d={len(self.embeddings_7d_scaled)}"
        
        # Add disease label
        self.metadata['label'] = (self.metadata['disease'] == 'PD').astype(int)
        
        print(f"   HC slices: {(self.metadata['label']==0).sum()}")
        print(f"   PD slices: {(self.metadata['label']==1).sum()}")
        
        # Verify no subject has mixed labels
        subj_label_check = self.metadata.groupby('subject_key')['label'].nunique()
        bad_subjects = subj_label_check[subj_label_check > 1]
        if len(bad_subjects) > 0:
            raise ValueError(f"Found subjects with mixed disease labels: {list(bad_subjects.index)[:5]}")
        print(f"   All subjects have consistent disease labels")
        
        return True
    
    def run_pd_loso_subject_level(self, cohort=None):
        """
        Task 1: PD Classification using Leave-One-Subject-Out.
        Evaluates at SUBJECT LEVEL (correct LOSO metric).
        Uses 7D scaled embeddings.
        """
        print("\n" + "="*70)
        print("     PD CLASSIFICATION - SUBJECT-LEVEL LOSO")
        print("="*70)
        
        # Filter by cohort if specified
        if cohort:
            mask = self.metadata['cohort'] == cohort
            metadata = self.metadata[mask].copy()
            X = self.embeddings_7d_scaled[mask]
            print(f"\nCohort: {cohort}")
        else:
            metadata = self.metadata.copy()
            X = self.embeddings_7d_scaled
            print(f"\nALL COHORTS COMBINED (warning: potential language confound)")
        
        y = metadata['label'].values
        groups = metadata['subject_key'].values
        
        # Subject-level stats
        unique_subjects = np.unique(groups)
        subjects_hc = metadata[metadata['label']==0]['subject_key'].nunique()
        subjects_pd = metadata[metadata['label']==1]['subject_key'].nunique()
        
        print(f"\nTotal slices: {len(X)}")
        print(f"Total subjects: {len(unique_subjects)} (HC:{subjects_hc}, PD:{subjects_pd})")
        
        if subjects_hc == 0 or subjects_pd == 0:
            print(f"    Skipping {cohort if cohort else 'combined'} - only one class present")
            return None
        
        # LOSO cross-validation
        logo = LeaveOneGroupOut()
        
        subject_true = []      # True label for each subject
        subject_score = []      # Mean predicted probability for each subject
        subject_names = []      # Subject keys
        fold_details = []
        
        for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
            test_subject = groups[test_idx[0]]
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # CRITICAL: Train must have both classes
            if len(np.unique(y_train)) < 2:
                print(f"   Fold {fold}: Skipping {test_subject} - train set has only one class")
                continue
            
            # Train classifier
            clf = LogisticRegression(max_iter=1000, class_weight='balanced')
            clf.fit(X_train, y_train)
            
            # Predict on all slices of left-out subject
            p_slices = clf.predict_proba(X_test)[:, 1]
            
            # Aggregate to subject level (mean)
            p_subject = float(np.mean(p_slices))
            y_subject = int(y_test[0])  # All slices same label
            
            subject_true.append(y_subject)
            subject_score.append(p_subject)
            subject_names.append(test_subject)
            
            fold_details.append({
                'fold': fold,
                'test_subject': test_subject,
                'true_label': y_subject,
                'pred_score': p_subject,
                'n_slices': len(X_test),
                'n_train_subjects': len(np.unique(groups[train_idx])),
                'n_train_hc': int(sum(y_train==0)),
                'n_train_pd': int(sum(y_train==1))
            })
            
            if fold % 10 == 0:
                print(f"   Fold {fold}: {test_subject} (true={y_subject}, pred={p_subject:.3f})")
        
        # Convert to numpy arrays
        subject_true = np.array(subject_true)
        subject_score = np.array(subject_score)
        
        # Subject-level metrics
        if len(np.unique(subject_true)) < 2:
            print("    Not enough class variety across subjects for AUROC")
            auroc = float('nan')
        else:
            auroc = roc_auc_score(subject_true, subject_score)
        
        # Convert scores to binary predictions (threshold 0.5)
        subject_pred = (subject_score >= 0.5).astype(int)
        
        acc = accuracy_score(subject_true, subject_pred)
        bal_acc = balanced_accuracy_score(subject_true, subject_pred)
        
        # Confusion matrix manually
        tp = int(sum((subject_true == 1) & (subject_pred == 1)))
        tn = int(sum((subject_true == 0) & (subject_pred == 0)))
        fp = int(sum((subject_true == 0) & (subject_pred == 1)))
        fn = int(sum((subject_true == 1) & (subject_pred == 0)))
        
        print("\n" + "="*70)
        print("     SUBJECT-LEVEL LOSO RESULTS")
        print("="*70)
        print(f"Subjects evaluated: {len(subject_true)}")
        print(f"AUROC: {auroc:.3f}")
        print(f"Accuracy: {acc:.3f}")
        print(f"Balanced accuracy: {bal_acc:.3f}")
        print("\nConfusion Matrix:")
        print(f"               Predicted")
        print(f"              HC    PD")
        print(f"Actual HC    {tn:3d}   {fp:3d}")
        print(f"       PD    {fn:3d}   {tp:3d}")
        
        # Sensitivity/Specificity
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
        print(f"\nSensitivity: {sensitivity:.3f}")
        print(f"Specificity: {specificity:.3f}")
        
        return {
            'task': 'pd_loso_subject_level',
            'cohort': cohort,
            'n_subjects_evaluated': len(subject_true),
            'n_hc': int(sum(subject_true==0)),
            'n_pd': int(sum(subject_true==1)),
            'auroc': float(auroc) if auroc == auroc else None,
            'accuracy': float(acc),
            'balanced_accuracy': float(bal_acc),
            'sensitivity': float(sensitivity) if sensitivity == sensitivity else None,
            'specificity': float(specificity) if specificity == specificity else None,
            'confusion_matrix': {
                'tn': tn, 'fp': fp,
                'fn': fn, 'tp': tp
            },
            'subject_results': fold_details
        }
    
    def run_speaker_id_closed_set(self, cohort, embedding_type='512_raw', min_recordings=2):
        """
        Task 2: Speaker Identification - CLOSED SET.
        Every subject in test also appears in train.
        Splits by recording_key, but ensures per-subject train/test.
        
        embedding_type: '512_raw' or '7d_scaled'
        """
        print("\n" + "="*70)
        print(f"SPEAKER IDENTIFICATION (CLOSED SET) - {cohort}")
        print("="*70)
        
        # Select embeddings
        if embedding_type == '512_raw':
            X_all = self.embeddings_512_raw
            emb_name = "512D Raw"
        elif embedding_type == '7d_scaled':
            X_all = self.embeddings_7d_scaled
            emb_name = "7D Scaled"
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
        
        # Filter by cohort
        cohort_mask = self.metadata['cohort'] == cohort
        metadata = self.metadata[cohort_mask].copy()
        X = X_all[cohort_mask]
        
        print(f"\nTotal slices in {cohort}: {len(X)}")
        print(f"Total subjects: {metadata['subject_key'].nunique()}")
        
        # Count recordings per subject
        recordings_per_subject = metadata.groupby('subject_key')['recording_key'].nunique()
        valid_subjects = recordings_per_subject[recordings_per_subject >= min_recordings].index
        
        if len(valid_subjects) < 2:
            print(f"    Not enough subjects with >= {min_recordings} recordings")
            return None
        
        print(f"Subjects with >= {min_recordings} recordings: {len(valid_subjects)}")
        
        # ===== CLOSED-SET SPLIT =====
        # Ensure each subject contributes at least 1 recording to train and test
        rng = np.random.RandomState(42)
        train_rec_keys = []
        test_rec_keys = []
        
        for subject in valid_subjects:
            subject_recs = metadata.loc[metadata['subject_key'] == subject, 'recording_key'].unique()
            
            # Shuffle recordings for this subject
            rng.shuffle(subject_recs)
            
            # Determine split: at least 1 test, rest train
            n_test = max(1, int(round(0.3 * len(subject_recs))))
            n_test = min(n_test, len(subject_recs) - 1)  # Keep at least 1 for train
            
            test_rec_keys.extend(subject_recs[:n_test])
            train_rec_keys.extend(subject_recs[n_test:])
        
        # Create masks
        train_mask = metadata['recording_key'].isin(train_rec_keys)
        test_mask = metadata['recording_key'].isin(test_rec_keys)
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train = metadata.loc[train_mask, 'subject_key'].values
        y_test = metadata.loc[test_mask, 'subject_key'].values
        
        # Verify closed-set property
        train_subjects = set(y_train)
        test_subjects = set(y_test)
        missing_from_train = test_subjects - train_subjects
        
        print(f"\nTrain recordings: {len(train_rec_keys)}")
        print(f"Test recordings: {len(test_rec_keys)}")
        print(f"Train samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Train subjects: {len(train_subjects)}")
        print(f"Test subjects: {len(test_subjects)}")
        
        if missing_from_train:
            print(f"    WARNING: {len(missing_from_train)} test subjects not in train!")
            print(f"   This violates closed-set assumption")
        else:
            print(f"Closed-set verified: all test subjects appear in train")
        
        # Train classifier (multi_class='ovr' is default, removed for compatibility)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        
        # Predict
        y_pred = clf.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        
        # Per-subject accuracy
        subject_accs = {}
        for subject in test_subjects:
            mask = (y_test == subject)
            if sum(mask) > 0:
                subject_accs[subject] = accuracy_score(y_test[mask], y_pred[mask])
        
        macro_acc = float(np.mean(list(subject_accs.values())))
        micro_acc = float(np.sum([v * len(y_test[y_test==k]) for k, v in subject_accs.items()]) / len(y_test))
        
        print(f"\n     Speaker ID Results ({emb_name}):")
        print(f"   Overall accuracy: {acc:.4f}")
        print(f"   Balanced accuracy: {bal_acc:.4f}")
        print(f"   Macro accuracy (per subject): {macro_acc:.4f}")
        print(f"   Micro accuracy: {micro_acc:.4f}")
        
        return {
            'task': 'speaker_id_closed_set',
            'cohort': cohort,
            'embedding_type': embedding_type,
            'embedding_name': emb_name,
            'n_train_subjects': len(train_subjects),
            'n_test_subjects': len(test_subjects),
            'n_train_slices': len(X_train),
            'n_test_slices': len(X_test),
            'accuracy': float(acc),
            'balanced_accuracy': float(bal_acc),
            'macro_accuracy': macro_acc,
            'micro_accuracy': micro_acc,
            'closed_set_verified': len(missing_from_train) == 0
        }
    
    def run_all_experiments(self, prefix='cross_lingual_real'):
        """
        Run complete experiment suite:
        - PD LOSO subject-level for each cohort (primary)
        - PD LOSO for combined (optional, with warning)
        - Speaker ID closed-set for each cohort (512D vs 7D)
        """
        # Load data (prefix may be overridden by kcl_30s_mode)
        self.load_data(prefix)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'prefix': prefix,
            'kcl_30s_mode': self.kcl_30s_mode,
            'pd_results': [],
            'speaker_id_results': []
        }
        
        # ===== PD LOSO Experiments (Primary: within each cohort) =====
        print("\n" + "="*70)
        print("     RUNNING PD LOSO EXPERIMENTS (SUBJECT-LEVEL)")
        print("="*70)
        
        # Get unique cohorts
        cohorts = [c for c in self.metadata['cohort'].unique() if pd.notna(c)]
        
        # If in KCL 30s mode, we might want to highlight that
        if self.kcl_30s_mode:
            print("\n     KCL 30s Mode: Analyzing Condition D segments")
        
        # Each cohort separately (primary analysis)
        for cohort in cohorts:
            result = self.run_pd_loso_subject_level(cohort=cohort)
            if result:
                results['pd_results'].append(result)
        
        # Combined cohorts (optional, with warning in report)
        if len(cohorts) > 1:  # Only run combined if multiple cohorts
            print("\n    Running combined cohorts (potential language confound)")
            combined = self.run_pd_loso_subject_level(cohort=None)
            if combined:
                combined['warning'] = 'Combined cohorts may learn language, not disease'
                results['pd_results'].append(combined)
        
        # ===== Speaker ID Experiments (Closed Set) =====
        print("\n" + "="*70)
        print("RUNNING SPEAKER ID EXPERIMENTS (CLOSED SET)")
        print("="*70)
        
        for cohort in cohorts:
            # Skip Speaker ID for KCL in Condition D mode (only 1 recording per subject)
            if self.kcl_30s_mode and cohort == 'KCL':
                print(f"\n    Skipping Speaker ID for {cohort} (Condition D has only one recording per subject)")
                continue
            
            # 512D
            res_512 = self.run_speaker_id_closed_set(cohort, embedding_type='512_raw')
            if res_512:
                results['speaker_id_results'].append(res_512)
            
            # 7D
            res_7d = self.run_speaker_id_closed_set(cohort, embedding_type='7d_scaled')
            if res_7d:
                results['speaker_id_results'].append(res_7d)
        
        # ===== Save Results =====
        self._save_results(results)
        
        # ===== Generate Identity vs Disease Comparison =====
        self._generate_comparison(results)
        
        return results
    
    def _save_results(self, results):
        """Save results with hash - with numpy type conversion"""
        
        # Convert numpy types to Python native types
        results_clean = convert_numpy(results)
        
        # Save JSON
        results_path = self.output_dir / f'{self.run_name}_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_clean, f, indent=2)
        
        # Compute hash
        results_hash = hashlib.sha256(
            json.dumps(results_clean, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        # Save hash
        hash_path = self.output_dir / f'{self.run_name}_hash.txt'
        with open(hash_path, 'w', encoding='utf-8') as f:
            f.write(f"Results hash: {results_hash}\n")
            f.write(f"Generated: {results['timestamp']}\n")
        
        print(f"\n    Results saved to: {results_path}")
        print(f"    Results hash: {results_hash}")
    
    def _generate_comparison(self, results):
        """Generate identity vs disease comparison table"""
        
        print("\n" + "="*70)
        print("     IDENTITY VS DISEASE COMPARISON")
        print("="*70)
        
        # Add note if KCL 30s mode
        if results.get('kcl_30s_mode'):
            print("\n     KCL 30s Mode: Condition D 30-second center segments")
        
        # Organize speaker ID results by cohort
        speaker_by_cohort = {}
        for r in results['speaker_id_results']:
            cohort = r['cohort']
            if cohort not in speaker_by_cohort:
                speaker_by_cohort[cohort] = {}
            speaker_by_cohort[cohort][r['embedding_type']] = r
        
        # Print speaker ID comparison table
        print("\nSPEAKER IDENTIFICATION (Macro Accuracy)")
        print("-"*70)
        print("{:<20} {:>15} {:>15} {:>15}".format(
            "Cohort", "512D Acc", "7D Acc", "Delta (7D-512D)"))
        print("-"*70)
        
        for cohort, res in sorted(speaker_by_cohort.items()):
            if '512_raw' in res and '7d_scaled' in res:
                acc_512 = res['512_raw']['macro_accuracy']
                acc_7d = res['7d_scaled']['macro_accuracy']
                delta = acc_7d - acc_512
                
                print("{:<20} {:>15.3f} {:>15.3f} {:>+15.3f}".format(
                    cohort[:20], acc_512, acc_7d, delta))
        
        # Print PD LOSO results
        print("\n\n     PD CLASSIFICATION (Subject-Level LOSO)")
        print("-"*70)
        print("{:<20} {:>12} {:>12} {:>12} {:>12}".format(
            "Cohort", "N_Subj", "AUROC", "Acc", "BalAcc"))
        print("-"*70)
        
        for r in results['pd_results']:
            cohort = r['cohort'] if r['cohort'] else "ALL COMBINED"
            if r.get('warning'):
                cohort += "*"
            
            n_subj = r.get('n_subjects_evaluated', 0)
            auroc = r.get('auroc', float('nan'))
            acc = r.get('accuracy', float('nan'))
            bal_acc = r.get('balanced_accuracy', float('nan'))
            
            print("{:<20} {:>12} {:>12.3f} {:>12.3f} {:>12.3f}".format(
                cohort[:20], n_subj, auroc, acc, bal_acc))
        
        print("\n* Combined cohorts - may learn language, not disease")


def main():
    parser = argparse.ArgumentParser(description='Step 2: Identity vs Disease Experiments')
    parser.add_argument('--features_dir', type=str, 
                       default=r'C:\Projects\hear_italian\features_certified',
                       help='Directory with Step 1 outputs')
    parser.add_argument('--output_dir', type=str,
                       default=r'C:\Projects\hear_italian\step2_results',
                       help='Directory for Step 2 results')
    parser.add_argument('--prefix', type=str, default='cross_lingual_real',
                       help='Prefix of Step 1 files (e.g., cross_lingual_real)')
    parser.add_argument('--run_name', type=str, default='complete_analysis',
                       help='Name for this run')
    parser.add_argument('--kcl_30s', action='store_true',
                       help='Analyze KCL Condition D 30s center segments')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("STEP 2 EXPERIMENTS")
    print("="*70)
    print(f"Features dir: {args.features_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Prefix: {args.prefix}")
    print(f"Run name: {args.run_name}")
    print(f"KCL 30s Mode: {args.kcl_30s}")
    
    # Run experiments
    step2 = Step2Experiments(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        kcl_30s_mode=args.kcl_30s
    )
    
    results = step2.run_all_experiments(prefix=args.prefix)
    
    print("\n" + "="*70)
    print("    STEP 2 COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":

    main()
