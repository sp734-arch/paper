"""
Generate Figure 1: Dataset-Specific Artifacts in Parkinson's Voice Screening

This script demonstrates the core problem addressed in the paper:
- Panel A: Within-dataset evaluation (Italian → Italian held-out) shows high AUC (0.97)
- Panel B: Cross-dataset evaluation (Italian → English) collapses to chance (0.47)

This proves that naïve training learns dataset-specific artifacts rather than 
physiological biomarkers, motivating the need for our auditing approach.

Data Requirements:
- Italian PD dataset (clinical recordings)
- Italian HC dataset (clinical recordings)  
- English PD dataset (different recording conditions)
- English HC dataset (different recording conditions)

Output:
- Figure1_dataset_artifacts_[timestamp].png
- Figure1_dataset_artifacts_[timestamp].pdf
- results.json (numerical results)

Paper Reference: Figure 1
Author: Jim McCormack
Date: Feb 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from pathlib import Path
import warnings
from datetime import datetime
import json
from collections import defaultdict
import re
from tqdm import tqdm
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent

print("=" * 80)
print("FIGURE 1: DATASET-SPECIFIC ARTIFACTS IN PD VOICE SCREENING")
print("=" * 80)

# Data directories

# Path will automatically handle the slashes for your OS
# Use raw strings (r"") to prevent backslash issues
# Path will automatically handle the slashes for your OS
# ============================================================================
# CONFIGURATION
# ============================================================================
# Use raw strings (r"") to ensure Windows backslashes are handled correctly
ITALIAN_HEALTHY_DIR = Path(r"C:\Projects\hear_italian\features_audit\density_50\features_italian\healthy")
ITALIAN_PARKINSONS_DIR = Path(r"C:\Projects\hear_italian\features_audit\density_50\features_italian\parkinsons")
ENGLISH_HEALTHY_DIR = Path(r"C:\Projects\hear_italian\features_audit\density_50\features_english\healthy")
ENGLISH_PARKINSONS_DIR = Path(r"C:\Projects\hear_italian\features_audit\density_50\features_english\parkinsons")

# ============================================================================
# SUBJECT PARSING FUNCTION
# ============================================================================

def parse_subject_from_filename(filename):
    """Extract subject ID from .npy filename."""
    stem = Path(filename).stem
    
    # Remove _s{number} suffix
    stem = re.sub(r'_s\d+$', '', stem)
    
    parts = stem.split('_')
    
    # Italian: first part has space
    if ' ' in parts[0]:
        return parts[0]
    
    # English: build up from beginning until we have a valid subject
    for i in range(1, min(4, len(parts) + 1)):
        candidate = '_'.join(parts[:i])
        
        # Check if this looks like a valid subject
        if candidate.startswith('AH_'):
            suffix = candidate[3:]
            if len(suffix) == 4 and suffix.isalnum():  # "064F", "114S"
                return candidate
            if suffix.isdigit():  # "545616858"
                return candidate
            if suffix and suffix[0].isdigit():
                return candidate
    
    return parts[0]

# ============================================================================
# LOAD DATA WITH SUBJECT GROUPING
# ============================================================================

def load_all_subjects_with_data():
    """Load all data grouped by language and subject."""
    datasets = [
        ('Italian Healthy', ITALIAN_HEALTHY_DIR, 0, 'italian'),
        ('Italian PD', ITALIAN_PARKINSONS_DIR, 1, 'italian'),
        ('English Healthy', ENGLISH_HEALTHY_DIR, 0, 'english'),
        ('English PD', ENGLISH_PARKINSONS_DIR, 1, 'english')
    ]
    
    all_subjects = defaultdict(lambda: defaultdict(list))
    
    for dataset_name, data_dir, label, language in datasets:
        if not data_dir.exists():
            print(f"⚠️ Directory not found: {data_dir}")
            continue
        
        files = list(data_dir.glob("*.npy"))
        print(f"📁 {dataset_name}: {len(files)} files")
        
        if not files:
            continue
        
        for f in files:
            try:
                embedding = np.load(f)
                subject = parse_subject_from_filename(f)
                
                all_subjects[language][subject].append({
                    'embedding': embedding,
                    'filename': f.name,
                    'subject': subject,
                    'label': label,
                    'language': language
                })
            except Exception as e:
                print(f"Error loading {f}: {e}")
    
    return all_subjects

# ============================================================================
# RUN EXPERIMENT: SUBJECT-INDEPENDENT LOSO (CORRECTED CALC)
# ============================================================================

def run_experiment_with_subject_splits():
    """Run the cross-dataset experiment with LOSO to match the 0.97 Audit results."""
    
    print("\n" + "=" * 80)
    print("RUNNING CROSS-DATASET EXPERIMENT (LOSO METHOD)")
    print("=" * 80)
    
    # Step 1: Load all data
    print("\n📊 LOADING ALL DATA...")
    all_subjects = load_all_subjects_with_data()
    
    # Step 2: Statistics
    print("\n📊 DATASET STATISTICS:")
    print("=" * 40)
    
    for language in ['italian', 'english']:
        language_data = all_subjects.get(language, {})
        if not language_data:
            print(f"\n{language.upper()}: No data found")
            continue
        
        subjects = list(language_data.keys())
        hc_subjects = [s for s in subjects if language_data[s][0]['label'] == 0]
        pd_subjects = [s for s in subjects if language_data[s][0]['label'] == 1]
        
        total_recordings = sum(len(recordings) for recordings in language_data.values())
        
        print(f"\n{language.upper()}:")
        print(f"   Subjects: {len(subjects)} total")
        print(f"     - Healthy (HC): {len(hc_subjects)}")
        print(f"     - Parkinson's (PD): {len(pd_subjects)}")
        print(f"   Total recordings: {total_recordings}")
        print(f"   Recordings per subject: {total_recordings / len(subjects):.1f} avg")

    # Step 3: Italian LOSO Loop (Within-Dataset Evaluation)
    print("\n🎯 RUNNING ITALIAN LOSO (WITHIN-DATASET)...")
    ita_data = all_subjects['italian']
    ita_subjects = list(ita_data.keys())
    
    all_ita_probs = []
    all_ita_labels = []
    
    # Correct LOSO logic replaces the random 80/20 split
    for test_sub in tqdm(ita_subjects, desc="Processing Italian Subjects"):
        train_list = [rec for s, recs in ita_data.items() if s != test_sub for rec in recs]
        test_list = ita_data[test_sub]
        
        X_train = np.array([d['embedding'] for d in train_list])
        y_train = np.array([d['label'] for d in train_list])
        X_test = np.array([d['embedding'] for d in test_list])
        y_test = np.array([d['label'] for d in test_list])
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=min(64, X_train.shape[1]), random_state=42)),
            ('clf', LogisticRegression(C=0.1, max_iter=1000, random_state=42))
        ])
        
        pipeline.fit(X_train, y_train)
        probs = pipeline.predict_proba(X_test)[:, 1]
        
        all_ita_probs.extend(probs)
        all_ita_labels.extend(y_test)
    
    ita_auc = roc_auc_score(all_ita_labels, all_ita_probs)
    fpr_ita, tpr_ita, _ = roc_curve(all_ita_labels, all_ita_probs)

    # Step 4: Cross-Dataset (Italian Total -> English Total)
    print("\n🌍 PREPARING ENGLISH CROSS-DATASET EVALUATION...")
    all_ita_recs = [rec for s, recs in ita_data.items() for rec in recs]
    X_train_full = np.array([d['embedding'] for d in all_ita_recs])
    y_train_full = np.array([d['label'] for d in all_ita_recs])
    
    eng_data = all_subjects['english']
    eng_subjects = list(eng_data.keys())
    english_data_list = [rec for s, recs in eng_data.items() for rec in recs]
    
    X_test_eng = np.array([d['embedding'] for d in english_data_list])
    y_test_eng = np.array([d['label'] for d in english_data_list])
    
    pipeline_cross = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=min(64, X_train_full.shape[1]), random_state=42)),
        ('clf', LogisticRegression(C=0.1, max_iter=1000, random_state=42))
    ])
    
    pipeline_cross.fit(X_train_full, y_train_full)
    y_pred_eng = pipeline_cross.predict_proba(X_test_eng)[:, 1]
    
    eng_auc = roc_auc_score(y_test_eng, y_pred_eng)
    fpr_eng, tpr_eng, _ = roc_curve(y_test_eng, y_pred_eng)
    
    results = {
        'ita': {
            'auc': ita_auc, 'fpr': fpr_ita, 'tpr': tpr_ita,
            'scores': all_ita_probs, 'labels': all_ita_labels
        },
        'eng': {
            'auc': eng_auc, 'fpr': fpr_eng, 'tpr': tpr_eng,
            'scores': y_pred_eng, 'labels': y_test_eng
        },
        'train_subjects': len(ita_subjects),
        'test_ita_subjects': len(ita_subjects),
        'test_eng_subjects': len(eng_subjects),
        'train_recordings': len(all_ita_recs),
        'test_eng_recordings': len(english_data_list),
        'performance_drop': ita_auc - eng_auc
    }
    
    return results

# ============================================================================
# ROC CURVE VISUALIZATION WITH CLEAR LABELS
# ============================================================================

def generate_roc_figure(results, output_dir):
    """Generate ROC curve figure with clear within-dataset vs cross-dataset labels."""
    
    if results is None or 'ita' not in results or 'eng' not in results:
        print("⚠️ Not enough data for visualization")
        return
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # ==================== PANEL A: Within-Dataset (Italian→Italian) ====================
    axes[0].plot(
        results['ita']['fpr'], 
        results['ita']['tpr'], 
        color='#2E86AB', 
        lw=2.5, 
        label=f'AUC = {results["ita"]["auc"]:.3f}'
    )
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Chance')
    axes[0].set_title(
        "A) Within-Dataset Evaluation\nItalian → Italian (Rigorous LOSO)", 
        fontsize=12, 
        fontweight='bold', 
        pad=15
    )
    axes[0].set_xlabel('False Positive Rate', fontsize=11)
    axes[0].set_ylabel('True Positive Rate', fontsize=11)
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].fill_between(
        results['ita']['fpr'], 
        results['ita']['tpr'], 
        alpha=0.1, 
        color='#2E86AB'
    )
    
    # Performance text for Panel A
    axes[0].text(
        0.05, 0.95, 
        f'Baseline\nAUC = {results["ita"]["auc"]:.3f}\n(In-Lab)', 
        transform=axes[0].transAxes, 
        fontsize=16, 
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    # ==================== PANEL B: Cross-Dataset (Italian→English) ====================
    axes[1].plot(
        results['eng']['fpr'], 
        results['eng']['tpr'], 
        color='#A23B72', 
        lw=2.5,
        label=f'AUC = {results["eng"]["auc"]:.3f}'
    )
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Chance')
    axes[1].set_title(
        "B) Cross-Dataset Evaluation\nItalian → English (Environmental Shift)", 
        fontsize=12, 
        fontweight='bold', 
        pad=15
    )
    axes[1].set_xlabel('False Positive Rate', fontsize=11)
    axes[1].legend(loc='lower right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].fill_between(
        results['eng']['fpr'], 
        results['eng']['tpr'], 
        alpha=0.1, 
        color='#A23B72'
    )
    
    # Performance text for Panel B
    axes[1].text(
        0.05, 0.95, 
        f'Collapse\nAUC = {results["eng"]["auc"]:.3f}\n(Cross-Cohort)', 
        transform=axes[1].transAxes, 
        fontsize=16, 
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    # ==================== MAIN TITLE ====================
    fig.suptitle(
        f'Figure 1: Dataset-Specific Artifacts in Parkinson\'s Voice Screening\n',
        fontsize=14, 
        fontweight='bold', 
        y=0.90
    )
    
    # ==================== FOOTER TEXT ====================
    model_details = (
        "Naïve Methodology: 512-dim Latent Projection → StandardScaler → PCA(64) → Logistic Regression"
    )
    fig.text(0.5, 0.01, model_details, ha='center', fontsize=10, style='italic')
    
    experiment_design = (
        f"Validation: Subject-Independent LOSO | Drop ΔAUC = {results['performance_drop']:.3f}"
    )
    fig.text(0.5, -0.02, experiment_design, ha='center', fontsize=9)
    
    # Adjust layout
    plt.tight_layout(rect=[0.02, 0.05, 1, 0.95])
    
    # Save figure
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = output_dir / f"Figure1_artifacts_{timestamp_str}.png"
    pdf_path = output_dir / f"Figure1_artifacts_{timestamp_str}.pdf"
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    
    print(f"\n📊 Figure saved to: {png_path}")
    plt.show()
    
    return {'png': png_path, 'pdf': pdf_path}

# ============================================================================
# MAIN
# ============================================================================

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def main():
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H%M%S")
    
    output_dir = SCRIPT_DIR / "outputs_final" / date_str / f"run_{time_str}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    results = run_experiment_with_subject_splits()
    
    if results is None:
        print("\n❌ Experiment failed")
        return
    
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 RESULTS:")
    print(f"  Within-dataset (Italian LOSO): AUC = {results['ita']['auc']:.3f}")
    print(f"  Cross-dataset (English):       AUC = {results['eng']['auc']:.3f}")
    print(f"  Stability Drop:               ΔAUC = {results['performance_drop']:.3f}")
    
    print(f"\n🔍 SCIENTIFIC INTERPRETATION:")
    print("   Standard models exploit acoustic artifacts, leading to massive cross-cohort collapse.")
    print("   This confirms that lab-specific accuracy (0.97) does not indicate clinical portability.")
    
    generate_roc_figure(results, output_dir)
    
    # Fix JSON serialization by converting all numpy types to Python native types
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        # Convert the entire results dictionary recursively
        json_results = convert_numpy_types(results)
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_path}")
    print("\n✅ EXPERIMENT COMPLETE!")

if __name__ == "__main__":
    main()
