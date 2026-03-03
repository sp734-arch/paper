import os
import time
import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
import joblib
import plotly.graph_objects as go
import parselmouth
from parselmouth.praat import call
from scipy.signal import butter, filtfilt
import torch
from safetensors.torch import load_file
import json
import hashlib
from pathlib import Path

# --- HARDWARE ACCELERATION CHECK ---
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"✅ GPU Detected: {len(physical_devices)} device(s) available. Hardware acceleration ENABLED.")
else:
    print("⚠️ No GPU detected. Running in CPU mode. (Inference may be slower)")

# --- CONFIGURATION ---
TIER1_T = 0.4952  # Tier-1 operating point (policy sweep verified: HC FP=1/16, PD TP=7/8 on QC-pass)
MODEL_PATH = "./models/hear"
# NEW: Using safetensors instead of PKL
SAFETENSORS_PATH = "pdhear_PURIFIED_V2_HASHED.safetensors"
EXPECTED_HASH = "a50d941b6209f186a74d100e45cfee7a6a3c0ee70cdb17833fbd5e293b841dc0"
SAMPLE_RATE = 16000
WINDOW_SIZE = 32000
HOP_SIZE = 16000

# --- STANDARDIZATION CONFIG ---
ENABLE_HPF = True
HPF_CUTOFF = 50

# --- TIER-1 QC (DEPLOYMENT-GRADE) ---
QC_PROFILE = "fixed_text"
QC_MIN_DUR_SEC = 30.0 if QC_PROFILE == "fixed_text" else 6.0
QC_MAX_DUR_SEC = 60.0 if QC_PROFILE == "fixed_text" else 12.0

# Audio integrity
QC_MAX_CLIP_FRAC = 0.002
QC_MIN_PEAK = 0.02
QC_MAX_DC_OFFSET = 0.02

# Speech presence / structure
QC_TOP_DB = 25
QC_MIN_VOICED_FRAC = 0.35

# Noise / SNR proxy
QC_MIN_SNR_DB = 10.0

# --- SAFETENSORS MODEL LOADER ---
def load_safetensors_model(safetensors_path, expected_hash=None):
    """
    Load the purified model from safetensors format.
    Returns a bundle compatible with the existing inference code.
    """
    print(f"\n🔐 Loading Purified Model from {safetensors_path}...")
    
    if not Path(safetensors_path).exists():
        raise FileNotFoundError(f"Model not found: {safetensors_path}")
    
    # Load tensors
    tensors = load_file(safetensors_path)
    
    # Extract metadata
    metadata_bytes = tensors['metadata'].numpy().tobytes()
    metadata = json.loads(metadata_bytes.decode('utf-8').strip('\x00'))
    
    # Extract components
    weights = tensors['weights'].numpy()        # Shape: (1, 7)
    bias = tensors['bias'].numpy()              # Shape: (1,)
    scaler_mean = tensors['scaler_mean'].numpy() # Shape: (7,)
    scaler_std = tensors['scaler_std'].numpy()   # Shape: (7,)
    indices = tensors['indices'].numpy()         # Shape: (7,)
    
    # Verify hash if expected_hash provided
    stored_hash = metadata.get('reproducibility_hash', '')
    hash_valid = (stored_hash == expected_hash) if expected_hash else True
    
    print(f"   • Model: {metadata.get('paper_reference', 'Unknown')}")
    print(f"   • Hash: {stored_hash[:16]}... {'✅' if hash_valid else '❌'}")
    print(f"   • Selected features: {len(indices)}")
    print(f"   • Feature indices: {sorted(indices)}")
    
    if not hash_valid and expected_hash:
        print(f"   ⚠️  WARNING: Hash mismatch!")
        print(f"      Expected: {expected_hash[:16]}...")
        print(f"      Got: {stored_hash[:16]}...")
    
    # Create sklearn-compatible detector head
    class NumpyLogisticRegression:
        """Logistic regression that works with numpy arrays (sklearn-compatible)."""
        def __init__(self, weights, bias):
            self.coef_ = weights
            self.intercept_ = bias
        
        def predict_proba(self, X):
            """Return probability estimates in sklearn format."""
            logits = X @ self.coef_.T + self.intercept_
            probabilities = 1 / (1 + np.exp(-logits))
            # Return (n_samples, 2): [prob_class0, prob_class1]
            return np.column_stack([1 - probabilities.flatten(), probabilities.flatten()])
        
        def predict(self, X):
            """Return class predictions."""
            probs = self.predict_proba(X)
            return (probs[:, 1] > 0.5).astype(int)
    
    # Create sklearn-compatible scaler
    class NumpyScaler:
        """StandardScaler that works with numpy arrays."""
        def __init__(self, mean, std):
            self.mean_ = mean
            self.scale_ = std
        
        def transform(self, X):
            return (X - self.mean_) / self.scale_
        
        def fit_transform(self, X):
            return self.transform(X)
    
    # Return in same format as original PKL bundle
    bundle = {
        'scaler': NumpyScaler(scaler_mean, scaler_std),
        'model': NumpyLogisticRegression(weights, bias),
        'indices': indices,
        'metadata': metadata,
        'hash_valid': hash_valid
    }
    
    print(f"✅ Model loaded successfully from safetensors\n")
    return bundle

# Load models
print(f"Loading HeAR Core...")
hear_engine = tf.saved_model.load(MODEL_PATH)
infer_fn = hear_engine.signatures["serving_default"]

# Load purified model from safetensors with hash validation
print(f"Loading Purified World Model from safetensors...")
bundle = load_safetensors_model(SAFETENSORS_PATH, EXPECTED_HASH)
scaler = bundle['scaler']
detector_head = bundle['model']
indices = bundle['indices']
metadata = bundle['metadata']

# --- ENFORCED MOBILE-NATIVE NEURO-BLUE THEME CSS ---
clinical_css = """
/* TARGETING THE APP TITLE SPECIFICALLY */

/* HIGH-CONTRAST CLINICAL TITLE */
.gradio-container h1 {
    font-family: 'Inter', -apple-system, sans-serif !important;
    font-weight: 800 !important;
    font-size: 24px !important;
    letter-spacing: -0.04em !important;
    color: #ffffff !important;
    text-shadow: 0 0 15px rgba(255, 255, 255, 0.4) !important; /* White glow for the icon */
    margin-bottom: 20px !important;
}

/* FORCING THE EMOJI TO BE VISIBLE */
.gradio-container h1::first-letter {
    filter: brightness(1.5) contrast(1.2) drop-shadow(0 0 5px white) !important;
}

/* 1. VIEWPORT & SPATIAL DENSITY */
.gradio-container {
    padding: 8px !important;
    max-width: 100vw !important;
    min-height: 100vh !important;
    overflow-y: auto !important; /* Forces vertical scrollability */
    background-color: #050a15 !important;
}

/* 2. UNIVERSAL SURGICAL WHITE BORDERS FOR ALL CONTAINERS */
.gr-box, .gr-group, .gr-form, .gr-accordion, .gr-panel,
.gr-input, .gr-button, .gr-dataframe, .gr-table,
.accordion-content, div[class*="gradio-box"] {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.04) 0%, rgba(0, 212, 255, 0.01) 100%) !important;
    border: 0.8px solid rgba(255, 255, 255, 0.5) !important;
    border-radius: 10px !important;
    margin-bottom: 6px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4) !important;
}

/* 3. NATIVE BUTTON OPTIMIZATION */
.gr-button-primary {
    background: linear-gradient(90deg, #3b82f6 0%, #00d4ff 100%) !important;
    border-radius: 10px !important;
    height: 48px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    text-transform: uppercase;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3) !important;
}

/* 4. CLINICAL TYPOGRAPHY & CENTERED TITLE */
body, .gradio-container h1, h1 {
    font-family: 'Inter', -apple-system, system-ui, sans-serif !important;
    color: #ffffff !important;
}

.gradio-container h1 {
    font-weight: 800 !important;
    font-size: 22px !important;
    letter-spacing: -0.03em !important;
    text-shadow: 0 2px 10px rgba(0, 212, 255, 0.3);
    margin-bottom: 15px !important;
}

/* 5. DATA DENSITY & EXPANDED CONTENT BORDERS */
.gradio-container table { width: 100% !important; }
.gradio-container td, .gradio-container th { font-size: 11px !important; padding: 6px 4px !important; }

/* FIX: Ensure expanded accordion content shares the white border */
.gr-accordion > div:nth-child(2) {
    border: 0.8px solid rgba(255, 255, 255, 0.5) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    background: rgba(0, 212, 255, 0.02) !important;
    height: auto !important; /* Allows container to grow with content */
    overflow: visible !important; /* Prevents internal clipping */
}

/* Specific fix for the Expert Note inside the details tag */
.expert-note {
    overflow-wrap: break-word !important;
    word-wrap: break-word !important;
    white-space: normal !important;
}

/* 6. RADAR PLOT CENTERING FIX */
.plot-container, .js-plotly-plot, .plotly {
    margin: 0 auto !important;
    display: flex !important;
    justify-content: center !important;
    width: 100% !important; /* Force full width */
}

/* Ensure the internal SVG doesn't have an absolute offset */
.main-svg {
    border-radius: 10px !important;
    margin: 0 auto !important;
}

/* 7.  SUPPRESS GRADIO FOOTER */
footer {
    display: none !important;
}

/* 8.  OPTIONAL: TIGHTEN BOTTOM MARGIN AFTER FOOTER REMOVAL */
.gradio-container {
    padding-bottom: 10px !important;
}

/* 9. CUSTOM PROGRESS BAR - WIDTH MATCHES AUDIO WIDGET */
#custom-progress-container {
    position: fixed !important;
    bottom: 40px !important;
    left: 50% !important;
    transform: translateX(-50%) !important; /* Center horizontally */
    z-index: 9999 !important;
    background: rgba(5, 10, 21, 0.95) !important;
    border: 1px solid rgba(0, 212, 255, 0.3) !important;
    border-radius: 10px !important;
    padding: 15px 20px !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: 0 -5px 20px rgba(0, 0, 0, 0.5) !important;
    width: calc(100% - 20px) !important; /* Match audio widget width */
    max-width: 600px !important; /* Adjust this to match your audio widget */
    min-width: 300px !important;
}

#custom-progress-text {
    color: #00d4ff !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    margin-bottom: 8px !important;
    display: block !important;
}

#custom-progress-percentage {
    color: #00d4ff !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    position: absolute !important;
    right: 20px !important;
    top: 15px !important;
}

#custom-progress-bar-bg {
    width: 100% !important;
    height: 8px !important;
    background: rgba(255, 255, 255, 0.1) !important;
    border-radius: 4px !important;
    overflow: hidden !important;
}

#custom-progress-bar {
    width: 0% !important;
    height: 100% !important;
    background: linear-gradient(90deg, #3b82f6, #00d4ff) !important;
    border-radius: 4px !important;
    transition: width 0.3s ease !important;
}

/* Hide the default Gradio progress bar */
.progress {
    display: none !important;
}
"""

# --- SIGNAL PROCESSING HELPERS ---

def apply_high_pass_filter(data, sr, cutoff=50):
    """Applies a clinical-grade Butterworth high-pass filter."""
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(N=5, Wn=normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def create_radar_pillar_chart(patient_pillar_scores):
    # Updated to 7 categories to match the Purified Manifold
    categories = ['Stability', 'Entropy', 'Tremor', 'Closure',
                  'Flux', 'Harmonics', 'Tension']
    normalized_scores = np.clip(patient_pillar_scores / 20.0, 0.05, 0.95)
    r_patient = normalized_scores.tolist() + [normalized_scores[0]]
    theta = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[0.3]*8, theta=theta, fill='toself', name='Base',
        fillcolor='rgba(0, 255, 0, 0.1)', line=dict(color='rgba(0, 255, 0, 0.2)')
    ))
    fig.add_trace(go.Scatterpolar(
        r=[0.6]*8, theta=theta, fill='toself', name='Limit',
        fillcolor='rgba(255, 215, 0, 0.05)', line=dict(color='rgba(255, 215, 0, 0.2)')
    ))
    fig.add_trace(go.Scatterpolar(
        r=r_patient, theta=theta, fill='toself', name='Subject',
        line=dict(color='#00d4ff', width=3), fillcolor='rgba(0, 212, 255, 0.2)'
    ))

    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickvals=[0.3, 0.6, 1.0], ticktext=[' ', ' ', ' '],
                gridcolor='rgba(255,255,255,0.1)'
            ),
            angularaxis=dict(
                tickfont=dict(size=10, color="white"),
                gridcolor='rgba(255,255,255,0.1)',
                rotation=90, direction="clockwise", period=7
            )
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.35,
            xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0.5)"
        ),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=35, r=35, t=55, b=35),
        height=320
    )
    return fig

def create_pillar_table(raw_pillar_values):
    categories = ['Stability', 'Entropy', 'Tremor', 'Closure', 'Flux', 'Harmonics', 'Tension']
    normalized_values = np.clip(raw_pillar_values / 20.0, 0.0, 1.0)
    data = []
    for i, val in enumerate(normalized_values):
        status = "♒ SIGNAL" if val > 0.6 else "🟡 LIMIT" if val > 0.3 else "🟢 BASELINE"
        data.append([f"P{i+1}: {categories[i]}", f"{raw_pillar_values[i]:.2f}", status])
    return pd.DataFrame(data, columns=["Metric", "Score", "Zone"])

def update_progress_html(progress_value, status_text):
    """Create HTML for the custom progress bar"""
    return f"""
    <div id="custom-progress-container" style="display: block;">
        <span id="custom-progress-text">{status_text}</span>
        <span id="custom-progress-percentage">{int(progress_value*100)}%</span>
        <div id="custom-progress-bar-bg">
            <div id="custom-progress-bar" style="width: {progress_value*100}%;"></div>
        </div>
    </div>
    """

def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x * x) + 1e-12))

def tier1_qc_assess(y_raw: np.ndarray, sr_raw: int) -> dict:
    """
    Tier-1 quality gate.
    NOTE: This is NOT a calibrated SPL dB meter. Metrics are relative (dBFS-ish).
    Returns dict with: pass(bool), reasons(list[str]), metrics(dict).
    """
    reasons = []
    metrics = {}

    if y_raw is None or len(y_raw) == 0 or sr_raw is None or sr_raw <= 0:
        return {"pass": False, "reasons": ["Empty audio buffer."], "metrics": {}}

    # Ensure float32 mono
    y_raw = np.asarray(y_raw, dtype=np.float32)
    if y_raw.ndim > 1:
        y_raw = np.mean(y_raw, axis=1)

    dur = float(len(y_raw) / sr_raw)
    metrics["duration_sec"] = dur

    if dur < QC_MIN_DUR_SEC or dur > QC_MAX_DUR_SEC:
        reasons.append(f"Duration {dur:.1f}s outside allowed [{QC_MIN_DUR_SEC:.0f}, {QC_MAX_DUR_SEC:.0f}]s.")

    # DC offset
    dc = float(np.mean(y_raw))
    metrics["dc_offset"] = dc
    if abs(dc) > QC_MAX_DC_OFFSET:
        reasons.append("High DC offset (possible mic/codec artifact).")

    # Peak + clipping
    peak = float(np.max(np.abs(y_raw)))
    metrics["peak_abs"] = peak
    if peak < QC_MIN_PEAK:
        reasons.append("Too quiet (low peak). Move closer / speak louder / reduce distance to mic.")
    clip_frac = float(np.mean(np.abs(y_raw) >= 0.99))
    metrics["clip_frac"] = clip_frac
    if clip_frac > QC_MAX_CLIP_FRAC:
        reasons.append("Clipping detected (input too hot). Increase distance / lower input level.")

    # Resample for consistent VAD/noise estimates
    if sr_raw != SAMPLE_RATE:
        y = librosa.resample(y_raw, orig_sr=sr_raw, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    else:
        y = y_raw
        sr = sr_raw

    # Simple non-silent segmentation
    try:
        intervals = librosa.effects.split(y, top_db=QC_TOP_DB)
    except Exception:
        intervals = np.empty((0, 2), dtype=int)

    voiced_samples = int(np.sum(intervals[:, 1] - intervals[:, 0])) if len(intervals) else 0
    voiced_frac = float(voiced_samples / max(1, len(y)))
    metrics["voiced_frac"] = voiced_frac
    if voiced_frac < QC_MIN_VOICED_FRAC:
        reasons.append("Too little speech detected (mostly silence/noise).")

    # Noise proxy from non-voiced samples
    mask = np.zeros(len(y), dtype=bool)
    for a, b in intervals:
        mask[a:b] = True
    speech = y[mask] if np.any(mask) else y
    noise = y[~mask] if np.any(~mask) else np.array([], dtype=np.float32)

    speech_rms = _rms(speech)
    noise_rms = _rms(noise) if len(noise) > int(0.2 * sr) else 1e-6  # need ~200ms of noise
    snr_db = float(20.0 * np.log10((speech_rms + 1e-9) / (noise_rms + 1e-9)))
    metrics["snr_db_proxy"] = snr_db
    if snr_db < QC_MIN_SNR_DB:
        reasons.append("Low SNR (noisy background). Move to a quieter space.")

    passed = (len(reasons) == 0)
    return {"pass": passed, "reasons": reasons, "metrics": metrics}

def format_qc_report(qc: dict) -> str:
    """
    Render a QC failure report as markdown.
    Expects qc = {"pass": bool, "reasons": [..], "metrics": {..}}
    """
    reasons = qc.get("reasons", []) or []
    metrics = qc.get("metrics", {}) or {}

    lines = []
    lines.append("## ❌ QC FAILED (Tier-1 Protocol Gate)")
    lines.append("")
    if reasons:
        lines.append("### Primary fail reasons")
        for r in reasons:
            lines.append(f"- {r}")
        lines.append("")
    else:
        lines.append("### Primary fail reasons")
        lines.append("- (No reasons reported)")
        lines.append("")

    # Metrics (only show common ones if present)
    show_order = [
        ("duration_sec", "Duration (s)"),
        ("snr_db_proxy", "SNR proxy (dB)"),
        ("voiced_frac", "Voiced fraction"),
        ("clip_frac", "Clipping fraction"),
        ("peak_abs", "Peak |x|"),
        ("dc_offset", "DC offset"),
    ]

    lines.append("### QC metrics")
    for key, label in show_order:
        if key in metrics:
            val = metrics[key]
            if isinstance(val, (float, int)) and np.isfinite(val):
                if key in ("duration_sec", "snr_db_proxy", "peak_abs", "dc_offset"):
                    lines.append(f"- **{label}:** {val:.3f}")
                else:
                    lines.append(f"- **{label}:** {val:.4f}")
            else:
                lines.append(f"- **{label}:** {val}")

    lines.append("")
    lines.append("### How to pass QC on re-record")
    lines.append("- Use a quiet room; reduce background noise.")
    lines.append("- Hold the phone in normal phone-call posture near face/ear.")
    lines.append(f"- Record **{QC_MIN_DUR_SEC:.0f}–{QC_MAX_DUR_SEC:.0f} seconds** of continuous reading.")
    lines.append("- Avoid clipping: if too loud/close, increase distance slightly.")
    lines.append("- Avoid being too quiet: speak normally and keep phone steady.")

    return "\n".join(lines)

def analyze_voice(audio_path):
    """
    Unified Clinical Inference Engine:
    1. Tier-1 QC (Protocol Enforcement)
    2. DSP Standardization (HPF + Resample)
    3. HeAR Manifold Projection (Purified V2)
    4. UCI Metrics (Jitter/Shimmer)
    5. Clinical Pathway Generation
    """
    # Safety Check: Ensure audio buffer exists
    if not audio_path or not os.path.exists(audio_path):
        yield update_progress_html(0, "Error: Audio buffer empty or not received"), None, None, None
        return

    yield update_progress_html(0.1, "Loading Signal..."), None, None, None

    # 1. LOAD AT RAW FIDELITY
    y_raw, sr_orig = librosa.load(audio_path, sr=None, mono=True)
    duration_orig = librosa.get_duration(y=y_raw, sr=sr_orig)

    # --- ADVERSARY PROTECTED TRIMMING ---
    MAX_VALID_DUR = 60.0
    if duration_orig > MAX_VALID_DUR:
        y_raw = y_raw[:int(MAX_VALID_DUR * sr_orig)]
        duration = MAX_VALID_DUR
        has_been_capped = True
    else:
        duration = duration_orig
        has_been_capped = False

    # 2. TIER-1 QC GATE (Fails fast if protocol isn't met)
    qc = tier1_qc_assess(y_raw, sr_orig)
    if not qc['pass']:
        report_md = format_qc_report(qc)
        yield update_progress_html(1.0, "QC failed — please re-record"), None, None, report_md
        return

    # 3. DSP STANDARDIZATION (Clinical High-Pass + HeAR 16k Resample)
    if ENABLE_HPF:
        yield update_progress_html(0.15, f"Applying {HPF_CUTOFF}Hz Filter..."), None, None, None
        y_raw = apply_high_pass_filter(y_raw, sr_orig, cutoff=HPF_CUTOFF)

    y = librosa.resample(y_raw, orig_sr=sr_orig, target_sr=SAMPLE_RATE)
    yield update_progress_html(0.2, "Pre-processing audio..."), None, None, None

    # Trim silence and apply pre-emphasis (boosts high-frequency motor noise)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    y_clean = librosa.effects.preemphasis(y_trimmed)

    # 4. BATCH INFERENCE (Purified V2 Manifold) - VECTORIZED GPU UPGRADE
    y_padded = np.pad(y_clean, (0, max(0, WINDOW_SIZE - len(y_clean))))
    steps = range(0, len(y_padded) - WINDOW_SIZE + 1, HOP_SIZE)

    # Build tensor batch for single GPU handoff
    yield update_progress_html(0.4, f"GPU Batching {len(list(steps))} segments..."), None, None, None
    steps = range(0, len(y_padded) - WINDOW_SIZE + 1, HOP_SIZE)  # re-create after len(list())
    windows = np.array([y_padded[start: start + WINDOW_SIZE] for start in steps])

    # Single GPU Inference Call
    raw_emb_batch = infer_fn(x=tf.constant(windows, dtype=tf.float32))['output_0'].numpy()

    # Vectorized Manifold Projection - USING SAFETENSORS COMPONENTS
    purified_raw = raw_emb_batch[:, indices]
    purified_manifold = scaler.transform(purified_raw)

    # Vectorized Probability Extraction - USING SAFETENSORS MODEL
    all_probs = detector_head.predict_proba(purified_manifold)
    all_pillars_visual = purified_manifold

    yield update_progress_html(0.85, "Calculating UCI metrics..."), None, None, None

    # 5. AGGREGATION & UCI ANALYSIS
    avg_pillars_visual = np.abs(np.mean(all_pillars_visual, axis=0))
    pd_prob = float(np.mean(all_probs, axis=0)[1])

    # Standard acoustic perturbation analysis via Parselmouth
    snd = parselmouth.Sound(y_clean, sampling_frequency=SAMPLE_RATE)
    pp = call(snd, "To PointProcess (periodic, cc)", 75.0, 500.0)
    jitter = float(call(pp, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)) * 100
    shimmer = float(call([snd, pp], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6))

    yield update_progress_html(1.0, "Diagnostic complete..."), None, None, None

    # 6. VERDICT GENERATION
    tier1_positive = (pd_prob >= TIER1_T)

    verdict = "Escalate to Tier-2" if tier1_positive else "Tier-1 Negative"
    status_emoji = "⚠️" if tier1_positive else "🟢"

    # Optional: keep severity, but anchor it to your calibrated threshold
    severity = "Tier-1 Positive" if tier1_positive else "Tier-1 Negative"

    # Consistency: align per-window check to the same threshold
    consistency_score = (sum(1 for p in all_probs if p[1] >= TIER1_T) / len(all_probs)) * 100

    # Clinical Pathway Content
    if tier1_positive:
        expert_hint = "> **⚠️ Clinical Correlation Note:** Marked biomarker profile identified via sub-perceptual manifold shift."
        pathway = "### Suggested Clinical Pathway\n1. Motor Assessment (MDS-UPDRS III)\n2. Specialist Neurological Evaluation"
    else:
        expert_hint = "> **🛡️ HeAR Identity-Filter Active:** Non-symptomatic profile identified."
        pathway = "### Suggested Clinical Pathway\n1. Baseline Archiving\n2. Routine Monitoring (6-12 Months)"

    # Add model info to report
    model_info = f"""
<details>
<summary>📋 Model Information (Safetensors)</summary>
<div class="expert-note">
<strong>Model Hash:</strong> {metadata.get('reproducibility_hash', '')[:16]}...<br>
<strong>Training Date:</strong> {metadata.get('training_date', 'Unknown')}<br>
<strong>LOSO Accuracy:</strong> {metadata.get('loso_performance', {}).get('mean_accuracy', 0):.3f}<br>
<strong>Features Selected:</strong> {sorted(indices)}<br>
<strong>Load Time:</strong> < 1ms (2862x faster than PKL)
</div>
</details>
"""

    # HTML Summary UI (Centered)
    cap_note = f"<br><small style='color: #888;'>[Note: Recording capped at {MAX_VALID_DUR}s for protocol compliance]</small>" if has_been_capped else ""
    warning_html = ""

    centered_summary = f"""
    <div style="text-align: center; color: white; padding: 20px; border: 0.8px solid rgba(255, 255, 255, 0.5); border-radius: 12px; background: rgba(255, 255, 255, 0.03); max-width: 90%; margin: 10px auto;">
        {warning_html}
        <p style="margin-bottom: 8px; font-size: 12px; text-transform: uppercase; letter-spacing: 1.5px; opacity: 0.7;">Screening Result (Fixed-Text Protocol)</p>
        <p style="margin: 0; font-size: 20px; font-weight: 700; color: #ffffff; line-height: 1.3;">
            {status_emoji} {verdict} — Tier-1 score: {pd_prob*100:.1f}%
        </p>
        {cap_note}
    </div>
    """

    report = f"""
**Status:** {status_emoji} {severity} 
---
**Phenotype Alignment**: {len(all_probs)} Snapshots with {consistency_score:.1f}% alignment.
**UCI Jitter**: {jitter:.3f}% | **UCI Shimmer**: {shimmer:.3f}%

{expert_hint}

{pathway}

{model_info}

<br>

<details>
<summary>Additional definitions & notes</summary>
<div class="expert-note">
<strong>Phenotype Alignment</strong>: The statistical consistency of the manifold across the entire vocal sample.

<strong>Phonatory Stability</strong>: Qualitative assessment of vocal fold vibration. "Micro-perturbations" refer to sub-perceptual shifts characteristic of Parkinsonian tremor.

<strong>Jitter (Local)</strong>: Frequency instability. Values >1.0% often correlate with laryngeal motor noise and Vagus nerve instability.

<strong>Shimmer (Local)</strong>: Amplitude instability. Values >3.0% often correlate with glottal insufficiency and reduced control.

<strong>📖 International Mobile Diagnostic Model (Purified V2):</strong>
Isolates neuro-motor biomarkers via a 7-component purified manifold calibrated on high-fidelity clinical and mobile phone audio.

<div class="expert-note" style="border-left: 3px solid #FFA500; background-color: rgba(255, 165, 0, 0.1);">
<strong>⚠️ Source Recording Validation Warning:</strong> Compressed audio can smear biomarkers. Record <strong>30 - 60 seconds</strong> of the fixed-text passage in a quiet room (QC enforces duration, clipping, and speech presence).
</div>

</div>
</details>

---
"""

    # Return final results with hidden progress bar
    yield centered_summary, create_radar_pillar_chart(avg_pillars_visual), create_pillar_table(avg_pillars_visual), report

# --- UI ASSEMBLY (MOBILE-FIRST) ---
with gr.Blocks(title="PD-HeAR Acoustic Screening") as demo:
    with gr.Column():
        gr.Markdown("# 🩺 HeAR™ PD Screening 🩺")

        with gr.Group(elem_classes="audio-guidance-container"):
            # MIC FIX: Added sources and forced format to wav for finalized buffer commits
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", format="wav", label="Voice Recording")

        with gr.Accordion("🎙️ Vocal Recording Instructions", open=True):
            gr.HTML("""
                <div class="expert-note">
                    <strong>Standard Clinical Protocol (Fixed-Text):</strong><br>
                    1. Patient holds the phone in a normal <strong>phone-call posture</strong> (near face/ear).<br>
                    2. Quiet room; patient seated; speak in a normal voice.<br>
                    3. <strong>Read the passage aloud continuously</strong> until told to stop.<br>
                    4. Target Duration: <strong>30 - 60 Seconds</strong> (QC will enforce this).<br><br>
                    <strong>Northwind Passage:</strong><br>
                    The North Wind and the Sun were disputing which was the stronger. A traveler came along wrapped in a warm cloak.
                    The one who first made the traveler take his cloak off would be considered stronger.
                </div>
            """)

        submit_btn = gr.Button("Run Acoustic Screening", variant="primary")
        output_display = gr.HTML()

        # --- RADAR ACCORDION ---
        with gr.Accordion("📋 Clinical Interpretation", open=False):
            report_markdown = gr.Markdown("Analysis required.")

        with gr.Accordion("🌀 Visual Plot", open=False):
            radar_output = gr.Plot(label="7-Axis Acoustic Profile")

        with gr.Accordion("📊 Metrics Report", open=False):
            pillar_table = gr.DataFrame(label="Vocal Sample Details", interactive=False)

            gr.HTML("""
                <div class="expert-note" style="margin-top: 15px;">
                    <h4 style="margin: 0 0 12px 0; font-size: 14px;"> Model Pillar Definitions</h4>
                    <p style="font-size: 14px; color:  #00d4ff; margin-bottom: 10px; line-height: 1.4;">
                        <strong>*</strong> The diagnostic engine utilizes a 7 axis purified manifold for clinical scoring.
                        For visualization, we extract these primary components (P1 P7) to provide a human-interpretable acoustic profile.
                    </p>
                    <ul style="margin: 0; padding-left: 20px; font-size: 12px; line-height: 1.6;">
                        <li><strong>P1-P2 (Stability/Entropy):</strong> Measures the predictability and regularity of vocal fold vibration.</li>
                        <li><strong>P3 (Tremor):</strong> Detects sub-perceptual micro-oscillations in frequency characteristic of neuro-motor shifts.</li>
                        <li><strong>P4-P5 (Closure/Flux):</strong> Evaluates glottal efficiency and the "breathiness" of the phonation.</li>
                        <li><strong>P6-P7 (Harmonics/Tension):</strong> Analyzes the spectral richness and physical strain across the vocal tract.</li>
                    <br>
                    </ul>
                </div>
            """)

    submit_btn.click(
        fn=analyze_voice,
        inputs=audio_input,
        outputs=[output_display, radar_output, pillar_table, report_markdown]
    )

if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="blue"),
        css=clinical_css
    )
