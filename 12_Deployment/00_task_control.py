#!/usr/bin/env python3
"""
Step 0: Automated Task Control & Segment Isolation (Tier-1 Protocol Gate)

Purpose:
Isolates standardized speech segments for the "North Wind and the Sun" reading task. 
This script enforces 'Task Control' by using Whisper v3 ASR to detect linguistic 
anchors, ensuring evaluation is restricted to a fixed-text task.

Validity Requirement: 
- Checklist Item 1: Task Control [Ref: Section 7.1.7].
- Prevents disease separation from collapsing due to task confounding [Ref: Table 2].

Technical Specs:
- Model: Faster-Whisper (large-v3)
- Audio Output: 16-bit PCM (Lossless)
- Anchor Tokens: "North Wind" (Start) -> "Stronger" (End)

# Normal production run
python 00_task_control.py --source /data/KCL/PD --cohort PD

# Validation only (no files written)
python 00_task_control.py --source /data/KCL/HC --cohort HC --validation-mode

# Negative control test
python 00_task_control.py --source /data/empty_room --negative-control --validation-mode

# With different model for speed
python 00_task_control.py --source /data/KCL/PD --model medium --device cpu


Author: Jim McCormack
Date: Feb 2026
"""

import os
import subprocess
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Handle missing dependencies gracefully
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    WhisperModel = None
    print("⚠️  WARNING: faster-whisper not installed. Run: pip install faster-whisper")
    print("   Continuing but will fail at model load...")

# --- CONFIGURATION (DEPLOYMENT STANDARDS) ---
# Aligns with Tier-1 Protocol Gate requirements for task isolation.

# Audio format standards - keep original but add validation
OUTPUT_SAMPLE_RATE = 16000  # Hz, 16kHz standard for speech
OUTPUT_CHANNELS = 1         # Mono
OUTPUT_CODEC = "pcm_s16le"  # 16-bit PCM lossless

# Anchor settings to ensure physiological onset/offset are captured
# ORIGINAL VALUES - restored after over-correction
START_BUFFER = 0.30
END_BUFFER = 0.40
MIN_DUR_SEC = 1.0   # Original: just discard truly junk segments (<1s)
# Note: The passage from "North Wind" to "Stronger" is ~10-12 seconds

class TaskControlValidator:
    """Logs and validates task isolation for audit trail"""
    
    def __init__(self, log_dir: Path, cohort: str = "PD"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.cohort = cohort
        self.stats = defaultdict(int)
        self.file_records = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def log_file_result(self, filename, status, details=None):
        """Record per-file processing results with metadata"""
        # Fix encoding issues for Windows console
        try:
            filename = filename.encode('ascii', 'ignore').decode('ascii')
        except:
            pass
            
        record = {
            "filename": filename,
            "cohort": self.cohort,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.file_records.append(record)
        self.stats[status] += 1
        self.stats["total_files"] += 1
        
    def write_report(self):
        """Generate comprehensive JSON and text audit reports"""
        
        # Calculate success rate
        success_count = self.stats.get("SUCCESS", 0) + self.stats.get("VALIDATION_PASS", 0)
        total_processed = len(self.file_records)
        success_rate = (success_count / total_processed * 100) if total_processed > 0 else 0
        
        # Compile audit data
        summary = {
            "session_id": self.session_id,
            "cohort": self.cohort,
            "timestamp": datetime.now().isoformat(),
            "total_files_processed": total_processed,
            "success_count": success_count,
            "success_rate": round(success_rate, 2),
            "statistics": dict(self.stats),
            "files": self.file_records
        }
        
        # Write JSON report (ASCII safe)
        report_path = self.log_dir / f"task_control_audit_{self.cohort}_{self.session_id}.json"
        with open(report_path, 'w', encoding='ascii', errors='ignore') as f:
            json.dump(summary, f, indent=2)
            
        # Write human-readable log (ASCII safe)
        log_path = self.log_dir / f"task_control_log_{self.cohort}_{self.session_id}.txt"
        with open(log_path, 'w', encoding='ascii', errors='ignore') as f:
            f.write(f"=== TASK CONTROL AUDIT REPORT ===\n")
            f.write(f"Cohort: {self.cohort}\n")
            f.write(f"Session: {self.session_id}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*50}\n\n")
            
            f.write(f"SUMMARY STATISTICS:\n")
            f.write(f"  Total files processed: {total_processed}\n")
            f.write(f"  Successful extractions: {success_count}\n")
            f.write(f"  Success rate: {success_rate:.1f}%\n\n")
            
            f.write("BREAKDOWN BY STATUS:\n")
            for status, count in sorted(self.stats.items()):
                if status != "total_files":
                    f.write(f"  {status}: {count}\n")
                    
        print(f"\n📊 Audit log written: {log_path}")
        print(f"📈 JSON report: {report_path}")
        return log_path, report_path

def clean(tok: str) -> str:
    """Normalize token strings for consistent comparison"""
    if tok is None:
        return ""
    return tok.lower().strip(".,?!:;\"'()[]{} ")

def find_start_end(words):
    """
    ORIGINAL IMPLEMENTATION - restored after over-correction
    Returns (start_ts, end_ts) in seconds or (None, None) if not found.
    Start: detect phrase "north wind" (north then wind within next 2 tokens)
    End: last occurrence of "stronger" after start
    """
    toks = [clean(w.word) for w in words]

    # --- Find start: "north wind" ---
    start_idx = None
    for i, t in enumerate(toks):
        if t == "north":
            # look ahead a couple tokens for "wind"
            for j in range(i + 1, min(i + 3, len(toks))):
                if toks[j] == "wind":
                    start_idx = i
                    break
        if start_idx is not None:
            break

    if start_idx is None:
        return None, None

    start_ts = max(0.0, words[start_idx].start - START_BUFFER)

    # --- Find end: last "stronger" after start ---
    end_ts = None
    for k in range(start_idx, len(toks)):
        if toks[k] == "stronger":
            end_ts = words[k].end + END_BUFFER

    if end_ts is None:
        return None, None

    # Sanity
    if end_ts <= start_ts:
        return None, None

    if (end_ts - start_ts) < MIN_DUR_SEC:
        return None, None

    return start_ts, end_ts

def slice_wav(input_path: Path, output_path: Path, start_ts: float, end_ts: float):
    """Extract segment with standardized audio format"""
    dur = end_ts - start_ts
    cmd = [
        "ffmpeg", "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{start_ts:.3f}",
        "-t",  f"{dur:.3f}",
        "-i", str(input_path),
        "-map", "0:a:0",
        "-acodec", OUTPUT_CODEC,
        "-ar", str(OUTPUT_SAMPLE_RATE),
        "-ac", str(OUTPUT_CHANNELS),
        str(output_path),
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ FFmpeg error: {e.stderr}")
        raise

def process_batch(source_dir: Path, cohort: str = "PD", 
                 validation_mode: bool = False, 
                 negative_control: bool = False):
    """
    Main processing pipeline - restored to original logic with added validation
    """
    print(f"\n{'='*60}")
    print(f"🚀 TASK CONTROL ISOLATION v1.0 (Original)")
    print(f"{'='*60}")
    print(f"Cohort: {cohort}")
    print(f"Source: {source_dir}")
    print(f"Mode: {'VALIDATION' if validation_mode else 'PRODUCTION'}")
    print(f"Negative Control: {'YES' if negative_control else 'NO'}")
    print(f"{'='*60}\n")
    
    # Setup directories
    target_dir = source_dir / f"northwindpci"
    if not validation_mode:
        target_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize validator
    validator = TaskControlValidator(target_dir / "logs", cohort)
    
    # Check dependencies
    if not FASTER_WHISPER_AVAILABLE:
        print("❌ ERROR: faster-whisper not installed.")
        print("   Run: pip install faster-whisper")
        return None
    
    # Load Whisper model
    print("📦 Loading Whisper v3 (large-v3)...")
    try:
        model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        print("✅ Model loaded successfully (GPU)\n")
    except:
        try:
            print("   GPU failed, falling back to CPU...")
            model = WhisperModel("large-v3", device="cpu", compute_type="int8")
            print("✅ Model loaded successfully (CPU)\n")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return None
    
    # Find audio files
    audio_extensions = (".wav", ".mp3", ".m4a", ".flac", ".ogg")
    audio_files = sorted([
        f for f in os.listdir(source_dir) 
        if f.lower().endswith(audio_extensions)
    ])
    
    print(f"📂 Found {len(audio_files)} audio files to process.\n")
    
    # Track statistics
    negative_control_detections = 0
    
    # Process each file
    for idx, filename in enumerate(audio_files, 1):
        input_path = source_dir / filename
        output_filename = f"{Path(filename).stem}_northwindpci.wav"
        output_path = target_dir / output_filename if not validation_mode else None
        
        # Fix filename for logging
        safe_filename = filename.encode('ascii', 'ignore').decode('ascii')
        print(f"[{idx}/{len(audio_files)}] 👂 Analyzing: {safe_filename}")
        
        details = {
            "start_time": None,
            "end_time": None,
            "duration": None,
            "word_count": 0
        }
        
        try:
            # Transcribe with word timestamps
            segments, info = model.transcribe(
                str(input_path), 
                word_timestamps=True,
                language="en",
                beam_size=5,
                condition_on_previous_text=False
            )
            
            # Collect all words with timestamps
            all_words = []
            for seg in segments:
                if seg.words:
                    all_words.extend(seg.words)
            
            details["word_count"] = len(all_words)
            details["language"] = info.language
            details["language_probability"] = round(info.language_probability, 3)
            
            print(f"   📝 Detected language: {info.language} (confidence: {info.language_probability:.2%})")
            
            if not all_words:
                print(f"   ❌ No words detected")
                validator.log_file_result(filename, "SKIP_NO_WORDS", details)
                continue
            
            # Find passage boundaries - ORIGINAL LOGIC
            start_ts, end_ts = find_start_end(all_words)
            
            if start_ts is None:
                print(f"   ❌ North Wind boundaries not detected")
                validator.log_file_result(filename, "SKIP_NO_BOUNDARIES", details)
                
                if negative_control:
                    print(f"   ✅ Negative control: expected skip")
                continue
            
            duration = end_ts - start_ts
            details.update({
                "start_time": round(start_ts, 3),
                "end_time": round(end_ts, 3),
                "duration": round(duration, 3)
            })
            
            # For negative control, detection is a failure
            if negative_control:
                negative_control_detections += 1
                print(f"   ❌ Negative control FAILED - passage detected")
            
            print(f"   📍 Segment: {start_ts:.2f}s → {end_ts:.2f}s (Δ {duration:.2f}s)")
            
            # Skip actual file writing in validation mode
            if validation_mode:
                print(f"   ✅ Validation pass (no file written)")
                validator.log_file_result(filename, "VALIDATION_PASS", details)
                continue
            
            # Extract audio segment
            try:
                slice_wav(input_path, output_path, start_ts, end_ts)
                
                if output_path.exists():
                    file_size = output_path.stat().st_size / (1024 * 1024)
                    print(f"   💾 Saved: {output_path.name} ({file_size:.1f} MB)")
                    validator.log_file_result(filename, "SUCCESS", details)
                else:
                    print(f"   ❌ Output file not created")
                    validator.log_file_result(filename, "FAIL_OUTPUT_MISSING", details)
                    
            except subprocess.CalledProcessError as e:
                print(f"   ❌ FFmpeg extraction failed")
                validator.log_file_result(filename, "FAIL_FFMPEG", {"error": str(e)})
            
        except Exception as e:
            print(f"   ❌ Error: {type(e).__name__}")
            validator.log_file_result(filename, "FAIL_EXCEPTION", {"error": str(e)})
        
        print()  # Blank line between files
    
    # Write final audit report
    log_path, json_path = validator.write_report()
    
    # Print summary
    print(f"\n{'='*60}")
    print("📋 TASK CONTROL VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Cohort:            {cohort}")
    print(f"Files processed:   {validator.stats['total_files']}")
    print(f"Success:           {validator.stats.get('SUCCESS', 0)}")
    print(f"Validation pass:   {validator.stats.get('VALIDATION_PASS', 0)}")
    print(f"Skipped (no words): {validator.stats.get('SKIP_NO_WORDS', 0)}")
    print(f"Skipped (no boundaries): {validator.stats.get('SKIP_NO_BOUNDARIES', 0)}")
    
    if negative_control:
        print(f"\n🧪 NEGATIVE CONTROL RESULTS:")
        print(f"   False detections: {negative_control_detections}")
        print(f"   {'✅ PASS' if negative_control_detections == 0 else '❌ FAIL'}")
    
    print(f"\n📊 Audit logs: {target_dir / 'logs'}")
    print(f"{'='*60}\n")
    
    return validator

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Task Control Isolation for North Wind and the Sun passage"
    )
    
    parser.add_argument("--source", type=str, required=True,
                       help="Source directory containing audio files")
    parser.add_argument("--cohort", type=str, default="PD",
                       help="Cohort label (PD, HC, etc.)")
    parser.add_argument("--validation-mode", action="store_true",
                       help="Run validation checks without writing output files")
    parser.add_argument("--negative-control", action="store_true",
                       help="Run as negative control - expect zero detections")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Compute device")
    
    args = parser.parse_args()
    
    # Fix Windows path if needed
    source_path = Path(args.source)
    if not source_path.exists():
        # Try relative path
        source_path = Path.cwd() / args.source
        if not source_path.exists():
            print(f"❌ Error: Source directory not found: {args.source}")
            return 1
    
    # Run processing
    try:
        validator = process_batch(
            source_dir=source_path,
            cohort=args.cohort,
            validation_mode=args.validation_mode,
            negative_control=args.negative_control
        )
        
        if validator is None:
            return 1
            
        # Return appropriate exit code
        if args.negative_control:
            return 0 if validator.stats.get('SUCCESS', 0) == 0 else 1
        else:
            return 0
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted")
        return 130
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
