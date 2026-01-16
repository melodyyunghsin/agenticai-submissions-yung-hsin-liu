import os
import json
import time
import threading
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import wfdb
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from models import EcgSegment, EcgFeatures, BioStatus, UserBaseline, PomodoroSummary

# Configuration
STORAGE_DIR = "data"
os.makedirs(STORAGE_DIR, exist_ok=True)
BIO_STATUS_FILE = os.path.join(STORAGE_DIR, "bio_status.json")
BASELINE_FILE = os.path.join(STORAGE_DIR, "user_baseline.json")
SUMMARY_FILE = os.path.join(STORAGE_DIR, "pomodoro_summary.json")

APP_ORIGINS = ["http://localhost:3000"]
app = FastAPI(title="ECG Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=APP_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for background tasks
processing_active = True

def get_mit_bih_data(record_name='100', sampto=3000):
    """Fetch MIT-BIH data for simulation."""
    try:
        record = wfdb.rdrecord(record_name, pn_dir='mitdb', sampto=sampto)
        return record.p_signal[:, 0], record.fs
    except Exception as e:
        print(f"Error fetching MIT-BIH data: {e}")
        # Return dummy sine wave if fail
        fs = 360
        t = np.linspace(0, sampto/fs, sampto)
        return 0.5 * np.sin(2 * np.pi * 1.2 * t) + 0.2 * np.random.randn(sampto), fs

def process_ecg(raw_signal: np.ndarray, fs: int):
    """
    Complete ECG processing pipeline:
    detrend -> filter -> smooth -> R peaks -> P/T/QRS -> Metrics -> Plot
    """
    # 1. Detrend and Filter (0.5 - 45 Hz)
    detrended = signal.detrend(raw_signal)
    b, a = signal.butter(3, [0.5, 45], btype='bandpass', fs=fs)
    filtered = signal.filtfilt(b, a, detrended)
    
    # 2. Smooth
    window_len = int(fs * 0.05)
    if window_len % 2 == 0: window_len += 1
    smoothed = signal.savgol_filter(filtered, window_len, 3)
    
    # 3. Find R peaks (using a simple threshold + distance method)
    # Refined: Use squared derivative for QRS detection (Pan-Tompkins like)
    diff = np.diff(smoothed)
    squared = diff ** 2
    # Moving average
    ma_len = int(fs * 0.12)
    integrated = np.convolve(squared, np.ones(ma_len)/ma_len, mode='same')
    
    peaks, _ = signal.find_peaks(integrated, distance=int(fs * 0.6), height=np.mean(integrated)*2)
    
    # 4. P-wave, T-wave, QRS complex extraction
    # Simplified search windows relative to R-peak
    p_waves = [] # List of (start, end) indices
    t_waves = []
    qrs_complexes = []
    
    rr_intervals = []
    
    for i in range(1, len(peaks) - 1):
        r = peaks[i]
        rr_intervals.append((peaks[i] - peaks[i-1]) / fs * 1000.0) # ms
        
        # QRS: simple window
        qrs_start = r - int(fs * 0.05)
        qrs_end = r + int(fs * 0.05)
        qrs_complexes.append((qrs_start, qrs_end))
        
        # T-wave: search after QRS
        t_search_start = r + int(fs * 0.1)
        t_search_end = r + int(fs * 0.4)
        if t_search_end < len(smoothed):
            t_peak_idx = np.argmax(smoothed[t_search_start:t_search_end]) + t_search_start
            t_waves.append((t_peak_idx - int(fs*0.05), t_peak_idx + int(fs*0.05)))
            
        # P-wave: search before QRS
        p_search_start = r - int(fs * 0.25)
        p_search_end = r - int(fs * 0.08)
        if p_search_start > 0:
            p_peak_idx = np.argmax(smoothed[p_search_start:p_search_end]) + p_search_start
            p_waves.append((p_peak_idx - int(fs*0.03), p_peak_idx + int(fs*0.03)))

    # Metrics
    hr = 0.0
    sdnn = 0.0
    if len(rr_intervals) > 1:
        hr = 60000.0 / np.mean(rr_intervals)
        sdnn = np.std(rr_intervals)
        
    return {
        "raw": raw_signal,
        "cleaned": smoothed,
        "peaks": peaks,
        "p_waves": p_waves,
        "t_waves": t_waves,
        "qrs": qrs_complexes,
        "hr_bpm": hr,
        "sdnn_ms": sdnn,
        "rr_ms": rr_intervals
    }

def generate_vis(data, fs, filename="ecg_plot.png"):
    """Generate plotting with annotations."""
    plt.figure(figsize=(12, 6))
    time_ax = np.arange(len(data['raw'])) / fs
    
    plt.subplot(2, 1, 1)
    plt.plot(time_ax, data['raw'], label='Raw ECG', alpha=0.5)
    plt.title("Raw Signal")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(time_ax, data['cleaned'], color='black', label='Cleaned')
    
    # R peaks
    plt.scatter(data['peaks'] / fs, data['cleaned'][data['peaks']], color='red', marker='v', label='R Peak')
    
    # P waves (Blue boxes)
    for start, end in data['p_waves']:
        plt.axvspan(start / fs, end / fs, color='blue', alpha=0.2, label='P wave' if start == data['p_waves'][0][0] else "")
        
    # T waves (Green boxes)
    for start, end in data['t_waves']:
        plt.axvspan(start / fs, end / fs, color='green', alpha=0.2, label='T wave' if start == data['t_waves'][0][0] else "")
        
    plt.title("Cleaned Signal with Features")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(STORAGE_DIR, filename)
    plt.savefig(path)
    plt.close()
    return path

# --- Background Task Implementation ---

def bio_status_loop():
    """High-frequency update to bio_status.json."""
    global processing_active
    while processing_active:
        try:
            # Simulate real-time processing
            raw, fs = get_mit_bih_data(sampto=360*5) # Default fs=360 for MITDB
            res = process_ecg(raw, fs)
            
            status = BioStatus(
                timestamp_ms=int(time.time() * 1000),
                hr_bpm=round(res['hr_bpm'], 2),
                hrv_sdnn_ms=round(res['sdnn_ms'], 2),
                signal_quality_score=1.0, # Placeholder
                is_stressed=res['sdnn_ms'] < 50.0 # Simple heuristic
            )
            
            with open(BIO_STATUS_FILE, "w") as f:
                json.dump(status.dict(), f)
        except Exception as e:
            print(f"Error in bio_status_loop: {e}")
            
        time.sleep(1) # Frequency: 1Hz

def maintain_baseline():
    """Hourly task to maintain user_baseline.json."""
    while processing_active:
        try:
            raw, fs = get_mit_bih_data(sampto=360*60) # 1 minute for baseline
            res = process_ecg(raw, fs)
            
            baseline = UserBaseline(
                user_id="demo_user",
                period_start_hour=datetime.now().hour,
                avg_hr=round(res['hr_bpm'], 2),
                avg_sdnn=round(res['sdnn_ms'], 2),
                last_updated=int(time.time())
            )
            
            with open(BASELINE_FILE, "w") as f:
                json.dump(baseline.dict(), f)
        except Exception as e:
            print(f"Error in maintain_baseline: {e}")
            
        time.sleep(3600) # Hourly

# Start background threads
threads = []
t1 = threading.Thread(target=bio_status_loop, daemon=True)
t2 = threading.Thread(target=maintain_baseline, daemon=True)
t1.start()
t2.start()
threads.extend([t1, t2])

@app.on_event("shutdown")
def shutdown_event():
    global processing_active
    processing_active = False

@app.get("/health")
def health():
    return {"status": "ok", "processing": processing_active}

@app.post("/ecg/pomodoro/end", response_model=PomodoroSummary)
def end_pomodoro(session_id: str, duration_min: int = 25):
    """Triggered when pomodoro ends to generate summary."""
    fs = 360
    # Simulate data for a representative 30-second segment
    raw, fs = get_mit_bih_data(sampto=fs * 30) 
    res = process_ecg(raw, fs)
    plot_path = generate_vis(res, fs, f"summary_{session_id}.png")
    
    summary = PomodoroSummary(
        session_id=session_id,
        start_time=int(time.time() - duration_min * 60),
        end_time=int(time.time()),
        avg_hr=round(res['hr_bpm'], 2),
        avg_sdnn=round(res['sdnn_ms'], 2),
        stress_percentage=30.0, # Placeholder
        total_samples=len(raw),
        plot_url=plot_path
    )
    
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary.dict(), f)
        
    return summary

@app.post("/ecg/features", response_model=EcgFeatures)
def ecg_features(segment: EcgSegment) -> EcgFeatures:
    """Manual feature extraction endpoint."""
    fs = segment.sampling_rate_hz
    # Flatten samples if provided as [[s1], [s2]]
    raw = np.array(segment.samples).flatten()
    res = process_ecg(raw, fs)
    
    # Calculate wave durations in ms
    p_wave_ms = (res['p_waves'][0][1] - res['p_waves'][0][0]) / fs * 1000 if res['p_waves'] else 0
    t_wave_ms = (res['t_waves'][0][1] - res['t_waves'][0][0]) / fs * 1000 if res['t_waves'] else 0
    qrs_ms = (res['qrs'][0][1] - res['qrs'][0][0]) / fs * 1000 if res['qrs'] else 0
    
    return EcgFeatures(
        segment_id=segment.segment_id,
        p_wave_ms=round(p_wave_ms, 2),
        t_wave_ms=round(t_wave_ms, 2),
        qrs_complex_ms=round(qrs_ms, 2),
        rr_intervals_ms=[round(x, 2) for x in res['rr_ms']],
        hrv_sdnn_ms=round(res['sdnn_ms'], 2),
        hr_bpm=round(res['hr_bpm'], 2),
        quality={"signal_ok": True}
    )
