from pathlib import Path
import uuid
import numpy as np
import pandas as pd
import mne
import joblib
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "neuroguard_rf.joblib"
SCALER_PATH = ROOT / "models" / "neuroguard_scaler.joblib"
SELECTED_PATH = ROOT / "models" / "selected_features.txt"

WINDOW_SEC = 5.0
DEFAULT_THRESHOLD = 0.50
DEFAULT_MERGE_GAP_SEC = 30.0
DEFAULT_MIN_DURATION_SEC = 10.0
BANDS = {
    "delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0), "gamma": (30.0, 45.0),
}


def _load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Missing scaler: {SCALER_PATH}")
    if not SELECTED_PATH.exists():
        raise FileNotFoundError(f"Missing selected feature list: {SELECTED_PATH}")
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH), [x.strip() for x in SELECTED_PATH.read_text().splitlines() if x.strip()]


def _bandpower(x, sf, low, high):
    freqs, psd = welch(x, fs=sf, nperseg=min(len(x), int(sf * 2)))
    mask = (freqs >= low) & (freqs <= high)
    return float(np.trapz(psd[mask], freqs[mask])) if np.any(mask) else 0.0


def _extract_window_features(data, sf):
    values = {}
    for ch_idx in range(data.shape[0]):
        x = np.asarray(data[ch_idx], dtype=float)
        prefix = f"ch{ch_idx}"
        values[f"{prefix}_mean"] = float(np.mean(x))
        values[f"{prefix}_std"] = float(np.std(x))
        values[f"{prefix}_variance"] = float(np.var(x))
        values[f"{prefix}_rms"] = float(np.sqrt(np.mean(x ** 2)))
        values[f"{prefix}_energy"] = float(np.sum(x ** 2))
        for band, (lo, hi) in BANDS.items():
            values[f"{prefix}_{band}_power"] = _bandpower(x, sf, lo, hi)
    return values


def _merge_events(intervals, merge_gap_sec, min_duration_sec):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for start, end, confidence in intervals[1:]:
        if start - merged[-1][1] <= merge_gap_sec:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2] = max(merged[-1][2], confidence)
        else:
            merged.append([start, end, confidence])
    return [
        {"start": round(s, 2), "end": round(e, 2), "duration": round(e-s, 2), "confidence": round(c, 4)}
        for s, e, c in merged if e - s >= min_duration_sec
    ]


def validate_edf(edf_path):
    """Validate extension and channel compatibility without running inference."""
    path = Path(edf_path)
    if path.suffix.lower() != ".edf":
        raise ValueError("Only .edf files are supported.")
    if not path.exists():
        raise FileNotFoundError("EDF file was not found.")
    _, _, selected = _load_artifacts()
    raw = mne.io.read_raw_edf(str(path), preload=False, verbose=False)
    channels = len(raw.ch_names)
    required = [int(x.split("_")[0][2:]) for x in selected if x.startswith("ch") and "_" in x]
    minimum = max(required) + 1 if required else 1
    if channels < minimum:
        raise ValueError(f"EDF has {channels} channels; this trained model requires at least {minimum} channels.")
    return {"channels": channels, "sampling_frequency": float(raw.info["sfreq"]), "duration_sec": float(raw.n_times / raw.info["sfreq"])}


def predict_edf(edf_path, threshold=DEFAULT_THRESHOLD, merge_gap_sec=DEFAULT_MERGE_GAP_SEC, min_duration_sec=DEFAULT_MIN_DURATION_SEC):
    model, scaler, selected = _load_artifacts()
    threshold = float(np.clip(threshold, 0.01, 0.99))
    merge_gap_sec = float(max(0, merge_gap_sec))
    min_duration_sec = float(max(0, min_duration_sec))

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data = raw.get_data()
    sf = float(raw.info["sfreq"])
    total_seconds = data.shape[1] / sf

    required_indices = [int(x.split("_")[0][2:]) for x in selected if x.startswith("ch") and "_" in x]
    if required_indices and max(required_indices) >= data.shape[0]:
        raise ValueError(f"This EDF has {data.shape[0]} channels, but the trained feature set requires at least {max(required_indices)+1} channels.")

    window_samples = int(round(WINDOW_SEC * sf))
    rows, starts = [], []
    n_windows = data.shape[1] // window_samples
    for i in range(n_windows):
        start_sample = i * window_samples
        window = data[:, start_sample:start_sample + window_samples]
        rows.append(_extract_window_features(window, sf))
        starts.append(i * WINDOW_SEC)
    if not rows:
        raise ValueError("EDF is shorter than one 5-second analysis window.")

    frame = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    missing = [f for f in selected if f not in frame.columns]
    if missing:
        raise ValueError(f"Uploaded EDF does not match the trained feature layout. Missing selected feature: {missing[0]}")

    probabilities = model.predict_proba(scaler.transform(frame[selected].to_numpy(dtype=float)))[:, 1]
    positive = probabilities >= threshold

    raw_intervals = []
    for start, prob, is_positive in zip(starts, probabilities, positive):
        if is_positive:
            raw_intervals.append((start, min(start + WINDOW_SEC, total_seconds), float(prob)))
    events = _merge_events(raw_intervals, merge_gap_sec, min_duration_sec)

    windows = [{"start": round(float(s), 2), "end": round(min(float(s + WINDOW_SEC), total_seconds), 2), "probability": round(float(p), 6), "prediction": int(p >= threshold)} for s, p in zip(starts, probabilities)]
    return {
        "filename": Path(edf_path).name, "windows_analyzed": int(n_windows), "positive_windows": int(positive.sum()),
        "peak_probability": float(probabilities.max()), "detected_events": len(events), "events": events,
        "threshold": threshold, "window_sec": WINDOW_SEC, "merge_gap_sec": merge_gap_sec,
        "min_duration_sec": min_duration_sec, "sampling_frequency": sf, "duration_sec": total_seconds,
        "channels": int(data.shape[0]), "windows": windows, "job_id": uuid.uuid4().hex,
    }
