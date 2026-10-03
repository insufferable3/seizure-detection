from pathlib import Path
import uuid
import numpy as np
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



def _read_edf_header(edf_path):
    """Read the fixed EDF header directly.

    This intentionally avoids MNE for preflight validation. Some EDF files can
    contain metadata quirks that make a full MNE open fail even though the EDF
    signal header is readable. The model itself still uses MNE for inference.
    """
    path = Path(edf_path)
    with path.open("rb") as f:
        header = f.read(256)
        if len(header) < 256:
            raise ValueError("The file is too small to be a valid EDF recording.")

        try:
            version = header[0:8].decode("ascii", errors="ignore").strip()
            header_bytes = int(header[184:192].decode("ascii", errors="ignore").strip())
            n_records = int(header[236:244].decode("ascii", errors="ignore").strip())
            record_duration = float(header[244:252].decode("ascii", errors="ignore").strip())
            n_signals = int(header[252:256].decode("ascii", errors="ignore").strip())
        except (ValueError, TypeError):
            raise ValueError("The EDF header could not be parsed. Please upload a standard EDF/EDF+ recording.")

        if header_bytes < 256 or n_signals < 1:
            raise ValueError("The EDF header contains invalid channel metadata.")

        # Read the complete per-signal header.
        f.seek(0)
        signal_header = f.read(header_bytes)
        if len(signal_header) < header_bytes:
            raise ValueError("The EDF header is incomplete or truncated.")

    labels_start = 256
    labels_end = labels_start + 16 * n_signals
    labels_raw = signal_header[labels_start:labels_end]
    labels = [
        labels_raw[i * 16:(i + 1) * 16].decode("latin-1", errors="ignore").strip()
        for i in range(n_signals)
    ]

    # Samples per record is the 216-byte block in the per-signal header.
    # Offsets are relative to the per-signal header:
    # labels 0, transducer 16, phys_dim 96, phys_min 104, phys_max 112,
    # dig_min 120, dig_max 128, prefilter 136, samples/record 216.
    samples_offset = 256 + (216 * n_signals)
    samples_raw = signal_header[samples_offset:samples_offset + 8 * n_signals]
    samples_per_record = []
    for i in range(n_signals):
        raw = samples_raw[i * 8:(i + 1) * 8]
        try:
            samples_per_record.append(int(raw.decode("ascii", errors="ignore").strip()))
        except ValueError:
            samples_per_record.append(0)

    if record_duration <= 0 or n_records <= 0 or not any(v > 0 for v in samples_per_record):
        raise ValueError("The EDF contains invalid recording-duration or sampling metadata.")

    duration_sec = float(n_records * record_duration)
    sfreqs = [v / record_duration for v in samples_per_record if v > 0]
    sfreq = float(sfreqs[0]) if sfreqs else 0.0

    return {
        "version": version,
        "channels": n_signals,
        "channel_names": labels,
        "sampling_frequency": sfreq,
        "duration_sec": duration_sec,
        "samples_per_record": samples_per_record,
        "record_duration": record_duration,
    }

def validate_edf(edf_path):
    """Preflight validation using the EDF header only.

    This avoids opening the entire recording with MNE during upload validation,
    so a metadata quirk cannot leak a low-level parser exception into the UI.
    """
    path = Path(edf_path)
    if path.suffix.lower() != ".edf":
        raise ValueError("Only .edf files are supported.")
    if not path.exists():
        raise FileNotFoundError("EDF file was not found.")

    _, _, selected = _load_artifacts()
    info = _read_edf_header(path)

    required = [
        int(x.split("_")[0][2:])
        for x in selected
        if x.startswith("ch") and "_" in x and x.split("_")[0][2:].isdigit()
    ]
    minimum = max(required) + 1 if required else 1

    if info["channels"] < minimum:
        raise ValueError(
            f"This recording has {info['channels']} channels. "
            f"NeuroGuard's trained feature layout requires at least {minimum} channels."
        )

    return {
        "channels": info["channels"],
        "sampling_frequency": info["sampling_frequency"],
        "duration_sec": info["duration_sec"],
        "channel_names": info["channel_names"],
        "compatible": True,
    }


def predict_edf(edf_path, threshold=DEFAULT_THRESHOLD, merge_gap_sec=DEFAULT_MERGE_GAP_SEC, min_duration_sec=DEFAULT_MIN_DURATION_SEC):
    model, scaler, selected = _load_artifacts()
    threshold = float(np.clip(threshold, 0.01, 0.99))
    merge_gap_sec = float(max(0, merge_gap_sec))
    min_duration_sec = float(max(0, min_duration_sec))

    # Keep the EDF backed by its file and read only one analysis window at a
    # time. preload=True plus get_data() materialized the entire recording in
    # RAM before feature extraction, which can exceed small hosted instances.
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    try:
        sf = float(raw.info["sfreq"])
        n_channels = len(raw.ch_names)
        total_samples = int(raw.n_times)
        total_seconds = total_samples / sf

        required_indices = [
            int(x.split("_")[0][2:]) for x in selected
            if x.startswith("ch") and "_" in x and x.split("_")[0][2:].isdigit()
        ]
        if required_indices and max(required_indices) >= n_channels:
            raise ValueError(f"This EDF has {n_channels} channels, but the trained feature set requires at least {max(required_indices)+1} channels.")

        window_samples = int(round(WINDOW_SEC * sf))
        n_windows = total_samples // window_samples
        if not n_windows:
            raise ValueError("EDF is shorter than one 5-second analysis window.")

        starts = [i * WINDOW_SEC for i in range(n_windows)]
        probabilities_list = []
        feature_columns = None
        feature_batch = []
        batch_size = 128

        def predict_batch():
            if not feature_batch:
                return
            values = np.asarray(feature_batch, dtype=float)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            scaled = scaler.transform(values)
            probabilities_list.extend(model.predict_proba(scaled)[:, 1].astype(float).tolist())
            feature_batch.clear()

        for i in range(n_windows):
            start_sample = i * window_samples
            stop_sample = start_sample + window_samples
            window = raw.get_data(start=start_sample, stop=stop_sample)
            features = _extract_window_features(window, sf)
            if feature_columns is None:
                missing = [name for name in selected if name not in features]
                if missing:
                    raise ValueError(f"Uploaded EDF does not match the trained feature layout. Missing selected feature: {missing[0]}")
                feature_columns = selected
            # Keep only a small bounded batch of feature rows in memory.
            feature_batch.append([features[name] for name in feature_columns])
            if len(feature_batch) >= batch_size:
                predict_batch()
        predict_batch()
        probabilities = np.asarray(probabilities_list, dtype=float)
    finally:
        raw.close()

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
        "channels": int(n_channels), "windows": windows, "job_id": uuid.uuid4().hex,
    }
