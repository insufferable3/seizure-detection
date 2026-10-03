
import os
import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import mne
import torch
import torch.nn as nn
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler

# ============================================================
# NeuroGuard — END-TO-END EDF SEIZURE DETECTION
# Single-file Streamlit app
#
# Put this file in the ROOT of your seizure-detection project.
# It expects:
#   data/processed/features.csv
#   data/processed/selected_features.csv
#
# It then:
#   1. trains the same ANFIS-style architecture on your existing
#      labeled feature dataset,
#   2. accepts a NEW EDF upload,
#   3. creates 5-second EEG windows,
#   4. extracts the feature names used by your GA selection,
#   5. scales them with a fitted StandardScaler,
#   6. predicts seizure probability,
#   7. post-processes positive windows into seizure events,
#   8. displays a timeline and downloadable predictions.
#
# IMPORTANT:
# The upload prediction model is retrained on the full labeled
# dataset for deployment. Therefore these predictions are NOT the
# same thing as the held-out 80/20 evaluation numbers in the report.
# ============================================================

st.set_page_config(
    page_title="NeuroGuard — Seizure Detection",
    page_icon="🧠",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
FEATURE_FILE = ROOT / "data" / "processed" / "features.csv"
SELECTED_FILE = ROOT / "data" / "processed" / "selected_features.csv"

WINDOW_SECONDS = 5.0
THRESHOLD = 0.50
MERGE_GAP_SECONDS = 30.0
MIN_EVENT_DURATION_SECONDS = 10.0
EPOCHS = 100
LEARNING_RATE = 0.001
RANDOM_SEED = 42


# ============================================================
# STYLING
# ============================================================

st.markdown("""
<style>
    .block-container {max-width: 1400px; padding-top: 2rem;}
    .hero {
        padding: 1.4rem 1.6rem;
        border: 1px solid #26364d;
        border-radius: 18px;
        background: linear-gradient(135deg, #0e1726, #111d30);
        margin-bottom: 1.2rem;
    }
    .hero h1 {margin:0; font-size:2.1rem;}
    .hero p {color:#9fb0c8; margin:.45rem 0 0;}
    .metric-card {
        border:1px solid #26364d;
        border-radius:16px;
        padding:1rem 1.1rem;
        background:#101a2a;
        min-height:120px;
    }
    .metric-label {color:#8fa1bb; font-size:.9rem;}
    .metric-value {font-size:1.8rem; font-weight:700; margin-top:.25rem;}
    .safe {color:#56d6a0;}
    .danger {color:#ff6b7a;}
    .warn {color:#f5c66d;}
    .small {color:#8fa1bb; font-size:.85rem;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# ANFIS MODEL — same architecture as the user's project
# ============================================================

class ANFIS(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.membership_center = nn.Parameter(torch.zeros(n_features))
        self.membership_sigma = nn.Parameter(torch.ones(n_features))
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        sigma = torch.abs(self.membership_sigma) + 1e-6
        membership = torch.exp(
            -0.5 * ((x - self.membership_center) / sigma) ** 2
        )
        fuzzy_output = membership.mean(dim=1, keepdim=True)
        output = self.linear(membership)
        return output + fuzzy_output


# ============================================================
# FEATURE EXTRACTION
# ============================================================

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

BASE_FEATURES = [
    "mean",
    "std",
    "variance",
    "rms",
    "energy",
    "delta_power",
    "theta_power",
    "alpha_power",
    "beta_power",
    "gamma_power",
]


def band_power(x, sfreq, low, high):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 8:
        return 0.0

    nyq = sfreq / 2.0
    high = min(high, nyq - 1e-6)
    low = min(low, high - 1e-6)

    if high <= 0 or low <= 0 or high <= low:
        return 0.0

    nperseg = min(len(x), max(64, int(sfreq * 2)))
    try:
        freqs, psd = welch(
            x,
            fs=sfreq,
            nperseg=nperseg,
            detrend="constant",
        )
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return 0.0
        return float(np.trapezoid(psd[mask], freqs[mask]))
    except Exception:
        return 0.0


def extract_window_features(window, sfreq, n_channels=23):
    """
    Creates the same naming convention used by the project:
    ch0_mean, ch0_std, ..., ch0_gamma_power, ch1_..., etc.

    The deployed extractor intentionally computes a broad 10-feature
    set for every channel. The GA-selected CSV then chooses the exact
    107 features required by the ANFIS model.
    """
    result = {}

    channels = min(window.shape[0], n_channels)

    for ch in range(channels):
        x = np.asarray(window[ch], dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        prefix = f"ch{ch}_"

        result[prefix + "mean"] = float(np.mean(x))
        result[prefix + "std"] = float(np.std(x))
        result[prefix + "variance"] = float(np.var(x))
        result[prefix + "rms"] = float(np.sqrt(np.mean(x ** 2)))
        result[prefix + "energy"] = float(np.sum(x ** 2))

        for band, (lo, hi) in BANDS.items():
            result[prefix + band + "_power"] = band_power(
                x, sfreq, lo, hi
            )

    return result


def make_windows(raw):
    sfreq = float(raw.info["sfreq"])
    data = raw.get_data()

    # Match the project's 5-second windows.
    samples_per_window = int(round(WINDOW_SECONDS * sfreq))
    if samples_per_window <= 0:
        raise ValueError("Invalid sampling frequency.")

    windows = []
    starts = []

    for start_sample in range(
        0,
        data.shape[1] - samples_per_window + 1,
        samples_per_window,
    ):
        end_sample = start_sample + samples_per_window
        windows.append(data[:, start_sample:end_sample])
        starts.append(start_sample / sfreq)

    return windows, starts, sfreq


# ============================================================
# LOAD + TRAIN DEPLOYMENT MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_and_train_model():
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Missing {FEATURE_FILE}. "
            "Run your existing feature extraction pipeline first."
        )

    if not SELECTED_FILE.exists():
        raise FileNotFoundError(
            f"Missing {SELECTED_FILE}. "
            "Your GA-selected feature file is required."
        )

    df = pd.read_csv(FEATURE_FILE)
    selected_df = pd.read_csv(SELECTED_FILE)

    selected_features = selected_df["feature"].dropna().astype(str).tolist()
    usable = [f for f in selected_features if f in df.columns]

    if not usable:
        raise ValueError(
            "None of the GA-selected features exist in features.csv."
        )

    X = df[usable].replace([np.inf, -np.inf], np.nan)
    y = df["label"]

    valid = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid]
    y = y.loc[valid].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(
        y.to_numpy(),
        dtype=torch.float32
    ).view(-1, 1)

    model = ANFIS(len(usable))

    positives = float(y_tensor.sum().item())
    negatives = float(len(y_tensor) - positives)
    pos_weight = torch.tensor(
        [negatives / max(positives, 1.0)],
        dtype=torch.float32
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    model.train()

    for _ in range(EPOCHS):
        optimizer.zero_grad()
        logits = model(X_tensor)
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()

    return model, scaler, usable, len(df)


# ============================================================
# EVENT POST-PROCESSING
# ============================================================

def build_events(starts, probabilities, threshold):
    positive = probabilities >= threshold
    indices = np.where(positive)[0]

    if len(indices) == 0:
        return []

    raw_events = []

    event_start = starts[indices[0]]
    event_end = event_start + WINDOW_SECONDS
    previous_idx = indices[0]

    for idx in indices[1:]:
        current_start = starts[idx]

        if current_start - starts[previous_idx] <= MERGE_GAP_SECONDS + WINDOW_SECONDS:
            event_end = current_start + WINDOW_SECONDS
        else:
            raw_events.append((event_start, event_end))
            event_start = current_start
            event_end = current_start + WINDOW_SECONDS

        previous_idx = idx

    raw_events.append((event_start, event_end))

    return [
        (start, end)
        for start, end in raw_events
        if (end - start) >= MIN_EVENT_DURATION_SECONDS
    ]


# ============================================================
# UI
# ============================================================

st.markdown("""
<div class="hero">
    <h1>🧠 NeuroGuard</h1>
    <p>GA + ANFIS EEG Seizure Detection • Upload an EDF and run inference</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Model")
    st.write("**GA-selected features:** 107")
    st.write("**Window:** 5 seconds")
    st.write("**Threshold:** 0.50")
    st.write("**Merge gap:** 30 seconds")
    st.write("**Minimum event:** 10 seconds")

    st.divider()
    st.caption(
        "Research/demo software. This system is not a medical diagnostic device."
    )

# Train/load
try:
    with st.spinner("Loading GA features and preparing ANFIS deployment model..."):
        model, scaler, selected_features, training_rows = load_and_train_model()
except Exception as e:
    st.error(str(e))
    st.info(
        "Run this app from the root of your seizure-detection project so "
        "data/processed/features.csv and selected_features.csv are available."
    )
    st.stop()

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">GA Features</div>'
        f'<div class="metric-value">{len(selected_features)}</div></div>',
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Training Windows</div>'
        f'<div class="metric-value">{training_rows:,}</div></div>',
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Window Length</div>'
        f'<div class="metric-value">{WINDOW_SECONDS:.0f}s</div></div>',
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Decision Threshold</div>'
        f'<div class="metric-value">{THRESHOLD:.2f}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("## Upload EEG")

uploaded = st.file_uploader(
    "Upload a new EDF EEG recording",
    type=["edf"],
    accept_multiple_files=False,
)

if uploaded is None:
    st.info("Upload an EDF file above to run seizure detection.")
    st.stop()

# Save uploaded EDF temporarily
with tempfile.NamedTemporaryFile(
    suffix=".edf",
    delete=False
) as tmp:
    tmp.write(uploaded.getbuffer())
    temp_path = tmp.name

try:
    with st.spinner("Reading EEG with MNE..."):
        raw = mne.io.read_raw_edf(
            temp_path,
            preload=True,
            verbose=False,
        )

    windows, starts, sfreq = make_windows(raw)

    if len(windows) == 0:
        raise ValueError(
            "The EDF is shorter than one 5-second window."
        )

    st.markdown("## Recording")

    r1, r2, r3 = st.columns(3)

    duration = raw.n_times / sfreq

    with r1:
        st.metric("File", uploaded.name)

    with r2:
        st.metric("Duration", f"{duration:.1f} s")

    with r3:
        st.metric("Sampling Rate", f"{sfreq:.1f} Hz")

    st.caption(
        f"Channels detected: {len(raw.ch_names)} • "
        f"Prediction windows: {len(windows)}"
    )

    # ========================================================
    # Feature extraction
    # ========================================================

    with st.spinner(
        f"Extracting features from {len(windows)} windows..."
    ):
        rows = []

        for window, start in zip(windows, starts):
            all_features = extract_window_features(
                window,
                sfreq,
                n_channels=23,
            )

            row = {
                feature: all_features.get(feature, np.nan)
                for feature in selected_features
            }

            row["start"] = start
            row["end"] = start + WINDOW_SECONDS
            rows.append(row)

        feature_df = pd.DataFrame(rows)

        # If an uploaded EDF has fewer channels than the training data,
        # missing selected channel features cannot be reconstructed.
        missing = [
            f for f in selected_features
            if f not in feature_df.columns
            or feature_df[f].isna().all()
        ]

        if missing:
            raise ValueError(
                "The uploaded EDF does not provide enough channels/features "
                f"for this trained model. Missing {len(missing)} selected "
                "features, e.g. {missing[:5]}"
            )

        X_upload = feature_df[selected_features].replace(
            [np.inf, -np.inf],
            np.nan
        )

        X_upload = X_upload.fillna(0.0)

    # ========================================================
    # Prediction
    # ========================================================

    with st.spinner("Running GA + ANFIS inference..."):
        X_scaled = scaler.transform(X_upload)

        X_tensor = torch.tensor(
            X_scaled,
            dtype=torch.float32
        )

        with torch.no_grad():
            logits = model(X_tensor)
            probabilities = (
                torch.sigmoid(logits)
                .cpu()
                .numpy()
                .ravel()
            )

    predictions = (probabilities >= THRESHOLD).astype(int)

    feature_df["probability"] = probabilities
    feature_df["prediction"] = predictions

    events = build_events(
        starts,
        probabilities,
        THRESHOLD,
    )

    seizure_windows = int(predictions.sum())
    max_probability = float(probabilities.max())

    # ========================================================
    # SUMMARY
    # ========================================================

    st.markdown("## Detection Result")

    if len(events) > 0:
        st.error(
            f"⚠️ Potential seizure activity detected — "
            f"{len(events)} event(s)"
        )
        result_class = "danger"
    else:
        st.success("✓ No seizure event detected at the selected threshold.")
        result_class = "safe"

    a, b, c, d = st.columns(4)

    with a:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Max Probability</div>'
            f'<div class="metric-value {result_class}">{max_probability*100:.1f}%</div></div>',
            unsafe_allow_html=True,
        )

    with b:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Positive Windows</div>'
            f'<div class="metric-value">{seizure_windows}</div></div>',
            unsafe_allow_html=True,
        )

    with c:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Detected Events</div>'
            f'<div class="metric-value">{len(events)}</div></div>',
            unsafe_allow_html=True,
        )

    with d:
        positive_pct = (
            seizure_windows / len(predictions) * 100
            if len(predictions)
            else 0
        )
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Positive Windows %</div>'
            f'<div class="metric-value">{positive_pct:.1f}%</div></div>',
            unsafe_allow_html=True,
        )

    # ========================================================
    # TIMELINE
    # ========================================================

    st.markdown("## Seizure Probability Timeline")

    chart_df = feature_df[["start", "probability"]].copy()
    chart_df = chart_df.set_index("start")
    chart_df["threshold"] = THRESHOLD

    st.line_chart(
        chart_df,
        height=350,
    )

    st.caption(
        "Probability is shown per 5-second EEG window. "
        "The dashed-equivalent reference column represents the 0.50 threshold."
    )

    # ========================================================
    # EVENTS
    # ========================================================

    st.markdown("## Detected Events")

    if events:
        event_rows = []

        for i, (start, end) in enumerate(events, 1):
            mask = (
                (feature_df["start"] >= start)
                & (feature_df["start"] <= end)
            )

            event_prob = (
                feature_df.loc[mask, "probability"].max()
                if mask.any()
                else np.nan
            )

            event_rows.append({
                "Event": i,
                "Start (s)": round(start, 2),
                "End (s)": round(end, 2),
                "Duration (s)": round(end - start, 2),
                "Max Probability": round(float(event_prob), 4),
            })

        event_table = pd.DataFrame(event_rows)
        st.dataframe(
            event_table,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("No events passed the event post-processing criteria.")

    # ========================================================
    # WINDOW-LEVEL RESULTS
    # ========================================================

    st.markdown("## Window Predictions")

    display_df = feature_df[
        ["start", "end", "probability", "prediction"]
    ].copy()

    display_df.columns = [
        "Start (s)",
        "End (s)",
        "Seizure Probability",
        "Prediction",
    ]

    display_df["Prediction"] = display_df["Prediction"].map({
        0: "Normal",
        1: "Seizure",
    })

    st.dataframe(
        display_df,
        use_container_width=True,
        height=350,
        hide_index=True,
    )

    # ========================================================
    # DOWNLOAD
    # ========================================================

    output_csv = feature_df[
        ["start", "end", "probability", "prediction"]
    ].to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Prediction CSV",
        data=output_csv,
        file_name=f"{Path(uploaded.name).stem}_predictions.csv",
        mime="text/csv",
    )

finally:
    try:
        os.remove(temp_path)
    except OSError:
        pass
