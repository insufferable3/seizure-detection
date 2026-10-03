import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

seizures = {
    "chb03_01.edf": [(362, 414)],
    "chb03_15.edf": [],
    "chb03_16.edf": [],
    "chb03_17.edf": [],
    "chb03_34.edf": [(1982, 2029)],
    "chb03_35.edf": [(2592, 2656)],
    "chb03_36.edf": [(1725, 1778)]
}

window_size = 5
rows = []

for filename, seizure_intervals in seizures.items():

    print(f"\nProcessing {filename}...")

    raw = mne.io.read_raw_edf(
        filename,
        preload=True,
        verbose=False
    )

    data = raw.get_data()
    sfreq = raw.info["sfreq"]

    samples_per_window = int(window_size * sfreq)

    for start in range(
        0,
        data.shape[1] - samples_per_window,
        samples_per_window
    ):

        end = start + samples_per_window

        start_time = start / sfreq
        end_time = end / sfreq

        # -------------------------
        # Label window
        # -------------------------

        label = 0

        for seizure_start, seizure_end in seizure_intervals:

            overlap_start = max(start_time, seizure_start)
            overlap_end = min(end_time, seizure_end)

            overlap = max(0, overlap_end - overlap_start)

            if overlap >= window_size / 2:
                label = 1
                break

        # -------------------------
        # Feature extraction
        # -------------------------

        features = {}

        for ch in range(data.shape[0]):

            signal = data[ch, start:end]

            prefix = f"ch{ch}"

            # Time-domain features
            features[f"{prefix}_mean"] = np.mean(signal)

            features[f"{prefix}_std"] = np.std(signal)

            features[f"{prefix}_variance"] = np.var(signal)

            features[f"{prefix}_rms"] = np.sqrt(
                np.mean(signal ** 2)
            )

            features[f"{prefix}_energy"] = np.sum(
                signal ** 2
            )

            # Frequency-domain features
            frequencies, power = welch(
                signal,
                fs=sfreq
            )

            bands = {
                "delta": (0.5, 4),
                "theta": (4, 8),
                "alpha": (8, 13),
                "beta": (13, 30),
                "gamma": (30, 100)
            }

            for band, (low, high) in bands.items():

                mask = (
                    (frequencies >= low) &
                    (frequencies < high)
                )

                features[
                    f"{prefix}_{band}_power"
                ] = np.trapz(
                    power[mask],
                    frequencies[mask]
                )

        features["file"] = filename
        features["start"] = start_time
        features["end"] = end_time
        features["label"] = label

        rows.append(features)


# -------------------------
# Create final dataset
# -------------------------

df = pd.DataFrame(rows)

df.to_csv("features.csv", index=False)

print("\n==============================")
print("FEATURE EXTRACTION COMPLETE")
print("==============================")

print("Dataset shape:", df.shape)

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nFiles:")
print(df["file"].value_counts())

print("\nSaved as features.csv")