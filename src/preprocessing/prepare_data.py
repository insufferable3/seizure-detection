import mne
import numpy as np
import pandas as pd

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

        label = 0

        # Check whether this window overlaps a seizure
        for seizure_start, seizure_end in seizure_intervals:

            overlap_start = max(start_time, seizure_start)
            overlap_end = min(end_time, seizure_end)

            overlap = max(0, overlap_end - overlap_start)

            if overlap >= window_size / 2:
                label = 1
                break

        rows.append({
            "file": filename,
            "start": start_time,
            "end": end_time,
            "label": label
        })

    print(f"Windows created: {len(rows)}")


df = pd.DataFrame(rows)

df.to_csv("windows.csv", index=False)

print("\n==============================")
print("DATASET CREATED")
print("==============================")

print("Total windows:", len(df))

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nPer-file distribution:")
print(
    df.groupby(["file", "label"])
      .size()
      .unstack(fill_value=0)
)

print("\nSaved as windows.csv")