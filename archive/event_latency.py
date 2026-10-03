import pandas as pd

INPUT_FILE = "results/csv/cv_all_features_predictions.csv"

THRESHOLD = 0.50
MERGE_GAP = 30
MIN_DURATION = 10


def get_events(df, use_prediction=False):

    events = []

    for file_name, g in df.groupby("file"):

        g = g.sort_values("start").reset_index(drop=True)

        if use_prediction:
            active = g["seizure_probability"] >= THRESHOLD
        else:
            active = g["actual"] == 1

        current = None

        for i in range(len(g)):

            if active.iloc[i]:

                start = float(g.loc[i, "start"])
                end = float(g.loc[i, "end"])

                if current is None:
                    current = [start, end]

                else:

                    gap = start - current[1]

                    if gap <= MERGE_GAP:
                        current[1] = end

                    else:

                        duration = current[1] - current[0]

                        if duration >= MIN_DURATION:
                            events.append(
                                (file_name, current[0], current[1])
                            )

                        current = [start, end]

        if current is not None:

            duration = current[1] - current[0]

            if duration >= MIN_DURATION:
                events.append(
                    (file_name, current[0], current[1])
                )

    return events


def overlaps(pred, actual):

    pf, ps, pe = pred
    af, a_s, ae = actual

    if pf != af:
        return False

    return min(pe, ae) > max(ps, a_s)


df = pd.read_csv(INPUT_FILE)

actual_events = get_events(df, False)
predicted_events = get_events(df, True)


rows = []


for actual in actual_events:

    file_name, actual_start, actual_end = actual

    matches = [
        pred for pred in predicted_events
        if overlaps(pred, actual)
    ]

    if not matches:
        continue

    # Earliest predicted event overlapping the seizure
    pred = min(matches, key=lambda x: x[1])

    _, pred_start, pred_end = pred

    onset_error = pred_start - actual_start

    offset_error = pred_end - actual_end

    detection_latency = max(0, onset_error)

    actual_duration = actual_end - actual_start
    predicted_duration = pred_end - pred_start

    overlap_start = max(actual_start, pred_start)
    overlap_end = min(actual_end, pred_end)

    overlap_duration = max(
        0,
        overlap_end - overlap_start
    )

    overlap_percentage = (
        100 * overlap_duration / actual_duration
    )

    rows.append({
        "file": file_name,
        "actual_start": actual_start,
        "actual_end": actual_end,
        "predicted_start": pred_start,
        "predicted_end": pred_end,
        "detection_latency": detection_latency,
        "onset_error": onset_error,
        "offset_error": offset_error,
        "actual_duration": actual_duration,
        "predicted_duration": predicted_duration,
        "overlap_duration": overlap_duration,
        "overlap_percentage": overlap_percentage
    })


results = pd.DataFrame(rows)


print()
print("=" * 80)
print("SEIZURE DETECTION LATENCY & TIMING ANALYSIS")
print("=" * 80)

print()

for _, r in results.iterrows():

    print("-" * 80)

    print(r["file"])

    print(
        f"Actual seizure     : "
        f"{r['actual_start']:.1f}s - {r['actual_end']:.1f}s"
    )

    print(
        f"Predicted seizure  : "
        f"{r['predicted_start']:.1f}s - {r['predicted_end']:.1f}s"
    )

    print(
        f"Detection latency  : "
        f"{r['detection_latency']:.1f}s"
    )

    print(
        f"Onset error        : "
        f"{r['onset_error']:+.1f}s"
    )

    print(
        f"Offset error       : "
        f"{r['offset_error']:+.1f}s"
    )

    print(
        f"Overlap            : "
        f"{r['overlap_percentage']:.2f}%"
    )


print()
print("=" * 80)
print("OVERALL TIMING RESULTS")
print("=" * 80)

print(
    f"Mean detection latency : "
    f"{results['detection_latency'].mean():.2f}s"
)

print(
    f"Median detection latency : "
    f"{results['detection_latency'].median():.2f}s"
)

print(
    f"Mean onset error       : "
    f"{results['onset_error'].mean():.2f}s"
)

print(
    f"Mean offset error      : "
    f"{results['offset_error'].mean():.2f}s"
)

print(
    f"Mean event overlap     : "
    f"{results['overlap_percentage'].mean():.2f}%"
)


results.to_csv(
    "event_latency_results.csv",
    index=False
)

print()
print("Saved as event_latency_results.csv")
