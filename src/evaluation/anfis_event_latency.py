import pandas as pd

df = pd.read_csv("results/csv/anfis_predictions.csv")

THRESHOLD = 0.45
MERGE_GAP = 30
MIN_DURATION = 10


def merge_events(events):
    if not events:
        return []

    events = sorted(events)
    merged = [list(events[0])]

    for start, end in events[1:]:
        if start - merged[-1][1] <= MERGE_GAP:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [
        (s, e)
        for s, e in merged
        if e - s >= MIN_DURATION
    ]


def get_events(group, column):

    events = []
    start = None
    end = None

    for _, row in group.iterrows():

        if row[column] == 1:

            if start is None:
                start = row["start"]

            end = row["end"]

        else:

            if start is not None:
                events.append((start, end))
                start = None
                end = None

    if start is not None:
        events.append((start, end))

    return merge_events(events)


df["predicted"] = (
    df["anfis_probability"] >= THRESHOLD
).astype(int)


latencies = []
onset_errors = []
offset_errors = []
overlaps = []

print("=" * 80)
print("ANFIS SEIZURE DETECTION — LATENCY & OVERLAP")
print("=" * 80)

for file_name, group in df.groupby("file"):

    group = group.sort_values("start")

    actual_events = get_events(group, "label")
    predicted_events = get_events(group, "predicted")

    for actual in actual_events:

        a_start, a_end = actual

        matching = []

        for pred in predicted_events:

            p_start, p_end = pred

            overlap_start = max(a_start, p_start)
            overlap_end = min(a_end, p_end)

            if overlap_end > overlap_start:
                matching.append(pred)

        if not matching:
            continue

        # Pick prediction with largest overlap
        pred = max(
            matching,
            key=lambda x: min(x[1], a_end) -
                          max(x[0], a_start)
        )

        p_start, p_end = pred

        latency = p_start - a_start
        onset_error = p_start - a_start
        offset_error = p_end - a_end

        intersection = max(
            0,
            min(a_end, p_end) -
            max(a_start, p_start)
        )

        union = (
            max(a_end, p_end) -
            min(a_start, p_start)
        )

        overlap = (
            intersection / union * 100
            if union > 0
            else 0
        )

        latencies.append(latency)
        onset_errors.append(onset_error)
        offset_errors.append(offset_error)
        overlaps.append(overlap)

        print("\n" + "-" * 70)
        print(file_name)

        print(
            f"Actual seizure     : "
            f"{a_start:.1f}s - {a_end:.1f}s"
        )

        print(
            f"Predicted seizure  : "
            f"{p_start:.1f}s - {p_end:.1f}s"
        )

        print(
            f"Detection latency  : {latency:.1f}s"
        )

        print(
            f"Onset error        : {onset_error:+.1f}s"
        )

        print(
            f"Offset error       : {offset_error:+.1f}s"
        )

        print(
            f"Overlap            : {overlap:.2f}%"
        )


print("\n" + "=" * 80)
print("OVERALL ANFIS TIMING RESULTS")
print("=" * 80)

print(
    f"Mean detection latency : "
    f"{sum(latencies) / len(latencies):.2f}s"
)

print(
    f"Median detection latency : "
    f"{pd.Series(latencies).median():.2f}s"
)

print(
    f"Mean onset error       : "
    f"{sum(onset_errors) / len(onset_errors):.2f}s"
)

print(
    f"Mean offset error      : "
    f"{sum(offset_errors) / len(offset_errors):.2f}s"
)

print(
    f"Mean event overlap     : "
    f"{sum(overlaps) / len(overlaps):.2f}%"
)

print("=" * 80)

results = pd.DataFrame({
    "detection_latency": latencies,
    "onset_error": onset_errors,
    "offset_error": offset_errors,
    "overlap": overlaps
})

results.to_csv(
    "anfis_event_latency_results.csv",
    index=False
)

print("Saved: anfis_event_latency_results.csv")
