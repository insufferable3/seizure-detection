import pandas as pd

df = pd.read_csv("results/csv/anfis_predictions.csv")

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


def overlap(pred, actual):

    ps, pe = pred
    a_s, a_e = actual

    return min(pe, a_e) > max(ps, a_s)


print("=" * 75)
print("ANFIS EVENT-LEVEL THRESHOLD SEARCH")
print("=" * 75)

results = []

for threshold in [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60
]:

    df["predicted"] = (
        df["anfis_probability"] >= threshold
    ).astype(int)

    actual_total = 0
    detected_total = 0
    predicted_total = 0
    false_total = 0

    for _, group in df.groupby("file"):

        group = group.sort_values("start")

        actual_events = get_events(
            group,
            "label"
        )

        predicted_events = get_events(
            group,
            "predicted"
        )

        detected = 0

        for actual in actual_events:

            if any(
                overlap(pred, actual)
                for pred in predicted_events
            ):
                detected += 1

        false_alarms = max(
            0,
            len(predicted_events) - detected
        )

        actual_total += len(actual_events)
        detected_total += detected
        predicted_total += len(predicted_events)
        false_total += false_alarms

    detection_rate = (
        detected_total / actual_total * 100
    )

    results.append([
        threshold,
        actual_total,
        detected_total,
        predicted_total,
        false_total,
        detection_rate
    ])

    print(
        f"\nThreshold: {threshold:.2f}"
        f"\nDetected : {detected_total}/{actual_total}"
        f"\nPredicted: {predicted_total}"
        f"\nFalse    : {false_total}"
        f"\nDetection: {detection_rate:.2f}%"
    )


result_df = pd.DataFrame(
    results,
    columns=[
        "threshold",
        "actual_events",
        "detected_events",
        "predicted_events",
        "false_alarm_events",
        "detection_rate"
    ]
)

# Prefer maximum detection, then minimum false alarms
result_df = result_df.sort_values(
    ["detection_rate", "false_alarm_events"],
    ascending=[False, True]
)

print("\n" + "=" * 75)
print("BEST EVENT CONFIGURATION")
print("=" * 75)

print(result_df.iloc[0])

result_df.to_csv(
    "anfis_event_threshold_results.csv",
    index=False
)

print("\nSaved: anfis_event_threshold_results.csv")
