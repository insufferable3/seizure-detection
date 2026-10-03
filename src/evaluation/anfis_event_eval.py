import pandas as pd

THRESHOLD = 0.45
MERGE_GAP = 30
MIN_DURATION = 10

df = pd.read_csv("results/csv/anfis_predictions.csv")

df["predicted"] = (
    df["anfis_probability"] >= THRESHOLD
).astype(int)


def merge_events(events, gap=30):
    if not events:
        return []

    events = sorted(events)

    merged = [list(events[0])]

    for start, end in events[1:]:

        if start - merged[-1][1] <= gap:
            merged[-1][1] = max(
                merged[-1][1],
                end
            )
        else:
            merged.append([start, end])

    return [
        (start, end)
        for start, end in merged
        if end - start >= MIN_DURATION
    ]


def get_events(group, column):

    events = []

    current_start = None
    current_end = None

    for _, row in group.iterrows():

        if row[column] == 1:

            if current_start is None:
                current_start = row["start"]

            current_end = row["end"]

        else:

            if current_start is not None:
                events.append(
                    (current_start, current_end)
                )

                current_start = None
                current_end = None

    if current_start is not None:
        events.append(
            (current_start, current_end)
        )

    return merge_events(
        events,
        MERGE_GAP
    )


def overlaps(pred, actual):

    p_start, p_end = pred
    a_start, a_end = actual

    return (
        min(p_end, a_end) -
        max(p_start, a_start)
    ) > 0


print("=" * 70)
print("ANFIS EVENT-LEVEL SEIZURE DETECTION")
print("=" * 70)

print(f"Threshold          : {THRESHOLD}")
print(f"Merge gap          : {MERGE_GAP}s")
print(f"Minimum duration   : {MIN_DURATION}s")

total_actual = 0
total_detected = 0
total_predicted = 0
total_false = 0
total_missed = 0

for file_name, group in df.groupby("file"):

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

        found = any(
            overlaps(pred, actual)
            for pred in predicted_events
        )

        if found:
            detected += 1

    false_alarms = max(
        0,
        len(predicted_events) - detected
    )

    missed = len(actual_events) - detected

    total_actual += len(actual_events)
    total_detected += detected
    total_predicted += len(predicted_events)
    total_false += false_alarms
    total_missed += missed

    print("\n" + "-" * 60)
    print(file_name)

    print("Actual events     :", len(actual_events))
    print("Detected events   :", detected)
    print("Predicted events  :", len(predicted_events))
    print("False alarms      :", false_alarms)
    print("Missed events     :", missed)

    if actual_events:
        print("\nActual:")
        for start, end in actual_events:
            print(
                f"  {start:.1f}s - {end:.1f}s"
            )

    if predicted_events:
        print("\nPredicted:")
        for start, end in predicted_events:
            print(
                f"  {start:.1f}s - {end:.1f}s"
            )


detection_rate = (
    total_detected / total_actual * 100
    if total_actual > 0
    else 0
)

print("\n" + "=" * 70)
print("FINAL ANFIS EVENT RESULTS")
print("=" * 70)

print("Actual seizure events :", total_actual)
print("Detected events       :", total_detected)
print("Predicted events      :", total_predicted)
print("False alarm events    :", total_false)
print("Missed seizure events :", total_missed)
print(
    f"Event detection rate  : {detection_rate:.2f}%"
)

print("=" * 70)
