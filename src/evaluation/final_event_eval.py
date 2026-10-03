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

detected = 0
missed = 0
false_alarms = 0

for actual in actual_events:

    if any(overlaps(pred, actual) for pred in predicted_events):
        detected += 1
    else:
        missed += 1


for pred in predicted_events:

    if not any(overlaps(pred, actual) for actual in actual_events):
        false_alarms += 1


print()
print("=" * 70)
print("FINAL SEIZURE EVENT DETECTION")
print("=" * 70)

print(f"Threshold              : {THRESHOLD}")
print(f"Merge gap              : {MERGE_GAP}s")
print(f"Minimum duration       : {MIN_DURATION}s")

print()
print(f"Actual seizure events  : {len(actual_events)}")
print(f"Detected events        : {detected}")
print(f"Missed events          : {missed}")
print(f"Predicted events       : {len(predicted_events)}")
print(f"False alarm events     : {false_alarms}")

if len(actual_events) > 0:
    detection_rate = 100 * detected / len(actual_events)
else:
    detection_rate = 0

print(f"Event detection rate   : {detection_rate:.2f}%")

print()
print("=" * 70)
print("EVENT DETAILS")
print("=" * 70)

for file_name in sorted(df["file"].unique()):

    actual = [
        x for x in actual_events
        if x[0] == file_name
    ]

    predicted = [
        x for x in predicted_events
        if x[0] == file_name
    ]

    print()
    print(file_name)

    print("Actual:")
    for _, start, end in actual:
        print(f"  {start:.1f}s - {end:.1f}s")

    print("Predicted:")
    for _, start, end in predicted:
        print(f"  {start:.1f}s - {end:.1f}s")


# Save final event table

rows = []

for file_name, start, end in predicted_events:

    detected_event = any(
        overlaps(
            (file_name, start, end),
            actual
        )
        for actual in actual_events
    )

    rows.append({
        "file": file_name,
        "predicted_start": start,
        "predicted_end": end,
        "duration": end - start,
        "detected_actual_event": detected_event
    })


pd.DataFrame(rows).to_csv(
    "final_event_predictions.csv",
    index=False
)

print()
print("Saved as final_event_predictions.csv")
