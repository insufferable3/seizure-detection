import pandas as pd

# ============================================
# SETTINGS
# ============================================

INPUT_FILE = "cv_predictions_detailed.csv"

THRESHOLD = 0.35

# If two predicted windows are within this gap,
# treat them as the same event.
MAX_GAP = 20.0


# ============================================
# LOAD
# ============================================

df = pd.read_csv(INPUT_FILE)

df = df.sort_values(
    ["file", "start"]
).reset_index(drop=True)

# Use probability instead of existing predicted column
df["predicted"] = (
    df["seizure_probability"] >= THRESHOLD
).astype(int)


# ============================================
# EVENT MERGING
# ============================================

def get_events(group):

    events = []

    current_start = None
    current_end = None

    for _, row in group.iterrows():

        start = row["start"]
        end = row["end"]
        pred = row["predicted"]

        if pred == 1:

            if current_start is None:

                current_start = start
                current_end = end

            else:

                gap = start - current_end

                if gap <= MAX_GAP:
                    current_end = end

                else:
                    events.append(
                        (current_start, current_end)
                    )

                    current_start = start
                    current_end = end

    # Save final event
    if current_start is not None:
        events.append(
            (current_start, current_end)
        )

    return events


# ============================================
# ACTUAL EVENTS
# ============================================

def get_actual_events(group):

    events = []

    current_start = None
    current_end = None

    for _, row in group.iterrows():

        start = row["start"]
        end = row["end"]
        actual = row["actual"]

        if actual == 1:

            if current_start is None:

                current_start = start
                current_end = end

            else:

                gap = start - current_end

                if gap <= MAX_GAP:
                    current_end = end

                else:

                    events.append(
                        (current_start, current_end)
                    )

                    current_start = start
                    current_end = end

    if current_start is not None:

        events.append(
            (current_start, current_end)
        )

    return events


# ============================================
# EVENT OVERLAP
# ============================================

def overlaps(pred_event, actual_event):

    p_start, p_end = pred_event
    a_start, a_end = actual_event

    return (
        max(p_start, a_start)
        < min(p_end, a_end)
    )


# ============================================
# EVALUATION
# ============================================

total_actual = 0
total_detected = 0
total_predicted = 0
total_false_alarm = 0


print("\n")
print("=" * 70)
print("POST-PROCESSED EVENT-LEVEL ANALYSIS")
print("=" * 70)

for file_name, group in df.groupby("file"):

    actual_events = get_actual_events(group)

    predicted_events = get_events(group)

    detected = 0

    for actual_event in actual_events:

        found = False

        for pred_event in predicted_events:

            if overlaps(pred_event, actual_event):
                found = True
                break

        if found:
            detected += 1

    false_alarms = 0

    for pred_event in predicted_events:

        found = False

        for actual_event in actual_events:

            if overlaps(pred_event, actual_event):
                found = True
                break

        if not found:
            false_alarms += 1

    total_actual += len(actual_events)
    total_predicted += len(predicted_events)
    total_detected += detected
    total_false_alarm += false_alarms

    print("\n" + "-" * 60)
    print(file_name)

    print(
        "Actual events     :",
        len(actual_events)
    )

    print(
        "Predicted events  :",
        len(predicted_events)
    )

    print(
        "Detected events   :",
        detected
    )

    print(
        "False alarms      :",
        false_alarms
    )

    if actual_events:

        print("\nActual:")

        for event in actual_events:
            print(
                f"  {event[0]:.1f}s - {event[1]:.1f}s"
            )

    if predicted_events:

        print("\nPredicted after merging:")

        for event in predicted_events:
            print(
                f"  {event[0]:.1f}s - {event[1]:.1f}s"
            )


# ============================================
# FINAL METRICS
# ============================================

if total_actual > 0:

    detection_rate = (
        total_detected /
        total_actual
    )

else:

    detection_rate = 0


print("\n")
print("=" * 70)
print("FINAL POST-PROCESSED RESULTS")
print("=" * 70)

print(
    f"Threshold              : {THRESHOLD}"
)

print(
    f"Merge gap              : {MAX_GAP}s"
)

print(
    f"Actual seizure events  : {total_actual}"
)

print(
    f"Detected events        : {total_detected}"
)

print(
    f"Predicted events       : {total_predicted}"
)

print(
    f"False alarm events     : {total_false_alarm}"
)

print(
    f"Event detection rate   : "
    f"{detection_rate * 100:.2f}%"
)