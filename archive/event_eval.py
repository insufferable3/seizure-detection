import pandas as pd
import numpy as np

# ============================================================
# SETTINGS
# ============================================================

CSV_FILE = "cv_predictions_detailed.csv"

# Use the threshold we already tested
THRESHOLD = 0.35

# Minimum number of consecutive positive windows
# required to call something a seizure event.
MIN_CONSECUTIVE = 1


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_FILE)

required_cols = [
    "file",
    "start",
    "end",
    "actual",
    "seizure_probability"
]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

print("Total windows:", len(df))


# ============================================================
# CREATE THRESHOLD PREDICTIONS
# ============================================================

df["predicted_threshold"] = (
    df["seizure_probability"] >= THRESHOLD
).astype(int)


# ============================================================
# FUNCTION: COUNT EVENTS
# ============================================================

def get_events(file_df, column):

    values = file_df[column].values
    events = []

    in_event = False
    start_time = None
    end_time = None

    for i, value in enumerate(values):

        if value == 1 and not in_event:

            in_event = True
            start_time = file_df.iloc[i]["start"]
            end_time = file_df.iloc[i]["end"]

        elif value == 1 and in_event:

            end_time = file_df.iloc[i]["end"]

        elif value == 0 and in_event:

            events.append(
                (start_time, end_time)
            )

            in_event = False
            start_time = None
            end_time = None

    # Handle event ending at final window
    if in_event:
        events.append(
            (start_time, end_time)
        )

    return events


# ============================================================
# EVENT OVERLAP
# ============================================================

def events_overlap(actual_event, predicted_event):

    actual_start, actual_end = actual_event
    pred_start, pred_end = predicted_event

    return (
        pred_start < actual_end
        and pred_end > actual_start
    )


# ============================================================
# ANALYSIS
# ============================================================

total_actual_events = 0
total_detected_events = 0
total_predicted_events = 0
total_false_alarm_events = 0

print("\n")
print("=" * 70)
print("EVENT-LEVEL SEIZURE ANALYSIS")
print("=" * 70)

for file_name, file_df in df.groupby("file"):

    file_df = file_df.sort_values("start").reset_index(drop=True)

    actual_events = get_events(
        file_df,
        "actual"
    )

    predicted_events = get_events(
        file_df,
        "predicted_threshold"
    )

    detected = 0

    for actual_event in actual_events:

        found = False

        for predicted_event in predicted_events:

            if events_overlap(
                actual_event,
                predicted_event
            ):
                found = True
                break

        if found:
            detected += 1

    false_alarms = 0

    for predicted_event in predicted_events:

        found = False

        for actual_event in actual_events:

            if events_overlap(
                actual_event,
                predicted_event
            ):
                found = True
                break

        if not found:
            false_alarms += 1

    total_actual_events += len(actual_events)
    total_predicted_events += len(predicted_events)
    total_detected_events += detected
    total_false_alarm_events += false_alarms

    print("\n" + "-" * 60)
    print(file_name)

    print(
        "Actual seizure events    :",
        len(actual_events)
    )

    print(
        "Predicted seizure events :",
        len(predicted_events)
    )

    print(
        "Detected seizure events  :",
        detected
    )

    print(
        "False alarm events       :",
        false_alarms
    )

    # Show actual seizure timings
    if len(actual_events) > 0:

        print("\nActual events:")

        for event in actual_events:
            print(
                f"  {event[0]:.1f}s - {event[1]:.1f}s"
            )

    # Show detected timings
    if len(predicted_events) > 0:

        print("\nPredicted events:")

        for event in predicted_events:
            print(
                f"  {event[0]:.1f}s - {event[1]:.1f}s"
            )


# ============================================================
# FINAL METRICS
# ============================================================

if total_actual_events > 0:

    detection_rate = (
        total_detected_events /
        total_actual_events
    )

else:

    detection_rate = 0


print("\n")
print("=" * 70)
print("FINAL EVENT-LEVEL RESULTS")
print("=" * 70)

print(
    f"Threshold               : {THRESHOLD}"
)

print(
    f"Actual seizure events   : {total_actual_events}"
)

print(
    f"Detected seizure events : {total_detected_events}"
)

print(
    f"Predicted events        : {total_predicted_events}"
)

print(
    f"False alarm events      : {total_false_alarm_events}"
)

print(
    f"Event detection rate    : "
    f"{detection_rate * 100:.2f}%"
)


# ============================================================
# SAVE EVENT PREDICTIONS
# ============================================================

df.to_csv(
    "event_predictions.csv",
    index=False
)

print("\nSaved as event_predictions.csv")