import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "results/csv/cv_all_features_predictions.csv"

THRESHOLD = 0.25

# Predicted events separated by <= this gap are merged
MERGE_GAP = 20.0


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["predicted"] = (
    df["seizure_probability"] >= THRESHOLD
).astype(int)

print("Total windows:", len(df))


# ============================================================
# BUILD EVENTS
# ============================================================

def build_events(file_df, column):

    rows = file_df.sort_values("start").reset_index(drop=True)

    events = []

    start = None
    end = None

    for _, row in rows.iterrows():

        if row[column] == 1:

            if start is None:
                start = row["start"]
                end = row["end"]

            elif row["start"] - end <= MERGE_GAP:
                end = row["end"]

            else:
                events.append((start, end))

                start = row["start"]
                end = row["end"]

        else:

            if start is not None:
                events.append((start, end))
                start = None
                end = None

    if start is not None:
        events.append((start, end))

    return events


# ============================================================
# EVENT OVERLAP
# ============================================================

def overlaps(actual, predicted):

    a_start, a_end = actual
    p_start, p_end = predicted

    return max(a_start, p_start) < min(a_end, p_end)


# ============================================================
# 1-TO-1 EVENT MATCHING
# ============================================================

def match_events(actual_events, predicted_events):

    matched_actual = set()
    matched_predicted = set()

    for p_idx, predicted in enumerate(predicted_events):

        for a_idx, actual in enumerate(actual_events):

            if a_idx in matched_actual:
                continue

            if overlaps(actual, predicted):

                matched_actual.add(a_idx)
                matched_predicted.add(p_idx)

                # IMPORTANT:
                # one actual event can only be detected once
                break

    detected = len(matched_actual)

    false_alarms = len(predicted_events) - len(matched_predicted)

    missed = len(actual_events) - detected

    return detected, false_alarms, missed


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 70)
print("ALL FEATURES — CORRECTED EVENT-LEVEL ANALYSIS")
print("=" * 70)


total_actual = 0
total_detected = 0
total_predicted = 0
total_false = 0
total_missed = 0


for file_name, file_df in df.groupby("file"):

    actual_events = build_events(file_df, "actual")
    predicted_events = build_events(file_df, "predicted")

    detected, false_alarms, missed = match_events(
        actual_events,
        predicted_events
    )

    total_actual += len(actual_events)
    total_detected += detected
    total_predicted += len(predicted_events)
    total_false += false_alarms
    total_missed += missed

    print()
    print("-" * 60)
    print(file_name)

    print("Actual events     :", len(actual_events))
    print("Predicted events  :", len(predicted_events))
    print("Detected events   :", detected)
    print("False alarms      :", false_alarms)
    print("Missed events     :", missed)

    if actual_events:
        print("\nActual:")

        for start, end in actual_events:
            print(f"  {start:.1f}s - {end:.1f}s")

    if predicted_events:
        print("\nPredicted:")

        for start, end in predicted_events:
            print(f"  {start:.1f}s - {end:.1f}s")


# ============================================================
# FINAL METRICS
# ============================================================

if total_actual > 0:
    detection_rate = (
        total_detected / total_actual
    ) * 100
else:
    detection_rate = 0


print()
print("=" * 70)
print("FINAL CORRECTED EVENT RESULTS")
print("=" * 70)

print(f"Threshold              : {THRESHOLD}")
print(f"Merge gap              : {MERGE_GAP}s")
print(f"Actual seizure events  : {total_actual}")
print(f"Detected events        : {total_detected}")
print(f"Predicted events       : {total_predicted}")
print(f"False alarm events     : {total_false}")
print(f"Missed seizure events  : {total_missed}")
print(f"Event detection rate   : {detection_rate:.2f}%")
