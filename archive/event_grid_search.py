import pandas as pd


INPUT_FILE = "results/csv/cv_all_features_predictions.csv"

THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

MERGE_GAPS = [5, 10, 20, 30, 40, 60]


df = pd.read_csv(INPUT_FILE)


def build_events(file_df, column, merge_gap):

    rows = file_df.sort_values("start").reset_index(drop=True)

    events = []

    start = None
    end = None

    for _, row in rows.iterrows():

        if row[column] == 1:

            if start is None:
                start = row["start"]
                end = row["end"]

            elif row["start"] - end <= merge_gap:
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


def overlaps(actual, predicted):

    a_start, a_end = actual
    p_start, p_end = predicted

    return max(a_start, p_start) < min(a_end, p_end)


def evaluate(threshold, merge_gap):

    temp = df.copy()

    temp["predicted"] = (
        temp["seizure_probability"] >= threshold
    ).astype(int)

    actual_total = 0
    detected_total = 0
    false_total = 0
    predicted_total = 0

    for _, file_df in temp.groupby("file"):

        actual_events = build_events(
            file_df,
            "actual",
            merge_gap
        )

        predicted_events = build_events(
            file_df,
            "predicted",
            merge_gap
        )

        actual_total += len(actual_events)
        predicted_total += len(predicted_events)

        matched_actual = set()
        matched_predicted = set()

        for p_idx, predicted in enumerate(predicted_events):

            for a_idx, actual in enumerate(actual_events):

                if a_idx in matched_actual:
                    continue

                if overlaps(actual, predicted):

                    matched_actual.add(a_idx)
                    matched_predicted.add(p_idx)

                    break

        detected_total += len(matched_actual)

        false_total += (
            len(predicted_events)
            - len(matched_predicted)
        )

    if actual_total > 0:
        detection_rate = (
            detected_total / actual_total
        ) * 100
    else:
        detection_rate = 0

    return (
        actual_total,
        detected_total,
        predicted_total,
        false_total,
        detection_rate
    )


# ============================================================
# GRID SEARCH
# ============================================================

results = []

print()
print("=" * 90)
print("EVENT-LEVEL THRESHOLD × MERGE-GAP SEARCH")
print("=" * 90)

for threshold in THRESHOLDS:

    for merge_gap in MERGE_GAPS:

        (
            actual,
            detected,
            predicted,
            false,
            detection_rate
        ) = evaluate(
            threshold,
            merge_gap
        )

        results.append({
            "threshold": threshold,
            "merge_gap": merge_gap,
            "actual_events": actual,
            "detected_events": detected,
            "predicted_events": predicted,
            "false_alarm_events": false,
            "detection_rate": detection_rate
        })


results_df = pd.DataFrame(results)


# ============================================================
# PRINT ALL RESULTS
# ============================================================

print(
    results_df.to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    "event_grid_results.csv",
    index=False
)


# ============================================================
# BEST CONFIGS
# ============================================================

# First: configurations detecting ALL seizures
all_detected = results_df[
    results_df["detected_events"]
    == results_df["actual_events"]
].copy()


print()
print("=" * 90)
print("CONFIGURATIONS WITH 100% EVENT DETECTION")
print("=" * 90)

if len(all_detected) > 0:

    best = all_detected.sort_values(
        [
            "false_alarm_events",
            "predicted_events",
            "threshold"
        ],
        ascending=[
            True,
            True,
            False
        ]
    )

    print(
        best.to_string(index=False)
    )

    print()
    print("=" * 90)
    print("BEST CONFIGURATION")
    print("=" * 90)

    print(
        best.iloc[0].to_string()
    )

else:

    print("No configuration detected all seizures.")


print()
print("Saved as event_grid_results.csv")
