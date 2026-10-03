import pandas as pd

df = pd.read_csv("results/csv/cv_all_features_predictions.csv")

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50]
MIN_DURATIONS = [5, 10, 15, 20, 25, 30]
MERGE_GAPS = [5, 10, 20, 30, 40, 60]


def make_events(group, threshold, min_duration, merge_gap):

    group = group.sort_values("start")

    positive = group[group["seizure_probability"] >= threshold]

    if positive.empty:
        return []

    events = []

    start = positive.iloc[0]["start"]
    end = positive.iloc[0]["end"]

    for _, row in positive.iloc[1:].iterrows():

        if row["start"] - end <= merge_gap:
            end = row["end"]

        else:
            if end - start >= min_duration:
                events.append((start, end))

            start = row["start"]
            end = row["end"]

    if end - start >= min_duration:
        events.append((start, end))

    return events


def get_actual_events(group):

    actual = group[group["actual"] == 1]

    if actual.empty:
        return []

    events = []

    start = actual.iloc[0]["start"]
    end = actual.iloc[0]["end"]

    for _, row in actual.iloc[1:].iterrows():

        if row["start"] - end <= 5:
            end = row["end"]

        else:
            events.append((start, end))
            start = row["start"]
            end = row["end"]

    events.append((start, end))

    return events


def overlap(pred, actual):

    p_start, p_end = pred
    a_start, a_end = actual

    return max(0, min(p_end, a_end) - max(p_start, a_start))


results = []


for threshold in THRESHOLDS:

    for min_duration in MIN_DURATIONS:

        for merge_gap in MERGE_GAPS:

            actual_total = 0
            detected_total = 0
            predicted_total = 0

            for filename, group in df.groupby("file"):

                actual_events = get_actual_events(group)

                predicted_events = make_events(
                    group,
                    threshold,
                    min_duration,
                    merge_gap
                )

                actual_total += len(actual_events)
                predicted_total += len(predicted_events)

                for actual in actual_events:

                    detected = False

                    for pred in predicted_events:

                        if overlap(pred, actual) > 0:
                            detected = True
                            break

                    if detected:
                        detected_total += 1

            false_alarms = predicted_total - detected_total

            detection_rate = (
                detected_total / actual_total * 100
                if actual_total > 0 else 0
            )

            results.append({
                "threshold": threshold,
                "min_duration": min_duration,
                "merge_gap": merge_gap,
                "actual_events": actual_total,
                "detected_events": detected_total,
                "predicted_events": predicted_total,
                "false_alarm_events": false_alarms,
                "detection_rate": detection_rate
            })


results_df = pd.DataFrame(results)

# First: only configurations detecting ALL seizures
perfect = results_df[
    results_df["detection_rate"] == 100
].copy()

# Among 100% detection configs, minimize false alarms
perfect = perfect.sort_values(
    ["false_alarm_events", "predicted_events"]
)

print("=" * 90)
print("THRESHOLD × DURATION × MERGE-GAP SEARCH")
print("=" * 90)

print(results_df.to_string(index=False))

print("\n")
print("=" * 90)
print("BEST 100% DETECTION CONFIGURATIONS")
print("=" * 90)

print(perfect.head(20).to_string(index=False))

best = perfect.iloc[0]

print("\n")
print("=" * 90)
print("BEST CONFIGURATION")
print("=" * 90)

print(best)

results_df.to_csv(
    "final_event_grid_results.csv",
    index=False
)

print("\nSaved as final_event_grid_results.csv")
