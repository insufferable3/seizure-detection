import pandas as pd

INPUT_FILE = "results/csv/cv_all_features_predictions.csv"

THRESHOLD = 0.50

MIN_DURATIONS = [5, 10, 15, 20, 25, 30]
MERGE_GAPS = [5, 10, 20, 30]


def make_events(df, threshold, merge_gap, min_duration):

    events = []

    for file_name, g in df.groupby("file"):

        g = g.sort_values("start").reset_index(drop=True)

        active = g["seizure_probability"] >= threshold

        current = None

        for i in range(len(g)):

            if active.iloc[i]:

                start = float(g.loc[i, "start"])
                end = float(g.loc[i, "end"])

                if current is None:
                    current = [start, end]

                else:
                    gap = start - current[1]

                    if gap <= merge_gap:
                        current[1] = end
                    else:
                        duration = current[1] - current[0]

                        if duration >= min_duration:
                            events.append(
                                (file_name, current[0], current[1])
                            )

                        current = [start, end]

        if current is not None:

            duration = current[1] - current[0]

            if duration >= min_duration:
                events.append(
                    (file_name, current[0], current[1])
                )

    return events


def get_actual_events(df):

    events = []

    for file_name, g in df.groupby("file"):

        g = g.sort_values("start").reset_index(drop=True)

        active = g["actual"] == 1

        current = None

        for i in range(len(g)):

            if active.iloc[i]:

                start = float(g.loc[i, "start"])
                end = float(g.loc[i, "end"])

                if current is None:
                    current = [start, end]

                else:
                    current[1] = end

            else:

                if current is not None:
                    events.append(
                        (file_name, current[0], current[1])
                    )
                    current = None

        if current is not None:
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

actual_events = get_actual_events(df)

results = []

print()
print("=" * 90)
print("EVENT DURATION SEARCH")
print("=" * 90)

for min_duration in MIN_DURATIONS:

    for merge_gap in MERGE_GAPS:

        predicted_events = make_events(
            df,
            THRESHOLD,
            merge_gap,
            min_duration
        )

        detected = 0

        for actual in actual_events:

            found = any(
                overlaps(pred, actual)
                for pred in predicted_events
            )

            if found:
                detected += 1

        false_alarms = 0

        for pred in predicted_events:

            if not any(
                overlaps(pred, actual)
                for actual in actual_events
            ):
                false_alarms += 1

        results.append({
            "threshold": THRESHOLD,
            "min_duration": min_duration,
            "merge_gap": merge_gap,
            "actual_events": len(actual_events),
            "detected_events": detected,
            "predicted_events": len(predicted_events),
            "false_alarm_events": false_alarms,
            "detection_rate":
                100 * detected / len(actual_events)
        })


results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))

print()
print("=" * 90)
print("CONFIGURATIONS WITH 100% DETECTION")
print("=" * 90)

good = results_df[
    results_df["detection_rate"] == 100
].sort_values(
    ["false_alarm_events", "predicted_events"]
)

print(good.to_string(index=False))

if len(good) > 0:

    best = good.iloc[0]

    print()
    print("=" * 90)
    print("BEST CONFIGURATION")
    print("=" * 90)

    print(best.to_string())

results_df.to_csv(
    "event_duration_results.csv",
    index=False
)

print()
print("Saved as event_duration_results.csv")
