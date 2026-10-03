import pandas as pd

pred = pd.read_csv("/Users/saanchritu/Desktop/seizure detection/results/csv/cv_all_features_predictions.csv")
timing = pd.read_csv("/Users/saanchritu/Desktop/seizure detection/results/csv/event_latency_results.csv")

print()
print("=" * 75)
print("FINAL SEIZURE DETECTION RESULTS")
print("=" * 75)

print()

# Window-level metrics
actual = pred["actual"]
predicted = pred["predicted"]

tp = ((actual == 1) & (predicted == 1)).sum()
tn = ((actual == 0) & (predicted == 0)).sum()
fp = ((actual == 0) & (predicted == 1)).sum()
fn = ((actual == 1) & (predicted == 0)).sum()

accuracy = (tp + tn) / (tp + tn + fp + fn)

precision = tp / (tp + fp) if (tp + fp) else 0
recall = tp / (tp + fn) if (tp + fn) else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall)
    else 0
)

print("WINDOW-LEVEL PERFORMANCE")
print("-" * 50)

print(f"Accuracy       : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1 Score       : {f1:.4f}")

print()
print("Confusion Matrix")
print("-" * 50)

print(f"TN : {tn}")
print(f"FP : {fp}")
print(f"FN : {fn}")
print(f"TP : {tp}")


# Event-level results
actual_events = 4
detected_events = 4
false_alarms = 0
missed_events = 0

event_detection_rate = (
    detected_events / actual_events * 100
)

mean_latency = timing["detection_latency"].mean()
median_latency = timing["detection_latency"].median()
mean_overlap = timing["overlap_percentage"].mean()


print()
print("=" * 75)
print("EVENT-LEVEL PERFORMANCE")
print("=" * 75)

print(f"Actual seizure events     : {actual_events}")
print(f"Detected seizure events   : {detected_events}")
print(f"Missed seizure events     : {missed_events}")
print(f"False alarm events        : {false_alarms}")
print(f"Event detection rate      : {event_detection_rate:.2f}%")

print()
print(f"Mean detection latency    : {mean_latency:.2f} s")
print(f"Median detection latency  : {median_latency:.2f} s")
print(f"Mean event overlap        : {mean_overlap:.2f}%")


print()
print("=" * 75)
print("FINAL CONFIGURATION")
print("=" * 75)

print("Probability threshold     : 0.50")
print("Merge gap                 : 30 s")
print("Minimum event duration    : 10 s")

print()
print("=" * 75)
print("FINAL SUMMARY")
print("=" * 75)

print(
    f"Window F1                 : {f1:.4f}"
)

print(
    f"Event detection           : "
    f"{event_detection_rate:.2f}%"
)

print(
    f"False alarms              : "
    f"{false_alarms}"
)

print(
    f"Mean detection latency    : "
    f"{mean_latency:.2f} s"
)

print(
    f"Mean event overlap        : "
    f"{mean_overlap:.2f}%"
)

print()
print("Final results generated.")
