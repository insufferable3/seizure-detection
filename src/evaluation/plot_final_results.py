import pandas as pd
import matplotlib.pyplot as plt


# ==============================
# LOAD RESULTS
# ==============================

timing = pd.read_csv("event_latency_results.csv")
grid = pd.read_csv("event_duration_results.csv")


# ==============================
# 1. DETECTION LATENCY
# ==============================

plt.figure(figsize=(8, 5))

plt.bar(
    timing["file"],
    timing["detection_latency"]
)

plt.xlabel("EEG File")
plt.ylabel("Detection Latency (seconds)")
plt.title("Seizure Detection Latency")

plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    "detection_latency.png",
    dpi=300
)

plt.show()


# ==============================
# 2. EVENT OVERLAP
# ==============================

plt.figure(figsize=(8, 5))

plt.bar(
    timing["file"],
    timing["overlap_percentage"]
)

plt.xlabel("EEG File")
plt.ylabel("Overlap with Actual Seizure (%)")
plt.title("Predicted vs Actual Seizure Overlap")

plt.xticks(rotation=30)

plt.ylim(0, 100)

plt.tight_layout()

plt.savefig(
    "event_overlap.png",
    dpi=300
)

plt.show()


# ==============================
# 3. ONSET ERROR
# ==============================

plt.figure(figsize=(8, 5))

plt.bar(
    timing["file"],
    timing["onset_error"]
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel("EEG File")
plt.ylabel("Onset Error (seconds)")
plt.title("Seizure Onset Detection Error")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    "onset_error.png",
    dpi=300
)

plt.show()


# ==============================
# 4. THRESHOLD VS FALSE ALARMS
# ==============================

summary = (
    grid.groupby("threshold")
    .agg({
        "false_alarm_events": "min",
        "detection_rate": "max"
    })
    .reset_index()
)

plt.figure(figsize=(8, 5))

plt.plot(
    summary["threshold"],
    summary["false_alarm_events"],
    marker="o"
)

plt.xlabel("Probability Threshold")
plt.ylabel("False Alarm Events")
plt.title("Threshold vs False Alarms")

plt.tight_layout()

plt.savefig(
    "threshold_vs_false_alarms.png",
    dpi=300
)

plt.show()


print()
print("=" * 60)
print("GRAPHS GENERATED")
print("=" * 60)

print("detection_latency.png")
print("event_overlap.png")
print("onset_error.png")
print("threshold_vs_false_alarms.png")