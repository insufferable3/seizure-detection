import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# FINAL MODEL COMPARISON
# ==============================

models = ["Baseline ML", "GA + ANFIS"]

window_f1 = [0.5429, 0.7059]
event_detection = [100.0, 100.0]
false_alarms = [0, 1]
latency = [12.50, -1.25]
overlap = [69.25, 47.98]

df = pd.DataFrame({
    "Model": models,
    "Window F1": window_f1,
    "Event Detection (%)": event_detection,
    "False Alarm Events": false_alarms,
    "Mean Detection Latency (s)": latency,
    "Mean Event Overlap (%)": overlap
})

print("=" * 75)
print("FINAL SEIZURE DETECTION MODEL COMPARISON")
print("=" * 75)
print(df.to_string(index=False))

df.to_csv("final_model_comparison.csv", index=False)

# ==============================
# WINDOW F1
# ==============================

plt.figure(figsize=(8, 5))
plt.bar(models, window_f1)
plt.ylabel("F1 Score")
plt.title("Window-Level F1 Comparison")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("final_window_f1_comparison.png", dpi=300)
plt.close()

# ==============================
# EVENT DETECTION
# ==============================

plt.figure(figsize=(8, 5))
plt.bar(models, event_detection)
plt.ylabel("Detection Rate (%)")
plt.title("Event-Level Seizure Detection")
plt.ylim(0, 110)
plt.tight_layout()
plt.savefig("final_event_detection_comparison.png", dpi=300)
plt.close()

# ==============================
# EVENT OVERLAP
# ==============================

plt.figure(figsize=(8, 5))
plt.bar(models, overlap)
plt.ylabel("Mean Event Overlap (%)")
plt.title("Mean Event Overlap Comparison")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("final_event_overlap_comparison.png", dpi=300)
plt.close()

print("\nSaved:")
print("  final_model_comparison.csv")
print("  final_window_f1_comparison.png")
print("  final_event_detection_comparison.png")
print("  final_event_overlap_comparison.png")