import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

df = pd.read_csv("results/csv/cv_all_features_predictions.csv")

y_true = df["actual"]
y_prob = df["seizure_probability"]


# ==============================
# ROC CURVE
# ==============================

fpr, tpr, thresholds = roc_curve(y_true, y_prob)

roc_auc = roc_auc_score(y_true, y_prob)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Seizure Detection")

plt.legend()
plt.tight_layout()

plt.savefig(
    "roc_curve.png",
    dpi=300
)

plt.show()


# ==============================
# PRECISION-RECALL CURVE
# ==============================

precision, recall, thresholds_pr = precision_recall_curve(
    y_true,
    y_prob
)

ap = average_precision_score(
    y_true,
    y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    recall,
    precision,
    label=f"Average Precision = {ap:.4f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve — Seizure Detection")

plt.legend()
plt.tight_layout()

plt.savefig(
    "precision_recall_curve.png",
    dpi=300
)

plt.show()


print()
print("=" * 60)
print("FINAL CURVE METRICS")
print("=" * 60)

print(f"ROC-AUC              : {roc_auc:.4f}")
print(f"Average Precision    : {ap:.4f}")

print()
print("Saved:")
print("roc_curve.png")
print("precision_recall_curve.png")
