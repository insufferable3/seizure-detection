import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

# ============================================================
# LOAD FINAL PREDICTIONS
# ============================================================

df = pd.read_csv("results/csv/cv_all_features_predictions.csv")

y_true = df["actual"]
y_pred = df["predicted"]
y_prob = df["seizure_probability"]


# ============================================================
# 1. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks([0, 1], ["Non-Seizure", "Seizure"])
plt.yticks([0, 1], ["Non-Seizure", "Seizure"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()


# ============================================================
# 2. ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300)
plt.show()


# ============================================================
# 3. PRECISION-RECALL CURVE
# ============================================================

precision, recall, _ = precision_recall_curve(
    y_true,
    y_prob
)

pr_auc = average_precision_score(
    y_true,
    y_prob
)

plt.figure(figsize=(7, 6))
plt.plot(
    recall,
    precision,
    label=f"PR-AUC = {pr_auc:.4f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()

plt.tight_layout()
plt.savefig("precision_recall_curve.png", dpi=300)
plt.show()


# ============================================================
# 4. SEIZURE TIMELINE
# ============================================================

seizure_files = [
    "chb03_01.edf",
    "chb03_34.edf",
    "chb03_35.edf",
    "chb03_36.edf"
]

for filename in seizure_files:

    temp = df[df["file"] == filename].copy()

    plt.figure(figsize=(12, 4))

    plt.plot(
        temp["start"],
        temp["seizure_probability"],
        label="Seizure Probability"
    )

    plt.axhline(
        0.50,
        linestyle="--",
        label="Threshold = 0.50"
    )

    actual = temp[temp["actual"] == 1]

    if not actual.empty:
        plt.axvspan(
            actual["start"].min(),
            actual["end"].max(),
            alpha=0.25,
            label="Actual Seizure"
        )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Seizure Probability")
    plt.title(f"Seizure Detection Timeline — {filename}")
    plt.legend()

    plt.tight_layout()

    output_name = filename.replace(".edf", "_timeline.png")

    plt.savefig(
        output_name,
        dpi=300
    )

    plt.show()


print("\n==============================================")
print("FINAL PLOTS GENERATED")
print("==============================================")
print("confusion_matrix.png")
print("roc_curve.png")
print("precision_recall_curve.png")
print("chb03_01_timeline.png")
print("chb03_34_timeline.png")
print("chb03_35_timeline.png")
print("chb03_36_timeline.png")
