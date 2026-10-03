import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# 1. LOAD CROSS-VALIDATION PREDICTIONS
# ============================================================

df = pd.read_csv("cv_predictions.csv")

y_true = df["actual"].values
prob = df["seizure_probability"].values

print("Total samples:", len(df))
print("Actual seizures:", np.sum(y_true == 1))
print("Actual non-seizures:", np.sum(y_true == 0))


# ============================================================
# 2. TEST DIFFERENT THRESHOLDS
# ============================================================

thresholds = [
    0.50,
    0.45,
    0.40,
    0.35,
    0.30,
    0.25,
    0.20,
    0.15,
    0.10,
    0.05
]

results = []

print("\n")
print("=" * 75)
print("THRESHOLD ANALYSIS")
print("=" * 75)

print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'FP':<8}"
    f"{'FN':<8}"
)

for threshold in thresholds:

    # Convert probability → class
    y_pred = (prob >= threshold).astype(int)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    # Make sure confusion matrix is 2x2
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positives": fp,
        "false_negatives": fn
    })

    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
        f"{fp:<8}"
        f"{fn:<8}"
    )


# ============================================================
# 3. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "threshold_results.csv",
    index=False
)

print("\nSaved as threshold_results.csv")


# ============================================================
# 4. BEST THRESHOLD BY F1
# ============================================================

best_f1_row = results_df.loc[
    results_df["f1"].idxmax()
]

print("\n")
print("=" * 50)
print("BEST THRESHOLD BY F1")
print("=" * 50)

print(
    "Threshold:",
    best_f1_row["threshold"]
)

print(
    "Precision:",
    round(best_f1_row["precision"], 4)
)

print(
    "Recall:",
    round(best_f1_row["recall"], 4)
)

print(
    "F1:",
    round(best_f1_row["f1"], 4)
)


# ============================================================
# 5. BEST THRESHOLD BY RECALL
# ============================================================

best_recall_row = results_df.loc[
    results_df["recall"].idxmax()
]

print("\n")
print("=" * 50)
print("BEST THRESHOLD BY RECALL")
print("=" * 50)

print(
    "Threshold:",
    best_recall_row["threshold"]
)

print(
    "Precision:",
    round(best_recall_row["precision"], 4)
)

print(
    "Recall:",
    round(best_recall_row["recall"], 4)
)

print(
    "F1:",
    round(best_recall_row["f1"], 4)
)