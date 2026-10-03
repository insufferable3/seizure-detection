import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# LOAD ALL-FEATURE CV PREDICTIONS
# ============================================================

df = pd.read_csv("results/csv/cv_all_features_predictions.csv")

y_true = df["actual"].values
prob = df["seizure_probability"].values

print("Total samples:", len(df))
print("Actual seizures:", np.sum(y_true == 1))
print("Actual non-seizures:", np.sum(y_true == 0))

# ============================================================
# THRESHOLDS
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
print("ALL FEATURES — THRESHOLD ANALYSIS")
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

    predictions = (
        prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
        f"{fp:<8}"
        f"{fn:<8}"
    )

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp": fp,
        "fn": fn
    })


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "threshold_all_features_results.csv",
    index=False
)

# ============================================================
# BEST F1
# ============================================================

best_f1 = results_df.loc[
    results_df["f1"].idxmax()
]

print("\n")
print("=" * 50)
print("BEST THRESHOLD BY F1")
print("=" * 50)

print(
    "Threshold:",
    best_f1["threshold"]
)

print(
    "Precision:",
    best_f1["precision"]
)

print(
    "Recall:",
    best_f1["recall"]
)

print(
    "F1:",
    best_f1["f1"]
)

# ============================================================
# BEST RECALL
# ============================================================

best_recall = results_df.loc[
    results_df["recall"].idxmax()
]

print("\n")
print("=" * 50)
print("BEST THRESHOLD BY RECALL")
print("=" * 50)

print(
    "Threshold:",
    best_recall["threshold"]
)

print(
    "Precision:",
    best_recall["precision"]
)

print(
    "Recall:",
    best_recall["recall"]
)

print(
    "F1:",
    best_recall["f1"]
)

print("\nSaved as threshold_all_features_results.csv")
