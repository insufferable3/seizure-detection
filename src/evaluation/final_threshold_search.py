import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("results/csv/cv_all_features_predictions.csv")

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

print("=" * 80)
print("FINAL THRESHOLD COMPARISON")
print("=" * 80)

for threshold in thresholds:

    pred = (df["seizure_probability"] >= threshold).astype(int)

    precision = precision_score(
        df["actual"], pred, zero_division=0
    )

    recall = recall_score(
        df["actual"], pred, zero_division=0
    )

    f1 = f1_score(
        df["actual"], pred, zero_division=0
    )

    tp = ((df["actual"] == 1) & (pred == 1)).sum()
    fp = ((df["actual"] == 0) & (pred == 1)).sum()
    fn = ((df["actual"] == 1) & (pred == 0)).sum()

    print(f"\nThreshold: {threshold:.2f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"TP        : {tp}")
    print(f"FP        : {fp}")
    print(f"FN        : {fn}")
