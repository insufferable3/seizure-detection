import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("results/csv/anfis_predictions.csv")

y_true = df["label"]
prob = df["anfis_probability"]

print("=" * 70)
print("ANFIS THRESHOLD SEARCH")
print("=" * 70)

results = []

for threshold in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:

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

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    results.append([
        threshold,
        precision,
        recall,
        f1,
        tp,
        fp,
        fn
    ])

    print(
        f"\nThreshold: {threshold:.2f}"
        f"\nPrecision : {precision:.4f}"
        f"\nRecall    : {recall:.4f}"
        f"\nF1        : {f1:.4f}"
        f"\nTP        : {tp}"
        f"\nFP        : {fp}"
        f"\nFN        : {fn}"
    )

result_df = pd.DataFrame(
    results,
    columns=[
        "threshold",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn"
    ]
)

result_df = result_df.sort_values(
    "f1",
    ascending=False
)

print("\n" + "=" * 70)
print("BEST ANFIS CONFIGURATION")
print("=" * 70)

print(result_df.iloc[0])

result_df.to_csv(
    "anfis_threshold_results.csv",
    index=False
)

print("\nSaved: anfis_threshold_results.csv")
