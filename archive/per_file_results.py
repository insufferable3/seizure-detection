import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("results/csv/cv_all_features_predictions.csv")

print("=" * 75)
print("PER-FILE SEIZURE DETECTION PERFORMANCE")
print("=" * 75)

results = []

for file, g in df.groupby("file"):

    y_true = g["actual"]
    y_pred = g["predicted"]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    print("\n" + "-" * 60)
    print(file)
    print(f"Actual seizures     : {y_true.sum()}")
    print(f"Predicted seizures  : {y_pred.sum()}")
    print(f"TP                  : {tp}")
    print(f"FP                  : {fp}")
    print(f"FN                  : {fn}")
    print(f"Precision           : {precision:.4f}")
    print(f"Recall              : {recall:.4f}")
    print(f"F1 Score            : {f1:.4f}")

    results.append({
        "file": file,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn
    })

result_df = pd.DataFrame(results)

result_df.to_csv("per_file_results.csv", index=False)

print("\n" + "=" * 75)
print("SAVED: per_file_results.csv")
print("=" * 75)
