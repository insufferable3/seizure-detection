import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

df = pd.read_csv("results/csv/cv_all_features_predictions.csv")

y_true = df["actual"]
y_score = df["seizure_probability"]

precision, recall, thresholds = precision_recall_curve(y_true, y_score)

pr_auc = average_precision_score(y_true, y_score)

plt.figure(figsize=(10, 7))
plt.plot(recall, precision, linewidth=2,
         label=f"PR-AUC = {pr_auc:.4f}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve — Seizure Detection")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("precision_recall_curve.png", dpi=300)
plt.show()

print("=" * 60)
print("PRECISION-RECALL ANALYSIS")
print("=" * 60)
print(f"PR-AUC / Average Precision : {pr_auc:.4f}")
