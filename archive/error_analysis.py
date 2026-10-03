import pandas as pd

df = pd.read_csv("results/csv/cv_all_features_predictions.csv")

print("=" * 80)
print("SEIZURE DETECTION — ERROR ANALYSIS")
print("=" * 80)

# False Negatives
fn = df[(df["actual"] == 1) & (df["predicted"] == 0)]

print("\n" + "=" * 80)
print("FALSE NEGATIVES — ACTUAL SEIZURE BUT MISSED")
print("=" * 80)

for file, g in fn.groupby("file"):
    print(f"\n{file}")
    for _, row in g.iterrows():
        print(
            f"  {row['start']:.1f}s - {row['end']:.1f}s"
            f" | probability = {row['seizure_probability']:.4f}"
        )

# False Positives
fp = df[(df["actual"] == 0) & (df["predicted"] == 1)]

print("\n" + "=" * 80)
print("FALSE POSITIVES — NON-SEIZURE BUT PREDICTED SEIZURE")
print("=" * 80)

for file, g in fp.groupby("file"):
    print(f"\n{file}")
    for _, row in g.iterrows():
        print(
            f"  {row['start']:.1f}s - {row['end']:.1f}s"
            f" | probability = {row['seizure_probability']:.4f}"
        )

print("\n" + "=" * 80)
print(f"Total False Negatives: {len(fn)}")
print(f"Total False Positives: {len(fp)}")
print("=" * 80)

fn.to_csv("false_negatives.csv", index=False)
fp.to_csv("false_positives.csv", index=False)

print("\nSaved:")
print("  false_negatives.csv")
print("  false_positives.csv")
