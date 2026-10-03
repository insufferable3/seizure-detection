import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("features.csv")

files = [
    "chb03_01.edf",
    "chb03_15.edf",
    "chb03_16.edf",
    "chb03_17.edf",
    "chb03_34.edf",
    "chb03_35.edf",
    "chb03_36.edf"
]

feature_cols = [
    col for col in df.columns
    if col not in ["file", "start", "end", "label"]
]

print("Total dataset:", df.shape)
print("Total features:", len(feature_cols))

# ============================================================
# 2. LEAVE-ONE-FILE-OUT CV
# ============================================================

all_actual = []
all_predicted = []
all_probability = []

detailed_results = []

for test_file in files:

    print("\n" + "=" * 60)
    print("TEST FILE:", test_file)
    print("=" * 60)

    train_df = df[df["file"] != test_file].copy()
    test_df = df[df["file"] == test_file].copy()

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    print("Training samples:", len(train_df))
    print("Testing samples :", len(test_df))

    print("\nTraining distribution:")
    print(train_df["label"].value_counts())

    print("\nTesting distribution:")
    print(test_df["label"].value_counts())

    # ========================================================
    # 3. RANDOM FOREST
    # ========================================================

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
        max_features="sqrt"
    )

    print("\nTraining model...")

    model.fit(X_train, y_train)

    print("Training complete!")

    # ========================================================
    # 4. PREDICTION
    # ========================================================

    probabilities = model.predict_proba(X_test)[:, 1]

    # Standard threshold
    predictions = (probabilities >= 0.5).astype(int)

    # ========================================================
    # 5. STORE RESULTS
    # ========================================================

    all_actual.extend(y_test)
    all_predicted.extend(predictions)
    all_probability.extend(probabilities)

    for i in range(len(test_df)):

        detailed_results.append({
            "file": test_df.iloc[i]["file"],
            "start": test_df.iloc[i]["start"],
            "end": test_df.iloc[i]["end"],
            "actual": y_test[i],
            "predicted": predictions[i],
            "seizure_probability": probabilities[i]
        })

    actual_seizures = np.sum(y_test == 1)
    predicted_seizures = np.sum(predictions == 1)

    print("\nActual seizure windows   :", actual_seizures)
    print("Predicted seizure windows:", predicted_seizures)


# ============================================================
# 6. GLOBAL RESULTS
# ============================================================

all_actual = np.array(all_actual)
all_predicted = np.array(all_predicted)
all_probability = np.array(all_probability)

accuracy = accuracy_score(
    all_actual,
    all_predicted
)

precision = precision_score(
    all_actual,
    all_predicted,
    zero_division=0
)

recall = recall_score(
    all_actual,
    all_predicted,
    zero_division=0
)

f1 = f1_score(
    all_actual,
    all_predicted,
    zero_division=0
)

roc_auc = roc_auc_score(
    all_actual,
    all_probability
)

cm = confusion_matrix(
    all_actual,
    all_predicted
)

print("\n")
print("=" * 60)
print("ALL FEATURES — LEAVE-ONE-FILE-OUT RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        all_actual,
        all_predicted,
        target_names=["Non-Seizure", "Seizure"],
        zero_division=0
    )
)

# ============================================================
# 7. SAVE DETAILED PREDICTIONS
# ============================================================

results_df = pd.DataFrame(detailed_results)

results_df.to_csv(
    "results/csv/cv_all_features_predictions.csv",
    index=False
)

print(
    "\nSaved as results/csv/cv_all_features_predictions.csv"
)

# ============================================================
# 8. PER-FILE RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("RESULTS PER FILE")
print("=" * 60)

for file_name in files:

    file_df = results_df[
        results_df["file"] == file_name
    ]

    actual = file_df["actual"].sum()
    predicted = file_df["predicted"].sum()

    print("\n" + file_name)
    print("Actual seizure windows    :", actual)
    print("Predicted seizure windows :", predicted)
