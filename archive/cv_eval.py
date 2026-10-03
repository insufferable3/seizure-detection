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
# LOAD DATA
# ============================================================

df = pd.read_csv("features.csv")

print("Total dataset:", df.shape)

# ============================================================
# GA FEATURES
# ============================================================

selected_features = pd.read_csv(
    "selected_features.csv"
)["feature"].tolist()

print("GA selected features:", len(selected_features))

# ============================================================
# FILES
# ============================================================

files = sorted(df["file"].unique())

print("\nFiles:")
for f in files:
    print(f)

# ============================================================
# STORAGE
# ============================================================

all_predictions = []

# ============================================================
# LEAVE-ONE-FILE-OUT CV
# ============================================================

for test_file in files:

    print("\n" + "=" * 60)
    print("TEST FILE:", test_file)
    print("=" * 60)

    train_df = df[df["file"] != test_file].copy()
    test_df = df[df["file"] == test_file].copy()

    X_train = train_df[selected_features]
    y_train = train_df["label"]

    X_test = test_df[selected_features]
    y_test = test_df["label"]

    print("Training samples:", len(train_df))
    print("Testing samples :", len(test_df))

    print("\nTraining distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting distribution:")
    print(y_test.value_counts().sort_index())

    # ========================================================
    # MODEL
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
    # PROBABILITY
    # ========================================================

    probability = model.predict_proba(X_test)[:, 1]

    # Default threshold
    prediction = (probability >= 0.50).astype(int)

    # ========================================================
    # SAVE TEST FOLD WITH METADATA
    # ========================================================

    fold_result = pd.DataFrame({
        "file": test_df["file"].values,
        "start": test_df["start"].values,
        "end": test_df["end"].values,
        "actual": y_test.values,
        "predicted": prediction,
        "seizure_probability": probability
    })

    all_predictions.append(fold_result)

    print(
        "\nActual seizure windows   :",
        np.sum(y_test.values == 1)
    )

    print(
        "Predicted seizure windows:",
        np.sum(prediction == 1)
    )


# ============================================================
# COMBINE ALL FOLDS
# ============================================================

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

# ============================================================
# SAVE DETAILED CSV
# ============================================================

predictions_df.to_csv(
    "cv_predictions_detailed.csv",
    index=False
)

# Also save old/simple version if needed
predictions_df[
    [
        "actual",
        "predicted",
        "seizure_probability"
    ]
].to_csv(
    "cv_predictions.csv",
    index=False
)

# ============================================================
# OVERALL METRICS
# ============================================================

y_true = predictions_df["actual"]
y_pred = predictions_df["predicted"]
y_prob = predictions_df["seizure_probability"]

accuracy = accuracy_score(y_true, y_pred)

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

roc_auc = roc_auc_score(
    y_true,
    y_prob
)

cm = confusion_matrix(
    y_true,
    y_pred
)

# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 50)
print("LEAVE-ONE-FILE-OUT CV RESULTS")
print("=" * 50)

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
        y_true,
        y_pred,
        target_names=[
            "Non-Seizure",
            "Seizure"
        ],
        zero_division=0
    )
)

# ============================================================
# RESULTS PER FILE
# ============================================================

print("\n")
print("=" * 50)
print("RESULTS PER FILE")
print("=" * 50)

for file_name, file_df in predictions_df.groupby("file"):

    actual = file_df["actual"].sum()

    predicted = file_df["predicted"].sum()

    print("\n" + file_name)

    print(
        "Actual seizure windows    :",
        actual
    )

    print(
        "Predicted seizure windows :",
        predicted
    )

# ============================================================
# FINAL CHECK
# ============================================================

print("\n")
print("=" * 50)
print("SAVED FILE")
print("=" * 50)

print("cv_predictions_detailed.csv")

print("\nColumns:")
print(predictions_df.columns.tolist())

print("\nFirst 10 rows:")
print(
    predictions_df
    .head(10)
    .to_string(index=False)
)