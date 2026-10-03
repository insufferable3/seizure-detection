import pandas as pd

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

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("features.csv")

selected = pd.read_csv("selected_features.csv")
selected_features = selected["feature"].tolist()

print("GA selected features:", len(selected_features))
print("Total dataset:", df.shape)

# ==========================================
# 2. FILE-LEVEL TRAIN / TEST SPLIT
# ==========================================
#
# TRAIN = complete recordings
# TEST  = completely different recordings
#
# Seizure files:
# chb03_01
# chb03_34
# chb03_35
# chb03_36
#
# Normal files:
# chb03_15
# chb03_16
# chb03_17
#

train_files = [
    "chb03_01.edf",
    "chb03_15.edf",
    "chb03_16.edf",
    "chb03_34.edf"
]

test_files = [
    "chb03_17.edf",
    "chb03_35.edf",
    "chb03_36.edf"
]

train_df = df[df["file"].isin(train_files)].copy()
test_df = df[df["file"].isin(test_files)].copy()

print("\nTraining files:")
print(train_files)

print("\nTesting files:")
print(test_files)

# ==========================================
# 3. CREATE X AND y
# ==========================================

X_train = train_df[selected_features]
y_train = train_df["label"]

X_test = test_df[selected_features]
y_test = test_df["label"]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

# ==========================================
# 4. TRAIN RANDOM FOREST
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training complete!")

# ==========================================
# 5. PREDICTION
# ==========================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# ==========================================
# 6. RESULTS
# ==========================================

print("\n==============================")
print("FILE-LEVEL TEST RESULTS")
print("==============================")

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))

# ==========================================
# 7. CONFUSION MATRIX
# ==========================================

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# 8. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Non-Seizure",
            "Seizure"
        ],
        zero_division=0
    )
)

# ==========================================
# 9. RESULTS PER FILE
# ==========================================

print("\n==============================")
print("RESULTS PER FILE")
print("==============================")

test_df["prediction"] = y_pred

for file in test_files:

    file_data = test_df[test_df["file"] == file]

    actual = file_data["label"].sum()
    predicted = file_data["prediction"].sum()

    print(f"\n{file}")
    print(f"Actual seizure windows    : {actual}")
    print(f"Predicted seizure windows : {predicted}")