# anfis.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# CONFIG
# ============================================================

FEATURE_FILE = "/Users/saanchritu/Desktop/seizure detection/data/processed/features.csv"
SELECTED_FILE = "/Users/saanchritu/Desktop/seizure detection/data/processed/selected_features.csv"

TEST_SIZE = 0.20
RANDOM_STATE = 42

EPOCHS = 100
LEARNING_RATE = 0.001

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("GA + ANFIS SEIZURE DETECTION")
print("=" * 70)

df = pd.read_csv(FEATURE_FILE)

selected_df = pd.read_csv(SELECTED_FILE)

selected_features = selected_df["feature"].tolist()

print(f"Total samples        : {len(df)}")
print(f"GA selected features : {len(selected_features)}")

# ============================================================
# CHECK FEATURES
# ============================================================

usable_features = [
    f for f in selected_features
    if f in df.columns
]

missing_features = [
    f for f in selected_features
    if f not in df.columns
]

print(f"Usable features      : {len(usable_features)}")

if missing_features:
    print(f"Missing features     : {len(missing_features)}")

if len(usable_features) == 0:
    raise ValueError(
        "No selected features found in features.csv"
    )

# ============================================================
# PREPARE X AND Y
# ============================================================

X = df[usable_features].copy()
y = df["label"].copy()

# Remove NaN / infinite values
X = X.replace([np.inf, -np.inf], np.nan)

valid = X.notna().all(axis=1) & y.notna()

X = X.loc[valid]
y = y.loc[valid]

print(f"Final samples        : {len(X)}")

print("\nClass distribution:")
print(y.value_counts())

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining samples     :", len(X_train))
print("Testing samples     :", len(X_test))

print("\nTraining distribution:")
print(y_train.value_counts())

print("\nTesting distribution:")
print(y_test.value_counts())

# ============================================================
# SCALE
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# TORCH TENSORS
# ============================================================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train.values,
    dtype=torch.float32
).view(-1, 1)

y_test_tensor = torch.tensor(
    y_test.values,
    dtype=torch.float32
).view(-1, 1)

print("\nInput dimensions:")
print("Training X:", X_train.shape)
print("Testing X :", X_test.shape)

# ============================================================
# SIMPLE ANFIS-STYLE NEURO-FUZZY MODEL
# ============================================================

class ANFIS(nn.Module):

    def __init__(self, n_features):
        super().__init__()

        self.membership_center = nn.Parameter(
            torch.zeros(n_features)
        )

        self.membership_sigma = nn.Parameter(
            torch.ones(n_features)
        )

        self.linear = nn.Linear(
            n_features,
            1
        )

    def forward(self, x):

        sigma = torch.abs(
            self.membership_sigma
        ) + 1e-6

        # Gaussian membership
        membership = torch.exp(
            -0.5 *
            ((x - self.membership_center) / sigma) ** 2
        )

        # Fuzzy aggregation
        fuzzy_output = membership.mean(
            dim=1,
            keepdim=True
        )

        # Linear rule layer
        output = self.linear(
            membership
        )

        # Combine fuzzy and rule output
        output = output + fuzzy_output

        return output


# ============================================================
# MODEL
# ============================================================

model = ANFIS(
    len(usable_features)
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        [(len(y_train) - y_train.sum()) /
         y_train.sum()]
    )
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("TRAINING ANFIS")
print("=" * 70)

model.train()

for epoch in range(EPOCHS):

    optimizer.zero_grad()

    logits = model(
        X_train
    )

    loss = criterion(
        logits,
        y_train
    )

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch [{epoch + 1:3d}/{EPOCHS}] "
            f"Loss: {loss.item():.6f}"
        )

# ============================================================
# PREDICTION
# ============================================================

model.eval()

with torch.no_grad():

    test_logits = model(
        X_test
    )

    probabilities = torch.sigmoid(
        test_logits
    ).numpy().ravel()

# ============================================================
# THRESHOLD
# ============================================================

THRESHOLD = 0.45

y_pred = (
    probabilities >= THRESHOLD
).astype(int)

y_true = y_test.values

# ============================================================
# RESULTS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

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

print("\n" + "=" * 70)
print("ANFIS RESULTS")
print("=" * 70)

print(f"Accuracy       : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1 Score       : {f1:.4f}")

print("\nConfusion Matrix")
print("-" * 40)

print(cm)

print("\nClassification Report")
print("-" * 40)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "Normal",
            "Seizure"
        ],
        zero_division=0
    )
)

# ============================================================
# SAVE PREDICTIONS
# ============================================================

results = df.loc[
    y_test.index,
    ["file", "start", "end", "label"]
].copy()

results["anfis_probability"] = probabilities
results["anfis_prediction"] = y_pred

import os

os.makedirs("results/csv", exist_ok=True)

results.to_csv(
    "results/csv/anfis_predictions.csv",
    index=False
)
print("=" * 70)
print("Saved: results/csv/anfis_predictions.csv")
print("=" * 70)