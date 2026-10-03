"""Compare recall-oriented Random Forest strategies on one untouched test split.

Run from the repository root with: python src/models/train.py
The threshold is chosen on validation predictions only. Oversampling is applied
only to the training partition.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20  # fraction of the remaining training data
BASELINE_THRESHOLD = 0.50
# Prefer recall while avoiding a threshold that achieves it by flagging nearly
# everything. Adjust this operating constraint for the intended use case.
MIN_VALIDATION_PRECISION = 0.25
THRESHOLDS = np.arange(0.05, 0.96, 0.05)


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    root = Path(__file__).resolve().parents[2]
    path = root / "data" / "processed" / "features.csv"
    df = pd.read_csv(path)
    X = df.drop(columns=["label", "file", "start", "end"])
    y = df["label"].astype(int)
    return X, y


def choose_recall_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Maximize validation recall subject to a minimum precision constraint."""
    candidates = sorted(set(THRESHOLDS.tolist() + [BASELINE_THRESHOLD]))
    scored = []
    for threshold in candidates:
        predictions = probabilities >= threshold
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        scored.append((threshold, precision, recall))
    eligible = [item for item in scored if item[1] >= MIN_VALIDATION_PRECISION]
    # If none meets the precision floor, take the best F1 threshold on validation.
    if not eligible:
        return max(
            scored,
            key=lambda item: (
                f1_score(y_true, probabilities >= item[0], zero_division=0),
                item[2],
                item[1],
            ),
        )[0]
    return max(eligible, key=lambda item: (item[2], item[1], -item[0]))[0]


def make_model(class_weight: str | None = None) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight=class_weight,
        n_jobs=-1,
    )


def oversample_minority(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Randomly duplicate minority training rows, without touching validation/test."""
    joined = X.copy()
    joined["__label__"] = y.to_numpy()
    counts = y.value_counts()
    majority = counts.idxmax()
    minority = counts.idxmin()
    minority_rows = joined[joined["__label__"] == minority]
    extra = minority_rows.sample(
        n=int(counts[majority] - counts[minority]),
        replace=True,
        random_state=RANDOM_STATE,
    )
    balanced = pd.concat([joined, extra], ignore_index=True).sample(
        frac=1, random_state=RANDOM_STATE
    )
    return balanced.drop(columns="__label__"), balanced["__label__"].astype(int)


def metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
    }


def main() -> None:
    X, y = load_dataset()
    print(f"Dataset: {len(y)} windows, {X.shape[1]} features")
    print("Class counts:", y.value_counts().sort_index().to_dict())

    # The test partition is fixed once and used only for final reporting.
    X_development, X_test, y_development, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_development,
    )
    print(
        f"Split sizes: train={len(y_train)}, validation={len(y_validation)}, "
        f"test={len(y_test)} (test positives={int(y_test.sum())})"
    )

    strategies = {
        "unweighted": (X_train, y_train, None),
        "class_weight_balanced": (X_train, y_train, "balanced"),
    }
    X_over, y_over = oversample_minority(X_train, y_train)
    strategies["random_oversampling"] = (X_over, y_over, None)

    rows = []
    for name, (fit_X, fit_y, class_weight) in strategies.items():
        model = make_model(class_weight=class_weight)
        model.fit(fit_X, fit_y)
        validation_prob = model.predict_proba(X_validation)[:, 1]
        selected_threshold = choose_recall_threshold(y_validation, validation_prob)
        # Test predictions are generated only after the threshold is fixed.
        test_prob = model.predict_proba(X_test)[:, 1]
        selected = metrics(y_test, test_prob, selected_threshold)
        fixed = metrics(y_test, test_prob, BASELINE_THRESHOLD)
        rows.append({"strategy": name, "evaluation": "validation_selected", **selected})
        rows.append({"strategy": name, "evaluation": "fixed_0.50", **fixed})

        print(f"\n{name} (validation-selected threshold={selected_threshold:.2f})")
        print(
            f"  held-out test: recall={selected['recall']:.3f}, "
            f"precision={selected['precision']:.3f}, F1={selected['f1']:.3f}, "
            f"FP={selected['false_positive']}, FN={selected['false_negative']}"
        )
        print(
            f"  same test at 0.50: recall={fixed['recall']:.3f}, "
            f"precision={fixed['precision']:.3f}, F1={fixed['f1']:.3f}, "
            f"FP={fixed['false_positive']}, FN={fixed['false_negative']}"
        )

    output_path = Path(__file__).resolve().parents[2] / "results" / "csv" / "recall_tuning_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\nSaved held-out comparison to {output_path}")
    print(
        "Threshold rule: maximize validation recall subject to precision >= "
        f"{MIN_VALIDATION_PRECISION:.2f}; test data did not select the threshold."
    )


if __name__ == "__main__":
    main()
