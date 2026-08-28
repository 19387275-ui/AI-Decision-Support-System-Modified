"""Evaluate the CBR method without train/test scaling leakage."""

from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from src.config import DATASET_PATH, EVALUATION_DIR, FEATURE_COLUMNS, NUMERICAL_COLUMNS
from src.preprocess import load_and_clean_dataset


def evaluate(test_size: float = 0.20, random_state: int = 42, k: int = 5):
    df = load_and_clean_dataset()
    X = df[FEATURE_COLUMNS].copy()
    y = df["loan_status"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit the scaler only on training data to prevent test-set information leakage.
    scaler = MinMaxScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[NUMERICAL_COLUMNS] = scaler.fit_transform(X_train[NUMERICAL_COLUMNS])
    X_test[NUMERICAL_COLUMNS] = scaler.transform(X_test[NUMERICAL_COLUMNS])

    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(X_train[FEATURE_COLUMNS])

    predictions = []
    for i in range(len(X_test)):
        _, indices = knn.kneighbors(X_test.iloc[[i]][FEATURE_COLUMNS])
        neighbour_labels = y_train.iloc[indices[0]]
        predictions.append(int(neighbour_labels.sum() >= (k // 2 + 1)))

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }
    cm = confusion_matrix(y_test, predictions)

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Actual": y_test.values, "Predicted": predictions}).to_csv(
        EVALUATION_DIR / "predictions.csv", index=False
    )
    with open(EVALUATION_DIR / "evaluation_results.txt", "w", encoding="utf-8") as f:
        f.write("CBR SYSTEM EVALUATION\n\n")
        f.write(f"Dataset: {Path(DATASET_PATH).name}\n")
        f.write(f"Test size: {test_size:.0%}\n")
        f.write(f"Random state: {random_state}\n")
        f.write(f"Number of neighbours (k): {k}\n")
        f.write("Scaler fitted on training data only: Yes\n\n")
        f.write(f"Accuracy : {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall   : {metrics['recall']:.4f}\n")
        f.write(f"F1 Score : {metrics['f1']:.4f}\n\n")
        f.write("Confusion Matrix\n")
        f.write(str(cm))
    return metrics, cm


if __name__ == "__main__":
    metrics, cm = evaluate()
    for name, value in metrics.items():
        print(f"{name.title():9}: {value:.4f}")
    print("Confusion Matrix")
    print(cm)
