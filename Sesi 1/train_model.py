import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_preprocessing import load_dataset, preprocess_data


def train_and_evaluate_model(model_name: str, model, X_train, X_test, y_train, y_test) -> None:
    """Melatih model, melakukan prediksi, dan menampilkan hasil evaluasi."""
    pipeline = Pipeline(
        steps=[
            ("preprocessor", model["preprocessor"]),
            ("model", model["estimator"]),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(f"\n=== {model_name} ===")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.colorbar()
    tick_labels = ["No Churn", "Churn"]
    plt.xticks([0, 1], tick_labels)
    plt.yticks([0, 1], tick_labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Proses utama: load data, preprocessing, split data, training, dan evaluasi model."""
    # Langkah 1: memuat dataset dari CSV lokal atau dataset publik default.
    raw_df = load_dataset()

    # Langkah 2: preprocessing awal untuk membersihkan data dan mempersiapkan feature.
    X, y, preprocessor = preprocess_data(raw_df, target_col="Churn")

    # Langkah 3: split data menjadi data latih dan data uji agar evaluasi lebih realistis.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=42),
            "preprocessor": preprocessor,
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
            "preprocessor": preprocessor,
        },
    }

    for model_name, model in models.items():
        train_and_evaluate_model(model_name, model, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
