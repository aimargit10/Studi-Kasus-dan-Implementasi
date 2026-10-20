import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def load_dataset():
    """Memuat dataset klasifikasi standar dari scikit-learn."""
    data = load_breast_cancer()
    X, y = data.data, data.target

    feature_names = data.feature_names
    target_names = data.target_names

    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    print("Dataset berhasil dimuat.")
    print(f"Jumlah sampel: {df.shape[0]}")
    print(f"Jumlah fitur: {df.shape[1] - 1}")
    print(f"Kelas target: {target_names.tolist()}\n")

    return df, target_names


def prepare_data(df: pd.DataFrame):
    """Memisahkan fitur dan label serta membagi data train-test 80:20."""
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def train_and_evaluate(model_name: str, model, X_train, X_test, y_train, y_test):
    """Melatih model, menghitung skor train/test, dan menampilkan evaluasi."""
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)

    y_pred = pipeline.predict(X_test)

    print(f"\n=== {model_name} ===")
    print(f"Training Accuracy: {train_score:.4f}")
    print(f"Testing Accuracy:  {test_score:.4f}")

    if test_score < train_score - 0.05:
        print("Potensi overfitting terdeteksi: selisih training-testing cukup besar.")
    elif test_score > train_score + 0.05:
        print("Potensi underfitting terdeteksi: performa testing lebih tinggi dari training.")
    else:
        print("Kinerja model relatif seimbang antara train dan test.")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))


def main():
    """Pipeline utama untuk load data, training, dan evaluasi model."""
    df, _ = load_dataset()
    X_train, X_test, y_train, y_test = prepare_data(df)

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(kernel="rbf", random_state=42),
    }

    for model_name, model in models.items():
        train_and_evaluate(model_name, model, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
