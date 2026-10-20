import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATASET_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-ibm-dataset/master/data/Telco-Customer-Churn.csv"


def load_dataset(csv_path: str | None = None) -> pd.DataFrame:
    """Memuat dataset churn pelanggan, mencoba CSV lokal lalu dataset publik default."""
    if csv_path:
        return pd.read_csv(csv_path)

    try:
        return pd.read_csv(DATASET_URL)
    except Exception:
        return generate_dummy_telco_data()


def generate_dummy_telco_data(n_samples: int = 1000) -> pd.DataFrame:
    """Membuat dataset dummy yang menyerupai karakteristik Telco Churn."""
    rng = np.random.default_rng(42)

    genders = rng.choice(["Male", "Female"], size=n_samples)
    senior_citizen = rng.integers(0, 2, size=n_samples)
    partner = rng.choice(["Yes", "No"], size=n_samples)
    dependents = rng.choice(["Yes", "No"], size=n_samples)
    phone_service = rng.choice(["Yes", "No"], size=n_samples)
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], size=n_samples)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n_samples)
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n_samples,
    )
    tenure = rng.integers(1, 72, size=n_samples)
    monthly_charges = rng.uniform(25, 120, size=n_samples)
    total_charges = monthly_charges * (tenure + rng.uniform(0.5, 3.0, size=n_samples))

    churn_prob = (
        0.15
        + 0.25 * (senior_citizen == 1)
        + 0.10 * (internet_service == "Fiber optic")
        + 0.12 * (contract == "Month-to-month")
        + 0.08 * (payment_method == "Electronic check")
        - 0.05 * (partner == "Yes")
        - 0.04 * (dependents == "Yes")
    )
    churn = rng.random(n_samples) < churn_prob
    churn_label = np.where(churn, "Yes", "No")

    df = pd.DataFrame(
        {
            "customerID": [f"CUST-{i:04d}" for i in range(1, n_samples + 1)],
            "gender": genders,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "InternetService": internet_service,
            "Contract": contract,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn_label,
        }
    )

    # Menyisipkan missing value secara sengaja agar proses preprocessing terlihat relevan.
    missing_cols = ["TotalCharges", "InternetService", "MonthlyCharges"]
    for col in missing_cols:
        num_missing = max(5, int(len(df) * 0.02))
        missing_idx = rng.choice(df.index, size=num_missing, replace=False)
        df.loc[missing_idx, col] = np.nan

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan nama kolom dan nilai target agar konsisten untuk pemodelan."""
    df = df.copy()
    df.columns = [col.strip().replace(" ", "") for col in df.columns]

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].astype(str).str.strip().str.lower()
        df["Churn"] = df["Churn"].map({"yes": "Yes", "no": "No", "true": "Yes", "false": "No", "1": "Yes", "0": "No"})
        df["Churn"] = df["Churn"].fillna("No")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Mengisi nilai hilang pada fitur numerik dan kategorikal menggunakan strategi yang sesuai."""
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if col != "Churn":
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")

    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Membuat preprocessor yang memisahkan fitur numerik dan kategorikal."""
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Menghapus customerID agar tidak dianggap sebagai feature predictive.
    if "customerID" in categorical_features:
        categorical_features.remove("customerID")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def prepare_features(df: pd.DataFrame, target_col: str = "Churn") -> tuple[pd.DataFrame, pd.Series]:
    """Mempersiapkan feature dan target untuk proses training model."""
    df = clean_dataset(df)
    df = handle_missing_values(df)

    if target_col not in df.columns:
        raise ValueError(f"Kolom target '{target_col}' tidak ditemukan pada dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col].map({"Yes": 1, "No": 0})

    # Menghilangkan kolom identifier yang tidak relevan untuk prediksi.
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])

    return X, y


def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """Fungsi utama preprocessing yang mengembalikan feature, target, dan preprocessor."""
    X, y = prepare_features(df, target_col=target_col)
    preprocessor = build_preprocessor(X)
    return X, y, preprocessor
