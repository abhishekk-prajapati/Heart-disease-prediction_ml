import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def _load_dataset(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path, header=None, names=COLUMN_NAMES, na_values="?")

    # Keep preprocessing aligned with the original app.
    for col in ["ca", "thal"]:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)
    return df


def _build_model(model_name: str):
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
        ),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=7),
    }

    if model_name not in models:
        valid = ", ".join(sorted(models.keys()))
        raise ValueError(f"Unsupported model '{model_name}'. Choose one of: {valid}")

    return models[model_name]


def load_and_train(data_path: str, model_name: str):
    """Load dataset, preprocess, and train the requested model."""
    df = _load_dataset(data_path)

    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = _build_model(model_name)
    model.fit(X_scaled, y)

    return scaler, model
