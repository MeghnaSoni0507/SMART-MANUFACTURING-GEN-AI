import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

TARGET_COL = "target"
ID_COL = "Id"


def load_data(csv_path):
    return pd.read_csv(csv_path)


def split_features_target(df):
    X = df.drop(columns=[TARGET_COL, ID_COL])
    y = df[TARGET_COL]
    return X, y


def build_preprocessor(X):
    """
    Create preprocessing pipeline for numeric + categorical data
    """
    numeric_features = X.select_dtypes(include=["number"]).columns
    categorical_features = X.select_dtypes(exclude=["number"]).columns

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor


def preprocess_data(X, y, test_size=0.2, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    return (
        X_train_processed,
        X_val_processed,
        y_train.values,
        y_val.values,
        preprocessor
    )
