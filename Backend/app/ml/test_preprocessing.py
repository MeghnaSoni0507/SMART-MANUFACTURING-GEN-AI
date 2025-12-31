import os
from ml.preprocessing import load_data, split_features_target, preprocess_data

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "merged_train_reduced.csv")

print("Loading data from:", DATA_PATH)

df = load_data(DATA_PATH)
X, y = split_features_target(df)

X_train, X_val, y_train, y_val, preprocessor = preprocess_data(X, y)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Target distribution (train):", y_train.mean())
