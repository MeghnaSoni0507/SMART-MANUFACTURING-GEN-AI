import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

from ml.preprocessing import load_data, split_features_target, preprocess_data

# ---------------- Paths ----------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "merged_train_reduced.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "Backend", "app", "models_artifacts")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_failure_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- Load & preprocess ----------------
print("Loading data...")
df = load_data(DATA_PATH)
X, y = split_features_target(df)

X_train, X_val, y_train, y_val, preprocessor = preprocess_data(X, y)

print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)

# ---------------- Train model ----------------
print("\nTraining RandomForest model...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# ---------------- Evaluate ----------------
print("\nEvaluating model...")

y_proba = model.predict_proba(X_val)[:, 1]

# Lower threshold to catch failures (important!)
THRESHOLD = 0.25
y_pred = (y_proba >= THRESHOLD).astype(int)

accuracy = accuracy_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_proba)
cm = confusion_matrix(y_val, y_pred)

print(f"Accuracy (threshold={THRESHOLD}): {accuracy:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_val, y_pred, zero_division=0))

# ---------------- Save model ----------------
joblib.dump(
    {
        "model": model,
        "preprocessor": preprocessor
    },
    MODEL_PATH
)

print(f"\nModel saved at: {MODEL_PATH}")
