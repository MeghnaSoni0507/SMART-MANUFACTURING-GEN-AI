import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# =====================
# PATH SETUP
# =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(SCRIPT_DIR)
BACKEND_DIR = os.path.dirname(APP_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ARTIFACT_DIR = os.path.join(APP_DIR, "models_artifacts")

os.makedirs(ARTIFACT_DIR, exist_ok=True)

print("Path verification:")
print(f"  Project root: {PROJECT_ROOT}")
print(f"  Data directory: {DATA_DIR}")
print(f"  Artifacts directory: {ARTIFACT_DIR}")

# =====================
# LOAD DATA
# =====================
csv_path = os.path.join(DATA_DIR, "merged_train_reduced.csv")
print(f"\nLoading data from: {csv_path}")

df = pd.read_csv(csv_path)
print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

# Check if target exists
if "target" not in df.columns:
    print(f"\n❌ ERROR: 'target' column not found!")
    print(f"Available columns: {', '.join(df.columns)}")
    exit(1)

y = df["target"].values
X = df.drop("target", axis=1)

# =====================
# ENCODE CATEGORICAL FEATURES
# =====================
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nFeature types:")
print(f"  Categorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"  Numerical columns ({len(numerical_cols)}): {numerical_cols}")

# Store label encoders for inference
label_encoders = {}
X_encoded = X.copy()

if categorical_cols:
    print("\nEncoding categorical features...")
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"  {col}: {len(le.classes_)} unique values")
    
    # Save label encoders
    encoders_path = os.path.join(ARTIFACT_DIR, "xgboost_label_encoders.pkl")
    joblib.dump(label_encoders, encoders_path)
    print(f"\n✅ Label encoders saved to: {encoders_path}")

# Save column names
columns_path = os.path.join(ARTIFACT_DIR, "xgboost_columns.pkl")
joblib.dump(X.columns.tolist(), columns_path)
print(f"✅ Column names saved to: {columns_path}")

# =====================
# TRAIN / VAL SPLIT
# =====================
X_train, X_val, y_train, y_val = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Features: {X_train.shape[1]}")

# =====================
# XGBOOST MODEL
# =====================
print("\nTraining XGBoost model...")

model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    verbosity=1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=50
)

# =====================
# EVALUATION
# =====================
print("\n" + "="*60)
print("Model Evaluation")
print("="*60)

# Training predictions
train_preds = model.predict(X_train)
train_mse = mean_squared_error(y_train, train_preds)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, train_preds)

print(f"\nTraining Set:")
print(f"  RMSE: {train_rmse:.4f}")
print(f"  R²:   {train_r2:.4f}")

# Validation predictions
val_preds = model.predict(X_val)
val_mse = mean_squared_error(y_val, val_preds)
val_rmse = np.sqrt(val_mse)
val_r2 = r2_score(y_val, val_preds)

print(f"\nValidation Set:")
print(f"  RMSE: {val_rmse:.4f}")
print(f"  R²:   {val_r2:.4f}")

# =====================
# FEATURE IMPORTANCE
# =====================
print("\n" + "="*60)
print("Top 10 Most Important Features")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:30s} {row['importance']:.4f}")

# =====================
# SAVE MODEL
# =====================
model_path = os.path.join(ARTIFACT_DIR, "xgboost_failure_model.pkl")
joblib.dump(model, model_path)

print("\n" + "="*60)
print("✅ Training Complete!")
print("="*60)
print(f"✅ XGBoost model saved to: {model_path}")
if label_encoders:
    print(f"✅ Label encoders saved to: {encoders_path}")
print(f"✅ Column names saved to: {columns_path}")
print(f"\nValidation RMSE: {val_rmse:.4f}")
print(f"Validation R²: {val_r2:.4f}")