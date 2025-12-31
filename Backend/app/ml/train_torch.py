import os
import torch
import torch.nn as nn
import pandas as pd
import joblib
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# =====================
# PATH SETUP - CORRECTED FOR PROJECT ROOT
# =====================
# Get the directory where this script is located (Backend/app/ml)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level to Backend/app
APP_DIR = os.path.dirname(SCRIPT_DIR)

# Go up one more level to Backend
BACKEND_DIR = os.path.dirname(APP_DIR)

# Go up one more level to project root (SMART MANUFACTURING GENAI)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

# Data is at project root level
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Models artifacts in Backend/app/models_artifacts
ARTIFACT_DIR = os.path.join(APP_DIR, "models_artifacts")

# Create artifacts directory if it doesn't exist
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# =====================
# VERIFY PATHS
# =====================
csv_path = os.path.join(DATA_DIR, "merged_train_reduced.csv")

print("Path verification:")
print(f"  Project root: {PROJECT_ROOT}")
print(f"  Data directory: {DATA_DIR}")
print(f"  CSV file: {csv_path}")
print(f"  Artifacts directory: {ARTIFACT_DIR}")

if not os.path.exists(csv_path):
    print(f"\n❌ ERROR: File not found at: {csv_path}")
    print("\nPlease verify:")
    print(f"  1. Data folder exists at: {DATA_DIR}")
    print(f"  2. File exists: merged_train_reduced.csv")
    exit(1)

# =====================
# PYTORCH MODEL
# =====================
class ManufacturingNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# =====================
# LOAD DATA
# =====================
print(f"\nLoading data from: {csv_path}")

df = pd.read_csv(csv_path)
print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

# Check if 'target' column exists
if 'target' not in df.columns:
    print(f"\n❌ ERROR: 'target' column not found!")
    print(f"Available columns: {', '.join(df.columns)}")
    exit(1)

# Separate features and target
y = df["target"].values.reshape(-1, 1)
X = df.drop("target", axis=1)

# =====================
# HANDLE CATEGORICAL FEATURES
# =====================
print("\nProcessing features...")

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"  Categorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"  Numerical columns ({len(numerical_cols)}): {numerical_cols}")

# Encode categorical features
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
    encoders_path = os.path.join(ARTIFACT_DIR, "torch_label_encoders.pkl")
    joblib.dump(label_encoders, encoders_path)
    print(f"\n✅ Label encoders saved to: {encoders_path}")

# =====================
# SCALING
# =====================
print("\nScaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

scaler_path = os.path.join(ARTIFACT_DIR, "torch_scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler saved to: {scaler_path}")

# Save column names for later use
columns_path = os.path.join(ARTIFACT_DIR, "torch_columns.pkl")
joblib.dump(X.columns.tolist(), columns_path)
print(f"✅ Column names saved to: {columns_path}")

# =====================
# TRAIN SPLIT
# =====================
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Features: {X_train.shape[1]}")

# =====================
# TORCH TENSORS
# =====================
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

dataset = TensorDataset(X_train_tensor, y_train_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# =====================
# TRAINING
# =====================
model = ManufacturingNet(input_dim=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nStarting training...")
print("-" * 50)

best_val_loss = float('inf')

for epoch in range(30):
    # Training
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor)
        val_loss = criterion(val_preds, y_val_tensor).item()
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_path = os.path.join(ARTIFACT_DIR, "torch_failure_model_best.pt")
        torch.save(model.state_dict(), best_model_path)
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:2d}/30 | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")

# =====================
# SAVE FINAL MODEL
# =====================
model_path = os.path.join(ARTIFACT_DIR, "torch_failure_model.pt")
torch.save(model.state_dict(), model_path)

print("-" * 50)
print("\n✅ Training complete!")
print(f"✅ Final model saved to: {model_path}")
print(f"✅ Best model saved to: {best_model_path}")
print(f"✅ Best validation loss: {best_val_loss:.4f}")
print(f"✅ Scaler saved to: {scaler_path}")
if label_encoders:
    print(f"✅ Label encoders saved to: {encoders_path}")
print(f"✅ Column names saved to: {columns_path}")