import sys
import os
import torch
import joblib
import numpy as np
import pandas as pd
from io import StringIO

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

# =====================
# PYTHON PATH FIX
# =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # Backend/app/api
APP_DIR = os.path.dirname(SCRIPT_DIR)                     # Backend/app

if APP_DIR not in sys.path:
    sys.path.append(APP_DIR)

# Import explainability, action, and batch prediction modules
from ml.explainability import (
    get_top_contributing_features, 
    analyze_risk_distribution,
    explain_machine
)
from ml.decision_engine import decision_intelligence
from ml.action_engine import generate_actions, classify_maintenance_urgency, estimate_delay_impact
from ml.batch_predictor import predict_batch

# =====================
# FASTAPI APP
# =====================
app = FastAPI(
    title="Smart Manufacturing API",
    description="PyTorch Manufacturing Failure Predictor",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================
# PYDANTIC MODELS
# =====================
class PredictionRequest(BaseModel):
    """Model for single prediction request"""
    data: Dict[str, Any]

class AnalyzeMachineRequest(BaseModel):
    """Model for machine analysis request"""
    features: Dict[str, Any]
    risk_score: int

class SimulationRequest(BaseModel):
    """Model for what-if simulation request"""
    base_features: Dict[str, Any]
    modifications: Dict[str, Any]

# =====================
# PATH SETUP
# =====================
ARTIFACT_DIR = os.path.join(APP_DIR, "models_artifacts")
print(f"Loading artifacts from: {ARTIFACT_DIR}")

# =====================
# CHECK IF MODELS EXIST
# =====================
required_files = [
    "torch_scaler.pkl",
    "torch_label_encoders.pkl",
    "torch_columns.pkl",
    "torch_failure_model_best.pt"
]

missing_files = []
for file in required_files:
    file_path = os.path.join(ARTIFACT_DIR, file)
    if not os.path.exists(file_path):
        missing_files.append(file)
    else:
        print(f"  ✅ Found: {file}")

if missing_files:
    print("\n" + "="*60)
    print("❌ ERROR: Missing required model files!")
    print("="*60)
    print("\nMissing files:")
    for file in missing_files:
        print(f"  - {file}")
    print(f"\nExpected location: {ARTIFACT_DIR}")
    print("\nPlease train the model first by running:")
    print("  python ml/train_torch.py")
    print("\nfrom the Backend/app directory")
    print("="*60)
    # Don't exit, let FastAPI start anyway for health checks

# =====================
# LOAD ARTIFACTS
# =====================
scaler = None
label_encoders = None
columns = None
model = None

try:
    print("\nLoading model artifacts...")
    scaler = joblib.load(os.path.join(ARTIFACT_DIR, "torch_scaler.pkl"))
    print("  ✅ Scaler loaded")

    label_encoders = joblib.load(os.path.join(ARTIFACT_DIR, "torch_label_encoders.pkl"))
    print(f"  ✅ Label encoders loaded ({len(label_encoders)} encoders)")

    columns = joblib.load(os.path.join(ARTIFACT_DIR, "torch_columns.pkl"))
    print(f"  ✅ Columns loaded ({len(columns)} features)")

    # =====================
    # PYTORCH MODEL
    # =====================
    class ManufacturingNet(torch.nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 64),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(64),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(32, 1)
            )

        def forward(self, x):
            return self.net(x)

    model = ManufacturingNet(input_dim=len(columns))

    # Robust model loading
    model_path = os.path.join(ARTIFACT_DIR, "torch_failure_model_best.pt")
    loaded = torch.load(model_path, map_location=torch.device("cpu"))

    if isinstance(loaded, dict):
        model.load_state_dict(loaded)
    else:
        try:
            model = loaded
        except Exception:
            if hasattr(loaded, 'state_dict'):
                model.load_state_dict(loaded.state_dict())

    model.eval()
    print("  ✅ PyTorch model loaded")
except Exception as e:
    print(f"⚠️  Warning: Could not load models: {str(e)}")

# =====================
# HELPER: PREPROCESS INPUT
# =====================
def preprocess_input(data: dict):
    """
    Preprocess input data for model prediction.
    Handles categorical encoding and scaling.
    """
    if not all([scaler, label_encoders, columns]):
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    row = {}

    for col in columns:
        value = data.get(col)
        
        if value is None:
            raise ValueError(f"Missing required feature: {col}")

        # Handle categorical features
        if col in label_encoders:
            le = label_encoders[col]
            value_str = str(value)
            
            if value_str not in le.classes_:
                print(f"Warning: Unknown value '{value}' for {col}, using fallback")
                value_str = le.classes_[0]
            
            value = le.transform([value_str])[0]

        row[col] = float(value)

    # Convert to numpy array and scale
    X = np.array(list(row.values())).reshape(1, -1)
    X = scaler.transform(X)
    
    return torch.tensor(X, dtype=torch.float32)

# =====================
# API ENDPOINTS
# =====================
@app.get("/")
def read_root():
    """Root endpoint - API information"""
    return {
        "status": "running",
        "message": "API is running",
        "model": "PyTorch Manufacturing Failure Predictor",
        "features": len(columns) if columns else 0,
        "categorical_features": len(label_encoders) if label_encoders else 0,
        "model_loaded": model is not None,
        "endpoints": {
            "/": "GET - API info",
            "/health": "GET - Health check",
            "/features": "GET - Get required features",
            "/predict/torch": "POST - Single prediction",
            "/upload-csv": "POST - Batch CSV prediction",
            "/analyze_machine": "POST - Machine analysis",
            "/simulate": "POST - What-if simulation"
        }
    }

@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features_count": len(columns) if columns else 0,
        "encoders_count": len(label_encoders) if label_encoders else 0
    }

@app.get("/features")
def get_features():
    """Get list of all required features"""
    if not columns:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    feature_info = []
    for col in columns:
        info = {
            "name": col,
            "type": "categorical" if col in label_encoders else "numerical"
        }
        if col in label_encoders:
            info["possible_values"] = label_encoders[col].classes_.tolist()
        feature_info.append(info)
    
    return {
        "features": feature_info,
        "total_count": len(columns)
    }

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """
    Process CSV file with batch machine failure predictions.
    
    Returns failure probability, risk classification, explainability, 
    and actions for each machine in the CSV.
    """
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")

        if not all([model, scaler, label_encoders, columns]):
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Read CSV
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))

        # Use batch predictor module
        batch_results = predict_batch(
            df=df,
            model=model,
            scaler=scaler,
            label_encoders=label_encoders,
            columns=columns,
            verbose=True
        )

        return {
            "total_rows": batch_results["total_rows"],
            "processed_rows": batch_results["processed_rows"],
            "errors": batch_results["errors"],
            "statistics": batch_results["statistics"],
            "results": batch_results["results"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"CSV processing failed: {str(e)}"
        )

@app.post("/predict/torch")
def predict_torch(request: Dict[str, Any]):
    """Single machine prediction endpoint"""
    try:
        if not request:
            raise HTTPException(status_code=400, detail="No data provided")

        if not all([model, scaler, label_encoders, columns]):
            raise HTTPException(status_code=503, detail="Model not loaded")

        x = preprocess_input(request)

        with torch.no_grad():
            logits = model(x)
            raw_score = logits.item()
            failure_probability = torch.sigmoid(logits).item()

        risk_score = int(failure_probability * 100)

        # Calibrated thresholds
        if risk_score >= 65:
            maintenance_priority = "High"
        elif risk_score >= 40:
            maintenance_priority = "Medium"
        else:
            maintenance_priority = "Low"

        # Extract top contributing risk factors
        top_factors = get_top_contributing_features(model, x, columns, top_k=3)
        
        # Generate explanations
        explanations, mapping = explain_machine(request, top_factors, columns)
        
        # Get recommended actions
        recommended_actions = generate_actions(maintenance_priority, top_factors)
        
        # Get urgency context
        urgency_info = classify_maintenance_urgency(maintenance_priority, failure_probability)
        delay_impact = estimate_delay_impact(maintenance_priority, top_factors)

        return {
            "model": "pytorch",
            "raw_score": round(raw_score, 2),
            "failure_probability": round(failure_probability, 4),
            "risk_score": risk_score,
            "maintenance_priority": maintenance_priority,
            "urgency": urgency_info.get("urgency"),
            "timeline": urgency_info.get("timeline"),
            "top_risk_factors": top_factors,
            "explanations": explanations,
            "explanation_mapping": mapping,
            "recommended_actions": recommended_actions,
            "delay_impact": delay_impact,
            "input_features": len(columns),
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/analyze_machine")
def analyze_machine(request: AnalyzeMachineRequest):
    """Machine analysis endpoint"""
    try:
        features = request.features
        risk_score = request.risk_score

        decision = decision_intelligence(features, risk_score)
        return decision

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.post("/simulate")
def simulate(request: SimulationRequest):
    """
    What-if analysis: Show how risk changes when features are modified.
    """
    try:
        if not all([model, scaler, label_encoders, columns]):
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        base_features = request.base_features
        modifications = request.modifications
        
        # Create modified feature dict
        modified_features = base_features.copy()
        modified_features.update(modifications)
        
        # Get original prediction
        x_original = preprocess_input(base_features)
        with torch.no_grad():
            original_prob = torch.sigmoid(model(x_original)).item()
            original_risk = int(original_prob * 100)
        
        # Get new prediction with modifications
        x_modified = preprocess_input(modified_features)
        with torch.no_grad():
            modified_prob = torch.sigmoid(model(x_modified)).item()
            modified_risk = int(modified_prob * 100)
        
        # Calculate impact
        risk_delta = modified_risk - original_risk
        
        return {
            "original_risk_score": original_risk,
            "original_failure_probability": round(original_prob, 4),
            "modified_risk_score": modified_risk,
            "modified_failure_probability": round(modified_prob, 4),
            "risk_delta": risk_delta,
            "improvement": risk_delta < 0,
            "modifications_applied": modifications,
            "status": "success"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Simulation failed: {str(e)}"
        )

# =====================
# STARTUP EVENT
# =====================
@app.on_event("startup")
async def startup_event():
    print("\n" + "="*60)
    print("🚀 FastAPI Server Starting")
    print("="*60)
    print(f"Model artifacts path: {ARTIFACT_DIR}")
    print(f"Features: {len(columns) if columns else 0}")
    print(f"Categorical features: {len(label_encoders) if label_encoders else 0}")
    print(f"Model loaded: {model is not None}")
    print("\nEndpoints:")
    print("  GET  /           - API info")
    print("  GET  /health     - Health check")
    print("  GET  /features   - List all features")
    print("  POST /predict/torch - Make predictions")
    print("  POST /upload-csv - Batch CSV predictions")
    print("  POST /analyze_machine - Machine analysis")
    print("  POST /simulate   - What-if simulation")
    print("  GET  /docs       - Interactive API documentation")
    print("="*60 + "\n")