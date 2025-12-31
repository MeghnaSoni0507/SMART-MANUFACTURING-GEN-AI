import sys
import os
import torch
import joblib
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

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

app = Flask(__name__)
CORS(app)

# =====================
# PATH SETUP
# =====================
# Get the directory where this script is located (Backend/app/api)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level to Backend/app
APP_DIR = os.path.dirname(SCRIPT_DIR)

# Artifacts directory
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
    exit(1)

# =====================
# LOAD ARTIFACTS
# =====================
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

# Robust model loading: support both state_dict and full model files
model_path = os.path.join(ARTIFACT_DIR, "torch_failure_model_best.pt")
loaded = torch.load(model_path, map_location=torch.device("cpu"))

if isinstance(loaded, dict):
    # Assume it's a state_dict
    model.load_state_dict(loaded)
else:
    # Assume it's a full model object — replace our instance
    try:
        model = loaded
    except Exception:
        # Fall back to loading state_dict from object's state_dict()
        if hasattr(loaded, 'state_dict'):
            model.load_state_dict(loaded.state_dict())

model.eval()
print("  ✅ PyTorch model loaded")

# =====================
# HELPER: PREPROCESS INPUT
# =====================
def preprocess_input(data: dict):
    """
    Preprocess input data for model prediction.
    Handles categorical encoding and scaling.
    """
    row = {}

    for col in columns:
        value = data.get(col)
        
        if value is None:
            raise ValueError(f"Missing required feature: {col}")

        # Handle categorical features
        if col in label_encoders:
            le = label_encoders[col]
            # Convert to string for encoding
            value_str = str(value)
            
            # Check if value is known
            if value_str not in le.classes_:
                print(f"Warning: Unknown value '{value}' for {col}, using fallback")
                value_str = le.classes_[0]  # Use first known value as fallback
            
            value = le.transform([value_str])[0]

        row[col] = float(value)

    # Convert to numpy array and scale
    X = np.array(list(row.values())).reshape(1, -1)
    X = scaler.transform(X)
    
    return torch.tensor(X, dtype=torch.float32)

# =====================
# API ENDPOINTS
# =====================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "model": "PyTorch Manufacturing Failure Predictor",
        "features": len(columns),
        "categorical_features": len(label_encoders),
        "endpoints": {
            "/predict/torch": "POST - Single prediction",
            "/upload-csv": "POST - Batch CSV prediction",
            "/health": "GET - Check model health",
            "/features": "GET - Get required features"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "features_count": len(columns),
        "encoders_count": len(label_encoders)
    })

@app.route("/features", methods=["GET"])
def get_features():
    feature_info = []
    for col in columns:
        info = {
            "name": col,
            "type": "categorical" if col in label_encoders else "numerical"
        }
        if col in label_encoders:
            info["possible_values"] = label_encoders[col].classes_.tolist()
        feature_info.append(info)
    
    return jsonify({
        "features": feature_info,
        "total_count": len(columns)
    })

@app.route("/upload-csv", methods=["POST"])
def upload_csv():
    """
    Process CSV file with batch machine failure predictions.
    
    Uses batch_predictor module for clean separation of prediction logic.
    Returns failure probability, risk classification, explainability, and actions
    for each machine in the CSV.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if not file.filename.endswith(".csv"):
            return jsonify({"error": "Only CSV files are supported"}), 400

        # Read CSV
        df = pd.read_csv(file)

        # Use batch predictor module (clean separation of concerns)
        batch_results = predict_batch(
            df=df,
            model=model,
            scaler=scaler,
            label_encoders=label_encoders,
            columns=columns,
            verbose=True  # Print statistics to console
        )

        return jsonify({
            "total_rows": batch_results["total_rows"],
            "processed_rows": batch_results["processed_rows"],
            "errors": batch_results["errors"],
            "statistics": batch_results["statistics"],
            "results": batch_results["results"]
        })

    except Exception as e:
        return jsonify({
            "error": "CSV processing failed",
            "message": str(e)
        }), 500

@app.route("/predict/torch", methods=["POST"])
def predict_torch():
    try:
        data = request.json

        if not data:
            return jsonify({"error": "No data provided"}), 400

        x = preprocess_input(data)

        with torch.no_grad():
            logits = model(x)
            raw_score = logits.item()
            
            # Apply sigmoid to convert logit to probability [0, 1]
            failure_probability = torch.sigmoid(logits).item()

        risk_score = int(failure_probability * 100)

        # Calibrated thresholds: High >= 65%, Medium >= 40%, Low < 40%
        if risk_score >= 65:
            maintenance_priority = "High"
        elif risk_score >= 40:
            maintenance_priority = "Medium"
        else:
            maintenance_priority = "Low"

        # Extract top contributing risk factors (using new explainability module)
        top_factors = get_top_contributing_features(model, x, columns, top_k=3)
        
        # Generate human-readable explanations (uses mapped sensor names and top factors)
        explanations, mapping = explain_machine(data, top_factors, columns)
        # Temporary debug print to verify per-machine output
        try:
            print(data.get("Id", data.get("id", "unknown")), explanations)
        except Exception:
            pass
        
        # Get recommended actions (using new action engine module)
        recommended_actions = generate_actions(maintenance_priority, top_factors)
        
        # Get urgency context
        urgency_info = classify_maintenance_urgency(maintenance_priority, failure_probability)
        delay_impact = estimate_delay_impact(maintenance_priority, top_factors)

        return jsonify({
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
        })

    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "message": str(e)
        }), 500

@app.route("/analyze_machine", methods=["POST"])
def analyze_machine():
    try:
        data = request.json

        if not data or "features" not in data or "risk_score" not in data:
            return jsonify({"error": "Missing 'features' or 'risk_score' in request"}), 400

        features = data.get("features", {})
        risk_score = int(data.get("risk_score", 0))

        decision = decision_intelligence(features, risk_score)

        return jsonify(decision)

    except Exception as e:
        return jsonify({"error": "Analysis failed", "message": str(e)}), 500

@app.route("/simulate", methods=["POST"])
def simulate():
    """
    What-if analysis: Show how risk changes when features are modified.
    
    Request body:
    {
        "base_features": {...current values...},
        "modifications": {"vibration": 0.2, "temperature": 25}
    }
    
    Returns:
        Original risk score and new risk score after modifications
    """
    try:
        data = request.json
        
        if not data or "base_features" not in data:
            return jsonify({"error": "Missing 'base_features'"}), 400
        
        base_features = data.get("base_features", {})
        modifications = data.get("modifications", {})
        
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
        
        return jsonify({
            "original_risk_score": original_risk,
            "original_failure_probability": round(original_prob, 4),
            "modified_risk_score": modified_risk,
            "modified_failure_probability": round(modified_prob, 4),
            "risk_delta": risk_delta,
            "improvement": risk_delta < 0,
            "modifications_applied": modifications,
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({
            "error": "Simulation failed",
            "message": str(e)
        }), 500

# =====================
# RUN APP
# =====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Flask API Server Starting")
    print("="*60)
    print(f"Model artifacts loaded from: {ARTIFACT_DIR}")
    print(f"Features: {len(columns)}")
    print(f"Categorical features: {len(label_encoders)}")
    print("\nEndpoints:")
    print("  GET  /           - API info")
    print("  GET  /health     - Health check")
    print("  GET  /features   - List all features")
    print("  POST /predict/torch - Make predictions")
    print("  POST /upload-csv - Batch CSV predictions")
    print("  POST /analyze_machine - Machine analysis")
    print("  POST /simulate   - What-if simulation")
    print("="*60 + "\n")
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)