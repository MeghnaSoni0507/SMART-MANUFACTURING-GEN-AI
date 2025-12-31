"""
Batch Prediction Engine for Manufacturing Machine Maintenance
============================================================

Provides efficient batch prediction of machine failure risk with explanations
and recommended maintenance actions. Designed for CSV/bulk processing.

Usage:
    from batch_predictor import predict_batch
    
    results = predict_batch(
        df=loaded_dataframe,
        model=pytorch_model,
        scaler=preprocessor,
        label_encoders=encoders_dict,
        columns=feature_list
    )
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Use relative imports since this module is in the ml package
try:
    # When imported as ml.batch_predictor
    from .explainability import get_top_contributing_features, explain_machine, set_thresholds
    from .action_engine import generate_actions, classify_maintenance_urgency
    from .decision_engine import decision_intelligence
except ImportError:
    # Fallback for direct imports
    from explainability import get_top_contributing_features, explain_machine, set_thresholds
    from action_engine import generate_actions, classify_maintenance_urgency
    from decision_engine import decision_intelligence


def preprocess_row(row_dict: Dict, scaler, label_encoders: Dict, columns: List[str]) -> torch.Tensor:
    """
    Convert a single row dict to preprocessed tensor.
    
    Args:
        row_dict: Dictionary with feature values
        scaler: Fitted StandardScaler object
        label_encoders: Dict of LabelEncoder objects for categorical features
        columns: List of feature names in correct order
    
    Returns:
        torch.Tensor: Preprocessed tensor [1, n_features]
    """
    row = {col: row_dict.get(col) for col in columns}
    
    # Encode categorical features
    for col, le in label_encoders.items():
        if col in row:
            value = row[col]
            value_str = str(value)
            
            # Check if value is known
            if value_str not in le.classes_:
                value_str = le.classes_[0]  # Use first known value as fallback
            
            value = le.transform([value_str])[0]
            row[col] = float(value)
    
    # Convert to numpy array and scale
    X = np.array(list(row.values())).reshape(1, -1)
    X = scaler.transform(X)
    
    return torch.tensor(X, dtype=torch.float32)


def predict_single(
    row_dict: Dict,
    model: torch.nn.Module,
    scaler,
    label_encoders: Dict,
    columns: List[str]
) -> Dict[str, Any]:
    """
    Generate a single prediction with explanations and actions.
    
    Args:
        row_dict: Dictionary with feature values for one machine
        model: Trained PyTorch model
        scaler: Fitted StandardScaler
        label_encoders: Dict of LabelEncoder objects
        columns: List of feature names
    
    Returns:
        Dictionary with:
        - failure_probability: Float [0, 1]
        - risk_score: Int [0, 100]
        - maintenance_priority: "High" | "Medium" | "Low"
        - top_risk_factors: List of dicts with {"feature": name, "impact_score": float}
        - explanations: List of human-readable diagnostic strings
        - recommended_actions: List of maintenance task strings
        - urgency: "CRITICAL" | "WARNING" | "NOMINAL"
        - timeline: "24h" | "7d" | "routine"
    """
    # Preprocess
    x = preprocess_row(row_dict, scaler, label_encoders, columns)
    
    # Inference
    with torch.no_grad():
        logits = model(x)
        failure_probability = torch.sigmoid(logits).item()
    
    risk_score = int(failure_probability * 100)
    
    # Classify priority (calibrated thresholds: High ≥ 65%, Medium ≥ 40%, Low < 40%)
    if risk_score >= 65:
        maintenance_priority = "High"
    elif risk_score >= 40:
        maintenance_priority = "Medium"
    else:
        maintenance_priority = "Low"
    
    # Extract explainability
    top_factors = get_top_contributing_features(model, x, columns, top_k=3)
    # Use explain_machine with both raw row and top contributing features
    explanations, mapping = explain_machine(row_dict, top_factors, columns)
    
    # Generate recommendations
    recommended_actions = generate_actions(maintenance_priority, top_factors)
    urgency_info = classify_maintenance_urgency(maintenance_priority, failure_probability)
    # Build a simple features dict for decision intelligence using mapped values when available
    features_for_decision = {}
    try:
        features_for_decision['vibration'] = mapping.get('vibration', {}).get('value')
    except Exception:
        features_for_decision['vibration'] = row_dict.get('vibration', row_dict.get('Vibration', 0))

    try:
        features_for_decision['temperature'] = mapping.get('temperature', {}).get('value')
    except Exception:
        features_for_decision['temperature'] = row_dict.get('temperature', row_dict.get('Temperature', 0))

    try:
        features_for_decision['pressure'] = mapping.get('pressure', {}).get('value')
    except Exception:
        features_for_decision['pressure'] = row_dict.get('pressure', row_dict.get('Pressure', 0))

    # Decision intelligence: combine risk with failure-mode rules
    try:
        decision = decision_intelligence(features_for_decision, risk_score)
    except Exception:
        decision = {"risk_score": risk_score, "risk_level": maintenance_priority, "failure_modes": [], "action_required": False}
    
    return {
        "failure_probability": round(failure_probability, 4),
        "risk_score": risk_score,
        "maintenance_priority": maintenance_priority,
        "top_risk_factors": top_factors,
        "explanations": explanations,
        "explanation_mapping": mapping,
        "recommended_actions": recommended_actions,
        "urgency": urgency_info.get("urgency"),
        "timeline": urgency_info.get("timeline"),
        "decision": decision
    }


def predict_batch(
    df: pd.DataFrame,
    model: torch.nn.Module,
    scaler,
    label_encoders: Dict,
    columns: List[str],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Batch predict machine failure risk for multiple machines in a DataFrame.
    
    This is the main entry point for batch CSV processing. Handles:
    - Row-by-row prediction with error handling
    - Feature extraction and explanation generation
    - Action mapping and prioritization
    - Debug logging and statistics
    
    Args:
        df: Pandas DataFrame with feature rows
        model: Trained PyTorch model
        scaler: Fitted StandardScaler for feature normalization
        label_encoders: Dict mapping column names to fitted LabelEncoders
        columns: List of feature column names (in order)
        verbose: If True, print probability distribution and debug info
    
    Returns:
        Dictionary containing:
        - total_rows: Number of rows in input DataFrame
        - processed_rows: Number of successfully processed rows
        - results: List of prediction dicts (one per row) with:
            * row_index: Original DataFrame index
            * failure_probability: Float [0, 1]
            * risk_score: Int [0, 100]
            * maintenance_priority: "High" | "Medium" | "Low"
            * top_risk_factors: Dicts with {"feature": name, "impact_score": float}
            * explanations: List of diagnostic strings
            * recommended_actions: List of task strings
            * urgency: "CRITICAL" | "WARNING" | "NOMINAL"
            * timeline: "24h" | "7d" | "routine"
        - statistics: Dict with min/max/mean probability and risk distribution counts
        - errors: Count of rows that failed processing
    
    Example:
        >>> results = predict_batch(csv_df, model, scaler, encoders, features)
        >>> for pred in results['results']:
        ...     if pred['maintenance_priority'] == 'High':
        ...         print(f"Machine {pred['row_index']}: {pred['explanations']}")
    """
    results = []
    errors_count = 0
    probabilities = []
    # Print column names once so users can verify expected feature keys
    try:
        print("Input DataFrame columns:", df.columns.tolist())
        expected = ["vibration", "temperature", "pressure"]
        missing = [c for c in expected if c not in df.columns]
        if missing:
            print("⚠️ Missing expected feature names:", missing)
            print("Ensure dataset columns include 'vibration', 'temperature', 'pressure' or adjust explainability rules.")
    except Exception:
        # Non-standard df, skip printing
        pass
    # Attempt to compute dataset-driven thresholds for vibration/temperature/pressure
    try:
        thresholds = {}
        cols = list(df.columns)
        lower_cols = [c.lower() for c in cols]

        # Vibration column detection
        vib_idx = None
        for i, c in enumerate(lower_cols):
            if 'vib' in c or 'vibration' in c:
                vib_idx = i
                break
        if vib_idx is not None:
            colname = cols[vib_idx]
            try:
                vals = pd.to_numeric(df[colname], errors='coerce').dropna()
                if len(vals) > 0:
                    thresholds['vibration'] = float(np.percentile(vals, 90))
                    print(f"Dataset vibration threshold (90p) from {colname}: {thresholds['vibration']}")
            except Exception:
                pass

        # Temperature column detection
        temp_idx = None
        for i, c in enumerate(lower_cols):
            if 'temp' in c or 'temperature' in c:
                temp_idx = i
                break
        if temp_idx is not None:
            colname = cols[temp_idx]
            try:
                vals = pd.to_numeric(df[colname], errors='coerce').dropna()
                if len(vals) > 0:
                    thresholds['temperature'] = float(np.percentile(vals, 90))
                    print(f"Dataset temperature threshold (90p) from {colname}: {thresholds['temperature']}")
            except Exception:
                pass

        # Pressure column detection
        pres_idx = None
        for i, c in enumerate(lower_cols):
            if 'press' in c or 'pressure' in c:
                pres_idx = i
                break
        if pres_idx is not None:
            colname = cols[pres_idx]
            try:
                vals = pd.to_numeric(df[colname], errors='coerce').dropna()
                if len(vals) > 0:
                    thresholds['pressure'] = float(np.percentile(vals, 90))
                    print(f"Dataset pressure threshold (90p) from {colname}: {thresholds['pressure']}")
            except Exception:
                pass

        if thresholds:
            try:
                set_thresholds(thresholds)
                print(f"Applied thresholds: {thresholds}")
            except Exception:
                pass
    except Exception:
        pass
    
    # Iterate via dicts for better performance than `iterrows()`
    for idx, row_dict in df.to_dict(orient="index").items():
        try:
            prediction = predict_single(row_dict, model, scaler, label_encoders, columns)
            
            # Track for statistics
            probabilities.append(prediction["failure_probability"])
            
            # Add row index to result
            prediction["row_index"] = idx
            results.append(prediction)
            # Temporary per-machine debug output (shows Id if present and explanations)
            try:
                id_val = row_dict.get("Id", row_dict.get("id", idx))
            except Exception:
                id_val = idx
            print(id_val, prediction.get("explanations"))
            
        except Exception as row_error:
            errors_count += 1
            results.append({
                "row_index": idx,
                "failure_probability": 0.0,
                "risk_score": 0,
                "maintenance_priority": "Low",
                "error": str(row_error)
            })
    
    # Calculate statistics
    stats = {}
    if probabilities:
        stats = {
            "min_probability": round(min(probabilities), 4),
            "max_probability": round(max(probabilities), 4),
            "mean_probability": round(sum(probabilities) / len(probabilities), 4),
            "std_probability": round(float(np.std(probabilities)), 4),
            "high_risk_count": sum(1 for r in results if r.get("maintenance_priority") == "High"),
            "medium_risk_count": sum(1 for r in results if r.get("maintenance_priority") == "Medium"),
            "low_risk_count": sum(1 for r in results if r.get("maintenance_priority") == "Low")
        }
    
    if verbose and probabilities:
        print("\n" + "="*60)
        print("BATCH PREDICTION STATISTICS")
        print("="*60)
        print(f"Min probability:        {stats['min_probability']:.4f}")
        print(f"Max probability:        {stats['max_probability']:.4f}")
        print(f"Mean probability:       {stats['mean_probability']:.4f}")
        print(f"Std Dev probability:    {stats['std_probability']:.4f}")
        print("-"*60)
        print(f"High Risk (≥65%):       {stats['high_risk_count']} machines")
        print(f"Medium Risk (40-65%):   {stats['medium_risk_count']} machines")
        print(f"Low Risk (<40%):        {stats['low_risk_count']} machines")
        print("="*60 + "\n")
    
    return {
        "total_rows": len(df),
        "processed_rows": len(results) - errors_count,
        "errors": errors_count,
        "results": results,
        "statistics": stats
    }
