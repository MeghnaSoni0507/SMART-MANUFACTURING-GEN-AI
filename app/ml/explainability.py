"""
Explainability Module: Feature Contribution Analysis
=====================================================

Provides interpretability for PyTorch-based failure predictions
using weight × input interaction scoring.

This approach is:
- Model-agnostic compatible (can extend to SHAP)
- Computationally efficient
- Production-ready
- Interview-defensible (mathematical foundation)
"""

import numpy as np
import torch


def get_top_contributing_features(model, input_tensor, columns, top_k=3):
    """
    Extract top-k contributing features using weight × input interaction.
    
    Args:
        model: PyTorch neural network with accessible first layer
        input_tensor: torch.Tensor of shape (1, num_features)
        columns: list of feature names
        top_k: number of top features to return
    
    Returns:
        List of dicts with 'feature' and 'impact_score' keys
    
    Example:
        >>> factors = get_top_contributing_features(model, x, columns, top_k=3)
        >>> # [{'feature': 'vibration', 'impact_score': 0.4521}, ...]
    """
    try:
        # Extract weights from first layer of model
        # model.net[0] is the first Linear layer
        weights = model.net[0].weight.detach().cpu().numpy()[0]
        
        # Convert input tensor to numpy
        inputs = input_tensor.detach().cpu().numpy()[0]
        
        # Compute interaction: absolute value of (weight * input)
        contributions = np.abs(weights * inputs)
        
        # Get top-k indices (sorted descending)
        top_indices = contributions.argsort()[-top_k:][::-1]
        
        # Build result list
        result = [
            {
                "feature": columns[i],
                "impact_score": round(float(contributions[i]), 4)
            }
            for i in top_indices
        ]
        
        return result
    
    except Exception as e:
        print(f"⚠️ Error in get_top_contributing_features: {e}")
        return []


def analyze_risk_distribution(probabilities):
    """
    Analyze probability distribution for model calibration.
    
    Args:
        probabilities: list of failure probabilities [0, 1]
    
    Returns:
        dict with min, max, mean, std
    """
    if not probabilities:
        return {}
    
    probs = np.array(probabilities)
    return {
        "min": round(float(probs.min()), 4),
        "max": round(float(probs.max()), 4),
        "mean": round(float(probs.mean()), 4),
        "std": round(float(probs.std()), 4),
        "count": len(probabilities)
    }


def percentile_risk_classification(failure_probability, percentiles=None):
    """
    Percentile-based risk classification (optional for advanced use).
    
    Instead of fixed thresholds, use data-driven percentiles.
    
    Args:
        failure_probability: single probability [0, 1]
        percentiles: dict with 'p75' and 'p40' keys (e.g., from training data)
    
    Returns:
        risk level string ('High', 'Medium', 'Low')
    """
    if percentiles is None:
        # Fallback to fixed thresholds
        if failure_probability >= 0.65:
            return "High"
        elif failure_probability >= 0.40:
            return "Medium"
        else:
            return "Low"
    
    # Percentile-based
    if failure_probability >= percentiles.get('p75', 0.65):
        return "High"
    elif failure_probability >= percentiles.get('p40', 0.40):
        return "Medium"
    else:
        return "Low"


def explain_machine(original_row, top_risk_factors=None, columns=None):
    """
    Generate human-readable explanations based on feature thresholds.
    
    Args:
        original_row: dict of raw feature values (before scaling/encoding)
        top_risk_factors: list from get_top_contributing_features()
        columns: list of feature names for context
    
    Returns:
        list of human-readable reason strings
    
    Example:
        >>> reasons = explain_machine(row, top_factors, columns)
        >>> # ['High vibration detected → possible bearing issue', ...]
    """
    # New simplified explain_machine implementation: only uses raw row features
    reasons = []

    if not isinstance(original_row, dict):
        try:
            original_row = dict(original_row)
        except Exception:
            return ["Unable to generate explanations"]

    # Map physical meanings to real dataset columns. Add known sensor names here.
    sensor_map = {
        "vibration": ["vibration", "L0_S0_F0", "sensor_12", "vib_mean"],
        "temperature": ["temperature", "temp_mean", "L1_S2_F3"],
        "pressure": ["pressure", "press_mean", "L2_S1_F4"]
    }

    def _find_value(mapping_candidates, rowdict):
        # Try exact matches first, then case-insensitive match
        for cand in mapping_candidates:
            if cand in rowdict:
                return rowdict.get(cand), cand
        # case-insensitive search
        keys = list(rowdict.keys())
        lower_map = {k.lower(): k for k in keys}
        for cand in mapping_candidates:
            if cand.lower() in lower_map:
                matched = lower_map[cand.lower()]
                return rowdict.get(matched), matched
        # If no direct match, try to auto-detect likely columns from provided `columns` list
        if columns:
            try:
                lower_cols = [c.lower() for c in columns]
                # Heuristic: if mapping_candidates includes 'vibration' look for 'vib' in column names
                if any('vibration' in str(m).lower() for m in mapping_candidates):
                    for i, col in enumerate(lower_cols):
                        if 'vib' in col or 'vibration' in col:
                            print(f"Auto-mapping vibration -> {columns[i]}")
                            return rowdict.get(columns[i]), columns[i]
                if any('temperature' in str(m).lower() or 'temp' in str(m).lower() for m in mapping_candidates):
                    for i, col in enumerate(lower_cols):
                        if 'temp' in col or 'temperature' in col:
                            print(f"Auto-mapping temperature -> {columns[i]}")
                            return rowdict.get(columns[i]), columns[i]
                if any('pressure' in str(m).lower() or 'press' in str(m).lower() for m in mapping_candidates):
                    for i, col in enumerate(lower_cols):
                        if 'press' in col or 'pressure' in col or 'pres' in col:
                            print(f"Auto-mapping pressure -> {columns[i]}")
                            return rowdict.get(columns[i]), columns[i]
            except Exception:
                pass

        return 0, None

    try:
        vib_val, vib_key = _find_value(sensor_map["vibration"], original_row)
        vibration = float(vib_val)
    except Exception:
        vibration = 0.0
        vib_key = None

    try:
        temp_val, temp_key = _find_value(sensor_map["temperature"], original_row)
        temperature = float(temp_val)
    except Exception:
        temperature = 0.0
        temp_key = None

    try:
        pres_val, pres_key = _find_value(sensor_map["pressure"], original_row)
        pressure = float(pres_val)
    except Exception:
        pressure = 0.0
        pres_key = None

    # Debug: show which raw columns were mapped (useful to verify mapping)
    try:
        row_id = original_row.get("Id", original_row.get("id", None))
        mapping = {
            "vibration": {"column": vib_key, "value": vibration},
            "temperature": {"column": temp_key, "value": temperature},
            "pressure": {"column": pres_key, "value": pressure},
        }
        print(f"explain_machine mapping for Id={row_id}: vibration->{vib_key}={vibration}, temperature->{temp_key}={temperature}, pressure->{pres_key}={pressure}")
    except Exception:
        mapping = {"vibration": {"column": None, "value": 0}, "temperature": {"column": None, "value": 0}, "pressure": {"column": None, "value": 0}}

    # Use configurable thresholds (dataset-driven if available)
    vib_thr = THRESHOLDS.get("vibration", 0.7)
    temp_thr = THRESHOLDS.get("temperature", 80)
    pres_thr = THRESHOLDS.get("pressure", 120)

    if vibration and vibration > vib_thr:
        reasons.append(f"High vibration → possible bearing wear (value={vibration})")

    if temperature and temperature > temp_thr:
        reasons.append(f"High temperature → overheating risk (value={temperature})")

    if pressure and pressure > pres_thr:
        reasons.append(f"Abnormal pressure → valve/blockage issue (value={pressure})")

    # If raw-sensor rules didn't produce results, try top contributing features
    if not reasons and top_risk_factors:
        for factor in top_risk_factors:
            feature_name = factor.get("feature", "").lower()
            impact_score = float(factor.get("impact_score", 0))

            if "vibration" in feature_name or "vib" in feature_name:
                if impact_score > 0.3:
                    reasons.append(f"🔴 High vibration impact ({impact_score:.2f}) → possible bearing wear")
                elif impact_score > 0.15:
                    reasons.append(f"🟡 Moderate vibration impact ({impact_score:.2f}) → monitor bearings")

            if "temperature" in feature_name or "temp" in feature_name:
                if impact_score > 0.3:
                    reasons.append(f"🔴 High temperature impact ({impact_score:.2f}) → overheating risk")
                elif impact_score > 0.15:
                    reasons.append(f"🟡 Moderate temperature impact ({impact_score:.2f}) → check cooling")

            if "pressure" in feature_name or "press" in feature_name:
                if impact_score > 0.3:
                    reasons.append(f"🔴 Pressure impact ({impact_score:.2f}) → valve/blockage issue")
                elif impact_score > 0.15:
                    reasons.append(f"🟡 Pressure deviation ({impact_score:.2f}) → monitor pressure")

            if "speed" in feature_name or "rpm" in feature_name:
                if impact_score > 0.3:
                    reasons.append(f"🔴 Speed anomaly ({impact_score:.2f}) → transmission/motor issue")
                elif impact_score > 0.15:
                    reasons.append(f"🟡 Speed variation ({impact_score:.2f}) → monitor motor")

            if "load" in feature_name:
                if impact_score > 0.3:
                    reasons.append(f"🔴 Excessive load ({impact_score:.2f}) → reduce load")
                elif impact_score > 0.15:
                    reasons.append(f"🟡 Load imbalance ({impact_score:.2f}) → verify distribution")

    if not reasons:
        reasons.append("No dominant anomaly detected; risk may be cumulative or latent")

    return reasons, mapping


def get_feature_statistics(features_data, columns):
    """
    Compute min/max/mean statistics for numerical features.
    Useful for understanding feature distributions.
    
    Args:
        features_data: list of feature arrays
        columns: list of column names
    
    Returns:
        dict with statistics per feature
    """
    try:
        features_array = np.array(features_data)
        stats = {}
        
        for i, col in enumerate(columns):
            col_data = features_array[:, i]
            stats[col] = {
                "min": round(float(col_data.min()), 4),
                "max": round(float(col_data.max()), 4),
                "mean": round(float(col_data.mean()), 4),
                "std": round(float(col_data.std()), 4)
            }
        
        return stats
    except Exception as e:
        print(f"⚠️ Error computing feature statistics: {e}")
        return {}


# Module-level thresholds (can be updated at runtime)
THRESHOLDS = {
    "vibration": 0.7,
    "temperature": 80,
    "pressure": 120
}


def set_thresholds(d: dict):
    """Update thresholds used by explain_machine. Only keys present in d are updated."""
    for k, v in (d or {}).items():
        if k in THRESHOLDS:
            try:
                THRESHOLDS[k] = float(v)
            except Exception:
                pass

