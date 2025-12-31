from services.genai_recommender import generate_maintenance_recommendations
import numpy as np
from ml.shap_explainer import (
    load_xgboost_model,
    create_shap_explainer,
    get_shap_values
)
from ml.root_cause import extract_root_causes

def compute_failure_risk(raw_prediction, min_val=0.0, max_val=1.0):
    """
    Convert raw model output into normalized risk score (0–100)
    using min-max scaling from validation distribution.
    """
    # Safety clamp
    raw_prediction = max(min(raw_prediction, max_val), min_val)

    risk_score = (raw_prediction - min_val) / (max_val - min_val + 1e-6)
    risk_score = risk_score * 100

    return round(float(risk_score), 2)


def compute_downtime_probability(risk_score):
    """
    Map risk score to probability of downtime
    """
    if risk_score >= 80:
        return 0.75
    elif risk_score >= 60:
        return 0.55
    elif risk_score >= 40:
        return 0.30
    else:
        return 0.10


def estimate_rul_days(risk_score):
    """
    Remaining Useful Life estimation (simple but explainable)
    """
    if risk_score >= 80:
        return 5
    elif risk_score >= 60:
        return 12
    elif risk_score >= 40:
        return 25
    else:
        return 60


def maintenance_priority(risk_score):
    if risk_score >= 80:
        return "Critical"
    elif risk_score >= 60:
        return "High"
    elif risk_score >= 40:
        return "Medium"
    else:
        return "Low"

def predict_failure_with_explainability(input_df):
    """
    Full failure prediction pipeline with SHAP-based root cause analysis
    """

    # 1️⃣ Load trained model
    model = load_xgboost_model()

    # 2️⃣ Predict failure probability
    failure_proba = model.predict_proba(input_df)[0][1]

    # 3️⃣ Convert to risk score (0–100)
    risk_score = compute_failure_risk(failure_proba)

    # 4️⃣ Business-level mappings
    downtime_prob = compute_downtime_probability(risk_score)
    rul_days = estimate_rul_days(risk_score)
    priority = maintenance_priority(risk_score)

    response = {
        "failure_probability": round(float(failure_proba), 3),
        "risk_score": risk_score,
        "downtime_probability": downtime_prob,
        "estimated_rul_days": rul_days,
        "maintenance_priority": priority
    }

    # 5️⃣ Run SHAP ONLY for risky cases (performance-safe)
    if risk_score >= 60:
        explainer = create_shap_explainer(model)
        shap_values = get_shap_values(explainer, input_df)

        root_causes = extract_root_causes(
            shap_values=shap_values,
            feature_names=input_df.columns.tolist(),
            top_k=3
        )


    recommendations = generate_maintenance_recommendations(root_causes)

    response["root_causes"] = root_causes
    response["maintenance_recommendations"] = recommendations
