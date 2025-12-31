import numpy as np

def extract_root_causes(shap_values, feature_names, top_k=3):
    """
    Returns top K features responsible for failure
    """
    shap_row = shap_values[0]   # single prediction
    impact = list(zip(feature_names, shap_row))

    impact_sorted = sorted(
        impact,
        key=lambda x: abs(x[1]),
        reverse=True
    )

    root_causes = [
        {"feature": f, "impact": round(float(v), 4)}
        for f, v in impact_sorted[:top_k]
    ]

    return root_causes
