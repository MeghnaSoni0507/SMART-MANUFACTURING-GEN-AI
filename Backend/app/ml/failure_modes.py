"""
Simple rule-based failure mode inference.

This module maps engineering signals (vibration, temperature, pressure)
to human-readable failure mode hypotheses with confidence and reasons.
"""

def infer_failure_modes(features: dict):
    """
    Infer mechanical failure modes based on sensor signals.

    Args:
        features: dict with engineering signals (vibration, temperature, pressure, ...)

    Returns:
        List of dicts: {mode, confidence, reason}
    """
    failure_modes = []

    vibration = float(features.get("vibration", 0) or 0)
    temperature = float(features.get("temperature", 0) or 0)
    pressure = float(features.get("pressure", 0) or 0)

    # Bearing Wear
    if vibration > 1.5:
        failure_modes.append({
            "mode": "Bearing Wear",
            "confidence": "High",
            "reason": "Vibration above safe threshold"
        })

    # Overheating
    if temperature > 80:
        failure_modes.append({
            "mode": "Overheating",
            "confidence": "Medium",
            "reason": "Temperature trending above nominal range"
        })

    # Hydraulic Leak / Low pressure
    if pressure < 20:
        failure_modes.append({
            "mode": "Hydraulic Leak",
            "confidence": "High",
            "reason": "Pressure below minimum operating level"
        })

    return failure_modes
