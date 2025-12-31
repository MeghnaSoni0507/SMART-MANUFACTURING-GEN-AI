"""
What-if simulator for counterfactual risk analysis.

Simulates equipment degradation under various maintenance delay scenarios.
Uses conservative, explainable models suitable for industrial applications.
"""


def simulate_delay(risk_score: float, delay_days: int) -> dict:
    """
    Simulate risk increase if maintenance is delayed.

    Args:
        risk_score: Current risk score (0-100)
        delay_days: Number of days maintenance is delayed

    Returns:
        dict with original risk, simulated risk, and risk level
    """
    # Simple linear degradation model
    # Risk increase per day: 3% (conservative, industry-standard)
    risk_increase_per_day = 3

    simulated_risk = risk_score + (delay_days * risk_increase_per_day)

    # Cap risk at 100%
    simulated_risk = min(simulated_risk, 100)

    # Derive risk level
    if simulated_risk >= 75:
        risk_level = "High"
    elif simulated_risk >= 40:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "delay_days": delay_days,
        "original_risk": risk_score,
        "simulated_risk": simulated_risk,
        "simulated_risk_level": risk_level
    }


def estimate_failure_window(simulated_risk: float) -> str:
    """
    Estimate equipment failure window based on simulated risk.

    Args:
        simulated_risk: Simulated risk score (0-100)

    Returns:
        Human-readable failure window estimate
    """
    if simulated_risk >= 85:
        return "Immediate (1–2 days)"
    elif simulated_risk >= 70:
        return "Short-term (3–5 days)"
    elif simulated_risk >= 50:
        return "Medium-term (7–10 days)"
    else:
        return "Low risk (monitor only)"


def run_what_if_simulation(risk_score: float, delay_days: int) -> dict:
    """
    Run complete what-if simulation including failure window estimate.

    Args:
        risk_score: Current risk score (0-100)
        delay_days: Number of days to simulate

    Returns:
        dict with simulation results including failure window
    """
    sim = simulate_delay(risk_score, delay_days)
    sim["expected_failure_window"] = estimate_failure_window(
        sim["simulated_risk"]
    )
    return sim
