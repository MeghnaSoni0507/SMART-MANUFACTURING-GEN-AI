"""
Decision engine combines ML risk score with failure mode inference
to produce actionable decision intelligence.
"""
from .failure_modes import infer_failure_modes
from .action_engine import generate_actions_from_failure_modes
from .what_if_simulator import run_what_if_simulation
import json


def decision_intelligence(features: dict, risk_score: float, delay_days: int = 0):
    """
    Converts raw ML risk into operational intelligence.

    Args:
        features: dict of engineering signals
        risk_score: integer 0-100
        delay_days: optional, number of days to simulate maintenance delay

    Returns:
        dict with risk_score, risk_level, failure_modes, actions, and optional what-if simulation
    """
    failure_modes = infer_failure_modes(features)

    actions = generate_actions_from_failure_modes(failure_modes)

    # Prepare what-if simulation if delay_days specified
    simulation = None
    if delay_days > 0:
        simulation = run_what_if_simulation(risk_score, delay_days)

    decision = {
        "risk_score": risk_score,
        "risk_level": (
            "High" if risk_score > 70
            else "Medium" if risk_score > 40
            else "Low"
        ),
        "failure_modes": failure_modes,
        "recommended_actions": actions,
        "action_required": len(actions) > 0,
        "what_if_simulation": simulation
    }

    # Log decision for debugging/validation
    try:
        print(json.dumps(decision, indent=2))
    except Exception:
        pass

    return decision
