"""
Action Engine: Rule-Based Maintenance Recommendations
======================================================

Translates ML predictions + feature contributions into 
domain-specific maintenance actions.

Design principle: Real factories combine ML + domain expertise.
This layer bridges the gap.
"""


def generate_actions(priority, top_risk_factors):
    """
    Generate maintenance recommendations based on priority and risk factors.
    
    Args:
        priority: str ('High', 'Medium', or 'Low')
        top_risk_factors: list of dicts with 'feature' and 'impact_score'
    
    Returns:
        list of actionable maintenance steps
    
    Example:
        >>> actions = generate_actions('High', [
        ...     {'feature': 'vibration', 'impact_score': 0.45},
        ...     {'feature': 'temperature', 'impact_score': 0.38}
        ... ])
        >>> # ['Schedule immediate inspection', 'Inspect bearings...', ...]
    """
    
    actions = set()
    
    # Priority-based actions
    if priority == "High":
        actions.add("🚨 Schedule immediate inspection (within 24 hours)")
        actions.add("Reduce machine operating load to 70%")
        actions.add("Prepare spare parts for critical components")
        actions.add("Assign maintenance team on standby")
    
    elif priority == "Medium":
        actions.add("Schedule inspection during next maintenance window (1-7 days)")
        actions.add("Increase monitoring frequency")
        actions.add("Verify spare parts availability")
    
    else:  # Low
        actions.add("Continue routine monitoring")
        actions.add("Include in standard preventive maintenance schedule")
    
    # Feature-specific actions (rule-based domain knowledge)
    if top_risk_factors:
        for factor in top_risk_factors:
            feature_name = factor.get("feature", "").lower()
            impact = factor.get("impact_score", 0)
            
            # Vibration-related issues
            if "vibration" in feature_name or "vib" in feature_name:
                actions.add("🔧 Inspect bearings and alignment")
                actions.add("Check for bearing wear or misalignment")
                if impact > 0.3:
                    actions.add("Consider bearing replacement")
            
            # Temperature-related issues
            if "temperature" in feature_name or "temp" in feature_name:
                actions.add("🌡️ Check cooling system performance")
                actions.add("Verify coolant levels and circulation")
                if impact > 0.3:
                    actions.add("Service cooling fan or radiator")
            
            # Pressure-related issues
            if "pressure" in feature_name or "press" in feature_name:
                actions.add("⚙️ Verify pressure regulation")
                actions.add("Check pressure relief valves")
                if impact > 0.3:
                    actions.add("Test and calibrate pressure sensors")
            
            # Speed/RPM issues
            if "speed" in feature_name or "rpm" in feature_name:
                actions.add("Check motor speed controller")
                actions.add("Inspect transmission and drive belt")
            
            # Load-related issues
            if "load" in feature_name:
                actions.add("Review load distribution")
                actions.add("Verify structural integrity")
                actions.add("Adjust operating parameters if safe")
    
    # Convert to sorted list (highest priority actions first)
    return sorted(list(actions))


def classify_maintenance_urgency(priority, failure_probability):
    """
    Determine maintenance urgency with additional context.
    
    Args:
        priority: str ('High', 'Medium', or 'Low')
        failure_probability: float (0-1)
    
    Returns:
        dict with urgency level and recommended timeline
    """
    
    if priority == "High":
        return {
            "urgency": "CRITICAL",
            "timeline": "Within 24 hours",
            "impact": "Imminent failure likely",
            "color": "red"
        }
    elif priority == "Medium":
        return {
            "urgency": "WARNING",
            "timeline": "Within 7 days",
            "impact": "Elevated risk; plan maintenance",
            "color": "yellow"
        }
    else:
        return {
            "urgency": "NOMINAL",
            "timeline": "Routine schedule",
            "impact": "Normal operation; continue monitoring",
            "color": "green"
        }


def estimate_delay_impact(priority, feature_contributions):
    """
    Estimate consequences of delaying maintenance.
    
    Args:
        priority: str ('High', 'Medium', or 'Low')
        feature_contributions: list of top risk factors
    
    Returns:
        str describing potential consequences
    """
    
    if priority == "High":
        return (
            "⏱️ CRITICAL: Delaying maintenance increases risk of:\n"
            "  • Catastrophic equipment failure (100-200% cost increase)\n"
            "  • Production downtime (cost: $K+ per hour)\n"
            "  • Safety hazards to operators\n"
            "  • Cascading damage to adjacent systems\n"
            "  → Take action immediately"
        )
    elif priority == "Medium":
        return (
            "⏱️ WARNING: Delaying maintenance may lead to:\n"
            "  • Escalation to high-risk state\n"
            "  • Increased repair costs (30-50%)\n"
            "  • Potential unplanned downtime\n"
            "  → Schedule inspection within 7 days"
        )
    else:
        return (
            "⏱️ NORMAL: Continue routine monitoring and maintenance.\n"
            "  • Early warning systems active\n"
            "  • Predictive alerts will trigger if risk increases"
        )


# New action catalog + helper to map failure modes -> detailed actions
ACTION_CATALOG = {
    "Bearing Wear": {
        "recommended_action": "Inspect bearings and schedule replacement",
        "urgency": "Within 72 hours",
        "estimated_downtime_min": 45,
        "estimated_cost_inr": "8000-12000",
        "impact_if_ignored": "Shaft damage and sudden breakdown"
    },
    "Overheating": {
        "recommended_action": "Check cooling system and airflow",
        "urgency": "Within 24 hours",
        "estimated_downtime_min": 20,
        "estimated_cost_inr": "2000-3000",
        "impact_if_ignored": "Thermal shutdown or motor burn"
    },
    "Hydraulic Leak": {
        "recommended_action": "Inspect seals and hoses",
        "urgency": "Immediate",
        "estimated_downtime_min": 30,
        "estimated_cost_inr": "5000-7000",
        "impact_if_ignored": "Loss of pressure and equipment damage"
    }
}


def generate_actions_from_failure_modes(failure_modes):
    """
    Convert a list of failure mode dicts into actionable maintenance tasks.

    Args:
        failure_modes: list of dicts with at least 'mode' and 'confidence'

    Returns:
        list of action dicts (catalog entry merged with failure mode info)
    """
    actions = []
    if not failure_modes:
        return actions

    for fm in failure_modes:
        mode = fm.get("mode")
        if not mode:
            continue

        catalog = ACTION_CATALOG.get(mode)
        if catalog:
            action = catalog.copy()
            action["failure_mode"] = mode
            action["confidence"] = fm.get("confidence")
            actions.append(action)

    return actions
