from services.maintenance_knowledge import MAINTENANCE_KB

def generate_maintenance_recommendations(root_causes):
    recommendations = []

    for rc in root_causes:
        feature = rc["feature"]

        if feature in MAINTENANCE_KB:
            kb_entry = MAINTENANCE_KB[feature]

            recommendations.append({
                "root_cause": feature,
                "identified_issue": kb_entry["issue"],
                "recommended_actions": kb_entry["actions"]
            })
        else:
            recommendations.append({
                "root_cause": feature,
                "identified_issue": "Unknown issue",
                "recommended_actions": [
                    "Perform manual inspection",
                    "Consult maintenance engineer"
                ]
            })

    return recommendations
