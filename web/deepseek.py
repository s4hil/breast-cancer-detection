# deepseek.py

def analyze_results(diagnosis: str, confidence: float, accuracy: float):
    """
    Analyze the endpoint result to provide interpretive diagnostics.
    """

    # Basic confidence interpretation
    if confidence >= 90:
        confidence_comment = "High confidence in prediction — strong indicator."
    elif confidence >= 70:
        confidence_comment = "Moderate confidence — clinical review recommended."
    else:
        confidence_comment = "Low confidence — further testing advised."

    # Accuracy interpretation
    if accuracy >= 98:
        accuracy_comment = "Model is highly reliable."
    elif accuracy >= 95:
        accuracy_comment = "Model is reasonably accurate."
    else:
        accuracy_comment = "Model accuracy is moderate; results should be verified."

    # Suggestion logic
    if diagnosis == "Malignant":
        action = "Immediate biopsy and oncologist consultation advised."
        severity = "High"
        alert_color = "danger"
    else:
        action = "Regular monitoring and routine check-up suggested."
        severity = "Low"
        alert_color = "success"

    return {
        "severity": severity,
        "confidence_comment": confidence_comment,
        "accuracy_comment": accuracy_comment,
        "suggested_action": action,
        "alert_color": alert_color
    }
