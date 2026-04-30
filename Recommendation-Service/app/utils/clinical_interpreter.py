from typing import Dict, Any, Optional, List


def _interpret_metrics(metrics: Dict[str, Any]) -> Dict[str, str]:
    """
    Translates raw health metric values into qualitative clinical interpretations.
    """
    interpretations = {}
    # (This function contains the business logic for clinical ranges)
    # BMI Interpretation
    if bmi := metrics.get("bmi"):
        if bmi >= 30:
            interpretations['bmi_status'] = "Obese"
        elif bmi >= 25:
            interpretations['bmi_status'] = "Overweight"
        elif bmi >= 23:
            interpretations['bmi_status'] = "At Risk"
        elif bmi >= 18.5:
            interpretations['bmi_status'] = "Normal"
        else:
            interpretations['bmi_status'] = "Underweight"
    # LDL Cholesterol Interpretation
    if ldl := metrics.get("ldl"):
        if ldl >= 190:
            interpretations['ldl_status'] = "Very High"
        elif ldl >= 160:
            interpretations['ldl_status'] = "High"
        elif ldl >= 130:
            interpretations['ldl_status'] = "Borderline High"
        elif ldl >= 100:
            interpretations['ldl_status'] = "Near Optimal"
        else:
            interpretations['ldl_status'] = "Optimal"
    # HDL Cholesterol Interpretation
    if hdl := metrics.get("hdl"):
        if hdl < 35:
            interpretations['hdl_status'] = "Critically Low"
        elif hdl < 40:
            interpretations['hdl_status'] = "Low (Risk Factor)"
        elif hdl >= 60:
            interpretations['hdl_status'] = "High (Protective)"
        else:
            interpretations['hdl_status'] = "Normal"
    # Triglycerides Interpretation
    if tg := metrics.get("triglycerides"):
        if tg >= 500:
            interpretations['tg_status'] = "Very High"
        elif tg >= 200:
            interpretations['tg_status'] = "High"
        elif tg >= 150:
            interpretations['tg_status'] = "Borderline High"
        else:
            interpretations['tg_status'] = "Normal"
    # Fasting Blood Sugar Interpretation
    if fbs := metrics.get("fasting_blood_sugar"):
        if fbs >= 126:
            interpretations['fbs_status'] = "in the Diabetic Range"
        elif fbs >= 100:
            interpretations['fbs_status'] = "in the Pre-diabetic Range"
        else:
            interpretations['fbs_status'] = "Normal"
    return interpretations


def format_metrics_to_question(
        metrics: Dict[str, Any],
        reported_conditions: Optional[List[str]] = None,
        reported_habits: Optional[List[str]] = None,
        additional_info: str = None
) -> str:
    """
    Formats all user health data into a rich, natural language query for the LLM.
    This is a key prompt engineering step.
    """
    interpretations = _interpret_metrics(metrics)
    # Build a descriptive paragraph from the user's data.
    query_parts = [f"The user is a {metrics.get('age')}-year-old {metrics.get('gender', 'person').lower()}."]
    if bmi_status := interpretations.get('bmi_status'):
        query_parts.append(f"Their BMI is {metrics.get('bmi')} kg/m², which is in the '{bmi_status}' category.")
    lab_results = []
    if ldl_status := interpretations.get('ldl_status'):
        lab_results.append(f"LDL cholesterol is {ldl_status} at {metrics.get('ldl')} mg/dL")
    if hdl_status := interpretations.get('hdl_status'):
        lab_results.append(f"HDL cholesterol is {hdl_status} at {metrics.get('hdl')} mg/dL")
    if tg_status := interpretations.get('tg_status'):
        lab_results.append(f"triglycerides are {tg_status} at {metrics.get('triglycerides')} mg/dL")
    if fbs_status := interpretations.get('fbs_status'):
        lab_results.append(f"fasting blood sugar is {fbs_status} at {metrics.get('fasting_blood_sugar')} mg/dL")
    if lab_results:
        query_parts.append(f"Key lab results indicate: {', '.join(lab_results)}.")
    if reported_conditions:
        conditions_str = ', '.join(reported_conditions)
        query_parts.append(f"The user also reports the following health conditions: {conditions_str}.")
    if reported_habits:
        habits_str = ', '.join(reported_habits)
        query_parts.append(f"Regarding lifestyle, the user mentions: {habits_str}.")
    if additional_info:
        query_parts.append(f"Additional user information: {additional_info}")
    return " ".join(query_parts)