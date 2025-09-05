from datetime import datetime
from typing import Dict, List, Any


def get_metric_status(metric_name: str, value: float) -> str:
    """
    Determines the clinical status of a health metric based on its value.
    Used for creating the deterministic table in the final report.
    """
    if value is None:
        return "N/A"

    ranges = {
        "Total Cholesterol": {"normal": 199, "borderline": 239},
        "LDL Cholesterol": {"normal": 99, "borderline": 159},
        "HDL Cholesterol": {"low": 39},
        "Fasting Blood Sugar": {"normal": 99, "prediabetes": 125},
    }

    if metric_name in ranges:
        if "low" in ranges[metric_name]:
            if value <= ranges[metric_name]["low"]: return "Low"
            return "Normal"
        else:
            if value <= ranges[metric_name]["normal"]: return "Normal"
            if value <= ranges[metric_name]["borderline"]: return "Borderline High"
            return "High"
    return "N/A"


def get_trend_indicator(series: List[Dict]) -> str:
    """Determines a simple trend from a series of data points."""
    if len(series) < 2:
        return "Stable"

    first_value = series[0].get('value')
    last_value = series[-1].get('value')

    if first_value is None or last_value is None:
        return "N/A"

    if last_value > first_value * 1.05:  # More than 5% increase
        return "📈 Worsening"
    if last_value < first_value * 0.95:  # More than 5% decrease
        return "📉 Improving"
    return "Stable"


def format_report_as_markdown(user_data: Dict, historical_data: Dict, ai_summary: str) -> str:
    """Assembles the final Markdown report string."""
    generation_date = datetime.now().strftime("%B %d, %Y")

    table_rows = ""
    for metric in historical_data.get('metrics', []):
        series = metric.get('series', [])
        latest_value_obj = series[-1] if series else {}
        previous_value_obj = series[-2] if len(series) > 1 else {}

        latest_value = latest_value_obj.get('value', 'N/A')
        previous_value = previous_value_obj.get('value', 'N/A')

        status = get_metric_status(metric['name'], latest_value) if isinstance(latest_value, (int, float)) else "N/A"
        trend = get_trend_indicator(series)

        unit = metric.get('unit', '')
        latest_str = f"{latest_value} {unit}" if isinstance(latest_value, (int, float)) else "N/A"
        prev_str = f"{previous_value} {unit}" if isinstance(previous_value, (int, float)) else "N/A"

        table_rows += f"| **{metric['name']}** | {latest_str} | {prev_str} | {trend} | {status} |\n"

    bmi_value = user_data.get('bmi', 'N/A')
    bmi_status = get_metric_status('BMI', bmi_value) if isinstance(bmi_value, (int, float)) else ""

    markdown_report = f"""
# Health Summary for {user_data.get('name', 'User')}

**Generated:** {generation_date}

## Patient Information
- **Age:** {user_data.get('age', 'N/A')}
- **Gender:** {user_data.get('gender', 'N/A')}
- **Weight:** {user_data.get('weight_kg', 'N/A')} kg
- **Height:** {user_data.get('height_cm', 'N/A')} cm
- **BMI:** {bmi_value} kg/m² ({bmi_status})

---

## AI-Generated Clinical Summary
{ai_summary}

---

## Key Metric Trends (Last 6 Months)

| Metric                | Latest Value | Previous Value | Trend       | Status    |
| --------------------- | ------------ | -------------- | ----------- | --------- |
{table_rows}
---

**Disclaimer:** This report is a summary of data from the Pulze app and is for informational purposes only. It is not a substitute for professional medical advice.
"""
    return markdown_report.strip()
