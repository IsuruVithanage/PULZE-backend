"""
Contains functions to parse raw text from medical reports and extract
structured data using regular expressions (regex).
"""

import re
from typing import Dict, Optional


def _find_metric(pattern: str, text: str) -> Optional[float]:
    """
    Searches for a regex pattern in text and returns the first captured group
    as a float. Handles complex patterns with multiple capture groups.
    """
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        # Find the first group that is not None
        value_str = next((g for g in match.groups() if g is not None), None)
        if value_str:
            try:
                return float(value_str)
            except ValueError:
                return None
    return None


def parse_lipid_report(text: str) -> Dict:
    """
    Parses raw text to find and extract lipid profile metrics.

    Args:
        text (str): The unstructured text from an OCR process.

    Raises:
        ValueError: If core metrics (like total cholesterol) cannot be found.

    Returns:
        Dict: A dictionary of the parsed metrics and their float values.
    """
    patterns = {
        "total_cholesterol": r"TOTAL CHOLESTEROL\s*[:\-]?\s*([\d.]+)",
        "hdl_cholesterol": r"HDL CHOLESTEROL\s*[:\-]?\s*([\d.]+)",
        "triglycerides": r"TRIGLYCERIDES\s*[:\-]?\s*([\d.]+)",
        "ldl_cholesterol": r"LDL CHOLESTEROL\s*[:\-]?\s*([\d.]+)",
        "vldl_cholesterol": r"VLDL CHOLESTEROL\s*[:\-]?\s*([\d.]+)",
        "non_hdl_cholesterol": r"NON\s*[-]?\s*HDL CHOLESTEROL\s*[:\-]?\s*([\d.]+)",
        "total_hdl_ratio": r"TOTAL CHOLESTEROL\s*/\s*HDL RATIO\s*[:\-]?\s*([\d.]+)",
        "triglycerides_hdl_ratio": r"TRIGLYCERIDES\s*/\s*HDL RATIO\s*[:\-]?\s*([\d.]+)"
    }
    data = {}
    for key, pattern in patterns.items():
        data[key] = _find_metric(pattern, text)

    # Validate that essential fields were found
    if data.get("total_cholesterol") is None or data.get("ldl_cholesterol") is None:
        raise ValueError("Could not parse core lipid profile metrics from the report.")

    return data


def parse_blood_sugar_report(text: str) -> Dict:
    """
    Parses raw text to find and extract blood sugar metrics.

    Args:
        text (str): The unstructured text from an OCR process.

    Raises:
        ValueError: If no blood sugar metrics can be found in the text.

    Returns:
        Dict: A dictionary of the parsed metrics and their float values.
    """
    patterns = {
        # This pattern looks for "Fasting" with "Plasma Glucose Venous" OR "Fasting Blood Sugar"
        "fasting_blood_sugar": r"PLASMA GLUCOSE VENOUS\s*-?\s*FASTING\s*([\d.]+)|FASTING BLOOD SUGAR\s+([\d.]+)",
        "random_blood_sugar": r"RANDOM BLOOD SUGAR\s*[:\-]?\s*([\d.]+)",
        "hba1c": r"HbA1c\s*[:\-]?\s*([\d.]+)"
    }

    data = {}
    for key, pattern in patterns.items():
        data[key] = _find_metric(pattern, text)

    # Validate that at least one metric was found
    if not any(value is not None for value in data.values()):
        raise ValueError("Could not parse any blood sugar metrics from the report.")

    return data