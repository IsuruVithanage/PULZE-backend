# app/utils/parser.py
import re
from typing import Dict


def parse_lipid_report(text: str) -> Dict:
    # ... (this function remains the same)
    patterns = {
        "total_cholesterol": r"TOTAL CHOLESTEROL\s+([\d.]+)",
        "hdl_cholesterol": r"HDL CHOLESTEROL\s+([\d.]+)",
        "triglycerides": r"TRIGLYCERIDES\s+([\d.]+)",
        "ldl_cholesterol": r"LDL CHOLESTEROL\s+([\d.]+)",
        "vldl_cholesterol": r"VLDL CHOLESTEROL\s+([\d.]+)",
        "non_hdl_cholesterol": r"NON HDL CHOLESTEROL\s+([\d.]+)",
        "total_hdl_ratio": r"TOTAL CHOLESTEROL/HDL RATIO\s+([\d.]+)",
        "triglycerides_hdl_ratio": r"TRIGLYCERIDES/HDL RATIO\s+([\d.]+)"
    }
    data = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        data[key] = float(match.group(1)) if match else None

    if data.get("total_cholesterol") is None or data.get("ldl_cholesterol") is None:
        raise ValueError("Could not parse core lipid profile metrics from the report.")

    return data


# --- UPDATE THIS FUNCTION ---
def parse_blood_sugar_report(text: str) -> Dict:
    # Add more variations to the patterns to match different lab reports
    patterns = {
        # This new pattern looks for "PLASMA GLUCOSE VENOUS" and "FASTING"
        # The `[\s\S]*?` part allows it to match across new lines
        "fasting_blood_sugar": r"PLASMA GLUCOSE VENOUS\s*-?\s*FASTING\s*([\d.]+)|\bFASTING BLOOD SUGAR\s+([\d.]+)",

        # Original patterns for other metrics
        "random_blood_sugar": r"RANDOM BLOOD SUGAR\s+([\d.]+)",
        "hba1c": r"HbA1c\s*[:\-]\s*([\d.]+)"
    }

    data = {}
    found_any = False

    for key, pattern in patterns.items():
        # The re.DOTALL flag allows '.' to match newline characters
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            # The new pattern has two capturing groups, so we check which one found the value
            # `match.groups()` returns a tuple of all captured groups, e.g., ('100', None) or (None, '98')
            # The next line filters out the `None` and gets the first valid number string.
            value = next((g for g in match.groups() if g is not None), None)
            if value:
                data[key] = float(value)
                found_any = True
            else:
                data[key] = None
        else:
            data[key] = None

    if not found_any:
        raise ValueError("Could not parse any blood sugar metrics from the report.")

    return data