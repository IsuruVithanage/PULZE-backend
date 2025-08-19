import re

def parse_report(text: str) -> dict:
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
        if match:
            data[key] = float(match.group(1))
        else:
            data[key] = 0.0
    return data
