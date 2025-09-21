# app/models/request_models.py
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum

# --- ADD THIS ENUM ---
class DocumentCategory(str, Enum):
    DIETARY = "dietary"
    GENERAL = "general"
    RISK_ASSESSMENT = "risk_assessment"
# ---------------------


class HealthMetrics(BaseModel):
    user_id: int = Field(..., description="The ID of the user this report belongs to.")
    gender: str = Field(..., description="Gender of the patient (Male/Female)")
    age: int = Field(..., description="Age in years", ge=0, le=120)
    bmi: float = Field(..., description="Body Mass Index", ge=0)

    # Lipid Profile Metrics (Optional)
    cholesterol: Optional[float] = Field(None, description="Total cholesterol in mg/dL", ge=0)
    hdl: Optional[float] = Field(None, description="HDL cholesterol in mg/dL", ge=0)
    ldl: Optional[float] = Field(None, description="LDL cholesterol in mg/dL", ge=0)
    triglycerides: Optional[float] = Field(None, description="Triglycerides in mg/dL", ge=0)

    # --- NEW: Blood Sugar Metric (Optional) ---
    fasting_blood_sugar: Optional[float] = Field(None, description="Fasting blood sugar in mg/dL", ge=0)
    hba1c: Optional[float] = Field(None, description="Glycated hemoglobin as a percentage (%)", ge=0)

    @field_validator('gender')
    def gender_must_be_valid(cls, v):
        normalized = v.lower().strip()
        if normalized not in ['male', 'female', 'other']:
            raise ValueError('Gender must be Male, Female, or Other')
        return v

class RecommendationRequest(BaseModel):
    additional_info: Optional[str] = Field(None, description="Any additional health information")


class RecommendationResponse(BaseModel):
    # ... your existing RecommendationResponse class ...
    recommendation: str = Field(..., description="Diet recommendation based on health metrics")