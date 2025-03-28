from pydantic import BaseModel, Field, field_validator
from typing import Optional


class HealthMetrics(BaseModel):
    gender: str = Field(..., description="Gender of the patient (Male/Female)")
    age: int = Field(..., description="Age in years", ge=0, le=120)
    cholesterol: float = Field(..., description="Total cholesterol in mg/dL", ge=0)
    hdl: float = Field(..., description="HDL cholesterol in mg/dL", ge=0)
    ldl: float = Field(..., description="LDL cholesterol in mg/dL", ge=0)
    triglycerides: float = Field(..., description="Triglycerides in mg/dL", ge=0)
    bmi: float = Field(..., description="Body Mass Index", ge=0)

    @field_validator('gender')
    def gender_must_be_valid(cls, v):
        normalized = v.lower().strip()
        if normalized not in ['male', 'female', 'other']:
            raise ValueError('Gender must be Male, Female, or Other')
        return v


class RecommendationRequest(BaseModel):
    health_metrics: HealthMetrics
    additional_info: Optional[str] = Field(None, description="Any additional health information")


class RecommendationResponse(BaseModel):
    recommendation: str = Field(..., description="Diet recommendation based on health metrics")