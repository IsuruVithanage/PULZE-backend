"""
Pydantic models for API request bodies and endpoint inputs.

This module defines the data structures and validation rules for the data
that the API expects to receive from clients.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum

class DocumentCategory(str, Enum):
    """Defines the allowed categories for uploaded knowledge base documents."""
    DIETARY = "dietary"
    GENERAL = "general"
    RISK_ASSESSMENT = "risk_assessment"

class HealthMetrics(BaseModel):
    """
    Represents the core health metrics for a user.
    Used internally to structure data for the RAG service.
    """
    # Core User Info
    user_id: int = Field(..., description="The ID of the user this report belongs to.")
    gender: str = Field(..., description="Gender of the patient (Male/Female/Other)")
    age: int = Field(..., description="Age in years", ge=0, le=120)
    bmi: float = Field(..., description="Body Mass Index", ge=0)

    # Optional Lipid Profile Metrics
    cholesterol: Optional[float] = Field(None, description="Total cholesterol in mg/dL", ge=0)
    hdl: Optional[float] = Field(None, description="HDL cholesterol in mg/dL", ge=0)
    ldl: Optional[float] = Field(None, description="LDL cholesterol in mg/dL", ge=0)
    triglycerides: Optional[float] = Field(None, description="Triglycerides in mg/dL", ge=0)

    # Optional Blood Sugar Metrics
    fasting_blood_sugar: Optional[float] = Field(None, description="Fasting blood sugar in mg/dL", ge=0)
    hba1c: Optional[float] = Field(None, description="Glycated hemoglobin as a percentage (%)", ge=0)

    @field_validator('gender')
    def gender_must_be_valid(cls, v: str) -> str:
        """Validate and normalize the gender field."""
        normalized = v.lower().strip()
        if normalized not in ['male', 'female', 'other']:
            raise ValueError('Gender must be Male, Female, or Other')
        return v
