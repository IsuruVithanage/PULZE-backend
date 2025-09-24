"""
Defines Pydantic schemas for API data validation and serialization.

These schemas determine the shape of the data for API requests and responses,
ensuring that data is valid and formatted correctly.
"""

from pydantic import BaseModel, EmailStr
from typing import Optional, List


class UserCreate(BaseModel):
    """Schema for validating new user registration data."""
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    """Schema for validating user login credentials."""
    email: EmailStr
    password: str


class UserProfileUpdate(BaseModel):
    """Schema for validating incoming user profile update data."""
    name: str
    age: int
    gender: str
    weight_kg: float
    height_cm: float


class AdditionalInfoUpdate(BaseModel):
    """Schema for updating a user's additional health information."""
    health_conditions: Optional[List[str]] = None
    lifestyle_habits: Optional[List[str]] = None


class AdditionalInfoResponse(BaseModel):
    """Schema for responding with the user's additional health information."""
    health_conditions: Optional[List[str]] = []
    lifestyle_habits: Optional[List[str]] = []

    class Config:
        from_attributes = True


class UserResponse(BaseModel):
    """Schema for formatting user data in API responses (excludes sensitive info)."""
    id: int
    email: str
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    health_conditions: Optional[List[str]] = []
    lifestyle_habits: Optional[List[str]] = []

    class Config:
        from_attributes = True


class UserProfileData(BaseModel):
    """Schema for user profile data, used for inter-service communication."""
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    health_conditions: Optional[List[str]] = []
    lifestyle_habits: Optional[List[str]] = []

    class Config:
        from_attributes = True


class Token(BaseModel):
    """Schema for the JWT access token response."""
    access_token: str
    token_type: str