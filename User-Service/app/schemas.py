from pydantic import BaseModel, EmailStr
from typing import Optional, List


# --- NEW: Schema for updating user profile data ---
class UserProfileUpdate(BaseModel):
    name: str
    age: int
    gender: str
    weight_kg: float
    height_cm: float # Assuming height is in cm now

# ... (keep your UserCreate and UserLogin schemas) ...
class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str


class AdditionalInfoUpdate(BaseModel):
    health_conditions: Optional[List[str]] = None
    lifestyle_habits: Optional[List[str]] = None


# --- [NEW] Schema for responding with the additional info ---
class AdditionalInfoResponse(BaseModel):
    health_conditions: Optional[List[str]] = []
    lifestyle_habits: Optional[List[str]] = []

    class Config:
        from_attributes = True

# --- UPDATED: UserResponse now includes all profile fields ---
class UserResponse(BaseModel):
    id: int
    email: str
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    # Add the new fields
    health_conditions: Optional[List[str]] = []
    lifestyle_habits: Optional[List[str]] = []

    class Config:
        from_attributes = True


class UserProfileData(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    # Add the new fields
    health_conditions: Optional[List[str]] = []
    lifestyle_habits: Optional[List[str]] = []

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str