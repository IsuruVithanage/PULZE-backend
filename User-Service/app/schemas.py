from pydantic import BaseModel, EmailStr
from typing import Optional

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


# --- UPDATED: UserResponse now includes all profile fields ---
class UserResponse(BaseModel):
    id: int
    email: EmailStr
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None

    class Config:
        from_attributes = True


class UserProfileData(BaseModel):
    id: int
    weight_kg: Optional[float]
    height_cm: Optional[float]
    gender: Optional[str]
    age: Optional[int]


class Token(BaseModel):
    access_token: str
    token_type: str