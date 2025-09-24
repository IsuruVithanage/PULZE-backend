"""
Defines the Pydantic models used for data validation and serialization.

These schemas ensure that incoming request bodies have the correct structure
and data types, and also define the structure of response bodies.
"""

from pydantic import BaseModel

class UserLogin(BaseModel):
    """Defines the expected request body for the user login endpoint."""
    email: str
    password: str

class Token(BaseModel):
    """Defines the response body for a successful authentication request."""
    access_token: str
    token_type: str