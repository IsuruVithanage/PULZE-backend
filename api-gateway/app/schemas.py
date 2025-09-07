from pydantic import BaseModel

class UserLogin(BaseModel):
    """The request body for the login endpoint."""
    email: str
    password: str

class Token(BaseModel):
    """The response body for a successful login."""
    access_token: str
    token_type: str
