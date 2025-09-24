"""
Handles authentication and authorization using JSON Web Tokens (JWT).

This module provides a dependency function (`get_current_user`) that can be
used in path operations to protect endpoints and identify the current user.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from dotenv import load_dotenv
import os

load_dotenv()

# --- Configuration ---
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"

# HTTPBearer is a security scheme that expects an "Authorization: Bearer <token>" header.
oauth2_scheme = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(oauth2_scheme)) -> str:
    """
    FastAPI dependency to secure endpoints and retrieve the current user's ID.

    It decodes the JWT token from the Authorization header, validates it, and
    extracts the user ID ('sub' claim).

    Args:
        credentials: The HTTP Authorization credentials automatically extracted by FastAPI.

    Raises:
        HTTPException(500): If the SECRET_KEY is not configured.
        HTTPException(401): If the token is invalid, malformed, or expired.

    Returns:
        str: The user ID extracted from the token's 'sub' claim.
    """
    if not SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT Secret Key is not configured."
        )

    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        return user_id
    except JWTError:
        raise credentials_exception