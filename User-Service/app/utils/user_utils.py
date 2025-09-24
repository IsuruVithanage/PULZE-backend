"""
Contains utility functions and dependencies for the user service.
"""

from fastapi import Header, HTTPException, status


def get_user_id_from_header(x_user_id: str | None = Header(None)) -> int:
    """
    FastAPI dependency to get the user ID from the X-User-ID header.

    This dependency is designed for a microservice architecture where an
    API Gateway validates the JWT and then passes the authenticated user's ID
    to this service in a secure header.

    Args:
        x_user_id (str | None): The value of the X-User-ID header.

    Raises:
        HTTPException: 400 if the header is missing or has an invalid format.

    Returns:
        int: The validated user ID.
    """
    if x_user_id is None:
        # This error should only be triggered if the service is accessed directly,
        # bypassing the API Gateway.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-User-ID header is missing. Access via API Gateway."
        )
    try:
        # Convert the header value to an integer
        return int(x_user_id)
    except ValueError:
        # Handle cases where the header value is not a valid integer
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid User ID format in X-User-ID header."
        )