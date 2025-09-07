from fastapi import Header, HTTPException, status

def get_user_id_from_header(x_user_id: str | None = Header(None)) -> int:
    """
    Dependency to get the user ID from the X-User-ID header, which is
    securely set by the API Gateway after JWT validation.
    """
    if x_user_id is None:
        # This error should ideally never be hit if the service is only accessed
        # through the correctly configured gateway.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-User-ID header is missing. Access via API Gateway."
        )
    try:
        return int(x_user_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid User ID format in X-User-ID header."
        )
