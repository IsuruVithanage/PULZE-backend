from fastapi import Header, HTTPException
from typing import Optional

async def get_user_id_from_gateway(x_user_id: Optional[int] = Header(None)) -> int:
    """
    A simple dependency to extract the user ID from a trusted 'X-User-ID' header
    that is expected to be added by the API Gateway after authentication.
    """
    if x_user_id is None:
        # This error is hit if the API Gateway fails to add the required header.
        raise HTTPException(
            status_code=401,
            detail="X-User-ID header is missing from gateway request."
        )
    return x_user_id