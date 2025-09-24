"""
Router for proxying requests to the Recommendation Service.

This router handles all requests prefixed with `/recommendation`. It is specifically
designed to handle potentially long-running requests (e.g., to an LLM) by
implementing a longer timeout and gracefully handling timeout errors.
"""

from fastapi import APIRouter, Request, Depends, Body
from fastapi.responses import JSONResponse, Response
from .. import auth
import httpx
import os

router = APIRouter(prefix="/recommendation", tags=["Recommendation Service"])

RECOMMENDATION_SERVICE_URL = os.getenv("RECOMMENDATION_SERVICE_URL")

# --- DEFINE A LONGER TIMEOUT FOR THIS SERVICE ---
# AI/LLM calls can be slow, so we need to give the downstream service more
TIMEOUT_CONFIG = httpx.Timeout(60.0, connect=5.0)


async def _proxy_to_recommendation_service(path: str, request: Request, user_id: str):
    """
    A shared helper function to proxy requests to the downstream Recommendation Service.

    This function forwards the original request's details and adds an `X-User-ID`
    header. It uses a custom timeout configuration suitable for slow services
    and handles timeout errors by returning a 504 Gateway Timeout response.

    Args:
        path (str): The path to be appended to the service URL.
        request (Request): The original incoming FastAPI request object.
        user_id (str): The authenticated user's ID from the JWT token.
    """
    if not RECOMMENDATION_SERVICE_URL:
        return JSONResponse({"detail": "Recommendation Service is not configured."}, status_code=503)

    # Apply the custom timeout configuration when creating the client.
    async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG) as client:
        url = f"{RECOMMENDATION_SERVICE_URL}/{path}"

        headers = dict(request.headers)
        headers["X-User-ID"] = user_id
        headers.pop('host', None)

        try:
            response = await client.request(
                method=request.method,
                url=url,
                headers=headers,
                params=request.query_params,
                content=await request.body()
            )
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type=response.headers.get('content-type')
            )
        except httpx.ReadTimeout:
            return JSONResponse(
                {"detail": "The request to the Recommendation Service timed out."},
                status_code=504
            )
        except httpx.ConnectError:
            return JSONResponse({"detail": "Could not connect to Recommendation Service."}, status_code=503)


# --- PROXY ENDPOINTS (Protected by auth.get_current_user) ---
# Each endpoint captures the sub-path and forwards the request using the helper function.

@router.get("/{path:path}", tags=["Protected Routes Proxy"])
async def get_recommendation_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user)):
    """Proxies GET requests to any path under /recommendation to the Recommendation Service."""
    return await _proxy_to_recommendation_service(path, request, user_id)


@router.post("/{path:path}", tags=["Protected Routes Proxy"])
async def post_recommendation_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user),
                                    body: dict = Body(None)):
    """Proxies POST requests to any path under /recommendation to the Recommendation Service."""
    return await _proxy_to_recommendation_service(path, request, user_id)


@router.put("/{path:path}", tags=["Protected Routes Proxy"])
async def put_recommendation_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user),
                                   body: dict = Body(None)):
    """Proxies PUT requests to any path under /recommendation to the Recommendation Service."""
    return await _proxy_to_recommendation_service(path, request, user_id)


@router.patch("/{path:path}", tags=["Protected Routes Proxy"])
async def patch_recommendation_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user),
                                     body: dict = Body(None)):
    """Proxies PATCH requests to any path under /recommendation to the Recommendation Service."""
    return await _proxy_to_recommendation_service(path, request, user_id)


@router.delete("/{path:path}", tags=["Protected Routes Proxy"])
async def delete_recommendation_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user)):
    """Proxies DELETE requests to any path under /recommendation to the Recommendation Service."""
    return await _proxy_to_recommendation_service(path, request, user_id)