"""
Router for proxying requests to the Health Metrics Service.

This module handles all requests prefixed with `/health-metrics`. It securely
forwards requests to the downstream Health Metrics Service, incorporating a
custom timeout to robustly handle potentially slow data processing tasks.
"""

from fastapi import APIRouter, Request, Depends, Body
from fastapi.responses import JSONResponse, Response
from .. import auth
import httpx
import os

router = APIRouter(prefix="/health-metrics", tags=["Health Metrics Service"])

HEALTH_METRICS_SERVICE_URL = os.getenv("HEALTH_METRICS_SERVICE_URL")

# Define a longer timeout. Since this service might involve data processing or
# small LLM calls, a longer timeout prevents the gateway from giving up too early.
TIMEOUT_CONFIG = httpx.Timeout(60.0)


async def _proxy_to_health_metrics_service(path: str, request: Request, user_id: str):
    """
    A shared helper function to proxy requests to the downstream Health Metrics Service.

    It constructs the target URL, forwards headers, body, and query parameters,
    and injects an `X-User-ID` header. It also uses a custom timeout and
    handles connection/timeout errors gracefully.

    Args:
        path (str): The sub-path to be appended to the service URL.
        request (Request): The original incoming FastAPI request object.
        user_id (str): The authenticated user's ID from the JWT token.
    """
    if not HEALTH_METRICS_SERVICE_URL:
        return JSONResponse({"detail": "Health Metrics Service is not configured."}, status_code=503)

    # Use the custom timeout configuration for the HTTP client.
    async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG) as client:
        url = f"{HEALTH_METRICS_SERVICE_URL}/{path}"

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
            return JSONResponse({"detail": "The request to the Health Metrics Service timed out."}, status_code=504)
        except httpx.ConnectError:
            return JSONResponse({"detail": "Could not connect to Health Metrics Service."}, status_code=503)


# --- PROXY ENDPOINTS (Protected by auth.get_current_user) ---
# The `{path:path}` parameter captures the entire sub-path of the request.

@router.get("/{path:path}", tags=["Protected Routes Proxy"])
async def get_health_metrics_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user)):
    """Proxies GET requests to the Health Metrics Service."""
    return await _proxy_to_health_metrics_service(path, request, user_id)


@router.post("/{path:path}", tags=["Protected Routes Proxy"])
async def post_health_metrics_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user),
                                    body: dict = Body(None)):
    """Proxies POST requests (with a body) to the Health Metrics Service."""
    return await _proxy_to_health_metrics_service(path, request, user_id)


@router.put("/{path:path}", tags=["Protected Routes Proxy"])
async def put_health_metrics_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user),
                                   body: dict = Body(None)):
    """Proxies PUT requests (with a body) to the Health Metrics Service."""
    return await _proxy_to_health_metrics_service(path, request, user_id)


@router.patch("/{path:path}", tags=["Protected Routes Proxy"])
async def patch_health_metrics_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user),
                                     body: dict = Body(None)):
    """Proxies PATCH requests (with a body) to the Health Metrics Service."""
    return await _proxy_to_health_metrics_service(path, request, user_id)


@router.delete("/{path:path}", tags=["Protected Routes Proxy"])
async def delete_health_metrics_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user)):
    """Proxies DELETE requests to the Health Metrics Service."""
    return await _proxy_to_health_metrics_service(path, request, user_id)