from fastapi import APIRouter, Request, Depends, Body
from fastapi.responses import JSONResponse, Response
from .. import auth
import httpx
import os

router = APIRouter(prefix="/health-metrics", tags=["Health Metrics Service"])

HEALTH_METRICS_SERVICE_URL = os.getenv("HEALTH_METRICS_SERVICE_URL")

# Define a longer timeout. Since this service might also have small LLM calls,
# a longer timeout prevents crashes during processing.
TIMEOUT_CONFIG = httpx.Timeout(60.0)


async def _proxy_to_health_metrics_service(path: str, request: Request, user_id: str):
    """
    A shared helper function to proxy requests to the downstream Health Metrics Service.
    """
    if not HEALTH_METRICS_SERVICE_URL:
        return JSONResponse({"detail": "Health Metrics Service is not configured."}, status_code=503)

    async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG) as client:
        url = f"{HEALTH_METRICS_SERVICE_URL}/{path}"
        headers = dict(request.headers)
        headers["X-User-ID"] = user_id
        headers.pop('host', None)

        try:
            response = await client.request(
                method=request.method, url=url, headers=headers,
                params=request.query_params, content=await request.body()
            )
            # Use a standard, reliable Response object to prevent crashes
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type=response.headers.get('content-type')
            )
        except httpx.ReadTimeout:
            # Gracefully handle timeouts
            return JSONResponse({"detail": "The request to the Health Metrics Service timed out."}, status_code=504)
        except httpx.ConnectError:
            # Gracefully handle connection errors
            return JSONResponse({"detail": "Could not connect to Health Metrics Service."}, status_code=503)


@router.get("/{path:path}")
async def get_health_metrics_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user)):
    """Proxies GET requests to the Health Metrics Service."""
    return await _proxy_to_health_metrics_service(path, request, user_id)


@router.post("/{path:path}")
async def post_health_metrics_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user), body: dict = Body(None)):
    """Proxies POST requests (with a body) to the Health Metrics Service."""
    return await _proxy_to_health_metrics_service(path, request, user_id)


@router.put("/{path:path}")
async def put_health_metrics_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user), body: dict = Body(None)):
    """Proxies PUT requests (with a body) to the Health Metrics Service."""
    return await _proxy_to_health_metrics_service(path, request, user_id)


@router.patch("/{path:path}")
async def patch_health_metrics_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user), body: dict = Body(None)):
    """Proxies PATCH requests (with a body) to the Health Metrics Service."""
    return await _proxy_to_health_metrics_service(path, request, user_id)


@router.delete("/{path:path}")
async def delete_health_metrics_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user)):
    """Proxies DELETE requests to the Health Metrics Service."""
    return await _proxy_to_health_metrics_service(path, request, user_id)

