from fastapi import APIRouter, Request, Depends
from starlette.responses import StreamingResponse

from .. import auth
import httpx
import os

router = APIRouter(prefix="/health-metrics", tags=["Health Metrics Service"])

HEALTH_METRICS_SERVICE_URL = os.getenv("HEALTH_METRICS_SERVICE_URL", "http://localhost:8003")

@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def route_to_health_metrics_service(
    path: str,
    request: Request,
    user_id: str = Depends(auth.get_current_user)
):
    """
    Routes requests to the Health Metrics Service.
    """
    async with httpx.AsyncClient() as client:
        url = f"{HEALTH_METRICS_SERVICE_URL}/{path}"
        headers = dict(request.headers)
        headers["X-User-ID"] = user_id

        try:
            response = await client.request(
                method=request.method,
                url=url,
                headers=headers,
                params=request.query_params,
                content=await request.body()
            )
            return StreamingResponse(
                response.aiter_raw(),
                status_code=response.status_code,
                headers=response.headers,
            )
        except httpx.ConnectError:
            return {"error": f"Could not connect to Health Metrics Service at {HEALTH_METRICS_SERVICE_URL}"}, 503
