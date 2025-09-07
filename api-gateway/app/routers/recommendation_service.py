from fastapi import APIRouter, Request, Depends
from starlette.responses import StreamingResponse

from .. import auth
import httpx
import os

router = APIRouter(prefix="/recommendation", tags=["Recommendation Service"])

RECOMMENDATION_SERVICE_URL = os.getenv("RECOMMENDATION_SERVICE_URL", "http://localhost:8002")


@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def route_to_recommendation_service(
        path: str,
        request: Request,
        user_id: str = Depends(auth.get_current_user)
):
    """
    Routes requests to the Recommendation Service.
    """
    async with httpx.AsyncClient() as client:
        url = f"{RECOMMENDATION_SERVICE_URL}/{path}"
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
            return {"error": f"Could not connect to Recommendation Service at {RECOMMENDATION_SERVICE_URL}"}, 503
