from fastapi import APIRouter, Request, Depends, Body
from fastapi.responses import JSONResponse, Response
from .. import auth
import httpx
import os

router = APIRouter(prefix="/recommendation", tags=["Recommendation Service"])

RECOMMENDATION_SERVICE_URL = os.getenv("RECOMMENDATION_SERVICE_URL")

# --- DEFINE A LONGER TIMEOUT FOR THIS SERVICE ---
# LLM calls can be slow, so we need to give them more time.
# A Timeout object can configure connect, read, write, and pool timeouts.
# Here we set a 60-second read timeout.
TIMEOUT_CONFIG = httpx.Timeout(60.0)


async def _proxy_to_recommendation_service(path: str, request: Request, user_id: str):
    """
    A shared helper function to proxy requests to the downstream Recommendation Service.
    """
    if not RECOMMENDATION_SERVICE_URL:
        return JSONResponse({"detail": "Recommendation Service is not configured."}, status_code=503)

    # --- APPLY THE TIMEOUT CONFIGURATION HERE ---
    async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG) as client:
        url = f"{RECOMMENDATION_SERVICE_URL}/{path}"
        headers = dict(request.headers)
        headers["X-User-ID"] = user_id
        headers.pop('host', None)

        try:
            response = await client.request(
                method=request.method, url=url, headers=headers,
                params=request.query_params, content=await request.body()
            )
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type=response.headers.get('content-type')
            )
        except httpx.ReadTimeout:
            # Handle the timeout gracefully instead of crashing
            return JSONResponse({"detail": "The request to the Recommendation Service timed out."}, status_code=504) # 504 Gateway Timeout
        except httpx.ConnectError:
            return JSONResponse({"detail": "Could not connect to Recommendation Service."}, status_code=503)


# The endpoint functions below do not need to be changed.
@router.get("/{path:path}")
async def get_recommendation_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user)):
    return await _proxy_to_recommendation_service(path, request, user_id)

@router.post("/{path:path}")
async def post_recommendation_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user), body: dict = Body(None)):
    return await _proxy_to_recommendation_service(path, request, user_id)

@router.put("/{path:path}")
async def put_recommendation_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user), body: dict = Body(None)):
    return await _proxy_to_recommendation_service(path, request, user_id)

@router.patch("/{path:path}")
async def patch_recommendation_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user), body: dict = Body(None)):
    return await _proxy_to_recommendation_service(path, request, user_id)

@router.delete("/{path:path}")
async def delete_recommendation_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user)):
    return await _proxy_to_recommendation_service(path, request, user_id)