from fastapi import APIRouter, Request, Depends, Body
from fastapi.responses import JSONResponse, Response
from .. import auth
from .. import schemas
import httpx
import os

router = APIRouter(prefix="/user", tags=["User Service"])

USER_SERVICE_URL = os.getenv("USER_SERVICE_URL")

@router.post("/auth/login", response_model=schemas.Token, tags=["Authentication"])
async def login(login_data: schemas.UserLogin):
    if not USER_SERVICE_URL:
        return JSONResponse({"detail": "User Service is not configured."}, status_code=503)
    async with httpx.AsyncClient() as client:
        url = f"{USER_SERVICE_URL}/auth/login"
        try:
            response = await client.post(url, json=login_data.dict())
            return Response(content=response.content, status_code=response.status_code, media_type=response.headers.get('content-type'))
        except httpx.ConnectError:
            return JSONResponse({"detail": "Could not connect to User Service."}, status_code=503)


async def _proxy_to_user_service(path: str, request: Request, user_id: str):
    if not USER_SERVICE_URL:
        return JSONResponse({"detail": "User Service is not configured."}, status_code=503)
    async with httpx.AsyncClient() as client:
        url = f"{USER_SERVICE_URL}/{path}"
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
        except httpx.ConnectError:
            return JSONResponse({"detail": "Could not connect to User Service."}, status_code=503)


# --- PROXY ENDPOINTS (Now with correct Swagger body) ---

@router.get("/{path:path}", tags=["Protected Routes Proxy"])
async def get_user_service_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user)):
    return await _proxy_to_user_service(path, request, user_id)

@router.post("/{path:path}", tags=["Protected Routes Proxy"])
async def post_user_service_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user), body: dict = Body(None)):
    return await _proxy_to_user_service(path, request, user_id)

@router.put("/{path:path}", tags=["Protected Routes Proxy"])
async def put_user_service_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user), body: dict = Body(None)):
    return await _proxy_to_user_service(path, request, user_id)

@router.patch("/{path:path}", tags=["Protected Routes Proxy"])
async def patch_user_service_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user), body: dict = Body(None)):
    return await _proxy_to_user_service(path, request, user_id)

@router.delete("/{path:path}", tags=["Protected Routes Proxy"])
async def delete_user_service_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user)):
    return await _proxy_to_user_service(path, request, user_id)

