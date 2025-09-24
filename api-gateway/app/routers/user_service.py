"""
Router for proxying requests to the User Service.

This router handles all requests prefixed with `/user`. It includes a special
public endpoint for login and protected "catch-all" endpoints that forward
all other requests to the downstream User Service.
"""

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
    """
    Proxies login requests to the User Service to obtain a JWT token.

    This is an unprotected endpoint. It takes user credentials, sends them to the
    actual User Service, and returns the response (containing the token) to the client.
    """
    if not USER_SERVICE_URL:
        return JSONResponse({"detail": "User Service is not configured."}, status_code=503)

    async with httpx.AsyncClient() as client:
        url = f"{USER_SERVICE_URL}/auth/login"
        try:
            response = await client.post(url, json=login_data.dict())
            return Response(content=response.content, status_code=response.status_code,
                            media_type=response.headers.get('content-type'))
        except httpx.ConnectError:
            return JSONResponse({"detail": "Could not connect to User Service."}, status_code=503)


async def _proxy_to_user_service(path: str, request: Request, user_id: str):
    """
    A helper function to forward requests to the downstream User Service.

    It constructs the target URL, copies headers, body, and query parameters,
    and adds a special `X-User-ID` header to identify the authenticated user
    for the downstream service.

    Args:
        path (str): The path to be appended to the user service URL.
        request (Request): The original incoming FastAPI request object.
        user_id (str): The authenticated user's ID from the JWT token.
    """
    if not USER_SERVICE_URL:
        return JSONResponse({"detail": "User Service is not configured."}, status_code=503)

    async with httpx.AsyncClient() as client:
        url = f"{USER_SERVICE_URL}/{path}"

        headers = dict(request.headers)
        headers["X-User-ID"] = user_id
        headers.pop('host', None)

        try:
            # Make a new request to the downstream service with all original details.
            response = await client.request(
                method=request.method,
                url=url,
                headers=headers,
                params=request.query_params,
                content=await request.body()
            )
            # Return the response from the downstream service to the original client.
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type=response.headers.get('content-type')
            )
        except httpx.ConnectError:
            return JSONResponse({"detail": "Could not connect to User Service."}, status_code=503)


# --- PROXY ENDPOINTS (Protected by auth.get_current_user) ---
# The `{path:path}` syntax allows the 'path' parameter to match any sub-path.

@router.get("/{path:path}", tags=["Protected Routes Proxy"])
async def get_user_service_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user)):
    """Proxies GET requests to any path under /user to the User Service."""
    return await _proxy_to_user_service(path, request, user_id)


@router.post("/{path:path}", tags=["Protected Routes Proxy"])
async def post_user_service_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user),
                                  body: dict = Body(None)):
    """Proxies POST requests to any path under /user to the User Service."""
    return await _proxy_to_user_service(path, request, user_id)


@router.put("/{path:path}", tags=["Protected Routes Proxy"])
async def put_user_service_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user),
                                 body: dict = Body(None)):
    """Proxies PUT requests to any path under /user to the User Service."""
    return await _proxy_to_user_service(path, request, user_id)


@router.patch("/{path:path}", tags=["Protected Routes Proxy"])
async def patch_user_service_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user),
                                   body: dict = Body(None)):
    """Proxies PATCH requests to any path under /user to the User Service."""
    return await _proxy_to_user_service(path, request, user_id)


@router.delete("/{path:path}", tags=["Protected Routes Proxy"])
async def delete_user_service_proxy(path: str, request: Request, user_id: str = Depends(auth.get_current_user)):
    """Proxies DELETE requests to any path under /user to the User Service."""
    return await _proxy_to_user_service(path, request, user_id)