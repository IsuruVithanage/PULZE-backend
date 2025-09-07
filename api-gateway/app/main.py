from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import httpx
from starlette.responses import StreamingResponse
import os
from dotenv import load_dotenv

from .routers import user_service, recommendation_service, health_metrics_service
from . import auth

load_dotenv()

app = FastAPI(
    title="Health App API Gateway",
    description="A single entry point for all health application services.",
    version="1.0.0"
)

# Include the routers for different services
app.include_router(user_service.router)
app.include_router(recommendation_service.router)
app.include_router(health_metrics_service.router)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    A simple middleware to add a process time header.
    In a real-world scenario, you might add more complex logic here,
    like logging or request validation.
    """
    import time
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# General exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )

@app.get("/")
def read_root():
    return {"message": "API Gateway is running"}
