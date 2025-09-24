"""
Main application file for the FastAPI API Gateway.

This file initializes the FastAPI application, includes the necessary routers
for the microservices, sets up middleware, and defines global exception handlers.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import httpx
from starlette.responses import StreamingResponse
import os
from dotenv import load_dotenv
from .routers import user_service, recommendation_service, health_metrics_service

load_dotenv()

app = FastAPI(
    title="Health App API Gateway",
    description="A single entry point for all health application services.",
    version="1.0.0"
)

# Include the routers for different microservices.
# Each router handles a specific path prefix (e.g., /user, /recommendations).
app.include_router(user_service.router)
app.include_router(recommendation_service.router)
app.include_router(health_metrics_service.router)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Middleware to calculate and add a custom `X-Process-Time` header
    to every response, indicating how long the request took to process.
    """
    import time
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# General exception handler to ensure consistent error responses.
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handles all FastAPI `HTTPException`s to return a standardized
    JSON error message.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )

@app.get("/")
def read_root():
    """
    Root endpoint for the API Gateway.
    Provides a simple health check to confirm the service is running.
    """
    return {"message": "API Gateway is running"}