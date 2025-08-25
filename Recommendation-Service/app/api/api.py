# app/api/api.py
from fastapi import APIRouter
from app.api.endpoints import recommendations

api_router = APIRouter()
api_router.include_router(recommendations.router, tags=["Health Recommendations"])