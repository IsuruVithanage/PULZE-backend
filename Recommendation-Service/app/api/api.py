# app/api/api.py
from fastapi import APIRouter
from app.api.endpoints import recommendation_router

api_router = APIRouter()
api_router.include_router(recommendation_router.router, tags=["Health Recommendations"])