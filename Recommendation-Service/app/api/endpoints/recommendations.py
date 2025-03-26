from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from app.models.request_models import RecommendationRequest, RecommendationResponse
from app.services.rag_service import RAGService
import os
from app.core.config import settings
import shutil

router = APIRouter(
    tags=["recommendations"],
    responses={404: {"description": "Not found"}},
)


def get_rag_service():
    """Dependency to get the RAG service"""
    try:
        return RAGService()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing RAG service: {str(e)}")


@router.post("/recommendation", response_model=RecommendationResponse)
async def get_diet_recommendation(
        request: RecommendationRequest,
        rag_service: RAGService = Depends(get_rag_service)
):
    """
    Get diet recommendations based on health metrics
    """
    try:
        # Convert Pydantic model to dict for the service
        metrics_dict = request.health_metrics.model_dump()

        # Get recommendation from RAG service
        recommendation = await rag_service.get_recommendation(
            metrics=metrics_dict,
            additional_info=request.additional_info
        )

        return RecommendationResponse(recommendation=recommendation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendation: {str(e)}")
