from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from app.models.request_models import RecommendationRequest, RecommendationResponse
from app.services.rag_service import RAGService
import os
from app.core.config import settings
import shutil

router = APIRouter(
    prefix="/recommendations",
    tags=["recommendations"],
    responses={404: {"description": "Not found"}},
)


def get_rag_service():
    """Dependency to get the RAG service"""
    try:
        return RAGService()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing RAG service: {str(e)}")


@router.post("/diet", response_model=RecommendationResponse)
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


@router.post("/upload-pdf", status_code=201)
async def upload_pdf(
        file: UploadFile = File(...),
        rag_service: RAGService = Depends(get_rag_service)
):
    """
    Upload a PDF file to be used by the RAG service
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        # Create the PDF directory if it doesn't exist
        os.makedirs(settings.PDF_DIRECTORY, exist_ok=True)

        # Define the file path
        file_path = os.path.join(settings.PDF_DIRECTORY, "diet.pdf")

        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load the PDF in the RAG service
        success = rag_service.load_pdf_from_file(file_path)

        if success:
            return {"message": f"PDF file '{file.filename}' uploaded successfully and loaded into the RAG service"}
        else:
            raise HTTPException(status_code=500, detail="Failed to load the PDF into the RAG service")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")


@router.get("/status")
async def get_status(
        rag_service: RAGService = Depends(get_rag_service)
):
    """
    Get the status of the RAG service
    """
    return {
        "pdf_loaded": rag_service.is_pdf_loaded,
        "documents_count": len(rag_service.doc_splits) if rag_service.is_pdf_loaded else 0,
        "model": settings.LLM_MODEL,
        "retriever_k": settings.RETRIEVER_K
    }