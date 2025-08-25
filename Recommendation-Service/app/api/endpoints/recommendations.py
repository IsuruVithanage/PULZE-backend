from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
# --- Import Form and your new DocumentCategory Enum ---
from app.models.request_models import RecommendationRequest, RecommendationResponse, DocumentCategory
from app.services.rag_service import RAGService
from app.core.config import settings
from app.utils.pdf_loader import save_uploaded_pdf
import os
import uuid

router = APIRouter()


# ... (your get_rag_service and get_diet_recommendation functions remain the same) ...
def get_rag_service():
    """Dependency to get the RAG service singleton instance."""
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
    Get diet recommendations based on health metrics.
    This endpoint uses the knowledge base indexed in Pinecone.
    """
    try:
        metrics_dict = request.health_metrics.model_dump()
        recommendation = await rag_service.get_recommendation(
            metrics=metrics_dict,
            additional_info=request.additional_info
        )
        return RecommendationResponse(recommendation=recommendation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendation: {str(e)}")


# --- UPDATE THIS ENTIRE FUNCTION ---
@router.post("/document/upload", status_code=201)
async def upload_document(
        category: DocumentCategory = Form(..., description="The category of the document."),
        file: UploadFile = File(...),
        rag_service: RAGService = Depends(get_rag_service)
):
    """
    Upload a PDF document to a specific category to be indexed into the knowledge base.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    destination_path = None
    try:
        # Construct the destination path based on the category
        # e.g., data/sources/dietary/
        destination_dir = os.path.join(settings.PDF_DIRECTORY, "sources", category.value)
        os.makedirs(destination_dir, exist_ok=True)

        # Use a unique filename to avoid conflicts
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        destination_path = os.path.join(destination_dir, unique_filename)

        # Save the uploaded file directly to its final destination
        await save_uploaded_pdf(file, destination_path)

        # Index the new PDF document from its final path
        await rag_service.add_pdf_to_index(destination_path)

        # NOTE: We no longer need to clean up a temporary file

        return {
            "message": f"Document '{file.filename}' has been successfully processed and indexed into the '{category.value}' category."}
    except Exception as e:
        # If an error occurs after saving, clean up the created file
        if destination_path and os.path.exists(destination_path):
            os.remove(destination_path)
        raise HTTPException(status_code=500, detail=f"Failed to process and index document: {str(e)}")