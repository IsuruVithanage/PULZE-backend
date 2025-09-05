import tempfile
from datetime import datetime

import boto3
import httpx
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from app.models.request_models import RecommendationRequest, RecommendationResponse, DocumentCategory
from app.services.rag_service import RAGService
from app.core.config import settings
from app.utils import pdf_generator
from app.utils.pdf_loader import save_uploaded_pdf
from app.models import request_models as schemas
import os
import uuid

from app.utils.report_formatting import format_report_as_markdown, get_metric_status

load_dotenv()
router = APIRouter()

s3_client = boto3.client("s3")


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
        destination_dir = os.path.join(settings.PDF_DIRECTORY, "sources", category.value)
        os.makedirs(destination_dir, exist_ok=True)
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        destination_path = os.path.join(destination_dir, unique_filename)

        await save_uploaded_pdf(file, destination_path)

        await rag_service.add_pdf_to_index(destination_path)

        return {
            "message": f"Document '{file.filename}' has been successfully processed and indexed into the '{category.value}' category."}
    except Exception as e:
        if destination_path and os.path.exists(destination_path):
            os.remove(destination_path)
        raise HTTPException(status_code=500, detail=f"Failed to process and index document: {str(e)}")



@router.post("/reindex-all", status_code=200)
async def trigger_reindexing(rag_service: RAGService = Depends(get_rag_service)):
    """
    Triggers a full re-indexing of all documents in the 'data/sources' directory.
    This will add all documents to the existing index.
    NOTE: This does not clear the index first.
    """
    try:
        summary = await rag_service.reindex_all_sources()
        return summary
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during re-indexing: {str(e)}")


@router.post("/generate-doctor-summary")
async def generate_doctor_summary(
        request: schemas.RecommendationRequest,
        rag_service: RAGService = Depends(get_rag_service)
):
    user_id = request.health_metrics.user_id

    user_profile_url = f"http://localhost:8001/api/auth/users/{user_id}/profile"
    historical_data_url = f"http://localhost:8000/api/users/{user_id}/historical-data"

    async with httpx.AsyncClient() as client:
        try:
            user_res = await client.get(user_profile_url)
            metrics_res = await client.get(historical_data_url)
            user_res.raise_for_status()
            metrics_res.raise_for_status()
            user_data = user_res.json()
            historical_data = metrics_res.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code,
                                detail=f"Error from downstream service: {e.response.text}")
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Service unavailable: {e}")

    height_cm = user_data.get("height_cm")
    weight_kg = user_data.get("weight_kg")
    if height_cm and weight_kg:
        user_data['bmi'] = round(weight_kg / ((height_cm / 100) ** 2), 1)

    prompt = f"""
    Act as an expert clinical data analyst. Your task is to generate a concise, objective clinical summary paragraph (3-5 sentences) based on the provided time-series patient data.

    Your summary MUST:
    1.  Analyze Trends: For each key metric, analyze the entire series of data points to determine the overall trend (e.g., "improving," "worsening," "stable," or "volatile/erratic").
    2.  Interpret the latest value based on general clinical guidelines (e.g., 'Normal', 'High', 'Prediabetic').
    3.  Synthesize the information into a single, holistic overview of the patient's health trajectory.
    4.  Be Objective: Do NOT provide recommendations or medical advice. Stick to summarizing the data's story.

    RAW TIME-SERIES PATIENT DATA:
    - User Profile: {user_data}
    - Historical Metrics: {historical_data['metrics']}

    Generate the clinical summary paragraph now.
    """

    ai_summary_result = await rag_service.llm.ainvoke(prompt)
    ai_summary = ai_summary_result.content
    markdown_report = format_report_as_markdown(user_data, historical_data, ai_summary)

    pdf_path = None
    try:
        template_context = {
            "user_data": user_data,
            "ai_summary": ai_summary,
            "generation_date": datetime.now().strftime("%B %d, %Y"),
            "historical_data": [
                {
                    "name": m['name'],
                    "unit": m.get('unit', ''),
                    "latest_value": m['series'][-1]['value'] if m.get('series') else 'N/A',
                    "previous_value": m['series'][-2]['value'] if len(m.get('series', [])) > 1 else 'N/A',
                    "status": get_metric_status(m['name'],
                                                                 m['series'][-1]['value'] if m.get('series') else None),
                }
                for m in historical_data.get('metrics', [])
            ]
        }

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf_path = tmp.name

        pdf_generator.create_pdf_from_html(template_context, pdf_path)

        s3_key = f"reports/{user_id}/health_summary_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.pdf"
        s3_client.upload_file(pdf_path, os.getenv("S3_BUCKET_NAME"), s3_key)

        presigned_pdf_url = s3_client.generate_presigned_url(
            'get_object', Params={'Bucket': os.getenv("S3_BUCKET_NAME"), 'Key': s3_key}, ExpiresIn=3600
        )

    except (NoCredentialsError, Exception) as e:
        print(f"Error with S3 or PDF generation: {e}")
        raise HTTPException(status_code=500, detail="Could not generate or save the report PDF.")
    finally:
        if pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)

    return {
        "report_markdown": markdown_report,
        "pdf_url": presigned_pdf_url
    }