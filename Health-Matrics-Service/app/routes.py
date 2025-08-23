import re
from datetime import datetime
from typing import List
from groq import Groq

import boto3
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import os, tempfile, requests
from urllib.parse import urlparse

from app import schemas, models, database
from app.database import SessionLocal, engine
from app.utils import ocr, parser

from pydantic import BaseModel, HttpUrl

models.Base.metadata.create_all(bind=engine)
load_dotenv()
router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def download_file_from_url(url: str) -> str:
    """Download file from a public S3 URL into a temp file and return path."""
    parsed = urlparse(url)
    if parsed.scheme not in ["http", "https"]:
        raise ValueError("Only http/https URLs are allowed")

    # Stream to temp file
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        suffix = os.path.splitext(parsed.path)[1]  # preserve extension
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        for chunk in r.iter_content(chunk_size=1024*1024):
            if chunk:
                tmp.write(chunk)
        tmp.close()
        return tmp.name

class S3UploadRequest(BaseModel):
    s3_url: HttpUrl
    user_id: int

@router.post("/upload", response_model=schemas.ReportResponse)
async def upload_s3_file(request: S3UploadRequest, db: Session = Depends(get_db)):
    try:
        file_location = download_file_from_url(str(request.s3_url))  # ✅ cast to str
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {e}")

    try:
        extracted_text = ocr.get_text_from_any_file(file_location)
        report_data = parser.parse_report(extracted_text)

        # include user_id in the report
        report_data["user_id"] = request.user_id

        new_report = models.Report(**report_data)
        db.add(new_report)
        db.commit()
        db.refresh(new_report)
        return new_report
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)



s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)

BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

class PresignedRequest(BaseModel):
    user_id: int
    filename: str
    content_type: str

class PresignedUrlResponse(BaseModel):
    upload_url: str
    file_url: str

def sanitize_filename(filename: str) -> str:
    # simple example: replace spaces and parentheses
    return "".join(c for c in filename if c.isalnum() or c in ("-", "_", ".")) or "file"

@router.post("/s3/presigned", response_model=PresignedUrlResponse)
async def generate_presigned_url(req: PresignedRequest):

    try:
        safe_filename = sanitize_filename(req.filename)
        key = f"uploads/{req.user_id}/{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{safe_filename}"

        content_type = str(req.content_type)

        presigned = s3_client.generate_presigned_url(
            "put_object",
            Params={"Bucket": os.getenv('S3_BUCKET_NAME'), "Key": key, "ContentType": content_type},
            ExpiresIn=3600,
        )

        file_url = f"https://{os.getenv('S3_BUCKET_NAME')}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{key}"
        return PresignedUrlResponse(upload_url=presigned, file_url=file_url)
    except Exception as e:
        print("Error generating presigned URL:", e)
        raise HTTPException(status_code=500, detail=str(e))


class FileItem(BaseModel):
    key: str
    url: str
    last_modified: datetime

@router.get("/s3/files/{user_id}", response_model=List[FileItem])
async def list_user_files(user_id: int):
    try:
        prefix = f"uploads/{user_id}/"
        response = s3_client.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=prefix
        )

        if "Contents" not in response:
            return []

        files = []
        for obj in response["Contents"]:
            key = obj["Key"]

            # Generate a presigned URL (GET) for viewing
            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": BUCKET_NAME, "Key": key},
                ExpiresIn=3600,  # 1 hour
            )

            files.append(
                FileItem(
                    key=key.split("/")[-1],  # only file name
                    url=url,
                    last_modified=obj["LastModified"],
                )
            )

        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {e}")



@router.get("/users/{user_id}/latest-report")
def get_latest_report(user_id: int, db: Session = Depends(get_db)):
    report = (
        db.query(models.Report)
        .filter(models.Report.user_id == user_id)
        .order_by(models.Report.updated_at.desc())
        .first()
    )
    if not report:
        raise HTTPException(status_code=404, detail="No report found")

    # Convert to dict so frontend can iterate dynamically
    return {
        "total_cholesterol": report.total_cholesterol,
        "hdl_cholesterol": report.hdl_cholesterol,
        "triglycerides": report.triglycerides,
        "ldl_cholesterol": report.ldl_cholesterol,
        "triglycerides_hdl_ratio": report.triglycerides_hdl_ratio,
        "updated_at": report.updated_at,
    }

try:
    groq_client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )
except Exception as e:
    # Handle cases where the API key might be missing
    print(f"Error initializing Groq client: {e}")
    groq_client = None

@router.get("/metric/recommendation")
def get_metric_recommendation(
    metric_name: str,    # e.g., "BMI", "Blood Pressure", "Blood Sugar"
    metric_value: float, # The actual value of the metric
    gender: str,         # e.g., "Male", "Female"
    age: int,            # User's age
    weight_kg: float,    # User's weight in kilograms
    height_cm: float     # User's height in centimeters
):
    """
    Provides a brief, AI-generated insight into a user's health metric.
    """
    if not groq_client:
        raise HTTPException(
            status_code=500,
            detail="Groq client is not initialized. Check API key."
        )

    # 1. Construct a detailed prompt for the language model.
    # This prompt guides the AI to give the specific kind of response you want.
    prompt = f"""
    A user has provided the following health data:
    - Gender: {gender}
    - Age: {age}
    - Weight: {weight_kg} kg
    - Height: {height_cm} cm
    - Health Metric: '{metric_name}'
    - Metric Value: {metric_value}

    Based on this data, act as a helpful health assistant and provide a brief, easy-to-understand insight into their '{metric_name}'.

    Your response MUST follow these rules:
    1.  Start by stating the user's current metric value and the category it falls into (e.g., "Overweight", "Normal", "High").
    2.  Briefly explain what this means for their health in simple terms.
    3.  Mention the generally accepted healthy range for this metric.
    4.  Keep the entire response to a single, concise paragraph (about 2-4 sentences).
    5.  Do NOT give medical advice. Focus on explaining the metric.
    6.  Maintain a supportive and encouraging tone.

    Here is an example for a BMI of 28:
    "Your current BMI is 28, which falls in the Overweight range. This means your body weight is higher than what is considered healthy for your height. Maintaining a BMI between 18.5 and 24.9 is ideal and can reduce the risk of developing health issues."

    Now, generate the insight for the user's data provided above.
    """

    try:
        # 2. Call the Groq API with the constructed prompt
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            # Using a powerful model like Llama 3 70b is great for this kind of nuanced task
            model="llama3-70b-8192",
        )

        # 3. Extract and return the content of the response
        recommendation = chat_completion.choices[0].message.content
        return {"recommendation": recommendation.strip()}

    except Exception as e:
        # 4. Handle potential API errors gracefully
        print(f"An error occurred with the Groq API: {e}")
        raise HTTPException(status_code=500, detail="Failed to get recommendation from AI model.")

