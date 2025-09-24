"""
Defines all API endpoints for the Health Metrics Service.

This includes file upload handling, OCR processing, data storage,
AI-powered recommendations, and inter-service communication.
"""

from datetime import datetime
from typing import List, Union

import httpx
from dateutil.relativedelta import relativedelta
from groq import Groq

import boto3
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import os, tempfile, requests
from urllib.parse import urlparse

from app import schemas, models
from app.database import engine, get_db
from app.schemas import CombinedReportResponse
from app.utils import ocr, parser

from pydantic import BaseModel, HttpUrl

models.Base.metadata.create_all(bind=engine)

load_dotenv()
router = APIRouter()

def download_file_from_url(url: str) -> str:
    """
    Downloads a file from a public URL into a temporary local file.

    Args:
        url (str): The public URL of the file to download.

    Returns:
        str: The local filesystem path to the downloaded temporary file.

    Note:
        The caller is responsible for deleting the temporary file after use.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ["http", "https"]:
        raise ValueError("Only http/https URLs are allowed")

    # Streams the download to a temp file to handle large files efficiently
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        suffix = os.path.splitext(parsed.path)[1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        for chunk in r.iter_content(chunk_size=1024*1024):
            if chunk:
                tmp.write(chunk)
        tmp.close()
        return tmp.name

def sanitize_filename(filename: str) -> str:
    """
    Removes potentially unsafe characters from a filename.
    """
    return "".join(c for c in filename if c.isalnum() or c in ("-", "_", ".")) or "file"


# --- File Upload Workflow ---

class S3UploadRequest(BaseModel):
    """Schema for the request to process a file already uploaded to S3."""
    s3_url: HttpUrl
    user_id: int
    report_type: schemas.ReportType

@router.post("/ocr", status_code=201)
async def upload_s3_file_v2(request: S3UploadRequest, db: Session = Depends(get_db)):
    """
    Processes a file after it has been uploaded to S3.

    This endpoint downloads the file, performs OCR, parses the text,
    and stores the structured data in the database.
    """
    file_location = None
    try:
        # Step 1: Download the file from its public S3 URL
        file_location = download_file_from_url(str(request.s3_url))
        # Step 2: Extract text from the downloaded file using OCR
        extracted_text = ocr.get_text_from_any_file(file_location)

        # Step 3: Parse the text and save to the correct database table
        if request.report_type == schemas.ReportType.LIPID:
            report_data = parser.parse_lipid_report(extracted_text)
            new_report = models.LipidReport(user_id=request.user_id, **report_data)
        elif request.report_type == schemas.ReportType.BLOOD_SUGAR:
            report_data = parser.parse_blood_sugar_report(extracted_text)
            new_report = models.BloodSugarReport(user_id=request.user_id, **report_data)
        else:
            raise HTTPException(status_code=400, detail="Invalid report type specified.")

        db.add(new_report)
        db.commit()
        db.refresh(new_report)
        return new_report

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Parsing failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    finally:
        # Crucial cleanup step: ensures the temporary file is deleted even if errors occur
        if file_location and os.path.exists(file_location):
            os.remove(file_location)


# --- S3 Management ---

# Initialize the Boto3 S3 client once for reuse
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

class PresignedRequest(BaseModel):
    """Schema for requesting a presigned URL."""
    user_id: int
    filename: str
    content_type: str
    report_type: schemas.ReportType

class PresignedUrlResponse(BaseModel):
    """Schema for the response containing the presigned URL."""
    upload_url: str
    file_url: str

@router.post("/s3/presigned", response_model=PresignedUrlResponse)
async def generate_presigned_url(req: PresignedRequest):
    """
    Generates a secure, temporary URL that a client can use to upload a
    file directly to S3, bypassing this server. This is a best practice.
    """
    try:
        safe_filename = sanitize_filename(req.filename)
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        # Creates a structured, unique object key for organization in S3
        key = f"uploads/{req.user_id}/{req.report_type.value}_{timestamp}_{safe_filename}"
        content_type = str(req.content_type)

        # Generate the URL for a PUT request, valid for 1 hour (3600 seconds)
        presigned = s3_client.generate_presigned_url(
            "put_object",
            Params={"Bucket": os.getenv('S3_BUCKET_NAME'), "Key": key, "ContentType": content_type},
            ExpiresIn=3600,
        )

        # Construct the permanent, public-facing URL of the file
        file_url = f"https://{os.getenv('S3_BUCKET_NAME')}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{key}"
        return PresignedUrlResponse(upload_url=presigned, file_url=file_url)
    except Exception as e:
        print("Error generating presigned URL:", e)
        raise HTTPException(status_code=500, detail=str(e))


class FileItem(BaseModel):
    """Schema for an item in the list of S3 files."""
    key: str
    url: str
    last_modified: datetime

@router.get("/s3/files/{user_id}", response_model=List[FileItem])
async def list_user_files(user_id: int):
    """
    Lists all files uploaded by a specific user to the S3 bucket.
    Generates temporary viewable URLs for each file.
    """
    try:
        prefix = f"uploads/{user_id}/"
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)

        if "Contents" not in response:
            return []

        files = []
        for obj in response["Contents"]:
            key = obj["Key"]
            # Generate a temporary GET URL for the client to view the file
            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": BUCKET_NAME, "Key": key},
                ExpiresIn=3600,
            )
            files.append(FileItem(key=key.split("/")[-1], url=url, last_modified=obj["LastModified"]))

        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {e}")


# --- Data Retrieval & Aggregation ---

@router.get("/users/{user_id}/latest-report", response_model=CombinedReportResponse)
def get_latest_combined_report(user_id: int, db: Session = Depends(get_db)):
    """
    Fetches the latest lipid and blood sugar reports for a user and
    combines them into a single response.
    """
    # Step 1: Fetch the most recent lipid report for the user
    latest_lipid_report = (
        db.query(models.LipidReport)
        .filter(models.LipidReport.user_id == user_id)
        .order_by(models.LipidReport.updated_at.desc())
        .first()
    )

    # Step 2: Fetch the most recent blood sugar report for the user
    latest_blood_sugar_report = (
        db.query(models.BloodSugarReport)
        .filter(models.BloodSugarReport.user_id == user_id)
        .order_by(models.BloodSugarReport.updated_at.desc())
        .first()
    )

    if not latest_lipid_report and not latest_blood_sugar_report:
        raise HTTPException(status_code=404, detail=f"No reports found for user {user_id}")

    # Step 3: Build the combined response dictionary from available data
    combined_data = {}
    last_updated_time = None

    if latest_lipid_report:
        combined_data.update({
            "total_cholesterol": latest_lipid_report.total_cholesterol,
            "hdl_cholesterol": latest_lipid_report.hdl_cholesterol,
            "triglycerides": latest_lipid_report.triglycerides,
            "ldl_cholesterol": latest_lipid_report.ldl_cholesterol,
            "triglycerides_hdl_ratio": latest_lipid_report.triglycerides_hdl_ratio,
        })
        last_updated_time = latest_lipid_report.updated_at

    if latest_blood_sugar_report:
        combined_data.update({
            "fasting_blood_sugar": latest_blood_sugar_report.fasting_blood_sugar,
            "random_blood_sugar": latest_blood_sugar_report.random_blood_sugar,
            "hba1c": latest_blood_sugar_report.hba1c,
        })
        # Determine which report is more recent for the 'last_updated' field
        if last_updated_time is None or latest_blood_sugar_report.updated_at > last_updated_time:
            last_updated_time = latest_blood_sugar_report.updated_at

    combined_data["last_updated"] = last_updated_time

    return combined_data


# --- AI-Powered Analysis ---

try:
    # Initialize the Groq client for AI-powered recommendations
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    groq_client = None

@router.get("/metric/recommendation")
def get_metric_recommendation(
        metric_name: str,
        metric_value: float,
        gender: str,
        age: int,
        weight_kg: float,
        height_cm: float,
        user_id: int,
        db: Session = Depends(get_db)
):
    """
    Provides a brief, AI-generated insight into a user's health metric,
    comparing it with the previous value if available.
    """
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq client is not initialized. Check API key.")

    # --- Find the previous metric value for comparison ---
    previous_value_text = ""
    previous_report = None
    metric_map = {
        "Total Cholesterol": "total_cholesterol", "HDL Cholesterol": "hdl_cholesterol",
        "LDL Cholesterol": "ldl_cholesterol", "Triglycerides": "triglycerides",
        "Fasting Blood Sugar": "fasting_blood_sugar", "Random Blood Sugar": "random_blood_sugar",
        "HbA1c": "hba1c"
    }
    api_metric_name = metric_map.get(metric_name)

    # Determine which database table to query
    if api_metric_name in {"total_cholesterol", "hdl_cholesterol", "ldl_cholesterol", "triglycerides"}:
        model_to_query = models.LipidReport
    elif api_metric_name in {"fasting_blood_sugar", "random_blood_sugar", "hba1c"}:
        model_to_query = models.BloodSugarReport
    else:
        model_to_query = None

    if model_to_query and api_metric_name:
        # Query for the second-to-last report (the one before the latest)
        # We use .offset(1) to skip the most recent record.
        previous_report = (
            db.query(model_to_query)
            .filter(model_to_query.user_id == user_id)
            .order_by(model_to_query.updated_at.desc())
            .offset(1)
            .first()
        )

    if previous_report:
        # Use getattr to dynamically access the attribute by its string name
        previous_value = getattr(previous_report, api_metric_name, None)
        if previous_value is not None:
            previous_date = previous_report.updated_at.strftime("%B %d, %Y")
            previous_value_text = f"For comparison, their previous value on {previous_date} was {previous_value}."

    # --- Dynamically construct a detailed prompt for the AI model ---
    prompt = f"""
    A user has provided the following health data:
    - Gender: {gender}
    - Age: {age}
    - Weight: {weight_kg} kg
    - Height: {height_cm} cm
    - Health Metric: '{metric_name}'
    - Metric Value: {metric_value}
    {'- Previous Metric Value: ' + previous_value_text if previous_value_text else ''}

    Based on this data, act as a helpful health assistant and provide a brief, easy-to-understand insight into their '{metric_name}'.

    Your response MUST follow these rules:
    1.  Start by stating the user's current metric value and the category it falls into (e.g., "Overweight", "Normal", "High").
    2.  Briefly explain what this means for their health in simple terms.
    3.  Mention the generally accepted healthy range for this metric.
    4.  **If a previous value is provided, briefly compare the current value to the previous one (e.g., "This is an improvement from your previous value of X..." or "This is higher than your previous reading of Y...").**
    5.  Keep the entire response to a single, concise paragraph (about 3-5 sentences).
    6.  Do NOT give medical advice. Focus on explaining the metric.
    7.  Maintain a supportive and encouraging tone.
    
    Now, generate the insight for the user's data provided above.
    """

    try:
        # Send the prompt to the Groq API
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-120b",
        )
        recommendation = chat_completion.choices[0].message.content
        return {"recommendation": recommendation.strip()}
    except Exception as e:
        print(f"An error occurred with the Groq API: {e}")
        raise HTTPException(status_code=500, detail="Failed to get recommendation from AI model.")


# --- Data Modification & Charting ---

@router.patch(
    "/users/{user_id}/report/metric",
    response_model=Union[schemas.LipidReportResponse, schemas.BloodSugarReportResponse]
)
def update_health_metric(
        user_id: int,
        request: schemas.UpdateMetricRequest,
        db: Session = Depends(get_db),
):
    """
    Updates a single metric in the latest health report for a given user.
    """
    model_to_query = None
    # Determine which report type to update based on the metric name
    if request.metric_name in {
        "total_cholesterol", "hdl_cholesterol", "triglycerides", "ldl_cholesterol",
        "vldl_cholesterol", "non_hdl_cholesterol", "total_hdl_ratio", "triglycerides_hdl_ratio"
    }:
        model_to_query = models.LipidReport
    elif request.metric_name in {"fasting_blood_sugar", "random_blood_sugar", "hba1c"}:
        model_to_query = models.BloodSugarReport
    else:
        raise HTTPException(status_code=400, detail=f"Invalid metric name: '{request.metric_name}'.")

    # Find the latest report of the correct type for the user
    latest_report = (
        db.query(model_to_query)
        .filter(model_to_query.user_id == user_id)
        .order_by(model_to_query.updated_at.desc())
        .first()
    )

    if not latest_report:
        raise HTTPException(status_code=404, detail=f"No report of the required type found for user {user_id} to update.")

    # Dynamically set the attribute on the report object and save it
    setattr(latest_report, request.metric_name, request.metric_value)
    try:
        db.commit()
        db.refresh(latest_report)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update report: {e}")

    return latest_report


@router.get("/users/{user_id}/reports/chart-data", response_model=schemas.ChartResponse)
def get_user_report_chart_data(
        user_id: int,
        report_type: schemas.ReportType,
        metric_name: str,
        months: int = 6,
        db: Session = Depends(get_db),
):
    """
    Retrieves historical data for a specific metric over N months,
    formatted for use in frontend chart libraries.
    """
    model_to_query = None
    allowed_metrics = set()
    # Determine which table to query and which metrics are allowed for that table
    if report_type == schemas.ReportType.LIPID:
        model_to_query = models.LipidReport
        allowed_metrics = {
            "total_cholesterol", "hdl_cholesterol", "triglycerides", "ldl_cholesterol",
            "vldl_cholesterol", "non_hdl_cholesterol", "total_hdl_ratio", "triglycerides_hdl_ratio"
        }
    elif report_type == schemas.ReportType.BLOOD_SUGAR:
        model_to_query = models.BloodSugarReport
        allowed_metrics = {"fasting_blood_sugar", "random_blood_sugar", "hba1c"}

    if metric_name not in allowed_metrics:
        raise HTTPException(status_code=400, detail=f"Invalid metric name '{metric_name}' for report type '{report_type}'.")

    # Calculate the start date for the query window
    start_date = datetime.now() - relativedelta(months=months)

    # Fetch all reports within the date range, oldest first
    historical_reports = (
        db.query(model_to_query)
        .filter(model_to_query.user_id == user_id, model_to_query.updated_at >= start_date)
        .order_by(model_to_query.updated_at.asc())
        .all()
    )

    if not historical_reports:
        return {"labels": [], "datasets": [{"data": []}]}

    # Format the data into labels (dates) and data points (metric values)
    labels = [report.updated_at.strftime("%b %d") for report in historical_reports]
    data_points = [getattr(report, metric_name, None) for report in historical_reports]

    return {"labels": labels, "datasets": [{"data": data_points}]}

@router.get("/users/{user_id}/latest-report")
def get_latest_combined_report(user_id: int, db: Session = Depends(get_db)):
    """
    Fetches the latest reports and combines them with user profile data
    (like BMI) fetched from the User-Service.
    """
    # Fetch latest reports (logic is simplified here for clarity, but it's the same as above)
    latest_lipid_report = db.query(models.LipidReport).filter(models.LipidReport.user_id == user_id).order_by(models.LipidReport.updated_at.desc()).first()
    latest_blood_sugar_report = db.query(models.BloodSugarReport).filter(models.BloodSugarReport.user_id == user_id).order_by(models.BloodSugarReport.updated_at.desc()).first()

    # --- Inter-Service Communication to get User Profile ---
    bmi = None
    weight = None
    try:
        # Make a server-to-server HTTP request to the User-Service
        profile_url = f"http://localhost:8001/api/auth/users/{user_id}/profile"
        with httpx.Client() as client:
            res = client.get(profile_url)
            res.raise_for_status()  # Check for errors from the other service
            profile_data = res.json()

        # Calculate BMI if height and weight are available
        height_cm = profile_data.get("height_cm")
        weight_kg = profile_data.get("weight_kg")
        weight = weight_kg

        if height_cm and weight_kg and height_cm > 0:
            height_m = height_cm / 100
            bmi = round(weight_kg / (height_m ** 2), 1)

    except httpx.RequestError as e:
        print(f"Could not connect to User-Service: {e}")
        # Continue without BMI if User-Service is down
    except Exception as e:
        print(f"Error processing user profile data: {e}")

    # Build the final combined dictionary with all available data
    combined_data = {"bmi": bmi, "weight": weight}
    if latest_lipid_report:
        combined_data.update(schemas.LipidReportResponse.from_orm(latest_lipid_report).dict())
    if latest_blood_sugar_report:
        combined_data.update(schemas.BloodSugarReportResponse.from_orm(latest_blood_sugar_report).dict())

    return combined_data

@router.get("/users/{user_id}/historical-data", response_model=schemas.HistoricalDataResponse)
def get_historical_data_for_summary(user_id: int, db: Session = Depends(get_db), months: int = 6):
    """
    Fetches a time-series of historical data for key metrics over the last N months.
    """
    metrics_to_track = {
        "Total Cholesterol": {"key": "total_cholesterol", "unit": "mg/dL", "model": models.LipidReport},
        "LDL Cholesterol": {"key": "ldl_cholesterol", "unit": "mg/dL", "model": models.LipidReport},
        "HDL Cholesterol": {"key": "hdl_cholesterol", "unit": "mg/dL", "model": models.LipidReport},
        "Triglycerides": {"key": "triglycerides", "unit": "mg/dL", "model": models.LipidReport},
        "Fasting Blood Sugar": {"key": "fasting_blood_sugar", "unit": "mg/dL", "model": models.BloodSugarReport},
    }

    time_series_data = []
    start_date = datetime.now() - relativedelta(months=months)

    # Iterate through each metric we want to track
    for name, config in metrics_to_track.items():
        model = config["model"]
        metric_key = config["key"]

        # Query for all reports of the correct type within the date range
        historical_reports = (
            db.query(model)
            .filter(model.user_id == user_id, model.updated_at >= start_date)
            .order_by(model.updated_at.asc())
            .all()
        )

        # Create a list of data points for the current metric
        series = [
            schemas.TimeSeriesDataPoint(date=report.updated_at, value=getattr(report, metric_key, None))
            for report in historical_reports
        ]

        time_series_data.append(schemas.MetricTimeSeries(name=name, unit=config["unit"], series=series))

    return schemas.HistoricalDataResponse(metrics=time_series_data)