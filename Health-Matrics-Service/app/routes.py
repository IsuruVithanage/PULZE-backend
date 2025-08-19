from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import os, tempfile, requests
from urllib.parse import urlparse

from app import schemas, models
from app.database import SessionLocal, engine
from app.utils import ocr, parser

from pydantic import BaseModel, HttpUrl

models.Base.metadata.create_all(bind=engine)
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

