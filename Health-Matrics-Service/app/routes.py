from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from sqlalchemy.orm import Session
from app import schemas, models
from app.database import SessionLocal, engine
from app.utils import ocr, parser

# Ensure tables are created (or use Alembic)
models.Base.metadata.create_all(bind=engine)

router = APIRouter()

# Dependency to get a DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/upload", response_model=schemas.ReportResponse)
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Save file to a temporary location
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    try:
        extracted_text = ocr.get_text_from_any_file(file_location)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # Optionally delete the temporary file
        import os
        os.remove(file_location)

    # Parse the extracted text
    report_data = parser.parse_report(extracted_text)

    # Create Report object and save to DB
    new_report = models.Report(**report_data)
    db.add(new_report)
    db.commit()
    db.refresh(new_report)

    return new_report
