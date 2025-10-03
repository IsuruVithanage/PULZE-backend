# app/scheduler.py

from datetime import datetime
from dateutil.relativedelta import relativedelta
from .database import SessionLocal
from . import models
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def delete_old_health_reports():
    """
    Deletes lipid and blood sugar reports older than 8 months.
    """
    db = SessionLocal()
    try:
        # Calculate the cutoff date (8 months ago from today)
        cutoff_date = datetime.now() - relativedelta(months=8)
        logger.info(f"Starting cleanup of reports older than {cutoff_date.strftime('%Y-%m-%d')}.")

        # Delete old lipid reports
        lipid_deleted_count = (
            db.query(models.LipidReport)
            .filter(models.LipidReport.updated_at < cutoff_date)
            .delete(synchronize_session=False)
        )

        # Delete old blood sugar reports
        sugar_deleted_count = (
            db.query(models.BloodSugarReport)
            .filter(models.BloodSugarReport.updated_at < cutoff_date)
            .delete(synchronize_session=False)
        )

        db.commit()
        logger.info(f"Cleanup complete. Deleted {lipid_deleted_count} lipid reports and {sugar_deleted_count} blood sugar reports.")

    except Exception as e:
        logger.error(f"Error during scheduled cleanup: {e}")
        db.rollback()
    finally:
        db.close()