"""
Defines all API endpoints related to user medication reminders.
"""
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from .. import schemas, models, database
from ..utils.user_utils import get_user_id_from_header

# Create a new router for reminders
router = APIRouter(prefix="/reminders", tags=["Reminders"])


@router.post("/", response_model=schemas.Reminder)
def create_reminder(
    reminder: schemas.ReminderCreate,
    db: Session = Depends(database.get_db),
    current_user_id: int = Depends(get_user_id_from_header) # CORRECT DEPENDENCY
):
    """
    Creates a new reminder for the authenticated user.
    """
    # Create the reminder using the user ID from the header
    db_reminder = models.Reminder(**reminder.dict(), user_id=current_user_id)
    db.add(db_reminder)
    db.commit()
    db.refresh(db_reminder)
    return db_reminder


@router.get("/", response_model=List[schemas.Reminder])
def get_user_reminders(
    db: Session = Depends(database.get_db),
    current_user_id: int = Depends(get_user_id_from_header) # CORRECT DEPENDENCY
):
    """
    Retrieves all reminders for the authenticated user.
    """
    reminders = db.query(models.Reminder).filter(models.Reminder.user_id == current_user_id).all()
    return reminders


@router.delete("/{reminder_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_reminder(
    reminder_id: int,
    db: Session = Depends(database.get_db),
    current_user_id: int = Depends(get_user_id_from_header) # CORRECT DEPENDENCY
):
    """
    Deletes a specific reminder for the authenticated user.
    """
    db_reminder = db.query(models.Reminder).filter(
        models.Reminder.id == reminder_id,
        models.Reminder.user_id == current_user_id
    ).first()

    if db_reminder is None:
        raise HTTPException(status_code=404, detail="Reminder not found")

    db.delete(db_reminder)
    db.commit()
    # For a 204 response, you shouldn't return a body
    return