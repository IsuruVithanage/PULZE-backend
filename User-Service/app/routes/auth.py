from fastapi import APIRouter, Depends, HTTPException, status, Response
from sqlalchemy.orm import Session
from .. import schemas, models, auth, database
from ..utils.user_utils import get_user_id_from_header

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=schemas.UserResponse)
def register_user(user: schemas.UserCreate, db: Session = Depends(database.get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = models.User(email=user.email, hashed_password=auth.get_password_hash(user.password))
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


# --- MODIFIED FOR MOBILE APP ---
# Returns the token in the response body instead of a cookie.
@router.post("/login", response_model=schemas.Token)
def login_for_access_token(form_data: schemas.UserLogin, db: Session = Depends(database.get_db)):
    user = db.query(models.User).filter(models.User.email == form_data.email).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")

    access_token = auth.create_access_token(data={"sub": str(user.id)})

    # Return the token directly in the body, which is standard for mobile apps.
    return {"access_token": access_token, "token_type": "bearer"}


# --- MODIFIED FOR STATELESS JWT ---
# Client is responsible for deleting the token. This endpoint is for acknowledgement.
@router.post("/logout")
def logout():
    # For a stateless JWT approach, logout is handled by the client deleting the token.
    # This endpoint can be used for token blocklisting in more complex setups.
    return {"message": "Logout successful. Please discard the token on the client side."}


# --- MODIFIED FOR GATEWAY ARCHITECTURE ---
# Uses the new dependency to get user ID from the header.
@router.get("/users/me", response_model=schemas.UserResponse)
def read_users_me(
    db: Session = Depends(database.get_db),
    current_user_id: int = Depends(get_user_id_from_header) # <-- USE the new dependency
):
    user = db.query(models.User).filter(models.User.id == current_user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# --- MODIFIED FOR GATEWAY ARCHITECTURE ---
# Also uses the new dependency.
@router.patch("/users/me/profile", response_model=schemas.UserResponse)
def update_user_profile(
    profile_data: schemas.UserProfileUpdate,
    db: Session = Depends(database.get_db),
    current_user_id: int = Depends(get_user_id_from_header) # <-- USE the new dependency
):
    user = db.query(models.User).filter(models.User.id == current_user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    user.name = profile_data.name
    user.age = profile_data.age
    user.gender = profile_data.gender
    user.weight_kg = profile_data.weight_kg
    user.height_cm = profile_data.height_cm

    db.commit()
    db.refresh(user)
    return user


# This endpoint is for inter-service communication and can remain as is.
# Another service would call this, providing a specific user_id.
@router.get("/users/{user_id}/profile", response_model=schemas.UserProfileData)
def get_user_profile_for_service(user_id: int, db: Session = Depends(database.get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user



@router.put("/users/me/additional-info", response_model=schemas.AdditionalInfoResponse)
def update_additional_info(
    info_data: schemas.AdditionalInfoUpdate,
    db: Session = Depends(database.get_db),
    current_user_id: int = Depends(get_user_id_from_header)
):
    user = db.query(models.User).filter(models.User.id == current_user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # Update fields if they are provided in the request
    if info_data.health_conditions is not None:
        user.health_conditions = info_data.health_conditions
    if info_data.lifestyle_habits is not None:
        user.lifestyle_habits = info_data.lifestyle_habits

    db.commit()
    db.refresh(user)
    return user