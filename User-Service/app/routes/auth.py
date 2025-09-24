"""
Defines all API endpoints related to user authentication and profile management.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from .. import schemas, models, auth, database
from ..utils.user_utils import get_user_id_from_header

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=schemas.UserResponse)
def register_user(user: schemas.UserCreate, db: Session = Depends(database.get_db)):
    """
    Registers a new user in the database.

    Args:
        user (schemas.UserCreate): The user's registration data (email, password).
        db (Session): The database session dependency.

    Raises:
        HTTPException: 400 if the email is already registered.

    Returns:
        models.User: The newly created user object.
    """
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create a new user instance with a hashed password
    hashed_password = auth.get_password_hash(user.password)
    new_user = models.User(email=user.email, hashed_password=hashed_password)

    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@router.post("/login", response_model=schemas.Token)
def login_for_access_token(form_data: schemas.UserLogin, db: Session = Depends(database.get_db)):
    """
    Authenticates a user and returns a JWT access token.

    Args:
        form_data (schemas.UserLogin): The user's login credentials.
        db (Session): The database session dependency.

    Raises:
        HTTPException: 401 if the credentials are incorrect.

    Returns:
        dict: A dictionary containing the access token and token type.
    """
    user = db.query(models.User).filter(models.User.email == form_data.email).first()

    # Verify user existence and password correctness
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    # Create a new JWT access token
    access_token = auth.create_access_token(data={"sub": str(user.id)})

    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/logout")
def logout():
    """
    Provides a formal endpoint for logging out.

    In a stateless JWT system, the client is responsible for discarding the token.
    This endpoint serves as an acknowledgment.
    """
    return {"message": "Logout successful. Please discard the token on the client side."}


@router.get("/users/me", response_model=schemas.UserResponse)
def read_users_me(
    db: Session = Depends(database.get_db),
    current_user_id: int = Depends(get_user_id_from_header)
):
    """
    Retrieves the profile of the currently authenticated user.

    Args:
        db (Session): The database session dependency.
        current_user_id (int): The user ID extracted from the X-User-ID header.

    Raises:
        HTTPException: 404 if the user is not found.

    Returns:
        models.User: The authenticated user's profile data.
    """
    user = db.query(models.User).filter(models.User.id == current_user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.patch("/users/me/profile", response_model=schemas.UserResponse)
def update_user_profile(
    profile_data: schemas.UserProfileUpdate,
    db: Session = Depends(database.get_db),
    current_user_id: int = Depends(get_user_id_from_header)
):
    """
    Updates the basic profile of the currently authenticated user.

    Args:
        profile_data (schemas.UserProfileUpdate): The new profile data to update.
        db (Session): The database session dependency.
        current_user_id (int): The user ID from the X-User-ID header.

    Raises:
        HTTPException: 404 if the user is not found.

    Returns:
        models.User: The updated user object.
    """
    user = db.query(models.User).filter(models.User.id == current_user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # Update user attributes with the new data
    user.name = profile_data.name
    user.age = profile_data.age
    user.gender = profile_data.gender
    user.weight_kg = profile_data.weight_kg
    user.height_cm = profile_data.height_cm

    db.commit()
    db.refresh(user)
    return user


@router.put("/users/me/additional-info", response_model=schemas.AdditionalInfoResponse)
def update_additional_info(
    info_data: schemas.AdditionalInfoUpdate,
    db: Session = Depends(database.get_db),
    current_user_id: int = Depends(get_user_id_from_header)
):
    """
    Updates the additional health information for the authenticated user.

    Args:
        info_data (schemas.AdditionalInfoUpdate): New health/lifestyle data.
        db (Session): The database session dependency.
        current_user_id (int): The user ID from the X-User-ID header.

    Raises:
        HTTPException: 404 if the user is not found.

    Returns:
        models.User: The updated user object.
    """
    user = db.query(models.User).filter(models.User.id == current_user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # Update fields only if they are provided in the request
    if info_data.health_conditions is not None:
        user.health_conditions = info_data.health_conditions
    if info_data.lifestyle_habits is not None:
        user.lifestyle_habits = info_data.lifestyle_habits

    db.commit()
    db.refresh(user)
    return user


@router.get("/users/{user_id}/profile", response_model=schemas.UserProfileData)
def get_user_profile_for_service(user_id: int, db: Session = Depends(database.get_db)):
    """
    Retrieves a user's profile data by their ID.

    This endpoint is intended for secure, internal, inter-service communication.
    """
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user