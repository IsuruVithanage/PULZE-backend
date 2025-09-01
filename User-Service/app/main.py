from dotenv import load_dotenv
from fastapi import FastAPI
from .routes import auth
from . import models
from .database import engine
load_dotenv()

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="User Authentication Service")

app.include_router(auth.router, prefix="/api")