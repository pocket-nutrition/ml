import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.model import load_model
from app.router import router

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="pocket-nutrition-ml", version="2.0.0", lifespan=lifespan)
app.include_router(router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
