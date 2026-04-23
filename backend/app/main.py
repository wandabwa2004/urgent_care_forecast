from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import historical, insights, predict
from app.services.model_service import ModelService


@asynccontextmanager
async def lifespan(app: FastAPI):
    ModelService.load()
    yield


app = FastAPI(
    title="Urgent Care Demand Forecast API",
    description="Daily patient-volume forecasts for a Melbourne walk-in clinic.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router,    prefix="/api", tags=["Predictions"])
app.include_router(historical.router, prefix="/api", tags=["Historical"])
app.include_router(insights.router,   prefix="/api", tags=["Insights"])


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Urgent Care Demand Forecast API"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "model_loaded": ModelService.is_loaded()}
