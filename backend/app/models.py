from datetime import date as Date
from typing import Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    date: Date = Field(..., description="Date to predict patient volume for")

    temperature: Optional[float] = Field(None, description="Expected daily mean temperature (°C)")
    precipitation: Optional[float] = Field(0.0, description="Expected precipitation (mm)")
    weather_type: Optional[str] = Field("Partly Cloudy", description="Sunny | Partly Cloudy | Cloudy | Rainy")
    humidity: Optional[int] = Field(65, description="Relative humidity 0-100")
    pollen_index: Optional[int] = Field(None, description="Pollen index 0-10 (auto-defaulted by month)")

    # Calendar flags — auto-detected if omitted
    is_public_holiday: Optional[int] = Field(None, description="1 if Victorian public holiday")
    is_school_holiday: Optional[int] = Field(None, description="1 if Victorian school holiday")

    # Optional overrides for epi drivers (defaults are month-derived)
    is_flu_peak_override: Optional[int] = Field(None, description="Force flu-peak flag")
    is_thunderstorm_asthma: Optional[int] = Field(0, description="Manual flag for thunderstorm-asthma event")


class PredictionResponse(BaseModel):
    date: Date
    predicted_patients: int
    lower_80: int
    upper_80: int
    lower_95: int
    upper_95: int
    staffing_tier: str          # Low | Medium | High
    tier_plan: str              # Short description of staffing plan
    confidence: float
    model_used: str


class HistoricalRecord(BaseModel):
    date: Date
    patients: int
    day_name: str
    month: int
    is_public_holiday: int
    is_school_holiday: int
    is_flu_peak: int
    is_hayfever_season: int
    weather_type: str
    temperature: float
    pollen_index: int


class HistoricalResponse(BaseModel):
    records: list[HistoricalRecord]
    total_records: int
    date_from: Date
    date_to: Date


class ModelMetrics(BaseModel):
    mae: float
    rmse: float
    r2: float
    mape: float
    bias: float


class InsightsResponse(BaseModel):
    best_model: str
    metrics: ModelMetrics
    feature_count: int
    training_records: int
    top_features: list[dict]
    all_model_results: dict
