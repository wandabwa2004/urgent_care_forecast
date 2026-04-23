from fastapi import APIRouter, HTTPException

from app.models import PredictionRequest, PredictionResponse
from app.services.model_service import ModelService
from app.services.supabase_client import log_prediction

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest) -> PredictionResponse:
    if not ModelService.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    feature_row = ModelService.build_feature_row(req, ModelService._historical_df)
    result = ModelService.predict(feature_row)

    response = PredictionResponse(date=req.date, **result)

    log_prediction({
        "date": str(req.date),
        "predicted_patients": response.predicted_patients,
        "staffing_tier": response.staffing_tier,
        "model_used": response.model_used,
        "temperature": req.temperature,
        "weather_type": req.weather_type,
        "is_public_holiday": req.is_public_holiday,
        "is_school_holiday": req.is_school_holiday,
        "pollen_index": req.pollen_index,
    })

    return response
