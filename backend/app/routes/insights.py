from fastapi import APIRouter, HTTPException

from app.models import InsightsResponse, ModelMetrics
from app.services.model_service import ModelService

router = APIRouter()


@router.get("/insights", response_model=InsightsResponse)
def insights() -> InsightsResponse:
    if not ModelService.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = ModelService.get_model_results()
    best_model_label = ModelService._best_model_name.title() if ModelService._best_model_name.islower() else ModelService._best_model_name
    best = results.get("XGBoost") or results.get(best_model_label) or {}

    metrics = ModelMetrics(
        mae=float(best.get("MAE", 0.0)),
        rmse=float(best.get("RMSE", 0.0)),
        r2=float(best.get("R2", 0.0)),
        mape=float(best.get("MAPE%", 0.0)),
        bias=float(best.get("Bias", 0.0)),
    )

    df = ModelService._historical_df
    training_records = int((df["date"] < df["date"].max() - __import__("pandas").Timedelta(days=90)).sum()) if df is not None else 0

    return InsightsResponse(
        best_model=ModelService._best_model_name,
        metrics=metrics,
        feature_count=len(ModelService._features),
        training_records=training_records,
        top_features=ModelService.get_feature_importances(15),
        all_model_results=results,
    )
