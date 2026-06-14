from fastapi import APIRouter, HTTPException

from app.models import PredictionRequest, StaffingResponse
from app.services.model_service import ModelService
from app.services.staffing_service import StaffingService

router = APIRouter()


@router.post("/staffing", response_model=StaffingResponse)
def staffing(req: PredictionRequest) -> StaffingResponse:
    """Forecast the day's patient volume, then recommend the minimum-cost roster
    that meets the wait-time service level. Assumptions are echoed back."""
    if not ModelService.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not StaffingService.is_loaded():
        raise HTTPException(status_code=503, detail="Staffing optimiser not loaded")
    if ModelService._historical_df is None:
        raise HTTPException(
            status_code=503,
            detail="Historical dataset not loaded — run the feature-engineering "
                   "pipeline to produce data/processed/clinic_patients_engineered.csv",
        )

    # Stage 0 — point forecast for the requested day.
    feature_row = ModelService.build_feature_row(req, ModelService._historical_df)
    prediction = ModelService.predict(feature_row)
    point = prediction["predicted_patients"]

    # A thunderstorm-asthma flag is the known shock driver that warrants standby.
    high_risk = bool(req.is_thunderstorm_asthma)

    rec = StaffingService.recommend(point, high_risk_day=high_risk)

    return StaffingResponse(
        date=req.date,
        predicted_patients=point,
        planned_demand=rec["planned_demand"],
        p95_demand=rec["p95_demand"],
        tail_demand=rec["tail_demand"],
        bodies_to_roster=rec["bodies_to_roster"],
        peak_concurrent=rec["peak_concurrent"],
        headcount_by_shift=rec["headcount_by_shift"],
        daily_cost=rec["daily_cost"],
        shortfall=rec["shortfall"],
        achieved_sla=rec["achieved_sla"],
        peak_hour=rec["peak_hour"],
        standby_recommended=rec["standby_recommended"],
        solver_status=rec["solver_status"],
        assumptions=rec["assumptions"],
    )
