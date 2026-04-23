from datetime import date as Date
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.models import HistoricalRecord, HistoricalResponse
from app.services.model_service import ModelService

router = APIRouter()


@router.get("/historical", response_model=HistoricalResponse)
def historical(
    date_from: Optional[Date] = Query(None),
    date_to: Optional[Date] = Query(None),
    limit: int = Query(365, ge=1, le=2000),
) -> HistoricalResponse:
    if not ModelService.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = ModelService.get_historical(date_from, date_to, limit)
    if df.empty:
        return HistoricalResponse(records=[], total_records=0,
                                  date_from=date_from or Date.today(),
                                  date_to=date_to or Date.today())

    records = [
        HistoricalRecord(
            date=r["date"].date(),
            patients=int(r["patients"]),
            day_name=r["day_name"],
            month=int(r["month"]),
            is_public_holiday=int(r["is_public_holiday"]),
            is_school_holiday=int(r["is_school_holiday"]),
            is_flu_peak=int(r["is_flu_peak"]),
            is_hayfever_season=int(r["is_hayfever_season"]),
            weather_type=r["weather_type"],
            temperature=float(r["temperature"]),
            pollen_index=int(r["pollen_index"]),
        )
        for _, r in df.iterrows()
    ]

    return HistoricalResponse(
        records=records,
        total_records=len(records),
        date_from=df["date"].min().date(),
        date_to=df["date"].max().date(),
    )
