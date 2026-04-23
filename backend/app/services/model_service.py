from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

BASE_DIR   = Path(__file__).resolve().parents[3]
MODELS_DIR = BASE_DIR / "ml-pipeline" / "models"
DATA_DIR   = BASE_DIR / "data"

# Victorian public holidays — matches generator
VIC_PUBLIC_HOLIDAYS = {
    "2023-01-01","2023-01-02","2023-01-26","2023-03-13","2023-04-07","2023-04-08",
    "2023-04-09","2023-04-10","2023-04-25","2023-06-12","2023-11-07","2023-12-25","2023-12-26",
    "2024-01-01","2024-01-26","2024-03-11","2024-03-29","2024-03-30","2024-03-31",
    "2024-04-01","2024-04-25","2024-06-10","2024-11-05","2024-12-25","2024-12-26",
    "2025-01-01","2025-01-27","2025-03-10","2025-04-18","2025-04-19","2025-04-20",
    "2025-04-21","2025-04-25","2025-06-09","2025-11-04","2025-12-25","2025-12-26",
}


def _is_vic_school_holiday(d: pd.Timestamp) -> int:
    year = d.year
    ranges = [
        (f"{year}-04-01", f"{year}-04-18"),
        (f"{year}-06-24", f"{year}-07-10"),
        (f"{year}-09-16", f"{year}-10-02"),
        (f"{year}-12-18", f"{year}-12-31"),
    ]
    return int(any(pd.Timestamp(s) <= d <= pd.Timestamp(e) for s, e in ranges))


def _default_temp(month: int) -> float:
    defaults = {1: 20.5, 2: 20.5, 3: 18.5, 4: 15.5, 5: 12.0, 6: 10.0,
                7: 9.5, 8: 11.0, 9: 12.5, 10: 15.0, 11: 17.0, 12: 19.5}
    return defaults.get(month, 16.0)


def _default_pollen(month: int) -> int:
    base = {1: 2, 2: 2, 3: 1, 4: 1, 5: 0, 6: 0,
            7: 0, 8: 1, 9: 3, 10: 7, 11: 9, 12: 6}
    return base.get(month, 3)


class ModelService:
    _model = None
    _features: list[str] = []
    _best_model_name: str = ""
    _historical_df: pd.DataFrame | None = None
    _model_results: dict = {}
    _residual_std: float = 0.0
    _cv_residuals: np.ndarray | None = None
    _xgb_log_offset: float = 0.0
    _tiers = {"low_max": 80, "medium_max": 150}

    @classmethod
    def load(cls):
        with open(MODELS_DIR / "features.json") as f:
            meta = json.load(f)
        cls._features = meta["features"]
        cls._best_model_name = meta["best_model"]
        cls._residual_std = float(meta.get("cv_residual_std", 25.0))
        cls._tiers = meta.get("tiers", cls._tiers)

        if cls._best_model_name == "xgboost":
            cls._model = xgb.XGBRegressor()
            cls._model.load_model(str(MODELS_DIR / "xgboost.json"))
        else:
            cls._model = joblib.load(MODELS_DIR / "random_forest.joblib")

        engineered_path = DATA_DIR / "processed" / "clinic_patients_engineered.csv"
        if engineered_path.exists():
            cls._historical_df = pd.read_csv(engineered_path, parse_dates=["date"])

        results_path = MODELS_DIR / "model_results.csv"
        if results_path.exists():
            res = pd.read_csv(results_path)
            cls._model_results = res.set_index("Model").to_dict(orient="index")

        cv_path = MODELS_DIR / "cv_residuals.npy"
        if cv_path.exists():
            cls._cv_residuals = np.load(cv_path)

        # Calibrate log-offset for XGBoost 2.x (base_score not added by predict())
        if cls._historical_df is not None and cls._best_model_name == "xgboost":
            train = cls._historical_df.iloc[:-90]
            X_train = train[cls._features]
            raw = cls._model.predict(X_train)
            log_actual = np.log1p(train["patients"].values)
            cls._xgb_log_offset = float((log_actual - raw).mean())

        print(f"[ModelService] Loaded: {cls._best_model_name} | features: {len(cls._features)} | "
              f"residual_std: {cls._residual_std:.1f} | xgb_log_offset: {cls._xgb_log_offset:.4f}")

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._model is not None

    @classmethod
    def _tier(cls, n: float) -> tuple[str, str]:
        if n < cls._tiers["low_max"]:
            return "Low", "2 doctors, 3 nurses — standard roster"
        if n <= cls._tiers["medium_max"]:
            return "Medium", "3 doctors, 4 nurses, extra triage nurse"
        return "High", "4+ doctors, surge protocol, ED overflow coordination"

    @classmethod
    def predict(cls, feature_row: pd.DataFrame) -> dict:
        raw = cls._model.predict(feature_row)
        if cls._best_model_name == "xgboost":
            point = float(np.expm1(raw[0] + cls._xgb_log_offset))
        else:
            point = float(np.expm1(raw[0]))
        point = max(0.0, point)

        std = cls._residual_std or 25.0
        lower_80 = max(0, int(point - 1.28 * std))
        upper_80 = int(point + 1.28 * std)
        lower_95 = max(0, int(point - 1.96 * std))
        upper_95 = int(point + 1.96 * std)

        tier, plan = cls._tier(point)

        # Confidence from test R² (best-model row)
        best_row = cls._model_results.get("XGBoost") or cls._model_results.get(cls._best_model_name.title()) or {}
        r2 = float(best_row.get("R2", 0.0))
        confidence = max(0.0, min(1.0, r2))

        return {
            "predicted_patients": int(round(point)),
            "lower_80": lower_80,
            "upper_80": upper_80,
            "lower_95": lower_95,
            "upper_95": upper_95,
            "staffing_tier": tier,
            "tier_plan": plan,
            "confidence": round(confidence, 3),
            "model_used": cls._best_model_name,
        }

    @classmethod
    def get_historical(cls, date_from=None, date_to=None, limit: int = 365) -> pd.DataFrame:
        if cls._historical_df is None:
            return pd.DataFrame()
        df = cls._historical_df.copy()
        if date_from:
            df = df[df["date"] >= pd.Timestamp(date_from)]
        if date_to:
            df = df[df["date"] <= pd.Timestamp(date_to)]
        return df.tail(limit)

    @classmethod
    def get_feature_importances(cls, top_n: int = 10) -> list[dict]:
        if cls._model is None or not hasattr(cls._model, "feature_importances_"):
            return []
        importances = cls._model.feature_importances_
        pairs = sorted(zip(cls._features, importances), key=lambda x: x[1], reverse=True)
        return [{"feature": f, "importance": round(float(v), 4)} for f, v in pairs[:top_n]]

    @classmethod
    def get_model_results(cls) -> dict:
        return cls._model_results

    @classmethod
    def build_feature_row(cls, req, historical_df: pd.DataFrame) -> pd.DataFrame:
        """Build a single-row feature DataFrame matching the model's expected feature list."""
        target = pd.Timestamp(req.date)
        dow   = target.dayofweek
        month = target.month
        day   = target.day
        doy   = target.dayofyear
        woy   = int(target.isocalendar()[1])

        row = {
            "year": target.year, "month": month, "day": day, "day_of_week": dow,
            "week_of_year": woy, "quarter": (month - 1) // 3 + 1, "doy": doy,
            "dow_sin":   math.sin(2 * math.pi * dow / 7),
            "dow_cos":   math.cos(2 * math.pi * dow / 7),
            "month_sin": math.sin(2 * math.pi * (month - 1) / 12),
            "month_cos": math.cos(2 * math.pi * (month - 1) / 12),
            "doy_sin":   math.sin(2 * math.pi * doy / 365),
            "doy_cos":   math.cos(2 * math.pi * doy / 365),
            "woy_sin":   math.sin(2 * math.pi * woy / 52),
            "woy_cos":   math.cos(2 * math.pi * woy / 52),
            "is_weekend": int(dow >= 5),
            "is_monday":  int(dow == 0),
        }

        # Calendar — auto-detect if not provided
        is_ph = req.is_public_holiday if req.is_public_holiday is not None else int(str(target.date()) in VIC_PUBLIC_HOLIDAYS)
        prev_day = target - pd.Timedelta(days=1)
        is_day_after_ph = int(str(prev_day.date()) in VIC_PUBLIC_HOLIDAYS)
        is_sh = req.is_school_holiday if req.is_school_holiday is not None else _is_vic_school_holiday(target)
        row["is_public_holiday"] = is_ph
        row["is_day_after_public_holiday"] = is_day_after_ph
        row["is_school_holiday"] = is_sh

        # Weather
        temp = req.temperature if req.temperature is not None else _default_temp(month)
        pollen = req.pollen_index if req.pollen_index is not None else _default_pollen(month)
        row["temperature"]    = float(temp)
        row["precipitation"]  = float(req.precipitation or 0.0)
        row["humidity"]       = int(req.humidity or 65)
        row["pollen_index"]   = int(pollen)
        row["temp_extreme_hot"]  = int(temp > 35)
        row["temp_extreme_cold"] = int(temp < 5)

        weather_type = req.weather_type or "Partly Cloudy"
        row["weather_encoded"] = {"Sunny": 3, "Partly Cloudy": 2, "Cloudy": 1, "Rainy": 0}.get(weather_type, 2)
        for wt in ["Cloudy", "Partly Cloudy", "Rainy", "Sunny"]:
            row[f"weather_{wt}"] = int(weather_type == wt)

        # Season
        def _season(m):
            if m in (12, 1, 2): return "Summer"
            if m in (3, 4, 5):  return "Autumn"
            if m in (6, 7, 8):  return "Winter"
            return "Spring"
        season = _season(month)
        for s in ["Autumn", "Spring", "Summer", "Winter"]:
            row[f"season_{s}"] = int(season == s)

        # Epi drivers
        row["is_flu_season"] = int(month in (6, 7, 8))
        row["is_flu_peak"] = req.is_flu_peak_override if req.is_flu_peak_override is not None else int(
            month == 7 or (month == 8 and day <= 15)
        )
        row["is_hayfever_season"] = int(month in (10, 11))
        row["is_thunderstorm_asthma"] = int(req.is_thunderstorm_asthma or 0)

        row["illness_driver_count"] = (row["is_flu_season"] + row["is_hayfever_season"]
                                       + row["is_day_after_public_holiday"]
                                       + row["temp_extreme_hot"] + row["temp_extreme_cold"])

        # Interactions
        row["ph_x_monday"] = row["is_public_holiday"] * row["is_monday"]
        row["flu_peak_x_rainy"] = row["is_flu_peak"] * int(weather_type == "Rainy")
        row["hayfever_x_high_pollen"] = row["is_hayfever_season"] * int(row["pollen_index"] >= 7)
        row["school_holiday_x_weekend"] = row["is_school_holiday"] * row["is_weekend"]
        row["extreme_cold_x_flu_season"] = row["temp_extreme_cold"] * row["is_flu_season"]
        row["day_after_ph_x_monday"] = row["is_day_after_public_holiday"] * row["is_monday"]

        # Lag / rolling features from history
        past = historical_df[historical_df["date"] < target].sort_values("date")
        series = past["patients"] if "patients" in past.columns else pd.Series(dtype=float)
        fallback = float(series.mean()) if len(series) else 100.0

        def lag(n):
            return float(series.iloc[-n]) if len(series) >= n else fallback

        def roll(n, stat):
            s = series.iloc[-n:] if len(series) >= n else series
            if len(s) == 0: return fallback
            if stat == "mean": return float(s.mean())
            if stat == "std":  return float(s.std()) if len(s) > 1 else 0.0
            if stat == "max":  return float(s.max())
            if stat == "min":  return float(s.min())
            return fallback

        for l in [1, 2, 3, 7, 14, 21, 28, 60, 90]:
            row[f"patients_lag_{l}"] = lag(l)
        for k in [7, 14, 21, 28]:
            row[f"lag_{k}_same_dow"] = lag(k)
        row["mean_last_4_same_dow"] = float(np.mean([lag(k) for k in [7, 14, 21, 28]]))

        for w in [3, 7, 14, 30, 60, 90]:
            row[f"rolling_mean_{w}d"] = roll(w, "mean")
            row[f"rolling_std_{w}d"]  = roll(w, "std")
            row[f"rolling_max_{w}d"]  = roll(w, "max")
            row[f"rolling_min_{w}d"]  = roll(w, "min")
        for s in [7, 14, 30]:
            row[f"ewma_{s}d"] = roll(s, "mean")

        feature_row = {f: row.get(f, 0.0) for f in cls._features}
        return pd.DataFrame([feature_row])
