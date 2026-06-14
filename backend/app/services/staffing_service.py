"""
Staffing service — bridges the FastAPI app to the optimisation layer that lives
in ml-pipeline/src/optimization.

It loads the model's empirical CV residuals and the staffing-config assumptions
once at startup, then turns a point forecast into a roster recommendation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[3]
ML_SRC = BASE_DIR / "ml-pipeline" / "src"
MODELS_DIR = BASE_DIR / "ml-pipeline" / "models"
CONFIG_PATH = ML_SRC / "optimization" / "staffing_config.yaml"

# Make the optimisation package importable from the backend.
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))

from optimization import StaffingOptimizer, load_assumptions  # noqa: E402


class StaffingService:
    _optimizer: StaffingOptimizer | None = None

    @classmethod
    def load(cls) -> None:
        residuals_path = MODELS_DIR / "cv_residuals.npy"
        if not residuals_path.exists():
            print("[StaffingService] cv_residuals.npy not found — staffing disabled.")
            return
        residuals = np.load(residuals_path)
        assumptions = load_assumptions(CONFIG_PATH if CONFIG_PATH.exists() else None)
        cls._optimizer = StaffingOptimizer(residuals, assumptions=assumptions)
        print(f"[StaffingService] Loaded | scenarios={assumptions.n_scenarios} | "
              f"coverage={assumptions.coverage:.0%} | "
              f"SLA=≤{assumptions.wait_prob:.0%} wait>{assumptions.wait_target_min:.0f}min")

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._optimizer is not None

    @classmethod
    def recommend(cls, point_forecast: float, high_risk_day: bool = False) -> dict:
        if cls._optimizer is None:
            raise RuntimeError("StaffingService not loaded")
        rec = cls._optimizer.recommend(point_forecast, high_risk_day=high_risk_day)
        out = rec.summary()
        out["assumptions"] = rec.assumptions_register
        return out
