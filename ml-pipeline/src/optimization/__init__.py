"""
Staffing-optimisation layer for the Melbourne urgent-care demand forecast.

Two stages sit on top of the daily patient-volume forecast:

  Stage 1 — Requirements:  intraday arrival profile + Erlang-C queueing +
            scenario sampling of the model's empirical residuals, producing the
            minimum role counts per hour that meet a wait-time service level
            on a target fraction of plausible days.

  Stage 2 — Rostering:     OR-Tools CP-SAT picks the minimum-cost legal roster
            (counts of each role per shift) that covers those requirements.

Every assumption is explicit and lives in `StaffingAssumptions` /
`staffing_config.yaml`, and is echoed back with each recommendation.
"""

from .assumptions import StaffingAssumptions, load_assumptions
from .optimizer import StaffingOptimizer, DayRecommendation

__all__ = [
    "StaffingAssumptions",
    "load_assumptions",
    "StaffingOptimizer",
    "DayRecommendation",
]
