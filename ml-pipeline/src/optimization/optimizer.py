"""
StaffingOptimizer — the public entry point.

Ties Stage 1 (requirements) and Stage 2 (rostering) together into a single
per-day recommendation the rostering manager can act on, with every assumption
echoed back and the achieved service level reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .assumptions import StaffingAssumptions, load_assumptions
from .demand_profile import hourly_arrivals, peak_hour
from .requirements import build_requirements, requirement_grid, service_metrics_at, sample_demand
from .rostering import optimise_roster, RosterResult


@dataclass
class DayRecommendation:
    point_forecast: float
    planned_demand: float                  # coverage-quantile demand the roster is sized for
    p95_demand: float
    tail_demand: float                     # P99 demand — the bad-day scenario standby guards against
    roster: Dict[str, Dict[str, int]]      # roster[role][shift] = headcount
    bodies_to_roster: Dict[str, int]       # total shift-assignments per role across the day
    peak_concurrent: Dict[str, int]        # most of each role on the floor at once
    headcount_by_shift: Dict[str, Dict[str, int]]
    daily_cost: float
    shortfall: int
    achieved_sla: Dict[str, float]         # role -> worst-hour P(wait > target) at planned demand
    peak_hour: int
    standby_recommended: bool
    requirement_grid: Dict[str, Dict[int, int]]
    coverage_grid: Dict[str, Dict[int, int]]
    service_metrics: List[dict]
    assumptions_register: List[dict]
    solver_status: str

    def summary(self) -> Dict:
        """Compact, serialisable summary for an API response."""
        return {
            "point_forecast": round(self.point_forecast, 1),
            "planned_demand": round(self.planned_demand, 1),
            "p95_demand": round(self.p95_demand, 1),
            "tail_demand": round(self.tail_demand, 1),
            "roster": self.roster,
            "bodies_to_roster": self.bodies_to_roster,
            "peak_concurrent": self.peak_concurrent,
            "headcount_by_shift": self.headcount_by_shift,
            "daily_cost": round(self.daily_cost, 2),
            "shortfall": self.shortfall,
            "achieved_sla": {k: round(v, 3) for k, v in self.achieved_sla.items()},
            "peak_hour": self.peak_hour,
            "standby_recommended": self.standby_recommended,
            "solver_status": self.solver_status,
        }


class StaffingOptimizer:
    def __init__(self, residuals: np.ndarray,
                 assumptions: Optional[StaffingAssumptions] = None,
                 config_path: Optional[str | Path] = None):
        if residuals is None or len(residuals) == 0:
            raise ValueError("Residuals are required for scenario-based requirements.")
        self.residuals = np.asarray(residuals, dtype=float)
        self.a = assumptions or load_assumptions(config_path)

    # ----------------------------------------------------------------------
    def recommend(self, point_forecast: float, high_risk_day: bool = False) -> DayRecommendation:
        """Recommend a roster for one day.

        `high_risk_day` flags a known shock driver (e.g. thunderstorm-asthma
        conditions: high pollen + spring storm). Because the demand distribution
        is built from a single, homoscedastic residual pool, the data-driven tail
        is the same relative size every day and cannot, on its own, tell a busy
        day from a genuinely dangerous one — so the standby recommendation keys
        off this explicit risk flag instead.
        """
        a = self.a
        scenarios = sample_demand(point_forecast, self.residuals, a)
        planned_demand = float(np.quantile(scenarios, a.coverage))
        p95_demand = float(np.quantile(scenarios, 0.95))
        tail_demand = float(np.quantile(scenarios, 0.99))

        # Stage 1 — requirements at the coverage quantile.
        reqs = build_requirements(point_forecast, self.residuals, a)
        grid = requirement_grid(reqs)

        # Stage 2 — min-cost roster covering those requirements.
        roster: RosterResult = optimise_roster(grid, a)

        # Confirm the achieved service level at the planned demand.
        metrics = service_metrics_at(roster.coverage, planned_demand, a)
        achieved_sla: Dict[str, float] = {}
        for r in a.roles:
            role_rows = [m for m in metrics if m["role"] == r.name]
            achieved_sla[r.name] = max((m["p_wait_gt_target"] for m in role_rows), default=0.0)

        peak_concurrent = {
            r.name: max(roster.coverage[r.name].values()) for r in a.roles
        }
        bodies_to_roster = {
            r.name: sum(roster.counts[r.name].values()) for r in a.roles
        }

        # Standby keys off a known shock driver (see docstring). The data-driven
        # tail (tail_demand) is reported for transparency but is homoscedastic.
        standby = bool(high_risk_day)

        return DayRecommendation(
            point_forecast=point_forecast,
            planned_demand=planned_demand,
            p95_demand=p95_demand,
            tail_demand=tail_demand,
            roster=roster.counts,
            bodies_to_roster=bodies_to_roster,
            peak_concurrent=peak_concurrent,
            headcount_by_shift=roster.headcount_by_shift,
            daily_cost=roster.total_cost,
            shortfall=roster.shortfall,
            achieved_sla=achieved_sla,
            peak_hour=peak_hour(planned_demand, a),
            standby_recommended=standby,
            requirement_grid=grid,
            coverage_grid=roster.coverage,
            service_metrics=metrics,
            assumptions_register=a.register(),
            solver_status=roster.status,
        )

    # ----------------------------------------------------------------------
    def recommend_week(self, forecasts: Dict[str, float]) -> Dict[str, DayRecommendation]:
        """Recommend for a date -> point-forecast mapping."""
        return {day: self.recommend(fc) for day, fc in forecasts.items()}
