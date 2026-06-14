"""
Stage 1 — requirements generation.

Given a point forecast for the day and the model's empirical CV residuals, we:

  1. Sample N plausible actual-demand scenarios: D_s = max(0, forecast + e_s),
     where e_s is drawn (with replacement) from the residuals (actual - pred).
  2. For each scenario, distribute demand across the day and size each role per
     hour with Erlang-C so the wait-time SLA and utilisation cap hold.
  3. Staff each (hour, role) to the `coverage` quantile of required servers
     across scenarios — i.e. enough for `coverage` fraction of plausible days.

The result is a per-hour, per-role requirement that already accounts for both
intraday shape and forecast uncertainty, before any rostering happens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .assumptions import StaffingAssumptions
from .demand_profile import hourly_arrivals
from .queueing import min_servers_for_sla, prob_wait_exceeds, expected_wait_hours, utilisation


@dataclass
class HourRoleRequirement:
    hour: int
    role: str
    required: int           # servers needed at the coverage quantile
    arrivals_planned: float  # arrival rate at the planned (coverage-quantile) demand


def sample_demand(point_forecast: float, residuals: np.ndarray,
                  a: StaffingAssumptions) -> np.ndarray:
    """Draw demand scenarios by resampling empirical residuals."""
    rng = np.random.default_rng(a.random_seed)
    draws = rng.choice(residuals, size=a.n_scenarios, replace=True)
    return np.maximum(0.0, point_forecast + draws)


def _required_per_hour_for_demand(daily_total: float, a: StaffingAssumptions
                                  ) -> Dict[int, Dict[str, int]]:
    """For one demand value, the Erlang-C server requirement per hour per role."""
    arrivals = hourly_arrivals(daily_total, a)
    t_hours = a.wait_target_min / 60.0
    out: Dict[int, Dict[str, int]] = {}
    for hour, lam in arrivals.items():
        out[hour] = {}
        for r in a.roles:
            out[hour][r.name] = min_servers_for_sla(
                lam=lam, mu=r.service_rate, t_hours=t_hours,
                wait_prob=a.wait_prob, target_utilisation=a.target_utilisation,
                floor=r.min_on_duty,
            )
    return out


def planned_demand(point_forecast: float, residuals: np.ndarray,
                   a: StaffingAssumptions) -> float:
    """The coverage-quantile (e.g. P90) demand we size the roster to."""
    return float(np.quantile(sample_demand(point_forecast, residuals, a), a.coverage))


def build_requirements(point_forecast: float, residuals: np.ndarray,
                       a: StaffingAssumptions) -> List[HourRoleRequirement]:
    """Per-hour, per-role requirement sized to the coverage-quantile demand day.

    We resample the model's empirical residuals to form the demand distribution,
    take its `coverage` quantile (the "plan-to" day — robust to forecast error on
    that fraction of days), distribute it across the day, and size each role per
    hour with Erlang-C so the wait-time SLA and utilisation cap hold *at that
    demand*. Because Erlang-C requirement is monotone in demand, meeting the SLA
    at the P90 day means meeting it on every lighter day too.
    """
    planned = planned_demand(point_forecast, residuals, a)
    req = _required_per_hour_for_demand(planned, a)
    planned_arrivals = hourly_arrivals(planned, a)

    requirements: List[HourRoleRequirement] = []
    for h in a.open_hours:
        for r in a.roles:
            requirements.append(HourRoleRequirement(
                hour=h, role=r.name, required=max(req[h][r.name], r.min_on_duty),
                arrivals_planned=round(planned_arrivals[h], 2),
            ))
    return requirements


def requirement_distribution(point_forecast: float, residuals: np.ndarray,
                             a: StaffingAssumptions, role: str, hour: int) -> np.ndarray:
    """Distribution of required servers for one (role, hour) across demand
    scenarios — used by the notebook to visualise how uncertainty drives the
    chosen coverage quantile."""
    scenarios = sample_demand(point_forecast, residuals, a)
    t_hours = a.wait_target_min / 60.0
    r = a.role(role)
    weights = dict(zip(a.open_hours, a.normalised_weights()))
    out = []
    cache: Dict[int, int] = {}
    for d in scenarios:
        lam_key = int(round(d * weights[hour] * 100))  # cache on rounded arrival rate
        if lam_key not in cache:
            lam = d * weights[hour]
            cache[lam_key] = min_servers_for_sla(
                lam=lam, mu=r.service_rate, t_hours=t_hours, wait_prob=a.wait_prob,
                target_utilisation=a.target_utilisation, floor=r.min_on_duty)
        out.append(cache[lam_key])
    return np.array(out)


def requirement_grid(requirements: List[HourRoleRequirement]
                     ) -> Dict[str, Dict[int, int]]:
    """Reshape requirements into grid[role][hour] = required count."""
    grid: Dict[str, Dict[int, int]] = {}
    for req in requirements:
        grid.setdefault(req.role, {})[req.hour] = req.required
    return grid


def service_metrics_at(grid: Dict[str, Dict[int, int]], daily_total: float,
                       a: StaffingAssumptions) -> List[dict]:
    """Achieved wait/utilisation per hour per role at a given demand, used to
    confirm the SLA is actually met by the chosen staffing."""
    arrivals = hourly_arrivals(daily_total, a)
    t_hours = a.wait_target_min / 60.0
    rows: List[dict] = []
    for r in a.roles:
        for h in a.open_hours:
            lam = arrivals[h]
            c = grid[r.name][h]
            rows.append({
                "hour": h,
                "role": r.name,
                "servers": c,
                "arrivals": round(lam, 2),
                "p_wait_gt_target": round(prob_wait_exceeds(c, lam, r.service_rate, t_hours), 3),
                "expected_wait_min": round(expected_wait_hours(c, lam, r.service_rate) * 60.0, 1),
                "utilisation": round(utilisation(c, lam, r.service_rate), 3),
            })
    return rows
