"""
Tests for the staffing-optimisation layer.

Run from the ml-pipeline directory:
    ../.venv/bin/python -m pytest tests/test_staffing.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from optimization import StaffingOptimizer, load_assumptions               # noqa: E402
from optimization.assumptions import StaffingAssumptions                   # noqa: E402
from optimization.queueing import (                                        # noqa: E402
    erlang_c, prob_wait_exceeds, expected_wait_hours, utilisation,
    min_servers_for_sla, offered_load,
)
from optimization.demand_profile import hourly_arrivals, peak_hour         # noqa: E402
from optimization.requirements import (                                    # noqa: E402
    build_requirements, requirement_grid, service_metrics_at, planned_demand,
)
from optimization.rostering import optimise_roster                         # noqa: E402

CONFIG = SRC / "optimization" / "staffing_config.yaml"
RESIDUALS = Path(__file__).resolve().parents[1] / "models" / "cv_residuals.npy"


@pytest.fixture(scope="module")
def a() -> StaffingAssumptions:
    return load_assumptions(CONFIG)


@pytest.fixture(scope="module")
def residuals() -> np.ndarray:
    return np.load(RESIDUALS)


@pytest.fixture(scope="module")
def opt(residuals, a) -> StaffingOptimizer:
    return StaffingOptimizer(residuals, assumptions=a)


# ---- Queueing primitives --------------------------------------------------

def test_erlang_c_in_unit_interval():
    for c in range(1, 20):
        p = erlang_c(c, a=5.0)
        assert 0.0 <= p <= 1.0


def test_erlang_c_decreases_with_more_servers():
    a_load = 6.0
    probs = [erlang_c(c, a_load) for c in range(7, 20)]
    assert all(x >= y - 1e-12 for x, y in zip(probs, probs[1:])), "P(wait) must fall as servers rise"


def test_unstable_queue_returns_one():
    # offered load >= servers -> everyone waits
    assert erlang_c(c=3, a=3.0) == 1.0
    assert prob_wait_exceeds(c=2, lam=10.0, mu=4.0, t_hours=0.5) == 1.0


def test_prob_wait_exceeds_bounds_and_monotonicity():
    lam, mu = 8.0, 4.0
    p_prev = 1.0
    for c in range(3, 12):
        p = prob_wait_exceeds(c, lam, mu, t_hours=0.5)
        assert 0.0 <= p <= 1.0
        assert p <= p_prev + 1e-12   # more servers -> not worse
        p_prev = p


def test_zero_arrivals_no_wait():
    assert prob_wait_exceeds(c=1, lam=0.0, mu=4.0, t_hours=0.5) == 0.0
    assert expected_wait_hours(c=1, lam=0.0, mu=4.0) == 0.0
    assert min_servers_for_sla(lam=0.0, mu=4.0, t_hours=0.5, wait_prob=0.2, floor=1) == 1


def test_min_servers_actually_meets_sla():
    lam, mu, t, target, util = 9.0, 4.0, 0.5, 0.20, 0.85
    c = min_servers_for_sla(lam, mu, t, target, target_utilisation=util, floor=1)
    assert prob_wait_exceeds(c, lam, mu, t) <= target + 1e-9
    assert utilisation(c, lam, mu) <= util + 1e-9
    # one fewer server must violate the SLA or the utilisation cap
    assert (prob_wait_exceeds(c - 1, lam, mu, t) > target or
            utilisation(c - 1, lam, mu) > util)


def test_min_servers_monotone_in_load():
    prev = 0
    for lam in range(1, 40):
        c = min_servers_for_sla(float(lam), mu=4.0, t_hours=0.5, wait_prob=0.2,
                                target_utilisation=0.85, floor=1)
        assert c >= prev, "required servers must not fall as arrivals rise"
        prev = c


# ---- Demand profile -------------------------------------------------------

def test_profile_weights_sum_to_one(a):
    assert math.isclose(sum(a.normalised_weights()), 1.0, rel_tol=1e-9)


def test_arrivals_sum_to_daily_total(a):
    arrivals = hourly_arrivals(150.0, a)
    assert math.isclose(sum(arrivals.values()), 150.0, rel_tol=1e-9)
    assert len(arrivals) == a.n_open_hours


def test_peak_hour_is_in_open_hours(a):
    assert peak_hour(150.0, a) in a.open_hours


# ---- Requirements ---------------------------------------------------------

def test_requirements_respect_safe_staffing_floor(residuals, a):
    reqs = build_requirements(100.0, residuals, a)
    for r in reqs:
        assert r.required >= a.role(r.role).min_on_duty


def test_requirements_monotone_in_forecast(residuals, a):
    grid_lo = requirement_grid(build_requirements(80.0, residuals, a))
    grid_hi = requirement_grid(build_requirements(160.0, residuals, a))
    for role in grid_lo:
        for hour in grid_lo[role]:
            assert grid_hi[role][hour] >= grid_lo[role][hour], \
                f"{role}@{hour}: requirement fell when demand rose"


def test_service_metrics_meet_sla_at_planned(residuals, a):
    # The requirement grid must meet the wait-time SLA at the planned demand,
    # by construction (we size Erlang-C directly to that demand).
    grid = requirement_grid(build_requirements(120.0, residuals, a))
    planned = planned_demand(120.0, residuals, a)
    metrics = service_metrics_at(grid, planned, a)
    for m in metrics:
        assert m["p_wait_gt_target"] <= a.wait_prob + 1e-6, m


# ---- Rostering ------------------------------------------------------------

def test_roster_covers_requirements(residuals, a):
    grid = requirement_grid(build_requirements(140.0, residuals, a))
    result = optimise_roster(grid, a)
    assert result.shortfall == 0
    assert result.status in ("OPTIMAL", "FEASIBLE")
    for role in grid:
        for hour in grid[role]:
            assert result.coverage[role][hour] >= grid[role][hour]


def test_roster_is_deterministic(residuals, a):
    grid = requirement_grid(build_requirements(140.0, residuals, a))
    r1 = optimise_roster(grid, a)
    r2 = optimise_roster(grid, a)
    assert r1.counts == r2.counts
    assert r1.total_cost == r2.total_cost


def test_roster_cost_matches_counts(residuals, a):
    grid = requirement_grid(build_requirements(140.0, residuals, a))
    r = optimise_roster(grid, a)
    recomputed = sum(
        r.counts[role.name][s.name] * role.hourly_cost * s.length
        for role in a.roles for s in a.shifts
    )
    assert math.isclose(recomputed, r.total_cost, rel_tol=1e-9)


# ---- End-to-end optimizer -------------------------------------------------

def test_cost_monotone_in_forecast(opt):
    prev = -1.0
    for fc in [40, 70, 100, 140, 190, 250]:
        rec = opt.recommend(fc)
        assert rec.daily_cost >= prev - 1e-6, "daily cost fell as forecast rose"
        prev = rec.daily_cost


def test_bodies_to_roster_monotone(opt):
    prev = {"doctor": 0, "nurse": 0, "admin": 0}
    for fc in [40, 70, 100, 140, 190, 250]:
        b = opt.recommend(fc).bodies_to_roster
        for role in prev:
            assert b[role] >= prev[role], f"{role} bodies fell as forecast rose"
        prev = b


def test_end_to_end_sla_met(opt, a):
    for fc in [50, 100, 150, 220]:
        rec = opt.recommend(fc)
        worst = max(rec.achieved_sla.values())
        assert worst <= a.wait_prob + 1e-6, f"SLA violated at fc={fc}: {worst}"
        assert rec.shortfall == 0


def test_recommendation_is_deterministic(opt):
    s1 = opt.recommend(120).summary()
    s2 = opt.recommend(120).summary()
    assert s1 == s2


def test_planned_demand_above_point_forecast(opt):
    # Residual mean is positive (model under-forecasts) and we plan to P90,
    # so the planned demand must exceed the point forecast.
    rec = opt.recommend(100)
    assert rec.planned_demand > rec.point_forecast
    assert rec.tail_demand >= rec.p95_demand >= rec.planned_demand


def test_assumptions_register_nonempty(opt):
    rec = opt.recommend(100)
    assert len(rec.assumptions_register) >= 7
    assert all({"assumption", "value", "unit", "justification"} <= set(r)
               for r in rec.assumptions_register)
