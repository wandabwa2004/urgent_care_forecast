"""
Stage 2 — rostering with OR-Tools CP-SAT.

Given the per-hour, per-role requirements from Stage 1, choose how many of each
role to place on each rosterable shift so that:

  * every open hour is covered to at least its required count (per role), and
  * each role meets its safe-staffing floor in every open hour,

while minimising total labour cost (count x shift-length x hourly-cost).

Coverage is enforced as a *soft* constraint with a large shortfall penalty, so
the model always returns a usable roster (and reports any shortfall) rather than
failing as infeasible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ortools.sat.python import cp_model

from .assumptions import StaffingAssumptions


# Cost (AUD) charged per unit of unmet coverage-hour. Set far above any real
# staff hourly cost so the solver covers demand whenever physically possible.
SHORTFALL_PENALTY = 100_000


@dataclass
class RosterResult:
    counts: Dict[str, Dict[str, int]]          # counts[role][shift_name] = headcount
    total_cost: float
    shortfall: int                             # total unmet coverage-hours (0 if fully covered)
    coverage: Dict[str, Dict[int, int]]        # achieved staff-on-duty[role][hour]
    status: str
    headcount_by_shift: Dict[str, Dict[str, int]] = field(default_factory=dict)


def optimise_roster(requirements_grid: Dict[str, Dict[int, int]],
                    a: StaffingAssumptions) -> RosterResult:
    """Solve the min-cost covering roster for one day."""
    model = cp_model.CpModel()

    roles = [r.name for r in a.roles]
    shifts = list(a.shifts)
    hours = a.open_hours

    # Decision variables: integer count of each role on each shift.
    # Upper bound: enough to cover the single busiest hour's requirement.
    x: Dict[str, Dict[str, cp_model.IntVar]] = {}
    for role in roles:
        peak_req = max(requirements_grid[role].values())
        x[role] = {
            s.name: model.NewIntVar(0, peak_req + 1, f"x_{role}_{s.name}")
            for s in shifts
        }

    # Shortfall variables: unmet coverage per role per hour (soft constraint).
    short: Dict[str, Dict[int, cp_model.IntVar]] = {
        role: {h: model.NewIntVar(0, 1000, f"short_{role}_{h}") for h in hours}
        for role in roles
    }

    # Coverage constraint: staff on duty in hour h >= requirement - shortfall.
    for role in roles:
        for h in hours:
            on_duty = sum(
                x[role][s.name] for s in shifts if s.start_hour <= h < s.end_hour
            )
            model.Add(on_duty + short[role][h] >= requirements_grid[role][h])

    # Over-coverage (staff on duty above requirement), used only as a tiny
    # tie-breaker so the solver prefers tight rosters over wasteful overlap
    # among equal-cost solutions.
    over: Dict[str, Dict[int, cp_model.IntVar]] = {
        role: {h: model.NewIntVar(0, 1000, f"over_{role}_{h}") for h in hours}
        for role in roles
    }
    for role in roles:
        for h in hours:
            on_duty = sum(
                x[role][s.name] for s in shifts if s.start_hour <= h < s.end_hour
            )
            model.Add(over[role][h] == on_duty + short[role][h] - requirements_grid[role][h])

    # Objective: labour cost + heavily-penalised shortfall + tiny over-coverage tie-break.
    labour_cost = sum(
        x[role][s.name] * int(round(a.role(role).hourly_cost)) * s.length
        for role in roles for s in shifts
    )
    total_shortfall = sum(short[role][h] for role in roles for h in hours)
    total_over = sum(over[role][h] for role in roles for h in hours)
    model.Minimize(labour_cost + SHORTFALL_PENALTY * total_shortfall + total_over)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    # Deterministic, reproducible solutions (no random tie-breaking across runs).
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = a.random_seed
    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return RosterResult(counts={}, total_cost=float("inf"), shortfall=-1,
                            coverage={}, status=status_name)

    counts = {
        role: {s.name: int(solver.Value(x[role][s.name])) for s in shifts}
        for role in roles
    }
    coverage = {
        role: {
            h: sum(int(solver.Value(x[role][s.name]))
                   for s in shifts if s.start_hour <= h < s.end_hour)
            for h in hours
        }
        for role in roles
    }
    total_cost = float(sum(
        counts[role][s.name] * a.role(role).hourly_cost * s.length
        for role in roles for s in shifts
    ))
    shortfall = int(sum(solver.Value(short[role][h]) for role in roles for h in hours))

    headcount_by_shift = {
        s.name: {role: counts[role][s.name] for role in roles} for s in shifts
    }

    return RosterResult(
        counts=counts, total_cost=total_cost, shortfall=shortfall,
        coverage=coverage, status=status_name, headcount_by_shift=headcount_by_shift,
    )
