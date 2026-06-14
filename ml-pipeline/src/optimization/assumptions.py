"""
StaffingAssumptions — the single, explicit register of every assumption the
optimisation layer relies on.

Nothing in the staffing engine is allowed to hard-code a number that a
rostering manager might reasonably want to challenge. It all lives here, can be
overridden from `staffing_config.yaml`, and is echoed back with every
recommendation so the decision is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple

import yaml


@dataclass(frozen=True)
class Role:
    """A staff role with its service time and hourly cost."""
    name: str
    service_minutes: float   # average minutes one staff member spends per patient
    hourly_cost: float       # fully-loaded cost per rostered hour (AUD)
    min_on_duty: int         # minimum on duty during every open hour (safe-staffing floor)

    @property
    def service_rate(self) -> float:
        """Patients served per hour by one staff member (mu)."""
        return 60.0 / self.service_minutes


@dataclass(frozen=True)
class Shift:
    """A rosterable shift, defined by the open hours it covers."""
    name: str
    start_hour: int
    end_hour: int

    @property
    def hours(self) -> List[int]:
        return list(range(self.start_hour, self.end_hour))

    @property
    def length(self) -> int:
        return self.end_hour - self.start_hour


# ---- Default intraday arrival profile -------------------------------------
# Relative arrival weight per open hour (08:00-20:00). Walk-in / urgent-care
# clinics see a mid-morning build, a midday plateau, and a post-work evening
# bump. We only have *daily* data, so this SHAPE is an assumption, normalised
# to sum to 1.0 at runtime and flagged in the register for replacement once
# hourly arrival data exists.
DEFAULT_ARRIVAL_WEIGHTS: Tuple[float, ...] = (
    0.040,  # 08:00
    0.070,  # 09:00
    0.090,  # 10:00
    0.100,  # 11:00
    0.100,  # 12:00
    0.090,  # 13:00
    0.080,  # 14:00
    0.075,  # 15:00
    0.080,  # 16:00
    0.095,  # 17:00
    0.080,  # 18:00
    0.060,  # 19:00
)


@dataclass(frozen=True)
class StaffingAssumptions:
    # --- Opening hours ---
    open_hour: int = 8
    close_hour: int = 20  # exclusive; clinic open 08:00-20:00 = 12 hours

    # --- Roles (service times, costs, safe-staffing floors) ---
    roles: Tuple[Role, ...] = (
        Role("doctor", service_minutes=15.0, hourly_cost=150.0, min_on_duty=1),
        Role("nurse",  service_minutes=10.0, hourly_cost=65.0,  min_on_duty=1),
        Role("admin",  service_minutes=5.0,  hourly_cost=40.0,  min_on_duty=1),
    )

    # --- Service-level target (wait time) ---
    # "No more than `wait_prob` of patients wait longer than `wait_target_min`
    # minutes to start being seen by the relevant role."
    wait_target_min: float = 30.0
    wait_prob: float = 0.20

    # --- Utilisation cap ---
    # Staff cannot be productively occupied 100% of the time; this also keeps the
    # queue away from the unstable region. Required servers are never sized below
    # offered_load / target_utilisation.
    target_utilisation: float = 0.85

    # --- Demand uncertainty handling ---
    # We sample the model's empirical CV residuals, build the requirement for each
    # sampled day, and staff to the `coverage` quantile of required servers — i.e.
    # the roster is sufficient on `coverage` fraction of plausible days.
    coverage: float = 0.90
    n_scenarios: int = 1000
    random_seed: int = 42

    # --- Intraday arrival profile ---
    arrival_weights: Tuple[float, ...] = DEFAULT_ARRIVAL_WEIGHTS

    # --- Rosterable shifts ---
    shifts: Tuple[Shift, ...] = (
        Shift("Early", 8, 14),
        Shift("Mid",   11, 17),
        Shift("Late",  14, 20),
    )

    # --- Tail / standby flag ---
    # If the 95th-percentile demand exceeds the planned demand by more than this
    # fraction, flag the day for an on-call locum on standby.
    standby_threshold: float = 0.25

    # ----------------------------------------------------------------------
    @property
    def open_hours(self) -> List[int]:
        return list(range(self.open_hour, self.close_hour))

    @property
    def n_open_hours(self) -> int:
        return self.close_hour - self.open_hour

    def role(self, name: str) -> Role:
        for r in self.roles:
            if r.name == name:
                return r
        raise KeyError(f"Unknown role: {name}")

    def normalised_weights(self) -> List[float]:
        """Arrival weights aligned to open hours and renormalised to sum to 1."""
        w = list(self.arrival_weights)
        n = self.n_open_hours
        if len(w) != n:
            # Fall back to a flat profile if the supplied profile is the wrong length.
            w = [1.0] * n
        total = sum(w)
        return [x / total for x in w]

    # ----------------------------------------------------------------------
    def register(self) -> List[dict]:
        """Human-readable assumptions register (name, value, unit, justification)."""
        reg = [
            ("Opening hours", f"{self.open_hour:02d}:00-{self.close_hour:02d}:00",
             "clock", "Clinic trading window; demand is distributed across these hours."),
            ("Wait-time target", f"≤{self.wait_prob:.0%} wait > {self.wait_target_min:.0f} min",
             "service level", "The clinical/experience promise the roster must keep (Erlang-C)."),
            ("Target utilisation cap", f"{self.target_utilisation:.0%}",
             "fraction", "Staff are never sized above this occupancy; keeps the queue stable."),
            ("Demand coverage", f"{self.coverage:.0%} of days",
             "fraction", "Roster is sufficient on this fraction of sampled-residual scenarios."),
            ("Scenarios sampled", f"{self.n_scenarios}",
             "count", "Empirical CV residuals resampled to build the demand distribution."),
            ("Arrival profile", "assumed mid-morning/evening peaks",
             "shape", "PLACEHOLDER: derived from typical urgent-care curves, not hourly data."),
            ("Standby trigger", "known shock driver (e.g. thunderstorm asthma)",
             "flag", "On-call locum standby; the global residual tail is homoscedastic "
                     "so a known risk driver, not the data tail, drives this."),
        ]
        for r in self.roles:
            reg.append((
                f"{r.name.title()} service time", f"{r.service_minutes:.0f} min/patient",
                "minutes", f"= {r.service_rate:.1f} patients/hour per {r.name}.",
            ))
            reg.append((
                f"{r.name.title()} cost", f"${r.hourly_cost:.0f}/hr",
                "AUD", f"Fully-loaded hourly cost; min {r.min_on_duty} on duty at all times.",
            ))
        return [
            {"assumption": a, "value": v, "unit": u, "justification": j}
            for a, v, u, j in reg
        ]


# --------------------------------------------------------------------------
def load_assumptions(path: str | Path | None = None) -> StaffingAssumptions:
    """Load assumptions from a YAML file, falling back to coded defaults.

    The YAML may override any scalar field and may redefine `roles`, `shifts`
    and `arrival_weights`. Anything omitted keeps its default.
    """
    if path is None:
        return StaffingAssumptions()

    path = Path(path)
    if not path.exists():
        return StaffingAssumptions()

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    defaults = StaffingAssumptions()
    kwargs: dict = {}

    for field_name in (
        "open_hour", "close_hour", "wait_target_min", "wait_prob",
        "target_utilisation", "coverage", "n_scenarios", "random_seed",
        "standby_threshold",
    ):
        if field_name in cfg:
            kwargs[field_name] = cfg[field_name]

    if "roles" in cfg:
        kwargs["roles"] = tuple(
            Role(
                name=r["name"],
                service_minutes=float(r["service_minutes"]),
                hourly_cost=float(r["hourly_cost"]),
                min_on_duty=int(r.get("min_on_duty", 1)),
            )
            for r in cfg["roles"]
        )

    if "shifts" in cfg:
        kwargs["shifts"] = tuple(
            Shift(name=s["name"], start_hour=int(s["start_hour"]), end_hour=int(s["end_hour"]))
            for s in cfg["shifts"]
        )

    if "arrival_weights" in cfg:
        kwargs["arrival_weights"] = tuple(float(x) for x in cfg["arrival_weights"])

    merged = {**asdict(defaults), **kwargs}
    # asdict turns nested dataclasses into dicts; restore the typed tuples.
    merged["roles"] = kwargs.get("roles", defaults.roles)
    merged["shifts"] = kwargs.get("shifts", defaults.shifts)
    merged["arrival_weights"] = kwargs.get("arrival_weights", defaults.arrival_weights)
    return StaffingAssumptions(**merged)
