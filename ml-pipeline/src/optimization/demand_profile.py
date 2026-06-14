"""
Intraday demand profile.

Turns a single daily patient total into an arrival *rate* (patients/hour) for
each open hour, using the normalised arrival-profile assumption.
"""

from __future__ import annotations

from typing import Dict, List

from .assumptions import StaffingAssumptions


def hourly_arrivals(daily_total: float, a: StaffingAssumptions) -> Dict[int, float]:
    """Map a daily patient total to expected arrivals in each open hour.

    Because each open hour is one hour wide, the arrival count in an hour *is*
    the arrival rate lam (patients/hour) used by the Erlang-C model.
    """
    weights = a.normalised_weights()
    return {hour: max(0.0, daily_total) * w for hour, w in zip(a.open_hours, weights)}


def peak_hour(daily_total: float, a: StaffingAssumptions) -> int:
    """Open hour with the highest expected arrival rate."""
    arrivals = hourly_arrivals(daily_total, a)
    return max(arrivals, key=arrivals.get)


def profile_table(daily_total: float, a: StaffingAssumptions) -> List[dict]:
    """Convenience table of hour -> arrivals, for inspection / plotting."""
    arrivals = hourly_arrivals(daily_total, a)
    return [{"hour": h, "arrivals": round(v, 2)} for h, v in arrivals.items()]
