"""
Erlang-C queueing primitives.

We model each role's hour as an M/M/c queue: Poisson patient arrivals at rate
`lam` (patients/hour), exponential service at rate `mu` (patients/hour per
server), and `c` identical servers. Patients wait in a single queue (no
balking, no abandonment — see caveats in the article / notebook; Erlang-A would
extend this to abandonment).

All times are in hours unless noted. `t` wait targets passed in minutes are
converted by the caller.
"""

from __future__ import annotations

import math


def offered_load(lam: float, mu: float) -> float:
    """Offered load a = lam / mu, in Erlangs. The work arriving per unit time."""
    if mu <= 0:
        raise ValueError("Service rate mu must be positive.")
    return lam / mu


def erlang_b(c: int, a: float) -> float:
    """Erlang-B blocking probability, computed with the numerically stable
    recurrence B(0,a)=1, B(k,a) = a*B / (k + a*B). Avoids factorial overflow."""
    b = 1.0
    for k in range(1, c + 1):
        b = (a * b) / (k + a * b)
    return b


def erlang_c(c: int, a: float) -> float:
    """Probability that an arriving patient has to wait at all, P(W > 0).

    Derived from Erlang-B: C = B / (1 - rho*(1 - B)), with rho = a/c.
    Requires rho < 1 for a stable queue; returns 1.0 if unstable.
    """
    if c <= 0:
        return 1.0
    rho = a / c
    if rho >= 1.0:
        return 1.0
    b = erlang_b(c, a)
    return b / (1.0 - rho * (1.0 - b))


def prob_wait_exceeds(c: int, lam: float, mu: float, t_hours: float) -> float:
    """P(wait > t) for an M/M/c queue.

    P(W > t) = C * exp(-(c*mu - lam) * t),  for a stable queue (rho < 1).
    """
    if lam <= 0:
        return 0.0
    a = offered_load(lam, mu)
    if c <= a:           # unstable (or exactly critical) — effectively everyone waits
        return 1.0
    c_prob = erlang_c(c, a)
    return c_prob * math.exp(-(c * mu - lam) * t_hours)


def expected_wait_hours(c: int, lam: float, mu: float) -> float:
    """Expected waiting time in queue, Wq = C / (c*mu - lam)."""
    if lam <= 0:
        return 0.0
    a = offered_load(lam, mu)
    if c <= a:
        return math.inf
    return erlang_c(c, a) / (c * mu - lam)


def utilisation(c: int, lam: float, mu: float) -> float:
    """Server utilisation rho = lam / (c*mu)."""
    if c <= 0:
        return math.inf
    return lam / (c * mu)


def min_servers_for_sla(
    lam: float,
    mu: float,
    t_hours: float,
    wait_prob: float,
    target_utilisation: float = 1.0,
    floor: int = 0,
    max_servers: int = 200,
) -> int:
    """Smallest server count c such that:

      * the queue is stable (rho < 1),
      * utilisation rho <= target_utilisation, and
      * P(wait > t_hours) <= wait_prob,

    never returning fewer than `floor`. Returns `floor` when lam == 0.
    """
    if lam <= 0:
        return floor

    a = offered_load(lam, mu)
    # Start from the smallest c that satisfies the utilisation cap and stability.
    c = max(floor, 1, math.ceil(a / max(target_utilisation, 1e-9)))
    while c <= max_servers:
        if utilisation(c, lam, mu) <= target_utilisation and \
           prob_wait_exceeds(c, lam, mu, t_hours) <= wait_prob:
            return c
        c += 1
    return max_servers
