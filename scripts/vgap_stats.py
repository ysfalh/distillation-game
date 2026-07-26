"""Rank statistics shared by the v_gap binning and reporting scripts.

Hand-rolled so the report job needs nothing beyond the standard library: the
CPU partition's environment is not guaranteed to have scipy.
"""
from __future__ import annotations

from itertools import permutations
from math import sqrt


def rankdata(values: list[float]) -> list[float]:
    """Ranks starting at 1, with ties sharing their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2 or n != len(y):
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return cov / sqrt(vx * vy)


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(rankdata(x), rankdata(y))


def spearman_permutation_p(x: list[float], y: list[float], max_exact: int = 9) -> tuple[float, bool]:
    """Two-sided p-value for Spearman rho, exact for small n.

    With five bins there are only 120 orderings, so the exact permutation
    distribution is cheap and avoids leaning on an asymptotic approximation
    that is meaningless at this sample size. Returns (p, exact).
    """
    n = len(x)
    if n < 3 or n != len(y):
        return float("nan"), False
    observed = spearman(x, y)
    if observed != observed:  # NaN
        return float("nan"), False
    if n > max_exact:
        return float("nan"), False

    rx = rankdata(x)
    ry = rankdata(y)
    at_least_as_extreme = 0
    total = 0
    for perm in permutations(ry):
        total += 1
        if abs(pearson(rx, list(perm))) >= abs(observed) - 1e-12:
            at_least_as_extreme += 1
    return at_least_as_extreme / total, True
