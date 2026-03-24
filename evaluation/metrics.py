"""Four-dimension social dynamics metrics."""
from __future__ import annotations
import logging
from typing import Optional
import numpy as np
from scipy import stats

__all__ = [
    "diffusion_rate", "network_density",
    "bimodality_coefficient", "social_response_lag",
]

logger = logging.getLogger(__name__)


def diffusion_rate(informed: int, total: int) -> float:
    """Fraction of agents who know about the injected event."""
    if total == 0:
        return 0.0
    return informed / total


def network_density(edge_count: int, n_agents: int) -> float:
    """Ratio of actual directed edges to maximum possible."""
    max_edges = n_agents * (n_agents - 1)
    if max_edges == 0:
        return 0.0
    return min(1.0, edge_count / max_edges)


def bimodality_coefficient(opinions: list[float]) -> float:
    """
    Bimodality Coefficient (BC) based on Sarle (1981).
    BC = (skewness^2 + 1) / (kurtosis + correction_term)
    BC > 0.555 indicates bimodal distribution.
    """
    n = len(opinions)
    if n < 4:
        return 0.0
    arr = np.array(opinions, dtype=float)
    skew = float(stats.skew(arr))
    kurt = float(stats.kurtosis(arr))  # Fisher's definition (excess kurtosis)
    # Correction term to handle small n
    correction = 3.0 * (n - 1) ** 2 / max((n - 2) * (n - 3), 1)
    denominator = kurt + correction
    if abs(denominator) < 1e-9:
        return 0.0
    bc = (skew ** 2 + 1) / denominator
    return float(bc)


def social_response_lag(
    inject_step: int,
    fifty_pct_step: Optional[int],
) -> Optional[int]:
    """Steps from event injection to 50% agent awareness. None if not yet reached."""
    if fifty_pct_step is None:
        return None
    return fifty_pct_step - inject_step
