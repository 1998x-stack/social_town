from __future__ import annotations
import pytest
import random
from evaluation.metrics import (
    diffusion_rate, network_density, bimodality_coefficient, social_response_lag
)


def test_diffusion_rate_zero():
    assert diffusion_rate(informed=0, total=10) == pytest.approx(0.0)


def test_diffusion_rate_full():
    assert diffusion_rate(informed=10, total=10) == pytest.approx(1.0)


def test_diffusion_rate_partial():
    assert diffusion_rate(informed=5, total=10) == pytest.approx(0.5)


def test_network_density_formula():
    # 4 agents, 6 directed edges (of 12 max) → density = 0.5
    assert network_density(edge_count=6, n_agents=4) == pytest.approx(0.5)


def test_network_density_empty():
    assert network_density(edge_count=0, n_agents=5) == pytest.approx(0.0)


def test_bc_uniform_low():
    """Uniform random opinions should have BC < 0.55."""
    rng = random.Random(42)
    opinions = [rng.uniform(-1, 1) for _ in range(100)]
    bc = bimodality_coefficient(opinions)
    assert bc < 0.55, f"BC={bc:.3f} for uniform dist should be < 0.55"


def test_bc_bimodal_high():
    """Two-peak distribution should have BC > 0.55."""
    opinions = [-0.9] * 50 + [0.9] * 50
    bc = bimodality_coefficient(opinions)
    assert bc > 0.55, f"BC={bc:.3f} for bimodal dist should be > 0.55"


def test_social_response_lag_returns_positive():
    lag = social_response_lag(inject_step=10, fifty_pct_step=25)
    assert lag == 15


def test_social_response_lag_none_when_not_reached():
    lag = social_response_lag(inject_step=10, fifty_pct_step=None)
    assert lag is None
