import pytest
from .main import POMDP

@pytest.fixture
def pomdp():
    thresholds = [0.5, 0.5, 0.5]  # active_links, bandwidth_consumption, write_time
    return POMDP()
def test_bellman_equation():