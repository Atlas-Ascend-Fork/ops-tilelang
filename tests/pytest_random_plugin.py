import hashlib
import random

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption('--seed', type=int, default=0)


@pytest.fixture(autouse=True)
def seed(request):
    base = request.config.getoption('--seed')
    node_hash = int(hashlib.sha256(request.node.nodeid.encode()).hexdigest(), 16) % (2**31)
    seed = base + node_hash
    if -(2**63) <= base <= 2**64 - 1 and seed > 2**64 - 1:
        seed %= 2**64
    random.seed(seed)
    torch.manual_seed(seed)
    return seed
