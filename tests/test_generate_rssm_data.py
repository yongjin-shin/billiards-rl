"""
tests/test_generate_rssm_data.py

Tests for the balanced (rejection-sampling) data generator in
world_model/generate_rssm_data.py.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_model.generate_rssm_data import generate_balanced
from world_model.rssm_dataset import ShotData


def _random_policy_fn():
    from simulator import BilliardsEnv
    env = BilliardsEnv(n_balls=1)

    def policy(obs):
        return env.action_space.sample()

    return policy, env


class TestGenerateBalanced:
    def test_respects_quotas(self):
        """Buckets should stop accepting shots once their quota is reached."""
        policy_fn, env = _random_policy_fn()
        try:
            quotas = {0: 3, 1: 1}
            shots, n_attempts = generate_balanced(
                quotas       = quotas,
                policy_fn    = policy_fn,
                n_balls      = 1,
                seed_start   = 0,
                max_attempts = 5_000,
                report_every = 10_000,
            )
        finally:
            env.close()

        assert n_attempts > 0
        counts = {k: 0 for k in quotas}
        for s in shots:
            assert isinstance(s, ShotData)
            counts[s.n_pocketed_targets()] += 1

        for k, q in quotas.items():
            assert counts[k] <= q, f"bucket {k} exceeded quota: {counts[k]} > {q}"

    def test_unreachable_bucket_does_not_hang(self):
        """A bucket that can never be filled (n_pocketed=2 with only 1 target ball)
        must not cause an infinite loop — max_attempts is a hard cap."""
        policy_fn, env = _random_policy_fn()
        try:
            quotas = {0: 1, 1: 1, 2: 5_000}
            shots, n_attempts = generate_balanced(
                quotas       = quotas,
                policy_fn    = policy_fn,
                n_balls      = 1,
                seed_start   = 0,
                max_attempts = 200,
                report_every = 10_000,
            )
        finally:
            env.close()

        assert n_attempts <= 200, "max_attempts safety cap must be respected"
        assert all(s.n_pocketed_targets() != 2 for s in shots), \
            "n_balls=1 has only one target ball; n_pocketed_targets==2 is impossible"
