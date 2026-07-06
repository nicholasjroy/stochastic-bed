"""Location-finding simulator: sensors observe signals with log-Gaussian noise."""

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.distributions as dist
from torch import Tensor


class Trajectory(NamedTuple):
    designs: Tensor                   # [*B, T, D, p]
    outcomes: Tensor                  # [*B, T, D]
    entropies: Tensor | None = None   # [*B, T]


class LocationFinding(nn.Module):
    def __init__(self,
            p: int = 2,                    # Number of coordinate dimensions
            K: int = 1,                    # Number of sources
            D: int = 1,                    # Number of sensors each time period
            T: int = 7,                    # Number of time periods
            a: float = 1.0,                # Source signal weight
            b: float = 0.1,                # Background signal weight
            m: float = 0.001,              # Min squared distance to a source
            noise_std: float = 0.5,        # Observation noise std
            prior_bound: float = 3.0,      # Half-width of uniform prior box
    ) -> None:
        super().__init__()

        self.p = p
        self.K = K
        self.D = D
        self.T = T
        self.a = a
        self.b = b
        self.m = m
        self.noise_std = noise_std
        self.prior_bound = prior_bound

    def prior(self):
        base = dist.Uniform(-self.prior_bound, self.prior_bound)
        return dist.Independent(base.expand([self.K * self.p]), 1)   # event shape: [K*p]

    def likelihood(self, theta: Tensor, designs: Tensor):
        """Outcome likelihood p(y | theta, design) as a Distribution."""
        batch_shape = theta.shape[:-1]
        theta = theta.reshape(*batch_shape, self.K, self.p)

        diffs = designs.unsqueeze(-2) - theta.unsqueeze(-3)   # [*B, D, K, p]
        sq_distances = diffs.pow(2).sum(-1)                   # [*B, D, K]
        signals = self.a / (self.m + sq_distances)
        total_signal = signals.sum(dim=-1) + self.b   # [*B, D]
        loc = torch.log(total_signal)

        return dist.Independent(dist.Normal(loc, self.noise_std), 1)

    def step(self, theta: Tensor, design: Tensor):
        """Sample an outcome y_t for a design given theta."""
        y_t = self.likelihood(theta, design).rsample()   # [*B, D]
        return y_t

    def rollout(
        self, theta: Tensor, policy: nn.Module, return_entropy: bool = False,
    ) -> Trajectory:
        """Simulate full trajectories under a given batch of thetas and policy."""
        batch_shape = theta.shape[:-1]
        designs, outcomes, entropies = [], [], []

        hist_designs = theta.new_zeros(*batch_shape, 0, self.D, self.p)   # [*B, 0, D, p]
        hist_outcomes = theta.new_zeros(*batch_shape, 0, self.D)          # [*B, 0, D]

        for _ in range(self.T):
            xi_t = policy(hist_designs, hist_outcomes)
            if return_entropy:
                entropies.append(policy.entropy(hist_designs, hist_outcomes))   # [*B]

            y_t = self.step(theta, xi_t)

            designs.append(xi_t)
            outcomes.append(y_t)
            hist_designs = torch.cat([hist_designs, xi_t.unsqueeze(-3)], dim=-3)
            hist_outcomes = torch.cat([hist_outcomes, y_t.unsqueeze(-2)], dim=-2)

        return Trajectory(
            designs=torch.stack(designs, dim=-3),
            outcomes=torch.stack(outcomes, dim=-2),
            entropies=torch.stack(entropies, dim=-1) if return_entropy else None,
        )
