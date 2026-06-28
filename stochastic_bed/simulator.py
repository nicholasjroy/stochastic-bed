import torch
import torch.nn as nn
import torch.distributions as dist
from torch import Tensor


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
        theta = theta.view(*batch_shape, self.K, self.p)

        distances = torch.cdist(designs, theta)       # [*B, D, K]
        signals = self.a / (self.m + distances**2)
        total_signal = signals.sum(dim=-1) + self.b   # [*B, D]
        loc = torch.log(total_signal)
        cov = self.noise_std**2 * torch.eye(self.D, device=loc.device, dtype=loc.dtype)   # [D, D]

        return dist.MultivariateNormal(loc=loc, covariance_matrix=cov)

    def step(self, theta: Tensor, design: Tensor):
        """Sample an outcome y_t for a design given theta."""
        y_t = self.likelihood(theta, design).rsample()   # [*B, D]
        return y_t

    def rollout(self, theta: Tensor, policy: nn.Module, return_entropy: bool = False):
        """Simulate full trajectories under a given batch of thetas and policy."""
        batch_shape = theta.shape[:-1]
        designs  = torch.zeros(*batch_shape, self.T, self.D, self.p, device=theta.device, dtype=theta.dtype)   # [*B, T, D, p]
        outcomes = torch.zeros(*batch_shape, self.T, self.D, device=theta.device, dtype=theta.dtype)           # [*B, T, D]
        entropies = None

        if return_entropy:
            entropies = torch.empty(*batch_shape, self.T, device=theta.device, dtype=theta.dtype)   # [*B, T]

        for t in range(self.T):
            hist_designs = designs[..., :t, :, :]   # [*B, t, D, p]
            hist_outcomes = outcomes[..., :t, :]    # [*B, t, D]

            xi_t = policy(hist_designs, hist_outcomes)
            if return_entropy:
                entropies[..., t] = policy.entropy(hist_designs, hist_outcomes)   # [*B]

            y_t = self.step(theta, xi_t)

            designs[..., t, :, :] = xi_t
            outcomes[..., t, :] = y_t

        return designs, outcomes, entropies
    