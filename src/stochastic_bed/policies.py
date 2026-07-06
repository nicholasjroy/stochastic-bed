"""Policy networks map history to [D, p] designs (D sensors, p coordinates)."""

import math

import torch
import torch.nn as nn
import torch.distributions as dist
from torch.distributions import transforms


class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: tuple[int, ...]):
        super().__init__()
        dims = (input_dim, *hidden_dims)
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class HistoryEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: tuple[int, ...] = (256,)):
        super().__init__()
        self.output_dim = output_dim
        self.mlp = MLP(input_dim, output_dim, hidden_dims)

    def forward(self, hist_designs, hist_outcomes):
        T = hist_designs.shape[-3]

        if T == 0:
            batch_shape = hist_designs.shape[:-3]
            device = hist_designs.device
            dtype = hist_designs.dtype
            return torch.zeros(*batch_shape, self.output_dim, device=device, dtype=dtype)

        xi_flat = hist_designs.flatten(start_dim=-2, end_dim=-1)   # [*B, T, D*p]
        x = torch.cat([xi_flat, hist_outcomes], dim=-1)            # [*B, T, D*p + D]

        z = self.mlp(x)       # [*B, T, output_dim]
        out = z.sum(dim=-2)   # [*B, output_dim]

        return out


class DeterministicPolicy(nn.Module):
    def __init__(self,
                 D: int,
                 p: int,
                 design_bound: float,
                 enc_hidden_dims: tuple[int, ...] = (256,),
                 enc_output_dim: int = 128,
                 hidden_dims: tuple[int, ...] = (64,)):
        super().__init__()
        self.D = D
        self.p = p
        self.design_bound = design_bound

        self.history_encoder = HistoryEncoder(
            input_dim=D*p + D,
            output_dim=enc_output_dim,
            hidden_dims=enc_hidden_dims
        )

        self.mlp = MLP(enc_output_dim, D*p, hidden_dims)

    def forward(self, hist_designs, hist_outcomes):
        batch_shape = hist_designs.shape[:-3]

        enc = self.history_encoder(hist_designs, hist_outcomes)

        z = self.mlp(enc)   # [*B, D*p]
        z = z.view(*batch_shape, self.D, self.p)
        out = torch.tanh(z) * self.design_bound

        return out


class StochasticPolicy(nn.Module):
    def __init__(self,
                 D: int,
                 p: int,
                 design_bound: float,
                 enc_hidden_dims: tuple[int, ...] = (256,),
                 enc_output_dim: int = 128,
                 hidden_dims: tuple[int, ...] = (64,),
                 min_std: float = 0.01,
                 init_mean: float = 0.0,
                 init_std: float = 0.5):
        super().__init__()
        self.D = D
        self.p = p
        self.design_bound = design_bound
        self.min_std = min_std

        self.history_encoder = HistoryEncoder(
            input_dim=D*p + D,
            output_dim=enc_output_dim,
            hidden_dims=enc_hidden_dims,
        )

        self.mean_mlp = MLP(enc_output_dim, D*p, hidden_dims)
        nn.init.constant_(self.mean_mlp.net[-1].bias, init_mean)

        self.log_std_mlp = MLP(enc_output_dim, D*p, hidden_dims)
        nn.init.constant_(self.log_std_mlp.net[-1].bias, math.log(max(init_std, 1e-8)))

        self.tanh_transform = transforms.TanhTransform(cache_size=1)
        self.scale_transform = transforms.AffineTransform(loc=0, scale=design_bound)

    def _base_distribution(self, hist_designs, hist_outcomes):
        batch_shape = hist_designs.shape[:-3]

        enc = self.history_encoder(hist_designs, hist_outcomes)   # [*B, enc_output_dim]

        mean = self.mean_mlp(enc)                        # [*B, D*p]
        mean = mean.view(*batch_shape, self.D, self.p)   # [*B, D, p]

        log_std = self.log_std_mlp(enc)
        log_std = log_std.view(*batch_shape, self.D, self.p)
        std = torch.exp(log_std) + self.min_std

        return dist.Independent(
            dist.Normal(mean, std), reinterpreted_batch_ndims=2,
        )   # Event shape: [D, p]

    def _distribution(self, hist_designs, hist_outcomes):
        base_dist = self._base_distribution(hist_designs, hist_outcomes)
        return dist.TransformedDistribution(
            base_dist, [self.tanh_transform, self.scale_transform],
        )

    def entropy(self, hist_designs, hist_outcomes, n_samples=100):
        base_dist = self._base_distribution(hist_designs, hist_outcomes)
        base_entropy = base_dist.entropy()    # [*B]
        x = base_dist.rsample((n_samples,))   # [n_samples, *B, D, p]

        y_tanh = self.tanh_transform(x)
        log_det_tanh = self.tanh_transform.log_abs_det_jacobian(x, y_tanh)
        y_final = self.scale_transform(y_tanh)
        log_det_scale = self.scale_transform.log_abs_det_jacobian(y_tanh, y_final)

        total_log_det = log_det_tanh + log_det_scale      # [n_samples, *B, D, p]
        total_log_det = total_log_det.sum(dim=[-2, -1])   # [n_samples, *B]

        # Total entropy: H(Y) = H(X) + E[log|det J|]
        return base_entropy + total_log_det.mean(dim=0)

    def forward(self, hist_designs, hist_outcomes):
        return self._distribution(hist_designs, hist_outcomes).rsample()   # [*B, D, p]


class RandomPolicy(nn.Module):
    def __init__(self,
                 D: int,
                 p: int,
                 design_bound: float):
        super().__init__()
        self.D = D
        self.p = p
        self.design_bound = design_bound

    def forward(self, hist_designs, hist_outcomes):
        batch_shape = hist_designs.shape[:-3]
        b = self.design_bound
        return hist_designs.new_empty(*batch_shape, self.D, self.p).uniform_(-b, b)
    
    def entropy(self, hist_designs, hist_outcomes):
        batch_shape = hist_designs.shape[:-3]
        b = self.design_bound
        return hist_designs.new_full(batch_shape, self.D * self.p * math.log(2 * b))
