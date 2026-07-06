"""Joint policy-posterior training loop for Barber-Agakov bound"""

import argparse
import copy
import csv
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from .policies import DeterministicPolicy, RandomPolicy, StochasticPolicy
from .posterior import PosteriorNet
from .simulator import LocationFinding


def train(simulator: LocationFinding,
          policy_cls: type[nn.Module] = DeterministicPolicy,
          posterior: PosteriorNet | None = None,
          *,
          design_bound: float = 3.0,
          num_steps: int = 3000,
          batch_size: int = 256,
          clip_norm: float | None = 2.0,
          alpha: float = 0.0,
          alpha_decay: float = 1.0,
          device: torch.device | None = None,
          verbose: bool = True):
    """Train policy and posterior jointly by minimizing the BA loss over rollouts."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    D, p, K = simulator.D, simulator.p, simulator.K
    policy = policy_cls(D=D, p=p, design_bound=design_bound).to(device)

    if posterior is None:
        posterior = PosteriorNet(D=D, p=p, K=K, flow_type="CouplingFlow").to(device)
    else:
        posterior = copy.deepcopy(posterior).to(device)

    trainable = nn.ModuleList([policy, posterior])
    optimizer = torch.optim.AdamW(trainable.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        total_steps=num_steps,
        pct_start=0.4,
        anneal_strategy="cos",
        div_factor=25,
        final_div_factor=25,
    )

    metrics = {
        "ba_loss": [],
        "grad_norm": [],
        "mean_entropy": [],
        "alpha": [],
        "learning_rate": [],
    }
    current_alpha = alpha

    with torch.enable_grad():
        for step in range(num_steps):
            theta = simulator.prior().sample((batch_size,)).to(device)   # [B, K*p]
            designs, outcomes = simulator.rollout(theta, policy)

            loss = posterior.loss(theta, designs, outcomes).mean()

            if hasattr(policy, "entropy"):
                mean_entropy = sum(
                    policy.entropy(designs[:, :t], outcomes[:, :t])
                    for t in range(simulator.T)
                ).mean()   # Sum over T, average over batch
            else:
                mean_entropy = torch.tensor(0.0, device=device)

            total_loss = loss - current_alpha * mean_entropy

            optimizer.zero_grad()
            total_loss.backward()

            if clip_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable.parameters(), clip_norm,
                )
            else:
                sq_grad_sums = [
                    p.grad.pow(2).sum()
                    for p in trainable.parameters() if p.grad is not None
                ]
                grad_norm = torch.sqrt(
                    sum(sq_grad_sums, torch.tensor(0.0, device=device))
                )

            optimizer.step()
            scheduler.step()

            current_alpha *= alpha_decay
            current_lr = optimizer.param_groups[0]["lr"]

            metrics["ba_loss"].append(loss.item())
            metrics["grad_norm"].append(grad_norm.item())
            metrics["mean_entropy"].append(mean_entropy.item())
            metrics["alpha"].append(current_alpha)
            metrics["learning_rate"].append(current_lr)

            if verbose and step % 50 == 0:
                parts = [
                    f"Step {step + 1}: BA loss {loss.item():.3f}",
                    f"grad_norm {metrics['grad_norm'][-1]:.3f}",
                ]
                if hasattr(policy, "entropy"):
                    parts.append(f"Entropy {metrics['mean_entropy'][-1]:.3f}")
                    parts.append(f"alpha {current_alpha:.4f}")
                parts.append(f"lr {current_lr:.2e}")
                print("   ".join(parts))

    return policy, posterior, metrics


def main():
    policy_classes = {
        "deterministic": DeterministicPolicy,
        "stochastic": StochasticPolicy,
        "random": RandomPolicy,
    }

    parser = argparse.ArgumentParser(
        description="Train a design policy on the location-finding problem.",
    )
    parser.add_argument("--policy", choices=policy_classes, default="deterministic")
    parser.add_argument("--num-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.0)
    args = parser.parse_args()

    policy, posterior, metrics = train(LocationFinding(),
                                       policy_classes[args.policy],
                                       num_steps=args.num_steps,
                                       batch_size=args.batch_size,
                                       alpha=args.alpha)

    run_dir = Path("runs") / f"{datetime.now():%Y-%m-%d_%H%M%S}_{args.policy}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(metrics.keys())
        writer.writerows(zip(*metrics.values()))
    torch.save(policy.state_dict(), run_dir / "policy.pt")
    torch.save(posterior, run_dir / "posterior.pt")
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
