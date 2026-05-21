import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from dataclasses import dataclass

from utilities import *

import json

# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    seed: int = 0
    n_train: int = 2**13
    n_test: int = 2**11
    n_hidden: int = 3
    hidden_width: int = 64
    lr: float = 1e-3
    epochs: int = 20000
    checkpoint_interval: int = epochs // 20
    topk_frac: float = 0.10
    compare_epoch: int = epochs // 20


cfg = Config()

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# ============================================================
# Dataset
# ============================================================

noise_multiplier = 1e-1 

def vanderpol_rhs(t, state, mu=3.0):
    x, v = state[..., 0:1], state[..., 1:2]
    dxdt = v
    dvdt = mu * (1.0 - x**2) * v - x
    return torch.cat([dxdt, dvdt], dim=-1)

def rk4_step(f, t, y, dt, mu=3.0):
    k1 = f(t, y, mu)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1, mu)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2, mu)
    k4 = f(t + dt, y + dt * k3, mu)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def simulate_vanderpol(t_grid, y0, mu=3.0):
    y = torch.empty((len(t_grid), 2), dtype=torch.float32)
    y[0] = y0
    for i in range(len(t_grid) - 1):
        dt = t_grid[i + 1] - t_grid[i]
        y[i + 1] = rk4_step(vanderpol_rhs, t_grid[i], y[i:i+1], dt, mu=mu).squeeze(0)
    return y

# time grids
t_train = torch.linspace(0.0, 20.0, cfg.n_train, dtype=torch.float32)
t_test  = torch.linspace(0.0, 20.0, cfg.n_test, dtype=torch.float32)

# underlying ODE trajectory
y0 = torch.tensor([2.0, 0.0], dtype=torch.float32)  # [position, velocity]
traj_train = simulate_vanderpol(t_train, y0, mu=3.0)
traj_test  = simulate_vanderpol(t_test,  y0, mu=3.0)

x_train = t_train.unsqueeze(1).to(device)
y_train = (traj_train[:, 0:1] + noise_multiplier * torch.randn(cfg.n_train, 1)).to(device)

x_test = t_test.unsqueeze(1).to(device)
y_test = (traj_test[:, 0:1] + noise_multiplier * torch.randn(cfg.n_test, 1)).to(device)

# ============================================================
# Model
# ============================================================

class SmallMLP(nn.Module):
    def __init__(self, width):
        super().__init__()

        layers = [
            nn.Linear(1, width),
            # nn.Tanh(),
            nn.LeakyReLU(),
        ]

        for _ in range(cfg.n_hidden - 1):
            layers.extend([
                nn.Linear(width, width),
                # nn.Tanh(),
                nn.LeakyReLU(),
            ])

        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model = SmallMLP(cfg.hidden_width).to(device)


# ============================================================
# Initial state and initial sensitivities
# ============================================================

initial_state = copy.deepcopy(model.state_dict())

initial_model = SmallMLP(cfg.hidden_width).to(device)
initial_model.load_state_dict(initial_state)
initial_model.eval()

J_init = compute_parameter_jacobian(initial_model, x_train)
C_init = compute_covariance(J_init)
S_init = sensitivity_scores(J_init)
init_topk_idx = topk_indices(S_init, cfg.topk_frac)


# ============================================================
# Optimisation
# ============================================================

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=cfg.lr)


# ============================================================
# Logging
# ============================================================

history = {
    "epoch": [],
    "train_loss": [],
    "test_loss": [],
    "spearman_init_current": [],
    "init_topk_mass": [],
    "spearman_ref_current": [],
    "ref_topk_mass": [],
}

# reference snapshot
S_ref = None
C_ref = None
ref_topk_idx = None
ref_epoch = None


# ============================================================
# Training
# ============================================================

print("\nTraining")
print("========")

for epoch in range(cfg.epochs):
    model.train()
    optimizer.zero_grad()

    pred = model(x_train)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()

    if epoch % cfg.checkpoint_interval == 0:
        model.eval()

        with torch.no_grad():
            train_loss = criterion(model(x_train), y_train).item()
            test_loss = criterion(model(x_test), y_test).item()

        J = compute_parameter_jacobian(model, x_train)
        S_curr = sensitivity_scores(J)

        rho_init_curr = spearman_corr(S_init, S_curr)
        init_topk_mass = mass_on_indices(S_curr, init_topk_idx)

        rho_ref_curr = np.nan
        ref_topk_mass = np.nan

        if (S_ref is None) and (epoch >= cfg.compare_epoch):
            S_ref = S_curr.detach().clone()
            C_ref = compute_covariance(J).detach().clone()
            ref_topk_idx = topk_indices(S_ref, cfg.topk_frac)
            ref_epoch = epoch
            print(f"Captured reference snapshot at epoch={ref_epoch}")

        if S_ref is not None:
            rho_ref_curr = spearman_corr(S_ref, S_curr)
            ref_topk_mass = mass_on_indices(S_curr, ref_topk_idx)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["spearman_init_current"].append(rho_init_curr)
        history["init_topk_mass"].append(init_topk_mass)
        history["spearman_ref_current"].append(rho_ref_curr)
        history["ref_topk_mass"].append(ref_topk_mass)

        print(
            f"epoch={epoch:6d} | "
            f"train={train_loss:.3e} | "
            f"test={test_loss:.3e} | "
            f"rho(init,current)={rho_init_curr:.3f} | "
            f"init-topk-mass={init_topk_mass:.3f} | "
            f"rho(ref,current)={rho_ref_curr:.3f} | "
            f"ref-topk-mass={ref_topk_mass:.3f}"
        )


# ============================================================
# Final Jacobians
# ============================================================

model.eval()
J_final = compute_parameter_jacobian(model, x_train)
C_final = compute_covariance(J_final)
S_final = sensitivity_scores(J_final)

eig_init = eigvals_from_covariance(C_init)
eig_final = eigvals_from_covariance(C_final)

if S_ref is not None:
    eig_ref = eigvals_from_covariance(C_ref)
else:
    eig_ref = None


# ============================================================
# Plotting
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Learned function
with torch.no_grad():
    y_pred = model(x_test).detach().cpu().numpy()

axes[0, 0].plot(x_test.detach().cpu().numpy(), y_test.detach().cpu().numpy(), label="true")
axes[0, 0].plot(x_test.detach().cpu().numpy(), y_pred, label="prediction")
axes[0, 0].set_title("Function approximation")
axes[0, 0].legend()

# 2. Loss curves
axes[0, 1].plot(history["epoch"], history["train_loss"], label="train")
axes[0, 1].plot(history["epoch"], history["test_loss"], label="test")
axes[0, 1].set_yscale("log")
axes[0, 1].set_title("Loss evolution")
axes[0, 1].legend()

# 3. Sensitivity persistence against init and compare_epoch
ax = axes[0, 2]
ax2 = ax.twinx()

l1 = ax.plot(history["epoch"], history["spearman_init_current"], label="Spearman(init, current)", color='b')
l2 = []
l3 = []
if any(~np.isnan(np.array(history["spearman_ref_current"], dtype=np.float64))):
    l2 = ax.plot(history["epoch"], history["spearman_ref_current"], label="Spearman(ref, current)", color='g')
    l3 = ax2.plot(history["epoch"], history["ref_topk_mass"], linestyle="--", label="Mass in ref top-k", color='g')
else:
    l2 = []
    l3 = []

m1 = ax2.plot(history["epoch"], history["init_topk_mass"], linestyle="--", label="Mass in init top-k", color='b')

ax.set_ylim(-1.0, 1.0)
ax2.set_ylim(0.0, 1.0)
ax.set_title("Sensitivity persistence over training")
ax.set_xlabel("epoch")
ax.set_ylabel("rank correlation")
ax2.set_ylabel("sensitivity mass")

lines = l1 + l2 + m1 + l3
labels = [line.get_label() for line in lines]
ax.legend(lines, labels, loc="best")

# 4. Covariance matrix at init
im0 = axes[1, 0].imshow(C_init.detach().cpu().numpy(), aspect="auto", cmap='RdBu_r')
axes[1, 0].set_title("Sensitivity covariance at init")
plt.colorbar(im0, ax=axes[1, 0])

# 5. Covariance matrix at end
im1 = axes[1, 1].imshow(np.abs((C_final-C_init).detach().cpu().numpy()), aspect="auto", cmap='RdBu_r')
axes[1, 1].set_title("Sensitivity covariance difference after training")
plt.colorbar(im1, ax=axes[1, 1])

# 6. Eigenvalue spectrum
axes[1, 2].plot(eig_init.detach().cpu().numpy(), label="initial")
if eig_ref is not None:
    axes[1, 2].plot(eig_ref.detach().cpu().numpy(), label=f"ref @ epoch {ref_epoch}")
axes[1, 2].plot(eig_final.detach().cpu().numpy(), label="final")
axes[1, 2].set_yscale("log")
axes[1, 2].set_title("Sensitivity covariance eigenspectrum")
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('Plots/Initial_Experiment.pdf')
# plt.show()


# ============================================================
# Final summary
# ============================================================

summary = {
    "parameter_count": parameter_count(model),
    "final_train_loss": history["train_loss"][-1],
    "final_test_loss": history["test_loss"][-1],
    "final_spearman_init_final": spearman_corr(S_init, S_final),
    "final_mass_in_init_topk": mass_on_indices(S_final, init_topk_idx),
    "reference_epoch": ref_epoch if S_ref is not None else None,
    "final_spearman_ref_final": spearman_corr(S_ref, S_final) if S_ref is not None else None,
    "final_mass_in_ref_topk": mass_on_indices(S_final, ref_topk_idx) if S_ref is not None else None,
    "largest_initial_eigenvalue": float(eig_init[0].item()),
    "largest_ref_eigenvalue": float(eig_ref[0].item()) if eig_ref is not None else None,
    "largest_final_eigenvalue": float(eig_final[0].item()),
}

with open("Plots/final_summary.json", "w") as f:
    json.dump(summary, f, indent=2)