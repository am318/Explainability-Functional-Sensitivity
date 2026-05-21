import copy
import json
import os
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from utilities import *  # noqa: F401,F403



# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    seed: int = 0

    # Training samples are random points in phase space: input = [x, v].
    n_train: int = 2**12
    n_test: int = 2**10

    # Sensitivity/Jacobian calculations are expensive for vector outputs, so use
    # a fixed subset for diagnostics rather than the full training set.
    n_sensitivity: int = 128

    n_hidden: int = 3
    hidden_width: int = 64
    lr: float = 1e-3
    epochs: int = 3000
    checkpoint_interval: int = epochs // 20
    topk_frac: float = 0.10
    compare_epoch: int = epochs // 20

    # Van der Pol oscillator parameter and phase-space domain.
    mu: float = 3.0
    x_min: float = -3.0
    x_max: float = 3.0
    v_min: float = -5.0
    v_max: float = 5.0

    noise_multiplier: float = 0.02
    plot_grid_size: int = 45


cfg = Config()


torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

os.makedirs("Plots", exist_ok=True)


# ============================================================
# Dataset: learn the 2D Van der Pol vector field
# ============================================================
# Input:  state = [x, v]
# Target: f(x, v) = [dx/dt, dv/dt]
#         dx/dt = v
#         dv/dt = mu * (1 - x^2) * v - x
# This is harder than learning a single 1D trajectory x(t), because the model
# must learn the whole phase-space dynamics, not one solution curve.

def vanderpol_vector_field(state, mu=3.0):
    x = state[..., 0:1]
    v = state[..., 1:2]
    dxdt = v
    dvdt = mu * (1.0 - x ** 2) * v - x
    return torch.cat([dxdt, dvdt], dim=-1)


def sample_phase_space(n):
    x = cfg.x_min + (cfg.x_max - cfg.x_min) * torch.rand(n, 1, dtype=torch.float32)
    v = cfg.v_min + (cfg.v_max - cfg.v_min) * torch.rand(n, 1, dtype=torch.float32)
    return torch.cat([x, v], dim=-1)


x_train = sample_phase_space(cfg.n_train).to(device)
y_train_clean = vanderpol_vector_field(x_train, mu=cfg.mu)
y_train = y_train_clean + cfg.noise_multiplier * torch.randn_like(y_train_clean)

x_test = sample_phase_space(cfg.n_test).to(device)
y_test_clean = vanderpol_vector_field(x_test, mu=cfg.mu)
y_test = y_test_clean + cfg.noise_multiplier * torch.randn_like(y_test_clean)

# Fixed subset for sensitivity analysis.
sens_idx = torch.randperm(cfg.n_train, device=device)[:cfg.n_sensitivity]
x_sens = x_train[sens_idx]


# ============================================================
# Model
# ============================================================

class SmallMLP(nn.Module):
    def __init__(self, width, input_dim=2, output_dim=2):
        super().__init__()

        layers = [
            nn.Linear(input_dim, width),
            nn.LayerNorm(width),
            nn.SiLU(),
        ]

        for _ in range(cfg.n_hidden - 1):
            layers.extend([
                nn.Linear(width, width),
                nn.LayerNorm(width),
                nn.SiLU(),
            ])

        layers.append(nn.Linear(width, output_dim))
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

J_init = compute_parameter_jacobian(initial_model, x_sens)
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

S_ref = None
C_ref = None
ref_topk_idx = None
ref_epoch = None


# ============================================================
# Training
# ============================================================

print("\nTraining")
print("========")

for epoch in range(cfg.epochs + 1):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    pred = model(x_train)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()

    if epoch % cfg.checkpoint_interval == 0:
        model.eval()

        with torch.no_grad():
            train_loss = criterion(model(x_train), y_train_clean).item()
            test_loss = criterion(model(x_test), y_test_clean).item()

        J = compute_parameter_jacobian(model, x_sens)
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
J_final = compute_parameter_jacobian(model, x_sens)
C_final = compute_covariance(J_final)
S_final = sensitivity_scores(J_final)

eig_init = eigvals_from_covariance(C_init)
eig_final = eigvals_from_covariance(C_final)

eig_ref = eigvals_from_covariance(C_ref) if S_ref is not None else None


# ============================================================
# Plotting
# ============================================================

grid_x = torch.linspace(cfg.x_min, cfg.x_max, cfg.plot_grid_size, dtype=torch.float32, device=device)
grid_v = torch.linspace(cfg.v_min, cfg.v_max, cfg.plot_grid_size, dtype=torch.float32, device=device)
X, V = torch.meshgrid(grid_x, grid_v, indexing="xy")
grid_state = torch.stack([X.reshape(-1), V.reshape(-1)], dim=-1)

with torch.no_grad():
    true_field = vanderpol_vector_field(grid_state, mu=cfg.mu)
    pred_field = model(grid_state)
    field_error = torch.linalg.norm(pred_field - true_field, dim=-1)

X_np = X.detach().cpu().numpy()
V_np = V.detach().cpu().numpy()
true_np = true_field.detach().cpu().numpy().reshape(cfg.plot_grid_size, cfg.plot_grid_size, 2)
pred_np = pred_field.detach().cpu().numpy().reshape(cfg.plot_grid_size, cfg.plot_grid_size, 2)
err_np = field_error.detach().cpu().numpy().reshape(cfg.plot_grid_size, cfg.plot_grid_size)

fig, axes = plt.subplots(2, 3, figsize=(19, 11))

# 1. True vector field
skip = max(1, cfg.plot_grid_size // 20)
axes[0, 0].quiver(
    X_np[::skip, ::skip], V_np[::skip, ::skip],
    true_np[::skip, ::skip, 0], true_np[::skip, ::skip, 1],
    angles="xy"
)
axes[0, 0].set_title("True Van der Pol vector field")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("v")

# 2. Learned vector field
axes[0, 1].quiver(
    X_np[::skip, ::skip], V_np[::skip, ::skip],
    pred_np[::skip, ::skip, 0], pred_np[::skip, ::skip, 1],
    angles="xy"
)
axes[0, 1].set_title("Learned vector field")
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("v")

# 3. Error heatmap
im_err = axes[0, 2].imshow(
    err_np,
    extent=[cfg.x_min, cfg.x_max, cfg.v_min, cfg.v_max],
    origin="lower",
    aspect="auto",
)
axes[0, 2].set_title("Vector-field error norm")
axes[0, 2].set_xlabel("x")
axes[0, 2].set_ylabel("v")
plt.colorbar(im_err, ax=axes[0, 2])

# 4. Loss curves
axes[1, 0].plot(history["epoch"], history["train_loss"], label="train")
axes[1, 0].plot(history["epoch"], history["test_loss"], label="test")
axes[1, 0].set_yscale("log")
axes[1, 0].set_title("Loss evolution")
axes[1, 0].set_xlabel("epoch")
axes[1, 0].legend()

# 5. Sensitivity persistence
ax = axes[1, 1]
ax2 = ax.twinx()
l1 = ax.plot(history["epoch"], history["spearman_init_current"], label="Spearman(init, current)")
l2 = []
l3 = []
if any(~np.isnan(np.array(history["spearman_ref_current"], dtype=np.float64))):
    l2 = ax.plot(history["epoch"], history["spearman_ref_current"], label="Spearman(ref, current)")
    l3 = ax2.plot(history["epoch"], history["ref_topk_mass"], linestyle="--", label="Mass in ref top-k")
m1 = ax2.plot(history["epoch"], history["init_topk_mass"], linestyle="--", label="Mass in init top-k")
ax.set_ylim(-1.0, 1.0)
ax2.set_ylim(0.0, 1.0)
ax.set_title("Sensitivity persistence over training")
ax.set_xlabel("epoch")
ax.set_ylabel("rank correlation")
ax2.set_ylabel("sensitivity mass")
lines = l1 + l2 + m1 + l3
labels = [line.get_label() for line in lines]
ax.legend(lines, labels, loc="best")

# 6. Eigenspectrum
axes[1, 2].plot(eig_init.detach().cpu().numpy(), label="initial")
if eig_ref is not None:
    axes[1, 2].plot(eig_ref.detach().cpu().numpy(), label=f"ref @ epoch {ref_epoch}")
axes[1, 2].plot(eig_final.detach().cpu().numpy(), label="final")
axes[1, 2].set_yscale("log")
axes[1, 2].set_title("Sensitivity covariance eigenspectrum")
axes[1, 2].legend()

plt.tight_layout()
plt.savefig("Plots/Initial_Experiment_2D_VanderPol.pdf")
plt.savefig("Plots/Initial_Experiment_2D_VanderPol.png", dpi=200)
plt.close(fig)


# ============================================================
# Final summary
# ============================================================

with torch.no_grad():
    final_train_mse = criterion(model(x_train), y_train_clean).item()
    final_test_mse = criterion(model(x_test), y_test_clean).item()
    mean_grid_error = float(field_error.mean().detach().cpu().item())
    max_grid_error = float(field_error.max().detach().cpu().item())

summary = {
    "target": "2D Van der Pol vector field f(x, v) = [v, mu * (1 - x^2) * v - x]",
    "mu": cfg.mu,
    "parameter_count": parameter_count(model),
    "final_train_mse_clean": final_train_mse,
    "final_test_mse_clean": final_test_mse,
    "mean_grid_vector_error": mean_grid_error,
    "max_grid_vector_error": max_grid_error,
    "final_spearman_init_final": spearman_corr(S_init, S_final),
    "final_mass_in_init_topk": mass_on_indices(S_final, init_topk_idx),
    "reference_epoch": ref_epoch if S_ref is not None else None,
    "final_spearman_ref_final": spearman_corr(S_ref, S_final) if S_ref is not None else None,
    "final_mass_in_ref_topk": mass_on_indices(S_final, ref_topk_idx) if S_ref is not None else None,
    "largest_initial_eigenvalue": float(eig_init[0].item()),
    "largest_ref_eigenvalue": float(eig_ref[0].item()) if eig_ref is not None else None,
    "largest_final_eigenvalue": float(eig_final[0].item()),
}

with open("Plots/final_summary_2d_vanderpol.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nFinal summary")
print("=============")
print(json.dumps(summary, indent=2))
