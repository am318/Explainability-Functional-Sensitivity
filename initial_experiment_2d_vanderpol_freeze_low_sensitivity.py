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

from utilities import * 

# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    seed: int = 0

    # Training samples are random points in phase space: input = [x, v].
    n_train: int = 2**12
    n_test: int = 2**10

    # # Sensitivity/Jacobian calculations are expensive for vector outputs, so use
    # # a fixed subset for diagnostics rather than the full training set.
    # n_sensitivity: int = 128

    n_hidden: int = 2 # 4
    hidden_width: int = 48 # 64
    lr: float = 1e-3
    epochs: int = 100000
    checkpoint_interval: int = epochs // 25
    topk_frac: float = 0.10
    compare_epoch: int = epochs // 25

    # Van der Pol oscillator parameter and phase-space domain.
    mu: float = 3.0
    x_min: float = -3.0
    x_max: float = 3.0
    v_min: float = -5.0
    v_max: float = 5.0

    noise_multiplier: float = 5e-2
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

# # Fixed subset for sensitivity analysis.
# sens_idx = torch.randperm(cfg.n_train, device=device)[:cfg.n_sensitivity]
# x_sens = x_train[sens_idx]

# Full training set for exact sensitivity analysis.
x_sens = x_train


# ============================================================
# Model
# ============================================================

class InputNormalizer(nn.Module):
    def __init__(self, x_min, x_max, eps=1e-12):
        super().__init__()

        x_min = torch.as_tensor(x_min, dtype=torch.float32)
        x_max = torch.as_tensor(x_max, dtype=torch.float32)

        self.register_buffer("x_min", x_min)
        self.register_buffer("x_max", x_max)
        self.eps = eps

    def forward(self, x):
        return 2.0 * (x - self.x_min) / (self.x_max - self.x_min + self.eps) - 1.0


class SmallMLP(nn.Module):
    def __init__(self, width, x_min, x_max, input_dim=2, output_dim=2):
        super().__init__()

        self.input_normalizer = InputNormalizer(x_min, x_max)

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
        x = self.input_normalizer(x)
        return self.net(x)


x_min = torch.tensor([0.0, -1.0], device=device)
x_max = torch.tensor([1.0,  1.0], device=device)

model = SmallMLP(
    cfg.hidden_width,
    x_min=x_min,
    x_max=x_max,
    input_dim=2,
    output_dim=2,
).to(device)

# ============================================================
# Initial state and initial sensitivities
# ============================================================

initial_state = copy.deepcopy(model.state_dict())

initial_model = SmallMLP(
    cfg.hidden_width,
    x_min=x_min,
    x_max=x_max,
    input_dim=2,
    output_dim=2,
).to(device)
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

fig, axes = plt.subplots(3, 3, figsize=(20, 16))

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

# 4. Sensitivity covariance at init
eps = 1e-30

def normalize_covariance(C):
    d = torch.diag(C)
    scale = torch.sqrt(d.clamp_min(eps))
    return C / (scale[:, None] * scale[None, :] + eps)

C_init_n = normalize_covariance(C_init)
C_final_n = normalize_covariance(C_final)

im0 = axes[1, 0].imshow(C_init_n.detach().cpu().numpy(),
                        aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)

axes[1, 0].set_title("Sensitivity covariance at init")
axes[1, 0].set_xlabel("parameter index")
axes[1, 0].set_ylabel("parameter index")
plt.colorbar(im0, ax=axes[1, 0])

# 5. Sensitivity covariance difference after training
im1 = axes[1, 1].imshow(np.abs((C_final_n - C_init_n).detach().cpu().numpy()),
                        aspect="auto", cmap="magma")
axes[1, 1].set_title("Sensitivity covariance difference after training")
axes[1, 1].set_xlabel("parameter index")
axes[1, 1].set_ylabel("parameter index")
plt.colorbar(im1, ax=axes[1, 1])

# 6. Eigenspectrum
axes[1, 2].plot((eig_init/(eig_init.max() + 1e-30)).detach().cpu().numpy(), label="initial")
if eig_ref is not None:
    axes[1, 2].plot((eig_ref/(eig_ref.max() + 1e-30)).detach().cpu().numpy(), label=f"ref @ epoch {ref_epoch}")
axes[1, 2].plot((eig_final/(eig_final.max()+ 1e-30)).detach().cpu().numpy(), label="final")
axes[1, 2].set_yscale("log")
axes[1, 2].set_title("Sensitivity covariance eigenspectrum")
axes[1, 2].legend()
axes[1, 2].set_xlabel('Number')
axes[1, 2].set_ylabel('Eigenvalue')

# 7. Loss curves
axes[2, 0].plot(history["epoch"], history["train_loss"], label="train")
axes[2, 0].plot(history["epoch"], history["test_loss"], label="test")
axes[2, 0].set_yscale("log")
axes[2, 0].set_title("Loss evolution")
axes[2, 0].set_xlabel("epoch")
axes[2, 0].set_ylabel("MSE Loss")
axes[2, 0].legend()

# 8. Sensitivity persistence
ax = axes[2, 1]
ax2 = ax.twinx()
l1 = ax.plot(history["epoch"], history["spearman_init_current"], label="Spearman(init, current)", color='b')
l2 = []
l3 = []
if any(~np.isnan(np.array(history["spearman_ref_current"], dtype=np.float64))):
    l2 = ax.plot(history["epoch"], history["spearman_ref_current"], label="Spearman(ref, current)", color='g')
    l3 = ax2.plot(history["epoch"], history["ref_topk_mass"], linestyle="--", label="Mass in ref top-k", color='g')
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

# 9. Early-vs-final normalised sensitivity scores
eps = 1e-30
# low_frac = 0.25
low_frac = 1-cfg.topk_frac

# S_early = S_ref if S_ref is not None else S_init

S_early = S_init

S_early_norm = S_early / (S_early.sum() + eps)
S_final_norm = S_final / (S_final.sum() + eps)

early_cutoff = torch.quantile(S_early_norm, low_frac)
final_cutoff = torch.quantile(S_final_norm, low_frac)

persistently_low_mask = (
    (S_early_norm <= early_cutoff) &
    (S_final_norm <= final_cutoff)
)

x_np = S_early_norm.detach().cpu().numpy()
y_np = S_final_norm.detach().cpu().numpy()
mask_np = persistently_low_mask.detach().cpu().numpy()

axes[2, 2].scatter(
    x_np[~mask_np],
    y_np[~mask_np],
    s=8,
    alpha=0.35,
    label="other parameters",
    marker = 'o',
    color = 'g'
)

axes[2, 2].scatter(
    x_np[mask_np],
    y_np[mask_np],
    s=10,
    alpha=0.9,
    label="persistently low",
    marker = 'x',
    color = 'b'
)

lims = [
    min(x_np.min(), y_np.min()),
    max(x_np.max(), y_np.max())
]

axes[2, 2].plot(
    lims,
    lims,
    linestyle=":",
    linewidth=1.2,
    color="k",
    label="y = x"
)

axes[2, 2].axvline(
    early_cutoff.detach().cpu().item(),
    linestyle="--",
    linewidth=1.2,
    color = 'r'
)

axes[2, 2].axhline(
    final_cutoff.detach().cpu().item(),
    linestyle="--",
    linewidth=1.2,
    color='r'
)

axes[2, 2].set_xscale("log")
axes[2, 2].set_yscale("log")
axes[2, 2].set_title("Persistently low sensitivity")
axes[2, 2].set_xlabel("initial sensitivity mass fraction")
axes[2, 2].set_ylabel("final sensitivity mass fraction")
axes[2, 2].legend()

plt.tight_layout()
plt.savefig(f"Plots/Initial_Experiment_2D_VanderPol_freeze_low_sensitivity_{parameter_count(model)}_Parameters.pdf")
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

with open(f"Plots/final_summary_2d_vanderpol_freeze_low_sensitivity_{parameter_count(model)}_Parameters.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nFinal summary")
print("=============")
print(json.dumps(summary, indent=2))
