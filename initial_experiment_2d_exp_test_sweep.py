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

def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))

# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    seed: int = _env_int("SEED", 0)

    # Training samples are random points in phase space: input = [x, y].
    n_train: int = 2**12
    n_test: int = 2**10

    # # Sensitivity/Jacobian calculations are expensive for vector outputs, so use
    # # a fixed subset for diagnostics rather than the full training set.
    # n_sensitivity: int = 128

    n_hidden: int = _env_int("N_HIDDEN", 2)
    hidden_width: int = _env_int("HIDDEN_WIDTH", 32)
    lr: float = 1e-2
    epochs: int = _env_int("EPOCHS", 100000)
    checkpoint_interval: int = epochs // 40
    topk_frac: float = 0.10
    compare_epoch: int = epochs // 40

    # Symmetric target field and sampling domain.
    x_min: float = -2.0
    x_max: float = 2.0
    v_min: float = -2.0
    v_max: float = 2.0

    field_scale: float = 1.0e1

    noise_multiplier: float = 5e-2
    plot_grid_size: int = 45


cfg = Config()

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

os.makedirs("Plots", exist_ok=True)


# ============================================================
# Dataset: learn a physically symmetric 2D vector field
# ============================================================
# Input:  state = [x, y]
# Target: F(x, y) = [-2x exp(-x^2 - y^2), 2y exp(-x^2 - y^2)]
# This is the gradient of phi(x, y) = exp(-x^2 + y^2).
#

def symmetric_vector_field(state, field_scale=1.0):
    x = state[..., 0:1]
    y = state[..., 1:2]

    phi = torch.exp((-x ** 2) + (-y ** 2))
    dfdx = -2.0 * field_scale * x * phi
    dfdy = 2.0 * field_scale * y * phi

    return torch.cat([dfdx, dfdy], dim=-1)


def sample_phase_space(n, x_min=-3.0, x_max=3.0, v_min=-3.0, v_max=3.0):
    x = x_min + (x_max - x_min) * torch.rand(n, 1, dtype=torch.float32)
    y = v_min + (v_max - v_min) * torch.rand(n, 1, dtype=torch.float32)
    return torch.cat([x, y], dim=-1)


x_train = sample_phase_space(cfg.n_train).to(device)
y_train_clean = symmetric_vector_field(x_train, field_scale=cfg.field_scale)
y_train = y_train_clean + cfg.noise_multiplier * torch.randn_like(y_train_clean)

x_test = sample_phase_space(cfg.n_test).to(device)
y_test_clean = symmetric_vector_field(x_test, field_scale=cfg.field_scale)
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


x_min = torch.tensor([cfg.x_min, cfg.v_min], device=device)
x_max = torch.tensor([cfg.x_max, cfg.v_max], device=device)

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

def physics_informed_loss(pred, state, field_scale):
    """Residual loss for the symmetric gradient-like vector field.

    The model predicts [F_x, F_y] for
        F(x, y) = [-2x exp(-x^2 + y^2), 2y exp(-x^2 + y^2)]
    so the loss penalizes the residual of those equations directly.
    """
    x = state[..., 0:1]
    y = state[..., 1:2]

    phi = torch.exp((-x ** 2) + (y ** 2))
    target_x = -2.0 * field_scale * x * phi
    target_y = 2.0 * field_scale * y * phi

    residual_x = pred[..., 0:1] - target_x
    residual_y = pred[..., 1:2] - target_y

    return (residual_x.pow(2).mean() + residual_y.pow(2).mean())

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
    "mean_abs_sensitivity": [],
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
    task_loss = criterion(pred, y_train)
    loss = task_loss

    loss.backward()
    optimizer.step()

    if epoch % cfg.checkpoint_interval == 0:
        model.eval()

        with torch.no_grad():
            train_loss = criterion(pred, y_train_clean).item()
            test_loss = criterion(model(x_test), y_test_clean).item()

        J = compute_parameter_jacobian(model, x_sens)
        S_curr = sensitivity_scores(J)

        rho_init_curr = spearman_corr(S_init, S_curr)
        init_topk_mass = mass_on_indices(S_curr, init_topk_idx)
        mean_abs_sens = mean_abs_sensitivity_by_output(model, x_sens).detach().cpu().numpy()

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
        history["mean_abs_sensitivity"].append(mean_abs_sens)

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

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


ACCENT_BLUE = "#4C72B0"
ACCENT_TEAL = "#55A868"
ACCENT_ORANGE = "#C44E52"
ACCENT_PURPLE = "#8172B3"
ACCENT_GRAY = "#6C757D"

grid_x = torch.linspace(cfg.x_min, cfg.x_max, cfg.plot_grid_size, dtype=torch.float32, device=device)
grid_y = torch.linspace(cfg.v_min, cfg.v_max, cfg.plot_grid_size, dtype=torch.float32, device=device)
X, Y = torch.meshgrid(grid_x, grid_y, indexing="xy")
grid_state = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)

with torch.no_grad():
    true_field = symmetric_vector_field(grid_state, field_scale=cfg.field_scale)
    pred_field = model(grid_state)
    field_error = torch.linalg.norm(pred_field - true_field, dim=-1)

X_np = X.detach().cpu().numpy()
Y_np = Y.detach().cpu().numpy()
true_np = true_field.detach().cpu().numpy().reshape(cfg.plot_grid_size, cfg.plot_grid_size, 2)
pred_np = pred_field.detach().cpu().numpy().reshape(cfg.plot_grid_size, cfg.plot_grid_size, 2)
err_np = field_error.detach().cpu().numpy().reshape(cfg.plot_grid_size, cfg.plot_grid_size)

skip = max(1, cfg.plot_grid_size // 20)

# Precompute magnitudes and use one shared normalization for both plots
true_mag = np.sqrt(true_np[..., 0]**2 + true_np[..., 1]**2)
pred_mag = np.sqrt(pred_np[..., 0]**2 + pred_np[..., 1]**2)

vmax = max(true_mag.max(), pred_mag.max())
norm = plt.Normalize(vmin=0.0, vmax=vmax)
cmap = "cividis"

# 1. True vector field
fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=False)
q = ax.quiver(
    X_np[::skip, ::skip], Y_np[::skip, ::skip],
    true_np[::skip, ::skip, 0], true_np[::skip, ::skip, 1],
    true_mag[::skip, ::skip],
    angles="xy",
    scale_units="xy",
    scale=22.0,          # larger = shorter arrows
    width=0.0022,
    headwidth=3.0,
    headlength=4.0,
    headaxislength=3.5,
    cmap=cmap,
    norm=norm,
    alpha=0.95,
)
cbar = fig.colorbar(q, ax=ax, pad=0.02)
cbar.set_label(r"$\|f(x,y)\|$")
ax.set_title("True symmetric vector field")
ax.set_xlabel("x")
ax.set_ylabel("y")
prettify_axes(ax)
save_pub_figure(fig, f"Plots/True_Symmetric_Vector_Field_{parameter_count(model)}_Parameters_{cfg.n_hidden}_Depth_{cfg.hidden_width}_Width.pdf")

# 2. Learned vector field
fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=False)
q = ax.quiver(
    X_np[::skip, ::skip], Y_np[::skip, ::skip],
    pred_np[::skip, ::skip, 0], pred_np[::skip, ::skip, 1],
    pred_mag[::skip, ::skip],
    angles="xy",
    scale_units="xy",
    scale=22.0,
    width=0.0022,
    headwidth=3.0,
    headlength=4.0,
    headaxislength=3.5,
    cmap=cmap,
    norm=norm,
    alpha=0.95,
)
cbar = fig.colorbar(q, ax=ax, pad=0.02)
cbar.set_label(r"$\|f_\theta(x,y)\|$")
ax.set_title("Learned symmetric vector field")
ax.set_xlabel("x")
ax.set_ylabel("y")
prettify_axes(ax)
save_pub_figure(fig, f"Plots/Learned_Symmetric_Vector_Field_{parameter_count(model)}_Parameters_{cfg.n_hidden}_Depth_{cfg.hidden_width}_Width.pdf")

# 3. Error heatmap
fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=False)
im_err = ax.imshow(
    err_np,
    extent=[cfg.x_min, cfg.x_max, cfg.v_min, cfg.v_max],
    origin="lower",
    aspect="auto",
    cmap="magma",
)
ax.set_title("Vector-field error norm")
ax.set_xlabel("x")
ax.set_ylabel("y")
prettify_axes(ax)
cbar = fig.colorbar(im_err, ax=ax, pad=0.02, fraction=0.046)
cbar.ax.tick_params(direction="in", length=4, width=0.7)
cbar.outline.set_linewidth(0.8)
save_pub_figure(fig, f"Plots/Symmetric_Vector_Field_Error_{parameter_count(model)}_Parameters_{cfg.n_hidden}_Depth_{cfg.hidden_width}_Width.pdf")

# 6. Eigenspectrum
eig_final_erank = effective_rank(eig_final)

fig, ax = plt.subplots(figsize=(7.4, 6.0), constrained_layout=False)
ax.plot(eig_init.detach().cpu().numpy(), label="initial", color=ACCENT_BLUE)
if eig_ref is not None:
    ax.plot(eig_ref.detach().cpu().numpy(), label=f"ref @ epoch {ref_epoch}", color=ACCENT_PURPLE, linestyle="--")
ax.plot(eig_final.detach().cpu().numpy(), label="final", color=ACCENT_ORANGE)
ax.set_yscale("log")
ax.set_title(
    f"Sensitivity covariance eigenspectrum "
    f"(effective rank = {eig_final_erank:.2f})"
)
ax.set_xlabel("Parameter  Index")
ax.set_ylabel("Eigenvalue")
prettify_axes(ax)
ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
beautify_legend(ax, loc="best")
save_pub_figure(fig, f"Plots/Eigenspectrum_{parameter_count(model)}_Parameters_{cfg.n_hidden}_Depth_{cfg.hidden_width}_Width.pdf")

# 7. Loss curves
fig, ax = plt.subplots(figsize=(7.4, 6.0), constrained_layout=False)
ax.plot(history["epoch"], history["train_loss"], label="train", color=ACCENT_BLUE)
ax.plot(history["epoch"], history["test_loss"], label="test", color=ACCENT_ORANGE)
ax.set_yscale("log")
ax.set_title("Loss evolution")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE loss")
prettify_axes(ax)
ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
beautify_legend(ax, loc="best")
save_pub_figure(fig, f"Plots/Loss_Evolution_{parameter_count(model)}_Parameters_{cfg.n_hidden}_Depth_{cfg.hidden_width}_Width.pdf")


# ============================================================
# Separate sensitivity distribution figure
# ============================================================

eps = 1e-30

# Normalize so the two distributions are comparable on the same scale.
S_init_dist = S_init.detach().cpu().numpy()
S_final_dist = S_final.detach().cpu().numpy()

# Use common log-spaced bins over the positive support.
positive_vals = np.concatenate([
    S_init_dist[S_init_dist > 0],
    S_final_dist[S_final_dist > 0],
])
positive_vals = positive_vals[np.isfinite(positive_vals)]

if positive_vals.size > 0:
    bins = np.logspace(
        np.log10(positive_vals.min()),
        np.log10(positive_vals.max()),
        50,
    )
else:
    bins = 50

fig3, ax3 = plt.subplots(figsize=(8.8, 5.8), constrained_layout=False)

ax3.hist(
    S_init_dist,
    bins=bins,
    density=True,
    alpha=0.42,
    label="initial",
    histtype="stepfilled",
    edgecolor=ACCENT_BLUE,
    linewidth=0.9,
    color=ACCENT_BLUE,
)
ax3.hist(
    S_final_dist,
    bins=bins,
    density=True,
    alpha=0.42,
    label="final",
    histtype="stepfilled",
    edgecolor=ACCENT_ORANGE,
    linewidth=0.9,
    color=ACCENT_ORANGE,
)


ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_title("Distribution of normalized sensitivities")
ax3.set_xlabel("Sensitivity mass fraction")
ax3.set_ylabel("Density")
prettify_axes(ax3)
ax3.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax3.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax3.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax3.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
beautify_legend(ax3, loc="best")
save_pub_figure(
    fig3,
    f"Plots/Sensitivity_Distribution_Initial_vs_Final_"
    f"{parameter_count(model)}_Parameters_{cfg.n_hidden}_Depth_{cfg.hidden_width}_Width.pdf",
)



# ============================================================
# Separate plot: unnormalised sensitivity vs parameter magnitude
# ============================================================


param_mag_init = flatten_param_magnitudes(initial_model)
param_mag_final = flatten_param_magnitudes(model)

sens_init_red = reduce_sensitivity_to_parameter_level(initial_model, S_init)
sens_final_red = reduce_sensitivity_to_parameter_level(model, S_final)

# Safety clamp for log axes.
eps = 1e-30
x_init = param_mag_init.clamp_min(eps).cpu().numpy()
y_init = sens_init_red.clamp_min(eps).cpu().numpy()

x_final = param_mag_final.clamp_min(eps).cpu().numpy()
y_final = sens_final_red.clamp_min(eps).cpu().numpy()

fig4, ax4 = plt.subplots(figsize=(8.2, 6.0), constrained_layout=False)

ax4.scatter(
    x_init,
    y_init,
    s=9,
    alpha=0.35,
    label="initial",
    marker="o",
    linewidths=0.0,
    color=ACCENT_BLUE,
    rasterized=True,
)

ax4.scatter(
    x_final,
    y_final,
    s=14,
    alpha=0.38,
    label="final",
    marker="x",
    linewidths=0.9,
    color=ACCENT_ORANGE,
    rasterized=True,
)


ax4.set_xscale("log")
ax4.set_yscale("log")
ax4.set_title("Unnormalised sensitivity vs parameter magnitude")
ax4.set_xlabel(r"$|	\theta_i|$")
ax4.set_ylabel(r"$S(\theta_i) $")
prettify_axes(ax4)
ax4.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax4.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax4.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax4.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
beautify_legend(ax4, loc="best")
save_pub_figure(
    fig4,
    f"Plots/Sensitivity_vs_Parameter_Magnitude_"
    f"{parameter_count(model)}_Parameters_{cfg.n_hidden}_Depth_{cfg.hidden_width}_Width.pdf",
)


# ============================================================
# Absolute sensitivity evolution
# ============================================================
Q_abs = np.stack(history["mean_abs_sensitivity"], axis=0)
checkpoint_epochs = np.array(history["epoch"], dtype=int)
n_checkpoints, n_outputs, n_params = Q_abs.shape

vmax = np.nanmax(Q_abs)
if not np.isfinite(vmax) or vmax <= 0:
    vmax = 1.0

fig, axes = plt.subplots(
    1,
    n_outputs,
    figsize=(5.0 * n_outputs + 1.0, 4.0),
    constrained_layout=True,
    sharex=True,
)
if n_outputs == 1:
    axes = [axes]

im = None
for out_idx, ax in enumerate(axes):
    im = ax.imshow(
        Q_abs[:, out_idx, :].T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
        extent=[checkpoint_epochs[0], checkpoint_epochs[-1], -0.5, n_params - 0.5],
    )
    # ax.set_title(
    #     rf"Output {out_idx}: $\mathbb{{E}}_x[|\partial f_{{{out_idx}}} / \partial \theta_i|]$",
    #     fontsize=9,
    #     pad=6,
    # )
    ax.set_ylabel("Parameter index")
    prettify_axes(ax)

for ax in axes:
    ax.set_xlabel("Iteration number")

# Optional: reduce tick clutter
tick_step = max(1, len(checkpoint_epochs) // 6)
xticks = checkpoint_epochs[::tick_step]
for ax in axes:
    ax.set_xticks(xticks)

cbar = fig.colorbar(im, ax=axes[-1], pad=0.04, fraction=0.046)
cbar.set_label(r"mean $|\partial f_k / \partial \theta_i|$")
cbar.ax.tick_params(direction="in", length=4, width=0.7)
cbar.outline.set_linewidth(0.8)

save_pub_figure(
    fig,
    f"Plots/Absolute_Sensitivity_Over_Training_"
    f"{parameter_count(model)}_Parameters_{cfg.n_hidden}_Depth_{cfg.hidden_width}_Width.pdf",
)







# ============================================================
# Low-dimensional structure analysis
# ============================================================

from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# ------------------------------------------------------------
# Prepare data
# ------------------------------------------------------------

# Jacobian:
# shape = [n_samples * output_dim, n_parameters]
J_np = J_final.detach().cpu().numpy()

# Covariance
C_np = C_final.detach().cpu().numpy()

# Sensitivity vector
S_np = S_final.detach().cpu().numpy()

# Standardize Jacobian rows
J_scaled = StandardScaler().fit_transform(J_np)

print("\n")
print("===================================================")
print("Low-dimensional structure diagnostics")
print("===================================================")

print(f"Jacobian shape: {J_scaled.shape}")
print(f"Covariance shape: {C_np.shape}")
print(f"Sensitivity shape: {S_np.shape}")


# ============================================================
# PCA on Jacobian rows
# ============================================================

pca = PCA(n_components=min(64, J_scaled.shape[1]))
J_pca = pca.fit_transform(J_scaled)

explained = pca.explained_variance_ratio_
cum_explained = np.cumsum(explained)

effective_dim_95 = np.searchsorted(cum_explained, 0.95) + 1

print(f"\nPCA effective dimension (95% variance): {effective_dim_95}")

fig, ax = plt.subplots(figsize=(7.2, 5.5))

ax.plot(cum_explained, linewidth=2.0)
ax.axhline(0.95, linestyle="--")

ax.set_title("Jacobian PCA cumulative explained variance")
ax.set_xlabel("Principal component")
ax.set_ylabel("Cumulative explained variance")

prettify_axes(ax)

save_pub_figure(
    fig,
    f"Plots/Jacobian_PCA_Cumulative_Variance_"
    f"{parameter_count(model)}.pdf",
)


# ------------------------------------------------------------
# PCA scatter
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6.5, 6.0))

ax.scatter(
    J_pca[:, 0],
    J_pca[:, 1],
    s=10,
    alpha=0.55,
    rasterized=True,
)

ax.set_title("Jacobian PCA projection")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")

prettify_axes(ax)

save_pub_figure(
    fig,
    f"Plots/Jacobian_PCA_Projection_"
    f"{parameter_count(model)}.pdf",
)


# ============================================================
# Isomap
# ============================================================

print("Running Isomap...")

isomap = Isomap(
    n_components=2,
    n_neighbors=20,
)

J_iso = isomap.fit_transform(J_scaled)

fig, ax = plt.subplots(figsize=(6.5, 6.0))

ax.scatter(
    J_iso[:, 0],
    J_iso[:, 1],
    s=10,
    alpha=0.55,
    rasterized=True,
)

ax.set_title("Jacobian Isomap embedding")
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")

prettify_axes(ax)

save_pub_figure(
    fig,
    f"Plots/Jacobian_Isomap_Embedding_"
    f"{parameter_count(model)}.pdf",
)


# ============================================================
# t-SNE
# ============================================================

print("Running t-SNE...")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    init="pca",
    learning_rate="auto",
    random_state=cfg.seed,
)

J_tsne = tsne.fit_transform(J_scaled)

fig, ax = plt.subplots(figsize=(6.5, 6.0))

ax.scatter(
    J_tsne[:, 0],
    J_tsne[:, 1],
    s=10,
    alpha=0.55,
    rasterized=True,
)

ax.set_title("Jacobian t-SNE embedding")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")

prettify_axes(ax)

save_pub_figure(
    fig,
    f"Plots/Jacobian_tSNE_Embedding_"
    f"{parameter_count(model)}.pdf",
)


# ============================================================
# UMAP
# ============================================================

if HAS_UMAP:

    print("Running UMAP...")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=25,
        min_dist=0.1,
        metric="euclidean",
        random_state=cfg.seed,
    )

    J_umap = reducer.fit_transform(J_scaled)

    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    ax.scatter(
        J_umap[:, 0],
        J_umap[:, 1],
        s=10,
        alpha=0.55,
        rasterized=True,
    )

    ax.set_title("Jacobian UMAP embedding")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    prettify_axes(ax)

    save_pub_figure(
        fig,
        f"Plots/Jacobian_UMAP_Embedding_"
        f"{parameter_count(model)}.pdf",
    )


# ============================================================
# Covariance eigenspectrum diagnostics
# ============================================================

eigvals = eig_final.detach().cpu().numpy()

eigvals = eigvals[eigvals > 1e-14]

participation_ratio = (
    (eigvals.sum() ** 2) /
    (np.square(eigvals).sum())
)

print(f"\nParticipation ratio dimension: {participation_ratio:.3f}")

fig, ax = plt.subplots(figsize=(7.0, 5.5))

ax.plot(
    eigvals / eigvals.max(),
    linewidth=2.0,
)

ax.set_yscale("log")

ax.set_title("Normalized covariance eigenspectrum")
ax.set_xlabel("Eigenvalue index")
ax.set_ylabel(r"$\lambda_i / \lambda_{\max}$")

prettify_axes(ax)

save_pub_figure(
    fig,
    f"Plots/Normalized_Covariance_Eigenspectrum_"
    f"{parameter_count(model)}.pdf",
)


# ============================================================
# Distance preservation diagnostics
# ============================================================

print("Computing pairwise distance diagnostics...")

subset_size = min(512, J_scaled.shape[0])

subset_idx = np.random.choice(
    J_scaled.shape[0],
    subset_size,
    replace=False,
)

X_sub = J_scaled[subset_idx]
P_sub = J_pca[subset_idx]

D_high = pairwise_distances(X_sub)
D_low = pairwise_distances(P_sub)

corr = np.corrcoef(
    D_high.ravel(),
    D_low.ravel(),
)[0, 1]

print(f"Distance correlation (PCA): {corr:.4f}")

fig, ax = plt.subplots(figsize=(6.2, 6.0))

ax.scatter(
    D_high.ravel(),
    D_low.ravel(),
    s=1,
    alpha=0.15,
    rasterized=True,
)

ax.set_title(
    f"PCA distance preservation\ncorr={corr:.3f}"
)

ax.set_xlabel("High-dimensional distance")
ax.set_ylabel("Low-dimensional distance")

prettify_axes(ax)

save_pub_figure(
    fig,
    f"Plots/PCA_Distance_Preservation_"
    f"{parameter_count(model)}.pdf",
)


# ============================================================
# Sensitivity ordering structure
# ============================================================

sorted_sens = np.sort(S_np)[::-1]

fig, ax = plt.subplots(figsize=(7.0, 5.5))

ax.plot(sorted_sens)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_title("Sorted sensitivity spectrum")
ax.set_xlabel("Parameter rank")
ax.set_ylabel("Sensitivity")

prettify_axes(ax)

save_pub_figure(
    fig,
    f"Plots/Sensitivity_Rank_Spectrum_"
    f"{parameter_count(model)}.pdf",
)







# ============================================================
# Final summary
# ============================================================

with torch.no_grad():
    final_train_mse = criterion(model(x_train), y_train_clean).item()
    final_test_mse = criterion(model(x_test), y_test_clean).item()
    mean_grid_error = float(field_error.mean().detach().cpu().item())
    max_grid_error = float(field_error.max().detach().cpu().item())

summary = {
    "target": "2D symmetric vector field F(x, y) = [-2x exp(-x^2 + y^2), 2y exp(-x^2 + y^2)]",
    "field_scale": cfg.field_scale,
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

with open(f"Plots/final_summary_2d_symmetric_field_{parameter_count(model)}_Parameters_{cfg.n_hidden}_Depth_{cfg.hidden_width}_Width.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nFinal summary")
print("=============")
print(json.dumps(summary, indent=2))