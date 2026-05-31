import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from datasets import (
    available_datasets,
    get_dataset_spec,
    make_dataset_for_sweep,
)
from utilities import (
    beautify_legend,
    compute_covariance,
    compute_parameter_jacobian,
    effective_rank,
    eigvals_from_covariance,
    flatten_param_magnitudes,
    mass_on_indices,
    mean_abs_sensitivity_by_output,
    parameter_count,
    prettify_axes,
    reduce_sensitivity_to_parameter_level,
    save_pub_figure,
    sensitivity_scores,
    spearman_corr,
    topk_indices,
)


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


def _env_str(name, default):
    return os.environ.get(name, default)


@dataclass
class Config:
    seed: int = _env_int("SEED", 0)
    dataset_name: str = _env_str("DATASET", "parabola")

    n_train: int = _env_int("N_TRAIN", 2**12)
    n_test: int = _env_int("N_TEST", 2**10)
    n_sensitivity: int = _env_int("N_SENSITIVITY", 0)

    n_hidden: int = _env_int("N_HIDDEN", 2)
    hidden_width: int = _env_int("HIDDEN_WIDTH", 8)
    lr: float = _env_float("LR", 1e-2)
    epochs: int = _env_int("EPOCHS", 10000)
    checkpoint_interval: int = _env_int("CHECKPOINT_INTERVAL", 0)
    compare_epoch: int = _env_int("COMPARE_EPOCH", 0)
    topk_frac: float = _env_float("TOPK_FRAC", 0.10)
    l1_lambda: float = _env_float("L1_LAMBDA", 1e-3)

    x_min: float | None = None
    x_max: float | None = None
    v_min: float | None = None
    v_max: float | None = None

    field_scale: float = _env_float("FIELD_SCALE", 1.0)
    noise_multiplier: float = _env_float("NOISE_MULTIPLIER", 5e-2)
    plot_grid_size: int = _env_int("PLOT_GRID_SIZE", 45)
    jacobian_chunk_size: int = _env_int("JACOBIAN_CHUNK_SIZE", 64)
    run_manifold: bool = _env_int("RUN_MANIFOLD", 0) != 0

    def __post_init__(self):
        self.checkpoint_interval = self.checkpoint_interval or max(1, self.epochs // 20)
        self.compare_epoch = self.compare_epoch or max(1, self.epochs // 20)
        self.x_min = _env_float("X_MIN", self.x_min) if "X_MIN" in os.environ else self.x_min
        self.x_max = _env_float("X_MAX", self.x_max) if "X_MAX" in os.environ else self.x_max
        self.v_min = _env_float("V_MIN", self.v_min) if "V_MIN" in os.environ else self.v_min
        self.v_max = _env_float("V_MAX", self.v_max) if "V_MAX" in os.environ else self.v_max


cfg = Config()

if cfg.dataset_name not in available_datasets():
    raise ValueError(
        f"Unknown DATASET={cfg.dataset_name!r}. Available datasets: {', '.join(available_datasets())}"
    )

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

spec = get_dataset_spec(cfg.dataset_name)
output_dir = Path(os.getenv("OUTPUT_DIR", f"Plots/{spec.name}"))
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Using device: {device}")
print(f"Using dataset: {spec.name} ({spec.description})")

# ============================================================
# Dataset
# ============================================================

generator = torch.Generator(device="cpu")
generator.manual_seed(cfg.seed)

sensitivity_size = cfg.n_sensitivity if cfg.n_sensitivity > 0 else None
split = make_dataset_for_sweep(
    spec.name,
    cfg.n_train,
    cfg.n_test,
    x_min=cfg.x_min,
    x_max=cfg.x_max,
    v_min=cfg.v_min,
    v_max=cfg.v_max,
    field_scale=cfg.field_scale,
    noise_multiplier=cfg.noise_multiplier,
    sensitivity_size=sensitivity_size,
    generator=generator,
    device=None,
)

x_train = split.x_train.to(device)
y_train = split.y_train.to(device)
x_test = split.x_test.to(device)
y_test = split.y_test.to(device)
y_train_clean = split.y_train_clean.to(device) if split.y_train_clean is not None else y_train
y_test_clean = split.y_test_clean.to(device) if split.y_test_clean is not None else y_test
x_sens = split.x_sens.to(device) if split.x_sens is not None else x_train

input_dim = x_train.shape[-1]
output_dim = y_train.shape[-1]

if input_dim != spec.input_dim or output_dim != spec.output_dim:
    raise RuntimeError(
        f"Dataset metadata mismatch for {spec.name}: spec says ({spec.input_dim}, {spec.output_dim}), "
        f"actual tensors are ({input_dim}, {output_dim})."
    )

# ============================================================
# Model
# ============================================================

class SmallMLP(nn.Module):
    def __init__(self, width, input_dim, output_dim, n_hidden):
        super().__init__()
        layers = [nn.Linear(input_dim, width), nn.LayerNorm(width), nn.SiLU()]
        for _ in range(n_hidden - 1):
            layers.extend([nn.Linear(width, width), nn.LayerNorm(width), nn.SiLU()])
        layers.append(nn.Linear(width, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model = SmallMLP(
    cfg.hidden_width,
    input_dim=input_dim,
    output_dim=output_dim,
    n_hidden=cfg.n_hidden,
).to(device)

initial_state = copy.deepcopy(model.state_dict())
initial_model = SmallMLP(
    cfg.hidden_width,
    input_dim=input_dim,
    output_dim=output_dim,
    n_hidden=cfg.n_hidden,
).to(device)
initial_model.load_state_dict(initial_state)
initial_model.eval()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

# ============================================================
# Initial sensitivities
# ============================================================

J_init = compute_parameter_jacobian(initial_model, x_sens, chunk_size=cfg.jacobian_chunk_size)
C_init = compute_covariance(J_init)
S_init = sensitivity_scores(J_init)
init_topk_idx = topk_indices(S_init, cfg.topk_frac)

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

print("\nTraining")
print("========")

for epoch in range(cfg.epochs + 1):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    pred = model(x_train)
    task_loss = criterion(pred, y_train)
    l1_penalty = sum(
        p.abs().sum()
        for name, p in model.named_parameters()
        if p.requires_grad and "bias" not in name
    )
    loss = task_loss + cfg.l1_lambda * l1_penalty
    loss.backward()
    optimizer.step()

    if epoch % cfg.checkpoint_interval == 0 or epoch == cfg.epochs:
        model.eval()
        with torch.no_grad():
            train_loss = criterion(model(x_train), y_train_clean).item()
            test_loss = criterion(model(x_test), y_test_clean).item()

        J = compute_parameter_jacobian(model, x_sens, chunk_size=cfg.jacobian_chunk_size)
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
# Final Jacobians and sensitivity summaries
# ============================================================

model.eval()
J_final = compute_parameter_jacobian(model, x_sens, chunk_size=cfg.jacobian_chunk_size)
C_final = compute_covariance(J_final)
S_final = sensitivity_scores(J_final)

eig_init = eigvals_from_covariance(C_init)
eig_final = eigvals_from_covariance(C_final)
eig_ref = eigvals_from_covariance(C_ref) if S_ref is not None else None

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


def _slug(text):
    return text.replace(" ", "_").replace("/", "_")


def parameter_location_metadata(model):
    labels = []
    spans = []
    cursor = 0
    for name, p in model.named_parameters():
        n = p.numel()
        labels.extend([name] * n)
        spans.append((cursor, cursor + n, name))
        cursor += n
    labels = np.asarray(labels, dtype=object)
    unique_labels = list(dict.fromkeys(labels.tolist()))
    cmap = plt.get_cmap("tab20")
    colour_map = {lab: cmap(i % cmap.N) for i, lab in enumerate(unique_labels)}
    colours = np.asarray([colour_map[lab] for lab in labels], dtype=object)
    return labels, colours, unique_labels, colour_map, spans


def add_parameter_location_legend(ax, unique_labels, colour_map, *, loc="best", max_labels=16):
    from matplotlib.lines import Line2D
    shown = unique_labels[:max_labels]
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=colour_map[label],
               markeredgecolor="none", markersize=5, label=label)
        for label in shown
    ]
    if len(unique_labels) > max_labels:
        handles.append(Line2D([0], [0], linestyle="None", label=f"+{len(unique_labels) - max_labels} more"))
    ax.legend(handles=handles, frameon=False, loc=loc, fontsize=7)


def add_parameter_location_boundaries(ax, spans, *, axis="y", color="white", alpha=0.55):
    for _, stop, _ in spans[:-1]:
        boundary = stop - 0.5
        if axis == "y":
            ax.axhline(boundary, color=color, linewidth=0.45, alpha=alpha)
        else:
            ax.axvline(boundary, color=color, linewidth=0.45, alpha=alpha)


param_location_labels, param_location_colours, param_location_unique, param_location_colour_map, param_location_spans = parameter_location_metadata(model)
run_tag = f"{spec.name}_{parameter_count(model)}_Parameters_{cfg.n_hidden}_Depth_{cfg.hidden_width}_Width"

# ============================================================
# Dataset-specific fit plots
# ============================================================

mean_grid_error = None
max_grid_error = None

if input_dim == 1:
    x_min_plot = float(x_train.detach().cpu().min())
    x_max_plot = float(x_train.detach().cpu().max())
    grid_x = torch.linspace(x_min_plot, x_max_plot, cfg.plot_grid_size, dtype=torch.float32, device=device).unsqueeze(1)
    with torch.no_grad():
        pred_y = model(grid_x)
        if spec.target_fn is not None and spec.name != "vanderpol_timeseries":
            true_y = cfg.field_scale * spec.target_fn(grid_x) if spec.name in {"parabola", "power", "sine", "symmetric_vector_field"} else spec.target_fn(grid_x)
        else:
            true_y = None

    fig, ax = plt.subplots(figsize=(7.4, 6.0), constrained_layout=False)
    x_np = grid_x.detach().cpu().numpy().squeeze(-1)
    pred_np = pred_y.detach().cpu().numpy().squeeze(-1)
    if true_y is not None:
        true_np = true_y.detach().cpu().numpy().squeeze(-1)
        ax.plot(x_np, true_np, label="target", color=ACCENT_BLUE)
        err_np = np.abs(pred_np - true_np)
        mean_grid_error = float(err_np.mean())
        max_grid_error = float(err_np.max())
    ax.plot(x_np, pred_np, label="learned", color=ACCENT_ORANGE)
    ax.scatter(
        x_train.detach().cpu().numpy().squeeze(-1),
        y_train_clean.detach().cpu().numpy().squeeze(-1),
        s=8,
        alpha=0.12,
        color=ACCENT_GRAY,
        label="train samples",
        rasterized=True,
    )
    ax.set_title(f"Learned fit: {spec.name}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    prettify_axes(ax)
    beautify_legend(ax, loc="best")
    save_pub_figure(fig, output_dir / f"Learned_Fit_{run_tag}.pdf")

    if true_y is not None:
        fig, ax = plt.subplots(figsize=(7.4, 6.0), constrained_layout=False)
        ax.plot(x_np, err_np, color=ACCENT_TEAL)
        ax.set_title(f"Absolute error: {spec.name}")
        ax.set_xlabel("x")
        ax.set_ylabel("absolute error")
        prettify_axes(ax)
        save_pub_figure(fig, output_dir / f"Fit_Absolute_Error_{run_tag}.pdf")

elif input_dim == 2 and output_dim == 2:
    with torch.no_grad():
        pred_test = model(x_test)
        vector_err = torch.linalg.norm(pred_test - y_test_clean, dim=1)
    mean_grid_error = float(vector_err.mean().item())
    max_grid_error = float(vector_err.max().item())

    fig, ax = plt.subplots(figsize=(7.0, 6.0), constrained_layout=False)
    xs = x_test.detach().cpu().numpy()
    ys = y_test_clean.detach().cpu().numpy()
    ps = pred_test.detach().cpu().numpy()
    n_plot = min(600, xs.shape[0])
    ax.quiver(xs[:n_plot, 0], xs[:n_plot, 1], ys[:n_plot, 0], ys[:n_plot, 1], angles="xy", scale_units="xy", scale=1, alpha=0.45)
    ax.quiver(xs[:n_plot, 0], xs[:n_plot, 1], ps[:n_plot, 0], ps[:n_plot, 1], angles="xy", scale_units="xy", scale=1, alpha=0.45)
    ax.set_title(f"Target and learned vector field: {spec.name}")
    ax.set_xlabel("x")
    ax.set_ylabel("v/y")
    prettify_axes(ax)
    save_pub_figure(fig, output_dir / f"Vector_Field_Fit_{run_tag}.pdf")

# ============================================================
# Shared plots
# ============================================================

fig, ax = plt.subplots(figsize=(7.4, 6.0), constrained_layout=False)
ax.plot(eig_init.detach().cpu().numpy(), label="initial", color=ACCENT_BLUE)
if eig_ref is not None:
    ax.plot(eig_ref.detach().cpu().numpy(), label=f"ref @ epoch {ref_epoch}", color=ACCENT_PURPLE, linestyle="--")
ax.plot(eig_final.detach().cpu().numpy(), label="final", color=ACCENT_ORANGE)
ax.set_yscale("log")
ax.set_title(f"Sensitivity covariance eigenspectrum (effective rank = {effective_rank(eig_final):.2f})")
ax.set_xlabel("Parameter index")
ax.set_ylabel("Eigenvalue")
prettify_axes(ax)
beautify_legend(ax, loc="best")
save_pub_figure(fig, output_dir / f"Eigenspectrum_{run_tag}.pdf")

fig, ax = plt.subplots(figsize=(7.4, 6.0), constrained_layout=False)
ax.plot(history["epoch"], history["train_loss"], label="train", color=ACCENT_BLUE)
ax.plot(history["epoch"], history["test_loss"], label="test", color=ACCENT_ORANGE)
ax.set_yscale("log")
ax.set_title("Loss evolution")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE loss")
prettify_axes(ax)
beautify_legend(ax, loc="best")
save_pub_figure(fig, output_dir / f"Loss_Evolution_{run_tag}.pdf")

S_init_dist = S_init.detach().cpu().numpy()
S_final_dist = S_final.detach().cpu().numpy()
positive_vals = np.concatenate([S_init_dist[S_init_dist > 0], S_final_dist[S_final_dist > 0]])
positive_vals = positive_vals[np.isfinite(positive_vals)]
bins = np.logspace(np.log10(positive_vals.min()), np.log10(positive_vals.max()), 50) if positive_vals.size else 50
fig, ax = plt.subplots(figsize=(8.8, 5.8), constrained_layout=False)
ax.hist(S_init_dist, bins=bins, density=True, alpha=0.42, label="initial", histtype="stepfilled", edgecolor=ACCENT_BLUE, linewidth=0.9, color=ACCENT_BLUE)
ax.hist(S_final_dist, bins=bins, density=True, alpha=0.42, label="final", histtype="stepfilled", edgecolor=ACCENT_ORANGE, linewidth=0.9, color=ACCENT_ORANGE)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("Distribution of sensitivities")
ax.set_xlabel("Sensitivity")
ax.set_ylabel("Density")
prettify_axes(ax)
beautify_legend(ax, loc="best")
save_pub_figure(fig, output_dir / f"Sensitivity_Distribution_Initial_vs_Final_{run_tag}.pdf")

param_mag_init = flatten_param_magnitudes(initial_model)
param_mag_final = flatten_param_magnitudes(model)
sens_init_red = reduce_sensitivity_to_parameter_level(initial_model, S_init)
sens_final_red = reduce_sensitivity_to_parameter_level(model, S_final)
eps = 1e-30
fig, ax = plt.subplots(figsize=(8.2, 6.0), constrained_layout=False)
ax.scatter(param_mag_init.clamp_min(eps).cpu().numpy(), sens_init_red.clamp_min(eps).cpu().numpy(), s=9, alpha=0.35, marker="o", linewidths=0.0, c=param_location_colours, rasterized=True)
ax.scatter(param_mag_final.clamp_min(eps).cpu().numpy(), sens_final_red.clamp_min(eps).cpu().numpy(), s=14, alpha=0.55, marker="x", linewidths=0.9, c=param_location_colours, rasterized=True)
add_parameter_location_legend(ax, param_location_unique, param_location_colour_map, loc="best")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("Unnormalised sensitivity vs parameter magnitude")
ax.set_xlabel(r"$|\theta_i|$")
ax.set_ylabel(r"$S(\theta_i)$")
prettify_axes(ax)
save_pub_figure(fig, output_dir / f"Sensitivity_vs_Parameter_Magnitude_{run_tag}.pdf")

Q_abs = np.stack(history["mean_abs_sensitivity"], axis=0)
checkpoint_epochs = np.array(history["epoch"], dtype=int)
n_checkpoints, n_outputs, n_params = Q_abs.shape
vmax = np.nanmax(Q_abs)
if not np.isfinite(vmax) or vmax <= 0:
    vmax = 1.0
fig, axes = plt.subplots(1, n_outputs, figsize=(5.0 * n_outputs + 1.0, 4.0), constrained_layout=True, sharex=True)
if n_outputs == 1:
    axes = [axes]
im = None
for out_idx, ax in enumerate(axes):
    im = ax.imshow(Q_abs[:, out_idx, :].T, aspect="auto", origin="lower", interpolation="nearest", cmap="magma", vmin=0.0, vmax=vmax, extent=[checkpoint_epochs[0], checkpoint_epochs[-1], -0.5, n_params - 0.5])
    ax.set_ylabel("Parameter index")
    ax.set_xlabel("Iteration number")
    add_parameter_location_boundaries(ax, param_location_spans, axis="y")
    prettify_axes(ax)
fig.colorbar(im, ax=axes[-1], pad=0.04, fraction=0.046).set_label(r"mean $|\partial f_k / \partial \theta_i|$")
save_pub_figure(fig, output_dir / f"Absolute_Sensitivity_Over_Training_{run_tag}.pdf")

# ============================================================
# Low-dimensional structure analysis
# ============================================================

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap, TSNE
    from sklearn.preprocessing import StandardScaler

    J_np = J_final.detach().cpu().numpy()
    C_np = C_final.detach().cpu().numpy()
    S_np = S_final.detach().cpu().numpy()
    J_param = J_np.T
    J_scaled = StandardScaler().fit_transform(J_param)
    analysis_colours = np.resize(param_location_colours, J_scaled.shape[0])

    print("\nLow-dimensional structure diagnostics")
    print("=====================================")
    print(f"Raw Jacobian shape [sample-output, parameter]: {J_np.shape}")
    print(f"Parameter-analysis matrix shape [parameter, sample-output]: {J_scaled.shape}")
    print(f"Covariance shape: {C_np.shape}")
    print(f"Sensitivity shape: {S_np.shape}")

    n_pca_components = min(64, J_scaled.shape[0], J_scaled.shape[1])
    pca = PCA(n_components=n_pca_components)
    J_pca = pca.fit_transform(J_scaled)
    cum_explained = np.cumsum(pca.explained_variance_ratio_)
    effective_dim_95 = int(np.searchsorted(cum_explained, 0.95) + 1)
    print(f"PCA effective dimension (95% variance): {effective_dim_95}")

    if J_pca.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(6.5, 6.0))
        ax.scatter(J_pca[:, 0], J_pca[:, 1], s=10, alpha=0.65, c=analysis_colours, rasterized=True)
        add_parameter_location_legend(ax, param_location_unique, param_location_colour_map, loc="best")
        ax.set_title("Parameter-Jacobian PCA projection")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        prettify_axes(ax)
        save_pub_figure(fig, output_dir / f"Jacobian_PCA_Projection_{run_tag}.pdf")

    if cfg.run_manifold and J_scaled.shape[0] > 3:
        isomap = Isomap(n_components=2, n_neighbors=min(20, max(2, J_scaled.shape[0] - 1)))
        J_iso = isomap.fit_transform(J_scaled)
        fig, ax = plt.subplots(figsize=(6.5, 6.0))
        ax.scatter(J_iso[:, 0], J_iso[:, 1], s=10, alpha=0.65, c=analysis_colours, rasterized=True)
        add_parameter_location_legend(ax, param_location_unique, param_location_colour_map, loc="best")
        ax.set_title("Parameter-Jacobian Isomap embedding")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        prettify_axes(ax)
        save_pub_figure(fig, output_dir / f"Jacobian_Isomap_Embedding_{run_tag}.pdf")

        tsne = TSNE(n_components=2, perplexity=min(30, max(1, (J_scaled.shape[0] - 1) // 3)), init="pca", learning_rate="auto", random_state=cfg.seed)
        J_tsne = tsne.fit_transform(J_scaled)
        fig, ax = plt.subplots(figsize=(6.5, 6.0))
        ax.scatter(J_tsne[:, 0], J_tsne[:, 1], s=10, alpha=0.65, c=analysis_colours, rasterized=True)
        add_parameter_location_legend(ax, param_location_unique, param_location_colour_map, loc="best")
        ax.set_title("Parameter-Jacobian t-SNE embedding")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        prettify_axes(ax)
        save_pub_figure(fig, output_dir / f"Jacobian_tSNE_Embedding_{run_tag}.pdf")

    sort_idx = np.argsort(S_np)[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.scatter(np.arange(S_np.size), S_np[sort_idx], s=10, alpha=0.65, c=np.resize(param_location_colours, S_np.size)[sort_idx], linewidths=0.0, rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Sorted sensitivity spectrum")
    ax.set_xlabel("Parameter rank")
    ax.set_ylabel("Sensitivity")
    prettify_axes(ax)
    save_pub_figure(fig, output_dir / f"Sensitivity_Rank_Spectrum_{run_tag}.pdf")
except Exception as exc:
    print(f"Skipping low-dimensional structure analysis: {exc}")
    effective_dim_95 = None

# ============================================================
# Final summary
# ============================================================

with torch.no_grad():
    final_train_mse = criterion(model(x_train), y_train_clean).item()
    final_test_mse = criterion(model(x_test), y_test_clean).item()

summary = {
    "dataset": spec.name,
    "target": spec.description,
    "input_dim": input_dim,
    "output_dim": output_dim,
    "field_scale": cfg.field_scale,
    "noise_multiplier": cfg.noise_multiplier,
    "parameter_count": parameter_count(model),
    "final_train_mse_clean": final_train_mse,
    "final_test_mse_clean": final_test_mse,
    "mean_grid_or_test_abs_error": mean_grid_error,
    "max_grid_or_test_abs_error": max_grid_error,
    "final_spearman_init_final": spearman_corr(S_init, S_final),
    "final_mass_in_init_topk": mass_on_indices(S_final, init_topk_idx),
    "reference_epoch": ref_epoch if S_ref is not None else None,
    "final_spearman_ref_final": spearman_corr(S_ref, S_final) if S_ref is not None else None,
    "final_mass_in_ref_topk": mass_on_indices(S_final, ref_topk_idx) if S_ref is not None else None,
    "largest_initial_eigenvalue": float(eig_init[0].item()),
    "largest_ref_eigenvalue": float(eig_ref[0].item()) if eig_ref is not None else None,
    "largest_final_eigenvalue": float(eig_final[0].item()),
    "pca_effective_dim_95": effective_dim_95,
}

summary_path = output_dir / f"final_summary_{run_tag}.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print("\nFinal summary")
print("=============")
print(json.dumps(summary, indent=2))
print(f"Wrote summary to {summary_path}")
