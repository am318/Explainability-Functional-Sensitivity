import copy
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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
    epochs: int = 10000
    checkpoint_interval: int = epochs // 10
    topk_frac: float = 0.10
    compare_epoch: int = epochs // 10


cfg = Config()

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)

# MPS-safe device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# ============================================================
# Dataset
# ============================================================

# noise_multiplier = 1e-2

# x_train = torch.linspace(-1.2, 1.2, cfg.n_train, dtype=torch.float32).unsqueeze(1)
# y_train = x_train ** 4 - x_train ** 2 + noise_multiplier * torch.randn_like(x_train)

# x_test = torch.linspace(-1.2, 1.2, cfg.n_test, dtype=torch.float32).unsqueeze(1)
# y_test = x_test ** 4 - x_test ** 2 + noise_multiplier * torch.randn_like(x_test)

# x_train = x_train.to(device)
# y_train = y_train.to(device)

# x_test = x_test.to(device)
# y_test = y_test.to(device)

noise_multiplier = 1e-1  # increase if you want a harder observation model

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

# observe only the position with noise
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
            nn.LeakyReLU(),
        ]

        for _ in range(cfg.n_hidden - 1):
            layers.extend([
                nn.Linear(width, width),
                nn.LeakyReLU(),
            ])

        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model = SmallMLP(cfg.hidden_width).to(device)


# ============================================================
# Utilities
# ============================================================

def flatten_grads(grads):
    return torch.cat([g.reshape(-1) for g in grads])


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def compute_parameter_jacobian(model, x):
    """
    J shape:
        [N_data, N_parameters]
    Row k:
        df(x_k)/dtheta
    """
    params = list(model.parameters())
    rows = []

    x = x.to(next(model.parameters()).device)

    for xi in x:
        yi = model(xi.unsqueeze(0)).squeeze()

        grads = torch.autograd.grad(
            yi,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )

        row = flatten_grads(grads)
        rows.append(row)

    return torch.stack(rows)


def compute_covariance(J):
    """
    C = (1/N) J^T J
    """
    return (J.T @ J) / J.shape[0]


def sensitivity_scores(J):
    """
    S_i = E[(df/dtheta_i)^2]
    """
    return torch.mean(J ** 2, dim=0)


def eigvals_from_covariance(C):
    C_cpu = C.detach().cpu().numpy()
    eig = np.linalg.eigvalsh(C_cpu)
    eig = np.clip(eig, 0.0, None)
    eig = eig[::-1]
    return torch.from_numpy(eig.astype(np.float32))


def topk_indices(scores, frac=0.1):
    k = max(1, int(frac * scores.numel()))
    return torch.topk(scores, k).indices


def mass_on_indices(scores, indices):
    return (scores[indices].sum() / (scores.sum() + 1e-12)).item()


def spearman_corr(a, b):
    """
    Rank correlation without scipy.
    """
    a = a.detach().cpu().numpy()
    b = b.detach().cpu().numpy()

    ra = np.argsort(np.argsort(a)).astype(np.float32)
    rb = np.argsort(np.argsort(b)).astype(np.float32)

    sa = ra.std()
    sb = rb.std()

    if sa < 1e-12 or sb < 1e-12:
        return 0.0

    return float(np.corrcoef(ra, rb)[0, 1])


def build_trainable_masks(model, scores, frac=0.1):
    """
    Build boolean masks for each parameter tensor.
    True  -> keep trainable
    False -> freeze after compare_epoch

    The masks are defined globally across all parameters by top-k functional
    sensitivity at the compare snapshot.
    """
    keep_idx = topk_indices(scores, frac=frac)
    global_mask = torch.zeros_like(scores, dtype=torch.bool)
    global_mask[keep_idx] = True

    masks = []
    offset = 0
    for p in model.parameters():
        n = p.numel()
        masks.append(global_mask[offset: offset + n].view_as(p).detach().clone())
        offset += n

    if offset != scores.numel():
        raise RuntimeError(
            f"Mask construction failed: consumed {offset} scores, expected {scores.numel()}."
        )

    return masks


def apply_masks_to_grads_and_state(model, optimizer, masks):
    """
    Zero gradients and Adam state entries for frozen parameters.
    This prevents post-compare updates and momentum carry-over.
    """
    for p, mask in zip(model.parameters(), masks):
        if p.grad is not None:
            p.grad.mul_(mask)

        state = optimizer.state.get(p, None)
        if not state:
            continue

        if "exp_avg" in state:
            state["exp_avg"].mul_(mask)
        if "exp_avg_sq" in state:
            state["exp_avg_sq"].mul_(mask)


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
    "frozen_fraction": [],
}

# Second reference snapshot and trainability mask
S_ref = None
C_ref = None
ref_topk_idx = None
ref_epoch = None
trainable_masks = None


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

    # After compare_epoch, freeze parameters with small functional sensitivity
    # at the compare snapshot by masking gradients and Adam state.
    if trainable_masks is not None:
        apply_masks_to_grads_and_state(model, optimizer, trainable_masks)

    optimizer.step()

    # Capture the compare snapshot exactly at compare_epoch, after the update
    # for that epoch, so subsequent epochs are frozen.
    if (epoch == cfg.compare_epoch) and (S_ref is None):
        model.eval()
        J_ref = compute_parameter_jacobian(model, x_train)
        S_ref = sensitivity_scores(J_ref).detach().clone()
        C_ref = compute_covariance(J_ref).detach().clone()
        ref_topk_idx = topk_indices(S_ref, cfg.topk_frac)
        ref_epoch = epoch
        trainable_masks = build_trainable_masks(model, S_ref, cfg.topk_frac)
        frozen_fraction_exact = 1.0 - (
            sum(m.sum().item() for m in trainable_masks) / sum(m.numel() for m in trainable_masks)
        )
        print(f"Captured reference snapshot at epoch={ref_epoch}")
        print(f"Freezing low-sensitivity weights; frozen fraction = {frozen_fraction_exact:.3f}")

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
        frozen_fraction = np.nan

        if S_ref is not None:
            rho_ref_curr = spearman_corr(S_ref, S_curr)
            ref_topk_mass = mass_on_indices(S_curr, ref_topk_idx)
            frozen_fraction = 1.0 - (
                sum(m.sum().item() for m in trainable_masks) / sum(m.numel() for m in trainable_masks)
            )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["spearman_init_current"].append(rho_init_curr)
        history["init_topk_mass"].append(init_topk_mass)
        history["spearman_ref_current"].append(rho_ref_curr)
        history["ref_topk_mass"].append(ref_topk_mass)
        history["frozen_fraction"].append(frozen_fraction)

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
im0 = axes[1, 0].imshow(C_init.detach().cpu().numpy(), aspect="auto")
axes[1, 0].set_title("Sensitivity covariance at init")
plt.colorbar(im0, ax=axes[1, 0])

# 5. Covariance matrix at end
im1 = axes[1, 1].imshow(C_final.detach().cpu().numpy(), aspect="auto")
axes[1, 1].set_title("Sensitivity covariance after training")
plt.colorbar(im1, ax=axes[1, 1])

# 6. Eigenvalue spectrum
axes[1, 2].plot(eig_init.cpu().numpy(), label="initial")
if eig_ref is not None:
    axes[1, 2].plot(eig_ref.cpu().numpy(), label=f"ref @ epoch {ref_epoch}")
axes[1, 2].plot(eig_final.cpu().numpy(), label="final")
axes[1, 2].set_yscale("log")
axes[1, 2].set_title("Sensitivity covariance eigenspectrum")
axes[1, 2].legend()

plt.tight_layout()
plt.show()


# ============================================================
# Additional sensitivity distribution plots
# ============================================================

fig, axes = plt.subplots(1, 3 if S_ref is not None else 2, figsize=(16, 4))

axes[0].hist(S_init.detach().cpu().numpy(), bins=20)
axes[0].set_title("Parameter sensitivities at init")

if S_ref is not None:
    axes[1].hist(S_ref.detach().cpu().numpy(), bins=20)
    axes[1].set_title(f"Parameter sensitivities at ref epoch {ref_epoch}")
    axes[2].hist(S_final.detach().cpu().numpy(), bins=20)
    axes[2].set_title("Parameter sensitivities after training")
else:
    axes[1].hist(S_final.detach().cpu().numpy(), bins=20)
    axes[1].set_title("Parameter sensitivities after training")

plt.tight_layout()
plt.show()


# ============================================================
# Final summary
# ============================================================

print("\nFinal summary")
print("=============")
print(f"parameter count: {parameter_count(model)}")
print(f"final train loss: {history['train_loss'][-1]:.6e}")
print(f"final test loss: {history['test_loss'][-1]:.6e}")
print(f"final Spearman(init, final): {spearman_corr(S_init, S_final):.3f}")
print(
    f"final mass in init top-{int(100 * cfg.topk_frac)}%: "
    f"{mass_on_indices(S_final, init_topk_idx):.3f}"
)

if S_ref is not None:
    print(f"reference epoch captured at: {ref_epoch}")
    print(f"final Spearman(ref, final): {spearman_corr(S_ref, S_final):.3f}")
    print(
        f"final mass in ref top-{int(100 * cfg.topk_frac)}%: "
        f"{mass_on_indices(S_final, ref_topk_idx):.3f}"
    )
else:
    print("reference snapshot was not captured; increase epochs or lower compare_epoch.")

print(f"largest initial eigenvalue: {eig_init[0].item():.6e}")
if eig_ref is not None:
    print(f"largest ref eigenvalue: {eig_ref[0].item():.6e}")
print(f"largest final eigenvalue: {eig_final[0].item():.6e}")
