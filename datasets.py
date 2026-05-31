import torch

# ============================================================
# 1D VDP
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
# Dataset: learn the 2D Morse potential dynamics
# ============================================================
# Input:  state = [x, v]
# Target: f(x, v) = [dx/dt, dv/dt]
#         dx/dt = v
#         dv/dt = -dV/dx
#
# Morse potential:
#   V(x) = D_e * (1 - exp(-a * (x - x_e)))^2
#
# Force:
#   -dV/dx = 2 * a * D_e * (1 - exp(-a * (x - x_e))) * exp(-a * (x - x_e))

def morse_vector_field(state, D_e=1.0, a=1.0, x_e=0.0):
    x = state[..., 0:1]
    v = state[..., 1:2]

    exp_term = torch.exp(-a * (x - x_e))
    dxdt = v
    dvdt = 2.0 * a * D_e * (1.0 - exp_term) * exp_term

    return torch.cat([dxdt, dvdt], dim=-1)


def sample_phase_space(n, x_min=-2.0, x_max=4.0, v_min=-3.0, v_max=3.0):
    x = x_min + (x_max - x_min) * torch.rand(n, 1, dtype=torch.float32)
    v = v_min + (v_max - v_min) * torch.rand(n, 1, dtype=torch.float32)
    return torch.cat([x, v], dim=-1)


x_train = sample_phase_space(cfg.n_train).to(device)
y_train_clean = morse_vector_field(x_train, D_e=cfg.D_e, a=cfg.a, x_e=cfg.x_e)
y_train = y_train_clean + cfg.noise_multiplier * torch.randn_like(y_train_clean)

x_test = sample_phase_space(cfg.n_test).to(device)
y_test_clean = morse_vector_field(x_test, D_e=cfg.D_e, a=cfg.a, x_e=cfg.x_e)
y_test = y_test_clean + cfg.noise_multiplier * torch.randn_like(y_test_clean)

# # Fixed subset for sensitivity analysis.
# sens_idx = torch.randperm(cfg.n_train, device=device)[:cfg.n_sensitivity]
# x_sens = x_train[sens_idx]

# Full training set for exact sensitivity analysis.
x_sens = x_train



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
# Dataset: learn the scalar parabola y = x^3
# ============================================================
# Input:  x
# Target: y = x^3

power = 3

def parabola_target(x, field_scale=1.0):
    # return field_scale * x.pow(power)
    return torch.sin(x)

def sample_x(n, x_min=-1.0, x_max=1.0):
    return x_min + (x_max - x_min) * torch.rand(n, 1, dtype=torch.float32)



# ============================================================
# Dataset: learn the scalar parabola y = x^3
# ============================================================
# Input:  x
# Target: y = x^3

power = 3

def parabola_target(x, field_scale=1.0):
    # return field_scale * x.pow(power)
    return torch.sin(x)

def sample_x(n, x_min=-1.0, x_max=1.0):
    return x_min + (x_max - x_min) * torch.rand(n, 1, dtype=torch.float32)