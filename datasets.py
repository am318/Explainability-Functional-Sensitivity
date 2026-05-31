"""Dataset generators for PyTorch training.

This module contains pure functions for constructing toy supervised datasets.
It deliberately avoids import-time dataset creation so callers can choose the
sample counts, device, noise level, and physical parameters explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import torch

Tensor = torch.Tensor
DeviceLike = Optional[Union[str, torch.device]]


@dataclass(frozen=True)
class DatasetSplit:
    """Container for train/test tensors and optional clean targets."""

    x_train: Tensor
    y_train: Tensor
    x_test: Tensor
    y_test: Tensor
    y_train_clean: Optional[Tensor] = None
    y_test_clean: Optional[Tensor] = None
    x_sens: Optional[Tensor] = None


# ============================================================
# Shared utilities
# ============================================================


def _as_device(device: DeviceLike) -> torch.device:
    return torch.device("cpu") if device is None else torch.device(device)


def _rand_uniform(
    n: int,
    low: float,
    high: float,
    *,
    generator: Optional[torch.Generator] = None,
    device: DeviceLike = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    return low + (high - low) * torch.rand(
        n,
        1,
        generator=generator,
        device=_as_device(device),
        dtype=dtype,
    )


def sample_phase_space(
    n: int,
    *,
    x_min: float,
    x_max: float,
    v_min: float,
    v_max: float,
    generator: Optional[torch.Generator] = None,
    device: DeviceLike = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Sample two-dimensional phase-space points ``[x, v]`` uniformly."""

    x = _rand_uniform(
        n,
        x_min,
        x_max,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    v = _rand_uniform(
        n,
        v_min,
        v_max,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    return torch.cat([x, v], dim=-1)


def sample_x(
    n: int,
    *,
    x_min: float = -1.0,
    x_max: float = 1.0,
    generator: Optional[torch.Generator] = None,
    device: DeviceLike = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Sample one-dimensional inputs uniformly."""

    return _rand_uniform(
        n,
        x_min,
        x_max,
        generator=generator,
        device=device,
        dtype=dtype,
    )


def add_noise(
    y: Tensor,
    noise_multiplier: float = 0.0,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Add independent Gaussian noise to a target tensor."""

    if noise_multiplier == 0.0:
        return y.clone()
    return y + noise_multiplier * torch.randn(
        y.shape,
        generator=generator,
        device=y.device,
        dtype=y.dtype,
    )


def make_supervised_split(
    x_train: Tensor,
    y_train_clean: Tensor,
    x_test: Tensor,
    y_test_clean: Tensor,
    *,
    noise_multiplier: float = 0.0,
    generator: Optional[torch.Generator] = None,
    sensitivity_size: Optional[int] = None,
) -> DatasetSplit:
    """Build a ``DatasetSplit`` from clean train/test targets."""

    y_train = add_noise(y_train_clean, noise_multiplier, generator=generator)
    y_test = add_noise(y_test_clean, noise_multiplier, generator=generator)

    if sensitivity_size is None:
        x_sens = x_train
    else:
        n_sens = min(sensitivity_size, x_train.shape[0])
        indices = torch.randperm(
            x_train.shape[0],
            generator=generator,
            device=x_train.device,
        )[:n_sens]
        x_sens = x_train[indices]

    return DatasetSplit(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        y_train_clean=y_train_clean,
        y_test_clean=y_test_clean,
        x_sens=x_sens,
    )


# ============================================================
# Van der Pol time-series dataset
# ============================================================


def vanderpol_rhs(t: Tensor, state: Tensor, mu: float = 3.0) -> Tensor:
    """Right-hand side for the Van der Pol oscillator."""

    del t  # The system is autonomous.
    x, v = state[..., 0:1], state[..., 1:2]
    dxdt = v
    dvdt = mu * (1.0 - x**2) * v - x
    return torch.cat([dxdt, dvdt], dim=-1)


def rk4_step(
    f: Callable[[Tensor, Tensor, float], Tensor],
    t: Tensor,
    y: Tensor,
    dt: Tensor,
    *,
    mu: float = 3.0,
) -> Tensor:
    """One fourth-order Runge-Kutta step."""

    k1 = f(t, y, mu)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1, mu)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2, mu)
    k4 = f(t + dt, y + dt * k3, mu)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_vanderpol(t_grid: Tensor, y0: Tensor, mu: float = 3.0) -> Tensor:
    """Simulate the Van der Pol oscillator on ``t_grid`` from ``y0``."""

    y = torch.empty((len(t_grid), 2), dtype=y0.dtype, device=y0.device)
    y[0] = y0
    for i in range(len(t_grid) - 1):
        dt = t_grid[i + 1] - t_grid[i]
        y[i + 1] = rk4_step(
            vanderpol_rhs,
            t_grid[i],
            y[i : i + 1],
            dt,
            mu=mu,
        ).squeeze(0)
    return y


def make_vanderpol_timeseries_dataset(
    n_train: int,
    n_test: int,
    *,
    t_min: float = 0.0,
    t_max: float = 20.0,
    y0: Sequence[float] = (2.0, 0.0),
    mu: float = 3.0,
    noise_multiplier: float = 1e-1,
    generator: Optional[torch.Generator] = None,
    device: DeviceLike = None,
    dtype: torch.dtype = torch.float32,
) -> DatasetSplit:
    """Dataset where input is time ``t`` and target is Van der Pol position ``x(t)``."""

    dev = _as_device(device)
    t_train = torch.linspace(t_min, t_max, n_train, device=dev, dtype=dtype)
    t_test = torch.linspace(t_min, t_max, n_test, device=dev, dtype=dtype)
    y0_tensor = torch.tensor(y0, device=dev, dtype=dtype)

    traj_train = simulate_vanderpol(t_train, y0_tensor, mu=mu)
    traj_test = simulate_vanderpol(t_test, y0_tensor, mu=mu)

    return make_supervised_split(
        x_train=t_train.unsqueeze(1),
        y_train_clean=traj_train[:, 0:1],
        x_test=t_test.unsqueeze(1),
        y_test_clean=traj_test[:, 0:1],
        noise_multiplier=noise_multiplier,
        generator=generator,
    )


# ============================================================
# Analytic vector fields
# ============================================================


def symmetric_vector_field(state: Tensor, field_scale: float = 1.0) -> Tensor:
    """Evaluate F(x, y) = [-2x, 2y] * field_scale * exp(-x^2 - y^2)."""

    x = state[..., 0:1]
    y = state[..., 1:2]
    phi = torch.exp(-(x**2) - y**2)
    dfdx = -2.0 * field_scale * x * phi
    dfdy = 2.0 * field_scale * y * phi
    return torch.cat([dfdx, dfdy], dim=-1)


def morse_vector_field(
    state: Tensor,
    *,
    D_e: float = 1.0,
    a: float = 1.0,
    x_e: float = 0.0,
) -> Tensor:
    """Evaluate Morse-potential dynamics for state ``[x, v]``."""

    x = state[..., 0:1]
    v = state[..., 1:2]
    exp_term = torch.exp(-a * (x - x_e))
    dxdt = v
    dvdt = 2.0 * a * D_e * (1.0 - exp_term) * exp_term
    return torch.cat([dxdt, dvdt], dim=-1)


def vanderpol_vector_field(state: Tensor, mu: float = 3.0) -> Tensor:
    """Evaluate Van der Pol vector-field targets for state ``[x, v]``."""

    x = state[..., 0:1]
    v = state[..., 1:2]
    dxdt = v
    dvdt = mu * (1.0 - x**2) * v - x
    return torch.cat([dxdt, dvdt], dim=-1)


def make_vector_field_dataset(
    vector_field: Callable[[Tensor], Tensor],
    n_train: int,
    n_test: int,
    *,
    x_min: float,
    x_max: float,
    v_min: float,
    v_max: float,
    noise_multiplier: float = 0.0,
    sensitivity_size: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    device: DeviceLike = None,
    dtype: torch.dtype = torch.float32,
) -> DatasetSplit:
    """Generic two-dimensional vector-field dataset factory."""

    x_train = sample_phase_space(
        n_train,
        x_min=x_min,
        x_max=x_max,
        v_min=v_min,
        v_max=v_max,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    x_test = sample_phase_space(
        n_test,
        x_min=x_min,
        x_max=x_max,
        v_min=v_min,
        v_max=v_max,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    return make_supervised_split(
        x_train=x_train,
        y_train_clean=vector_field(x_train),
        x_test=x_test,
        y_test_clean=vector_field(x_test),
        noise_multiplier=noise_multiplier,
        generator=generator,
        sensitivity_size=sensitivity_size,
    )


def make_symmetric_vector_field_dataset(
    n_train: int,
    n_test: int,
    *,
    field_scale: float = 1.0,
    x_min: float = -3.0,
    x_max: float = 3.0,
    y_min: float = -3.0,
    y_max: float = 3.0,
    noise_multiplier: float = 0.0,
    sensitivity_size: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    device: DeviceLike = None,
    dtype: torch.dtype = torch.float32,
) -> DatasetSplit:
    """Dataset for the symmetric 2D vector field."""

    return make_vector_field_dataset(
        lambda state: symmetric_vector_field(state, field_scale=field_scale),
        n_train,
        n_test,
        x_min=x_min,
        x_max=x_max,
        v_min=y_min,
        v_max=y_max,
        noise_multiplier=noise_multiplier,
        sensitivity_size=sensitivity_size,
        generator=generator,
        device=device,
        dtype=dtype,
    )


def make_morse_vector_field_dataset(
    n_train: int,
    n_test: int,
    *,
    D_e: float = 1.0,
    a: float = 1.0,
    x_e: float = 0.0,
    x_min: float = -2.0,
    x_max: float = 4.0,
    v_min: float = -3.0,
    v_max: float = 3.0,
    noise_multiplier: float = 0.0,
    sensitivity_size: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    device: DeviceLike = None,
    dtype: torch.dtype = torch.float32,
) -> DatasetSplit:
    """Dataset for Morse-potential dynamics."""

    return make_vector_field_dataset(
        lambda state: morse_vector_field(state, D_e=D_e, a=a, x_e=x_e),
        n_train,
        n_test,
        x_min=x_min,
        x_max=x_max,
        v_min=v_min,
        v_max=v_max,
        noise_multiplier=noise_multiplier,
        sensitivity_size=sensitivity_size,
        generator=generator,
        device=device,
        dtype=dtype,
    )


def make_vanderpol_vector_field_dataset(
    n_train: int,
    n_test: int,
    *,
    mu: float = 3.0,
    x_min: float = -3.0,
    x_max: float = 3.0,
    v_min: float = -3.0,
    v_max: float = 3.0,
    noise_multiplier: float = 0.0,
    sensitivity_size: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    device: DeviceLike = None,
    dtype: torch.dtype = torch.float32,
) -> DatasetSplit:
    """Dataset for Van der Pol vector-field learning."""

    return make_vector_field_dataset(
        lambda state: vanderpol_vector_field(state, mu=mu),
        n_train,
        n_test,
        x_min=x_min,
        x_max=x_max,
        v_min=v_min,
        v_max=v_max,
        noise_multiplier=noise_multiplier,
        sensitivity_size=sensitivity_size,
        generator=generator,
        device=device,
        dtype=dtype,
    )


# ============================================================
# Scalar function datasets
# ============================================================


def power_target(x: Tensor, *, power: int = 3, field_scale: float = 1.0) -> Tensor:
    """Evaluate ``field_scale * x**power``."""

    return field_scale * x.pow(power)


def sine_target(x: Tensor, *, field_scale: float = 1.0) -> Tensor:
    """Evaluate ``field_scale * sin(x)``."""

    return field_scale * torch.sin(x)


def make_scalar_function_dataset(
    target: Callable[[Tensor], Tensor],
    n_train: int,
    n_test: int,
    *,
    x_min: float = -1.0,
    x_max: float = 1.0,
    noise_multiplier: float = 0.0,
    sensitivity_size: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    device: DeviceLike = None,
    dtype: torch.dtype = torch.float32,
) -> DatasetSplit:
    """Generic scalar supervised dataset factory."""

    x_train = sample_x(
        n_train,
        x_min=x_min,
        x_max=x_max,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    x_test = sample_x(
        n_test,
        x_min=x_min,
        x_max=x_max,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    return make_supervised_split(
        x_train=x_train,
        y_train_clean=target(x_train),
        x_test=x_test,
        y_test_clean=target(x_test),
        noise_multiplier=noise_multiplier,
        generator=generator,
        sensitivity_size=sensitivity_size,
    )


def make_power_dataset(
    n_train: int,
    n_test: int,
    *,
    power: int = 3,
    field_scale: float = 1.0,
    x_min: float = -1.0,
    x_max: float = 1.0,
    noise_multiplier: float = 0.0,
    sensitivity_size: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    device: DeviceLike = None,
    dtype: torch.dtype = torch.float32,
) -> DatasetSplit:
    """Dataset for ``y = field_scale * x**power``."""

    return make_scalar_function_dataset(
        lambda x: power_target(x, power=power, field_scale=field_scale),
        n_train,
        n_test,
        x_min=x_min,
        x_max=x_max,
        noise_multiplier=noise_multiplier,
        sensitivity_size=sensitivity_size,
        generator=generator,
        device=device,
        dtype=dtype,
    )


def make_sine_dataset(
    n_train: int,
    n_test: int,
    *,
    field_scale: float = 1.0,
    x_min: float = -1.0,
    x_max: float = 1.0,
    noise_multiplier: float = 0.0,
    sensitivity_size: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    device: DeviceLike = None,
    dtype: torch.dtype = torch.float32,
) -> DatasetSplit:
    """Dataset for ``y = field_scale * sin(x)``."""

    return make_scalar_function_dataset(
        lambda x: sine_target(x, field_scale=field_scale),
        n_train,
        n_test,
        x_min=x_min,
        x_max=x_max,
        noise_multiplier=noise_multiplier,
        sensitivity_size=sensitivity_size,
        generator=generator,
        device=device,
        dtype=dtype,
    )

# ============================================================
# Experimental sweep compatibility layer
# ============================================================

@dataclass(frozen=True)
class DatasetSpec:
    """Metadata and factory used by ``Experimental_Sweep.py``.

    The sweep needs to know the input/output dimensions before constructing the
    model, and it needs a single callable that returns a ``DatasetSplit``.  This
    spec keeps that information beside each dataset definition.
    """

    name: str
    factory: Callable[..., DatasetSplit]
    input_dim: int
    output_dim: int
    default_domain: Tuple[float, ...]
    description: str
    plot_kind: str = "generic"
    target_fn: Optional[Callable[[Tensor], Tensor]] = None


def parabola_target(x: Tensor, *, field_scale: float = 1.0) -> Tensor:
    """Evaluate the legacy sweep target ``field_scale * x**2``."""

    return power_target(x, power=2, field_scale=field_scale)


def make_parabola_dataset(
    n_train: int,
    n_test: int,
    *,
    field_scale: float = 1.0,
    x_min: float = -1.0,
    x_max: float = 1.0,
    noise_multiplier: float = 0.0,
    sensitivity_size: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    device: DeviceLike = None,
    dtype: torch.dtype = torch.float32,
) -> DatasetSplit:
    """Dataset for the legacy scalar parabola target ``y = field_scale * x**2``."""

    return make_scalar_function_dataset(
        lambda x: parabola_target(x, field_scale=field_scale),
        n_train,
        n_test,
        x_min=x_min,
        x_max=x_max,
        noise_multiplier=noise_multiplier,
        sensitivity_size=sensitivity_size,
        generator=generator,
        device=device,
        dtype=dtype,
    )


def _make_power3_dataset(
    n_train: int,
    n_test: int,
    **kwargs,
) -> DatasetSplit:
    return make_power_dataset(n_train, n_test, power=3, **kwargs)


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "parabola": DatasetSpec(
        name="parabola",
        factory=make_parabola_dataset,
        input_dim=1,
        output_dim=1,
        default_domain=(-2.0, 2.0),
        description="scalar parabola y = field_scale * x^2",
        plot_kind="scalar_1d",
        target_fn=lambda x: parabola_target(x),
    ),
    "power": DatasetSpec(
        name="power",
        factory=_make_power3_dataset,
        input_dim=1,
        output_dim=1,
        default_domain=(-2.0, 2.0),
        description="scalar cubic y = field_scale * x^3",
        plot_kind="scalar_1d",
        target_fn=lambda x: power_target(x, power=3),
    ),
    "sine": DatasetSpec(
        name="sine",
        factory=make_sine_dataset,
        input_dim=1,
        output_dim=1,
        default_domain=(-6.283185307179586, 6.283185307179586),
        description="scalar sine y = field_scale * sin(x)",
        plot_kind="scalar_1d",
        target_fn=lambda x: sine_target(x),
    ),
    "symmetric_vector_field": DatasetSpec(
        name="symmetric_vector_field",
        factory=make_symmetric_vector_field_dataset,
        input_dim=2,
        output_dim=2,
        default_domain=(-3.0, 3.0, -3.0, 3.0),
        description="2D symmetric vector field",
        plot_kind="vector_field_2d",
        target_fn=symmetric_vector_field,
    ),
    "morse_vector_field": DatasetSpec(
        name="morse_vector_field",
        factory=make_morse_vector_field_dataset,
        input_dim=2,
        output_dim=2,
        default_domain=(-2.0, 4.0, -3.0, 3.0),
        description="Morse-potential phase-space vector field",
        plot_kind="vector_field_2d",
        target_fn=morse_vector_field,
    ),
    "vanderpol_vector_field": DatasetSpec(
        name="vanderpol_vector_field",
        factory=make_vanderpol_vector_field_dataset,
        input_dim=2,
        output_dim=2,
        default_domain=(-3.0, 3.0, -3.0, 3.0),
        description="Van der Pol phase-space vector field",
        plot_kind="vector_field_2d",
        target_fn=vanderpol_vector_field,
    ),
    "vanderpol_timeseries": DatasetSpec(
        name="vanderpol_timeseries",
        factory=make_vanderpol_timeseries_dataset,
        input_dim=1,
        output_dim=1,
        default_domain=(0.0, 20.0),
        description="Van der Pol time-series position x(t)",
        plot_kind="scalar_1d",
        target_fn=None,
    ),
}


def available_datasets() -> tuple[str, ...]:
    """Return dataset names accepted by ``make_dataset_for_sweep``."""

    return tuple(DATASET_REGISTRY.keys())


def get_dataset_spec(name: str) -> DatasetSpec:
    """Return the dataset spec for ``name`` with a clear error on typos."""

    key = name.strip().lower()
    if key not in DATASET_REGISTRY:
        options = ", ".join(available_datasets())
        raise KeyError(f"Unknown dataset {name!r}. Available datasets: {options}")
    return DATASET_REGISTRY[key]


def make_dataset_for_sweep(
    name: str,
    n_train: int,
    n_test: int,
    *,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    v_min: Optional[float] = None,
    v_max: Optional[float] = None,
    field_scale: float = 1.0,
    noise_multiplier: float = 0.0,
    sensitivity_size: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    device: DeviceLike = None,
    dtype: torch.dtype = torch.float32,
) -> DatasetSplit:
    """Create a dataset by registry name for the experimental sweep.

    Scalar datasets receive ``x_min/x_max``.  Vector-field datasets receive
    ``x_min/x_max/v_min/v_max``.  Time-series datasets map ``x_min/x_max`` to
    ``t_min/t_max``.
    """

    spec = get_dataset_spec(name)
    domain = spec.default_domain

    if spec.name == "vanderpol_timeseries":
        return spec.factory(
            n_train,
            n_test,
            t_min=domain[0] if x_min is None else x_min,
            t_max=domain[1] if x_max is None else x_max,
            noise_multiplier=noise_multiplier,
            generator=generator,
            device=device,
            dtype=dtype,
        )

    if spec.input_dim == 1:
        return spec.factory(
            n_train,
            n_test,
            field_scale=field_scale,
            x_min=domain[0] if x_min is None else x_min,
            x_max=domain[1] if x_max is None else x_max,
            noise_multiplier=noise_multiplier,
            sensitivity_size=sensitivity_size,
            generator=generator,
            device=device,
            dtype=dtype,
        )

    common = dict(
        n_train=n_train,
        n_test=n_test,
        x_min=domain[0] if x_min is None else x_min,
        x_max=domain[1] if x_max is None else x_max,
        noise_multiplier=noise_multiplier,
        sensitivity_size=sensitivity_size,
        generator=generator,
        device=device,
        dtype=dtype,
    )

    if spec.name == "symmetric_vector_field":
        return spec.factory(
            field_scale=field_scale,
            y_min=domain[2] if v_min is None else v_min,
            y_max=domain[3] if v_max is None else v_max,
            **common,
        )

    return spec.factory(
        v_min=domain[2] if v_min is None else v_min,
        v_max=domain[3] if v_max is None else v_max,
        **common,
    )

