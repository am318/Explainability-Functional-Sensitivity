"""
Fast correctness check for the two-moons pipeline: data, model forward /
backward, sensitivity scoring, a short instrumented training run with
plotting and checkpointing, and the full rank-stability analysis on top of
it. Does not train to convergence -- this only confirms the pipeline runs
and produces finite, well-shaped outputs.

Because it exercises common/experiment.py, common/sweep.py's building
blocks and common/rank_stability_runner.py, it is also the quickest check
that a change to those shared modules has not broken anything.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))

from dataset import build_moons_datasets
from experiment import Experiment, run_training, select_device, set_seed
from model import MoonsMLP
from sensitivity import compute_sensitivity, flatten_scores, pool_rows, summarize_sensitivity
from train import Config, build_experiment


def smoke_config(output_root: Path) -> Config:
    return replace(
        Config(),
        output_root=str(output_root),
        experiment_name="unit_test",
        epochs=3,
        n_train=256,
        n_test=128,
        batch_size=32,
        probe_samples=64,
        probe_batch_size=32,
        sensitivity_probes=2,
        checkpoint_interval=1,
        heatmap_rows=17,  # deliberately not a divisor of the parameter count
    )


def test_data_and_model(device: torch.device) -> None:
    print(f"\n=== Data / model / sensitivity on device: {device} ===")
    set_seed(0)
    train_pool, test_set = build_moons_datasets(n_train=256, n_test=128, noise=0.2, seed=0)
    assert len(train_pool) == 256 and len(test_set) == 128
    x, y = train_pool[0]
    assert x.shape == (2,) and x.dtype == torch.float32
    assert y.dtype == torch.int64 and int(y) in (0, 1)
    print(f"Data OK: train={len(train_pool)}, test={len(test_set)}, x={tuple(x.shape)}")

    model = MoonsMLP(hidden_dim=32).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == 1218, f"expected 1218 parameters, got {n_params}"

    inputs = torch.stack([train_pool[i][0] for i in range(16)]).to(device)
    targets = torch.stack([train_pool[i][1] for i in range(16)]).to(device)
    logits = model(inputs)
    assert logits.shape == (16, 2) and torch.isfinite(logits).all()

    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, targets)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print(f"Forward/backward OK: {n_params} parameters, loss={loss.item():.4f}")

    probe_loader = torch.utils.data.DataLoader(test_set, batch_size=32)
    unsigned, signed = compute_sensitivity(model, probe_loader, device, n_probes=2, include_signed=False)
    assert signed is None, "include_signed=False should return None for the signed score"
    assert set(unsigned) == {n for n, p in model.named_parameters() if p.requires_grad}
    for name, tensor in unsigned.items():
        assert torch.isfinite(tensor).all() and (tensor >= 0).all(), name
    summary = summarize_sensitivity(unsigned)
    assert summary["total"] > 0
    print(f"Sensitivity OK: {summary}")

    flat, boundaries = flatten_scores(model, unsigned)
    assert flat.numel() == n_params
    assert [g for g, _, _ in boundaries] == ["input", "hidden", "output"]
    assert boundaries[0][1] == 0 and boundaries[-1][2] == n_params
    pooled = pool_rows(flat, 17)
    assert pooled.shape == (17,) and torch.isfinite(pooled).all()
    print(f"Flatten/pool OK: groups={[g for g, _, _ in boundaries]}")


def test_training_and_rank_stability(device: torch.device) -> None:
    print("\n=== Short instrumented run + rank stability ===")
    tmp_dir = Path(tempfile.mkdtemp(prefix="twomoons_mlp_smoke_"))
    try:
        cfg = smoke_config(tmp_dir)
        history = run_training(cfg, build_experiment, device, progress=False, verbose=False)
        assert len(history) == cfg.epochs
        assert all("val_loss" in row and "unsigned_total" in row for row in history)
        assert all("signed_total" not in row for row in history), "signed tracking should be off by default"

        run_dir = cfg.output_dir
        for name in [
            "history.json",
            "loss_and_sensitivity.png",
            "parameter_sensitivity_heatmap.png",
            "parameter_sensitivity_heatmap_data.npz",
        ]:
            path = run_dir / name
            assert path.exists() and path.stat().st_size > 0, f"missing {name}"
        ckpts = sorted(run_dir.glob("ckpt_epoch*.pt"))
        assert len(ckpts) == cfg.epochs, f"expected {cfg.epochs} checkpoints, got {len(ckpts)}"
        saved = json.loads((run_dir / "history.json").read_text())
        assert saved["config"]["hidden_dim"] == cfg.hidden_dim
        print(f"Training OK: {len(history)} epochs, {len(ckpts)} checkpoints, artifacts written")

        cmd = [sys.executable, str(HERE / "rank_stability.py"), str(run_dir), "--probes", "2"]
        subprocess.run(cmd, check=True, env={**os.environ, "DEVICE": str(device)})
        for name in [
            "rank_stability_unsigned.png",
            "rank_stability_consecutive_unsigned.png",
            "rank_stability_curves.npz",
            "full_resolution_sensitivity_cache.npz",
        ]:
            path = run_dir / name
            assert path.exists() and path.stat().st_size > 0, f"missing {name}"
        assert not (run_dir / "rank_stability_signed.png").exists(), "signed figure should not be produced"

        from rank_stability_runner import load_curves
        curves = load_curves(run_dir, kind="unsigned")
        assert set(curves["checkpoint"]["spearman"]) == {"total", "input", "hidden", "output"}
        assert curves["checkpoint"]["spearman"]["total"][-1] == 1.0, "final vs. final must be exactly 1"
        assert len(curves["checkpoint_epochs"]) == cfg.epochs + 1, "epoch 0 should be prepended"
        print("Rank stability OK: figures, curves and cache written")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    device = select_device()
    test_data_and_model(device)
    test_training_and_rank_stability(device)
    print("\nAll smoke tests passed.")
