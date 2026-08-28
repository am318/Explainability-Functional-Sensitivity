"""
Fast correctness check for the MNIST pipeline: dataset download and IDX
parsing, model forward / backward, sensitivity scoring, a short instrumented
training run (on a small subset) with plotting and checkpointing, and the
full rank-stability analysis on top of it. Does not train to convergence --
this only confirms the pipeline runs and produces finite, well-shaped
outputs before committing a cluster job to it.
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

from dataset import MNIST_MEAN, MNIST_STD, build_mnist_datasets
from experiment import Experiment, run_training, select_device, set_seed
from model import MnistCNN
from sensitivity import compute_sensitivity, flatten_scores, pool_rows, summarize_sensitivity
from train import Config, build_experiment


def test_data_and_model(device: torch.device) -> None:
    print(f"\n=== Data / model / sensitivity on device: {device} ===")
    set_seed(0)
    train, test = build_mnist_datasets(Path(Config().data_dir))
    assert len(train) == 60000, f"expected 60000 training images, got {len(train)}"
    assert len(test) == 10000, f"expected 10000 test images, got {len(test)}"
    x, y = train[0]
    assert x.shape == (1, 28, 28) and x.dtype == torch.float32
    assert y.dtype == torch.int64 and 0 <= int(y) <= 9
    # Standardisation should put the pooled pixel distribution near (0, 1).
    sample = train.tensors[0][:2000]
    assert abs(float(sample.mean())) < 0.1, f"normalised mean {float(sample.mean()):.3f} is off"
    assert abs(float(sample.std()) - 1.0) < 0.1, f"normalised std {float(sample.std()):.3f} is off"
    labels = train.tensors[1]
    assert set(labels.unique().tolist()) == set(range(10))
    print(f"Data OK: train={len(train)}, test={len(test)}, x={tuple(x.shape)}, "
          f"mean={float(sample.mean()):.3f}, std={float(sample.std()):.3f}")

    model = MnistCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == 9098, f"expected 9098 parameters, got {n_params}"

    inputs = train.tensors[0][:16].to(device)
    targets = train.tensors[1][:16].to(device)
    logits = model(inputs)
    assert logits.shape == (16, 10) and torch.isfinite(logits).all()

    loss = nn.CrossEntropyLoss()(logits, targets)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print(f"Forward/backward OK: {n_params} parameters, loss={loss.item():.4f}")

    probe = torch.utils.data.DataLoader(torch.utils.data.Subset(test, range(128)), batch_size=64)
    unsigned, signed = compute_sensitivity(model, probe, device, n_probes=2, include_signed=False)
    assert signed is None
    for name, tensor in unsigned.items():
        assert torch.isfinite(tensor).all() and (tensor >= 0).all(), name
    assert summarize_sensitivity(unsigned)["total"] > 0

    flat, boundaries = flatten_scores(model, unsigned)
    assert flat.numel() == n_params
    assert [g for g, _, _ in boundaries] == ["conv1", "conv2", "head"]
    pooled = pool_rows(flat, 17)
    assert pooled.shape == (17,) and torch.isfinite(pooled).all()
    print(f"Sensitivity OK: groups={[g for g, _, _ in boundaries]}, total={summarize_sensitivity(unsigned)['total']:.4e}")


def test_training_and_rank_stability(device: torch.device) -> None:
    print("\n=== Short instrumented run + rank stability ===")
    tmp_dir = Path(tempfile.mkdtemp(prefix="mnist_cnn_smoke_"))
    try:
        cfg = replace(
            Config(),
            output_root=str(tmp_dir),
            experiment_name="unit_test",
            epochs=2,
            batch_size=256,
            probe_samples=128,
            probe_batch_size=64,
            eval_batch_size=1000,
            sensitivity_probes=2,
            checkpoint_interval=1,
            heatmap_rows=17,
        )
        history = run_training(cfg, build_experiment, device, progress=False, verbose=False)
        assert len(history) == cfg.epochs
        assert all("val_loss" in row and "unsigned_total" in row for row in history)
        assert history[-1]["val_acc"] > 0.8, f"2 epochs should clear 80% accuracy, got {history[-1]['val_acc']:.3f}"

        run_dir = cfg.output_dir
        for name in [
            "history.json",
            "loss_and_sensitivity.png",
            "parameter_sensitivity_heatmap.png",
            "parameter_sensitivity_heatmap_data.npz",
        ]:
            assert (run_dir / name).exists() and (run_dir / name).stat().st_size > 0, f"missing {name}"
        assert len(sorted(run_dir.glob("ckpt_epoch*.pt"))) == cfg.epochs
        print(f"Training OK: {len(history)} epochs, test acc {history[-1]['val_acc']:.3f}, artifacts written")

        cmd = [sys.executable, str(HERE / "rank_stability.py"), str(run_dir), "--probes", "2"]
        subprocess.run(cmd, check=True, env={**os.environ, "DEVICE": str(device)})
        for name in ["rank_stability_unsigned.png", "rank_stability_consecutive_unsigned.png",
                     "rank_stability_curves.npz"]:
            assert (run_dir / name).exists() and (run_dir / name).stat().st_size > 0, f"missing {name}"
        assert not (run_dir / "rank_stability_signed.png").exists()

        from rank_stability_runner import load_curves
        curves = load_curves(run_dir, kind="unsigned")
        assert set(curves["checkpoint"]["spearman"]) == {"total", "conv1", "conv2", "head"}
        assert curves["checkpoint"]["spearman"]["total"][-1] == 1.0
        print("Rank stability OK: figures, curves and cache written")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    device = select_device()
    test_data_and_model(device)
    test_training_and_rank_stability(device)
    print("\nAll smoke tests passed.")
