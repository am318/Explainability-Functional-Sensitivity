"""
Fast correctness check for the WikiText-2 AWD-LSTM port: exercises data
loading, the model's forward/backward pass (incl. that WeightDrop's
DropConnect mask actually changes between forward calls and that gradients
reach the raw hidden-to-hidden weight through it), an optimizer step,
sensitivity tracking, a couple of full training epochs with plotting, a tiny
pruning-experiment branch, and rank-stability analysis -- on every available
device (CPU always; MPS/CUDA if present). Does not train to convergence --
this only confirms the pipeline runs and produces finite, well-shaped
outputs.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))

from dataset import WordSequenceDataset, build_corpus
from model import AWDLSTM
from sensitivity import (
    compute_sensitivity,
    flatten_scores,
    pool_rows,
    pooled_group_boundaries,
    summarize_sensitivity,
)


def available_devices() -> list:
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


def write_tiny_corpus(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    vocab_words = [f"word{i}" for i in range(40)]
    import random
    rng = random.Random(0)

    def make_split(n_lines: int) -> str:
        lines = [" ".join(rng.choices(vocab_words, k=15)) for _ in range(n_lines)]
        return "\n".join(lines) + "\n"

    (data_dir / "train.txt").write_text(make_split(60), encoding="utf-8")
    (data_dir / "valid.txt").write_text(make_split(20), encoding="utf-8")
    (data_dir / "test.txt").write_text(make_split(20), encoding="utf-8")


def run_smoke_test(device: torch.device, data_dir: Path) -> None:
    print(f"\n=== Smoke test on device: {device} ===")
    torch.manual_seed(0)

    dictionary, train_ids, valid_ids, _test_ids = build_corpus(str(data_dir))
    seq_length = 8
    train_dataset = WordSequenceDataset(train_ids, seq_length)
    valid_dataset = WordSequenceDataset(valid_ids, seq_length)
    loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
    probe_loader = DataLoader(Subset(valid_dataset, list(range(min(8, len(valid_dataset))))), batch_size=4)

    inputs, targets = next(iter(loader))
    inputs, targets = inputs.to(device), targets.to(device)
    assert inputs.shape == (8, seq_length)
    assert targets.shape == (8, seq_length)
    print(f"Data OK: vocab_size={len(dictionary)}, batch shape={tuple(inputs.shape)}")

    model = AWDLSTM(
        vocab_size=len(dictionary), embedding_dim=16, rnn_units=16, nlayers=2,
        dropout=0.4, dropouth=0.3, dropouti=0.4, dropoute=0.1, wdrop=0.5, tie_weights=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    logits, hidden = model(inputs)
    assert logits.shape == (8, seq_length, len(dictionary))
    assert torch.isfinite(logits).all()
    print(f"Forward pass OK: logits shape={tuple(logits.shape)}")

    # WeightDrop correctness: in train() mode, two forward calls on the same
    # input should give different outputs (fresh DropConnect mask each time),
    # and the raw hidden-to-hidden weight must receive a gradient (i.e. the
    # _flat_weights refresh in model.py's WeightDrop actually works -- if it
    # didn't, the LSTM backend would silently keep using a stale weight
    # tensor and the dropped-out mask would have no effect on the output).
    model.train()
    with torch.no_grad():
        out1, _ = model(inputs)
        out2, _ = model(inputs)
    assert not torch.allclose(out1, out2), "WeightDrop mask does not appear to vary between forward calls"
    raw_weight = model.rnns[0].module.weight_hh_l0_raw
    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(inputs)
    criterion(logits.reshape(-1, len(dictionary)), targets.reshape(-1)).backward()
    assert raw_weight.grad is not None and torch.isfinite(raw_weight.grad).all() and raw_weight.grad.abs().sum() > 0
    print("WeightDrop OK: DropConnect mask varies across calls, gradient reaches weight_hh_l0_raw")

    loss_before = criterion(logits.reshape(-1, len(dictionary)), targets.reshape(-1))
    assert torch.isfinite(loss_before)
    optimizer.step()
    print(f"Backward + optimizer step OK: loss={loss_before.item():.4f}")

    with torch.no_grad():
        logits_after, _ = model(inputs)
        loss_after = criterion(logits_after.reshape(-1, len(dictionary)), targets.reshape(-1))
    print(f"Loss after one step: {loss_after.item():.4f} (changed: {loss_after.item() != loss_before.item()})")

    unsigned, signed = compute_sensitivity(model, probe_loader, device, n_probes=3)
    assert set(unsigned.keys()) == {n for n, p in model.named_parameters() if p.requires_grad}
    for name, tensor in unsigned.items():
        assert torch.isfinite(tensor).all()
        assert (tensor >= 0).all(), f"unsigned sensitivity must be non-negative ({name})"
        assert tensor.shape == dict(model.named_parameters())[name].shape
    for name, tensor in signed.items():
        assert torch.isfinite(tensor).all()
        assert tensor.shape == dict(model.named_parameters())[name].shape
    unsigned_summary = summarize_sensitivity(unsigned)
    signed_summary = summarize_sensitivity(signed)
    assert unsigned_summary["total"] > 0
    print(f"Sensitivity OK: unsigned={unsigned_summary}, signed={signed_summary}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    unsigned_flat, boundaries = flatten_scores(model, unsigned)
    assert unsigned_flat.numel() == total_params
    groups = [g for g, _, _ in boundaries]
    assert groups == ["embedding", "rnns", "head"], groups
    assert boundaries[0][1] == 0 and boundaries[-1][2] == total_params
    n_rows = 17  # deliberately not a divisor of total_params, to exercise padding
    pooled = pool_rows(unsigned_flat, n_rows)
    assert pooled.shape == (n_rows,)
    assert torch.isfinite(pooled).all()
    pooled_bounds = pooled_group_boundaries(boundaries, total_params, n_rows)
    assert all(0 <= row < n_rows for _, row in pooled_bounds)
    print(f"Flatten/pool OK: total_params={total_params}, pooled_shape={tuple(pooled.shape)}, bounds={pooled_bounds}")

    print(f"=== Device {device} passed ===")


def _patch_env(env: dict):
    old_env = {k: os.environ.get(k) for k in env}
    os.environ.update(env)

    class _Restorer:
        def __exit__(self, *a):
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        def __enter__(self):
            return self

    return _Restorer()


def run_training_with_plot_smoke_test(data_dir: Path) -> Path:
    print("\n=== Smoke test: short training run with sensitivity tracking + plot ===")
    import train as train_module

    tmp_dir = Path(tempfile.mkdtemp(prefix="wikitext_lstm_smoke_"))
    experiment_name = "unit_test"
    env = {
        "DATA_DIR": str(data_dir),
        "OUTPUT_ROOT": str(tmp_dir),
        "EXPERIMENT_NAME": experiment_name,
        "EPOCHS": "2",
        "SEQ_LENGTH": "8",
        "EMBEDDING_DIM": "16",
        "RNN_UNITS": "16",
        "NLAYERS": "2",
        "BATCH_SIZE": "8",
        "CHECKPOINT_INTERVAL": "1",
        "SENSITIVITY_CHUNKS": "8",
        "SENSITIVITY_BATCH_SIZE": "4",
        "SENSITIVITY_PROBES": "2",
        "SENSITIVITY_INTERVAL": "1",
    }
    with _patch_env(env):
        import importlib
        importlib.reload(train_module)
        train_module.main()

    experiment_dir = tmp_dir / experiment_name
    history_path = experiment_dir / "history.json"
    plot_path = experiment_dir / "loss_and_sensitivity.png"
    heatmap_path = experiment_dir / "parameter_sensitivity_heatmap.png"
    assert experiment_dir.exists(), "experiment subfolder was not created under OUTPUT_ROOT"
    assert history_path.exists(), "history.json was not written"
    assert plot_path.exists() and plot_path.stat().st_size > 0
    assert heatmap_path.exists() and heatmap_path.stat().st_size > 0
    for e in (1, 2):
        assert (experiment_dir / f"ckpt_epoch{e}.pt").exists()

    import json
    history = json.loads(history_path.read_text())["history"]
    assert len(history) == 2
    assert all("unsigned_total" in row and "signed_total" in row for row in history)
    print(f"Training+plotting OK: history rows={len(history)}, plot={plot_path}")
    print("=== Training/plot smoke test passed ===")
    return experiment_dir


def run_pruning_and_rank_stability_smoke_test(experiment_dir: Path) -> None:
    # Run as subprocesses (rather than importing): `common/rank_stability.py`
    # (correlation-statistics library) and this directory's own
    # `rank_stability.py` (CLI script) share a module name, and this
    # process's sys.path already has common/ prepended (from the imports at
    # the top of this file) -- so `import rank_stability` here would
    # silently resolve to the wrong module. Subprocesses sidestep that
    # entirely and also exercise these files exactly as a real invocation
    # would (`python pruning_experiment.py ...`).
    print("\n=== Smoke test: pruning_experiment.py + rank_stability.py ===")
    import subprocess

    script_dir = Path(__file__).resolve().parent

    subprocess.run(
        [sys.executable, str(script_dir / "pruning_experiment.py"), str(experiment_dir),
         "--keep-fraction", "0.5", "--prune-epochs", "0", "1", "--probes", "2"],
        check=True,
    )
    out_dir = experiment_dir / "pruning_keep0.50"
    assert (out_dir / "results.json").exists()
    assert (out_dir / "pruning_vs_epoch.png").exists()
    assert (out_dir / "branch_prune0_final.pt").exists()
    assert (out_dir / "branch_prune1_final.pt").exists()
    print("pruning_experiment.py OK")

    subprocess.run(
        [sys.executable, str(script_dir / "rank_stability.py"), str(experiment_dir), "--probes", "2"],
        check=True,
    )
    assert (experiment_dir / "full_resolution_sensitivity_cache.npz").exists()
    print("rank_stability.py OK")

    print("=== Pruning/rank-stability smoke test passed ===")


if __name__ == "__main__":
    devices = available_devices()
    print(f"Testing on devices: {devices}")

    corpus_dir = Path(tempfile.mkdtemp(prefix="wikitext_lstm_smoke_corpus_"))
    try:
        write_tiny_corpus(corpus_dir)
        for device in devices:
            run_smoke_test(device, corpus_dir)
        experiment_dir = run_training_with_plot_smoke_test(corpus_dir)
        run_pruning_and_rank_stability_smoke_test(experiment_dir)
        shutil.rmtree(experiment_dir.parent, ignore_errors=True)
    finally:
        shutil.rmtree(corpus_dir, ignore_errors=True)

    print("\nAll smoke tests passed.")
