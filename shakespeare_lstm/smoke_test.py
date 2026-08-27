"""
Fast correctness check for the PyTorch ShakespearAI port: exercises data
loading, the model's forward/backward pass, an optimizer step, text
generation, sensitivity tracking, and a couple of full training epochs with
plotting -- on every available device (CPU always; MPS/CUDA if present).
Does not train to convergence -- this only confirms the pipeline runs and
produces finite, well-shaped outputs.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from dataset import CharSequenceDataset, build_vocab, encode, load_text
from generate import generate_text
from model import CharLSTM
from plotting import plot_training_history
from sensitivity import (
    compute_sensitivity,
    flatten_scores,
    pool_rows,
    pooled_group_boundaries,
    summarize_sensitivity,
)

DATA_PATH = Path(__file__).resolve().parents[1] / "ShakespearAI" / "dataset" / "shakespeare.txt"


def available_devices() -> list:
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


def run_smoke_test(device: torch.device) -> None:
    print(f"\n=== Smoke test on device: {device} ===")
    torch.manual_seed(0)

    text = load_text(str(DATA_PATH))[:5000]
    vocab, char2idx = build_vocab(text)
    idx2char = vocab
    text_as_int = encode(text, char2idx)

    seq_length = 20
    dataset = CharSequenceDataset(text_as_int, seq_length)
    train_indices = list(range(0, len(dataset) - 8))
    probe_indices = list(range(len(dataset) - 8, len(dataset)))
    loader = DataLoader(Subset(dataset, train_indices), batch_size=8, shuffle=True, drop_last=True)
    probe_loader = DataLoader(Subset(dataset, probe_indices), batch_size=4, shuffle=False)

    inputs, targets = next(iter(loader))
    inputs, targets = inputs.to(device), targets.to(device)
    assert inputs.shape == (8, seq_length)
    assert targets.shape == (8, seq_length)
    print(f"Data OK: vocab_size={len(vocab)}, batch shape={tuple(inputs.shape)}")

    model = CharLSTM(vocab_size=len(vocab), embedding_dim=16, rnn_units=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    logits, hidden = model(inputs)
    assert logits.shape == (8, seq_length, len(vocab))
    assert torch.isfinite(logits).all()
    print(f"Forward pass OK: logits shape={tuple(logits.shape)}")

    loss_before = criterion(logits.reshape(-1, len(vocab)), targets.reshape(-1))
    assert torch.isfinite(loss_before)

    optimizer.zero_grad(set_to_none=True)
    loss_before.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0 and all(g == g for g in grad_norms)  # no NaNs
    optimizer.step()
    print(f"Backward + optimizer step OK: loss={loss_before.item():.4f}")

    with torch.no_grad():
        logits_after, _ = model(inputs)
        loss_after = criterion(logits_after.reshape(-1, len(vocab)), targets.reshape(-1))
    print(f"Loss after one step: {loss_after.item():.4f} (changed: {loss_after.item() != loss_before.item()})")

    generated = generate_text(
        model, start_string="ROMEO", char2idx=char2idx, idx2char=idx2char,
        device=device, num_generate=30, temperature=1.0,
    )
    assert len(generated) == len("ROMEO") + 30
    print(f"Generation OK: {generated!r}")

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
    assert [g for g, _, _ in boundaries] == ["embedding", "lstm", "head"]
    assert boundaries[0][1] == 0 and boundaries[-1][2] == total_params
    n_rows = 17  # deliberately not a divisor of total_params, to exercise padding
    pooled = pool_rows(unsigned_flat, n_rows)
    assert pooled.shape == (n_rows,)
    assert torch.isfinite(pooled).all()
    pooled_bounds = pooled_group_boundaries(boundaries, total_params, n_rows)
    assert [g for g, _ in pooled_bounds] == ["embedding", "lstm", "head"]
    assert all(0 <= row < n_rows for _, row in pooled_bounds)
    print(f"Flatten/pool OK: total_params={total_params}, pooled_shape={tuple(pooled.shape)}, bounds={pooled_bounds}")

    print(f"=== Device {device} passed ===")


def run_training_with_plot_smoke_test() -> None:
    print("\n=== Smoke test: short training run with sensitivity tracking + plot ===")
    import train as train_module

    tmp_dir = Path(tempfile.mkdtemp(prefix="shakespeare_lstm_smoke_"))
    try:
        import os

        small_corpus_path = tmp_dir / "shakespeare_small.txt"
        small_corpus_path.write_text(load_text(str(DATA_PATH))[:20000], encoding="utf-8")

        experiment_name = "unit_test"
        env = {
            "DATA_PATH": str(small_corpus_path),
            "OUTPUT_ROOT": str(tmp_dir),
            "EXPERIMENT_NAME": experiment_name,
            "EPOCHS": "2",
            "SEQ_LENGTH": "20",
            "EMBEDDING_DIM": "16",
            "RNN_UNITS": "32",
            "BATCH_SIZE": "8",
            "CHECKPOINT_INTERVAL": "1",
            "SENSITIVITY_CHUNKS": "16",
            "SENSITIVITY_BATCH_SIZE": "8",
            "SENSITIVITY_PROBES": "2",
            "SENSITIVITY_INTERVAL": "1",
        }
        old_env = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            import importlib
            importlib.reload(train_module)
            train_module.main()
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        experiment_dir = tmp_dir / experiment_name
        history_path = experiment_dir / "history.json"
        plot_path = experiment_dir / "loss_and_sensitivity.png"
        heatmap_path = experiment_dir / "parameter_sensitivity_heatmap.png"
        assert experiment_dir.exists(), "experiment subfolder was not created under OUTPUT_ROOT"
        assert history_path.exists(), "history.json was not written"
        assert plot_path.exists(), "loss_and_sensitivity.png was not written"
        assert plot_path.stat().st_size > 0
        assert heatmap_path.exists(), "parameter_sensitivity_heatmap.png was not written"
        assert heatmap_path.stat().st_size > 0

        import json
        history = json.loads(history_path.read_text())["history"]
        assert len(history) == 2
        assert all("unsigned_total" in row and "signed_total" in row for row in history)
        print(f"Training+plotting OK: history rows={len(history)}, plot={plot_path}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("=== Training/plot smoke test passed ===")


if __name__ == "__main__":
    devices = available_devices()
    print(f"Testing on devices: {devices}")
    for device in devices:
        run_smoke_test(device)
    run_training_with_plot_smoke_test()
    print("\nAll smoke tests passed.")
