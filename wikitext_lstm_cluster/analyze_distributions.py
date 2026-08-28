"""
Compare parameter-wise sensitivity distributions before vs. after training.
See shakespeare_lstm/analyze_distributions.py for the full explanation --
same analysis, on the WikiText-2 AWD-LSTM instead.

Usage:
    python analyze_distributions.py outputs/full_run_1
    python analyze_distributions.py outputs/full_run_1 --probes 16
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))

from plotting import (
    group_labels,
    plot_sensitivity_distributions_all,
    plot_sensitivity_distributions_by_module,
)
from sensitivity import compute_sensitivity, flatten_scores
from train import Config, build_loaders, build_model, select_device, set_seed


def load_config(experiment_dir: Path) -> Config:
    history = json.loads((experiment_dir / "history.json").read_text())
    cfg_dict = {k: v for k, v in history["config"].items() if k != "output_dir"}
    return Config(**cfg_dict)


def find_final_checkpoint(experiment_dir: Path) -> Path:
    ckpts = sorted(
        experiment_dir.glob("ckpt_epoch*.pt"),
        key=lambda p: int(p.stem.split("epoch")[-1]),
    )
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {experiment_dir}")
    return ckpts[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=str, help="e.g. outputs/full_run_1")
    parser.add_argument("--probes", type=int, default=8, help="Hutchinson probes for both before/after scoring")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    cfg = load_config(experiment_dir)
    ckpt_path = find_final_checkpoint(experiment_dir)
    device = select_device()
    print(f"Using device: {device}")
    print(f"Before = initialization (seed={cfg.seed}); After = {ckpt_path.name}")

    set_seed(cfg.seed)
    _, probe_loader, vocab_size = build_loaders(cfg)
    before_model = build_model(cfg, vocab_size, device)

    ckpt = torch.load(ckpt_path, map_location=device)
    after_model = build_model(cfg, ckpt["vocab_size"], device)
    after_model.load_state_dict(ckpt["model_state"])

    print("Scoring initialization...")
    before_unsigned, before_signed = compute_sensitivity(
        before_model, probe_loader, device, n_probes=args.probes, show_progress=True
    )
    print("Scoring final checkpoint...")
    after_unsigned, after_signed = compute_sensitivity(
        after_model, probe_loader, device, n_probes=args.probes, show_progress=True
    )

    before_u_flat, boundaries = flatten_scores(before_model, before_unsigned)
    after_u_flat, _ = flatten_scores(after_model, after_unsigned)
    before_s_flat, _ = flatten_scores(before_model, before_signed)
    after_s_flat, _ = flatten_scores(after_model, after_signed)
    labels = group_labels(boundaries, before_u_flat.numel())
    groups = [g for g, _, _ in boundaries]

    before_u_np = before_u_flat.numpy()
    after_u_np = after_u_flat.numpy()
    before_s_np = before_s_flat.numpy()
    after_s_np = after_s_flat.numpy()

    out_all = experiment_dir / "sensitivity_distributions_all.png"
    plot_sensitivity_distributions_all(before_u_np, after_u_np, before_s_np, after_s_np, out_all)
    print(f"Saved {out_all}")

    out_by_module = experiment_dir / "sensitivity_distributions_by_module.png"
    plot_sensitivity_distributions_by_module(
        before_u_np, after_u_np, before_s_np, after_s_np, labels, groups, out_by_module
    )
    print(f"Saved {out_by_module}")


if __name__ == "__main__":
    main()
