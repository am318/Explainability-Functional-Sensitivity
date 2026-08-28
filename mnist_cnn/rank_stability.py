"""
How much does parameter-wise sensitivity agree with its final-epoch value
(and, separately, its previous-checkpoint value), over training? Reports
three complementary statistics -- Pearson r, Spearman rho, Kendall tau --
see common/rank_stability.py for what each captures and
common/rank_stability_runner.py for the shared orchestration (identical
logic is used by twomoons_mlp/ and shakespeare_lstm/).

The whole-model ("total") curve is shown at two resolutions: a dense one
(every tracked epoch) from the pooled heatmap data already saved by
train.py, and an exact one recomputed at full parameter resolution for each
saved checkpoint (plus epoch 0, reconstructed from the seed). Per-module
(conv1/conv2/head) breakdowns are *only* shown at full resolution, and this
model is the reason why: the pooled heatmap allocates rows proportionally to
parameter count, so conv1's 80 parameters (0.9% of the model) collapse to
about two of the 300 pooled rows -- far too few for any of these statistics
to mean anything.

Full-resolution scoring is cached (per checkpoint) in
<experiment_dir>/full_resolution_sensitivity_cache.npz, since it's the
expensive step and doesn't change if you just want to look at it again or
add another statistic.

Usage:
    python rank_stability.py outputs/best/adam_lr0.003
    python rank_stability.py outputs/best/adam_lr0.003 --probes 16
    python rank_stability.py outputs/best/adam_lr0.003 --recompute  # ignore cache
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from experiment import select_device, set_seed
from model import MnistCNN
from rank_stability_runner import run_rank_stability_analysis
from train import Config, build_experiment


def load_config(experiment_dir: Path) -> Config:
    history = json.loads((experiment_dir / "history.json").read_text())
    cfg_dict = {k: v for k, v in history["config"].items() if k != "output_dir"}
    return Config(**cfg_dict)


def analyze(experiment_dir: Path, device: torch.device, probes: int = 8, recompute: bool = False) -> None:
    cfg = load_config(experiment_dir)

    def build_probe_loader_and_init_model():
        # Same seed, same construction order as training: build_experiment
        # rebuilds the data and loaders before the model, so the model's
        # initialization is bit-for-bit the one the run started from.
        set_seed(cfg.seed)
        exp = build_experiment(cfg, device)
        return exp.probe_loader, exp.model

    def load_model_from_checkpoint(ckpt_path: Path):
        model = MnistCNN(channels1=cfg.channels1, channels2=cfg.channels2).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"])
        return model

    run_rank_stability_analysis(
        experiment_dir, device, probes, recompute,
        build_probe_loader_and_init_model, load_model_from_checkpoint,
        include_signed=cfg.track_signed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=str, help="e.g. outputs/best/adam_lr0.003")
    parser.add_argument("--probes", type=int, default=8, help="Hutchinson probes for the full-resolution scoring")
    parser.add_argument("--recompute", action="store_true", help="Ignore any cached full-resolution scores")
    args = parser.parse_args()

    device = select_device()
    print(f"Using device: {device}")
    analyze(Path(args.experiment_dir), device, args.probes, args.recompute)


if __name__ == "__main__":
    main()
