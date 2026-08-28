"""
How much does parameter-wise sensitivity agree with its final-epoch value
(and, separately, its previous-checkpoint value), over training? See
shakespeare_lstm/rank_stability.py for the full explanation -- this is the
same analysis, reusing common/rank_stability_runner.py, on the WikiText-2
AWD-LSTM instead.

Usage:
    python rank_stability.py outputs/full_run_1
    python rank_stability.py outputs/full_run_1 --probes 8
    python rank_stability.py outputs/full_run_1 --recompute  # ignore cache
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))

from rank_stability_runner import run_rank_stability_analysis
from train import Config, build_loaders, build_model, select_device, set_seed


def load_config(experiment_dir: Path) -> Config:
    history = json.loads((experiment_dir / "history.json").read_text())
    cfg_dict = {k: v for k, v in history["config"].items() if k != "output_dir"}
    return Config(**cfg_dict)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=str, help="e.g. outputs/full_run_1")
    parser.add_argument("--probes", type=int, default=8, help="Hutchinson probes for the full-resolution scoring")
    parser.add_argument("--recompute", action="store_true", help="Ignore any cached full-resolution scores")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    cfg = load_config(experiment_dir)
    device = select_device()
    print(f"Using device: {device}")

    def build_probe_loader_and_init_model():
        set_seed(cfg.seed)
        _, probe_loader, vocab_size = build_loaders(cfg)
        model = build_model(cfg, vocab_size, device)
        return probe_loader, model

    def load_model_from_checkpoint(ckpt_path: Path):
        state = torch.load(ckpt_path, map_location=device)
        model = build_model(cfg, state["vocab_size"], device)
        model.load_state_dict(state["model_state"])
        return model

    run_rank_stability_analysis(
        experiment_dir, device, args.probes, args.recompute,
        build_probe_loader_and_init_model, load_model_from_checkpoint,
    )


if __name__ == "__main__":
    main()
