"""
How much does parameter-wise sensitivity agree with its final-epoch value
(and, separately, its previous-checkpoint value), over training? Reports
three complementary statistics -- Pearson r, Spearman rho, Kendall tau --
see common/rank_stability.py for what each captures and
common/rank_stability_runner.py for the shared orchestration (identical
logic is used by function_regression/rank_stability.py).

The whole-model ("total") curve is shown at two resolutions: a dense one
(every tracked epoch) from the pooled heatmap data already saved by
train.py, and an exact one recomputed at full parameter resolution for each
saved checkpoint (plus epoch 0, reconstructed from the seed). Per-module
(embedding/lstm/head) breakdowns are *only* shown at full resolution -- the
pooled heatmap allocates rows to modules proportionally to their parameter
count, so a small module like the embedding layer can collapse to a handful
of pooled rows (or, for this model, zero -- it's smaller than one pooling
bin), too few points for any of these statistics to mean anything.

Full-resolution scoring is cached (per checkpoint) in
<experiment_dir>/full_resolution_sensitivity_cache.npz, since it's the
expensive step and doesn't change if you just want to look at it again or
add another statistic.

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from dataset import CharSequenceDataset, build_vocab, encode, load_text
from model import CharLSTM
from rank_stability_runner import run_rank_stability_analysis
from train import Config, build_loaders, select_device, set_seed


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

    text = load_text(cfg.data_path)
    vocab, char2idx = build_vocab(text)
    text_as_int = encode(text, char2idx)

    def build_probe_loader_and_init_model():
        set_seed(cfg.seed)
        dataset = CharSequenceDataset(text_as_int, cfg.seq_length)
        _, probe_loader = build_loaders(cfg, dataset)
        model = CharLSTM(len(vocab), cfg.embedding_dim, cfg.rnn_units).to(device)
        return probe_loader, model

    def load_model_from_checkpoint(ckpt_path: Path):
        model = CharLSTM(len(vocab), cfg.embedding_dim, cfg.rnn_units).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"])
        return model

    run_rank_stability_analysis(
        experiment_dir, device, args.probes, args.recompute,
        build_probe_loader_and_init_model, load_model_from_checkpoint,
    )


if __name__ == "__main__":
    main()
