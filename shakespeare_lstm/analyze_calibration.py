"""
Why can a pruned+retrained model have LOWER cross-entropy loss but LOWER
top-1 accuracy than the dense baseline, at the same time?

Cross-entropy is -log p(true token) -- it's unbounded above, so a handful of
tokens where the model is confidently *wrong* can dominate the mean even
though they're a tiny fraction of tokens. Top-1 accuracy only asks whether
the single highest-probability token is correct, and doesn't care how
confident that guess was, or how bad the other guesses were. So a model that
hedges more (flatter, less confident predictions) can lose a few points of
accuracy while avoiding the handful of catastrophic high-loss tokens that
were dragging the confident dense model's *mean* loss up -- even if the
dense model is "right" more often per-token.

This script checks that story directly against the data: mean vs. median
per-token loss (a big mean/median gap indicates a heavy tail of costly
mistakes), predicted-distribution entropy and max-probability (confidence),
and, at the token level, who is right/wrong/how-costly on which model.

Usage:
    python analyze_calibration.py outputs/full_run_60ep --branch-checkpoint outputs/full_run_60ep/pruning_keep0.20/branch_prune40_final.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from dataset import CharSequenceDataset, build_vocab, encode, load_text
from model import CharLSTM
from train import Config, build_loaders, select_device


def load_config(experiment_dir: Path) -> Config:
    history = json.loads((experiment_dir / "history.json").read_text())
    cfg_dict = {k: v for k, v in history["config"].items() if k != "output_dir"}
    return Config(**cfg_dict)


@torch.no_grad()
def per_token_stats(model: CharLSTM, loader, device: torch.device):
    """Returns flat numpy arrays, one entry per token in the probe set:
    loss (-log p_true), entropy of the predicted distribution, max
    probability (confidence), and whether the top-1 guess was correct."""
    model.eval()
    losses, entropies, confidences, corrects = [], [], [], []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits, _ = model(inputs)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        true_log_p = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        loss = -true_log_p
        entropy = -(probs * log_probs).sum(dim=-1)
        confidence, pred = probs.max(dim=-1)
        correct = (pred == targets)

        losses.append(loss.reshape(-1).cpu().numpy())
        entropies.append(entropy.reshape(-1).cpu().numpy())
        confidences.append(confidence.reshape(-1).cpu().numpy())
        corrects.append(correct.reshape(-1).cpu().numpy())

    return (
        np.concatenate(losses), np.concatenate(entropies),
        np.concatenate(confidences), np.concatenate(corrects),
    )


def summarize(name: str, loss: np.ndarray, entropy: np.ndarray, confidence: np.ndarray, correct: np.ndarray) -> None:
    print(f"\n--- {name} ---")
    print(f"n_tokens={len(loss)}")
    print(f"accuracy: {correct.mean():.4f}")
    print(f"loss: mean={loss.mean():.4f}  median={np.median(loss):.4f}  "
          f"p90={np.percentile(loss, 90):.4f}  p99={np.percentile(loss, 99):.4f}  max={loss.max():.4f}")
    print(f"predicted-distribution entropy: mean={entropy.mean():.4f}  median={np.median(entropy):.4f} "
          f"(max possible for 66 classes = {np.log(66):.4f})")
    print(f"confidence (max softmax prob): mean={confidence.mean():.4f}  median={np.median(confidence):.4f}")
    # contribution of the worst 1% of tokens to the total loss
    worst_1pct = np.sort(loss)[-max(1, len(loss) // 100):]
    print(f"share of total loss from worst 1% of tokens: {worst_1pct.sum() / loss.sum():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=str)
    parser.add_argument("--branch-checkpoint", type=str, required=True,
                         help="e.g. outputs/full_run_60ep/pruning_keep0.20/branch_prune40_final.pt")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    cfg = load_config(experiment_dir)
    device = select_device()
    print(f"Using device: {device}")

    text = load_text(cfg.data_path)
    vocab, char2idx = build_vocab(text)
    text_as_int = encode(text, char2idx)
    dataset = CharSequenceDataset(text_as_int, cfg.seq_length)
    _, probe_loader = build_loaders(cfg, dataset)

    baseline = CharLSTM(len(vocab), cfg.embedding_dim, cfg.rnn_units).to(device)
    baseline.load_state_dict(torch.load(experiment_dir / f"ckpt_epoch{cfg.epochs}.pt", map_location=device)["model_state"])

    pruned = CharLSTM(len(vocab), cfg.embedding_dim, cfg.rnn_units).to(device)
    pruned.load_state_dict(torch.load(args.branch_checkpoint, map_location=device)["model_state"])

    b_loss, b_ent, b_conf, b_correct = per_token_stats(baseline, probe_loader, device)
    p_loss, p_ent, p_conf, p_correct = per_token_stats(pruned, probe_loader, device)

    summarize("Baseline (dense, unpruned)", b_loss, b_ent, b_conf, b_correct)
    summarize("Pruned + retrained", p_loss, p_ent, p_conf, p_correct)

    # Token-level cross-tabulation: where do they disagree, and at what cost?
    both_right = b_correct & p_correct
    both_wrong = (~b_correct) & (~p_correct)
    only_baseline_right = b_correct & (~p_correct)
    only_pruned_right = (~b_correct) & p_correct

    print("\n--- Token-level agreement ---")
    print(f"both right:          {both_right.mean():.4f} of tokens")
    print(f"both wrong:          {both_wrong.mean():.4f} of tokens")
    print(f"only baseline right: {only_baseline_right.mean():.4f} of tokens "
          f"(baseline loss here: mean={b_loss[only_baseline_right].mean():.3f}, "
          f"pruned loss here: mean={p_loss[only_baseline_right].mean():.3f})")
    print(f"only pruned right:   {only_pruned_right.mean():.4f} of tokens "
          f"(baseline loss here: mean={b_loss[only_pruned_right].mean():.3f}, "
          f"pruned loss here: mean={p_loss[only_pruned_right].mean():.3f})")

    print("\n--- Where does baseline's extra loss come from? ---")
    print(f"baseline mean loss on 'both wrong' tokens:  {b_loss[both_wrong].mean():.4f}")
    print(f"pruned   mean loss on 'both wrong' tokens:  {p_loss[both_wrong].mean():.4f}")
    print(f"(if baseline is much higher here, it is being confidently wrong on tokens neither model gets right)")


if __name__ == "__main__":
    main()
