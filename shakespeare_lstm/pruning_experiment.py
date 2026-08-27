"""
Does WHEN you prune matter, for the same amount pruned?

For each candidate epoch e in --prune-epochs: take the model as it was at
epoch e (epoch 0 = reconstructed initialization, else a saved checkpoint),
rank all parameters globally by unsigned functional sensitivity computed at
that point, keep only the top --keep-fraction, zero the rest, and:

  1. evaluate loss/accuracy on the held-out probe set immediately (no
     further training) -- "how much does this pruning hurt right now",
  2. continue training the pruned model (mask enforced every step) up to
     the same total epoch budget the source run used, then evaluate again
     -- "does pruning early vs late change what you converge to".

Compared throughout against the source run's own (unpruned) final loss.
Reuses the source run's saved checkpoints as branch points rather than
retraining each candidate from scratch -- see notes.md.

Usage:
    python pruning_experiment.py outputs/full_run_60ep
    python pruning_experiment.py outputs/full_run_60ep --keep-fraction 0.2 \\
        --prune-epochs 0 10 20 30 40 50 60
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from dataset import CharSequenceDataset, build_vocab, encode, load_text
from model import CharLSTM
from pruning import apply_mask_, compute_topk_mask, mask_stats, zero_grad_for_mask_
from sensitivity import compute_sensitivity
from train import Config, build_loaders, select_device, set_seed


def load_config(experiment_dir: Path) -> Config:
    history = json.loads((experiment_dir / "history.json").read_text())
    cfg_dict = {k: v for k, v in history["config"].items() if k != "output_dir"}
    return Config(**cfg_dict)


@torch.no_grad()
def evaluate(model: CharLSTM, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    # No non_blocking=True: unsafe on MPS with non-pinned tensors, can
    # silently corrupt data (see module docstring investigation).
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits, _ = model(inputs)
        loss = criterion(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
        bsz = inputs.shape[0] * inputs.shape[1]
        loss_sum += float(loss.item()) * bsz
        correct += int((logits.argmax(dim=-1) == targets).sum().item())
        n += bsz
    return loss_sum / max(1, n), correct / max(1, n)


def train_one_masked_epoch(
    model: CharLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    masks: Dict[str, torch.Tensor],
    grad_clip: float,
    desc: str,
) -> float:
    model.train()
    loss_sum, n = 0.0, 0
    batches = tqdm(loader, desc=desc, leave=False)
    for inputs, targets in batches:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(inputs)
        loss = criterion(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
        loss.backward()
        zero_grad_for_mask_(model, masks)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        apply_mask_(model, masks)
        bsz = inputs.shape[0]
        loss_sum += float(loss.item()) * bsz
        n += bsz
        batches.set_postfix(loss=loss.item())
    return loss_sum / max(1, n)


def build_model_at_epoch(
    epoch: int, cfg: Config, vocab_size: int, device: torch.device, experiment_dir: Path
) -> CharLSTM:
    if epoch == 0:
        set_seed(cfg.seed)
        return CharLSTM(vocab_size, cfg.embedding_dim, cfg.rnn_units).to(device)
    ckpt_path = experiment_dir / f"ckpt_epoch{epoch}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path} -- pick a prune epoch with a saved checkpoint")
    model = CharLSTM(vocab_size, cfg.embedding_dim, cfg.rnn_units).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    return model


def run_branch(
    prune_epoch: int,
    final_epoch: int,
    cfg: Config,
    vocab_size: int,
    device: torch.device,
    experiment_dir: Path,
    train_loader: DataLoader,
    probe_loader: DataLoader,
    criterion: nn.Module,
    keep_fraction: float,
    n_probes: int,
    out_dir: Path,
) -> Dict:
    print(f"\n=== Branch: prune at epoch {prune_epoch} ===")
    model = build_model_at_epoch(prune_epoch, cfg, vocab_size, device, experiment_dir)

    unsigned, _ = compute_sensitivity(model, probe_loader, device, n_probes=n_probes, show_progress=True)
    masks = compute_topk_mask(model, unsigned, keep_fraction)
    stats = mask_stats(masks)
    apply_mask_(model, masks)
    print(f"Mask: kept {stats['kept']}/{stats['total']} params ({stats['kept_fraction']:.4f})")

    immediate_loss, immediate_acc = evaluate(model, probe_loader, criterion, device)
    print(f"Immediate post-prune: loss={immediate_loss:.4f} acc={immediate_acc:.4f}")

    remaining = final_epoch - prune_epoch
    epoch_losses: List[float] = []
    test_losses: List[float] = []
    test_accs: List[float] = []
    if remaining > 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        start = time.time()
        for i in range(1, remaining + 1):
            loss = train_one_masked_epoch(
                model, train_loader, optimizer, criterion, device, masks, cfg.grad_clip,
                desc=f"prune@{prune_epoch} retrain {i}/{remaining}",
            )
            epoch_losses.append(loss)
            # Cheap (forward passes only, ~512 held-out chunks) relative to
            # the training epoch itself, so evaluated every epoch to give a
            # real test-loss trajectory rather than just endpoints.
            test_loss, test_acc = evaluate(model, probe_loader, criterion, device)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            tqdm.write(f"  prune@{prune_epoch} | retrain epoch {i}/{remaining} | "
                       f"train_loss={loss:.4f} | test_loss={test_loss:.4f}")
        print(f"Retraining took {time.time() - start:.1f}s")

    final_loss, final_acc = evaluate(model, probe_loader, criterion, device)
    print(f"Final (epoch {final_epoch}): loss={final_loss:.4f} acc={final_acc:.4f}")

    ckpt_path = out_dir / f"branch_prune{prune_epoch}_final.pt"
    torch.save({"model_state": model.state_dict(), "masks": masks, "prune_epoch": prune_epoch}, ckpt_path)
    print(f"Saved final pruned+retrained model to {ckpt_path}")

    return {
        "prune_epoch": prune_epoch,
        "kept_fraction": stats["kept_fraction"],
        "immediate_loss": immediate_loss,
        "immediate_acc": immediate_acc,
        "final_loss": final_loss,
        "final_acc": final_acc,
        "retrain_epoch_losses": epoch_losses,
        "retrain_test_losses": test_losses,
        "retrain_test_accs": test_accs,
    }


def plot_results(
    results: List[Dict], baseline_loss: float, baseline_acc: float, keep_fraction: float, out_path: Path,
    baseline_train_loss: float = None,
) -> None:
    epochs = [r["prune_epoch"] for r in results]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13.0, 5.5), constrained_layout=True)

    ax_loss.axhline(baseline_loss, color="black", linestyle="--", label="unpruned baseline, held-out (final)")
    if baseline_train_loss is not None:
        ax_loss.axhline(baseline_train_loss, color="gray", linestyle=":",
                         label="unpruned baseline, TRAIN loss (final) -- overfitting gap")
    ax_loss.plot(epochs, [r["immediate_loss"] for r in results], "o-", color="tab:red",
                 label="immediately after pruning, no retraining")
    ax_loss.plot(epochs, [r["final_loss"] for r in results], "o-", color="tab:blue",
                 label="after pruning + retraining to final epoch (held-out)")
    train_final = [r["retrain_epoch_losses"][-1] if r["retrain_epoch_losses"] else r["final_loss"] for r in results]
    ax_loss.plot(epochs, train_final, "o--", color="tab:cyan", markersize=4, alpha=0.7,
                 label="after pruning + retraining (TRAIN loss)")
    ax_loss.set_xlabel("Epoch pruning was applied")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_yscale("log")
    ax_loss.set_title(f"Loss vs. when pruning was applied (keep {keep_fraction:.0%})")
    ax_loss.legend(fontsize=7)

    ax_acc.axhline(baseline_acc, color="black", linestyle="--", label="unpruned baseline (final)")
    ax_acc.plot(epochs, [r["immediate_acc"] for r in results], "o-", color="tab:red",
                label="immediately after pruning, no retraining")
    ax_acc.plot(epochs, [r["final_acc"] for r in results], "o-", color="tab:blue",
                label="after pruning + retraining to final epoch")
    ax_acc.set_xlabel("Epoch pruning was applied")
    ax_acc.set_ylabel("Held-out next-char accuracy")
    ax_acc.set_title(f"Accuracy vs. when pruning was applied (keep {keep_fraction:.0%})")
    ax_acc.legend(fontsize=8)

    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=str, help="e.g. outputs/full_run_60ep")
    parser.add_argument("--keep-fraction", type=float, default=0.2)
    parser.add_argument("--prune-epochs", type=int, nargs="+", default=[0, 10, 20, 30, 40, 50, 60])
    parser.add_argument("--probes", type=int, default=8, help="Hutchinson probes for the pruning-ranking sensitivity")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    cfg = load_config(experiment_dir)
    final_epoch = cfg.epochs
    device = select_device()
    print(f"Using device: {device}")
    print(f"Source run: {experiment_dir}, final_epoch={final_epoch}, keep_fraction={args.keep_fraction}")

    text = load_text(cfg.data_path)
    vocab, char2idx = build_vocab(text)
    text_as_int = encode(text, char2idx)
    dataset = CharSequenceDataset(text_as_int, cfg.seq_length)
    train_loader, probe_loader = build_loaders(cfg, dataset)
    criterion = nn.CrossEntropyLoss()

    baseline_model = build_model_at_epoch(final_epoch, cfg, len(vocab), device, experiment_dir)
    baseline_loss, baseline_acc = evaluate(baseline_model, probe_loader, criterion, device)
    baseline_train_loss = json.loads((experiment_dir / "history.json").read_text())["history"][-1]["loss"]
    print(f"Unpruned baseline (epoch {final_epoch}): held-out loss={baseline_loss:.4f} acc={baseline_acc:.4f} "
          f"(train loss was {baseline_train_loss:.4f})")

    out_dir = experiment_dir / f"pruning_keep{args.keep_fraction:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for prune_epoch in args.prune_epochs:
        result = run_branch(
            prune_epoch, final_epoch, cfg, len(vocab), device, experiment_dir,
            train_loader, probe_loader, criterion, args.keep_fraction, args.probes, out_dir,
        )
        results.append(result)
        with open(out_dir / "results.json", "w") as f:
            json.dump(
                {
                    "config": asdict(cfg),
                    "keep_fraction": args.keep_fraction,
                    "baseline_loss": baseline_loss,
                    "baseline_acc": baseline_acc,
                    "results": results,
                },
                f, indent=2,
            )
        print(f"Saved progress to {out_dir / 'results.json'}")

    plot_path = out_dir / "pruning_vs_epoch.png"
    plot_results(results, baseline_loss, baseline_acc, args.keep_fraction, plot_path, baseline_train_loss)
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
