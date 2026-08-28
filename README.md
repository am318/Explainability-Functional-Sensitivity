# Repository Structure

```
.
├── common/                  shared across every sensitivity experiment
│   ├── sensitivity.py       unsigned S_i and signed Sbar_i estimators
│   ├── experiment.py        the train-and-track loop
│   ├── optimizers.py        adam / sgd / sgd_momentum
│   ├── sweep.py             optimizer x learning-rate sweeps
│   ├── plotting.py          loss, sensitivity and heatmap figures
│   ├── rank_stability.py    Pearson / Spearman / Kendall curves
│   ├── rank_stability_runner.py
│   └── pruning.py
├── twomoons_mlp/            2->32->32->2 MLP, 1,218 parameters
├── mnist_cnn/               8/16-channel CNN, 9,098 parameters
├── shakespeare_lstm/        char-level LSTM
├── summary_figure.py        cross-experiment summary figure
├── Pruning_diagnostics/     zero-shot pruning methods (SNIP, SynFlow, …)
├── Old Pruning Tests/       legacy implementations, kept for reference
└── requirements.txt
```

- `common/` holds everything shared: an experiment directory supplies only its dataset, model and config.
- Each experiment directory follows the same layout — `dataset.py`, `model.py`, `train.py`, `rank_stability.py`, `smoke_test.py`, and (where there is a sweep) `sweep.py`.
- `Pruning_diagnostics/` contains the main implementation of the zero-shot pruning methods and supporting modules.
- `Old Pruning Tests/` contains legacy implementations and experimental code retained for reference.
- `requirements.txt` lists the Python package dependencies.

# Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

# Sensitivity Tracking Experiments

Each experiment trains a model while estimating per-parameter functional
sensitivity on a held-out probe set, then measures how much the *ordering*
of parameters by sensitivity at epoch `e` agrees with its final-epoch
ordering.

Each `sweep.py` tries one shared learning-rate grid on each of Adam, SGD and
SGD+momentum, picks the best learning rate per optimizer, then re-trains
those three with sensitivity tracking and per-epoch checkpointing before
running the rank-stability analysis over them.

```bash
# Two moons: 15 selection runs + 3 instrumented runs, ~1 minute on CPU
DEVICE=cpu python twomoons_mlp/sweep.py

# MNIST: the same, and worth a GPU (DEVICE is auto-detected)
python mnist_cnn/sweep.py

# One figure spanning every experiment that has been run
python summary_figure.py
```

MNIST downloads itself into `mnist_cnn/data/` on first use. `DEVICE` (cpu /
cuda / mps) overrides device selection; see each `train.py` for the other
environment variables.

Each sweep writes to `<experiment>/outputs/`: `sweep_learning_rates.png`,
`optimizer_comparison.png`, `sweep_summary.json`, and a directory per run
holding `history.json`, `loss_and_sensitivity.png`,
`parameter_sensitivity_heatmap.png` and `rank_stability_*.png`.

Before committing a cluster job, `smoke_test.py` in each experiment
directory exercises the whole pipeline in seconds.

# Running the Zero-Shot Pruning Algorithms

The main pruning scripts are located in the `Pruning_diagnostics` directory. Each script implements a different zero-shot pruning method for the ViT-Tiny model.

### SNIP

```bash
python Pruning_diagnostics/Pruning_SNIP_ViT_Tiny.py
```

### SynFlow

```bash
python Pruning_diagnostics/Pruning_SYNFLOW_ViT_Tiny.py
```

### Sensitivity

```bash
python Pruning_diagnostics/Pruning_Sensitivity_ViT_Tiny.py
```

Each script runs the corresponding zero-shot pruning algorithm (SNIP, SynFlow, or Sensitivity) using the shared utility modules in `Pruning_diagnostics`.
