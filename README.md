# Early Freezing of Parameter-Wise Functional Sensitivity

Code for the AXIOM @ NeurIPS 2026 submission. The claim, the controls that defend it, and
the experiments that test it are laid out in **[NARRATIVE.md](NARRATIVE.md)** — read that
first; it is the spine, and every module points back to a claim in it.

## The object of study

$$S_i(\theta) = \mathbb{E}_x \big\| \partial F_\theta(x) / \partial \theta_i \big\|_2^2$$

Label-free and loss-free — a property of the realised function. It is the diagonal of the
parameter-space Gram matrix $Q_\theta = \frac1n G_\theta^\top G_\theta$, dual to the
empirical NTK $K_\theta = \frac1n G_\theta G_\theta^\top$, with
$\mathrm{tr}(K_\theta) = \mathrm{tr}(Q_\theta) = \sum_i S_i(\theta)$.

We study the **ordering** $i \mapsto S_i(\theta_t)$ and how it moves during training.

## Layout

```
fsd/                  library
  sensitivity.py        S(theta) via per-example gradients; exact and Hutchinson estimators
  rank_metrics.py       overlap, chance correction, within-layer rho, budget/placement split
  theory.py             kernel velocity, drift model, counting bound, tail statistics
  run.py                one run: train, measure on a log schedule, emit metrics.json
  prune.py              C6 prune-and-continue panel
  models/, data/        ViT / ResNet-20 / MLP / tiny GPT; CIFAR-10/100 / char-level text
experiments/          e1 core measurement, e2 t* scaling, e3 pruning, e4 laziness, e5 failure modes
analysis/
  claims.py             reads results/ and prints each claim SUPPORTED / NOT SUPPORTED
  figures.py            the four paper figures
paper/                axiom2026.tex (4 pages) + figures
tests/                smoke test, sample-budget calibration, timing pilot
```

## Reproducing

```bash
python tests/fetch_data.py
```

```bash
python -m experiments.e1_rank_stability --run --skip-done
```

```bash
python -m analysis.claims --tag e1 --per-run && python -m analysis.figures --all
```

Every experiment takes `--list`, `--run`, and `--emit DIR`. `--emit` writes one JSON config
per run plus a Slurm array script, so moving a sweep to a cluster changes nothing except
which flag you pass:

```bash
python -m experiments.e2_tstar_scaling --emit jobs/e2 && sbatch jobs/e2/submit.sh
```

## Two measurement decisions that carry the paper

**Per-example gradients.** Squaring a batch-averaged gradient estimates $(\mathbb E g)^2$,
not $\mathbb E[g^2]$; the two differ by the gradient covariance. `fsd/sensitivity.py` uses
`torch.func.vmap` to square per example, which is the definition in $S_i$ and has lower
variance for the same compute. For $d_y \le 32$ the estimator is exact — no probe noise.

**Chance correction.** Sensitivity spans orders of magnitude across layers, so a global
top-$k$ set mostly encodes a per-layer budget. On a synthetic control whose two rankings
share *nothing* within any layer, raw top-$k$ overlap is 0.83 and Spearman $\rho$ is 0.98,
while the adjusted overlap is $-0.004$. Reported numbers are adjusted.

The sample budget (2048 inputs) comes from `tests/calibrate_samples.py`, which measures the
between-fold agreement — the ceiling on any stability number — as a function of set size.
