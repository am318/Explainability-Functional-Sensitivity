# Paper narrative — the spine of this repository

**Working title:** *When does a network decide what matters? The early freezing of
parameterwise functional sensitivity.*

**Venue:** AXIOM @ NeurIPS 2026 — 4 pages excl. references/appendix, deadline 2026-08-29 23:59 UTC.

**One-sentence claim.** The *ordering* of parameters by functional sensitivity
`S(theta) = E_x || d f(x) / d theta ||^2` — a label-free, loss-free quantity — is
essentially fixed after a short, measurable, and predictable initial phase of training,
and it is fixed *most strongly exactly at the top-k boundary that pruning depends on*.

**Why it matters here.** This is a *predictive principle for efficient AI*: it says the
information a pruning criterion needs is available at step `t*`, and it tells you how to
find `t*` without training to completion.

**Relationship to the follow-up paper.** This paper establishes the *measurement and the
mechanism*. The follow-up paper spends the result: pruning early in training by functional
sensitivity. We therefore include only a single feasibility panel here (C6), not a full
pruning study.

---

## Claims, and what would falsify each

Every experiment module names the claim it tests. `analysis/claims.py` reads the results
and prints each claim's current status, so the narrative is executable rather than aspirational.

### C1 — Freezing
The adjusted top-k overlap between `S_t` and `S_final` rises quickly and plateaus:
there is a `t*` with `t* << T` beyond which the ordering barely moves.

**`t*` is defined without a free threshold.** `S` is estimated on finitely many inputs, so
there is a hard ceiling on any agreement we can measure: the agreement of two *independent
estimates at the same parameters*. We therefore set

> `t*` = the first step after which **further training** moves the ordering **less than
> resampling the estimation data** does.

Two error sources — trajectory drift and estimation noise — placed on one axis, with the
crossing reported. This is the trajectory counterpart of the estimation-error lemma for the
conditional population ranking in the companion draft, which bounds `|S-hat - S|` at fixed
`theta`; here we bound the analogous quantity across `theta_t`. The fold comparison uses
half-size folds, so it is Spearman–Brown corrected to the full sample size before being
used as the ceiling — the correction moves it in the *tighter*, not the flattering,
direction.

*Falsified if:* overlap rises smoothly and only reaches its plateau near the end of
training, or if `t*` is a large fraction of `T`.

### C2 — It is not an artefact
The freezing survives three controls that kill most "ranking is stable" claims:

- **C2a Noise floor.** Two *disjoint data folds* at the same checkpoint give an upper bound
  on any overlap we can measure. Every stability curve is plotted against this band.
  *Falsified if:* the reported plateau sits at or below the noise floor.
- **C2b Layerwise confound.** Sensitivity varies by orders of magnitude across layers, so a
  global ranking mostly encodes "which layer". We report **adjusted overlap**
  (chance-corrected against a layer-budget-matched random baseline) and **within-layer**
  rank correlation. *Falsified if:* adjusted overlap is near zero, i.e. all the apparent
  stability is the layer budget.
- **C2c Not simply fixed at init.** `S_0` must be measurably worse than the plateau.
  *Falsified if:* `S_0` already achieves plateau overlap — the story then becomes
  "determined at initialisation", which is a different paper and we would say so.

### C3 — Boundary beats bulk
Stability is *higher at the top-k decision boundary* than in the bulk of the spectrum.
Pruning only ever asks "is this parameter in the top k?", so this is the quantity that matters.

*Falsified if:* top-k overlap tracks (or lags) global Spearman rather than exceeding it.

### C4 — The timescale is predictable
`t*` is not a magic constant: it moves systematically with width, depth, learning rate and
batch size, and it coincides with the departure-from-lazy transition measured by
**kernel velocity** (normalised drift of the empirical NTK Gram).

*Falsified if:* `t*` is unrelated to kernel velocity, or scatters unsystematically across
hyperparameters.

### C5 — Theory: drift vs. gap
Two mechanisms compose:

1. **Lazy conservation.** `S` is the diagonal of the Gauss–Newton / NTK operator in
   parameter space. In the linearised regime the Jacobian is constant, so the ordering is
   *exactly* conserved. Rank churn is therefore a direct measurement of departure from laziness.
2. **Heavy-tail gap.** A rank swap at the cut index `k` needs drift larger than the local
   gap in the sorted sensitivity spectrum. That spectrum is heavy-tailed, so gaps near the
   top are large while the bulk is dense — swaps concentrate in the bulk and avoid the
   boundary, which is exactly C3.

Formally, writing `a_i = sqrt(S_i)` for the L2 norm of the i-th Jacobian column and
`Delta_i` for its movement between two checkpoints (Prop. 1 in the paper):

- `|a_i(t) - a_i(T)| <= Delta_i`, and `sum_i Delta_i^2 <= L^2 ||theta_t - theta_T||^2`
  under the same local-Lipschitz Jacobian assumption the companion draft uses for a
  *pruning* perturbation — applied here along the trajectory instead;
- the k-th order statistic is 1-Lipschitz, so a parameter can only leave the top-k if it
  started within `2*Delta` of the cut, giving
  `overlap >= 1 - #{i : |a_i(T) - q_k(T)| <= 2 Delta} / k`.

Two corollaries: in the linearised regime `Delta = 0` and the ordering is **exactly
conserved** (so measured churn lower-bounds departure from lazy training); and with local
spectral density `rho_k` at the cut, `overlap >= 1 - 2 Delta rho_k p / k`, which is
non-vacuous precisely at high sparsity and degrades in the dense bulk — C3, derived rather
than asserted.

Alongside the bound we give a sharper *point* prediction: from the marginal distribution of
`log S` plus **one measured scalar per checkpoint** (the drift scale), predict top-k overlap
simultaneously at every sparsity, with nothing fitted to the overlap curve.

*Falsified if:* the predicted overlap curve does not track the measured one, or the bound
is violated.

### C6 — Consequence (feasibility panel; the follow-up paper's seed)
Pruning at step `t` using `S_t` recovers the accuracy of pruning at `S_final` once `t >= t*`,
and degrades below it.

*Falsified if:* accuracy keeps improving well past `t*`, i.e. `t*` does not predict the
usable prune time.

---

## Figure plan (4 pages)

| Fig | Content | Claims |
|-----|---------|--------|
| 1 | Stability vs. training step: adjusted top-k overlap at several sparsities, with noise-floor band, within-layer curve, and `t=0` marker | C1, C2, C3 |
| 1b | Raw overlap and Spearman against the chance level and the adjusted curve — the panel that answers "your ranking is just a layer budget" | C2b |
| 2 | `t*` across architectures and hyperparameters, plotted against kernel velocity | C4 |
| 3 | Predicted vs. observed overlap from the drift/gap model; heavy-tailed spectrum inset | C5 |
| 4 | Accuracy vs. prune-time `t`, two sparsities (feasibility panel) | C6 |

Settings: ViT-Tiny / ResNet-20 / MLP on CIFAR-10 (+CIFAR-100), and a tiny GPT on
character-level text — the last one matters because `S` is label-free, so it transfers to
next-token prediction without redefinition.

---

## Known related work to position against (check before submitting)

- You et al. 2020, *Early-Bird Tickets* — masks stabilise early, measured by mask Hamming distance.
- Frankle, Dziugaite, Roy, Carbin 2020, *The Early Phase of Neural Network Training*.
- Jastrzebski et al. 2020, *The Break-Even Point on the Optimization Trajectory*.
- Frankle et al. 2021, *Pruning Neural Networks at Initialization: Why Are We Missing the Mark?*
- Su et al. 2020, *Sanity-Checking Pruning Methods: Random Tickets Can Win the Jackpot*.
- Lee et al. 2019 (SNIP), Wang et al. 2020 (GraSP), Tanaka et al. 2020 (SynFlow).
- Fort et al. 2020, *Deep Learning vs. Kernel Learning* — kernel velocity / NTK drift.
- Chizat, Oyallon, Bach 2019, *On Lazy Training in Differentiable Programming*.

Our differentiators: (i) the quantity is **functional and label-free**, not a loss saliency;
(ii) we study the **ordering as a dynamical object with a timescale**, not a binary
"converged" flag; (iii) we give a **mechanism that predicts the curve**, tested causally
via an explicit laziness parameter (E4) rather than only correlationally.

---

## Measurement decisions that the claims rest on

**Per-example gradients.** Squaring a batch-averaged gradient estimates `(E g)^2`, not
`E[g^2]`; the two differ by the gradient covariance, and only the latter is the definition
of `S`. We use `torch.func.vmap` to square per example. For `d_y <= 32` (all the vision
settings) the estimator sums exactly over output coordinates, so there is **no probe noise
at all** and the noise floor is purely a data-sampling quantity.

**Sample budget.** `tests/calibrate_samples.py` measures the ceiling as a function of the
estimation-set size: 64 samples gives 0.74, 512 gives 0.88, 2048 gives 0.96 (adjusted
overlap at sparsity 0.9, ViT-Tiny). Headline runs use 2048; sweeps, which only need `t*`,
use 1024.

**Known limitations to state in the paper.** `t*` is measured against `S_final` of the
*same* run, so it is a statement about a trajectory, not about transfer between runs.
Cross-seed parameterwise comparison is meaningless under permutation symmetry, so the
question "is the frozen ordering a property of the initialisation or of the data?" needs a
same-init/different-data-order design — noted as future work, and directly relevant to
whether early masks transfer.

---

## Findings log

Kept so the narrative tracks what the runs actually showed, including where they
contradicted the plan. Entries are appended, never rewritten.

### 2026-08-25 — E1 pilot (MLP, ResNet-20 on CIFAR-10, 4000 steps, cosine)

**The measurement is sound.** Noise-floor ceilings came in at 0.95 (MLP) and 0.96–0.99
(ResNet), so there is ample headroom above the estimator. Step-0 adjusted overlap was 0.044
(MLP) and 0.020 (ResNet), so the ordering is emphatically *not* fixed at initialisation
(C2c holds, and holds strongly).

**The chance correction is doing real work.** At step 503 the MLP's raw top-$k$ overlap is
0.699 against a layer-budget chance level of 0.397 — adjusted, 0.501. Reporting the raw
number would have looked like early freezing. This is C2b working as designed.

**C1 did not reproduce, but the run was too short to test it.** Adjusted overlap rose
monotonically to 1.0 at $t=T$ with no plateau, giving $t^*=T$. The MLP's training loss was
still descending at step 4000 (1.78 → ~1.1; 10.2 epochs of CIFAR-10), so `S_final` was not
a settled reference and "agreement with final" was substantially measuring *convergence*.

> **Design lesson.** A stability claim needs its reference taken from a converged model.
> Runs must outlast convergence by a margin, or the measurement cannot distinguish "the
> ordering froze" from "training stopped". Default run length raised 4000 → 12000 steps;
> **E7** tests the plateau hypothesis directly at 20000 steps with constant LR.

**A premature reframe, recorded because it was wrong.** The MLP showed a clean 12×
separation between budget (settles step 145, 3.6%) and placement (90% at step 1745, 43.6%),
which looked like a better thesis. ResNet-20 did not reproduce it — budget settles at 43.6%,
placement at 66.0%. One architecture is not a finding. The budget/placement decomposition
stays as an instrument and a reported quantity; it is not (yet) the headline.

**Cosine decay is a live confound.** It drives the LR to ~0 near the end, which flattens
*any* agreement-with-final curve. The constant-LR arm was promoted from an appendix control
into the main E1 panel so the headline cannot be a schedule artefact.

### 2026-08-25 — E1 pilot complete (all three architectures, 4000 steps, cosine)

ViT-Tiny matches MLP and ResNet-20: adjusted overlap rises monotonically, no plateau,
`t* = T`. Three architectures agreeing is a real pattern rather than a single noisy run —
but all three were still training at the cut-off, so what the pattern establishes is
"no freezing within 10 epochs", not "no freezing".

| | MLP | ResNet-20 | ViT-Tiny |
|---|---|---|---|
| final acc | 0.570 | 0.758 | 0.643 |
| ceiling | 0.954 | 0.993 | 0.968 |
| adjusted @ step 0 | 0.044 | 0.020 | 0.057 |
| **raw** overlap @ step 0 | 0.162 | 0.137 | **0.319** |
| chance (layer budget) @ step 0 | 0.123 | 0.119 | **0.278** |
| budget settles | 3.6% | 43.6% | 43.6% |
| placement 90% | 43.6% | 66.0% | 43.6% |

**The sharpest illustration of why chance correction is necessary.** At step 0 — before a
single gradient step — the ViT's *raw* top-k overlap with its own final ordering is 0.319.
Chance, from the layer budget alone, is 0.278. A paper reporting raw overlap would be
reporting a third of its headline agreement from an untrained network.

**The budget/placement separation is architecture-dependent**, not universal: 12× for the
MLP, ~1.5× for ResNet-20, none for ViT. It stays an instrument, not the thesis.

> **Instrumentation gap found and closed.** Agreement with `S_final` is pinned to 1.0 at
> `t = T`, so a monotonic rise cannot distinguish "froze at `t*`" from "still churning,
> and `t` is merely closer to `T`". Every run now emits the **full pairwise
> checkpoint × checkpoint agreement matrix** (~6s/run). Freezing shows as a saturated
> bottom-right block whose corner is `t*`; constant-rate churn shows as diagonal banding
> that never saturates. This, not the vs-final curve, is the diagnostic that settles C1.

### 2026-08-25 — C5 mechanism check on the pilot (n=3)

The heavy-tail corollary predicts three signs, all between independently measured
quantities, and all fixed by the theory before the numbers were looked at:

| run | Hill α | gap/spread at cut | boundary advantage |
|---|---|---|---|
| ViT-Tiny | 1.399 | 0.1295 | +0.160 |
| ResNet-20 | 1.519 | 0.0606 | +0.045 |
| MLP | 1.687 | 0.0222 | −0.048 |

    corr(alpha, boundary advantage)       -0.99   [predicted negative]  OK
    corr(gap/spread, boundary advantage)  +1.00   [predicted positive]  OK
    corr(alpha, gap/spread)               -0.97   [predicted negative]  OK

Getting the sign right on the *intermediate* variable as well as the endpoints is what
separates a mechanism from a correlation.

**Do not overclaim this.** n = 3, and a perfect rank ordering of three items happens with
probability ~1/6 under the null. It is a pre-registered prediction that survived its first
contact with data, nothing more. E1 and E2 supply ~40 further points across widths, depths,
datasets and architectures; `analysis/mechanism.py` scores them automatically.

**Why it matters even if C1 fails.** The claim here is not that the ordering freezes. It is
that *how usable the ordering is at the pruning boundary* is predicted by a cheaply
measurable property of the sensitivity spectrum. That is a predictive principle in the
sense the venue asks for, and it is directly actionable for the follow-up: measure α, and
it tells you how aggressively an early mask can be trusted.

### 2026-08-26 — E7 verdict: C1 is falsified for MLP and ResNet-20

Both architectures, 20000 steps (51 epochs), **constant LR** — so neither truncation nor
schedule decay can account for the result.

| | MLP | ResNet-20 |
|---|---|---|
| final accuracy | 0.585 | **0.866** |
| final train loss | 1.115 | 0.320 |
| adjusted overlap at 66% of training | 0.734 | 0.573 |
| measurement ceiling | 0.953 | 0.938 |
| `t*` | 20000 (= T) | 20000 (= T) |

The pairwise matrix is the decisive read. "Weakest agreement among all checkpoints after
`t`" climbs steadily (MLP 0.027 → 0.734; ResNet 0.000 → 0.621) and **never saturates**.
Freezing would appear as a jump to the ceiling followed by a flat block. There is none in
either. The ResNet is well trained — 86.6% accuracy, train loss 0.32 — so this is not a
story about undertrained models.

**Schedule confound, quantified.** MLP constant vs cosine agree until the cosine LR starts
falling, then diverge: at step 13238 cosine reads 0.853 against constant's 0.734. Cosine
decay inflates the late curve by ~0.12. Neither plateaus.

### 2026-08-26 — What `S` at initialisation actually knows (all runs)

| run | raw@0.9 | chance | adjusted | within-layer ρ |
|---|---|---|---|---|
| MLP / constant | 0.144 | 0.120 | 0.027 | 0.027 |
| ResNet-20 / constant | 0.126 | 0.120 | 0.007 | **−0.071** |
| ViT / cosine | 0.319 | 0.278 | 0.057 | 0.164 |

Raw ≈ chance in every case. **Essentially all of the apparent agreement between the
initialisation ordering and the final ordering is the per-layer budget.** Within a layer,
`S_0` carries almost no information about `S_final`, and for ResNet-20 it is slightly
anti-correlated.

> **Implication for the companion zero-shot pruning draft.** This does *not* say
> sensitivity-based zero-shot pruning fails — `S_final` is not ground truth for "which
> weights should have been kept", since pruning at init changes the trajectory it would be
> compared against. What it says is that the *mechanism* is most likely the **layer
> budget**, not within-layer selection. That is directly testable with the
> budget-matched-random baseline already implemented in `fsd/prune.py`, and it sharpens
> what that paper should claim rather than undermining it.

### 2026-08-26 — C1 falsified for all three architectures; C5 mechanism does NOT replicate

**C1 (freezing): falsified, robustly.** Five long runs (20000 steps ≈ 51 epochs), three
architectures, both schedules. Every one gives `t* = T`, and the pairwise matrix saturates
in none of them.

| arch | sched | acc | ceiling | adj @66% | adj @init | within-layer ρ @init | raw @init | chance @init |
|---|---|---|---|---|---|---|---|---|
| MLP | constant | 0.585 | 0.953 | 0.734 | 0.027 | 0.027 | 0.144 | 0.120 |
| MLP | cosine | 0.610 | 0.953 | 0.853 | 0.031 | 0.062 | 0.151 | 0.123 |
| ResNet-20 | constant | 0.866 | 0.991 | 0.573 | 0.007 | −0.071 | 0.126 | 0.120 |
| ViT | constant | 0.783 | 0.980 | 0.658 | 0.031 | 0.050 | 0.219 | 0.194 |
| ViT | cosine | 0.777 | 0.980 | 0.827 | 0.050 | 0.054 | 0.270 | 0.231 |

The ViT was the architecture most likely to freeze by our own mechanism, and it does not.

**C5 mechanism: NOT replicated.** The n=3 pilot had all three predicted signs correct with
|r| > 0.96. On the five long runs:

    corr(alpha, boundary advantage)        -0.68   [predicted negative]  ok
    corr(gap/spread, boundary advantage)   -0.33   [predicted POSITIVE]  WRONG SIGN
    corr(alpha, gap/spread)                +0.67   [predicted NEGATIVE]  WRONG SIGN

Two of three signs invert. The earlier caveat — a perfect ordering of three items occurs
with probability ~1/6 under the null — was the right one, and the result was a fluke of
short runs. Boundary advantage also shrank from a ±0.16 range at 4000 steps to ±0.06 at
20000. **The tail-index prediction is withdrawn.** C3 is likewise inconsistent: high
sparsity beats low at 3/24 checkpoints in one run and 24/24 in another, with no pattern.

**What survives.** The counting bound of Prop. 1 is violated in 0 of 5 runs, as an upper
bound must be. The drift model's accuracy is architecture-dependent: MAE 0.042–0.053 (ViT),
0.151 (ResNet), 0.243–0.321 (MLP) — good where the spectrum is well separated, poor
otherwise, and we report it that way rather than averaging it into a single flattering
number.

### 2026-08-27 — Reconciling the original observation: Spearman vs. top-$k$

The original observation was made with Spearman $\rho$ over all parameters (per the
`rank_stability` module on the ShakespeareLSTM branch, which computes Pearson/Spearman/Kendall
against the final checkpoint, total and per module). That observation is **correct**, and it
does not conflict with the top-$k$ result. They measure different objects.

Step at which each measure first reaches a level (constant LR, 20000 steps):

| measure | MLP ≥0.7 | ResNet ≥0.7 | ViT ≥0.7 |
|---|---|---|---|
| Spearman ρ (all params) | 13% | 19% | 19% |
| top-$k$ raw @0.9 | 29% | 13% | 66% |
| top-$k$ raw @0.99 | never | 44% | 66% |
| component-level @0.9 | 29% | 19% | 44% |

$\rho$ settles early because it is dominated by the bulk: millions of low-sensitivity
parameters whose relative order is easy to get right and irrelevant to pruning. The top-$k$
set — the only thing a pruner reads — lags it, and the lag grows with sparsity.

**This is the paper, if it survives more settings.** Not "the ordering does/doesn't settle"
but *at which granularity and at which threshold it settles*, with the practical consequence
that the statistic normally reported is not the one a pruner depends on.

**Taken from the ShakespeareLSTM branch.** (i) The three-statistic view — Pearson alongside
Spearman and Kendall, since disagreement localises what changed: high $\rho$ with low
Pearson means the ordering held while magnitudes moved. Pearson (raw and log) is now in
`rank_metrics.compare`. (ii) Per-module breakdown, generalised here into a granularity
ladder: layer ordering → component (per output unit) → individual parameter. (iii) The
CharLSTM, now a fourth architecture family — neither convolutional nor attentional, and the
setting the original observation came from.

Queued locally: E3 prune panel, E9 (ViT/CIFAR-100, GPT/WikiText-2), E10 (LSTM). Cluster
arrays for E1/E2/E4–E10 are emitted and validated.

### 2026-08-27 — E3 prune-and-continue panel (ResNet-20, CIFAR-10)

Dense baseline: 0.830. Prune-and-continue from steps {0, 9, 400, 8000} of an 8000-step run,
sparsities {0.9, 0.99}, criteria {sensitivity, layer-budget-matched random, magnitude}.

| prune step | sp | sensitivity | random | magnitude |
|---|---|---|---|---|
| 0 | 0.9 | 0.547 | 0.658 | **0.700** |
| 0 | 0.99 | 0.103 | 0.103 | **0.490** |
| 9 | 0.9 | 0.583 | 0.665 | **0.700** |
| 9 | 0.99 | 0.101 | 0.097 | **0.495** |
| 400 | 0.9 | 0.693 | **0.748** | 0.737 |
| 400 | 0.99 | 0.395 | **0.526** | 0.497 |

**Sensitivity loses to both baselines at every prune time and sparsity tested.** At sp=0.99
and early prune times it is statistically indistinguishable from pure random (0.103 vs
0.103, 0.101 vs 0.097) — the layer-budget structure buys nothing there — and magnitude beats
it by ~40 points. The gap narrows by step 400 but never closes.

**`prune@8000` is not usable evidence.** Pruning at the final checkpoint of an 8000-step run
leaves `range(prune_step, cfg.train.steps)` empty, so there is zero fine-tuning after
masking. All three criteria collapse identically to ~0.10–0.14 regardless of criterion
(sensitivity 0.137/0.101, random 0.138/0.099, magnitude 0.099/0.099) — confirming this is an
artifact of the test condition, not a property of any criterion. `fsd/prune.py` should be
fixed to either skip `prune_step == steps` or fine-tune for a fixed post-prune budget
regardless of when pruning happens, before this experiment is trusted again.

**One architecture, one seed, one dataset — not general evidence against the method — but a
reproducible result on the paper's own machinery, and the opposite of what "early
sensitivity carries early signal" would predict.** Consistent with the granularity-ladder
finding (fine-grained top-k selection lags badly, especially at high sparsity): if early
sensitivity mostly knows the layer budget, and a *specified* budget-matched random baseline
already captures that, sensitivity has nothing left to contribute at fine granularity.

**Implication flagged for the user, not yet acted on.** This is an uncomfortable data point
for the companion zero-shot pruning paper as currently conceived. Needs replication (more
seeds, more architectures) before being treated as a real finding either way.

### 2026-08-27 — E9 coverage: ViT/CIFAR-100 and GPT/WikiText-2

Both confirm the rho-vs-top-k gap outside CIFAR-10.

| setting | acc | rho>=0.7 (%T) | top-k@0.9 raw>=0.7 (%T) | top-k@0.99 raw>=0.7 (%T) |
|---|---|---|---|---|
| ViT/CIFAR-10 (constant) | 0.783 | 19% | 66% | 66% |
| ViT/CIFAR-100 (constant) | 0.450 | 44% | 66% | 66% |
| GPT/WikiText-2 char-level (constant) | 0.599 (ppl 3.92) | 13% | 29% | 13%* |

*GPT's top-k@0.99 crossing is unreliable: raw overlap is non-monotonic near init (0.138 at
step 0, dips to 0.080 at step 41, recovers by step 141), most likely Hutchinson probe noise
-- GPT is the only setting using the noisy estimator (d_y=18112). Flagged, not trusted at
face value; would want more probes or an exact check before citing this number.

Both settings replicate the core pattern: rho settles well before the top-k set a pruner
would actually use. Three architectures, three datasets/tasks (image classification x2,
character-level next-token prediction), all constant LR, all t*=T. This is now a solid
empirical base for the paper's central claim.

E10 (LSTM, WikiText-2, the setting the original observation came from) is running.

### 2026-08-27 — E10: LSTM confirms, with two wrinkles

LSTM / WikiText-2 char-level, constant LR, 20000 steps, acc 0.555, ppl 4.74.

| | rho>=0.7 | top-k@0.5>=0.7 | top-k@0.9>=0.7 | top-k@0.99>=0.7 |
|---|---|---|---|---|
| step (%T) | 5800 (29%) | 3839 (19%) | 8762 (44%) | 13238 (66%) |

Top-k still lags rho, consistent with every other setting. Two things not to smooth over:

1. **rho starts at ~0 (0.005), not 0.2-0.55 like every other architecture**, and crosses 0.7
   *later* (29%) than most settings rather than earlier. LSTM is the architecture the
   original stability observation came from -- worth being precise that its parameter-level
   rho is not unusually fast to settle; the observation likely rested on a coarser view
   (layer/module) or a different measurement window.
2. **Layer-level ordering (7 layers) is non-monotonic**: rho_mean=1.000 at step 141, drops to
   0.891 at step 737, stays there. With only 7 items a single swap moves the statistic a lot;
   read as small-n noise, not "settled then unsettled", but reported rather than hidden.

### All 10 runs complete -- summary table

| setting | rho>=0.7 | top-k@0.9>=0.7 | ratio |
|---|---|---|---|
| MLP/CIFAR-10 | 13% | 29% | 2.2x |
| ResNet-20/CIFAR-10 | 19% | 13%* | -- |
| ViT/CIFAR-10 | 19% | 66% | 3.5x |
| ViT/CIFAR-100 | 44% | 66% | 1.5x |
| GPT/WikiText-2 | 13% | 29% | 2.2x |
| LSTM/WikiText-2 | 29% | 44% | 1.5x |

*ResNet's top-k crossing is non-monotone/architecture-specific per the earlier E7 log entry.

Five of six settings: top-k lags rho by 1.5-3.5x. Now cross-architecture (MLP, ResNet, ViT,
GPT, LSTM) and cross-task (2x image classification, 2x next-token prediction), all constant
LR so no schedule confound. This is the empirical base for the paper's central claim:
**Spearman rho over all parameters settles early; the top-k set a pruner reads lags it
substantially, and the lag grows with sparsity.**

Local queue is now empty. Remaining work: write this up in the paper, and optionally submit
the cluster arrays (jobs/e1, e2, e8 especially) for more seeds if time allows before the
Aug 29 deadline.

### 2026-08-28 — E3 RETRACTED: the masks were degenerate

Probing the E3 prune result (sensitivity losing to magnitude and to budget-matched random)
found a cause that is neither "too few iterations" nor "architecture-specific". The masks
themselves were invalid.

**Unconstrained global top-k empties whole layers.** Dead layers (zero weights kept) on
ResNet-20, mask taken at step 0:

| sparsity | sensitivity | magnitude | random (budget-matched) |
|---|---|---|---|
| 0.9 | 0/21 | **5/21** | 0/21 |
| 0.99 | **12/21** | **11/21** | **12/21** |

A disconnected network scores at chance regardless of ranking quality. So:

- **E3's sp=0.99 row is meaningless.** 0.103 / 0.103 / 0.490 measured which disconnection
  pattern ResNet-20's residual paths tolerate, not criterion quality. Sensitivity killed
  `layers.0-3` (early, fatal); magnitude killed `layers.3-6` (middle, where the skip path
  carries signal). That is the whole 0.10-vs-0.49 gap.
- **E3's sp=0.9 row is confounded.** Magnitude "won" (0.700 vs 0.547) while killing 5 layers
  outright. In a ResNet a dead conv degrades to identity, so magnitude was effectively doing
  *depth reduction*, not weight selection -- a comparison the architecture flatters.

**All E3 numbers are withdrawn** pending E11. They should not appear in either paper.

**Root cause:** `sensitivity_metrics.min_keep_count` in the earlier codebase guarded against
this; the `fsd/prune.py` rewrite dropped it. Restored as per-layer *count allocation* --
global ranking sets each layer's count, counts are clamped up to a floor, the excess is
reclaimed proportionally from layers with headroom, then each layer keeps its own top-k.
Implemented this way rather than as top-k-then-repair because a repair pass can itself
re-empty a layer whose weights are uniformly low-scoring (observed: magnitude still had 7
dead layers under the repair version). Now 0 dead layers at every sparsity and criterion,
with the global kept-fraction preserved exactly (0.1073 / 0.0181).

**E11 launched** to redo the comparison on valid masks and answer the two open questions:
*(i)* fixed generous fine-tune budget with periodic evaluation, so "slower to recover" is
distinguishable from "permanently worse"; *(ii)* MLP / ResNet-20 / ViT, since ResNet's
residual paths structurally flatter layer-killing criteria and MLP has no such escape route.

**Methodological note for the paper.** This is worth stating explicitly: unconstrained
global top-k pruning at high sparsity produces disconnected networks, and comparisons that
do not check for dead layers can silently measure architecture tolerance instead of
criterion quality. Cheap to check, easy to miss.

### 2026-08-28 — Unit-level disconnection: also present, also fixed

User's follow-up questions ("does naive sensitivity pruning disproportionately affect
information flow?" / "does the deficiency exist at lower sparsity?") led to a finer-grained
check: even with the *layer* floor in place, does sensitivity empty individual *output
units* (neurons, conv filters) within surviving layers?

**Yes, and starting far earlier than the layer-collapse bug.** Dead output units (of 794
total, ResNet-20, mask at init, layer floor active):

| sparsity | sensitivity | magnitude | random(bm) |
|---|---|---|---|
| 0.5 | **18** | 0 | 0 |
| 0.7 | 102 | 0 | 0 |
| 0.8 | 186 | 0 | 10 |
| 0.9 | 357 (45%) | 0 | 41 |
| 0.95 | **454 (57%)** | 0 | 65 |

At sp=0.5 -- minimal pruning -- sensitivity already deletes 18 channels outright; by 0.95,
57% of all output units have every incoming weight pruned. Magnitude never does this below
0.99. **Every E3/E11-pre-fix number, including sp=0.5-0.9, was contaminated**, not just the
sp>=0.95 rows flagged in the layer-level entry above.

**Mechanistically linked to disproportionate early-layer damage** (the second half of the
user's question): at sp=0.9, sensitivity keeps 22.8% of early-layer weights vs magnitude's
45.2%; at sp=0.95, 11.1% vs 31.2% -- roughly 2x more early-layer pruning at every sparsity
tested. Likely cause: sensitivity scores near init are far more concentrated (a handful of
channels dominate the Jacobian) than magnitude scores, so global top-k empties whole
channels before it has exhausted any single layer's budget. This is the same failure mode
SynFlow's design targets, one level finer than the layer-collapse literature usually
discusses.

**Fixed**: `enforce_min_count` (generalised from the layer-only version) applied twice --
layer floor (1%) then unit floor (5%) -- via count allocation, not top-k-then-repair.
Verified: 0 dead layers AND 0 dead units, sensitivity vs magnitude vs random, at every
sparsity from 0.5 to 0.99, with the kept fraction matching exactly across all three
criteria (0.5041 / 0.1073 / 0.0577 / 0.0576). The random baseline was also upgraded to
match unit-level counts (previously only layer counts), which by itself had let it go from
0 to 22 dead units under the old matching -- fixed by sampling within units rather than
within layers when unit ids are available.

**E11 relaunched** with both floors active. This is now the first prune comparison run on
genuinely valid masks at every sparsity tested. Whatever it shows, it will be the first
trustworthy number this project has produced for the companion pruning paper.

**Worth stating in the paper as a methodological contribution in its own right**, independent
of whether sensitivity ultimately wins or loses the comparison: naive top-k pruning by a
gradient-based saliency score collapses structure (layers, then finer, individual output
units) well before the nominal sparsity would suggest, at a much lower threshold (0.5, not
0.9+) than the pruning literature's "high sparsity" framing usually assumes, and a
comparison that does not check for this measures architecture tolerance to collapse, not
criterion quality. Directly relevant to why some zero-shot pruning results in the
literature may be measuring saliency-scale-driven structural collapse rather than a
genuinely good ranking.

### 2026-08-28 — dead_units=2 on MLP: false alarm, counter bug not mask bug

First E11 result reported `dead_units=2` on MLP despite the unit-floor fix having been
verified (on ResNet-20) to give exactly 0. Investigated before trusting either the run or
the fix.

**Cause: the diagnostic counter, not the mask.** `n_dead_units` looped over unit ids
`0..max(unit_ids)`, which includes ids belonging to non-prunable tensors (bias, head) that
own zero prunable weight by construction -- MLP has 1538 unit ids total but only 1536 with
any prunable weight. Those 2 phantom units trivially read "0 kept" and were counted as
"dead" despite never having anything prunable to lose.

**Verified the actual mask is clean**: reconstructed MLP's masks directly with random
scores, filtered dead-unit counting to unit ids that actually occur among prunable entries
(`unit_ids.unique()`), and confirmed 0 dead units at sparsity 0.5/0.9/0.95/0.99 with exact
budget matching (0.5016/0.1029/0.0533/0.0533). Fixed the counter in `prune_and_continue`
the same way. E11 relaunched.

**Lesson for the collapse-detection code generally**: any per-group accounting over a
grouping id must restrict to groups that are actually populated in the domain being
counted, not `0..max(id)+1` -- the same class of off-by-inclusion error as the original
layer/unit collapse bug, just one level removed (in the *measurement* of the fix rather
than the fix itself). Worth a second look at `dead_layers` for the same issue: currently
`sum(1 for n in masks if masks[n].numel()>0 ...)` already filters to tensors with nonzero
numel, which is the layer-level analogue of the same correctness requirement, so that one
was fine.

### 2026-08-28 — Objection tested: is churn just noise from unimportant parameters diluting rho?

User's hypothesis: if many low-sensitivity parameters fluctuate near a noise floor, their
rank churn could mechanically depress the measured stability of the WHOLE ranking, even
though none of them are ever "meaningful" -- i.e. the reported instability might be an
artifact of a universe larger than the useful parameter set, not a property of the
parameters that matter. Tested directly on saved per-checkpoint score trajectories
(ResNet-20, 14 checkpoints; ViT, 21 checkpoints -- no new training needed), sp=0.9,
mid-training checkpoint vs. final.

**(1) Estimator noise is flat across sensitivity deciles**, not concentrated in the low end:
relative fold-disagreement 0.05-0.06 (ResNet) and 0.04-0.07 (ViT) from the lowest decile to
the highest. Rules out the most literal version of the hypothesis (noise scale itself is
not what's different about small-S parameters).

**(2) Weights entering the top-k are not mostly near-miss boundary cases.** Only 16-18%
were already in the top 15% at the earlier checkpoint; 26-29% were BELOW MEDIAN at that
checkpoint and later rose into the top 10%. Weights leaving the top-k land solidly below
the cut at the final checkpoint (median percentile 0.59-0.78 against a 0.90 cut), not just
barely below it. Churn looks like real reordering, not boundary jitter among near-ties.

**(3) The decisive test: restrict to an "ever-plausible" candidate pool and remeasure.**
Spearman rho computed WITHIN the union of each checkpoint's own top-5%/10%/20%/30%/50% (a
population that necessarily excludes any parameter that never looked important):

| candidate pool | ResNet-20 rho | ViT rho |
|---|---|---|
| full parameter set | 0.295 | 0.366 |
| ever-top-50% | 0.210 | 0.320 |
| ever-top-30% | 0.158 | 0.299 |
| ever-top-10% | 0.232 | 0.235 |
| ever-top-5%  | 0.348 | 0.241 |

At every threshold, both architectures: restricting to the plausible-candidate set makes
rho the same or WORSE than the full-parameter measurement, never better. **Mechanism is the
opposite of the hypothesis**: bulk rho is inflated by a large population of parameters that
are stably, boringly unimportant throughout training -- they cost nothing to rank correctly
and prop up the full-population correlation for free. The contested set (parameters that
are ever plausibly important) churns as much or more than the population as a whole.

Note: the "top-k overlap restricted to candidates" comparison is a TAUTOLOGY (the top-k set
is always a subset of any generous candidate pool by construction) and was checked and
discarded as uninformative before writing this up -- only the Spearman-within-candidates
comparison is a real test.

**Conclusion for the paper: the measured instability is not an artifact of dilution by a
large low-importance tail. It is concentrated among the parameters a pruner would actually
be choosing between.** This strengthens rather than weakens the central claim. Caveat:
n=2 runs, one mid-training snapshot each, single sparsity (0.9) -- the direction is
consistent across both runs and every threshold tested, which is reassuring, but a fuller
version (multiple checkpoint pairs, multiple sparsities, more architectures) would be
worth doing with more time.

### 2026-08-28 — sp=0.5 prune-accuracy comparison: not yet run, now queued

User asked how sensitivity/random/magnitude compare at 50% sparsity. Checked: neither the
retracted E3 nor the running E11 tested sp=0.5 for accuracy -- both went straight to 0.9/
0.99 (ViT: 0.9 only). The only sp=0.5 data that existed was the RANK-STABILITY set-overlap
sweep from the "top x%" question (a different measurement -- does the set stay the same,
not does pruning it hurt) and the mask-validity diagnostic from the collapse-detection work
(dead-unit counts, not accuracy).

That diagnostic is still relevant context while E11b runs: at sp=0.5, UNCONSTRAINED (no
floor) masks give sensitivity 18/794 dead units on ResNet-20 vs 0 for magnitude and 0 for
random. So even at this mild a sparsity, sensitivity is already disconnecting structure
that the other two criteria leave untouched -- a mechanistic head start for magnitude/random
independent of any accuracy result.

Added `--sparsities` override to `experiments/e11_prune_probe.py` and queued a dedicated
sp=0.5 pass (E11b) behind the running E11 (sp 0.9/0.99), so it starts automatically without
contending for the GPU. Both floors (layer 1%, unit 5%) active as before.

### 2026-08-28 — "Do early sensitivities resemble trained ones?" Three pre-specified tests

User asked to demonstrate, by experiment, that early parameterwise sensitivities resemble a
fully trained network's. Pre-specified three tests before looking, and report all three
(not only the favourable one).

**TEST 1 -- extreme tail: NEGATIVE, after removing a severe confound.**
Initial result looked strongly positive: ViT top-0.5% set overlaps its final self by 0.613
*at initialisation*. This was almost entirely artifactual. The saved score vectors cover ALL
parameters, and **86.8% of the ViT's top-0.5% by S_final are NON-PRUNABLE** (LayerNorm,
biases, pos_embed, cls_token, head) despite being only 1.5% of parameters. Norm/bias
sensitivities are structurally huge and permanently top-ranked, so the "stability" was
architecture, not learning -- and irrelevant to pruning, which never touches them.
Restricted to prunable weights:

| top-0.5% overlap at init | all params | prunable only |
|---|---|---|
| ViT | 0.613 | **0.239** |
| ResNet-20 | 0.181 | **0.040** |

Would have been an embarrassing error in the paper. Any future top-k analysis on the saved
vectors MUST restrict to prunable first.

**TEST 3 -- distributional resemblance: NEGATIVE.** The spectrum's shape changes materially
during training. ViT Hill alpha 1.23 -> 0.67, log10 IQR 0.90 -> 1.67 (spread nearly
doubles). ResNet alpha 1.83 -> 1.54, IQR 1.37 -> 0.84. "The statistics are fixed early even
if identities change" is not supported.

**TEST 2 -- comparative predictive power: POSITIVE, and the real result.**
Predictors of S_T evaluated within-layer on prunable params (layer-scale confound stripped):

| predictor | ResNet-20 | ViT |
|---|---|---|
| S_0 (sensitivity at init) | **+0.0437** | **+0.1635** |
| abs(theta_0) (magnitude at init) | +0.0007 | +0.0041 |
| random | -0.0002 | -0.0002 |

S_0 carries genuine information about the trained network that magnitude does not --
40-60x more, with the free architectural structure removed. This is a real and defensible
version of the user's claim.

### The methodological gap this exposed, and E12

Whether rho=0.164 counts as "resembles" depends entirely on a ceiling nobody had measured.
Every stability number in this project has implicitly targeted 1.0, but S_T is the endpoint
of a stochastic trajectory: some of it is irreducibly run-specific (batch order), and NO
early predictor can access that. Measuring against 1.0 systematically understates early
knowledge.

**E12 launched**: pairs of runs, same initialisation, `data_seed` differing only, MLP and
ResNet-20, 12000 steps, keep_scores=all. Gives rho(S_T^A, S_T^B) -- the achievable ceiling.
The normalised quantity rho(S_0, S_T)/rho(S_T^A, S_T^B) is the fraction of *knowable*
structure present at init. Same-init/different-data-order is the only well-posed form:
different inits are related by a hidden unit permutation, making parameterwise correlation
meaningless.

Pre-specified interpretation (recorded before results):
- ceiling near 1.0 -> S_T highly reproducible; low rho(S_0,S_T) genuinely means early
  sensitivity does not resemble the trained network.
- ceiling comparable to rho(S_0,S_T) -> most of what is knowable IS known at init, and the
  earlier "does not settle" findings were substantially measuring irreducible run-to-run
  noise rather than a failure of early prediction.

`analysis/ceiling.py` written to compute this.

### 2026-08-28 — Baseline mismatch caught and fixed; prune-timing gap being filled

User asked three sharp questions: how does magnitude exceed dense, when are these models
actually pruned, and could timing/training-length be the explanation. All three exposed
real gaps.

**Baseline mismatch, confirmed and fixed.** The "0.585 dense baseline" quoted against E11's
MLP numbers was from a DIFFERENT experiment (E7: 20000 steps, constant LR) with a
mismatched step count and schedule -- not a control matched to E11's actual budget (8000
steps, cosine LR, same seed). Built `experiments/e11_dense_control.py` to train the
correct control. Result: **0.592**, not 0.585. Corrected table:

| sparsity | sensitivity | random | magnitude | dense (correct) |
|---|---|---|---|---|
| 0.9 | 0.523 | 0.585 | 0.593 | 0.592 |
| 0.99 | 0.288 | 0.560 | 0.561 | 0.592 |

Magnitude at sp=0.9 is now statistically indistinguishable from dense (0.593 vs 0.592) --
the earlier "exceeds baseline" read was entirely the mismatch, not a real effect. Sensitivity
remains the clear outlier at both sparsities: 6.9pp below dense at sp=0.9, 30.4pp at sp=0.99.

**Prune timing: confirmed E11 tests ONLY prune@0 (initialisation).** Checked the code
directly (`prune_and_continue(cfg, 0, ...)`, hardcoded). No valid late-pruning comparison
existed anywhere -- E3's step-9/400/8000 arms are retracted (degenerate masks). This is a
real, not cosmetic, gap: "sensitivity loses" has so far only been shown for the single
hardest case (prune from a random, untrained network).

**E13 launched**: prune-time sweep for MLP, prune points at 1% (step 80) and 5% (step 400)
of the 8000-step budget, sp=0.9, all three criteria, SAME fixed 8000-step fine-tune budget
at every prune time (isolates "how good is the mask taken at step X" from "how much total
compute did this arm get" -- a later prune point does not get more total training).
prune@0 numbers already exist from E11 as the free t=0 comparison point.

### 2026-08-28 — MLP reproducibility ceiling: rho=0.29, S_0 captures ~36% of it

E12 MLP pair complete (data_seed 0 vs 1, same init seed=0, 12000 steps cosine LR).

**Ceiling is low.** rho(S_T^A, S_T^B) = 0.288 within-layer, 0.300 global. Two runs from an
identical initialisation, trained to convergence, differing ONLY in batch order, agree with
each other at only ~0.29. Most of what S_T looks like is not a deterministic function of
architecture+init -- it is substantially decided by data order. This is a first-class
finding independent of early-prediction: the target a pruning mask aims at is not very
reproducible to begin with.

**S_0 as a fraction of the ceiling:**

| step | %T | raw within-layer rho | frac of ceiling |
|---|---|---|---|
| 0 | 0% | 0.105 | 36.4% |
| 7 | 0% | 0.101 | 35.1% |
| 15 | 0% | 0.058 | 20.2% |
| 29 | 0% | 0.003 | 1.1% |
| 56 | 0% | 0.141 | 48.8% |
| 110 | 1% | 0.308 | 107% (see caveat below) |

At init, before any training, S_0 captures ~36% of everything reproducible about the
trained network -- a genuine, defensible, quantitative version of "early sensitivity
resembles the trained network."

**Two things flagged rather than smoothed over.**
1. Non-monotonic wobble at steps 15-56 (36%->20%->1%->49%). Could be a real early
   transient or single-seed-pair noise. Needs more seeds/checkpoints before trusting the
   shape; do not cite the step-29 near-zero point as meaningful without replication.
2. **Methodological caveat, important for the paper**: once frac_of_ceiling exceeds 1.0
   (~1% of training here), the comparison is no longer apples-to-apples. raw_wl at that
   point measures INTRA-run correlation (checkpoint vs its own eventual endpoint), which is
   structurally easier than the INTER-run ceiling (two independent endpoints) -- a run
   trivially correlates more with where it personally ends up than with a sibling run's
   independent endpoint, once trajectory-specific structure starts accumulating. "Exceeds
   the ceiling" past this point is NOT beating a maximum; the two curves answer different
   questions from there on. Only the pre-crossing region (roughly the first 1% of training
   here) is a fair test of what early sensitivity knows about what is knowable.

Same top-10%-set version: raw=0.1575 vs ceiling=0.3963 at step 0 -> 39.8% of ceiling,
consistent with the within-layer rho reading.

ResNet-20 pair (3/4 of E12) in progress; checking whether ~36% at init replicates.

### 2026-08-28 — Accuracy ceiling: MLP dropped as the primary architecture for the grid

User flagged (correctly) that the prune-comparison grid needs a model reaching 80-90%+
accuracy to be meaningful, and MLP's dense control landed at only 0.592.

**This is architectural, not a training-length artifact.** A plain MLP has a hard accuracy
ceiling on CIFAR-10 around 55-60% -- no convolutional or attention inductive bias for
images. Corroborating evidence already in hand: the longest MLP run all session (E7,
20000 steps ~= 51 epochs, constant LR) topped out at 58.5% with train loss still ~1.1-1.2.
More training does not fix this.

**ResNet-20 already clears the target**: 86.6% in the E7 long run. Switched the grid
priority to ResNet-20 as the primary/hero architecture. MLP's partial grid data (5 cells,
seeded from E11/E13/E11b) is retained as a secondary/appendix data point -- useful for the
sparsity-dependence and prune-time-dependence SHAPE of the result, just not as evidence
about a "properly trained" model, given its low accuracy ceiling.

**Grid reduced from 4x4x3=48 to 3x3x3=27 cells for time budget.** ResNet-20's per-cell
cost (~19 min, ft=16000 steps) makes the full 48-cell grid an ~15h commitment -- too close
to the Aug 29 deadline to risk. Reduced to prune fractions {0%, 25%, 50%} x sparsities
{20%, 50%, 90%} x 3 methods, keeping both axes' endpoints and midpoint so the grid's shape
is still legible. Estimated ~8.4h, launched now to run overnight.

**MLP grid stopped** (was mid-run at 43 cells outstanding) once the accuracy ceiling was
confirmed -- would have produced 48 numbers with no upper accuracy value >60%, not useful
for the "compares to a well-trained dense model" framing the user wants.

### 2026-08-28 — Positive-evidence sweep: four new tests, targeted at finding a genuine signal

User asked explicitly for experiments most likely to surface positive evidence for "early
sensitivity predicts trained-network importance" (not just re-running the same negative
framing). Four tests, all free (existing saved trajectories, no new training).

**TEST 1 -- calibration (S_0 decile -> mean S_T): POSITIVE, strong for ResNet, weaker for ViT.**
Binning parameters by S_0 decile and taking the mean S_T per bin: ResNet-20 gives a
PERFECT monotonic staircase (spearman-of-means=1.000), decile 9 having ~12x the mean final
sensitivity of decile 0. ViT is directionally right at the extremes but noisier in the
middle (spearman-of-means=0.564, not monotonic). This matters because rank correlation
demands exact ordering, which pruning does not need -- "is this parameter in the important
BAND" is the operative question, and that survives even where exact rank churns.

**TEST 2 -- AUROC of S_0 for top-k(S_T) membership: POSITIVE, clean, and improves with
sparsity.**

| sparsity | ResNet-20 AUROC | ViT AUROC |
|---|---|---|
| 0.5 | 0.507 | 0.549 |
| 0.9 | 0.649 | 0.660 |
| 0.99 | **0.797** | **0.820** |

0.5 = random, 1.0 = perfect separation. At the sparsities pruning actually uses, S_0 alone
discriminates "will end up in the top-1%" from "won't" with AUROC ~0.8 -- a strong, clean,
easily-communicated result, and the RIGHT framing for the practical question (binary
keep/prune, not exact rank).

**TEST 3 -- naive windowed averaging (first 5 checkpoints, steps 0-9): NULL.** No
improvement over single-snapshot S_0 (rho 0.044->0.038 ResNet, 0.157->0.158 ViT). Reported
honestly rather than omitted.

**TEST 4 -- wider windows, WITH a confound caught and separated out.** Extending the window
further (n=13, up to step 3783 = 47% of training for ResNet) gives a huge rho jump
(0.044->0.451) -- but this is NOT early prediction, it's an average that has started
including LATE checkpoints that are trivially close to S_final. Restricting to genuinely
early windows (<2% of training): ResNet rho 0.044->0.089 (~2x) by step 89 (1.1%), ViT
0.157->0.181 (~15%) by step 42 (1%). Modest but real, honestly bounded.

**Action taken**: AUROC and calibration added as first-class, reusable metrics
(`fsd/rank_metrics.py`) rather than one-off scripts, wired into `fsd/run.py`'s per-checkpoint
analysis so every future/cluster run reports them automatically. New experiment
`experiments/e15_early_window.py` for the properly-bounded early-window-averaging test
(previous scripts share saved-trajectory data with this but nothing tested a
*deliberately* bounded early window before).
