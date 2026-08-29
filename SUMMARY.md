# Session Summary — Functional Sensitivity Dynamics

**Purpose of this file.** A self-contained account of what this session established, got
wrong and corrected, and still needs, for the AXIOM @ NeurIPS 2026 submission (workshop
deadline: 2026-08-29) and its planned pruning follow-up paper. Read this before
`NARRATIVE.md` (the full chronological log with every number) or `Test/neurips_2026.tex`
(the paper draft) — this file is the map; those are the territory.

---

## 1. The question, and how it moved

**Starting point.** The user's own zero-shot pruning draft (`neurips_2026.tex`, companion
work) proposes ranking parameters by functional sensitivity
`S_i(θ) = E_x‖∂F_θ(x)/∂θ_i‖²` — label-free, loss-free, exactly the diagonal of the
parameter-space Gram matrix dual to the empirical NTK — computed **once**, at
initialisation or shortly after. That method's soundness rests on an unstated assumption:
that the ranking `S` induces does not change much once computed.

**This session's task**: build the measurement to test that assumption, and write it up
for AXIOM. The framing went through several real revisions as evidence came in — each one
driven by a genuine finding, not scope creep:

1. *"Does the ranking freeze early?"* → tested via `t*` (threshold-free settling) and a
   pairwise checkpoint-agreement matrix. **Answer: no**, not at the individual-parameter
   level, in any of 6 long runs (MLP/ResNet-20/ViT, both LR schedules, trained well past
   convergence).
2. *User's counter-observation*: their own prior work measured **Spearman ρ over all
   parameters**, which genuinely does settle early (ρ≥0.7 by 13–29% of training in 5/6
   settings). Both are correct — they measure different objects. The reconciliation
   (bulk ρ vs. the top-k *set* a pruner reads) became a central finding.
3. *User's request*: reframe around **"early sensitivity resembles the fully-trained
   network"** as the headline. Tested this directly and rigorously — see §3. The honest
   answer is architecture-dependent: real but modest for MLP, negative-then-recovering for
   ResNet-20, once done correctly.
4. *User's scope-down*: "I only care about two things" — (a) does `S_t` resemble `S_T`
   within a run, (b) does pruning by each criterion at various times/sparsities compare
   well to a dense model. Both are now the paper's two headline tables.
5. *User's accuracy-floor requirement*: the dense reference model must hit 80–90%+, which
   ruled out MLP as the primary architecture (hard ceiling ~59% on CIFAR-10, architectural,
   not a training-length artifact) and made **ResNet-20 (86.6–87.2%)** the primary target.

---

## 2. What is true, stated as plainly as the evidence allows

### 2.1 The ordering does not settle at the parameter level
Six long runs (20,000 steps ≈ 51 epochs, both constant and cosine LR, MLP/ResNet-20/ViT),
all past the point of accuracy improvement. `t* = T` in every case — the pairwise
checkpoint-agreement matrix never saturates. This is **not** a truncation artifact (loss
had flattened) and **not** a schedule artifact (constant-LR runs show it too; cosine
specifically inflates the *late* portion of the curve by up to +0.12, measured directly).

### 2.2 Bulk ρ settles early; the top-k set a pruner reads settles much later
This is the paper's central reconciliation. Across 6 settings, Spearman ρ over all
parameters reaches 0.7 within 13–44% of training. The raw top-k overlap (what a pruning
mask actually shares) reaches the same threshold 1.5–3.5× later, and the gap widens with
sparsity. ρ is dominated by millions of stably-unimportant parameters that cost nothing to
rank correctly; the top-k boundary is where the actual difficulty lives.

### 2.3 Chance correction is not optional
Sensitivity spans orders of magnitude **across layers**, so two independent rankings that
merely agree on layer budgets look falsely stable. Synthetic control: raw top-k overlap
0.83, Spearman ρ 0.98, **adjusted overlap −0.004** (i.e., zero real agreement). Real-data
instances of the same trap: ViT step-0 raw overlap 0.319 vs. chance 0.278; ResNet-20
global AUROC 0.80 collapsing to ~chance (0.52) within-layer (§2.5). **Any statistic
computed on this project's data must be checked against its within-layer / chance-corrected
version before being trusted.** This is now a standing rule, not a one-off caveat.

### 2.4 Naive top-k pruning collapses structure, well below "high sparsity"
Checked directly (ResNet-20, init, unconstrained global top-k): sensitivity disconnects
18/794 output units by sparsity **0.5** (magnitude: 0), rising to 454/794 (57%) by 0.95.
Also prunes ~2× more of early layers than magnitude at matched global sparsity. Fixed with
a count-preserving two-level floor (layer 1%, unit 5%) in `fsd/prune.py`
(`enforce_min_count`), verified to give 0 dead layers/units at every sparsity 0.5–0.99 for
every criterion, budgets matched exactly. **This fix invalidated and required retraction of
the first prune-accuracy comparison (E3)** — a real methodology bug caught before it
reached the paper, not a nuance.

### 2.5 The reproducibility ceiling: what's even knowable
Two runs sharing an initialisation, differing only in batch order, only agree with each
other (within-layer ρ) at:
- **MLP: 0.288.** S₀ captures **36.4%** of that — real, modest, positive.
- **ResNet-20: 0.278.** S₀ captures **−32%** of that at init (worse than useless),
  crossing positive around 1% of training, reaching parity with the ceiling around 7%.

This means most of what `S_T` looks like is not a deterministic function of
architecture+init — it's substantially decided by data order. Measuring early agreement
against a target of 1.0 (as every naive stability number implicitly does) is unfair to any
early predictor; the ceiling is the right denominator.

### 2.6 AUROC/calibration "positive evidence" — corrected
A late-session sweep (user's explicit request: find experiments most likely to show
positive evidence) found AUROC of S₀ for top-k(S_T) membership at 0.65–0.82 — a clean,
strong-looking result. **This did not survive the within-layer control**: ResNet-20's 0.80
collapses to 0.52 (chance) within-layer; ViT's 0.82 shrinks to 0.57. The calibration-decile
test almost certainly has the same problem and is **unverified** (a within-layer version
was not run before the session ended — see §4). This correction is logged in full in
`NARRATIVE.md` (2026-08-29 entry) and must be reflected in the paper; do not cite the raw
0.80/0.82 AUROC numbers anywhere.

### 2.7 Prune-and-finetune accuracy: partial, sparsity-dependent, architecture-priority ResNet-20
With valid (floor-corrected) masks and a **correctly matched dense control** (a real bug:
an earlier "0.585 dense baseline" was from a different experiment entirely — different
step count, different LR schedule — and made magnitude look like it "beat" dense, which was
entirely the mismatch, not a real effect):

| arch | sparsity | prune@0 sensitivity | random (budget-matched) | magnitude | dense |
|---|---|---|---|---|---|
| MLP | 0.5 | 0.595 | 0.600 | — | 0.592 |
| MLP | 0.9 | 0.523 | 0.585 | 0.593 | 0.592 |
| MLP | 0.99 | 0.288 | 0.560 | 0.561 | 0.592 |
| ResNet-20 | 0.2 | 0.869 | 0.864 | — | **0.872** |

Sensitivity's deficit relative to random/magnitude is large at high sparsity and **shrinks
to ~0 at moderate sparsity (0.5)** — not a fixed weakness of the criterion, a
sparsity-dependent one. ResNet-20's grid (the priority architecture, since it's the only
one clearing the 80–90% accuracy requirement) is **only 2/27 cells complete** — the
background job died when the session was interrupted for a usage-limit reset and was never
relaunched. **This is the single most important unfinished piece of evidence** (§4).

### 2.8 A withdrawn result, kept visible on purpose
An early n=3 finding (boundary stability tracks the sensitivity spectrum's Hill tail index,
all three predicted correlation signs correct) looked like a strong mechanism. It did
**not replicate** on n=5 long runs (2 of 3 signs inverted) and is **withdrawn** — recorded
in `NARRATIVE.md` rather than quietly dropped, as a reminder that n=3 "perfect" results
need replication before they're trusted (base-rate: 1/6 by chance).

---

## 3. Bugs found and fixed this session (all in `fsd/` or `experiments/`, all logged with dates in `NARRATIVE.md`)

| # | Bug | Where | Fix |
|---|---|---|---|
| 1 | Batch-averaged gradient squared ≠ `E[g²]` | original `sensitivity_calculation.py` | Per-example gradients via `torch.func.vmap` (`fsd/sensitivity.py`) |
| 2 | Layer collapse: unconstrained top-k empties whole layers | `fsd/prune.py` | `enforce_min_count`, layer floor |
| 3 | Unit collapse: same failure one level finer, onset at sparsity 0.5 | `fsd/prune.py` | Second floor pass at the output-unit level |
| 4 | Random baseline only matched layer counts, not unit counts → up to 22/784 dead units of its own | `fsd/prune.py::budget_matched_random` | Matches at unit granularity when available |
| 5 | `dead_units` counter included non-prunable "phantom" units with zero prunable weight, falsely flagging them dead | `fsd/prune.py` | Restrict counting to `unit_ids.unique()` |
| 6 | Mismatched dense baseline (wrong experiment's number quoted) | analysis, not code | `experiments/e11_dense_control.py` — dense trained under the *exact* budget/schedule/seed being compared against |
| 7 | 86.8% of ViT's "top-0.5% most sensitive" parameters were non-prunable (LayerNorm/bias/embed/head) — architecture, not learning | analysis | Always restrict to `prunable` mask before any top-k analysis on saved scores |
| 8 | Windowed-averaging test showed huge gains that were pure late-checkpoint leakage, not early prediction | analysis | Bounded the window explicitly (<2% of training) before reporting |
| 9 | Global AUROC/calibration inflated by the layer-budget confound (§2.6) | analysis | Within-layer AUROC computed; calibration not yet fixed (see §4) |

---

## 4. Experiments still to do, in priority order

**P0 — blocking the paper's second headline table.**
1. **Finish the ResNet-20 prune-accuracy grid.** Currently 2/27 cells. Relaunch:
   ```bash
   nohup ./venv/bin/python -u -m experiments.e14_grid --archs resnet20 --out results/_probe > results/e14_resnet20.log 2>&1 &
   ```
   Estimated ~8.4h for the reduced 3×3×3 grid (prune@{0,25,50}% × sparsity{0.2,0.5,0.9} ×
   3 criteria). **Use `nohup` and confirm the process survives a terminal/session
   disconnect** — the previous attempt died silently across an interruption with no error
   logged, which is itself worth a fix (e.g. wrap in `setsid` or check via `disown`).
2. **Within-layer AUROC and calibration**, added as first-class metrics
   (`auroc_topk_within_layer` mirroring `within_layer_spearman`; `calibration_deciles`
   needs the same within-layer treatment) and re-run before §2.6's numbers are cited in the
   paper. This is the last unresolved thread in the "does early sensitivity predict
   importance" question.

**P1 — cluster-scale, needed for statistical weight (n=1 → n=3, more coverage).**
3. Fix the cluster script failure the user reported (`bash run_cluster_sweep.sh` on the
   4-GPU node: every `fsd.cli --config` task FAILED). Likely causes to check first, in
   order of probability: (a) the remote repo path is `~/Documents/Sensitivity-Pruning`,
   not this session's `Explainability-Functional-Sensitivity` — check whether it's a stale
   clone missing recent fixes (esp. #1–#5 above) or missing `data/` (CIFAR download) or a
   missing/different `venv`; (b) `CUDA_VISIBLE_DEVICES` pinning interacting badly with
   however that cluster's torch/CUDA is installed; (c) relative-path assumptions in
   `fsd/cli.py` or `experiments/_common.py` breaking under the cluster's actual working
   directory. Start by reading one `results/_cluster_logs/gpu0.log` for the actual
   traceback — none was captured in this session's transcript.
4. Run `jobs/run_cluster_sweep.sh` (TIER=1) and `jobs/run_positive_evidence_sweep.sh` once
   fixed — full 4×4×3 grid per architecture, 3-seed E1 stability runs, E12 ceiling pairs
   for every architecture (currently only MLP and ResNet-20 have ceilings; ViT, GPT, LSTM
   do not).
5. Full component/layer-granularity table for Table 1 in the paper — currently only
   populated for LSTM; the data exists for the other 5 settings via `compare_structured`,
   just needs tabulating (`analysis/tables.py` extension).

**P2 — strengthens specific paper sections, not blocking.**
6. Same-init/different-data-order **mask transfer** test (E6/`analysis/cross_run.py`) —
   does an early mask from run A predict run B's final ordering? Directly the question the
   pruning follow-up paper needs answered; implemented, not yet analysed.
7. ViT and GPT/LSTM reproducibility ceilings (only MLP/ResNet-20 done).
8. A genuine ≥10× parameter-count scale-up (everything so far is ≤2.7M params) — the
   single biggest external-validity gap, flagged in every limitations paragraph so far.

---

## 5. What's already reflected in `Test/neurips_2026.tex`

The paper draft (see file for full detail; `%% CUT-CANDIDATE` and `%% PENDING` comments
mark every open decision) currently covers: the sensitivity definition and NTK-trace
identity (§2, unchanged from the companion draft), the measurement protocol including the
chance-correction control, the fold-based measurement ceiling, and the pairwise-agreement
diagnostic (§3), the ρ-vs-top-k reconciliation as the central result (§4), the
naive-pruning-collapses-structure finding as an independent contribution (§5), the
trajectory-rank-stability bound and its proof (§6, appendix), and a placeholder §7 for the
prune-accuracy consequence. **This session's edits after that draft was last touched**
(the reproducibility ceiling, the corrected dense baselines, the AUROC confound correction,
and the partial ResNet-20 grid) are recorded in `NARRATIVE.md` but were not yet folded into
the `.tex` before this summary was written — see the accompanying tex edits made alongside
this file for what has now been added.

---

## 6. File map

- `NARRATIVE.md` — full chronological findings log, every number, every retraction. The
  source of truth; this file is a compressed derivative of it.
- `Test/neurips_2026.tex` — the paper draft.
- `fsd/` — the library (sensitivity estimator, rank metrics, pruning with structural
  floors, the training/measurement loop).
- `experiments/e1`–`e15` — one script per experiment; `e14_grid.py` is the prune-accuracy
  grid, `e12_ceiling.py` the reproducibility ceiling, `e11_dense_control.py` the matched
  dense baseline.
- `analysis/` — report generators (`claims.py`, `grid_report.py`, `ceiling.py`,
  `early_window.py`, `mechanism.py`, `tables.py`, `figures.py`, `pairwise.py`).
- `jobs/run_cluster_sweep.sh`, `jobs/run_positive_evidence_sweep.sh` — 4-GPU sweep scripts,
  `bash <script>.sh` invocation, currently broken on the user's cluster (P1.3 above).
- `results/_probe/`, `results/_probe_cluster/` — grid outputs (JSON), read by
  `analysis/grid_report.py`.
