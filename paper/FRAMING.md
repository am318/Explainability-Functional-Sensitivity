# Two framings, one measurement

E7 decides which paper this is. Both are written against the *same* runs, figures and
theory — only the emphasis and the abstract change. Drafting both now means the deadline
costs nothing whichever way the result goes.

The measurement is not in question either way: ceilings of 0.95–0.99, step-0 adjusted
overlap of 0.02–0.04, and a chance correction that provably separates real agreement from
layer-budget agreement.

---

## Framing A — "The ordering freezes early" (if E7 shows a plateau at t* << T)

**Claim.** Beyond `t*`, further training perturbs the ordering less than resampling the
estimation data does. `t*` is a small fraction of training and moves predictably with
width, depth, LR and batch size.

**Then the paper is:** a predictive principle. `t*` tells a practitioner when the
information a pruning criterion needs is already available, and the drift/gap theory says
why the top-k boundary is the part that stabilises first.

**Leans on:** E1 (curve), E2 (t* scaling), E4 (laziness intervention), E3 (prune-at-t*).

---

## Framing B — "It does not freeze, and the literature's metrics hide that"
(if E7 shows monotonic rise even past convergence, at constant LR)

**Claim.** The sensitivity ordering tracks training essentially all the way to
convergence. Reports of early stability rest on metrics that are dominated by cross-layer
scale separation: on a control whose two rankings share *nothing* within any layer, raw
top-k overlap is 0.83 and Spearman ρ is 0.98, while chance-corrected agreement is −0.004.

**Then the paper is:** a correction plus a mechanism. It says what *does* settle early (the
per-layer budget, in some architectures), what does not (within-layer placement), gives the
drift/gap model that predicts the observed churn rate, and draws the consequence for
pruning: early *allocation* is licensed, early *masks* are not.

**Leans on:** E1 + E7 (the curve and its controls), the synthetic control, the theory
section unchanged, E6 (does an early ordering predict a *different* run's final ordering?),
E3 (accuracy cost of pruning too early).

**Why this is still an AXIOM paper.** The call asks for predictive principles for efficient
AI. A well-measured negative — with a mechanism, a bound, and a decomposition saying which
component *is* usable early — is a predictive principle. It also directly explains why
prune-at-init methods recover layer budgets and little else.

---

## What does not change

- Proposition 1 and both corollaries. The bound holds regardless of whether Δ is small;
  under Framing B it is the quantity explaining why churn is as large as it is.
- The measurement protocol section (chance correction, fold ceiling, threshold-free `t*`).
  Under Framing B this becomes the primary methodological contribution.
- Every figure. Fig. 1 shows the curve and its controls; only the caption's verdict changes.
- The follow-up paper. Under A it inherits a `t*` to prune at; under B it inherits a sharp
  negative result about *what* can be decided early, which is more useful than a wrong
  positive.
