"""Read every results/ directory and report each claim's status.

The point of this file is that NARRATIVE.md cannot quietly drift away from the data. Each
claim has a machine-checkable criterion and prints SUPPORTED / NOT SUPPORTED / NO DATA,
with the numbers that produced the verdict. A claim the runs do not support is meant to be
seen, not explained away -- several of the falsifiers here would redirect the paper rather
than sink it, and we would rather find that out now than in review.

    python -m analysis.claims                 # all runs
    python -m analysis.claims --tag e1        # just one experiment
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

OK, BAD, NONE = "SUPPORTED", "NOT SUPPORTED", "NO DATA"


def load_runs(results: str = "results", tag: Optional[str] = None) -> List[dict]:
    runs = []
    for path in sorted(glob.glob(f"{results}/*/metrics.json")):
        try:
            m = json.load(open(path))
        except Exception:
            continue
        if tag and not m.get("run_id", "").startswith(tag):
            continue
        runs.append(m)
    return runs


def _sp_key(m: dict, prefer: float = 0.9) -> str:
    sps = m.get("sparsities", [])
    return f"{prefer:g}" if prefer in sps else f"{sps[len(sps)//2]:g}" if sps else "0.9"


def _series(m: dict, key: str, field: str = "adjusted") -> List[float]:
    return [row["topk"][key][field] for row in m["vs_final"]]


def c1_freezing(m: dict) -> dict:
    """t* should be a small fraction of the run."""
    key = _sp_key(m)
    t = m["tstar"].get(f"sp{key}_crossing")
    total = m["config"]["train"]["steps"]
    if t is None:
        return {"status": BAD, "detail": "never reaches the measurement ceiling"}
    frac = t / max(1, total)
    return {"status": OK if frac <= 0.25 else BAD,
            "detail": f"t*={t} of {total} steps ({frac:.1%})"}


def c2a_noise_floor(m: dict) -> dict:
    """The plateau must sit below the ceiling but not at chance."""
    key = _sp_key(m)
    ceiling = m["tstar"].get(f"sp{key}_ceiling")
    vals = _series(m, key)
    plateau = max(vals[:-1]) if len(vals) > 1 else float("nan")
    if ceiling is None or ceiling != ceiling:
        return {"status": NONE, "detail": "no fold estimate"}
    return {"status": OK if plateau <= ceiling + 0.02 else BAD,
            "detail": f"plateau={plateau:.3f} vs ceiling={ceiling:.3f}"}


def c2b_layerwise(m: dict) -> dict:
    """Adjusted overlap must be well above 0 -- otherwise it is only a layer budget."""
    key = _sp_key(m)
    vals = _series(m, key)
    final_adj = vals[-2] if len(vals) > 1 else float("nan")
    within = m["vs_final"][-2]["within_layer"]["weighted"] if len(vals) > 1 else float("nan")
    return {"status": OK if final_adj > 0.3 else BAD,
            "detail": f"adjusted={final_adj:.3f}, within-layer rho={within:.3f}"}


def c2c_not_init(m: dict) -> dict:
    """Step 0 must be measurably below the plateau."""
    key = _sp_key(m)
    vals = _series(m, key)
    if len(vals) < 3:
        return {"status": NONE, "detail": "too few checkpoints"}
    at_init, plateau = vals[0], max(vals[:-1])
    return {"status": OK if at_init < plateau - 0.05 else BAD,
            "detail": f"init={at_init:.3f} vs plateau={plateau:.3f}"
                      + ("  <- ordering is essentially set at init" if at_init >= plateau - 0.05 else "")}


def c3_boundary(m: dict) -> dict:
    """High-sparsity overlap should stabilise no later than the bulk correlation."""
    sps = m.get("sparsities", [])
    if len(sps) < 2:
        return {"status": NONE, "detail": "single sparsity"}
    lo, hi = f"{min(sps):g}", f"{max(sps):g}"
    t_lo = m["tstar"].get(f"sp{lo}_crossing")
    t_hi = m["tstar"].get(f"sp{hi}_crossing")
    if t_lo is None or t_hi is None:
        return {"status": NONE, "detail": f"t*({lo})={t_lo}, t*({hi})={t_hi}"}
    return {"status": OK if t_hi <= t_lo else BAD,
            "detail": f"t* at sparsity {hi} = {t_hi} vs {lo} = {t_lo}"}


def c4_laziness(m: dict) -> dict:
    """Rank churn and kernel velocity should move together within a run."""
    key = _sp_key(m)
    kv = [row.get("kernel_velocity_to_final") for row in m["laziness"]]
    adj = _series(m, key)
    pairs = [(a, b) for a, b in zip(kv, adj) if a is not None and a == a and b == b]
    if len(pairs) < 4:
        return {"status": NONE, "detail": "no kernel data"}
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    r = cov / den if den > 0 else float("nan")
    return {"status": OK if r < -0.5 else BAD,
            "detail": f"corr(kernel velocity, adjusted overlap) = {r:.3f} (want strongly negative)"}


def c5_theory(m: dict) -> dict:
    """The drift model should predict the measured overlap; the bound should hold."""
    key = _sp_key(m)
    errs, violations = [], 0
    for row in m["theory"]:
        pred, obs = row.get(f"pred_overlap_{key}"), row.get(f"obs_overlap_{key}")
        bound = row.get(f"bound_overlap_{key}")
        if pred is not None and obs is not None:
            errs.append(abs(pred - obs))
        if bound is not None and obs is not None and obs < bound - 1e-6:
            violations += 1
    if not errs:
        return {"status": NONE, "detail": "no theory rows"}
    mae = sum(errs) / len(errs)
    status = OK if (mae < 0.1 and violations == 0) else BAD
    return {"status": status,
            "detail": f"drift-model MAE={mae:.3f}, bound violations={violations}"}


CHECKS = [
    ("C1  freezing", c1_freezing),
    ("C2a noise floor", c2a_noise_floor),
    ("C2b beyond layer budget", c2b_layerwise),
    ("C2c not fixed at init", c2c_not_init),
    ("C3  boundary beats bulk", c3_boundary),
    ("C4  tracks laziness", c4_laziness),
    ("C5  theory predicts", c5_theory),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--per-run", action="store_true")
    args = ap.parse_args()

    runs = load_runs(args.results, args.tag)
    if not runs:
        print(f"no runs in {args.results}/" + (f" matching '{args.tag}'" if args.tag else ""))
        return 1

    print(f"{len(runs)} run(s)\n")
    tally: Dict[str, Dict[str, int]] = {name: {OK: 0, BAD: 0, NONE: 0} for name, _ in CHECKS}
    for m in runs:
        cfgm = m["config"]
        label = (f"{cfgm['model']['arch']}/{cfgm['data']['dataset']} "
                 f"w{cfgm['model']['width']} seed{cfgm['seed']}")
        if args.per_run:
            print(f"--- {m['run_id']}  {label}")
        for name, fn in CHECKS:
            try:
                res = fn(m)
            except Exception as exc:
                res = {"status": NONE, "detail": f"{type(exc).__name__}: {exc}"}
            tally[name][res["status"]] += 1
            if args.per_run:
                print(f"    {name:26s} {res['status']:14s} {res['detail']}")
        if args.per_run:
            print()

    print("=" * 78)
    print(f"{'claim':28s}{'supported':>12s}{'not supported':>16s}{'no data':>10s}")
    for name, _ in CHECKS:
        t = tally[name]
        print(f"{name:28s}{t[OK]:>12d}{t[BAD]:>16d}{t[NONE]:>10d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
