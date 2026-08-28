"""Run directory layout.

    results/<run_id>/config.json      exact config, so any run is reproducible from disk
                    /metrics.json     the analysis-ready result (what figures read)
                    /train_log.jsonl  loss/accuracy trace
                    /scores/step_N.npz   raw score vectors (optional; large)
                    /state/step_N.pt     weights for the C6 prune-and-retrain panel

Runs never coordinate: a cluster job array is N invocations writing N directories, and
`analysis/` globs whatever is present.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch


class Run:
    def __init__(self, root: str, run_id: str):
        self.dir = Path(root) / run_id
        (self.dir / "scores").mkdir(parents=True, exist_ok=True)
        (self.dir / "state").mkdir(parents=True, exist_ok=True)
        self.run_id = run_id

    def write_json(self, name: str, payload: Dict[str, Any]) -> None:
        tmp = self.dir / f".{name}.tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=_default)
        os.replace(tmp, self.dir / name)

    def append_jsonl(self, name: str, payload: Dict[str, Any]) -> None:
        with open(self.dir / name, "a") as fh:
            fh.write(json.dumps(payload, default=_default) + "\n")

    def save_scores(self, step: int, flat: torch.Tensor, folds: Iterable[torch.Tensor],
                    ntk: Optional[torch.Tensor]) -> None:
        payload = {"scores": flat.numpy().astype(np.float32)}
        for i, f in enumerate(folds):
            payload[f"fold{i}"] = f.numpy().astype(np.float32)
        if ntk is not None:
            payload["ntk"] = ntk.numpy().astype(np.float32)
        np.savez(self.dir / "scores" / f"step_{step}.npz", **payload)

    def save_state(self, step: int, model) -> None:
        torch.save(model.state_dict(), self.dir / "state" / f"step_{step}.pt")

    def load_state(self, step: int):
        return torch.load(self.dir / "state" / f"step_{step}.pt", map_location="cpu")

    def exists(self, name: str) -> bool:
        return (self.dir / name).exists()


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
