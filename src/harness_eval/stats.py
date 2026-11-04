"""Aggregates with task-level bootstrap CIs (the HarnessTax method).

Average each task's attempts first, then average across tasks; resample tasks
with replacement 10,000 times for the 95% interval.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .results import Trial

N_BOOT = 10_000
SEED = 20260906


@dataclass
class Estimate:
    mean: float
    lo: float
    hi: float
    n: int

    def fmt(self, kind: str = "pct") -> str:
        if kind == "pct":
            return f"{self.mean:.1%} [{self.lo:.1%}, {self.hi:.1%}]"
        if kind == "usd":
            return f"${self.mean:.3f} [${self.lo:.3f}, ${self.hi:.3f}]"
        if kind == "int":
            return f"{self.mean:,.0f} [{self.lo:,.0f}, {self.hi:,.0f}]"
        return f"{self.mean:.3g} [{self.lo:.3g}, {self.hi:.3g}]"


def task_means(trials: list[Trial], value) -> dict[str, float]:
    """value(trial) -> float | None; tasks with no valid value are dropped."""
    groups: dict[str, list[float]] = defaultdict(list)
    for t in trials:
        v = value(t)
        if v is not None:
            groups[t.task_name].append(float(v))
    return {k: float(np.mean(v)) for k, v in groups.items() if v}


def bootstrap(values: list[float], seed: int = SEED) -> Estimate:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(N_BOOT, arr.size))
    boots = arr[idx].mean(axis=1)
    return Estimate(float(arr.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), int(arr.size))


def estimate(trials: list[Trial], value) -> Estimate:
    return bootstrap(list(task_means(trials, value).values()))


def success(t: Trial) -> float:
    return 1.0 if t.passed else 0.0


def summarize(trials: list[Trial]) -> dict[str, Estimate | float | int]:
    """Headline numbers for one experiment."""
    return {
        "success": estimate(trials, success),
        "cost_usd": estimate(trials, lambda t: t.cost_usd),
        "tokens": estimate(trials, lambda t: t.tokens if t.n_llm_calls else None),
        "llm_calls": estimate(trials, lambda t: t.n_llm_calls if t.n_llm_calls else None),
        "first_call_prompt_tokens": estimate(trials, lambda t: t.first_call_prompt_tokens),
        "agent_minutes": estimate(trials, lambda t: None if t.agent_seconds is None else t.agent_seconds / 60),
        "n_trials": len(trials),
        "n_tasks": len({t.task_name for t in trials}),
        # Harbor still grades a trial whose agent crashed, so count exceptions, not just missing rewards.
        "n_errors": sum(1 for t in trials if t.status == "error" or t.exception_type),
        "n_turn_cap_hits": sum(1 for t in trials if t.turn_cap_hit),
        "n_unpriced_trials": sum(1 for t in trials if t.unpriced_calls),
    }


def paired(a: list[Trial], b: list[Trial], value, seed: int = SEED) -> dict[str, float | int]:
    """Difference b - a over tasks both runs share, with a paired bootstrap CI."""
    ma, mb = task_means(a, value), task_means(b, value)
    common = sorted(set(ma) & set(mb))
    if not common:
        return {"n": 0}
    xa = np.array([ma[k] for k in common])
    xb = np.array([mb[k] for k in common])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(common), size=(N_BOOT, len(common)))
    diffs = (xb[idx] - xa[idx]).mean(axis=1)
    out = {
        "n": len(common),
        "a": float(xa.mean()),
        "b": float(xb.mean()),
        "diff": float((xb - xa).mean()),
        "lo": float(np.percentile(diffs, 2.5)),
        "hi": float(np.percentile(diffs, 97.5)),
    }
    if (xa > 0).all() and (xb > 0).all():
        out["geo_mean_ratio"] = float(np.exp(np.mean(np.log(xb / xa))))
    return out
