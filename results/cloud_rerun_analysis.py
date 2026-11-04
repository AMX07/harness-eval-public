"""Extra numbers for results/cloud-rerun-haiku55.md (beyond `hx report` / `hx compare`).

    uv run python results/cloud_rerun_analysis.py <cc job dir> <dsh job dir>

- geometric mean of per-task cost ratios (B/A) with a 10,000-resample task bootstrap
- ratio of mean costs, with and without winning-avg-corewars
- infra failures per run: agent timeouts, verifier timeouts, OOM kills (exit code
  137 in the agent logs), turn-cap hits, other exceptions
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from harness_eval import stats
from harness_eval.results import job_meta, read_job

OUTLIER = "winning-avg-corewars"
EXIT_137 = re.compile(r"(exit(ed)?[ _]?(with )?(code|status)?[\"'\s:=]*137\b|returncode[\"'\s:=]*137\b|\b137\b[^\n]{0,40}(killed|oom))", re.I)


def trials(job: Path):
    return read_job(job, job_meta(job).get("pricing_path"))  # same as `hx report`


def oom_trials(ts) -> list[str]:
    hits = []
    for t in ts:
        agent_dir = Path(t.trial_dir) / "agent"
        found = False
        for log in list(agent_dir.rglob("*.txt")) + list(agent_dir.rglob("*.log")) + list(agent_dir.rglob("*.jsonl")) + [agent_dir / "trajectory.json"]:
            if not log.is_file():
                continue
            if EXIT_137.search(log.read_text(errors="replace")):
                found = True
                break
        if found or (t.exception_message and EXIT_137.search(t.exception_message)):
            hits.append(f"{t.task_name}#{t.attempt}{' (passed)' if t.passed else ''}")
    return hits


def infra(ts) -> dict:
    exc = Counter(t.exception_type for t in ts if t.exception_type)
    named = lambda kind: [f"{t.task_name}#{t.attempt}{' (passed)' if t.passed else ''}" for t in ts if t.exception_type == kind]
    return {
        "trials": len(ts),
        "agent_timeouts": named("AgentTimeoutError"),
        "verifier_timeouts": named("VerifierTimeoutError"),
        "oom_137": oom_trials(ts),
        "turn_cap_hits": [f"{t.task_name}#{t.attempt}{' (passed)' if t.passed else ''}" for t in ts if t.turn_cap_hit],
        "exceptions": dict(exc),
        "unscored": [f"{t.task_name}#{t.attempt}" for t in ts if t.reward is None],
    }


def geo_ratio(a: dict[str, float], b: dict[str, float]) -> dict:
    shared = sorted(k for k in a.keys() & b.keys() if a[k] > 0 and b[k] > 0)
    logs = np.log(np.array([b[k] / a[k] for k in shared]))
    rng = np.random.default_rng(stats.SEED)
    idx = rng.integers(0, logs.size, size=(stats.N_BOOT, logs.size))
    boots = np.exp(logs[idx].mean(axis=1))
    return {
        "n_tasks": len(shared),
        "geo_mean": float(np.exp(logs.mean())),
        "lo": float(np.percentile(boots, 2.5)),
        "hi": float(np.percentile(boots, 97.5)),
        "b_cheaper": int(sum(b[k] < a[k] for k in shared)),
    }


def main(a_dir: str, b_dir: str) -> None:
    A, B = trials(Path(a_dir)), trials(Path(b_dir))
    ca = stats.task_means(A, lambda t: t.cost_usd)
    cb = stats.task_means(B, lambda t: t.cost_usd)
    shared = sorted(ca.keys() & cb.keys())
    mean = lambda d, ks: float(np.mean([d[k] for k in ks]))
    no_out = [k for k in shared if k != OUTLIER]
    gap = mean(cb, shared) - mean(ca, shared)
    out_gap = (cb.get(OUTLIER, 0) - ca.get(OUTLIER, 0)) / len(shared) if OUTLIER in shared else None
    out = {
        "geo_mean_cost_ratio_b_over_a": geo_ratio(ca, cb),
        "ratio_of_means": mean(cb, shared) / mean(ca, shared),
        "ratio_of_means_without_outlier": mean(cb, no_out) / mean(ca, no_out),
        "mean_cost_a": mean(ca, shared),
        "mean_cost_b": mean(cb, shared),
        "outlier": {
            "task": OUTLIER,
            "cost_a": ca.get(OUTLIER),
            "cost_b": cb.get(OUTLIER),
            "share_of_mean_gap": (out_gap / gap) if (out_gap is not None and gap) else None,
        },
        "infra_a": infra(A),
        "infra_b": infra(B),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:3])
