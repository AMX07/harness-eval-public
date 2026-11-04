"""Read Harbor job directories into one flat record per trial.

Harness-agnostic: everything comes from Harbor's own files (result.json, the
ATIF trajectory at agent/trajectory.json, verifier output). Any harness whose
Harbor adapter writes an ATIF trajectory gets token, cost and step accounting.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .pricing import Usage, call_cost, normalize_model


@dataclass
class LlmCall:
    step_id: int
    model: str | None
    usage: Usage
    cost_usd: float | None
    timestamp: str | None = None


@dataclass
class Trial:
    job_name: str
    trial_name: str
    task_name: str
    attempt: int
    harness: str
    harness_version: str | None
    model: str | None
    reward: float | None
    status: str  # passed | failed | error
    exception_type: str | None
    exception_message: str | None
    started_at: str | None
    finished_at: str | None
    agent_seconds: float | None
    setup_seconds: float | None
    verifier_seconds: float | None
    n_steps: int
    n_llm_calls: int
    n_tool_calls: int
    turn_cap_hit: bool
    usage: Usage
    first_call_prompt_tokens: int | None
    cost_usd: float | None  # frozen price table, from per-call usage
    cost_usd_reported: float | None  # what the harness / Harbor reported
    unpriced_calls: int
    tests_passed: list[str] = field(default_factory=list)
    tests_failed: list[str] = field(default_factory=list)
    trial_dir: str = ""
    trajectory_path: str | None = None
    calls: list[LlmCall] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    @property
    def tokens(self) -> int:
        """Prompt + output tokens, the HarnessTax 'Tokens' metric."""
        return self.usage.prompt + self.usage.output

    def summary(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("calls")
        d["usage"] = asdict(self.usage) | {"prompt": self.usage.prompt}
        d["tokens"] = self.tokens
        return d


def _seconds(timing: dict | None) -> float | None:
    if not timing or not timing.get("started_at") or not timing.get("finished_at"):
        return None
    start = datetime.fromisoformat(timing["started_at"])
    end = datetime.fromisoformat(timing["finished_at"])
    return (end - start).total_seconds()


def usage_from_metrics(metrics: dict | None) -> Usage | None:
    """ATIF step metrics -> billing split.

    Convention (Harbor's Claude Code converter, also used by our adapters):
    prompt_tokens = uncached input + cache read + cache creation,
    cached_tokens = cache read, extra.cache_creation_input_tokens = cache writes,
    extra.cache_creation.{ephemeral_5m,ephemeral_1h}_input_tokens = TTL split.
    """
    if not metrics:
        return None
    prompt = metrics.get("prompt_tokens")
    output = metrics.get("completion_tokens") or 0
    if prompt is None and not output:
        return None
    extra = metrics.get("extra") or {}
    cache_read = metrics.get("cached_tokens") or extra.get("cache_read_input_tokens") or 0
    creation = extra.get("cache_creation_input_tokens") or 0
    split = extra.get("cache_creation") if isinstance(extra.get("cache_creation"), dict) else {}
    write_1h = split.get("ephemeral_1h_input_tokens") or 0
    write_5m = split.get("ephemeral_5m_input_tokens")
    if write_5m is None:
        write_5m = max(creation - write_1h, 0)
    uncached = max((prompt or 0) - cache_read - creation, 0)
    return Usage(input=uncached, output=output, cache_read=cache_read, cache_write_5m=write_5m, cache_write_1h=write_1h)


def read_trajectory(path: Path, default_model: str | None, pricing: str | None = None) -> dict[str, Any]:
    data = json.loads(path.read_text())
    steps = data.get("steps") or []
    calls: list[LlmCall] = []
    n_tool_calls = 0
    for step in steps:
        n_tool_calls += len(step.get("tool_calls") or [])
        usage = usage_from_metrics(step.get("metrics"))
        if usage is None:
            continue
        model = step.get("model_name") or default_model
        calls.append(LlmCall(step["step_id"], model, usage, call_cost(model, usage, pricing), step.get("timestamp")))
    agent_steps = [s for s in steps if s.get("source") == "agent"]
    final = data.get("final_metrics") or {}
    return {
        "calls": calls,
        "n_steps": len(agent_steps),
        "n_tool_calls": n_tool_calls,
        "reported_cost": final.get("total_cost_usd"),
        "agent_extra": (data.get("agent") or {}).get("extra") or {},
        "trajectory_extra": data.get("extra") or {},
    }


def read_tests(verifier_dir: Path) -> tuple[list[str], list[str]]:
    ctrf = verifier_dir / "ctrf.json"
    if not ctrf.is_file():
        return [], []
    try:
        tests = json.loads(ctrf.read_text())["results"]["tests"]
    except (KeyError, ValueError, TypeError):
        return [], []
    passed = [t["name"] for t in tests if t.get("status") == "passed"]
    failed = [t["name"] for t in tests if t.get("status") != "passed"]
    return passed, failed


def _turn_cap_hit(trial_dir: Path, traj: dict[str, Any]) -> bool:
    if traj.get("trajectory_extra", {}).get("turn_cap_hit"):
        return True
    for log in (trial_dir / "agent").glob("*.txt"):
        try:
            text = log.read_text(errors="replace")
        except OSError:
            continue
        if '"subtype":"error_max_turns"' in text or '"subtype": "error_max_turns"' in text:
            return True
    return False


def read_trial(trial_dir: Path, pricing: str | None = None) -> Trial | None:
    result_path = trial_dir / "result.json"
    if not result_path.is_file():
        return None
    r = json.loads(result_path.read_text())
    if "trial_name" not in r:
        return None  # job-level result.json
    agent_info = r.get("agent_info") or {}
    model_info = agent_info.get("model_info") or {}
    model = model_info.get("name") or (r.get("config", {}).get("agent", {}) or {}).get("model_name")
    rewards = (r.get("verifier_result") or {}).get("rewards") or {}
    reward = rewards.get("reward")
    exc = r.get("exception_info") or {}
    if reward is not None and float(reward) >= 1.0:
        status = "passed"
    elif reward is not None:
        status = "failed"
    else:
        status = "error"

    traj_path = trial_dir / "agent" / "trajectory.json"
    traj: dict[str, Any] = {"calls": [], "n_steps": 0, "n_tool_calls": 0, "reported_cost": None}
    if traj_path.is_file():
        try:
            traj = read_trajectory(traj_path, model, pricing)
        except (ValueError, KeyError) as e:
            traj["error"] = str(e)
    calls: list[LlmCall] = traj["calls"]
    usage = sum((c.usage for c in calls), Usage())
    unpriced = sum(1 for c in calls if c.cost_usd is None)
    cost = sum(c.cost_usd for c in calls if c.cost_usd is not None) if calls else None
    agent_result = r.get("agent_result") or {}
    passed_tests, failed_tests = read_tests(trial_dir / "verifier")
    return Trial(
        job_name=trial_dir.parent.name,
        trial_name=r["trial_name"],
        task_name=r["task_name"],
        attempt=0,
        harness=agent_info.get("name", "?"),
        harness_version=agent_info.get("version"),
        model=normalize_model(model),
        reward=None if reward is None else float(reward),
        status=status,
        exception_type=exc.get("exception_type"),
        exception_message=(exc.get("exception_message") or "")[:2000] or None,
        started_at=r.get("started_at"),
        finished_at=r.get("finished_at"),
        agent_seconds=_seconds(r.get("agent_execution")),
        setup_seconds=_seconds(r.get("agent_setup")),
        verifier_seconds=_seconds(r.get("verifier")),
        n_steps=traj["n_steps"],
        n_llm_calls=len(calls),
        n_tool_calls=traj["n_tool_calls"],
        turn_cap_hit=_turn_cap_hit(trial_dir, traj),
        usage=usage,
        first_call_prompt_tokens=calls[0].usage.prompt if calls else None,
        cost_usd=cost,
        cost_usd_reported=agent_result.get("cost_usd", traj.get("reported_cost")),
        unpriced_calls=unpriced,
        tests_passed=passed_tests,
        tests_failed=failed_tests,
        trial_dir=str(trial_dir),
        trajectory_path=str(traj_path) if traj_path.is_file() else None,
        calls=calls,
    )


def read_job(job_dir: Path, pricing: str | None = None) -> list[Trial]:
    """All trials of one Harbor job, with attempt numbers assigned per task by start time."""
    trials = [t for d in sorted(job_dir.iterdir()) if d.is_dir() and (t := read_trial(d, pricing))]
    by_task: dict[str, list[Trial]] = defaultdict(list)
    for t in trials:
        by_task[t.task_name].append(t)
    for group in by_task.values():
        group.sort(key=lambda t: (t.started_at or "", t.trial_name))
        for i, t in enumerate(group, start=1):
            t.attempt = i
    return sorted(trials, key=lambda t: (t.task_name, t.attempt))


def job_meta(job_dir: Path) -> dict[str, Any]:
    """The experiment record `hx run` writes next to Harbor's own files."""
    meta = job_dir / "hx-experiment.json"
    return json.loads(meta.read_text()) if meta.is_file() else {}
