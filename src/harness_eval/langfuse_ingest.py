"""Push Harbor results into Langfuse so every task, step and score is browsable.

Layout in Langfuse:
  - Dataset `<benchmark>`: one item per task (instruction + metadata).
  - Experiment run `<job> · a<k>`: one run per attempt index, linking each task
    to the trace of that attempt, so runs can be compared task by task. Langfuse
    v4 derives runs from `langfuse.experiment.*` span attributes (set on every
    span, as the official SDK's run_experiment does).
  - Trace per trial (name = task): model calls as generations with token usage
    and cost, tool calls with their outputs, and the verifier's test results.
    Session = the job, so a job's trials list together with total cost.
  - Scores per trace: reward, cost_usd, tokens, llm_calls, turn_cap_hit.

Traces go through Langfuse's OpenTelemetry endpoint (OTLP/JSON) with explicit
IDs and timestamps, so re-ingesting a job overwrites rather than duplicates.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from .pricing import cost_breakdown
from .results import Trial, job_meta, read_job, usage_from_metrics
from .tasks import TaskInfo, load_tasks

MAX_FIELD = 60_000  # characters kept per input/output field in the UI; full logs stay on disk


def _hex(seed: str, n: int) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()[:n]


def _clip(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_FIELD:
        return value[: MAX_FIELD // 2] + f"\n\n… [{len(value) - MAX_FIELD:,} chars clipped; full text in the job directory] …\n\n" + value[-MAX_FIELD // 2 :]
    return value


def _ns(ts: datetime) -> str:
    return str(int(ts.timestamp() * 1e9))


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _attr(key: str, value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return {"key": key, "value": {"arrayValue": {"values": [{"stringValue": str(v)} for v in value]}}}
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, default=str)
    return {"key": key, "value": {"stringValue": _clip(value)}}


@dataclass
class Span:
    span_id: str
    parent_id: str | None
    name: str
    start: datetime
    end: datetime
    attrs: dict[str, Any]
    error: str | None = None

    def otlp(self, trace_id: str) -> dict[str, Any]:
        attributes = [a for k, v in self.attrs.items() if (a := _attr(k, v))]
        span = {
            "traceId": trace_id,
            "spanId": self.span_id,
            "name": self.name,
            "kind": 1,
            "startTimeUnixNano": _ns(self.start),
            "endTimeUnixNano": _ns(max(self.end, self.start)),
            "attributes": attributes,
            "status": {"code": 2, "message": self.error} if self.error else {},
        }
        if self.parent_id:
            span["parentSpanId"] = self.parent_id
        return span


def _metadata(prefix: str, meta: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}.{k}": json.dumps(v, ensure_ascii=False, default=str) for k, v in meta.items() if v is not None}


def _observation_text(obs: dict[str, Any] | None) -> str | None:
    if not obs:
        return None
    parts = []
    for r in obs.get("results") or []:
        content = r.get("content")
        if isinstance(content, list):
            content = "\n".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
        parts.append(str(content or ""))
    return "\n\n".join(parts) if parts else None


def experiment_id(project_id: str, dataset_id: str, run_name: str) -> str:
    """Langfuse's stable experiment id (must match the server's derivation byte for byte)."""
    payload = json.dumps(["langfuse-experiment-v1", project_id, dataset_id, run_name], separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class ExperimentLink:
    """Ties a trace to a dataset item inside one experiment run (Langfuse v4 reads this from span attributes)."""

    experiment_id: str
    run_name: str
    dataset_id: str
    item_id: str
    metadata: dict[str, Any]
    item_metadata: dict[str, Any]
    description: str

    def attrs(self, root_id: str) -> dict[str, Any]:
        return {
            "langfuse.environment": "sdk-experiment",
            "langfuse.experiment.id": self.experiment_id,
            "langfuse.experiment.name": self.run_name,
            "langfuse.experiment.description": self.description,
            "langfuse.experiment.metadata": json.dumps(self.metadata, default=str),
            "langfuse.experiment.dataset.id": self.dataset_id,
            "langfuse.experiment.item.id": self.item_id,
            "langfuse.experiment.item.metadata": json.dumps(self.item_metadata, default=str),
            "langfuse.experiment.item.root_observation_id": root_id,
        }


def build_spans(trial: Trial, label: str, experiment: str, pricing: str | None, link: ExperimentLink | None = None) -> tuple[str, list[Span]]:
    trace_id = _hex(f"{trial.job_name}/{trial.trial_name}", 32)
    root_id = _hex(f"{trace_id}/root", 16)
    start = _parse_ts(trial.started_at) or datetime.now(timezone.utc)
    end = _parse_ts(trial.finished_at) or start
    traj = json.loads(Path(trial.trajectory_path).read_text()) if trial.trajectory_path else {}
    steps = traj.get("steps") or []
    instruction = next((s.get("message") for s in steps if s.get("source") == "user"), None)

    spans: list[Span] = []
    prev_ts = start
    pending_input: Any = instruction
    call_no = 0
    for i, step in enumerate(steps):
        ts = _parse_ts(step.get("timestamp")) or prev_ts
        if step.get("source") != "agent":
            pending_input = step.get("message")
            prev_ts = ts
            continue
        usage = usage_from_metrics(step.get("metrics"))
        tool_calls = step.get("tool_calls") or []
        output = {"text": step.get("message") or None, "reasoning": step.get("reasoning_content") or None,
                  "tool_calls": [{"name": t.get("function_name"), "arguments": t.get("arguments")} for t in tool_calls] or None}
        output = {k: v for k, v in output.items() if v}
        if usage is not None:
            call_no += 1
            model = step.get("model_name") or trial.model
            costs = cost_breakdown(model, usage, pricing) or {}
            attrs = {
                "langfuse.observation.type": "generation",
                "langfuse.observation.model.name": model,
                "langfuse.observation.input": pending_input,
                "langfuse.observation.output": output,
                "langfuse.observation.usage_details": {
                    "input": usage.input,
                    "output": usage.output,
                    "cache_read_input_tokens": usage.cache_read,
                    "cache_creation_input_tokens": usage.cache_write_5m + usage.cache_write_1h,
                },
                **({"langfuse.observation.cost_details": costs} if costs else {}),
                **_metadata("langfuse.observation.metadata", {"step_id": step.get("step_id"), "prompt_tokens": usage.prompt}),
            }
            spans.append(Span(_hex(f"{trace_id}/gen/{i}", 16), root_id, f"llm #{call_no}", prev_ts, ts, attrs))
        # Tool calls run after the model step; their results come back on the same step.
        results = {r.get("source_call_id"): r for r in ((step.get("observation") or {}).get("results") or [])}
        next_ts = _parse_ts(steps[i + 1].get("timestamp")) if i + 1 < len(steps) else end
        next_ts = next_ts or ts
        for j, call in enumerate(tool_calls):
            res = results.get(call.get("tool_call_id"))
            out = _observation_text({"results": [res]}) if res else None
            spans.append(Span(
                _hex(f"{trace_id}/tool/{i}/{j}", 16), root_id, call.get("function_name") or "tool", ts, next_ts,
                {"langfuse.observation.type": "tool", "langfuse.observation.input": call.get("arguments"), "langfuse.observation.output": out},
            ))
        pending_input = _observation_text(step.get("observation")) if step.get("observation") else None
        prev_ts = ts

    verifier_dir = Path(trial.trial_dir) / "verifier"
    stdout = verifier_dir / "test-stdout.txt"
    spans.append(Span(
        _hex(f"{trace_id}/verifier", 16), root_id, "verifier", end - timedelta(seconds=trial.verifier_seconds or 0), end,
        {
            "langfuse.observation.type": "evaluator",
            "langfuse.observation.input": {"tests": len(trial.tests_passed) + len(trial.tests_failed)},
            "langfuse.observation.output": {
                "reward": trial.reward,
                "failed": trial.tests_failed,
                "passed": trial.tests_passed,
                "stdout_tail": stdout.read_text(errors="replace")[-20_000:] if stdout.is_file() else None,
            },
        },
    ))

    final_text = next((s.get("message") for s in reversed(steps) if s.get("source") == "agent" and s.get("message")), None)
    root_meta = {
        "experiment": experiment,
        "job": trial.job_name,
        "trial": trial.trial_name,
        "task": trial.task_name,
        "attempt": trial.attempt,
        "harness": label,
        "harness_version": trial.harness_version,
        "model": trial.model,
        "status": trial.status,
        "reward": trial.reward,
        "cost_usd": trial.cost_usd,
        "cost_usd_reported": trial.cost_usd_reported,
        "tokens": trial.tokens,
        "llm_calls": trial.n_llm_calls,
        "tool_calls": trial.n_tool_calls,
        "turn_cap_hit": trial.turn_cap_hit,
        "first_call_prompt_tokens": trial.first_call_prompt_tokens,
        "agent_seconds": trial.agent_seconds,
        "exception": trial.exception_type,
        "trial_dir": trial.trial_dir,
    }
    root = Span(
        root_id, None, trial.task_name, start, end,
        {
            "langfuse.observation.type": "agent",
            "langfuse.trace.name": trial.task_name,
            "session.id": trial.job_name,
            "langfuse.trace.tags": [label, trial.model or "?", experiment, trial.status],
            "langfuse.observation.input": instruction,
            "langfuse.observation.output": final_text,
            "langfuse.release": trial.harness_version,
            **_metadata("langfuse.trace.metadata", root_meta),
            **({"langfuse.observation.level": "ERROR", "langfuse.observation.status_message": trial.exception_message} if trial.status == "error" else {}),
        },
        error=trial.exception_type if trial.status == "error" else None,
    )
    all_spans = [root, *spans]
    if link:
        # Langfuse propagates experiment attributes to every span of an item's trace.
        for span in all_spans:
            span.attrs.update(link.attrs(root_id))
    return trace_id, all_spans


class LangfuseClient:
    def __init__(self, host: str | None = None, public_key: str | None = None, secret_key: str | None = None):
        self.host = (host or os.environ.get("LANGFUSE_HOST") or "http://localhost:3000").rstrip("/")
        auth = (public_key or os.environ["LANGFUSE_PUBLIC_KEY"], secret_key or os.environ["LANGFUSE_SECRET_KEY"])
        self.http = httpx.Client(base_url=self.host, auth=auth, timeout=60)

    def _ok(self, r: httpx.Response) -> dict[str, Any]:
        if r.status_code >= 300:
            raise RuntimeError(f"Langfuse {r.request.method} {r.request.url.path} -> {r.status_code}: {r.text[:500]}")
        return r.json() if r.content else {}

    def send_spans(self, trace_id: str, spans: list[Span]) -> None:
        body = {"resourceSpans": [{
            "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "harness-eval"}}]},
            "scopeSpans": [{"scope": {"name": "harness-eval"}, "spans": [s.otlp(trace_id) for s in spans]}],
        }]}
        self._ok(self.http.post("/api/public/otel/v1/traces", json=body))

    def project_id(self) -> str:
        return self._ok(self.http.get("/api/public/projects"))["data"][0]["id"]

    def upsert_dataset(self, name: str, description: str, metadata: dict[str, Any]) -> str:
        """Create or update a dataset; returns its id."""
        return self._ok(self.http.post("/api/public/v2/datasets", json={"name": name, "description": description, "metadata": metadata}))["id"]

    def upsert_item(self, dataset: str, item_id: str, task: TaskInfo) -> None:
        self._ok(self.http.post("/api/public/dataset-items", json={
            "datasetName": dataset,
            "id": item_id,
            "input": {"task": task.name, "instruction": task.instruction},
            "metadata": {"difficulty": task.difficulty, "category": task.category, "tags": task.tags,
                         "agent_timeout_sec": task.agent_timeout_sec, "cpus": task.cpus, "memory": task.memory,
                         "docker_image": task.docker_image},
        }))

    def score(self, trace_id: str, name: str, value: float, comment: str | None = None) -> None:
        self._ok(self.http.post("/api/public/scores", json={
            "id": f"{trace_id}-{name}", "traceId": trace_id, "name": name, "value": float(value),
            "dataType": "NUMERIC", **({"comment": comment} if comment else {}),
        }))


def ingest_job(job_dir: Path, force: bool = False, client: LangfuseClient | None = None) -> dict[str, Any]:
    meta = job_meta(job_dir)
    exp = meta.get("experiment") or {}
    experiment = exp.get("name") or job_dir.name.split("__")[0]
    label = meta.get("harness_label") or "?"
    dataset = exp.get("dataset") or "terminal-bench@2.0"
    pricing = meta.get("pricing_path")
    marker = job_dir / "hx-langfuse.json"
    done: dict[str, Any] = json.loads(marker.read_text()) if marker.is_file() and not force else {"trials": {}}

    client = client or LangfuseClient()
    trials = read_job(job_dir, pricing)
    tasks = load_tasks(dataset)
    dataset_id = client.upsert_dataset(dataset, f"Tasks of {dataset}", {"source": "harbor"})
    project_id = client.project_id()
    sent = 0
    for trial in trials:
        key = trial.trial_name
        if key in done["trials"]:
            continue
        link = None
        if trial.task_name in tasks:
            item_id = f"{dataset}/{trial.task_name}"
            client.upsert_item(dataset, item_id, tasks[trial.task_name])
            run_name = f"{job_dir.name} · a{trial.attempt}"
            link = ExperimentLink(
                experiment_id=experiment_id(project_id, dataset_id, run_name),
                run_name=run_name,
                dataset_id=dataset_id,
                item_id=item_id,
                metadata={"experiment": experiment, "harness": label, "model": trial.model, "attempt": trial.attempt, "job": job_dir.name},
                item_metadata={"task": trial.task_name, "difficulty": tasks[trial.task_name].difficulty, "category": tasks[trial.task_name].category},
                description=f"{label} + {trial.model}: {experiment} (attempt {trial.attempt})",
            )
        trace_id, spans = build_spans(trial, label, experiment, pricing, link)
        client.send_spans(trace_id, spans)
        client.score(trace_id, "reward", trial.reward if trial.reward is not None else 0.0,
                     None if trial.reward is not None else f"no verifier reward: {trial.exception_type}")
        if trial.cost_usd is not None:
            client.score(trace_id, "cost_usd", trial.cost_usd)
        client.score(trace_id, "tokens", trial.tokens)
        client.score(trace_id, "llm_calls", trial.n_llm_calls)
        client.score(trace_id, "turn_cap_hit", 1.0 if trial.turn_cap_hit else 0.0)
        done["trials"][key] = trace_id
        marker.write_text(json.dumps(done, indent=1))
        sent += 1
    return {"trials": len(trials), "sent": sent, "host": client.host, "dataset": dataset}
