"""Convert DeepSeek Harness session events into an ATIF trajectory.

Event names come from dsh's core session format (packages/core/session):
  step/start          one model call plus the tools it requested
  assistant/message   the model's reply; carries `usage` (TokenUsage, disjoint counts)
  tool/call           name + raw JSON arguments, paired by callId
  tool/result         the tool's model-facing result
Any other event that carries a TokenUsage (e.g. compaction summaries) becomes an
extra agent step so its cost is still counted.

Metrics follow the convention results.py prices: prompt_tokens = uncached input
+ cache read + cache write, cached_tokens = cache read, and
extra.cache_creation_input_tokens = cache writes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)


def _type(event: dict[str, Any]) -> str | None:
    for key in ("type", "kind", "name"):
        if isinstance(event.get(key), str):
            return event[key]
    return None


def _data(event: dict[str, Any]) -> dict[str, Any]:
    for key in ("data", "payload", "body"):
        if isinstance(event.get(key), dict):
            return event[key]
    return event


def _iso(t: float | None) -> str | None:
    return None if t is None else datetime.fromtimestamp(t, tz=timezone.utc).isoformat()


def _text(content: Any, kinds: tuple[str, ...] = ("text",)) -> str:
    if isinstance(content, str):
        return content if "text" in kinds else ""
    if isinstance(content, dict):
        content = content.get("content", [])
    parts = []
    for block in content or []:
        if isinstance(block, dict) and block.get("type") in kinds:
            parts.append(str(block.get("text") or block.get("thinking") or block.get("reasoning") or ""))
    return "\n".join(p for p in parts if p)


def _is_usage(value: Any) -> bool:
    return isinstance(value, dict) and "inputTokens" in value and "outputTokens" in value


def _metrics(usage: dict[str, Any], long_cache: bool) -> Metrics:
    uncached = int(usage.get("inputTokens") or 0)
    read = int(usage.get("cacheReadTokens") or 0)
    write = int(usage.get("cacheWriteTokens") or 0)
    ttl = "ephemeral_1h_input_tokens" if long_cache else "ephemeral_5m_input_tokens"
    return Metrics(
        prompt_tokens=uncached + read + write,
        completion_tokens=int(usage.get("outputTokens") or 0),
        cached_tokens=read,
        extra={
            "cache_creation_input_tokens": write,
            "cache_read_input_tokens": read,
            "cache_creation": {ttl: write},
            **({"reasoning_tokens": usage["reasoningTokens"]} if usage.get("reasoningTokens") else {}),
            "dsh_usage": usage,
        },
    )


def read_events(path: Path) -> list[tuple[float | None, dict[str, Any]]]:
    """(timestamp, session event) pairs from the runner's dsh-events.jsonl."""
    out = []
    if not path.is_file():
        return out
    for line in path.read_text(errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("method") != "session.event":
            continue
        event = (rec.get("payload") or {}).get("event")
        if isinstance(event, dict):
            out.append((rec.get("t"), event))
    return out


def convert(logs_dir: Path, instruction: str | None, model: str | None, version: str | None) -> Trajectory | None:
    events = read_events(logs_dir / "dsh-events.jsonl")
    summary_path = logs_dir / "dsh-summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    if not events and not summary:
        return None
    long_cache = summary.get("cache_retention") == "long"

    steps: list[Step] = []
    first_t = events[0][0] if events else summary.get("started_at")
    steps.append(Step(step_id=1, timestamp=_iso(first_t), source="user", message=instruction or ""))
    by_key: dict[tuple[Any, Any], Step] = {}
    calls: dict[str, ToolCall] = {}

    def agent_step(key: tuple[Any, Any], t: float | None) -> Step:
        if key not in by_key:
            step = Step(step_id=len(steps) + 1, timestamp=_iso(t), source="agent", model_name=model, message="", llm_call_count=1)
            steps.append(step)
            by_key[key] = step
        return by_key[key]

    for t, event in events:
        kind, data = _type(event), _data(event)
        key = (data.get("turn"), data.get("step"))
        if kind == "assistant/message":
            step = agent_step(key, t)
            message = data.get("message") or {}
            step.message = _text(message) or step.message
            reasoning = _text(message, ("reasoning", "thinking"))
            if reasoning:
                step.reasoning_content = reasoning
            if _is_usage(data.get("usage")):
                step.metrics = _metrics(data["usage"], long_cache)
            step.timestamp = _iso(t) or step.timestamp
        elif kind == "tool/call":
            step = agent_step(key, t)
            try:
                args = json.loads(data.get("arguments") or "{}")
            except ValueError:
                args = {"raw": data.get("arguments")}
            call = ToolCall(tool_call_id=str(data.get("callId")), function_name=str(data.get("name")), arguments=args if isinstance(args, dict) else {"value": args})
            calls[call.tool_call_id] = call
            step.tool_calls = [*(step.tool_calls or []), call]
        elif kind == "tool/result":
            step = agent_step(key, t)
            message = data.get("message") or {}
            call_id = message.get("callId") or message.get("toolCallId") or data.get("callId")
            result = ObservationResult(source_call_id=None if call_id is None else str(call_id), content=_text(message) or json.dumps(message)[:20000])
            obs = step.observation or Observation(results=[])
            obs.results.append(result)
            step.observation = obs
        elif kind not in ("step/start", "step/end", "turn/end", "user/message", "system/message") and _is_usage(data.get("usage")):
            # A model call outside the main loop (compaction, titles): keep its cost.
            steps.append(Step(
                step_id=len(steps) + 1, timestamp=_iso(t), source="agent", model_name=model,
                message=f"[auxiliary model call: {kind}]", llm_call_count=1, metrics=_metrics(data["usage"], long_cache),
            ))

    metered = [s.metrics for s in steps if s.metrics]
    final = FinalMetrics(
        total_prompt_tokens=sum(m.prompt_tokens or 0 for m in metered),
        total_completion_tokens=sum(m.completion_tokens or 0 for m in metered),
        total_cached_tokens=sum(m.cached_tokens or 0 for m in metered),
        total_steps=len(steps),
    )
    if summary.get("final_response") and not any(s.source == "agent" and s.message for s in steps[-1:]):
        steps.append(Step(step_id=len(steps) + 1, timestamp=_iso(summary.get("finished_at")), source="agent", model_name=model, message=summary["final_response"]))
    return Trajectory(
        session_id="hx-task",
        agent=Agent(name="deepseek-harness", version=version or "unknown", model_name=model),
        steps=steps,
        final_metrics=final,
        extra={k: summary.get(k) for k in ("turn_cap_hit", "model_responses", "finish_reason", "error", "model_source", "profile", "max_turns", "reasoning_effort", "cache_retention", "session_log_upload")},
    )
