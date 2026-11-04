#!/usr/bin/env python3
"""Run one task with the DeepSeek Harness Python SDK (runs inside the task container).

Uploaded by the Harbor adapter (harness_eval.agents.deepseek_harness) and run
with the isolated interpreter at /opt/hx-dsh, so nothing is installed into the
task's own Python. Writes, under --out-dir:
  dsh-events.jsonl   every session event, streamed as recorded (live log)
  dsh-summary.json   final response, finish reason, turn count, errors
  sessions/          the harness's own JSONL session logs
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

from deepseek_harness import DeepSeekHarness


def _event_type(event: dict) -> str | None:
    for key in ("type", "kind", "name"):
        if isinstance(event.get(key), str):
            return event[key]
    return None


def write_patch(path: Path, args: argparse.Namespace, declare_model: bool) -> None:
    lines = [
        "- insert:",
        "    - id: llm-pi-ai",
        "      name: '@deepseek-ai/dsh-llm-pi-ai'",
        "      config:",
        "        providers:",
        f"          {args.provider}:",
        f"            apiKeyEnv: {args.api_key_env}",
    ]
    if args.base_url:
        lines.append(f"            baseURL: {json.dumps(args.base_url)}")
    if args.cache_retention:
        lines.append(f"            cacheRetention: {args.cache_retention}")
    if declare_model:
        lines += [
            "            models:",
            f"              - id: {args.model}",
            f"                contextWindow: {args.context_window}",
            f"                maxTokens: {args.max_output_tokens}",
            "                input: [text]",
        ]
    if not args.upload_session_log:
        # Off by default: the shipped profiles upload session logs alongside official DeepSeek requests.
        lines += ["- id: session-log-deepseek", "  config:", "    enabled: false"]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--instruction-file", required=True)
    p.add_argument("--provider", default="anthropic")
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-env", default="ANTHROPIC_API_KEY")
    p.add_argument("--base-url")
    p.add_argument("--profile", default="sdk-minimal")
    p.add_argument("--patch", action="append", default=[], help="Extra Cordis patch files, applied after ours")
    p.add_argument("--dsh-home", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-turns", type=int, default=0, help="Stop after this many model responses (0 = no cap)")
    p.add_argument("--reasoning-effort")
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--cache-retention", choices=["none", "short", "long"])
    p.add_argument("--context-window", type=int, default=200_000)
    p.add_argument("--max-output-tokens", type=int, default=64_000)
    p.add_argument("--upload-session-log", action="store_true")
    p.add_argument("--declare-model", action="store_true", help="Declare the model instead of using the bundled catalog")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    home = Path(args.dsh_home)
    home.mkdir(parents=True, exist_ok=True)
    events_path = out / "dsh-events.jsonl"
    instruction = Path(args.instruction_file).read_text()
    summary: dict = {
        "provider": args.provider,
        "model": args.model,
        "profile": args.profile,
        "max_turns": args.max_turns,
        "reasoning_effort": args.reasoning_effort,
        "cache_retention": args.cache_retention,
        "session_log_upload": bool(args.upload_session_log),
        "started_at": time.time(),
    }
    state = {"model_responses": 0, "turn_cap_hit": False}
    lock = threading.Lock()
    harness_ref: dict = {}

    def on_notification(n) -> None:
        with lock:
            with events_path.open("a") as f:
                f.write(json.dumps({"t": time.time(), "method": n.method, "payload": n.payload}, default=str) + "\n")
            event = (n.payload or {}).get("event") if n.method == "session.event" else None
            if isinstance(event, dict) and _event_type(event) == "assistant/message":
                state["model_responses"] += 1
                if args.max_turns and state["model_responses"] >= args.max_turns and not state["turn_cap_hit"]:
                    state["turn_cap_hit"] = True
                    # Closing from the notification thread could deadlock; do it on its own thread.
                    threading.Thread(target=lambda: harness_ref["h"].close(), daemon=True).start()

    exit_code = 0
    attempts = [args.declare_model] if args.declare_model else [False, True]
    for declare in attempts:
        patch = home / "hx-route.patch.yml"
        write_patch(patch, args, declare)
        harness = DeepSeekHarness(
            provider=args.provider,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_tokens=args.max_tokens,
            cwd=os.getcwd(),
            dsh_home=str(home),
            profile=args.profile,
            patches=(str(patch), *args.patch),
            initialize_timeout_seconds=300,
        )
        harness_ref["h"] = harness
        summary["model_source"] = "declared" if declare else "catalog"
        try:
            harness.start()
        except Exception as e:  # unknown model in the catalog -> retry with a declared model
            summary.setdefault("init_errors", []).append(f"{type(e).__name__}: {e}")
            harness.close()
            continue
        try:
            result = harness.run(instruction, session_id="hx-task", on_notification=on_notification)
            summary["final_response"] = result.final_response
            summary["finish_reason"] = result.finish_reason
        except Exception as e:
            summary["error"] = f"{type(e).__name__}: {e}"
            summary["traceback"] = traceback.format_exc()
            if not state["turn_cap_hit"] and state["model_responses"] == 0:
                exit_code = 1  # failed before the model ever answered: a setup or credential problem
        finally:
            harness.close()
        break
    else:
        summary["error"] = "initialization failed with both the catalog and a declared model"
        exit_code = 1

    summary.update(state)
    summary["finished_at"] = time.time()
    sessions = home / "sessions"
    if sessions.is_dir():
        shutil.copytree(sessions, out / "sessions", dirs_exist_ok=True)
    (out / "dsh-summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary.get(k) for k in ("finish_reason", "model_responses", "turn_cap_hit", "error")}))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
