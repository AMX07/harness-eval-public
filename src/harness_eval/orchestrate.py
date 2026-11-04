"""`hx run`: experiment YAML -> Harbor job -> results -> Langfuse + report."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from . import experiment as ex
from .paths import CACHE_DIR, CHECKS_DIR, JOBS_DIR, load_env

# How to point each harness at the mock API ({base} is filled in per run).
# Extend per harness, or set `harness.mock_kwargs` in the experiment YAML.
MOCK_KWARGS = {
    "claude-code": {"base_url": "{base}"},
    "harness_eval.agents.deepseek_harness:DeepSeekHarness": {"api_base_url": "{base}"},
}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_mock(record_dir: Path, port: int) -> None:
    """Serve the mock API on the host loopback; containers reach it via host.docker.internal."""
    from aiohttp import web

    from .mock_api import make_app

    def serve() -> None:
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        runner = web.AppRunner(make_app(record_dir))
        loop.run_until_complete(runner.setup())
        loop.run_until_complete(web.TCPSite(runner, "127.0.0.1", port).start())
        loop.run_forever()

    threading.Thread(target=serve, daemon=True).start()


def mock_kwargs(exp: ex.Experiment, port: int) -> dict[str, Any]:
    extra = exp.harness.mock_kwargs or MOCK_KWARGS.get(exp.harness.agent)
    if extra is None:
        raise SystemExit(f"--mock: no mock settings for agent '{exp.harness.agent}'. Add `mock_kwargs` to the experiment's harness.")
    base = f"http://host.docker.internal:{port}"
    return {k: v.replace("{base}", base) if isinstance(v, str) else v for k, v in extra.items()}


def run(
    exp_ref: str,
    tasks: list[str] | None = None,
    attempts: int | None = None,
    concurrency: int | None = None,
    model: str | None = None,
    mock: bool = False,
    dry_run: bool = False,
    ingest: bool = True,
) -> Path | None:
    load_env()
    exp = ex.load(exp_ref)
    if not mock and not dry_run:
        missing = [s for s in exp.secrets if not os.environ.get(s)]
        if missing:
            raise SystemExit(f"Missing {', '.join(missing)}. Add them to .env (see .env.example).")

    prebake_key = None
    if exp.environment.prebake and exp.environment.type == "docker":
        from . import prebake

        prebake_key = prebake.recipe_for(exp).key
        if not dry_run:
            prebake.ensure(exp, tasks if tasks is not None else exp.tasks.resolve(), log=lambda m: print(m, flush=True))

    name = ex.job_name(exp, "mock" if mock else None)
    jobs_dir = CHECKS_DIR if mock else JOBS_DIR
    port = free_port() if mock else None
    if mock:
        exp = exp.model_copy(deep=True)
        exp.harness.kwargs = {**exp.harness.kwargs, **mock_kwargs(exp, port)}
    config = ex.harbor_job_config(
        exp, name, task_names=tasks, attempts=attempts, concurrency=concurrency, model_override=model,
        prebake_key=prebake_key, jobs_dir=jobs_dir,
    )

    CACHE_DIR.joinpath("configs").mkdir(parents=True, exist_ok=True)
    config_path = CACHE_DIR / "configs" / f"{name}.json"
    config_path.write_text(json.dumps(config, indent=2))
    n_tasks = len(config["datasets"][0].get("task_names") or []) or "all"
    print(f"[hx] job {name}: {exp.harness_label} + {config['agents'][0]['model_name']}, tasks={n_tasks}, attempts={config['n_attempts']}, concurrency={config['n_concurrent_trials']}{' (MOCK API)' if mock else ''}", flush=True)
    print(f"[hx] harbor config: {config_path}", flush=True)
    if dry_run:
        print(json.dumps(config, indent=2))
        return None

    env = dict(os.environ)
    if mock:
        start_mock(jobs_dir / name / "mock-requests", port)
        env["ANTHROPIC_API_KEY"] = "mock-key"  # never send a real key to the mock

    started = datetime.now(timezone.utc).isoformat()
    harbor = str(Path(sys.executable).parent / "harbor")
    proc = subprocess.run([harbor, "run", "-c", str(config_path), "-y"], env=env, cwd=str(JOBS_DIR.parent))
    job_dir = jobs_dir / name
    ex.write_meta(job_dir, exp, config, {
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "harbor_exit_code": proc.returncode,
        "harbor_version": version("harbor"),
        "mock": mock,
        "pricing_path": exp.pricing_path,
        "prebake_key": prebake_key,
    })
    if proc.returncode != 0:
        print(f"[hx] harbor exited with {proc.returncode}; partial results are in {job_dir}")

    from .report import print_report

    print_report([job_dir])
    if ingest and exp.langfuse:
        from . import langfuse_stack
        from .langfuse_ingest import ingest_job

        if langfuse_stack.healthy():
            res = ingest_job(job_dir)
            print(f"[hx] Langfuse: {res['sent']} trials sent -> {res['host']} (dataset {res['dataset']})")
        else:
            print("[hx] Langfuse is not running; start it with `uv run hx langfuse up`, then `uv run hx ingest " + str(job_dir) + "`")
    return job_dir
