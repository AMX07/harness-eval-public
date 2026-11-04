"""Experiment definitions (experiments/*.yaml) and their Harbor job configs.

One experiment = one harness + one model + one task set + run settings. Keep
everything that should stay fixed across a comparison in the YAML, so two
experiments differ only in the field under study.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from .paths import EXPERIMENTS_DIR, JOBS_DIR, ROOT, registry_path
from .tasks import read_taskset


class TaskSelection(BaseModel):
    taskset: str | None = Field(None, description="File in tasksets/ (one task name per line)")
    names: list[str] | None = None
    all: bool = False

    @model_validator(mode="after")
    def exactly_one(self):
        chosen = [self.taskset is not None, self.names is not None, self.all]
        if sum(chosen) != 1:
            raise ValueError("tasks: set exactly one of taskset, names or all")
        return self

    def resolve(self) -> list[str] | None:
        if self.all:
            return None
        return list(self.names) if self.names is not None else read_taskset(self.taskset)


class Harness(BaseModel):
    agent: str = Field(description="Harbor agent name (claude-code) or import path (module:Class)")
    model: str = Field(description="provider/model, e.g. anthropic/claude-haiku-4-5")
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Agent options (harbor agent schema <name>)")
    env: dict[str, str] = Field(default_factory=dict, description="Non-secret env for the agent")
    label: str | None = Field(None, description="Display name in reports; defaults to the agent name")
    mock_kwargs: dict[str, Any] | None = Field(None, description="Agent options that point it at the mock API (`hx run --mock`)")


class Environment(BaseModel):
    type: str = "docker"
    kwargs: dict[str, Any] = Field(default_factory=dict)
    override_cpus: float | None = None
    override_memory_mb: int | None = None
    resource_limit_multiplier: float = Field(
        1.0, ge=1.0,
        description="Hard kill limit = task cpus/memory x this; the task spec stays the guaranteed allocation (PrebakedDocker)",
    )
    delete: bool = True
    prebake: bool = Field(True, description="Start trials from task images with the harness pre-installed (prebake.py)")


PREBAKED_ENVIRONMENT = "harness_eval.environments.prebaked:PrebakedDocker"


class Baseline(BaseModel):
    source: str = "harnesstax"
    system: str


class Experiment(BaseModel):
    name: str
    description: str = ""
    dataset: str = "terminal-bench@2.0"
    tasks: TaskSelection
    attempts: int = 3
    concurrency: int = 4
    timeout_multiplier: float = 1.0
    max_retries: int = 0
    harness: Harness
    environment: Environment = Field(default_factory=Environment)
    pricing: str = "pricing/anthropic.yaml"
    secrets: list[str] = Field(default_factory=list, description="Env vars that must be set (from .env); never written to job files")
    baseline: Baseline | None = None
    langfuse: bool = True

    @property
    def harness_label(self) -> str:
        return self.harness.label or self.harness.agent.split(":")[-1]

    @property
    def pricing_path(self) -> str:
        return str((ROOT / self.pricing).resolve())


def load(ref: str | Path) -> Experiment:
    path = Path(ref)
    if not path.is_file():
        path = EXPERIMENTS_DIR / (str(ref) if str(ref).endswith(".yaml") else f"{ref}.yaml")
    return Experiment.model_validate(yaml.safe_load(path.read_text()))


def job_name(exp: Experiment, suffix: str | None = None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{exp.name}__{stamp}" + (f"__{suffix}" if suffix else "")


def harbor_job_config(
    exp: Experiment,
    name: str,
    task_names: list[str] | None = None,
    attempts: int | None = None,
    concurrency: int | None = None,
    model_override: str | None = None,
    prebake_key: str | None = None,
    jobs_dir: Path = JOBS_DIR,
) -> dict[str, Any]:
    """Translate an experiment into Harbor's JobConfig (see `harbor job schema`)."""
    ds_name, _, ds_version = exp.dataset.partition("@")
    tasks = task_names if task_names is not None else exp.tasks.resolve()
    agent: dict[str, Any] = {
        "model_name": model_override or exp.harness.model,
        "kwargs": exp.harness.kwargs,
        "env": exp.harness.env,
    }
    if ":" in exp.harness.agent:
        agent["import_path"] = exp.harness.agent
    else:
        agent["name"] = exp.harness.agent
    env: dict[str, Any] = {"type": exp.environment.type, "delete": exp.environment.delete, "kwargs": exp.environment.kwargs}
    if prebake_key:
        env = {"import_path": PREBAKED_ENVIRONMENT, "delete": exp.environment.delete, "kwargs": {**exp.environment.kwargs, "prebake_key": prebake_key}}
    if exp.environment.resource_limit_multiplier != 1.0:
        if "import_path" not in env:
            if exp.environment.type != "docker":
                raise ValueError("environment.resource_limit_multiplier needs the local docker environment")
            env = {"import_path": PREBAKED_ENVIRONMENT, "delete": exp.environment.delete, "kwargs": dict(exp.environment.kwargs)}
        env["kwargs"]["resource_limit_multiplier"] = exp.environment.resource_limit_multiplier
    if exp.environment.override_cpus is not None:
        env["override_cpus"] = exp.environment.override_cpus
    if exp.environment.override_memory_mb is not None:
        env["override_memory_mb"] = exp.environment.override_memory_mb
    dataset: dict[str, Any] = {"name": ds_name}
    if ds_version:
        dataset["version"] = ds_version
    if tasks is not None:
        dataset["task_names"] = tasks
    if rp := registry_path():
        dataset["registry_path"] = rp
    return {
        "job_name": name,
        "jobs_dir": str(jobs_dir),
        "n_attempts": attempts or exp.attempts,
        "timeout_multiplier": exp.timeout_multiplier,
        "n_concurrent_trials": concurrency or exp.concurrency,
        "retry": {"max_retries": exp.max_retries},
        "environment": env,
        "agents": [agent],
        "datasets": [dataset],
    }


def write_meta(job_dir: Path, exp: Experiment, config: dict[str, Any], extra: dict[str, Any]) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "experiment": exp.model_dump(),
        "harness_label": exp.harness_label,
        "harbor_config": config,
        **extra,
    }
    (job_dir / "hx-experiment.json").write_text(json.dumps(record, indent=2, default=str))
