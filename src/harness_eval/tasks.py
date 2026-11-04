"""Task catalog for a Harbor dataset, and reproducible task samples."""

from __future__ import annotations

import random
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .paths import DATASETS_CACHE, TASKSETS_DIR, registry_path


@dataclass
class TaskInfo:
    name: str
    instruction: str
    difficulty: str | None
    category: str | None
    tags: list[str]
    agent_timeout_sec: float | None
    cpus: float | None
    memory: str | None
    docker_image: str | None
    path: Path


def dataset_dir(dataset: str) -> Path:
    """Local copy of `name@version`, downloaded through Harbor on first use."""
    name = dataset.split("@")[0]
    target = DATASETS_CACHE / name
    if not target.is_dir() or not any(target.iterdir()):
        DATASETS_CACHE.mkdir(parents=True, exist_ok=True)
        registry = ["--registry-path", rp] if (rp := registry_path()) else []
        subprocess.run(["harbor", "download", dataset, "-o", str(DATASETS_CACHE), *registry], check=True)
    return target


def load_tasks(dataset: str) -> dict[str, TaskInfo]:
    tasks: dict[str, TaskInfo] = {}
    for toml_path in sorted(dataset_dir(dataset).glob("*/task.toml")):
        cfg = tomllib.loads(toml_path.read_text())
        meta = cfg.get("metadata", {})
        env = cfg.get("environment", {})
        instr = toml_path.parent / "instruction.md"
        tasks[toml_path.parent.name] = TaskInfo(
            name=toml_path.parent.name,
            instruction=instr.read_text() if instr.is_file() else "",
            difficulty=meta.get("difficulty"),
            category=meta.get("category"),
            tags=list(meta.get("tags") or []),
            agent_timeout_sec=(cfg.get("agent") or {}).get("timeout_sec"),
            cpus=env.get("cpus"),
            memory=env.get("memory"),
            docker_image=env.get("docker_image"),
            path=toml_path.parent,
        )
    return tasks


def sample(dataset: str, n: int, seed: int) -> list[str]:
    """Uniform random sample of task names; same (dataset, n, seed) -> same list."""
    names = sorted(load_tasks(dataset))
    if n >= len(names):
        return names
    return sorted(random.Random(seed).sample(names, n))


def write_taskset(names: list[str], path: Path, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {header}\n" + "\n".join(names) + "\n")


def read_taskset(ref: str) -> list[str]:
    """A taskset file (one task per line, # comments) by path or by name in tasksets/."""
    path = Path(ref)
    if not path.is_file():
        path = TASKSETS_DIR / (ref if ref.endswith(".txt") else f"{ref}.txt")
    lines = [line.split("#")[0].strip() for line in path.read_text().splitlines()]
    return [line for line in lines if line]
