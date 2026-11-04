"""Project paths and .env loading shared by every command."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOBS_DIR = ROOT / "jobs"
CHECKS_DIR = JOBS_DIR / "checks"  # mock runs and other pipeline checks, out of the way of real runs
EXPERIMENTS_DIR = ROOT / "experiments"
TASKSETS_DIR = ROOT / "tasksets"
BASELINES_DIR = ROOT / "baselines"
PRICING_DIR = ROOT / "pricing"
CACHE_DIR = ROOT / ".cache"
DATASETS_CACHE = CACHE_DIR / "datasets"
LANGFUSE_DIR = ROOT / "infra" / "langfuse"
LANGFUSE_ENV = LANGFUSE_DIR / ".env"
DOTENV = ROOT / ".env"


def registry_path() -> str | None:
    """Optional local legacy registry.json for Harbor (HX_HARBOR_REGISTRY_PATH).

    For machines that can't reach Harbor's hosted registry (e.g. a network
    policy that blocks it): Harbor then resolves datasets from this file, whose
    tasks are pinned git commits.
    """
    value = os.environ.get("HX_HARBOR_REGISTRY_PATH")
    return str(Path(value).resolve()) if value else None


def read_env_file(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines; ignores blanks, comments and `export ` prefixes."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def load_env() -> dict[str, str]:
    """Merge .env (API keys) and the Langfuse stack env into os.environ.

    Values already set in the shell win, so a one-off `ANTHROPIC_API_KEY=...`
    prefix still overrides the file.
    """
    merged: dict[str, str] = {}
    lf = read_env_file(LANGFUSE_ENV)
    if lf:
        port = lf.get("LANGFUSE_PORT", "3000")
        merged.update(
            {
                "LANGFUSE_HOST": f"http://localhost:{port}",
                "LANGFUSE_PUBLIC_KEY": lf.get("LANGFUSE_INIT_PROJECT_PUBLIC_KEY", ""),
                "LANGFUSE_SECRET_KEY": lf.get("LANGFUSE_INIT_PROJECT_SECRET_KEY", ""),
            }
        )
    merged.update(read_env_file(DOTENV))
    for key, value in merged.items():
        if value and not os.environ.get(key):
            os.environ[key] = value
    return merged
