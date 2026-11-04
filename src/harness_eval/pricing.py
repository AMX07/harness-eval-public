"""Price one model call from its token usage with a frozen price table."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from .paths import PRICING_DIR

_DATE_SUFFIX = re.compile(r"-\d{8}$")


@dataclass(frozen=True)
class Usage:
    """Token usage of one model request, split the way Anthropic bills it."""

    input: int = 0  # uncached input tokens
    output: int = 0
    cache_read: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0

    @property
    def prompt(self) -> int:
        return self.input + self.cache_read + self.cache_write_5m + self.cache_write_1h

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input + other.input,
            self.output + other.output,
            self.cache_read + other.cache_read,
            self.cache_write_5m + other.cache_write_5m,
            self.cache_write_1h + other.cache_write_1h,
        )


def normalize_model(name: str | None) -> str | None:
    """'anthropic/claude-haiku-4-5-20251001' -> 'claude-haiku-4-5'."""
    if not name:
        return None
    return _DATE_SUFFIX.sub("", name.split("/")[-1].strip())


@lru_cache
def load_table(path: str | None = None) -> dict:
    return yaml.safe_load(Path(path or PRICING_DIR / "anthropic.yaml").read_text())


def cost_breakdown(model: str | None, usage: Usage, table_path: str | None = None) -> dict[str, float] | None:
    """USD cost of one request split by token kind, or None when the model is not in the table."""
    entry = load_table(table_path)["models"].get(normalize_model(model) or "")
    if entry is None:
        return None
    if "tiers" in entry:
        rates = entry["tiers"][-1]
        for tier in entry["tiers"]:
            if usage.prompt <= tier.get("max_prompt_tokens", float("inf")):
                rates = tier
                break
    else:
        rates = entry
    parts = {
        "input": usage.input * rates["input"] / 1e6,
        "output": usage.output * rates["output"] / 1e6,
        "cache_read": usage.cache_read * rates["cache_read"] / 1e6,
        "cache_write": (usage.cache_write_5m * rates["cache_write_5m"] + usage.cache_write_1h * rates["cache_write_1h"]) / 1e6,
    }
    parts["total"] = sum(parts.values())
    return parts


def call_cost(model: str | None, usage: Usage, table_path: str | None = None) -> float | None:
    """USD cost of one request, or None when the model is not in the table."""
    parts = cost_breakdown(model, usage, table_path)
    return None if parts is None else parts["total"]


def known_models(table_path: str | None = None) -> list[str]:
    return sorted(load_table(table_path)["models"])
