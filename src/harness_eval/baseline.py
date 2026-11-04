"""Extract the HarnessTax per-rollout data from the blog's embedded charts.

The post (https://arena.ai/blog/coding-agents-harness-tax) embeds each chart as
an iframe whose script carries a gzip+base64 JSON blob. Figure 2's cumulative
cost-success curves contain every rollout's cost (and tokens, where the
harness reported them), so per-rollout distributions can be recovered:
c_i = (x_i - x_{i-1}) * n. Task IDs are not published.
"""

from __future__ import annotations

import base64
import gzip
import html
import json
import re
from datetime import date
from typing import Any

import httpx

from .paths import BASELINES_DIR

URL = "https://arena.ai/blog/coding-agents-harness-tax"
_ATOB = re.compile(r'atob\("([A-Za-z0-9+/=]{200,})"\)')


def _blobs(page: str) -> list[dict[str, Any]]:
    text = html.unescape(page).replace('\\"', '"')
    out = []
    for match in _ATOB.finditer(text):
        try:
            out.append(json.loads(gzip.decompress(base64.b64decode(match.group(1)))))
        except (ValueError, OSError):
            continue
    return out


def _find(blobs: list[dict[str, Any]], marker: str) -> dict[str, Any] | None:
    for blob in blobs:
        for key, value in blob.items():
            if marker in key:
                return value
    return None


def fetch(benchmark: str = "tb") -> dict[str, Any]:
    page = httpx.get(URL, timeout=60, follow_redirects=True, headers={"User-Agent": "harness-eval/0.1"}).text
    blobs = _blobs(page)
    accrual = _find(blobs, f"system-cumulative-accrual-{benchmark}")
    frontier = _find(blobs, f"system-frontier-{benchmark}")
    if accrual is None:
        raise RuntimeError("Could not find the cumulative-curve chart data on the page; the post layout may have changed.")
    systems = []
    for curve in accrual["accrual"]["curves"]:
        n = curve["n"]
        cost = curve["series"]["cost"]["exact"]
        costs = [round((b - a) * n, 6) for a, b in zip(cost["x"], cost["x"][1:])]
        toks = None
        tok_series = (curve["series"].get("gross_tokens") or {}).get("exact")
        if tok_series:
            toks = [round((b - a) * n * 1e6) for a, b in zip(tok_series["x"], tok_series["x"][1:])]
        systems.append({
            "system": curve["label"],
            "model": curve["model_label"],
            "harness": curve["harness_label"],
            "cohort": curve["cohort"],
            "n": n,
            "mean_cost_usd": round(cost["x"][-1], 6),
            "success_rate": round(cost["y"][-1], 6),
            "rollout_costs_sorted_usd": costs,
            "rollout_tokens_sorted": toks,
        })
    frontier_points = []
    if frontier:
        for p in frontier["frontier"]["points"]:
            frontier_points.append({k: p.get(k) for k in ("model_label", "harness_label", "x", "y", "x_ci", "y_ci", "sample", "frontier")})
    return {
        "source": URL,
        "benchmark": {"tb": "terminal-bench@2.0", "swe": "swe-bench-lite"}[benchmark],
        "extracted": date.today().isoformat(),
        "notes": "Per-rollout costs reconstructed from Figure 2's cumulative curves (c_i = (x_i - x_{i-1}) * n). "
                 "30 random tasks x 3 attempts, 100-turn cap, high effort, fixed price list dated 2026-09-01. Task IDs not published.",
        "systems": systems,
        "frontier_points": frontier_points,
    }


def save(benchmark: str = "tb") -> str:
    data = fetch(benchmark)
    path = BASELINES_DIR / f"harnesstax-{benchmark}2.json" if benchmark == "tb" else BASELINES_DIR / f"harnesstax-{benchmark}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))
    return str(path)
