"""Terminal + markdown reports: one job, several jobs, a paired comparison, a baseline check."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from . import stats
from .paths import BASELINES_DIR
from .results import Trial, job_meta, read_job

console = Console()


def _label(job_dir: Path) -> str:
    meta = job_meta(job_dir)
    exp = meta.get("experiment") or {}
    model = ((meta.get("harbor_config") or {}).get("agents") or [{}])[0].get("model_name") or (exp.get("harness") or {}).get("model")
    return f"{meta.get('harness_label', '?')} + {str(model).split('/')[-1]}"


def load(job_dir: Path) -> list[Trial]:
    return read_job(job_dir, job_meta(job_dir).get("pricing_path"))


def baseline_system(job_dir: Path) -> dict[str, Any] | None:
    base = (job_meta(job_dir).get("experiment") or {}).get("baseline")
    path = BASELINES_DIR / "harnesstax-tb2.json"
    if not base or not path.is_file():
        return None
    systems = json.loads(path.read_text())["systems"]
    return next((s for s in systems if s["system"] == base["system"]), None)


def summary_table(job_dirs: list[Path]) -> Table:
    table = Table(title="Results (mean per attempt; 95% CI from 10k task-level bootstrap resamples)")
    for col in ("run", "tasks×attempts", "success", "cost / attempt", "tokens / attempt", "LLM calls", "1st-call prompt", "errors", "turn-cap hits"):
        table.add_column(col)
    for job_dir in job_dirs:
        trials = load(job_dir)
        s = stats.summarize(trials)
        attempts = max((t.attempt for t in trials), default=0)
        table.add_row(
            f"{_label(job_dir)}\n[dim]{job_dir.name}[/dim]",
            f"{s['n_tasks']}×{attempts} = {s['n_trials']}",
            s["success"].fmt("pct"),
            s["cost_usd"].fmt("usd") + (f"\n[yellow]{s['n_unpriced_trials']} trials unpriced[/yellow]" if s["n_unpriced_trials"] else ""),
            s["tokens"].fmt("int"),
            f"{s['llm_calls'].mean:.1f}",
            f"{s['first_call_prompt_tokens'].mean:,.0f}",
            str(s["n_errors"]),
            str(s["n_turn_cap_hits"]),
        )
    return table


def baseline_table(job_dir: Path) -> Table | None:
    base = baseline_system(job_dir)
    if not base:
        return None
    trials = load(job_dir)
    s = stats.summarize(trials)
    ours_costs = sorted(t.cost_usd for t in trials if t.cost_usd is not None)
    theirs = base["rollout_costs_sorted_usd"]
    table = Table(title=f"Replication check vs HarnessTax: {base['system']} ({base['cohort']}, n={base['n']})")
    for col in ("metric", "HarnessTax", "this run", "verdict"):
        table.add_column(col)

    def inside(value: float, est: stats.Estimate) -> str:
        return "[green]within our 95% CI[/green]" if est.lo <= value <= est.hi else "[red]outside our 95% CI[/red]"

    table.add_row("success rate", f"{base['success_rate']:.1%}", s["success"].fmt("pct"), inside(base["success_rate"], s["success"]))
    table.add_row("cost / attempt", f"${base['mean_cost_usd']:.3f}", s["cost_usd"].fmt("usd"), inside(base["mean_cost_usd"], s["cost_usd"]))
    if base.get("rollout_tokens_sorted"):
        mean_tok = float(np.mean(base["rollout_tokens_sorted"]))
        table.add_row("tokens / attempt", f"{mean_tok:,.0f}", s["tokens"].fmt("int"), inside(mean_tok, s["tokens"]))
    if ours_costs:
        for q in (0.1, 0.5, 0.9):
            table.add_row(f"cost p{int(q * 100)}", f"${np.quantile(theirs, q):.3f}", f"${np.quantile(ours_costs, q):.3f}", "")
    return table


def print_report(job_dirs: list[Path]) -> None:
    console.print(summary_table(job_dirs))
    for job_dir in job_dirs:
        bt = baseline_table(job_dir)
        if bt:
            console.print(bt)
        write_markdown(job_dir)


def per_task_table(job_dir: Path) -> Table:
    trials = load(job_dir)
    table = Table(title=f"Per task — {_label(job_dir)}")
    for col in ("task", "attempts", "passed", "mean cost", "mean tokens", "notes"):
        table.add_column(col)
    by_task: dict[str, list[Trial]] = {}
    for t in trials:
        by_task.setdefault(t.task_name, []).append(t)
    for task, ts in sorted(by_task.items()):
        costs = [t.cost_usd for t in ts if t.cost_usd is not None]
        notes = ", ".join(sorted({t.exception_type for t in ts if t.exception_type} | ({"turn cap"} if any(t.turn_cap_hit for t in ts) else set())))
        table.add_row(task, str(len(ts)), "".join("✓" if t.passed else ("!" if t.status == "error" else "·") for t in ts),
                      f"${np.mean(costs):.3f}" if costs else "—", f"{np.mean([t.tokens for t in ts]):,.0f}", notes)
    return table


def compare(job_a: Path, job_b: Path) -> None:
    a, b = load(job_a), load(job_b)
    la, lb = _label(job_a), _label(job_b)
    table = Table(title=f"Paired comparison on shared tasks: B = {lb} vs A = {la}")
    for col in ("metric", "A", "B", "B − A [95% CI]", "B / A (geo. mean of task ratios)"):
        table.add_column(col)
    rows = [
        ("success", stats.success, "pct"),
        ("cost / attempt", lambda t: t.cost_usd, "usd"),
        ("tokens / attempt", lambda t: t.tokens if t.n_llm_calls else None, "int"),
        ("LLM calls", lambda t: t.n_llm_calls if t.n_llm_calls else None, "num"),
        ("1st-call prompt tokens", lambda t: t.first_call_prompt_tokens, "int"),
        ("agent minutes", lambda t: None if t.agent_seconds is None else t.agent_seconds / 60, "num"),
    ]
    n = 0
    for name, fn, kind in rows:
        p = stats.paired(a, b, fn)
        if not p.get("n"):
            continue
        n = p["n"]
        fmt = {"pct": "{:.1%}", "usd": "${:.3f}", "int": "{:,.0f}", "num": "{:.2f}"}[kind]
        diff = (fmt.format(p["diff"]) + f" [{fmt.format(p['lo'])}, {fmt.format(p['hi'])}]").replace("$-", "-$")
        ratio = f"{p['geo_mean_ratio']:.2f}×" if "geo_mean_ratio" in p and kind != "pct" else ""
        table.add_row(name, fmt.format(p["a"]), fmt.format(p["b"]), diff, ratio)
    table.caption = f"{n} shared tasks; per task, attempts are averaged first"
    console.print(table)

    # Tasks where the harnesses disagree most: the ones worth opening in Langfuse.
    sa = stats.task_means(a, stats.success)
    sb = stats.task_means(b, stats.success)
    gaps = sorted(((sb[k] - sa[k], k) for k in set(sa) & set(sb)), reverse=True)
    gap_table = Table(title="Largest per-task success gaps (B − A)")
    gap_table.add_column("task")
    gap_table.add_column("A")
    gap_table.add_column("B")
    for gap, task in [g for g in gaps if g[0] > 0][:5] + [g for g in gaps if g[0] < 0][-5:]:
        gap_table.add_row(task, f"{sa[task]:.0%}", f"{sb[task]:.0%}")
    console.print(gap_table)


def write_markdown(job_dir: Path) -> Path:
    trials = load(job_dir)
    s = stats.summarize(trials)
    lines = [
        f"# {job_dir.name}",
        "",
        f"- Harness + model: **{_label(job_dir)}**",
        f"- Tasks × attempts: {s['n_tasks']} tasks, {s['n_trials']} trials ({s['n_errors']} errors, {s['n_turn_cap_hits']} turn-cap hits)",
        f"- Success: **{s['success'].fmt('pct')}**",
        f"- Cost per attempt: **{s['cost_usd'].fmt('usd')}** (frozen price table)",
        f"- Tokens per attempt (prompt + output): {s['tokens'].fmt('int')}",
        f"- First model call prompt tokens: {s['first_call_prompt_tokens'].mean:,.0f}",
        "",
        "| task | attempt | status | reward | cost | tokens | LLM calls | turn cap | exception |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t in trials:
        lines.append(
            f"| {t.task_name} | {t.attempt} | {t.status} | {t.reward} | "
            f"{'' if t.cost_usd is None else f'${t.cost_usd:.4f}'} | {t.tokens:,} | {t.n_llm_calls} | {'yes' if t.turn_cap_hit else ''} | {t.exception_type or ''} |"
        )
    base = baseline_system(job_dir)
    if base:
        lines += ["", f"HarnessTax baseline ({base['system']}): success {base['success_rate']:.1%}, cost ${base['mean_cost_usd']:.3f}/attempt."]
    path = job_dir / "hx-report.md"
    path.write_text("\n".join(lines) + "\n")
    (job_dir / "hx-trials.json").write_text(json.dumps([t.summary() for t in trials], indent=1, default=str))
    return path
