"""`hx`: run harness experiments on Terminal-Bench and browse them in Langfuse."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from .paths import CHECKS_DIR, JOBS_DIR, LANGFUSE_ENV, TASKSETS_DIR, load_env, read_env_file

app = typer.Typer(help=__doc__, no_args_is_help=True, add_completion=False)
tasks_app = typer.Typer(help="List and sample benchmark tasks.", no_args_is_help=True)
lf_app = typer.Typer(help="Run the local Langfuse stack.", no_args_is_help=True)
app.add_typer(tasks_app, name="tasks")
app.add_typer(lf_app, name="langfuse")
console = Console()


def resolve_job(ref: str) -> Path:
    """A job directory by path, by name (also in jobs/checks), or the newest real run whose name starts with `ref`."""
    path = Path(ref)
    if path.is_dir():
        return path
    for parent in (JOBS_DIR, CHECKS_DIR):
        if (parent / ref).is_dir():
            return parent / ref
    matches = sorted((d for d in JOBS_DIR.glob(f"{ref}*") if d.is_dir() and d != CHECKS_DIR), key=lambda d: d.stat().st_mtime)
    if not matches:
        raise typer.BadParameter(f"no job matching '{ref}' in {JOBS_DIR}")
    return matches[-1]


@app.command()
def run(
    experiment: Annotated[str, typer.Argument(help="Experiment name in experiments/ or a YAML path")],
    tasks: Annotated[Optional[str], typer.Option(help="Comma-separated task names (overrides the experiment's task set)")] = None,
    attempts: Annotated[Optional[int], typer.Option(help="Attempts per task")] = None,
    concurrency: Annotated[Optional[int], typer.Option(help="Trials in parallel")] = None,
    model: Annotated[Optional[str], typer.Option(help="Override the model, e.g. anthropic/claude-haiku-5-5")] = None,
    mock: Annotated[bool, typer.Option(help="Use the local mock API instead of a real model (free; tests the pipeline)")] = False,
    dry_run: Annotated[bool, typer.Option(help="Print the Harbor job config and exit")] = False,
    ingest: Annotated[bool, typer.Option(help="Send results to Langfuse when it is running")] = True,
):
    """Run an experiment: Harbor job -> report -> Langfuse."""
    from .orchestrate import run as do_run

    names = [t.strip() for t in tasks.split(",")] if tasks else None
    do_run(experiment, names, attempts, concurrency, model, mock, dry_run, ingest)


@app.command()
def prebake(
    experiment: Annotated[str, typer.Argument(help="Experiment name in experiments/ or a YAML path")],
    tasks: Annotated[Optional[str], typer.Option(help="Comma-separated task names (default: the experiment's task set)")] = None,
    jobs: Annotated[int, typer.Option(help="Images built in parallel")] = 4,
):
    """Build the experiment's task images with the harness pre-installed (hx run builds missing ones itself)."""
    from . import experiment as ex
    from .prebake import ensure

    exp = ex.load(experiment)
    names = [t.strip() for t in tasks.split(",")] if tasks else exp.tasks.resolve()
    recipe = ensure(exp, names, jobs=jobs, log=lambda m: console.print(m, markup=False))
    console.print(f"Images ready for {recipe.harness} {recipe.version} (tag key {recipe.key})")


@app.command()
def report(
    jobs: Annotated[list[str], typer.Argument(help="Job directories or names (prefix = newest match)")],
    per_task: Annotated[bool, typer.Option(help="Also show a per-task table")] = False,
):
    """Summarize one or more finished jobs (and check against the HarnessTax baseline)."""
    from .report import per_task_table, print_report

    dirs = [resolve_job(j) for j in jobs]
    print_report(dirs)
    if per_task:
        for d in dirs:
            console.print(per_task_table(d))


@app.command()
def compare(job_a: str, job_b: str):
    """Paired comparison of two jobs on their shared tasks (B vs A)."""
    from .report import compare as do_compare

    do_compare(resolve_job(job_a), resolve_job(job_b))


@app.command()
def ingest(
    job: str,
    force: Annotated[bool, typer.Option(help="Resend trials already sent")] = False,
):
    """Send a finished job to Langfuse."""
    load_env()
    from .langfuse_ingest import ingest_job

    res = ingest_job(resolve_job(job), force=force)
    console.print(f"Sent {res['sent']} of {res['trials']} trials to {res['host']} (dataset {res['dataset']}).")


@app.command()
def jobs():
    """List jobs with their headline numbers."""
    from . import stats
    from .report import _label, load

    table = Table(title=str(JOBS_DIR))
    for col in ("job", "run", "trials", "success", "cost / attempt"):
        table.add_column(col)
    for d in sorted((p for p in JOBS_DIR.iterdir() if p.is_dir()), key=lambda p: p.name):
        trials = load(d)
        if not trials:
            continue
        s = stats.summarize(trials)
        table.add_row(d.name, _label(d), str(s["n_trials"]), f"{s['success'].mean:.1%}", f"${s['cost_usd'].mean:.3f}")
    console.print(table)


@tasks_app.command("list")
def tasks_list(dataset: str = "terminal-bench@2.0"):
    """All tasks in a dataset with difficulty, category and resources."""
    from .tasks import load_tasks

    table = Table(title=dataset)
    for col in ("task", "difficulty", "category", "cpus", "memory", "timeout (s)"):
        table.add_column(col)
    for t in load_tasks(dataset).values():
        table.add_row(t.name, t.difficulty or "", t.category or "", str(t.cpus or ""), t.memory or "", str(t.agent_timeout_sec or ""))
    console.print(table)


@tasks_app.command("sample")
def tasks_sample(
    n: int = 30,
    seed: int = 2026,
    dataset: str = "terminal-bench@2.0",
    name: Annotated[Optional[str], typer.Option(help="Taskset file name (default tb2-n{n}-seed{seed})")] = None,
):
    """Write a reproducible random task sample to tasksets/."""
    from .tasks import sample, write_taskset

    names = sample(dataset, n, seed)
    path = TASKSETS_DIR / f"{name or f'tb2-n{n}-seed{seed}'}.txt"
    write_taskset(names, path, f"{dataset}: {len(names)} tasks, uniform random sample, seed={seed} (hx tasks sample)")
    console.print(f"Wrote {len(names)} tasks to {path}")


@lf_app.command("up")
def lf_up():
    """Start Langfuse (first run generates local secrets and a login)."""
    from . import langfuse_stack

    created = langfuse_stack.init()
    langfuse_stack.up()
    env = read_env_file(LANGFUSE_ENV)
    console.print(f"Langfuse is running at {langfuse_stack.host()}")
    console.print(f"Login: {env.get('LANGFUSE_INIT_USER_EMAIL')} — password is LANGFUSE_INIT_USER_PASSWORD in {LANGFUSE_ENV}" + (" (just generated)" if created else ""))


@lf_app.command("down")
def lf_down(remove_data: Annotated[bool, typer.Option(help="Also delete stored traces")] = False):
    """Stop Langfuse (data is kept unless --remove-data)."""
    from . import langfuse_stack

    langfuse_stack.down(remove_data)


@lf_app.command("status")
def lf_status():
    """Show container status and health."""
    from . import langfuse_stack

    console.print(langfuse_stack.status())
    console.print("healthy" if langfuse_stack.healthy() else "not reachable")


@app.command("baseline")
def baseline_fetch(benchmark: str = "tb"):
    """Re-extract the HarnessTax per-attempt data from the blog into baselines/."""
    from .baseline import save

    console.print(f"Wrote {save(benchmark)}")


@app.command("mock-api")
def mock_api(
    port: int = 8789,
    record: Annotated[Optional[Path], typer.Option(help="Save every request body here")] = None,
):
    """Serve the scripted mock Anthropic API (for debugging a harness by hand)."""
    from .mock_api import serve

    console.print(f"Mock Anthropic API on http://127.0.0.1:{port} (containers: http://host.docker.internal:{port})")
    serve("127.0.0.1", port, record)


def main() -> None:
    app()
