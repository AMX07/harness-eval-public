# harness-eval

Run coding-agent **harnesses** on **Terminal-Bench 2.0** in local Docker, price every run with one frozen price list, and browse every task, step and score in a local **Langfuse**.

The first study: replicate [HarnessTax](https://arena.ai/blog/coding-agents-harness-tax) (Claude Code + Claude Haiku 4.5), then swap only the harness for [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) and measure the difference.

```
experiments/*.yaml ──hx run──> Harbor job (Docker, one container per trial)
                                   │  agent/trajectory.json (ATIF), verifier/ctrf.json, result.json
                                   ▼
                     results.py (tokens, cost, tests) ──> hx report / hx compare
                                   └────────────────────> Langfuse (datasets, runs, traces, scores)
```

## Setup

1. **Docker Desktop**: Settings → Resources → give it at least **32 GB memory** and 10 CPUs (it ships with ~8 GB; Terminal-Bench tasks need 2–8 GB each, Langfuse ~3 GB). Keep "Use Rosetta for x86_64/amd64 emulation" on: every Terminal-Bench image is amd64.
2. **Python deps** (uv): `uv sync`
3. **API key**: `cp .env.example .env` and set `ANTHROPIC_API_KEY`. Keys stay in `.env`; they are passed to Harbor's process and never written into job files.
4. **Langfuse**: `uv run hx langfuse up` → http://localhost:3000. The first run generates local secrets and a login in `infra/langfuse/.env` (email `admin@harness-eval.local`, password `LANGFUSE_INIT_USER_PASSWORD`).

## Run

```bash
# Free pipeline check: a scripted mock replaces the model (no key, no spend)
uv run hx run tb2-cc-haiku45 --mock --tasks fix-git --attempts 1 --concurrency 1

# One real task first, to check the setup with a few cents of spend
uv run hx run tb2-cc-haiku45 --tasks fix-git --attempts 1

# The replication: 30 tasks x 3 attempts (blog: $0.253/attempt -> ~$23)
uv run hx run tb2-cc-haiku45
uv run hx report tb2-cc-haiku45 --per-task      # includes the check against the blog's numbers

# Same everything, DeepSeek Harness instead of Claude Code
uv run hx run tb2-dsh-haiku45
uv run hx compare tb2-cc-haiku45 tb2-dsh-haiku45
```

The first run of an experiment pre-installs its harness into each task image (pulling the task images too), about 1–2 minutes per image, 4 at a time. Later runs and every attempt reuse them, so trials start in seconds instead of spending 3–6 minutes installing the harness, which under emulation can exceed Harbor's 6-minute setup limit. `uv run hx prebake <exp>` builds them ahead of time; `environment.prebake: false` in an experiment turns this off.

A job name prefix (`tb2-cc-haiku45`) resolves to the newest matching job in `jobs/`. Mock runs go to `jobs/checks/` (as do earlier pipeline checks), so a prefix only ever picks a real run; name a check in full to report on it. Other commands: `hx jobs`, `hx ingest <job>`, `hx tasks list`, `hx tasks sample --n 30 --seed 2026`, `hx baseline`, `hx mock-api --record DIR`, `hx langfuse down|status`.

## Where to look in Langfuse

- **Datasets → terminal-bench@2.0 → Experiments**: one experiment per job and attempt (`<job> · a1`, `a2`, `a3`) with total cost, latency and average scores (reward = success rate). Select experiments to compare them task by task; **Items** lists every task with its instruction.
- **Sessions**: one session per job, listing its trials in order with total cost.
- **Traces**: one per trial, named after the task, tagged with harness, model, experiment and passed/failed/error. Inside: every model call (tokens, cache reads and writes, cost), every tool call with its output, and the verifier's passed/failed tests.
- **Scores**: `reward`, `cost_usd`, `tokens`, `llm_calls`, `turn_cap_hit`.

## Adding experiments and harnesses

- **New experiment**: copy a YAML in `experiments/`. Keep the fields you are not studying identical, so `hx compare` isolates one variable. `hx run <exp> --model anthropic/claude-haiku-5-5` overrides the model for one run.
- **New harness**: any Harbor agent works. Built-ins by name (`claude-code`, `codex`, `pi`, `opencode`, `cline-cli`, …; options: `uv run harbor agent schema <name>`), your own by import path (`module:Class`, see `src/harness_eval/agents/deepseek_harness.py`). To pre-install it, add a recipe to `prebake.recipe_for` (its install script, and an install that skips when it finds itself installed), or set `environment.prebake: false`. Token and cost accounting needs the adapter to write an ATIF `agent/trajectory.json` with per-step metrics: `prompt_tokens` = uncached + cache read + cache write, `cached_tokens` = cache read, `extra.cache_creation_input_tokens` = cache writes. For `--mock`, add `mock_kwargs` with `{base}` where the mock API's URL goes (e.g. `base_url: "{base}"`); each mock run gets its own port.
- **New model or prices**: `pricing/anthropic.yaml` (add a new file rather than editing old prices, and point `pricing:` at it).
- **New task sample**: `hx tasks sample --n 30 --seed <s>` and set `tasks.taskset`.

## How the DeepSeek Harness adapter works

`harness_eval.agents.deepseek_harness:DeepSeekHarness` installs `deepseek-harness-sdk` (it bundles the native `dsh` runtime) with uv into `/opt/hx-*`, isolated from the task's Python, and skips the install when the pre-installed image already has it. It runs one SDK session with the `sdk-minimal` profile, which dsh's `BENCHMARK.md` prescribes for benchmarks: a one-line system prompt and three tools (`bash`, `read_image`, `working_directory`). A profile patch registers the Anthropic route through dsh's built-in pi-ai adapter (prompt caching on) and **turns off dsh's session-log upload**, which the shipped profiles enable for official DeepSeek requests. The 100-turn cap counts model responses and stops the session at the cap. Session events are converted to ATIF (`agent/trajectory.json`), so cost and tokens are computed exactly as for Claude Code.

## Method notes (read before comparing with the blog)

- **Tasks**: HarnessTax sampled 30 tasks but did not publish which. `tb2-n30-seed2026` is our own random 30, so expect differences from task mix alone; use per-attempt cost distributions (`hx report` shows p10/p50/p90 against the blog's) as well as means.
- **Container contents**: both harnesses' containers get the same system packages (curl, bash, Node.js, npm, procps), the set Harbor's Claude Code installer adds. 29 of the 30 task images lack Node and none have procps (`ps`, `pgrep`), so without this only Claude Code's agent would have them. On Ubuntu 24.04 images, apt's npm also brings in python3 (3.12), so both agents have it there; DeepSeek Harness itself runs on uv's own Python under `/opt/hx-python`, never the image's. Claude Code is pinned to 2.1.316 so every trial runs the same release.
- **Hardware**: tasks run x86 images under emulation on Apple Silicon, so wall-clock is slower than cloud x86. Timeouts are the task defaults (`timeout_multiplier: 1.0`), so slow tasks can time out where they would not on x86.
- **Effort**: Claude Code gets `--effort high`. dsh's `high` on Haiku 4.5 sends a 16,384-token thinking budget. These are each harness's own "high" setting, as in the blog, not identical settings.
- **Cost**: computed from each model call's recorded usage with the frozen table, the same for every harness; the harness's own reported cost is kept as `cost_usd_reported`.
- **Statistics**: per task, attempts are averaged first; 95% intervals come from 10,000 task-level bootstrap resamples (the blog's method). 30 tasks gives wide intervals (roughly ±10 points on success).

## Layout

```
experiments/   one YAML per experiment            tasksets/   task lists (seeded samples)
pricing/       frozen price tables                baselines/  HarnessTax per-attempt data (hx baseline)
infra/langfuse docker compose for Langfuse        jobs/       Harbor outputs + hx-report.md per job
src/harness_eval/
  cli.py  orchestrate.py  experiment.py  tasks.py  results.py  pricing.py  stats.py  report.py
  langfuse_ingest.py  langfuse_stack.py  mock_api.py  baseline.py
  prebake.py  environments/prebaked.py   pre-installed task images, and the Harbor environment that uses them
  agents/deepseek_harness.py  agents/dsh/{runner.py, atif.py}
```
