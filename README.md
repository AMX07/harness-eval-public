# harness-eval

Does the harness matter as much as [HarnessTax](https://arena.ai/blog/coding-agents-harness-tax) says? We held the model fixed (**Claude Haiku 5.5**) and swapped only the harness: **Claude Code** (2.1.316) vs **[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)** (dsh v0.2.1-rc.2, `sdk-minimal` profile: 3 tools, one-line system prompt).

Setup: Terminal-Bench 2.0, 30 random tasks × 3 attempts, 100-turn cap, local Docker on Apple Silicon, one frozen price list. We ran it twice: once with the task defaults, and once with 3× CPU/memory headroom ("x3"). Statistics follow HarnessTax: average per task, then a 10,000-resample task bootstrap for 95% intervals.

## HarnessTax claims, checked

| HarnessTax claim | Verdict | What we found (DSH vs Claude Code) |
|---|---|---|
| **The harness changes cost far more than success** | ✅ Validated | Cost differs clearly; the success gap does not (−5.6 and −8.9 pts, both intervals include 0) |
| Claude Code costs **~1.5×** a minimal harness on Terminal-Bench (geometric mean of per-task cost ratios) | ✅ Validated | Claude Code costs **1.49×** (first run) and **1.54×** (x3); DSH is cheaper on 25 and 26 of 30 tasks |
| Similar turn counts can cost very different amounts | ✅ Validated | Model calls 0.96× (about equal), cost 0.65× |
| The tax starts at the first call: Claude Code's initial context is **>10×** a minimal harness | ✅ Validated | First call: 17,625 vs 1,419 prompt tokens (**12.4×**): 12 tool schemas, a 9k-char system prompt and injected reminders vs 3 tools and 46 chars |
| A simple harness can be competitive | 🟡 Partly | Competitive on cost (~60% fewer tokens), but trails on success in both runs (not significantly) |
| Models can do better outside their provider's harness ("Claude may not need Claude Code") | ❌ Not supported | Haiku 5.5 scored higher in Claude Code in both runs (72.2% vs 66.7%, then 72.2% vs 63.3%); not significant, but pointing the other way |

**Caveats:**
- **"Little effect on success" holds only on average.** Per task, the harness can decide the outcome: `pypi-server` and `hf-model-inference` are 3/3 for Claude Code and 0/3 for DSH in both runs, because dsh's shutdown kills the background server the verifier needs.
- **Mean cost is fragile.** In the first run DSH's *mean* cost was higher ($0.077 vs $0.062), even though it was cheaper on 25/30 tasks, because of two runaway attempts. Use per-task ratios.
- **Infrastructure moves results.** DSH's 6 timeouts and out-of-memory kills disappeared with 3× headroom.

**Not tested here:** SWE-bench Lite, Codex or Pi, other models, the blog's "up to 5×" cost spread.

## Results

| Haiku 5.5 | Success [95% CI] | Cost / attempt | Tokens / attempt |
|---|---|---|---|
| Claude Code | 72.2% [57.8, 84.4] | $0.062 | 0.93 M |
| Claude Code, x3 | 72.2% [57.8, 85.6] | $0.071 | 1.13 M |
| DeepSeek Harness | 66.7% [51.1, 81.1] | $0.077 | 0.94 M |
| DeepSeek Harness, x3 | 63.3% [47.8, 77.8] | $0.049 | 0.69 M |

Paired, x3 (DSH vs CC): cost ratio **0.65× [0.56, 0.76]**, token ratio **0.40×**, success **−8.9 pts [−22.2, +3.3]**.

Raw run data (trajectories, logs, write-ups) is not in this repository.

## Run it

Needs Docker (≥32 GB memory for the VM; Rosetta on, since task images are amd64) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync && cp .env.example .env            # then set ANTHROPIC_API_KEY in .env
uv run hx run tb2-cc-haiku55-x3 --mock --tasks fix-git --attempts 1   # free pipeline check
uv run hx run tb2-cc-haiku55-x3 && uv run hx run tb2-dsh-haiku55-x3
uv run hx compare tb2-cc-haiku55-x3 tb2-dsh-haiku55-x3
uv run hx langfuse up && uv run hx ingest tb2-cc-haiku55-x3     # browse traces at localhost:3000
```

Experiments are YAML files in `experiments/`, prices are in `pricing/`, and the DSH adapter is `src/harness_eval/agents/deepseek_harness.py`. `uv run hx --help` lists every command.
