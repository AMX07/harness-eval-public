"""Harbor adapter for DeepSeek Harness (dsh), via its Python SDK.

Use in an experiment as
    agent: harness_eval.agents.deepseek_harness:DeepSeekHarness
    model: anthropic/claude-haiku-4-5

Install: uv puts Python and `deepseek-harness-sdk` (which bundles the native
dsh runtime) under /opt/hx-*, isolated from the task's own Python packages,
after the same system packages Claude Code's installer adds (prebake.py), so
both harnesses see the same container. `hx run` normally bakes all of this into
the task image, and install() skips when it finds the SDK already there.
Run: dsh/runner.py drives one SDK session in the task's working directory with
the `sdk-minimal` profile, the setup dsh's BENCHMARK.md prescribes for
benchmarks, plus a profile patch that registers the model route and turns off
session-log upload. Afterwards the session events are converted to an ATIF
trajectory, so tokens, cost and steps are accounted like any other harness.
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Annotated, Any, override

from pydantic import Field

from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.agents.options import InstalledAgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from ..pricing import Usage, call_cost
from ..prebake import SYSTEM_PACKAGES
from .dsh import atif

UV_VERSION = "0.13.5"
RUNNER = Path(__file__).parent / "dsh" / "runner.py"
REMOTE_RUNNER = "/installed-agent/hx_dsh_runner.py"
REMOTE_INSTRUCTION = "/installed-agent/hx-instruction.md"
PROVIDER_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def install_script(sdk_version: str, python_version: str) -> str:
    """Root shell script that installs uv, Python and the SDK under /opt/hx-* (also used by prebake).

    uv must use its own Python: many task images have a system 3.12 (on Ubuntu 24.04, apt's npm brings in
    python3), and a venv on it would tie the harness to an interpreter the agent can change.
    """
    return f"""
set -euo pipefail
export UV_INSTALL_DIR=/opt/hx-uv UV_PYTHON_INSTALL_DIR=/opt/hx-python UV_CACHE_DIR=/opt/hx-uv-cache UV_NO_MODIFY_PATH=1
export UV_PYTHON_PREFERENCE=only-managed
if [ ! -x /opt/hx-uv/uv ]; then curl -LsSf https://astral.sh/uv/{UV_VERSION}/install.sh | sh >/dev/null; fi
/opt/hx-uv/uv venv --quiet --python {shlex.quote(python_version)} /opt/hx-dsh
/opt/hx-uv/uv pip install --quiet --python /opt/hx-dsh/bin/python {shlex.quote(f"deepseek-harness-sdk=={sdk_version}")}
chmod -R a+rX /opt/hx-uv /opt/hx-python /opt/hx-dsh
/opt/hx-dsh/bin/python -c 'import deepseek_harness, importlib.metadata as m; print("deepseek-harness-sdk", m.version("deepseek-harness-sdk"))'
"""


def installed_check(sdk_version: str) -> str:
    """Exits 0 when the SDK is already installed at `sdk_version` (a pre-installed image)."""
    code = f"import importlib.metadata as m, sys; sys.exit(m.version('deepseek-harness-sdk') != {sdk_version!r})"
    return f"/opt/hx-dsh/bin/python -c {shlex.quote(code)}"


class DeepSeekHarnessOptions(InstalledAgentOptions):
    profile: Annotated[str, Field(description="dsh profile; sdk-minimal is dsh's documented benchmark setup")] = "sdk-minimal"
    sdk_version: Annotated[str, Field(description="deepseek-harness-sdk version from PyPI")] = "0.2.1rc2"
    python_version: Annotated[str, Field(description="Interpreter uv installs for the SDK")] = "3.12"
    max_turns: Annotated[int | None, Field(description="Stop after this many model responses")] = None
    reasoning_effort: Annotated[str | None, Field(description="Adapter reasoning level (route-specific)")] = None
    max_tokens: Annotated[int | None, Field(description="Max output tokens per model call")] = None
    cache_retention: Annotated[str | None, Field(description="pi-ai prompt caching: none | short | long")] = None
    api_base_url: Annotated[str | None, Field(description="Override the provider endpoint (e.g. the mock API)")] = None
    declare_model: Annotated[bool, Field(description="Declare the model instead of using the bundled catalog")] = False
    context_window: Annotated[int, Field(description="Used only when the model is declared")] = 200_000
    max_output_tokens: Annotated[int, Field(description="Used only when the model is declared")] = 64_000
    upload_session_log: Annotated[bool, Field(description="Keep dsh's DeepSeek session-log upload on")] = False


class DeepSeekHarness(BaseInstalledAgent):
    capabilities = AgentCapabilities(atif=True, reasoning_effort=True)
    options_model = DeepSeekHarnessOptions

    @staticmethod
    @override
    def name() -> str:
        return "deepseek-harness"

    @classmethod
    @override
    def check_model_credentials(cls, connection, *, kwargs, env, explicit_env=None) -> None:
        options = kwargs or {}
        if options.get("api_base_url"):
            return  # the mock API needs no real key
        provider = (getattr(connection, "provider", None) or "anthropic").lower()
        key = PROVIDER_KEYS.get(provider)
        if key and not env.get(key):
            raise ValueError(f"deepseek-harness: set {key} (e.g. in .env) to run provider '{provider}'")

    @override
    def get_version_command(self) -> str | None:
        return "/opt/hx-dsh/bin/python -c \"import importlib.metadata as m; print(m.version('deepseek-harness-sdk'))\""

    @override
    def parse_version(self, stdout: str) -> str:
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        return lines[-1] if lines else stdout.strip()

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        o = self.options
        if (await environment.exec(command=installed_check(o.sdk_version), user="root")).return_code == 0:
            return  # pre-installed image
        await self.ensure_system_dependencies(environment, SYSTEM_PACKAGES)
        await self.exec_as_root(environment, command=install_script(o.sdk_version, o.python_version))

    def _provider_and_model(self) -> tuple[str, str]:
        name = self.model_name or ""
        if "/" in name:
            provider, model = name.split("/", 1)
            return provider, model
        return "anthropic", name

    @override
    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        self._instruction = instruction
        provider, model = self._provider_and_model()
        o = self.options
        key_env = PROVIDER_KEYS.get(provider, f"{provider.upper()}_API_KEY")
        env = {key_env: os.environ.get(key_env) or ("mock-key" if o.api_base_url else "")}
        # Uploaded per run (not in install) so a pre-baked image still gets the current runner.
        await environment.upload_file(RUNNER, REMOTE_RUNNER)
        await self._upload_config_text(environment, content=instruction, remote_path=REMOTE_INSTRUCTION, filename="instruction.md")
        await self.exec_as_root(environment, command=f"chmod a+rx {REMOTE_RUNNER} && chmod a+r {REMOTE_INSTRUCTION}")
        args = [
            "/opt/hx-dsh/bin/python", REMOTE_RUNNER,
            "--instruction-file", REMOTE_INSTRUCTION,
            "--provider", provider, "--model", model, "--api-key-env", key_env,
            "--profile", o.profile, "--dsh-home", "/tmp/hx-dsh-home", "--out-dir", "/logs/agent",
            "--context-window", str(o.context_window), "--max-output-tokens", str(o.max_output_tokens),
        ]
        if o.max_turns:
            args += ["--max-turns", str(o.max_turns)]
        if o.reasoning_effort:
            args += ["--reasoning-effort", o.reasoning_effort]
        if o.max_tokens:
            args += ["--max-tokens", str(o.max_tokens)]
        if o.cache_retention:
            args += ["--cache-retention", o.cache_retention]
        if o.api_base_url:
            args += ["--base-url", o.api_base_url]
        if o.declare_model:
            args.append("--declare-model")
        if o.upload_session_log:
            args.append("--upload-session-log")
        command = "set -o pipefail; " + " ".join(shlex.quote(a) for a in args) + " 2>&1 | tee /logs/agent/dsh-runner.txt"
        await self.exec_as_agent(environment, command=command, env=env)

    @override
    def convert_trajectory(self, logs_dir: Path):
        _, model = self._provider_and_model()
        return atif.convert(logs_dir, getattr(self, "_instruction", None), model, self.version())

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        try:
            trajectory = self.convert_trajectory(self.logs_dir)
        except Exception as exc:  # never fail the trial over accounting
            self.logger.warning(f"deepseek-harness: could not convert session events: {exc}")
            return
        if trajectory is None:
            return
        data: dict[str, Any] = trajectory.to_json_dict()
        (self.logs_dir / "trajectory.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
        fm = trajectory.final_metrics
        if fm:
            context.n_input_tokens = fm.total_prompt_tokens or 0
            context.n_cache_tokens = fm.total_cached_tokens or 0
            context.n_output_tokens = fm.total_completion_tokens or 0
        _, model = self._provider_and_model()
        total = 0.0
        for step in trajectory.steps:
            m = step.metrics
            if not m:
                continue
            extra = m.extra or {}
            write = int(extra.get("cache_creation_input_tokens") or 0)
            long_ttl = "ephemeral_1h_input_tokens" in (extra.get("cache_creation") or {})
            usage = Usage(
                input=(m.prompt_tokens or 0) - (m.cached_tokens or 0) - write,
                output=m.completion_tokens or 0,
                cache_read=m.cached_tokens or 0,
                cache_write_5m=0 if long_ttl else write,
                cache_write_1h=write if long_ttl else 0,
            )
            cost = call_cost(model, usage)
            if cost is None:
                return
            total += cost
        context.cost_usd = total
