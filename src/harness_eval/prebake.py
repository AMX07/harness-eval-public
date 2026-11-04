"""Harness installs baked into the task images, so trials skip them.

Harbor installs the harness inside every trial container. Under amd64
emulation on Apple Silicon that takes 3-6 minutes, and with several trials
installing at once it can pass Harbor's 6-minute agent-setup limit, which
fails the trial before the agent starts. So `hx run` first builds one image
per task and harness, FROM the task's own image, running what the trial would
run: the system packages (Harbor's package table and commands), then the
harness install as the same user. In the trial the harness finds itself
installed and skips straight to the agent. The image tag hashes the recipe,
so a new version builds new images instead of reusing stale ones.

Every harness gets the same system packages, the set Harbor's Claude Code
installer adds, so a comparison differs only in the harness.
"""

from __future__ import annotations

import hashlib
import re
import shlex
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent

from . import experiment as ex
from .tasks import load_tasks

SYSTEM_PACKAGES = ("curl", "bash", "nodejs", "npm", "procps")
PLATFORM = "linux/amd64"  # every Terminal-Bench image
REPO = "hx-prebaked"
DSH_AGENT = "harness_eval.agents.deepseek_harness:DeepSeekHarness"


@dataclass(frozen=True)
class Recipe:
    harness: str  # short name used in image tags
    version: str
    install: str  # bash script, run after the system packages
    as_root: bool  # else as the image's default user, like Harbor's exec_as_agent

    @property
    def key(self) -> str:
        digest = hashlib.sha256("\n".join([packages_script(), self.install, str(self.as_root)]).encode()).hexdigest()[:8]
        return f"{self.harness}-{self.version}-{digest}"


def packages_script(dependencies: tuple[str, ...] = SYSTEM_PACKAGES) -> str:
    """Harbor's ensure_system_dependencies as a shell script: install the packages whose commands are missing."""
    lines = [
        "set -e",
        'manager=""',
        'for m in apt-get dnf yum apk; do if command -v "$m" >/dev/null 2>&1; then manager="$m"; break; fi; done',
        'pkgs=""',
    ]
    for dep in dependencies:
        spec = BaseInstalledAgent.SYSTEM_PACKAGES[dep]
        add = 'case "$manager" in ' + " ".join(f'{m}) pkgs="$pkgs {" ".join(p)}";;' for m, p in spec.packages.items()) + " esac"
        if spec.always_install or not spec.commands:
            lines.append(add)
        else:
            check = " && ".join(f"command -v {shlex.quote(c)} >/dev/null 2>&1" for c in spec.commands)
            lines.append(f"if ! {{ {check}; }}; then {add}; fi")
    lines += [
        '[ -n "$pkgs" ] || exit 0',
        'case "$manager" in',
        "  apt-get) export DEBIAN_FRONTEND=noninteractive; apt-get update && apt-get install -y $pkgs;;",
        "  dnf) dnf install -y $pkgs;;",
        "  yum) yum install -y $pkgs;;",
        "  apk) apk add --no-cache $pkgs;;",
        '  *) echo "hx prebake: no supported package manager for:$pkgs" >&2;;',  # Harbor warns and carries on
        "esac",
    ]
    return "\n".join(lines) + "\n"


def claude_code_install(version: str) -> str:
    """Harbor's Claude Code install command (agents/installed/claude_code.py), pinned to `version`."""
    v = shlex.quote(version)
    return (
        "set -euo pipefail\n"
        "if command -v apk &> /dev/null; then\n"
        f"  npm install -g @anthropic-ai/claude-code@{v}\n"
        "else\n"
        f"  curl -fsSL https://downloads.claude.ai/claude-code-releases/bootstrap.sh | bash -s -- {v}\n"
        "fi\n"
        "echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> ~/.bashrc\n"
        'export PATH="$HOME/.local/bin:$PATH"\n'
        "claude --version\n"
    )


def recipe_for(exp: ex.Experiment) -> Recipe:
    agent, kwargs = exp.harness.agent, exp.harness.kwargs
    if agent == "claude-code":
        version = kwargs.get("version")
        if not version:
            raise SystemExit(f"{exp.name}: pin harness.kwargs.version (e.g. \"2.1.316\") so Claude Code can be pre-installed, or set environment.prebake: false")
        return Recipe("claude-code", str(version), claude_code_install(str(version)), as_root=False)
    if agent == DSH_AGENT:
        from .agents.deepseek_harness import DeepSeekHarnessOptions, install_script

        fields = DeepSeekHarnessOptions.model_fields
        sdk = str(kwargs.get("sdk_version", fields["sdk_version"].default))
        python = str(kwargs.get("python_version", fields["python_version"].default))
        return Recipe("dsh", sdk, install_script(sdk, python), as_root=True)
    raise SystemExit(f"{exp.name}: no pre-install recipe for agent '{agent}'; add one to prebake.recipe_for or set environment.prebake: false")


def image_name(base: str, key: str) -> str:
    """hx-prebaked/<base repo>:<base tag>-<key>, e.g. hx-prebaked/alexgshaw.fix-git:20251031-claude-code-2.1.316-1a2b3c4d."""
    ref, _, digest = base.partition("@")
    repo, tag = ref, "latest"
    if ":" in ref.rsplit("/", 1)[-1]:
        repo, tag = ref.rsplit(":", 1)
    if digest:
        tag = digest.split(":")[-1][:12]
    slug = re.sub(r"[^a-z0-9._-]+", ".", repo.lower().replace("/", ".")).strip("._-")
    return f"{REPO}/{slug}:{f'{tag}-{key}'[:128]}"


def image_exists(image: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0


def _docker(*args: str) -> str:
    proc = subprocess.run(["docker", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args[:2])} failed:\n{(proc.stderr or proc.stdout)[-3000:]}")
    return proc.stdout


def build(base: str, recipe: Recipe) -> str:
    image = image_name(base, recipe.key)
    if not image_exists(base):
        _docker("pull", "--platform", PLATFORM, base)
    user = _docker("image", "inspect", "--format", "{{.Config.User}}", base).strip() or "root"
    with tempfile.TemporaryDirectory(prefix="hx-prebake-") as tmp:
        ctx = Path(tmp)
        (ctx / "packages.sh").write_text(packages_script())
        (ctx / "install.sh").write_text(recipe.install)
        (ctx / "Dockerfile").write_text(
            f"FROM {base}\n"
            "USER root\n"
            "COPY packages.sh install.sh /tmp/hx-prebake/\n"
            "RUN sh /tmp/hx-prebake/packages.sh\n"
            f"USER {'root' if recipe.as_root else user}\n"
            "RUN bash /tmp/hx-prebake/install.sh\n"
            "USER root\n"
            "RUN rm -rf /tmp/hx-prebake\n"
            f"USER {user}\n"
            f'LABEL hx.prebake.base="{base}" hx.prebake.key="{recipe.key}"\n'
        )
        _docker("build", "--platform", PLATFORM, "-t", image, str(ctx))
    return image


def ensure(exp: ex.Experiment, task_names: list[str] | None, jobs: int = 4, log=print) -> Recipe:
    """Build the pre-installed images the run needs that don't exist yet; returns the recipe."""
    recipe = recipe_for(exp)
    catalog = load_tasks(exp.dataset)
    names = task_names if task_names is not None else sorted(catalog)
    unknown = [n for n in names if n not in catalog]
    if unknown:
        raise SystemExit(f"Unknown task(s) in {exp.dataset}: {', '.join(unknown)}")
    # A task without docker_image is built from its Dockerfile by Harbor; PrebakedDocker leaves it alone.
    bases = sorted({catalog[n].docker_image for n in names if catalog[n].docker_image})
    missing = [b for b in bases if not image_exists(image_name(b, recipe.key))]
    if not missing:
        return recipe
    log(f"[hx] pre-installing {recipe.harness} {recipe.version} into {len(missing)} task image(s), {jobs} at a time; later runs reuse them")
    errors = []
    with ThreadPoolExecutor(jobs) as pool:
        futures = {pool.submit(build, b, recipe): b for b in missing}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                log(f"[hx]   {done}/{len(missing)} {future.result()}")
            except Exception as exc:
                errors.append(f"{futures[future]}: {exc}")
                log(f"[hx]   {done}/{len(missing)} FAILED {futures[future]}")
    if errors:
        raise SystemExit("Pre-install failed for:\n" + "\n\n".join(errors))
    return recipe
