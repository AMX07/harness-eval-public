import json

from harness_eval import stats
from harness_eval.mock_api import plan_reply
from harness_eval.pricing import Usage, call_cost, normalize_model
from harness_eval.results import usage_from_metrics


def test_normalize_model():
    assert normalize_model("anthropic/claude-haiku-4-5-20251001") == "claude-haiku-4-5"
    assert normalize_model("claude-haiku-5-5") == "claude-haiku-5-5"


def test_haiku45_cost():
    u = Usage(input=1_000_000, output=1_000_000, cache_read=1_000_000, cache_write_5m=1_000_000, cache_write_1h=1_000_000)
    assert abs(call_cost("claude-haiku-4-5", u) - (1 + 5 + 0.1 + 1.25 + 2)) < 1e-9


def test_haiku55_long_context_tier():
    short = Usage(input=100_000)
    long = Usage(input=100_001)
    assert abs(call_cost("claude-haiku-5-5", short) - 0.01) < 1e-9
    assert abs(call_cost("claude-haiku-5-5", long) - 100_001 * 0.50 / 1e6) < 1e-9


def test_unknown_model_is_unpriced():
    assert call_cost("glm-5.2", Usage(input=10)) is None


def test_usage_from_claude_code_metrics():
    # Harbor's Claude Code converter: prompt = input + cache read + cache creation
    m = {"prompt_tokens": 1500, "completion_tokens": 40, "cached_tokens": 1000,
         "extra": {"cache_creation_input_tokens": 300, "cache_creation": {"ephemeral_5m_input_tokens": 300}}}
    assert usage_from_metrics(m) == Usage(input=200, output=40, cache_read=1000, cache_write_5m=300, cache_write_1h=0)


def test_mock_replies_with_tool_then_text():
    tools = [{"name": "Bash", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}, "description": {"type": "string"}}, "required": ["command"]}}]
    first = plan_reply({"tools": tools, "messages": [{"role": "user", "content": "do it"}]})
    assert first[-1]["type"] == "tool_use" and first[-1]["name"] == "Bash"
    second = plan_reply({"tools": tools, "messages": [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": first},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": first[-1]["id"], "content": "ok"}]},
    ]})
    assert [b["type"] for b in second] == ["text"]


def test_bootstrap_is_deterministic():
    a = stats.bootstrap([0, 1, 1, 0, 1])
    b = stats.bootstrap([0, 1, 1, 0, 1])
    assert a == b and a.lo <= a.mean <= a.hi
    json.dumps(a.__dict__)


def test_prebaked_image_name_is_a_valid_docker_reference():
    import re

    from harness_eval.prebake import image_name

    name = image_name("alexgshaw/fix-git:20251031", "claude-code-2.1.316-1a2b3c4d")
    assert name == "hx-prebaked/alexgshaw.fix-git:20251031-claude-code-2.1.316-1a2b3c4d"
    component = r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
    assert re.fullmatch(rf"{component}(?:/{component})*:[\w][\w.-]{{0,127}}", name)


def test_prebake_key_changes_with_the_harness_version():
    from harness_eval.prebake import Recipe, claude_code_install

    a = Recipe("claude-code", "2.1.316", claude_code_install("2.1.316"), as_root=False)
    b = Recipe("claude-code", "2.1.317", claude_code_install("2.1.317"), as_root=False)
    assert a.key.startswith("claude-code-2.1.316-") and a.key != b.key


def test_both_harnesses_get_claude_codes_system_packages():
    from harness_eval.prebake import packages_script

    script = packages_script()
    for package in ("curl", "bash", "nodejs", "npm", "procps"):
        assert f" {package}" in script
    assert "apt-get update && apt-get install -y $pkgs" in script


def test_resource_limit_multiplier(tmp_path):
    import json as _json

    from harbor.environments.docker import write_resources_compose_file
    from harbor.models.task.config import EnvironmentConfig as TaskEnv
    from harbor.models.trial.config import ResourceMode

    from harness_eval import experiment as ex
    from harness_eval.environments.prebaked import PrebakedDocker, scaled_limit

    # Config: the multiplier reaches PrebakedDocker's kwargs, with or without prebake.
    exp = ex.Experiment.model_validate({
        "name": "t", "tasks": {"names": ["fix-git"]},
        "harness": {"agent": "claude-code", "model": "anthropic/claude-haiku-5-5"},
        "environment": {"resource_limit_multiplier": 3},
    })
    for key in (None, "abc"):
        env = ex.harbor_job_config(exp, "j", prebake_key=key, jobs_dir=tmp_path)["environment"]
        assert env["import_path"] == ex.PREBAKED_ENVIRONMENT
        assert env["kwargs"]["resource_limit_multiplier"] == 3
    assert "import_path" not in ex.harbor_job_config(exp.model_copy(update={"environment": ex.Environment()}), "j", jobs_dir=tmp_path)["environment"]

    # Limits: CPU is capped at the host's count (Docker rejects more), memory is not.
    assert scaled_limit("cpu", 1, 3.0, 4) == 3.0 and scaled_limit("cpu", 2, 3.0, 4) == 4.0
    assert scaled_limit("memory", 2048, 3.0, 4) == 6144 and scaled_limit("memory", 2048, 1.0, 4) == 2048

    # Compose override: request = 1x task spec, hard limit = 3x.
    env = object.__new__(PrebakedDocker)
    env._limit_multiplier = 3.0
    env._cpu_resource_mode = env._memory_resource_mode = ResourceMode.AUTO
    env.task_env_config = TaskEnv(cpus=1, memory_mb=2048)
    path = write_resources_compose_file(
        tmp_path / "r.json",
        cpu_request=env._resource_request_value("cpu", auto_mode=ResourceMode.LIMIT),
        cpu_limit=env._resource_limit_value("cpu", auto_mode=ResourceMode.LIMIT),
        memory_request_mb=env._resource_request_value("memory", auto_mode=ResourceMode.LIMIT),
        memory_limit_mb=env._resource_limit_value("memory", auto_mode=ResourceMode.LIMIT),
    )
    main = _json.loads(path.read_text())["services"]["main"]
    assert main["cpus"] == 3.0 and main["mem_limit"] == "6144m"
    assert main["deploy"]["resources"]["reservations"] == {"cpus": "1", "memory": "2048M"}
