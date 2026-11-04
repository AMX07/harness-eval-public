"""Harbor's Docker environment, started from the task's pre-installed image (see prebake.py).

`hx run` uses it when `environment.prebake` is on:
    environment.import_path: harness_eval.environments.prebaked:PrebakedDocker
    environment.kwargs.prebake_key: <recipe key>

It also adds resource headroom (`environment.resource_limit_multiplier`), as in
Anthropic's infrastructure-noise study: the task's own cpus/memory become the
guaranteed allocation (compose `deploy.resources.reservations`), and the hard
kill limit (`cpus`, `mem_limit`) is that times the multiplier. Plain Harbor
Docker turns the task spec into a hard limit at 1x.
"""

from __future__ import annotations

import os

from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.trial.config import ResourceMode

from ..prebake import image_exists, image_name


def scaled_limit(resource: str, value: int | float | None, multiplier: float, host_cpus: int | None) -> int | float | None:
    """Hard limit = task spec x multiplier. Docker rejects `cpus` above the host's CPU count, so cap there."""
    if value is None or multiplier == 1.0:
        return value
    if resource == "cpu":
        limit = float(value) * multiplier
        return min(limit, float(host_cpus)) if host_cpus else limit
    return int(round(value * multiplier))


class PrebakedDocker(DockerEnvironment):
    def __init__(self, *args, prebake_key: str | None = None, resource_limit_multiplier: float = 1.0, **kwargs):
        if resource_limit_multiplier < 1.0:
            raise ValueError("resource_limit_multiplier must be >= 1")
        self._limit_multiplier = float(resource_limit_multiplier)
        config = kwargs.get("task_env_config")
        if prebake_key and config is not None and config.docker_image:
            image = image_name(config.docker_image, prebake_key)
            if not image_exists(image):
                raise RuntimeError(f"Pre-installed image {image} not found; build it with `hx prebake <experiment>`")
            kwargs["task_env_config"] = config.model_copy(update={"docker_image": image})
        super().__init__(*args, **kwargs)

    def _resource_request_value(self, resource, *, auto_mode):
        if self._limit_multiplier == 1.0:
            return super()._resource_request_value(resource, auto_mode=auto_mode)
        # Guaranteed allocation = the task's own spec (unless the policy ignores this resource).
        return self._resource_value(resource) if self._resource_mode(resource) != ResourceMode.IGNORE else None

    def _resource_limit_value(self, resource, *, auto_mode):
        if self._limit_multiplier == 1.0:
            return super()._resource_limit_value(resource, auto_mode=auto_mode)
        if self._resource_mode(resource) == ResourceMode.IGNORE:
            return None
        return scaled_limit(resource, self._resource_value(resource), self._limit_multiplier, os.cpu_count())
