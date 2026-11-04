"""A scripted stand-in for the Anthropic Messages API, for free offline tests.

Point a harness at it (ANTHROPIC_BASE_URL / base_url) and every model call gets
a deterministic reply: first a tool call that runs one harmless shell command,
then, once a tool result comes back, a final text answer. That exercises the
whole pipeline (Harbor -> container -> harness -> logs -> trajectory -> Langfuse)
without an API key or spend.

With --record DIR, every request body is saved, which shows exactly what a
harness sends: system prompt, tool schemas, caching markers.
"""

from __future__ import annotations

import itertools
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from aiohttp import web

# HX_MOCK_COMMAND swaps in another command, e.g. to probe what a harness's shell does to background processes.
MOCK_COMMAND = os.environ.get("HX_MOCK_COMMAND", "echo 'hx mock step' && pwd && ls | head -5")
_counter = itertools.count(1)


def _shell_tool(tools: list[dict[str, Any]]) -> tuple[str, dict[str, Any]] | None:
    """Pick the harness's shell tool and build a valid input for it."""
    def props(tool):
        return (tool.get("input_schema") or {}).get("properties") or {}

    candidates = [t for t in tools if t.get("name") in ("Bash", "bash")] or [t for t in tools if "command" in props(t)]
    if not candidates:
        return None
    tool = candidates[0]
    schema = tool.get("input_schema") or {}
    args: dict[str, Any] = {"command": MOCK_COMMAND}
    for key in schema.get("required") or []:
        if key in args:
            continue
        kind = (props(tool).get(key) or {}).get("type")
        args[key] = "mock step" if kind in (None, "string") else ({} if kind == "object" else 0)
    return tool["name"], args


def _last_user_has_tool_result(messages: list[dict[str, Any]]) -> bool:
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            return any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
        return False
    return False


def plan_reply(body: dict[str, Any]) -> list[dict[str, Any]]:
    messages = body.get("messages") or []
    tool = _shell_tool(body.get("tools") or [])
    if tool and not _last_user_has_tool_result(messages):
        name, args = tool
        return [
            {"type": "text", "text": "I'll inspect the workspace first."},
            {"type": "tool_use", "id": f"toolu_mock{uuid.uuid4().hex[:16]}", "name": name, "input": args},
        ]
    return [{"type": "text", "text": "Mock agent finished. (No real model was called.)"}]


def _usage(body: dict[str, Any], output_tokens: int) -> dict[str, int]:
    return {
        "input_tokens": max(len(json.dumps(body)) // 4, 1),
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


async def messages(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    record_dir: Path | None = request.app["record_dir"]
    n = next(_counter)
    if record_dir:
        (record_dir / f"{int(time.time() * 1000)}-{n:05d}.json").write_text(
            json.dumps({"path": request.path, "headers": {k: v for k, v in request.headers.items() if k.lower().startswith("anthropic-")}, "body": body}, indent=1)
        )
    blocks = plan_reply(body)
    stop_reason = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
    out_tokens = 25
    message = {
        "id": f"msg_mock{uuid.uuid4().hex[:20]}",
        "type": "message",
        "role": "assistant",
        "model": body.get("model", "mock"),
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": _usage(body, out_tokens),
    }
    if not body.get("stream"):
        return web.json_response(message)

    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
    await resp.prepare(request)
    start = dict(message, content=[], stop_reason=None, usage=dict(message["usage"], output_tokens=1))
    await resp.write(_sse("message_start", {"type": "message_start", "message": start}))
    for i, block in enumerate(blocks):
        if block["type"] == "text":
            await resp.write(_sse("content_block_start", {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}}))
            await resp.write(_sse("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": block["text"]}}))
        else:
            await resp.write(_sse("content_block_start", {"type": "content_block_start", "index": i, "content_block": {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}}))
            await resp.write(_sse("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}}))
        await resp.write(_sse("content_block_stop", {"type": "content_block_stop", "index": i}))
    await resp.write(_sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": {"output_tokens": out_tokens}}))
    await resp.write(_sse("message_stop", {"type": "message_stop"}))
    await resp.write_eof()
    return resp


async def count_tokens(request: web.Request) -> web.Response:
    body = await request.json()
    return web.json_response({"input_tokens": max(len(json.dumps(body)) // 4, 1)})


async def models(request: web.Request) -> web.Response:
    ids = ["claude-haiku-4-5", "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]
    data = [{"type": "model", "id": i, "display_name": i, "created_at": "2026-01-01T00:00:00Z"} for i in ids]
    return web.json_response({"data": data, "has_more": False, "first_id": ids[0], "last_id": ids[-1]})


async def fallback(request: web.Request) -> web.Response:
    path = request.path
    if request.method == "POST" and path.endswith("/v1/messages"):
        return await messages(request)
    if request.method == "POST" and path.endswith("/v1/messages/count_tokens"):
        return await count_tokens(request)
    if request.method == "GET" and path.rstrip("/").endswith("/v1/models"):
        return await models(request)
    return web.json_response({"type": "error", "error": {"type": "not_found_error", "message": f"mock: no route {request.method} {path}"}}, status=404)


def make_app(record_dir: Path | None = None) -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["record_dir"] = record_dir
    if record_dir:
        record_dir.mkdir(parents=True, exist_ok=True)
    app.router.add_route("*", "/{tail:.*}", fallback)
    return app


def serve(host: str = "127.0.0.1", port: int = 8789, record_dir: Path | None = None) -> None:
    web.run_app(make_app(record_dir), host=host, port=port, print=lambda *_: None)
