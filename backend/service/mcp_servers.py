# Copyright (C) 2025 AIDC-AI
# Licensed under the MIT License.
"""MCP servers the Copilot agent connects to.

Upstream hard-coded its hosted Copilot MCP server and a ModelScope Bing search, and one unreachable
server failed every chat. Here:

- ``builtin`` keeps those two upstream servers (on by default);
- ``local`` adds the MCP endpoints other custom nodes serve on this ComfyUI (POST routes ending in
  ``/mcp``, e.g. civitai-comfyui's ``/civitai-fill/mcp``), so they work without configuration;
- ``servers`` lists any other MCP server (Streamable HTTP or SSE).

The settings live in ``<ComfyUI user dir>/default/comfyui-copilot/mcp_servers.json`` (or
``COPILOT_MCP_CONFIG``) and are edited through ``GET/PUT /api/copilot/mcp-servers``. A server that
fails to connect is skipped, and a tool whose name is already taken is hidden from later servers.
"""
import asyncio
import json
import os
import re
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, Tuple

from ..utils.globals import BACKEND_BASE_URL
from ..utils.logger import log

BING_SSE_URL = "https://mcp.api-inference.modelscope.net/8c9fe550938e4f/sse"
TRANSPORTS = ("streamable_http", "sse")
NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
DEFAULTS = {"builtin": True, "local": True, "servers": []}


def config_path() -> str:
    if os.getenv("COPILOT_MCP_CONFIG"):
        return os.getenv("COPILOT_MCP_CONFIG")
    import folder_paths
    return os.path.join(folder_paths.get_user_directory(), "default", "comfyui-copilot", "mcp_servers.json")


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    try:
        with open(path or config_path(), encoding="utf-8") as file:
            return {**DEFAULTS, **json.load(file)}
    except FileNotFoundError:
        return dict(DEFAULTS)


def validate_config(config: Dict[str, Any], previous: Dict[str, Any]) -> Dict[str, Any]:
    """Check a PUT body. A server sent without ``headers`` keeps the headers it had (they are secrets
    and are never returned)."""
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
    result = {key: config.get(key, previous.get(key, DEFAULTS[key])) for key in DEFAULTS}
    for key in ("builtin", "local"):
        if not isinstance(result[key], bool):
            raise ValueError(f"{key} must be true or false")
    if not isinstance(result["servers"], list):
        raise ValueError("servers must be a list")
    old_headers = {s.get("name"): s.get("headers", {}) for s in previous.get("servers", [])}
    servers, names = [], set()
    for entry in result["servers"]:
        if not isinstance(entry, dict):
            raise ValueError("Each server must be an object")
        name, url = entry.get("name"), entry.get("url")
        if not isinstance(name, str) or not NAME.match(name) or name in names:
            raise ValueError(f"Server names must be unique, 1-64 letters, digits, '_', '.' or '-': {name!r}")
        if not isinstance(url, str) or not re.match(r"^https?://", url):
            raise ValueError(f"{name}: url must start with http:// or https://")
        transport = entry.get("transport", "streamable_http")
        if transport not in TRANSPORTS:
            raise ValueError(f"{name}: transport must be one of {', '.join(TRANSPORTS)}")
        headers = entry.get("headers", old_headers.get(name, {}))
        if not isinstance(headers, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in headers.items()):
            raise ValueError(f"{name}: headers must map names to strings")
        timeout = entry.get("timeout", 300)
        if not isinstance(timeout, (int, float)) or not 1 <= timeout <= 3600:
            raise ValueError(f"{name}: timeout must be 1-3600 seconds")
        names.add(name)
        servers.append({"name": name, "url": url, "transport": transport, "headers": headers,
                        "timeout": timeout, "enabled": entry.get("enabled", True) is not False})
    result["servers"] = servers
    return result


def save_config(config: Dict[str, Any], path: Optional[str] = None) -> None:
    path = path or config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    # Headers may carry credentials: keep the file private to the user.
    with open(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def public_config(config: Dict[str, Any], local: List[Dict[str, Any]]) -> Dict[str, Any]:
    servers = [{**{k: v for k, v in s.items() if k != "headers"}, "header_names": sorted(s.get("headers", {}))}
               for s in config["servers"]]
    return {"builtin": config["builtin"], "local": config["local"], "servers": servers, "discovered": local,
            "path": config_path()}


def local_endpoints(routes, address: Optional[str], port: Optional[int], scheme: str = "http") -> List[Dict[str, Any]]:
    """MCP endpoints other custom nodes registered on this ComfyUI (POST ``/<prefix>/mcp``)."""
    if not port:
        return []
    host = address if address and address not in ("0.0.0.0", "::") else "127.0.0.1"
    host = f"[{host}]" if ":" in host else host
    endpoints, seen = [], set()
    for route in routes:
        path, method = getattr(route, "path", ""), getattr(route, "method", "")
        if method != "POST" or not path.endswith("/mcp") or path.startswith("/api/copilot") or path in seen:
            continue
        prefix = path[: -len("/mcp")].strip("/")
        if not prefix or "{" in prefix:
            continue
        seen.add(path)
        endpoints.append({"name": prefix.replace("/", "-"), "url": f"{scheme}://{host}:{port}{path}",
                          "transport": "streamable_http", "headers": {}, "timeout": 300, "enabled": True})
    return endpoints


def comfy_local_endpoints() -> List[Dict[str, Any]]:
    import server
    from comfy.cli_args import args
    instance = server.PromptServer.instance
    return local_endpoints(instance.routes, getattr(instance, "address", None), getattr(instance, "port", None),
                           "https" if args.tls_keyfile and args.tls_certfile else "http")


def server_specs(config: Dict[str, Any], session_id: str, api_key: Optional[str], local: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    specs = []
    if config["builtin"]:
        headers = {"X-Session-Id": session_id, "Authorization": f"Bearer {api_key}"}
        specs.append({"name": "copilot", "url": BACKEND_BASE_URL + "/mcp-server/mcp", "transport": "sse", "headers": headers, "timeout": 300})
        specs.append({"name": "bing", "url": BING_SSE_URL, "transport": "sse", "headers": headers, "timeout": 300})
    if config["local"]:
        specs.extend(local)
    specs.extend(s for s in config["servers"] if s.get("enabled", True))
    return specs


def create_server(spec: Dict[str, Any]):
    from agents.mcp import MCPServerSse, MCPServerStreamableHttp
    params = {"url": spec["url"], "headers": spec.get("headers", {}), "timeout": float(spec.get("timeout", 300))}
    cls = MCPServerSse if spec["transport"] == "sse" else MCPServerStreamableHttp
    return cls(params=params, cache_tools_list=True, name=spec["name"], client_session_timeout_seconds=float(spec.get("timeout", 300)))


async def _hold(server, ready: asyncio.Future, close: asyncio.Event) -> None:
    """Keep one server connected inside its own task. The MCP clients' anyio cancel scopes must be
    entered and exited in one task, and a failed connect cancels that task's scope; in the caller's
    task it would cancel the whole chat (upstream's failure with an unreachable hosted server)."""
    try:
        async with server:
            ready.set_result({tool.name for tool in await server.list_tools()})
            await close.wait()
    except BaseException as error:
        if not ready.done():
            ready.set_result(error)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise


async def _release(task: asyncio.Task, close: asyncio.Event) -> None:
    close.set()
    await asyncio.wait({task}, timeout=10)
    task.cancel()


async def connect_servers(stack: AsyncExitStack, specs: List[Dict[str, Any]], reserved=(), create=create_server,
                          connect_timeout: float = 30) -> List[Tuple[str, Any]]:
    """Connect the servers concurrently, closed with ``stack``. Skip ones that fail or do not answer in
    ``connect_timeout`` seconds, and hide tool names an earlier server (or ``reserved``) already has."""
    holders = []
    for spec in specs:
        server, ready, close = create(spec), asyncio.get_running_loop().create_future(), asyncio.Event()
        task = asyncio.create_task(_hold(server, ready, close))
        stack.push_async_callback(_release, task, close)
        holders.append((spec, server, ready, task))
    if holders:
        await asyncio.wait([ready for _, _, ready, _ in holders], timeout=connect_timeout)
    connected, taken = [], set(reserved)
    for spec, server, ready, task in holders:
        names = ready.result() if ready.done() else TimeoutError(f"no answer in {connect_timeout}s")
        if isinstance(names, BaseException):
            task.cancel()
            log.warning(f"[MCP] Skipping server {spec['name']}: {type(names).__name__}: {names}")
            continue
        duplicates = names & taken
        if duplicates:
            log.warning(f"[MCP] {spec['name']}: hiding tools already provided elsewhere: {sorted(duplicates)}")
            server.tool_filter = {"blocked_tool_names": sorted(duplicates)}
        taken |= names
        connected.append((spec["name"], server))
    return connected
