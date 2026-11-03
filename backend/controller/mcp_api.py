# Copyright (C) 2025 AIDC-AI
# Licensed under the MIT License.
"""Settings for the MCP servers the Copilot agent uses (see backend/service/mcp_servers.py)."""
import json

from aiohttp import web
from server import PromptServer

from ..service.mcp_servers import comfy_local_endpoints, load_config, public_config, save_config, validate_config


@PromptServer.instance.routes.get("/api/copilot/mcp-servers")
async def get_mcp_servers(request):
    return web.json_response(public_config(load_config(), comfy_local_endpoints()))


@PromptServer.instance.routes.put("/api/copilot/mcp-servers")
async def put_mcp_servers(request):
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("Expected a JSON object")
        config = validate_config(body, load_config())
    except (ValueError, json.JSONDecodeError) as error:
        return web.json_response({"error": str(error)}, status=400)
    save_config(config)
    return web.json_response(public_config(config, comfy_local_endpoints()))
