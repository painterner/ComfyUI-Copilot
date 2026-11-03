# MCP servers

The chat agent's MCP servers are configurable. Upstream hard-coded its hosted Copilot MCP server and a
ModelScope Bing search, and a single unreachable server failed every chat.

| Source | Default | What |
|---|---|---|
| `builtin` | on | Upstream's hosted Copilot MCP server and Bing search |
| `local` | on | MCP endpoints other custom nodes serve on this ComfyUI: any `POST /<prefix>/mcp` route (Streamable HTTP). For example, [civitai-comfyui](https://github.com/painterner/civitai-comfyui) serves `/civitai-fill/mcp`, so its `civitai_*` tools are available in the chat with no setup |
| `servers` | none | Any other MCP server, Streamable HTTP or SSE |

The servers connect in parallel when a chat starts. One that fails or does not answer within 30 seconds is
skipped and logged (`[MCP] Skipping server …`). When two servers offer the same tool name, the later one's
tool is hidden; Copilot's own `get_current_workflow` always wins.

## Settings

Stored in `ComfyUI/user/default/comfyui-copilot/mcp_servers.json` (mode 600; set `COPILOT_MCP_CONFIG` to use
another file):

```json
{
  "builtin": false,
  "local": true,
  "servers": [
    {"name": "my-tools", "url": "https://example.com/mcp", "transport": "streamable_http",
     "headers": {"Authorization": "Bearer …"}, "timeout": 300, "enabled": true}
  ]
}
```

The same settings through the API (header values are never returned; a server sent without `headers` keeps
its saved ones):

```bash
curl http://127.0.0.1:8188/api/copilot/mcp-servers            # settings + discovered local endpoints
curl -X PUT http://127.0.0.1:8188/api/copilot/mcp-servers \
  -H 'Content-Type: application/json' -d '{"builtin": false}'
```

## Tests

```bash
python tests/test_mcp_servers.py      # settings, discovery, failures skipped, duplicate tools hidden
# In a running ComfyUI with civitai-comfyui: discovery + a chat (local fake model) calling a plugin tool
python tests/chat_mcp_check.py --comfy http://127.0.0.1:8188 --tool civitai_list_rules --expect generic
```
