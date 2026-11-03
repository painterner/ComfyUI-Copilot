"""MCP server settings, local discovery and tolerant connection, against local MCP servers.

    python tests/test_mcp_servers.py      # needs openai-agents, fastapi, uvicorn
"""
import asyncio
import json
import os
import stat
import sys
import tempfile
import unittest
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.service import mcp_servers  # noqa: E402


def mcp_app(tools, broken=False):
    """A Streamable HTTP MCP server answering with plain JSON, like civitai-comfyui's /civitai-fill/mcp."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response
    app = FastAPI()

    @app.post("/mcp")
    async def mcp(request: Request):
        if broken:
            return Response(status_code=503)
        message = await request.json()
        if "id" not in message:
            return Response(status_code=202)
        method, params = message["method"], message.get("params", {})
        if method == "initialize":
            result = {"protocolVersion": params["protocolVersion"], "capabilities": {"tools": {}}, "serverInfo": {"name": "t", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": [{"name": name, "description": name, "inputSchema": {"type": "object"}} for name in tools]}
        elif method == "tools/call":
            result = {"content": [{"type": "text", "text": json.dumps({"tool": params["name"], "args": params.get("arguments")})}]}
        else:
            return JSONResponse({"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32601, "message": method}})
        return JSONResponse({"jsonrpc": "2.0", "id": message["id"], "result": result})
    return app


async def serve(app):
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.02)
    return server, task, server.servers[0].sockets[0].getsockname()[1]


class ConfigTest(unittest.TestCase):
    def test_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "sub", "mcp_servers.json")
            self.assertEqual(mcp_servers.load_config(path), {"builtin": True, "local": True, "servers": []})
            config = mcp_servers.validate_config({"builtin": False, "servers": [
                {"name": "tools", "url": "https://example.com/mcp", "headers": {"Authorization": "Bearer secret"}}]}, mcp_servers.DEFAULTS)
            self.assertEqual(config["servers"][0], {"name": "tools", "url": "https://example.com/mcp", "transport": "streamable_http",
                                                    "headers": {"Authorization": "Bearer secret"}, "timeout": 300, "enabled": True})
            mcp_servers.save_config(config, path)
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600, "headers can carry credentials")
            saved = mcp_servers.load_config(path)
            self.assertFalse(saved["builtin"])
            # A server sent back without headers (the API never returns them) keeps its headers.
            again = mcp_servers.validate_config({"servers": [{"name": "tools", "url": "https://example.com/mcp", "enabled": False}]}, saved)
            self.assertEqual(again["servers"][0]["headers"], {"Authorization": "Bearer secret"})
            self.assertFalse(again["servers"][0]["enabled"])
            os.environ["COPILOT_MCP_CONFIG"] = path
            try:
                public = mcp_servers.public_config(saved, [])
            finally:
                del os.environ["COPILOT_MCP_CONFIG"]
            self.assertNotIn("secret", json.dumps(public))
            self.assertEqual(public["servers"][0]["header_names"], ["Authorization"])
            for body, fragment in [({"other": 1}, "Unknown"), ({"local": "yes"}, "true or false"),
                                   ({"servers": [{"name": "a b", "url": "https://x"}]}, "names"),
                                   ({"servers": [{"name": "a", "url": "ftp://x"}]}, "http"),
                                   ({"servers": [{"name": "a", "url": "https://x", "transport": "ws"}]}, "transport"),
                                   ({"servers": [{"name": "a", "url": "https://x"}, {"name": "a", "url": "https://y"}]}, "unique")]:
                with self.assertRaises(ValueError) as caught:
                    mcp_servers.validate_config(body, mcp_servers.DEFAULTS)
                self.assertIn(fragment, str(caught.exception))

    def test_local_endpoints_and_specs(self):
        routes = [SimpleNamespace(method="POST", path="/civitai-fill/mcp"), SimpleNamespace(method="GET", path="/civitai-fill/mcp"),
                  SimpleNamespace(method="POST", path="/api/copilot/mcp"), SimpleNamespace(method="POST", path="/x/{id}/mcp"),
                  SimpleNamespace(method="POST", path="/civitai-fill/inspect"), SimpleNamespace(method="POST", path="/a/b/mcp")]
        local = mcp_servers.local_endpoints(routes, "0.0.0.0", 8188)
        self.assertEqual([(e["name"], e["url"]) for e in local], [("civitai-fill", "http://127.0.0.1:8188/civitai-fill/mcp"),
                                                                 ("a-b", "http://127.0.0.1:8188/a/b/mcp")])
        self.assertEqual(mcp_servers.local_endpoints(routes, "::1", 8188, "https")[0]["url"], "https://[::1]:8188/civitai-fill/mcp")
        self.assertEqual(mcp_servers.local_endpoints(routes, None, None), [])
        custom = {"name": "mine", "url": "https://m/mcp", "transport": "sse", "headers": {}, "timeout": 30, "enabled": True}
        specs = mcp_servers.server_specs({"builtin": True, "local": True, "servers": [custom, {**custom, "name": "off", "enabled": False}]}, "s1", "k", local)
        self.assertEqual([s["name"] for s in specs], ["copilot", "bing", "civitai-fill", "a-b", "mine"])
        self.assertEqual(specs[0]["headers"], {"X-Session-Id": "s1", "Authorization": "Bearer k"})
        self.assertEqual([s["name"] for s in mcp_servers.server_specs({"builtin": False, "local": False, "servers": []}, "s", None, local)], [])


class ConnectTest(unittest.IsolatedAsyncioTestCase):
    async def test_connect_skips_failures_and_hides_duplicate_tools(self):
        from agents import Agent, RunContextWrapper
        running = [await serve(mcp_app(["civitai_inspect", "get_current_workflow"])),
                   await serve(mcp_app(["civitai_inspect", "other"])),
                   await serve(mcp_app([], broken=True))]
        try:
            a, b, broken = (port for _, _, port in running)
            spec = lambda name, url, transport="streamable_http": {"name": name, "url": url, "transport": transport, "headers": {}, "timeout": 10}
            specs = [spec("a", f"http://127.0.0.1:{a}/mcp"), spec("refused", "http://127.0.0.1:9/mcp"),
                     spec("broken", f"http://127.0.0.1:{broken}/mcp"), spec("sse-down", f"http://127.0.0.1:{broken}/sse", "sse"),
                     spec("b", f"http://127.0.0.1:{b}/mcp")]
            async with AsyncExitStack() as stack:
                connected = await mcp_servers.connect_servers(stack, specs, reserved={"get_current_workflow"})
                self.assertEqual([name for name, _ in connected], ["a", "b"])
                context, agent = RunContextWrapper(context=None), Agent(name="t")
                tools = {name: [t.name for t in await server.list_tools(context, agent)] for name, server in connected}
                self.assertEqual(tools, {"a": ["civitai_inspect"], "b": ["other"]})
                result = await connected[0][1].call_tool("civitai_inspect", {"url": "u"})
                self.assertEqual(json.loads(result.content[0].text), {"tool": "civitai_inspect", "args": {"url": "u"}})
        finally:
            for server, task, _ in running:
                server.should_exit = True
                await task


if __name__ == "__main__":
    unittest.main()
