"""End-to-end check in a running ComfyUI: Copilot discovers a plugin's MCP endpoint, and a chat calls
one of its tools. A local fake model (OpenAI-compatible, streaming) asks for the tool, then echoes
the tool result, so no paid model is used.

    python tests/chat_mcp_check.py --comfy http://127.0.0.1:8188 --server civitai-fill \\
        --tool civitai_list_rules --expect generic
"""
import argparse
import asyncio
import json
import sys
import uuid

from aiohttp import ClientSession, ClientTimeout, web


def chunk(delta, finish=None):
    return {"id": "fake", "object": "chat.completion.chunk", "created": 0, "model": "fake-model",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def fake_model(tool, seen):
    async def completions(request):
        body = await request.json()
        seen.append(body)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        last = body["messages"][-1]
        if last["role"] == "tool":
            content = last["content"] if isinstance(last["content"], str) else json.dumps(last["content"], ensure_ascii=False)
            events = [chunk({"role": "assistant", "content": "工具结果：" + content[:2000]}), chunk({}, "stop")]
        else:
            call = {"index": 0, "id": "call_1", "type": "function", "function": {"name": tool, "arguments": "{}"}}
            events = [chunk({"role": "assistant", "tool_calls": [call]}), chunk({}, "tool_calls")]
        usage = {"id": "fake", "object": "chat.completion.chunk", "created": 0, "model": "fake-model", "choices": [],
                 "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        for event in [*events, usage]:
            await response.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        return response
    return completions


async def main(options):
    seen = []
    app = web.Application()
    app.router.add_post("/v1/chat/completions", fake_model(options.tool, seen))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    model_url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1"
    comfy = options.comfy.rstrip("/")
    try:
        async with ClientSession(timeout=ClientTimeout(total=300)) as http:
            async with http.get(comfy + "/api/copilot/mcp-servers") as response:
                settings = await response.json()
            discovered = [server["name"] for server in settings["discovered"]]
            assert options.server in discovered, f"{options.server} not discovered: {discovered}"
            session_id = str(uuid.uuid4())
            body = {"session_id": session_id, "prompt": "check", "messages": [{"role": "user", "content": f"Use {options.tool}"}],
                    "ext": [{"type": "model_select", "data": ["fake-model"]}]}
            headers = {"Openai-Base-Url": model_url, "Openai-Api-Key": "sk-fake", "Accept-Language": "zh-CN"}
            text = ""
            async with http.post(comfy + "/api/chat/invoke", json=body, headers=headers) as response:
                async for line in response.content:
                    line = line.decode().strip()
                    if line:
                        text = json.loads(line).get("text") or text
        tools = [t["function"]["name"] for t in seen[0].get("tools", [])] if seen else []
        report = {"discovered": discovered, "model_requests": len(seen), "tool_offered": options.tool in tools,
                  "tools": tools, "answer": text[:300]}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        assert options.tool in tools, "the plugin tool was not offered to the model"
        assert len(tools) == len(set(tools)), "duplicate tool names"
        assert len(seen) >= 2 and options.expect in text, "the tool result did not reach the answer"
        print("PASS")
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy", default="http://127.0.0.1:8188")
    parser.add_argument("--server", default="civitai-fill")
    parser.add_argument("--tool", default="civitai_list_rules")
    parser.add_argument("--expect", default="generic")
    try:
        asyncio.run(main(parser.parse_args()))
    except AssertionError as error:
        print("FAIL:", error)
        sys.exit(1)
