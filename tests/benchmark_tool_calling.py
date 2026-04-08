#!/usr/bin/env python3
"""
Tool-calling quality + speed benchmark for pi-mono use cases.

Tests the model against realistic tool call scenarios:
  - Simple single-tool calls (weather, lookup)
  - Multi-argument TypeBox-style schemas
  - Nested object arguments
  - Multi-step tool chains
  - Edge cases: optional fields, enums, arrays

Usage:
  python tests/benchmark_tool_calling.py [port]
  python tests/benchmark_tool_calling.py 11435
"""

import sys
import json
import time
import urllib.request
import urllib.error

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 11435
BASE = f"http://127.0.0.1:{PORT}"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_codebase",
            "description": "Search the codebase for a pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex or literal string to search for"},
                    "file_glob": {"type": "string", "description": "Glob pattern to filter files, e.g. '*.ts'"},
                    "case_sensitive": {"type": "boolean", "default": False},
                    "max_results": {"type": "integer", "default": 20},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Make an exact string replacement in a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to the file"},
                    "old_string": {"type": "string", "description": "Exact text to replace"},
                    "new_string": {"type": "string", "description": "Replacement text"},
                    "replace_all": {"type": "boolean", "default": False},
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command and return output",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run"},
                    "timeout_ms": {"type": "integer", "default": 30000},
                    "cwd": {"type": "string", "description": "Working directory (optional)"},
                },
                "required": ["command"],
            },
        },
    },
]

CASES = [
    # (name, messages, expected_tool_name, check_fn)
    (
        "simple_single_tool",
        [{"role": "user", "content": "What's the weather in Vienna?"}],
        "get_weather",
        lambda args: args.get("location", "").lower() in ("vienna", "wien"),
    ),
    (
        "enum_parameter",
        [{"role": "user", "content": "What's the weather in Tokyo in fahrenheit?"}],
        "get_weather",
        lambda args: args.get("unit") == "fahrenheit" and "tokyo" in args.get("location", "").lower(),
    ),
    (
        "multi_arg_required",
        [{"role": "user", "content": "Search for 'useState' in TypeScript files in the codebase"}],
        "search_codebase",
        lambda args: "useState" in args.get("pattern", "") and ("ts" in args.get("file_glob", "").lower() or args.get("file_glob") is None),
    ),
    (
        "boolean_arg",
        [{"role": "user", "content": "Search for 'TODO' case-sensitively in all files"}],
        "search_codebase",
        lambda args: args.get("case_sensitive") is True and "TODO" in args.get("pattern", ""),
    ),
    (
        "file_edit_exact",
        [{"role": "user", "content": "In /src/app.ts replace the string 'const PORT = 3000' with 'const PORT = 8080'"}],
        "edit_file",
        lambda args: args.get("file_path") == "/src/app.ts"
            and "3000" in args.get("old_string", "")
            and "8080" in args.get("new_string", ""),
    ),
    (
        "command_with_optional",
        [{"role": "user", "content": "Run 'npm test' in /home/user/project with a 60 second timeout"}],
        "run_command",
        lambda args: "npm test" in args.get("command", "")
            and (args.get("timeout_ms", 0) >= 60000 or args.get("cwd", "") == "/home/user/project"),
    ),
    (
        "multi_step_chain",
        [
            {"role": "user", "content": "First search for all files containing 'deprecated' then show me the weather in Berlin"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "search_codebase", "arguments": json.dumps({"pattern": "deprecated"})}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"matches": ["src/old.ts:42", "src/legacy.ts:10"]})},
        ],
        "get_weather",
        lambda args: "berlin" in args.get("location", "").lower(),
    ),
]


def call_model(messages, tools, max_tokens=256):
    payload = json.dumps({
        "model": "local",
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())
    elapsed = time.perf_counter() - t0
    return body, elapsed


def check_health():
    try:
        req = urllib.request.Request(f"{BASE}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return data.get("status") == "ok"
    except Exception:
        return False


def run():
    print(f"Tool-calling benchmark — {BASE}")
    print("=" * 60)

    if not check_health():
        print("ERROR: Server not reachable or not ready. Check /health.")
        sys.exit(1)

    passed = 0
    failed = 0
    total_tps = 0.0
    results = []

    for name, messages, expected_tool, check_fn in CASES:
        try:
            resp, elapsed = call_model(messages, TOOLS, max_tokens=256)
        except Exception as e:
            print(f"  FAIL  {name}: request error: {e}")
            failed += 1
            results.append({"name": name, "status": "error", "error": str(e)})
            continue

        choice = resp.get("choices", [{}])[0]
        msg = choice.get("message", {})
        tool_calls = msg.get("tool_calls") or []
        usage = resp.get("usage", {})
        ctok = usage.get("completion_tokens", 0)
        tps = ctok / elapsed if elapsed > 0 else 0.0
        total_tps += tps

        if not tool_calls:
            # Check if the model responded with text instead of a tool call
            content = msg.get("content") or ""
            print(f"  FAIL  {name}: no tool_calls (got text: {content[:80]!r})")
            failed += 1
            results.append({"name": name, "status": "no_tool_call", "text": content[:120]})
            continue

        tc = tool_calls[0]
        fn_name = tc.get("function", {}).get("name", "")
        raw_args = tc.get("function", {}).get("arguments", "{}")

        try:
            args = json.loads(raw_args)
            json_ok = True
        except json.JSONDecodeError as e:
            print(f"  FAIL  {name}: invalid JSON in arguments: {e}")
            print(f"        raw: {raw_args[:120]}")
            failed += 1
            results.append({"name": name, "status": "invalid_json", "raw": raw_args[:200]})
            continue

        if fn_name != expected_tool:
            print(f"  FAIL  {name}: expected tool={expected_tool!r}, got={fn_name!r}")
            failed += 1
            results.append({"name": name, "status": "wrong_tool", "got": fn_name, "args": args})
            continue

        if not check_fn(args):
            print(f"  FAIL  {name}: tool={fn_name}, args check failed: {args}")
            failed += 1
            results.append({"name": name, "status": "wrong_args", "tool": fn_name, "args": args})
            continue

        print(f"  PASS  {name}: {fn_name}({json.dumps(args)[:80]}) — {tps:.1f} t/s ({ctok} tok, {elapsed:.2f}s)")
        passed += 1
        results.append({"name": name, "status": "pass", "tool": fn_name, "args": args, "tps": tps})

    print()
    print("=" * 60)
    avg_tps = total_tps / len(CASES) if CASES else 0
    print(f"Results: {passed}/{len(CASES)} passed, {failed} failed")
    print(f"Avg generation speed: {avg_tps:.1f} t/s")
    print()

    if failed > 0:
        print("Failed cases:")
        for r in results:
            if r["status"] != "pass":
                print(f"  {r['name']}: {r}")

    # Write JSON summary
    summary = {
        "port": PORT,
        "passed": passed,
        "failed": failed,
        "total": len(CASES),
        "avg_tps": round(avg_tps, 2),
        "cases": results,
    }
    out = f"logs/tool_calling_bench_{PORT}.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {out}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run())
