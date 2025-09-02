"""
inspect_redis_checkpoints.py

Resolve checkpoint_latest:<session>:<ns> -> JSON.GET the checkpoint and pretty-print messages.
Usage:
  python inspect_redis_checkpoints.py <session_id> [--ns __empty__] [--host redis-host]
"""
from __future__ import annotations
import argparse
import json
import os
import pprint
import redis
from typing import Any, List

def _get_redis_host(host_arg: str | None) -> str:
    if host_arg:
        return host_arg
    # prefer project config if available
    try:
        from agentic_rag.config import settings  # type: ignore
        return getattr(settings, "REDIS_HOST", "localhost")
    except Exception:
        return os.environ.get("REDIS_HOST", "localhost")

def flatten_messages(items: Any) -> List[dict]:
    out: List[dict] = []
    if isinstance(items, list):
        for it in items:
            out.extend(flatten_messages(it))
    elif isinstance(items, dict):
        out.append(items)
    return out

def inspect_checkpoint(session_id: str, ns: str = "__empty__", host: str | None = None, port: int = 6379):
    host = _get_redis_host(host)
    r = redis.Redis(host=host, port=port, decode_responses=True)
    latest_key = r.get(f"checkpoint_latest:{session_id}:{ns}")
    if not latest_key:
        print(f"No checkpoint_latest found for session={session_id} ns={ns}")
        return 1
    print(f"Latest checkpoint key: {latest_key}")

    raw = r.execute_command("JSON.GET", latest_key, ".")
    try:
        payload = json.loads(raw)
    except Exception:
        print("Failed to parse JSON payload, raw:")
        print(raw)
        return 2

    pp = pprint.PrettyPrinter(indent=2, width=140)
    print("\n--- checkpoint metadata ---")
    meta = {k: payload.get(k) for k in ("thread_id", "checkpoint_id", "checkpoint_ts", "parent_checkpoint_id")}
    pp.pprint(meta)

    checkpoint = payload.get("checkpoint", {})
    channel_values = checkpoint.get("channel_values", {})
    raw_messages = channel_values.get("messages", [])
    messages = flatten_messages(raw_messages)

    print(f"\n--- {len(messages)} messages ---")
    for idx, msg in enumerate(messages):
        kwargs = msg.get("kwargs", {})
        role = "unknown"
        mid = msg.get("id")
        if isinstance(mid, list) and len(mid) > 0:
            last = mid[-1]
            if isinstance(last, str) and last.endswith("Message"):
                # normalize to 'human' / 'ai' / 'system'
                if "HumanMessage" in last:
                    role = "human"
                elif "AIMessage" in last:
                    role = "ai"
                else:
                    role = last
        if role == "unknown":
            role = kwargs.get("type") or msg.get("type") or "unknown"
        content = kwargs.get("content", "<no-content>")
        # safe truncation for long docs
        if isinstance(content, str) and len(content) > 1000:
            content_preview = content[:1000] + "...(truncated)"
        else:
            content_preview = content
        print(f"[{idx}] role={role} id={msg.get('id')} lc={msg.get('lc')}")
        print(f"      content: {content_preview}\n")

    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session_id", help="session id / thread id to inspect")
    parser.add_argument("--ns", default="__empty__", help="checkpoint namespace (default: __empty__)")
    parser.add_argument("--host", default=None, help="redis host (overrides config/REDIS_HOST)")
    parser.add_argument("--port", type=int, default=6379, help="redis port")
    args = parser.parse_args()
    code = inspect_checkpoint(args.session_id, args.ns, args.host, args.port)
    raise SystemExit(code)

if __name__ == "__main__":
    main()