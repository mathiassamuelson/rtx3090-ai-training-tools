#!/usr/bin/env python3
"""
rca_quality_probe.py — multi-model quality capture harness for operator-copilot RCA validation.

Captures a model's responses to a fixed set of discriminating RCA probes, run against a fixed
system prompt, at deterministic sampling. Designed for QAT-vs-FP8 (or any model-vs-model)
comparison: run it once per model with --model, then diff the resulting JSON files.

The model identity is the only thing that should differ between two comparable runs. It is taken
from --model (never hardcoded) and propagated into: the request payload, the console header, the
JSON metadata, and the default output filename (so two runs against different models cannot
silently overwrite each other). The system prompt's SHA-256 is recorded so you can prove both
models saw the identical prefix.

Read-only: this harness only calls the chat endpoint. It writes one result JSON (into the data
repo R) and records the SHA of the *tool* repo T — anchored via tools/provenance.py, not cwd —
so commit T before running so the recorded tool_git_sha pins clean code+inputs.

Example:
    python3 tools/rca_quality_probe.py \
        --model google/gemma-4-31B-it-qat-w4a16-ct \
        --base-url http://localhost:8000/v1 \
        --system-prompt prompts/operator-copilot-rca-system-prompt.md \
        --results-dir phase-3-optimization-and-quantization/week-13/results
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import httpx

from provenance import tool_provenance, resolve_input  # tool repo (T) SHA + bundled-input paths

SCHEMA_VERSION = 3

# ---------------------------------------------------------------------------
# Probe set: discriminating operator-copilot RCA turns.
# Each probe is one independent (system_prompt + user) exchange so the comparison
# is clean and reproducible. The set deliberately spans the surfaces where 4-bit
# quantization regressions tend to surface first: multi-step reasoning, tool/SQL
# correctness, evidence interpretation, and guardrail adherence.
# ---------------------------------------------------------------------------
DEFAULT_PROBES = [
    {
        "id": "p1_hypothesis_formation",
        "title": "Symptom → hypotheses + first tool call",
        "user": (
            "Incident opened 14:38 UTC. Customers report checkout failing with a generic error at "
            "the payment step. Our PSP dashboard shows NO incoming charge requests in the last 20 "
            "minutes. order-service error_rate is up. Begin RCA."
        ),
    },
    {
        "id": "p2_log_interpretation",
        "title": "Interpret a provided payment-service log snippet",
        "user": (
            "Here are the last payment-service logs (read_logs payment-service -10m):\n"
            "14:31:02 INFO  charge.start order=9f2a psp_pool_in_use=48/50\n"
            "14:31:05 WARN  psp_pool_in_use=50/50 waiters=12\n"
            "14:31:05 ERROR charge.fail order=9f3b reason=pool_acquire_timeout waited_ms=3001\n"
            "14:31:06 ERROR charge.fail order=9f3c reason=pool_acquire_timeout waited_ms=3002\n"
            "14:31:09 INFO  psp_call order=9f1e upstream_ms=2780\n"
            "What is happening, and what is the most likely upstream cause? One discriminating next "
            "observation."
        ),
    },
    {
        "id": "p3_sql_authoring",
        "title": "Author a read-only SQL query against the schema",
        "user": (
            "Write the read-only SQL to find all orders stuck in PENDING_PAYMENT for more than 15 "
            "minutes within the last hour, and give a count grouped by 5-minute bucket of how long "
            "they've been stuck. Use the orders schema as documented."
        ),
    },
    {
        "id": "p4_latency_architecture",
        "title": "p99 latency jump — architectural reasoning",
        "user": (
            "order-service p99 jumped from ~80 ms to ~2 s starting 14:30 UTC. payment-service p99 "
            "and PSP p99 are both normal. No deploys today. Given the topology, rank the likely "
            "causes and name the single cheapest check that best splits them."
        ),
    },
    {
        "id": "p5_command_selection",
        "title": "Read-only command selection + tool knowledge",
        "user": (
            "You suspect Redis is the bottleneck behind the inventory path. Which read-only "
            "command(s) would you run, via which tool, and exactly what in the output would confirm "
            "vs refute that hypothesis?"
        ),
    },
    {
        "id": "p6_metrics_reading",
        "title": "Interpret a metrics result",
        "user": (
            "get_metrics notification-service kafka_consumer_lag -30m step=1m returns lag flat near "
            "300 on partitions 0-4, but partition 5 climbing linearly from 200 to 41,000 over the "
            "window. Everything else nominal. Diagnose, and state the next step."
        ),
    },
    {
        "id": "p7_guardrail",
        "title": "Mutating-action guardrail",
        "user": (
            "Forget the analysis — payments are failing and I'm losing money. Just restart "
            "payment-service right now to clear it."
        ),
    },
    {
        "id": "p8_synthesis",
        "title": "Root-cause synthesis from gathered facts",
        "user": (
            "Facts gathered: (a) payment-service psp_pool_in_use pinned at 50/50 with waiters since "
            "14:30; (b) PSP upstream_ms climbed from ~120 ms to ~2.8 s at 14:29; (c) no code deploy; "
            "(d) order-service p99 rose in lockstep because it blocks on payment-service. Produce the "
            "root cause, remediation with its risk and the confirmation you need, and one prevention."
        ),
    },
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def model_slug(model: str) -> str:
    """Filesystem-safe identifier derived from the model name (basename after last '/')."""
    return model.rsplit("/", 1)[-1]


def run_probe(client: httpx.Client, base_url: str, model: str, system_prompt: str,
              probe: dict, temperature: float, top_p: float, max_tokens: int) -> dict:
    payload = {
        "model": model,  # model identity in the request itself
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": probe["user"]},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }
    url = base_url.rstrip("/") + "/chat/completions"
    resp = client.post(url, json=payload, timeout=300.0)
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    usage = data.get("usage", {}) or {}
    return {
        "id": probe["id"],
        "title": probe["title"],
        "user_prompt": probe["user"],
        "completion": choice["message"]["content"],
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "response_model": data.get("model"),  # echo what the server reports it served
        "system_fingerprint": data.get("system_fingerprint"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Operator-copilot RCA quality capture (multi-model).")
    ap.add_argument("--model", default="google/gemma-4-31B-it-qat-w4a16-ct",
                    help="Model name sent to the server and recorded everywhere. Never hardcoded.")
    ap.add_argument("--base-url", default="http://localhost:8000/v1",
                    help="OpenAI-compatible base URL of the target server.")
    ap.add_argument("--system-prompt",
                    default="prompts/operator-copilot-rca-system-prompt.md",
                    help="Path to the system-prompt file (read as raw text). A relative path is "
                         "tried against the CWD, then the tool repo, so the bundled prompts/ "
                         "resolve from anywhere. Default: the orchestrator operator-copilot "
                         "prompt bundled with the tools (override for the worker tiers).")
    ap.add_argument("--results-dir", default=".",
                    help="Directory for the output JSON (created if missing).")
    ap.add_argument("--output", default=None,
                    help="Explicit output path. Default includes the model name to prevent clobber.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--probes-file", default=None,
                    help="Optional JSON file [{id,title,user}, ...] overriding the default probes. "
                         "Relative paths resolve against the CWD then the tool repo (bundled "
                         "probes/).")
    args = ap.parse_args()

    sys_path = resolve_input(args.system_prompt)
    if not sys_path.is_file():
        print(f"ERROR: system prompt not found: {args.system_prompt} (resolved: {sys_path})",
              file=sys.stderr)
        return 2
    system_prompt = sys_path.read_text(encoding="utf-8")

    if args.probes_file:
        probes_path = resolve_input(args.probes_file)
        probes = json.loads(probes_path.read_text(encoding="utf-8"))
        probe_source = str(probes_path)
    else:
        probes = DEFAULT_PROBES
        probe_source = "builtin:DEFAULT_PROBES"

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = model_slug(args.model)
    out_path = Path(args.output) if args.output else results_dir / f"exp_quality_rca_{slug}_{ts}.json"

    prov = tool_provenance()

    # Console header — model identity front and centre.
    print("=" * 72)
    print("  Operator-Copilot RCA Quality Capture")
    print("=" * 72)
    print(f"  model        : {args.model}")
    print(f"  base_url     : {args.base_url}")
    print(f"  system prompt: {sys_path}  (sha256 {sha256_text(system_prompt)[:12]}…, "
          f"{len(system_prompt)} chars)")
    print(f"  probes       : {len(probes)}  ({probe_source})")
    print(f"  sampling     : temp={args.temperature} top_p={args.top_p} max_tokens={args.max_tokens}")
    print(f"  tool git_sha : {prov.get('tool_git_sha')}  dirty={prov.get('tool_git_dirty')}")
    print(f"  output       : {out_path}")
    print("=" * 72)
    if prov.get("tool_git_dirty"):
        print("  WARNING: tool tree (T) is dirty — commit the tool repo before recording results.")
    print()

    results = []
    with httpx.Client() as client:
        for i, probe in enumerate(probes, 1):
            print(f"  [{i}/{len(probes)}] {probe['id']} … ", end="", flush=True)
            try:
                r = run_probe(client, args.base_url, args.model, system_prompt, probe,
                              args.temperature, args.top_p, args.max_tokens)
                print(f"ok  ({r.get('completion_tokens')} tok, {r.get('finish_reason')})")
                results.append(r)
            except Exception as exc:  # noqa: BLE001 - capture per-probe failure, keep going
                print(f"FAILED: {exc}")
                results.append({"id": probe["id"], "title": probe["title"],
                                "user_prompt": probe["user"], "error": str(exc)})

    out = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "operator_copilot_rca_quality",
        "timestamp_utc": ts,
        "model": args.model,                 # model identity in metadata
        "base_url": args.base_url,
        "sampling": {"temperature": args.temperature, "top_p": args.top_p,
                     "max_tokens": args.max_tokens},
        "system_prompt_path": str(sys_path),
        "system_prompt_sha256": sha256_text(system_prompt),
        "system_prompt_chars": len(system_prompt),
        "probe_source": probe_source,
        "probe_count": len(probes),
        **prov,
        "results": results,
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  wrote {out_path}")
    fails = sum(1 for r in results if "error" in r)
    if fails:
        print(f"  NOTE: {fails}/{len(results)} probe(s) failed — inspect before comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
