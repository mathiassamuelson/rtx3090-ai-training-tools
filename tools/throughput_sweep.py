#!/usr/bin/env python3
"""
throughput_sweep.py — throughput sweep for OpenAI-compatible LLM endpoints.

Supports vLLM (native OpenAI-compatible) and llama.cpp-server (OpenAI-compatible mode).
Uses streaming completions to measure per-request prefill and decode rates, and emits a
self-describing JSON results file. As of schema v3 the script can issue N requests
concurrently per measured wave and report system-level aggregate throughput under load.

Timing methodology (per request):
  - Wall clock is captured around the HTTP request.
  - Time-to-first-token (TTFT) is the wall-clock from request send to the first chunk
    containing non-empty text. This is treated as the prefill window.
  - Token counts come from the final chunk's `usage` block (requires
    `stream_options.include_usage=true`, which vLLM and llama.cpp-server both support).
  - Decode rate is (completion_tokens - 1) / (wall_time - ttft): the first generated
    token is produced at TTFT, so only the remaining tokens belong to the decode window.

Concurrency model (schema v3):
  A "wave" is N requests dispatched simultaneously via asyncio + httpx.AsyncClient and
  gathered. `--concurrency N` sets the wave width; `--iterations` sets how many measured
  waves are run per prompt size (and `--warmup` how many discarded waves precede them).
  At --concurrency 1 a wave is a single request, so `iterations` waves of width 1 reduce
  exactly to the v2 single-request-per-iteration behavior.

  Per-request records are always preserved. The concurrent aggregate is computed from
  the individual records, never measured separately:
    aggregate_gen_throughput = sum(completion_tokens over a wave's OK requests)
                             / (max completion_time - min dispatch_time over those requests)
  i.e. total generated tokens divided by wall-clock from the wave's first dispatch to its
  last completion. This denominator includes prefill time and is therefore a system
  throughput figure, distinct from the per-request decode_rate (which excludes prefill).

  dispatch_time and completion_time on each record are seconds relative to that wave's
  dispatch epoch (a shared monotonic reference taken just before the wave is gathered),
  so they are directly comparable within a wave and the wave wall-clock is well defined.

  Client-side caveat under concurrency: when N coroutines share one event loop, the
  instant the client *observes* a request's first token can be delayed by the loop
  servicing its siblings. At N=1 this is negligible. At N>1, per-request TTFT therefore
  reflects genuine server-side contention PLUS a small client-side observation artifact;
  it should be read as "observed TTFT under load," not isolated prefill latency. The
  aggregate throughput figure is unaffected by this, as it is derived from token counts
  and wave wall-clock, not from individual TTFTs.

Backend-agnostic by construction: the same request/response path is used for both
backends. The `--backend` flag is recorded in results metadata for provenance, not to
switch code paths.

Prefix cache avoidance:
  Each request's prompt is prepended with a short random nonce so that no two requests
  share a token prefix. This is necessary for backends with automatic cross-request
  prefix caching (vLLM enables it by default) — without it, repeated iterations of the
  same prompt are served from the prefix cache and measure KV lookup speed instead of
  prefill compute. The nonce adds only a handful of tokens, negligible at the prompt
  sizes this script is designed to measure.

  Defense in depth: if the server reports `usage.prompt_tokens_details.cached_tokens`
  on the final chunk (both vLLM and llama.cpp do), a non-trivial value on a measured
  iteration means the nonce strategy is failing and the result is invalid. The script
  captures this field into each iteration record and prints a live warning when it
  exceeds a small threshold. The threshold (max of 5 tokens or 5% of prompt) tolerates
  the BOS token, which is cached across requests on both backends independent of any
  nonce strategy — the first token is always shared between prompts, so cached_tokens=1
  on every request is expected and not a signal.

Server-side timings cross-check:
  llama.cpp-server emits a top-level `timings` block on the final SSE chunk with
  server-computed prefill and decode rates (prompt_ms, prompt_per_second, predicted_ms,
  predicted_per_second). When present, this is captured into the iteration record as
  `server_timings` and serves as an independent ground-truth cross-check against the
  script's own wall-clock measurements. vLLM does not emit this block; the field is
  omitted in that case.

Prompt size calibration:
  At startup, the script sends two small throwaway requests to the server to measure
  (a) the character-to-token ratio for its filler text under the target model's
  tokenizer, and (b) the token overhead of the nonce prefix. These measured values
  are then used to construct prompts that hit the requested token count accurately,
  rather than relying on a fixed heuristic that systematically misses on many
  tokenizers. The calibration constants are recorded in results metadata.

Output schema v3:
  {
    "metadata": { schema_version: 3, script, run, backend, endpoint, model,
                  sweep_config: { ..., concurrency } },
    "results": [
      {
        "prompt_size_requested": int,
        "concurrency": int,
        "waves": [
          {
            "wave_index": int,
            "requests": [ <per-request record>, ... ],   # supersedes v2 "iterations"
            "aggregate": { wall_time_s, gen_tokens, prompt_tokens,
                           gen_throughput_tok_s, n_ok, n_failed }
          }, ...
        ],
        "summary": {
          "per_request": { wall_time_s, ttft_s, prefill_rate_tok_s, decode_rate_tok_s },
          "aggregate":   { wall_time_s, gen_tokens, gen_throughput_tok_s }
        }
      }, ...
    ]
  }

  Per-request record fields are a superset of v2's iteration records: all v2 fields
  (prompt_tokens, completion_tokens, cached_tokens, wall_time_s, ttft_s, prefill_time_s,
  decode_time_s, prefill_rate_tok_s, decode_rate_tok_s, and server_timings when present)
  plus v3 additions request_id, dispatch_time, completion_time. The v2 per-prompt-size
  "summary" stats live under summary.per_request. Everything a v2 file carried is present;
  the structure is reorganized around waves, which is why schema_version is bumped to 3.
"""

import argparse
import asyncio
import json
import platform
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Optional

import httpx


BACKENDS = ("llamacpp", "vllm-openai")
SCHEMA_VERSION = 3

# Repo root = parent of tools/ (this script lives in tools/). Anchoring the default results
# directory and the git provenance to the SCRIPT's location — not the CWD — keeps the output
# location and the recorded git SHA invariant to where the sweep is launched from. (Mirrors
# the BASH_SOURCE/repo-root anchoring used in tools/start-stack.sh.)
REPO_ROOT = Path(__file__).resolve().parent.parent


def anchor_path(p: Path) -> Path:
    """Resolve a relative path against the repo root (script location), not the CWD.
    Absolute paths are returned unchanged, so callers retain an explicit escape hatch."""
    return p if p.is_absolute() else (REPO_ROOT / p)


def get_git_info(repo_root: Path, exclude: Optional[Path] = None) -> dict:
    """Return git SHA and dirty-tree status for the script's repo.

    Anchored to `repo_root` (via `git -C`), NOT the current working directory, so the
    recorded SHA reflects the code that actually ran regardless of where the sweep was
    launched. The dirty check excludes `exclude` (the results dir) when it lies inside the
    repo: result files are expected to be uncommitted at write time, so only changes
    OUTSIDE results/ mean the recorded SHA won't reflect the code/tools.
    """
    git = ["git", "-C", str(repo_root)]
    try:
        sha = subprocess.check_output(
            git + ["rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        status_cmd = git + ["status", "--porcelain", "--", "."]
        if exclude is not None:
            try:
                rel = exclude.resolve().relative_to(repo_root.resolve())
                status_cmd.append(f":(exclude){rel.as_posix()}")
            except ValueError:
                pass  # exclude is outside the repo — nothing to exclude, whole-tree check
        dirty = bool(subprocess.check_output(
            status_cmd,
            stderr=subprocess.DEVNULL,
        ).decode().strip())
        return {"git_sha": sha, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_sha": None, "dirty": None}


def slugify_model_name(name: str) -> str:
    """Make a model name safe for use in filenames."""
    name = name.split("/")[-1]  # strip org prefix if present
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in name)


def discover_model_name(endpoint: str, timeout: float = 5.0) -> Optional[str]:
    """Query /v1/models and return the first model id, or None on failure."""
    url = endpoint.rstrip("/") + "/v1/models"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
        models = data.get("data", [])
        if not models:
            return None
        return models[0].get("id")
    except (httpx.HTTPError, ValueError):
        return None


def calibrate_prompt_parameters(
    endpoint: str,
    model_name: str,
    request_timeout: float,
) -> tuple:
    """
    Measure two calibration constants for the target model's tokenizer by issuing
    small throwaway completion requests:

      - chars_per_filler_token: how many characters of the filler text ("lorem ")
        correspond to one token under this tokenizer.
      - nonce_tokens: how many tokens a nonce prefix (12 hex chars) consumes. A
        specific sample nonce is used; subsequent nonces should tokenize to roughly
        the same count, within a token or two of variance.

    Returns (chars_per_filler_token, nonce_tokens). Raises RuntimeError on failure.
    """
    url = endpoint.rstrip("/") + "/v1/completions"

    def count_tokens(text: str) -> int:
        payload = {
            "model": model_name,
            "prompt": text,
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }
        try:
            with httpx.Client(timeout=request_timeout) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()["usage"]["prompt_tokens"]
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise RuntimeError(f"Calibration request failed: {e}") from e

    filler_sample = ("lorem " * 500).strip()
    filler_tokens = count_tokens(filler_sample)
    if filler_tokens <= 0:
        raise RuntimeError("Calibration returned zero filler tokens.")
    chars_per_filler_token = len(filler_sample) / filler_tokens

    sample_nonce = uuid.uuid4().hex[:12]
    nonce_tokens = count_tokens(sample_nonce)

    return chars_per_filler_token, nonce_tokens


def build_prompt(
    target_tokens: int,
    chars_per_filler_token: float,
    nonce_tokens: int,
) -> str:
    """
    Build a prompt of approximately target_tokens tokens, with a unique nonce
    prefix to defeat cross-request prefix caching.

    Uses measured calibration constants (see calibrate_prompt_parameters) to size
    the filler accurately for the target model's tokenizer. The actual token count
    for each request is captured from the server response in the result record —
    build_prompt is best-effort, and final analysis should group by actual tokens,
    not requested tokens.
    """
    filler_tokens_needed = max(target_tokens - nonce_tokens, 1)
    filler_chars_needed = max(int(filler_tokens_needed * chars_per_filler_token), 1)

    word = "lorem "
    words_needed = max(1, (filler_chars_needed + len(word) - 1) // len(word))
    filler = (word * words_needed).strip()

    nonce = uuid.uuid4().hex[:12]
    return f"{nonce} {filler}"


async def run_streaming_request(
    client: httpx.AsyncClient,
    endpoint: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    request_id: str,
    wave_epoch: float,
) -> dict:
    """
    Issue one streaming completion request and return a per-request timing + token record.

    Raises RuntimeError if the response stream does not include a usage block, or if the
    server returns an error status.

    dispatch_time and completion_time in the returned record are seconds relative to
    wave_epoch (the shared monotonic reference for the wave this request belongs to), so
    that aggregate wall-clock can be computed across the wave's requests.

    Optional fields captured when the server emits them:
      - cached_tokens: from usage.prompt_tokens_details.cached_tokens. Non-zero on a
        measured (post-warmup) iteration indicates the nonce strategy is failing.
      - server_timings: llama.cpp-server's top-level `timings` block, used as an
        independent cross-check against the script's wall-clock measurements. vLLM
        does not emit this block.
    """
    url = endpoint.rstrip("/") + "/v1/completions"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_start = time.perf_counter()
    t_first_token = None
    usage = None
    server_timings = None

    async with client.stream("POST", url, json=payload, timeout=timeout) as resp:
        if resp.status_code >= 400:
            body = await resp.aread()
            raise RuntimeError(f"HTTP {resp.status_code}: {body[:500]!r}")
        async for line in resp.aiter_lines():
            if not line:
                continue
            if line.startswith("data: "):
                line = line[len("data: "):]
            if line.strip() == "[DONE]":
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk.get("timings"):
                server_timings = chunk["timings"]

            choices = chunk.get("choices") or []
            if choices and t_first_token is None:
                text = choices[0].get("text", "")
                if text:
                    t_first_token = time.perf_counter()

    t_end = time.perf_counter()

    if usage is None:
        raise RuntimeError(
            "No usage block in streamed response. "
            "Backend may not support stream_options.include_usage."
        )

    prompt_tokens = usage["prompt_tokens"]
    completion_tokens = usage["completion_tokens"]
    cached_tokens = (
        usage.get("prompt_tokens_details", {}).get("cached_tokens")
        if isinstance(usage.get("prompt_tokens_details"), dict)
        else None
    )

    wall_time = t_end - t_start
    if t_first_token is None:
        # No text chunks observed — degenerate case, treat entire wall time as prefill.
        ttft = wall_time
    else:
        ttft = t_first_token - t_start

    prefill_time = ttft
    decode_time = max(wall_time - ttft, 1e-9)

    prefill_rate = prompt_tokens / prefill_time if prefill_time > 0 else 0.0
    # Subtract 1: the first decoded token is produced at TTFT, not during decode.
    decode_rate = max(completion_tokens - 1, 0) / decode_time

    record = {
        "request_id": request_id,
        "dispatch_time": t_start - wave_epoch,
        "completion_time": t_end - wave_epoch,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "wall_time_s": wall_time,
        "ttft_s": ttft,
        "prefill_time_s": prefill_time,
        "decode_time_s": decode_time,
        "prefill_rate_tok_s": prefill_rate,
        "decode_rate_tok_s": decode_rate,
    }
    if server_timings is not None:
        record["server_timings"] = server_timings
    return record


async def run_wave(
    client: httpx.AsyncClient,
    endpoint: str,
    model_name: str,
    size: int,
    concurrency: int,
    max_tokens: int,
    timeout: float,
    chars_per_filler_token: float,
    nonce_tokens: int,
    wave_label: str,
) -> tuple:
    """
    Dispatch `concurrency` streaming requests simultaneously and gather them.

    Returns (ok_records, failures), where failures is a list of error strings. A shared
    wave_epoch is captured immediately before gather so per-request dispatch_time /
    completion_time are mutually comparable.
    """
    prompts = [
        build_prompt(size, chars_per_filler_token, nonce_tokens)
        for _ in range(concurrency)
    ]
    wave_epoch = time.perf_counter()
    tasks = [
        run_streaming_request(
            client, endpoint, model_name, prompts[j], max_tokens, timeout,
            request_id=f"{wave_label}-r{j}", wave_epoch=wave_epoch,
        )
        for j in range(concurrency)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    ok_records, failures = [], []
    for res in results:
        if isinstance(res, Exception):
            failures.append(f"{type(res).__name__}: {res}")
        else:
            ok_records.append(res)
    return ok_records, failures


def aggregate_wave(ok_records: list, n_failed: int) -> Optional[dict]:
    """
    System-level aggregate for a single wave: total generated tokens over OK requests
    divided by wall-clock from the wave's first dispatch to its last completion.

    Returns None if the wave had no successful requests.
    """
    if not ok_records:
        return None
    first_dispatch = min(r["dispatch_time"] for r in ok_records)
    last_completion = max(r["completion_time"] for r in ok_records)
    wall = max(last_completion - first_dispatch, 1e-9)
    gen_tokens = sum(r["completion_tokens"] for r in ok_records)
    prompt_tokens = sum(r["prompt_tokens"] for r in ok_records)
    return {
        "wall_time_s": wall,
        "gen_tokens": gen_tokens,
        "prompt_tokens": prompt_tokens,
        "gen_throughput_tok_s": gen_tokens / wall,
        "n_ok": len(ok_records),
        "n_failed": n_failed,
    }


def _stat_block(values: list) -> dict:
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
        "stdev": stdev(values) if len(values) > 1 else 0.0,
    }


def summarize_per_request(records: list) -> dict:
    """Summary statistics across all per-request records for a prompt size."""
    def stat(key):
        return _stat_block([r[key] for r in records])
    return {
        "wall_time_s": stat("wall_time_s"),
        "ttft_s": stat("ttft_s"),
        "prefill_rate_tok_s": stat("prefill_rate_tok_s"),
        "decode_rate_tok_s": stat("decode_rate_tok_s"),
    }


def summarize_aggregates(wave_aggs: list) -> dict:
    """Summary statistics across per-wave aggregate figures for a prompt size."""
    def stat(key):
        return _stat_block([w[key] for w in wave_aggs])
    return {
        "wall_time_s": stat("wall_time_s"),
        "gen_tokens": stat("gen_tokens"),
        "gen_throughput_tok_s": stat("gen_throughput_tok_s"),
    }


async def run_sweep(
    endpoint: str,
    model_name: str,
    prompt_sizes: list,
    concurrency: int,
    max_tokens: int,
    iterations: int,
    warmup: int,
    request_timeout: float,
    chars_per_filler_token: float,
    nonce_tokens: int,
) -> list:
    """Run warmup + measured waves for each prompt size and assemble the results list."""
    results = []
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(limits=limits) as client:
        for size in prompt_sizes:
            print(f"[prompt_size={size}]", file=sys.stderr)

            for i in range(warmup):
                ok, fail = await run_wave(
                    client, endpoint, model_name, size, concurrency,
                    max_tokens, request_timeout,
                    chars_per_filler_token, nonce_tokens,
                    wave_label=f"s{size}-warm{i+1}",
                )
                status = f"{len(ok)} ok" + (f", {len(fail)} FAILED" if fail else "")
                print(f"  warmup wave {i+1}/{warmup}: {status}", file=sys.stderr)
                for f in fail:
                    print(f"    [warmup fail] {f}", file=sys.stderr)

            waves = []
            all_ok_records = []
            wave_aggs = []
            for w in range(iterations):
                ok, fail = await run_wave(
                    client, endpoint, model_name, size, concurrency,
                    max_tokens, request_timeout,
                    chars_per_filler_token, nonce_tokens,
                    wave_label=f"s{size}-w{w+1}",
                )
                agg = aggregate_wave(ok, len(fail))
                waves.append({
                    "wave_index": w,
                    "requests": ok,
                    "aggregate": agg,
                })
                all_ok_records.extend(ok)
                if agg is not None:
                    wave_aggs.append(agg)

                # Live console output.
                if concurrency == 1 and ok:
                    rec = ok[0]
                    server_note = ""
                    if rec.get("server_timings"):
                        st = rec["server_timings"]
                        sp = st.get("prompt_per_second")
                        sd = st.get("predicted_per_second")
                        if sp is not None and sd is not None:
                            server_note = f" | server prefill={sp:.1f} decode={sd:.1f}"
                    print(
                        f"  iter {w+1}/{iterations}: "
                        f"prompt={rec['prompt_tokens']}tok "
                        f"gen={rec['completion_tokens']}tok "
                        f"prefill={rec['prefill_rate_tok_s']:.1f}tok/s "
                        f"decode={rec['decode_rate_tok_s']:.1f}tok/s"
                        f"{server_note}",
                        file=sys.stderr,
                    )
                elif ok:
                    decodes = [r["decode_rate_tok_s"] for r in ok]
                    print(
                        f"  wave {w+1}/{iterations}: "
                        f"{len(ok)}/{concurrency} ok "
                        f"agg_gen={agg['gen_throughput_tok_s']:.1f}tok/s "
                        f"wall={agg['wall_time_s']:.2f}s "
                        f"| per-req decode {min(decodes):.1f}-{max(decodes):.1f}tok/s",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"  wave {w+1}/{iterations}: 0/{concurrency} ok — all failed",
                        file=sys.stderr,
                    )
                for f in fail:
                    print(f"    [fail] {f}", file=sys.stderr)

                # Live warning: cached_tokens exceeding a small threshold on a measured
                # request means the nonce strategy is failing. The BOS token is always
                # cached across requests on both backends, so cached_tokens=1 is expected
                # and not a signal. Threshold catches meaningful cache hits.
                for rec in ok:
                    cached = rec.get("cached_tokens")
                    if cached is not None and cached > max(5, int(0.05 * rec["prompt_tokens"])):
                        print(
                            f"    [warn] cached_tokens={cached} on {rec['request_id']} "
                            f"— prefix cache is hitting, result is invalid.",
                            file=sys.stderr,
                        )

            entry = {
                "prompt_size_requested": size,
                "concurrency": concurrency,
                "waves": waves,
            }
            summary = {}
            if all_ok_records:
                summary["per_request"] = summarize_per_request(all_ok_records)
            if wave_aggs:
                summary["aggregate"] = summarize_aggregates(wave_aggs)
            if summary:
                entry["summary"] = summary
            results.append(entry)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Throughput sweep for OpenAI-compatible LLM endpoints "
                    "(single-request or concurrent).",
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=BACKENDS,
        help="Backend serving the endpoint. Recorded in results metadata for provenance.",
    )
    parser.add_argument(
        "--placement",
        default="na",
        choices=("naive", "steered", "na"),
        help="Stage->GPU placement provenance for pipeline-parallel runs (default: %(default)s). "
             "Recorded in metadata and folded into the default filename; does NOT switch code "
             "paths and is NOT verified by this script (confirm actual placement via nvidia-smi "
             "uuid-join). Use 'na' for configs where stage ordering is not a variable (e.g. TP).",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000",
        help="Base URL of the OpenAI-compatible server (default: %(default)s).",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model identifier. If omitted, queried from /v1/models; exits if discovery fails.",
    )
    parser.add_argument(
        "--prompt-sizes",
        type=int,
        nargs="+",
        default=[128, 512, 1024, 2048, 4096],
        help="Approximate prompt sizes in tokens to sweep (default: %(default)s). "
             "Actual counts come from the server.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Requests dispatched simultaneously per wave (default: %(default)s). "
             "At 1, behavior reduces to single-request-per-iteration (v2 semantics).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Completion tokens requested per call (default: %(default)s).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Measured waves per prompt size (default: %(default)s).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup waves per prompt size, discarded (default: %(default)s).",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=600.0,
        help="Per-request timeout in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file. Default: "
             "<results-dir>/throughput_sweep_<backend>_<model>_c<N>[_<placement>]_<timestamp>.json "
             "(placement segment included only when not 'na'). A relative path is anchored to "
             "the repo root, same as --results-dir.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for the default output filename (default: %(default)s). A RELATIVE "
             "path resolves against the repo root (the script's location), NOT the current "
             "directory, so output lands in the same place regardless of where you launch "
             "from. Pass an absolute path to write elsewhere.",
    )
    args = parser.parse_args()

    if args.concurrency < 1:
        print("[error] --concurrency must be >= 1.", file=sys.stderr)
        sys.exit(2)

    # Resolve model name (explicit or discovered)
    model_source = "explicit"
    model_name = args.model_name
    if model_name is None:
        print(
            f"[info] --model-name not specified; querying {args.endpoint}/v1/models",
            file=sys.stderr,
        )
        model_name = discover_model_name(args.endpoint)
        if model_name is None:
            print(
                "[error] Could not discover model name from /v1/models. "
                "Pass --model-name explicitly.",
                file=sys.stderr,
            )
            sys.exit(2)
        model_source = "discovered"
        print(f"[info] discovered model: {model_name}", file=sys.stderr)

    # Resolve output path (model name and concurrency level both in the default filename
    # so runs against different models or concurrency levels never silently overwrite).
    # Relative paths anchor to the repo root, not the CWD, so output lands in the same place
    # regardless of where the sweep is launched from (see anchor_path).
    if args.output:
        output_path = anchor_path(Path(args.output))
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = slugify_model_name(model_name)
        placement_seg = "" if args.placement == "na" else f"_{args.placement}"
        filename = (
            f"throughput_sweep_{args.backend}_{slug}_c{args.concurrency}"
            f"{placement_seg}_{timestamp}.json"
        )
        output_path = anchor_path(Path(args.results_dir)) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calibrate tokenizer behavior for this model
    print(f"[info] calibrating tokenizer via {args.endpoint}/v1/completions",
          file=sys.stderr)
    try:
        chars_per_filler_token, nonce_tokens = calibrate_prompt_parameters(
            args.endpoint, model_name, args.request_timeout,
        )
    except RuntimeError as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(3)
    print(
        f"[info] chars_per_filler_token={chars_per_filler_token:.4f} "
        f"nonce_tokens={nonce_tokens}",
        file=sys.stderr,
    )

    # Run header
    print(f"throughput_sweep: backend={args.backend} model={model_name}", file=sys.stderr)
    print(f"  endpoint={args.endpoint}", file=sys.stderr)
    print(f"  placement={args.placement} (provenance only — verify via nvidia-smi uuid-join)",
          file=sys.stderr)
    print(f"  prompt_sizes={args.prompt_sizes}  max_tokens={args.max_tokens}", file=sys.stderr)
    print(f"  concurrency={args.concurrency}  iterations={args.iterations}  "
          f"warmup={args.warmup}", file=sys.stderr)
    print(f"  output={output_path}", file=sys.stderr)
    print("", file=sys.stderr)

    # Sweep
    results = asyncio.run(run_sweep(
        endpoint=args.endpoint,
        model_name=model_name,
        prompt_sizes=args.prompt_sizes,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        iterations=args.iterations,
        warmup=args.warmup,
        request_timeout=args.request_timeout,
        chars_per_filler_token=chars_per_filler_token,
        nonce_tokens=nonce_tokens,
    ))

    # Metadata (no model-specific hardcoded fields)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "script": {
            "name": "throughput_sweep.py",
            "git": get_git_info(REPO_ROOT, exclude=output_path.parent),
        },
        "run": {
            "run_id": str(uuid.uuid4()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        },
        "backend": args.backend,
        "endpoint": args.endpoint,
        "model": {
            "name": model_name,
            "source": model_source,
            "tokenizer_calibration": {
                "chars_per_filler_token": chars_per_filler_token,
                "nonce_tokens": nonce_tokens,
            },
        },
        "sweep_config": {
            "prompt_sizes_requested": args.prompt_sizes,
            "concurrency": args.concurrency,
            "placement": args.placement,
            "max_tokens": args.max_tokens,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "request_timeout_s": args.request_timeout,
            "prompt_generation": "nonce_prefixed",
        },
    }

    output = {
        "metadata": metadata,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
