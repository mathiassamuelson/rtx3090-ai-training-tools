#!/usr/bin/env python3
"""
interference_probe.py — cross-tier interference measurement for the multi-tier stack.

Question: given that GPU memory and GPU compute are DISJOINT per tier (the 31B on the
NVLink pair GPUs 0+2, the 12B workers on PCIe-x1 GPUs 1+3), does saturating the other
tiers degrade a victim tier's latency — and if so, by how much? Any interference must
flow through SHARED HOST resources (CPU schedulers/tokenizers, system-memory bandwidth,
PCIe to the shared root complex), since the GPUs cannot contend.

Method (one direction per invocation, `--victim 31b` or `--victim 12b`):
  1. Verify victim + aggressor GPU placement empirically (nvidia-smi index->uuid->pid join).
  2. Start a sustained CONCURRENT chat-endpoint flood against the aggressor tier(s). This
     is the load shape that actually stresses the shared host path — many simultaneous
     decode loops — unlike single-stream, which only pegs the (disjoint) aggressor GPUs.
       - victim=31b: flood the nginx POOL (/v1/chat/completions on :8080). least_conn fans
         it across both workers, so this saturates them AND produces the upstream split in
         nginx's access log (folds in the deferred least_conn characterization).
       - victim=12b: flood the OTHER worker (:8003) and the 31B (:8000) DIRECTLY. No pool
         (the pool would route to the victim worker); no upstream-split capture.
  3. After a ramp, sample aggressor-GPU utilisation as saturation evidence, then run the
     VICTIM probe: throughput_sweep.py at concurrency 1, identical args to the solo
     baseline, so loaded-vs-solo is a clean diff on the same measurement path.
  4. Stop the flood; pull the nginx upstream distribution from `docker logs` (victim=31b);
     diff the loaded victim sweep against its committed solo baseline.

Aggressor load uses nonce-prefixed prompts to defeat automatic prefix caching (which is
live on the workers — a cache hit would skip prefill and collapse the intended load).

Self-describing output: model identities are discovered from each live endpoint and
propagated into request payloads, metadata, and the default output filename. Nothing
about a particular model is hardcoded into the recorded results.

Usage:
  tools/interference_probe.py --victim 31b
  tools/interference_probe.py --victim 12b
  tools/interference_probe.py --victim 31b --aggressor-concurrency 24 --ramp-seconds 25
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

# ---- repo anchoring (script-location-derived, NOT cwd) -------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # .../tools
REPO_ROOT = SCRIPT_DIR.parent                         # repo root
DEFAULT_SWEEP = SCRIPT_DIR / "throughput_sweep.py"
DEFAULT_RESULTS_DIR = REPO_ROOT / "phase-3-optimization-and-quantization" / "week-13" / "results"

# ---- direction presets (deployment topology; every field overridable on the CLI) ----------
# aggressor target: {url, gpus, via_pool}. gpus is the saturation-evidence sample only.
PRESETS = {
    "31b": {
        "victim_endpoint": "http://localhost:8000",
        "victim_gpus": [0, 2],
        "aggressors": [{"url": "http://localhost:8080", "gpus": [1, 3], "via_pool": True}],
        "nginx_container": "nginx-frontdoor",
    },
    "12b": {
        "victim_endpoint": "http://localhost:8001",
        "victim_gpus": [1],
        "aggressors": [
            {"url": "http://localhost:8003", "gpus": [3], "via_pool": False},
            {"url": "http://localhost:8000", "gpus": [0, 2], "via_pool": False},
        ],
        "nginx_container": None,
    },
}


# ---- small helpers -------------------------------------------------------------------------
def slugify(name: str) -> str:
    name = name.split("/")[-1]
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in name)


def ts_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_info() -> dict:
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip())
        return {"git_sha": sha, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_sha": None, "dirty": None}


async def sh(*cmd: str, timeout: float = 30.0) -> str:
    """Run a command, return stdout (best-effort; empty string on failure)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return out.decode(errors="replace")
    except (asyncio.TimeoutError, FileNotFoundError, OSError):
        return ""


async def discover_model(client: httpx.AsyncClient, base_url: str) -> Optional[str]:
    try:
        r = await client.get(base_url.rstrip("/") + "/v1/models", timeout=5.0)
        r.raise_for_status()
        data = r.json().get("data", [])
        return data[0]["id"] if data else None
    except (httpx.HTTPError, ValueError, KeyError, IndexError):
        return None


# ---- placement verification (index->uuid->pid->mem join) -----------------------------------
async def gpu_maps() -> tuple:
    idx_uuid, idx_used = {}, {}
    out = await sh("nvidia-smi", "--query-gpu=index,uuid,memory.used",
                   "--format=csv,noheader,nounits")
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            idx_uuid[int(parts[0])] = parts[1]
            idx_used[int(parts[0])] = int(parts[2]) if parts[2].isdigit() else 0
    return idx_uuid, idx_used


def placement_record(idx_uuid: dict, idx_used: dict, gpus: list) -> dict:
    obs = {}
    ok = True
    for i in gpus:
        used = idx_used.get(i, 0)
        obs[str(i)] = {"uuid": idx_uuid.get(i, "NA"), "used_mib": used}
        if used < 1000:   # a loaded model sits well above driver idle
            ok = False
    return {"gpus": gpus, "observed": obs, "placement_ok": ok}


# ---- aggressor flood (concurrent chat-endpoint load) ---------------------------------------
class FloodStats:
    def __init__(self):
        self.ok = 0
        self.fail = 0
        self.errors = {}


async def _flood_worker(client, url, model, prompt, max_tokens, stop, stats, timeout):
    chat_url = url.rstrip("/") + "/v1/chat/completions"
    while not stop.is_set():
        nonce = uuid.uuid4().hex
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": f"[{nonce}] {prompt}"}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
        try:
            r = await client.post(chat_url, json=payload, timeout=timeout)
            if r.status_code == 200:
                stats.ok += 1
            else:
                stats.fail += 1
                key = f"http_{r.status_code}"
                stats.errors[key] = stats.errors.get(key, 0) + 1
        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            stats.fail += 1
            key = type(e).__name__
            stats.errors[key] = stats.errors.get(key, 0) + 1


def _build_filler(approx_tokens: int) -> str:
    # ~6 chars/token (matches the workers' measured calibration closely enough; aggressor
    # prompt size need not be exact — it only needs to create real prefill+decode work).
    sentence = "The quick brown fox jumps over the lazy dog near the riverbank. "
    return (sentence * max(1, (approx_tokens * 6) // len(sentence))).strip()


# ---- nginx upstream-distribution capture ---------------------------------------------------
async def nginx_distribution(container: str, since_iso: str) -> dict:
    out = await sh("docker", "logs", "--since", since_iso, container, timeout=30.0)
    dist = {}
    for line in out.splitlines():
        marker = "upstream="
        if marker in line:
            frag = line.split(marker, 1)[1].split()[0].strip().strip('"')
            for addr in frag.split(","):          # nginx may list multiple on retries
                addr = addr.strip()
                if addr and addr != "-":
                    dist[addr] = dist.get(addr, 0) + 1
    return dist


# ---- victim probe (reuse throughput_sweep.py at c=1) ---------------------------------------
async def run_victim_sweep(sweep_path, endpoint, results_dir, out_path,
                           prompt_sizes, max_tokens, iterations, warmup):
    cmd = [
        sys.executable, str(sweep_path),
        "--backend", "vllm-openai",
        "--endpoint", endpoint,
        "--prompt-sizes", *[str(s) for s in prompt_sizes],
        "--concurrency", "1",
        "--max-tokens", str(max_tokens),
        "--iterations", str(iterations),
        "--warmup", str(warmup),
        "--results-dir", str(results_dir),
        "--output", str(out_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out, err = await proc.communicate()
    # surface the sweep's stderr (its live per-iter lines) to our stderr for visibility
    sys.stderr.write(err.decode(errors="replace"))
    return proc.returncode


def load_sweep_means(path: Path) -> dict:
    """Return {prompt_size: {decode, prefill, ttft}} from a schema-v3 sweep JSON."""
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    out = {}
    for entry in doc.get("results", []):
        size = entry.get("prompt_size_requested")
        per = (entry.get("summary") or {}).get("per_request") or {}
        def m(k):
            v = per.get(k)
            return v.get("mean") if isinstance(v, dict) else None
        out[size] = {
            "decode_tok_s": m("decode_rate_tok_s"),
            "prefill_tok_s": m("prefill_rate_tok_s"),
            "ttft_s": m("ttft_s"),
        }
    return out


def diff_tables(solo: dict, loaded: dict) -> list:
    rows = []
    for size in sorted(set(solo) & set(loaded), key=lambda x: (x is None, x)):
        s, l = solo[size], loaded[size]
        row = {"prompt_size": size}
        for metric in ("decode_tok_s", "prefill_tok_s", "ttft_s"):
            sv, lv = s.get(metric), l.get(metric)
            ratio = (lv / sv) if (sv and lv) else None
            row[metric] = {"solo": sv, "loaded": lv,
                           "loaded_over_solo": round(ratio, 4) if ratio else None}
        rows.append(row)
    return rows


# ---- main ----------------------------------------------------------------------------------
async def amain(args):
    preset = PRESETS[args.victim]
    victim_endpoint = args.victim_endpoint or preset["victim_endpoint"]
    victim_gpus = preset["victim_gpus"]
    aggressors = preset["aggressors"]
    nginx_container = preset["nginx_container"]
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    start_iso = ts_utc()
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(limits=limits) as client:
        # --- discover models from live endpoints (captured, never hardcoded) ---
        victim_model = await discover_model(client, victim_endpoint)
        if victim_model is None:
            print(f"[error] could not discover victim model at {victim_endpoint}", file=sys.stderr)
            return 2
        for a in aggressors:
            a["model"] = await discover_model(client, a["url"])
            if a["model"] is None:
                print(f"[error] could not discover aggressor model at {a['url']}", file=sys.stderr)
                return 2

        victim_slug = slugify(victim_model)
        out_stem = f"interference_victim-{victim_slug}_{args.victim}-direction"
        victim_loaded_json = results_dir / f"{out_stem}_victim-sweep.json"
        harness_json = results_dir / f"{out_stem}.json"

        # --- placement verification (before any load) ---
        idx_uuid, idx_used = await gpu_maps()
        placement = {
            "victim": placement_record(idx_uuid, idx_used, victim_gpus),
            "aggressors": [placement_record(idx_uuid, idx_used, a["gpus"]) for a in aggressors],
        }
        all_ok = placement["victim"]["placement_ok"] and all(
            p["placement_ok"] for p in placement["aggressors"])
        print(f"[info] placement_ok={all_ok}  victim={victim_gpus} "
              f"aggressors={[a['gpus'] for a in aggressors]}", file=sys.stderr)
        if not all_ok and not args.ignore_placement:
            print("[error] placement check failed (a target GPU is near-idle). "
                  "Re-verify the stack, or pass --ignore-placement to override.", file=sys.stderr)
            return 3

        # --- start the aggressor flood ---
        stop = asyncio.Event()
        prompt = _build_filler(args.aggressor_prompt_tokens)
        flood_tasks = []
        flood_stats = []
        for a in aggressors:
            st = FloodStats()
            flood_stats.append(st)
            for _ in range(args.aggressor_concurrency):
                flood_tasks.append(asyncio.create_task(_flood_worker(
                    client, a["url"], a["model"], prompt, args.aggressor_max_tokens,
                    stop, st, args.aggressor_request_timeout)))
        total_inflight = args.aggressor_concurrency * len(aggressors)
        print(f"[info] aggressor flood up: {total_inflight} in-flight "
              f"({args.aggressor_concurrency}/target x {len(aggressors)} targets); "
              f"ramping {args.ramp_seconds}s", file=sys.stderr)

        # --- ramp, then saturation evidence ---
        await asyncio.sleep(args.ramp_seconds)
        util_out = await sh("nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits")
        sat_sample = {}
        for line in util_out.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 2 and p[0].isdigit():
                sat_sample[p[0]] = {"util_pct": p[1], "mem_used_mib": p[2] if len(p) > 2 else None}
        agg_gpu_idxs = sorted({i for a in aggressors for i in a["gpus"]})
        print(f"[info] saturation sample (aggressor GPUs {agg_gpu_idxs}): "
              + ", ".join(f"gpu{i}={sat_sample.get(str(i), {}).get('util_pct','?')}%"
                          for i in agg_gpu_idxs), file=sys.stderr)

        # --- victim probe (loaded) ---
        print(f"[info] running loaded victim sweep -> {victim_loaded_json}", file=sys.stderr)
        rc = await run_victim_sweep(
            args.sweep, victim_endpoint, results_dir, victim_loaded_json,
            args.victim_prompt_sizes, args.victim_max_tokens,
            args.victim_iterations, args.victim_warmup)

        # --- stop the flood ---
        stop.set()
        await asyncio.gather(*flood_tasks, return_exceptions=True)
        agg_summary = [{
            "url": a["url"], "model": a["model"], "via_pool": a["via_pool"],
            "requests_ok": st.ok, "requests_failed": st.fail, "errors": st.errors,
        } for a, st in zip(aggressors, flood_stats)]
        print("[info] aggressor flood stopped: "
              + "; ".join(f"{a['url']} ok={s['requests_ok']} fail={s['requests_failed']}"
                          for a, s in zip(aggressors, agg_summary)), file=sys.stderr)

        # --- nginx upstream distribution (pool direction only) ---
        nginx_dist = {}
        if nginx_container:
            nginx_dist = await nginx_distribution(nginx_container, start_iso)
            print(f"[info] nginx upstream distribution: {nginx_dist or '(none captured)'}",
                  file=sys.stderr)

    # --- diff vs solo baseline ---
    baseline_path = Path(args.baseline) if args.baseline else (
        results_dir / f"interference_solo_baseline_victim-{victim_slug}.json")
    solo = load_sweep_means(baseline_path)
    loaded = load_sweep_means(victim_loaded_json)
    diff = diff_tables(solo, loaded) if solo and loaded else []
    if not solo:
        print(f"[warn] solo baseline not found/parsed at {baseline_path} — diff skipped",
              file=sys.stderr)

    # --- emit self-describing harness JSON ---
    doc = {
        "experiment": "cross_tier_interference",
        "direction": args.victim,
        "timestamp_utc": start_iso,
        "git": git_info(),
        "host": __import__("socket").gethostname(),
        "victim": {"model": victim_model, "endpoint": victim_endpoint,
                   "gpus": victim_gpus, "probe": "throughput_sweep.py c=1"},
        "aggressors": agg_summary,
        "aggressor_config": {
            "concurrency_per_target": args.aggressor_concurrency,
            "prompt_tokens_approx": args.aggressor_prompt_tokens,
            "max_tokens": args.aggressor_max_tokens,
            "ramp_seconds": args.ramp_seconds,
            "load_endpoint": "/v1/chat/completions (nonce-prefixed, APC-defeating)",
        },
        "placement": placement,
        "saturation_sample": sat_sample,
        "nginx_upstream_distribution": nginx_dist,
        "victim_sweep_file": str(victim_loaded_json),
        "baseline_file": str(baseline_path),
        "victim_probe_rc": rc,
        "diff_loaded_vs_solo": diff,
    }
    harness_json.write_text(json.dumps(doc, indent=2))
    print(f"\n[info] wrote {harness_json}", file=sys.stderr)

    # --- human summary ---
    print(f"\n=== interference summary ({args.victim}-as-victim) ===")
    for row in diff:
        d = row["decode_tok_s"]; p = row["prefill_tok_s"]
        print(f"  prompt={row['prompt_size']}: "
              f"decode {d['solo']}->{d['loaded']} (x{d['loaded_over_solo']})  "
              f"prefill {p['solo']}->{p['loaded']} (x{p['loaded_over_solo']})")
    if nginx_container:
        print(f"  nginx upstream split: {nginx_dist or '(none)'}")
    return 0 if rc == 0 else 4


def main():
    ap = argparse.ArgumentParser(description="Cross-tier interference probe.")
    ap.add_argument("--victim", required=True, choices=("31b", "12b"),
                    help="Tier to probe for interference while the others are saturated.")
    ap.add_argument("--victim-endpoint", default=None, help="Override victim endpoint URL.")
    ap.add_argument("--sweep", default=str(DEFAULT_SWEEP), help="Path to throughput_sweep.py.")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--baseline", default=None,
                    help="Solo baseline sweep JSON. Default: "
                         "<results-dir>/interference_solo_baseline_victim-<victim-model>.json")
    # victim probe (match the solo baseline)
    ap.add_argument("--victim-prompt-sizes", type=int, nargs="+", default=[512, 4096])
    ap.add_argument("--victim-max-tokens", type=int, default=256)
    ap.add_argument("--victim-iterations", type=int, default=5)
    ap.add_argument("--victim-warmup", type=int, default=1)
    # aggressor load
    ap.add_argument("--aggressor-concurrency", type=int, default=16,
                    help="In-flight requests per aggressor target (default: %(default)s).")
    ap.add_argument("--aggressor-prompt-tokens", type=int, default=512,
                    help="Approx aggressor prompt size; small keeps decode loops cycling "
                         "(host pressure) without piling on disjoint GPU prefill.")
    ap.add_argument("--aggressor-max-tokens", type=int, default=256)
    ap.add_argument("--aggressor-request-timeout", type=float, default=120.0)
    ap.add_argument("--ramp-seconds", type=float, default=20.0,
                    help="Wait after starting the flood before measuring (clears the "
                         "aggressor sweep calibration/warmup ramp).")
    ap.add_argument("--ignore-placement", action="store_true",
                    help="Proceed even if a target GPU looks near-idle.")
    args = ap.parse_args()
    sys.exit(asyncio.run(amain(args)))


if __name__ == "__main__":
    main()
