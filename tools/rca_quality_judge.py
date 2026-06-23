#!/usr/bin/env python3
"""
rca_quality_judge.py — LLM-judge quality scorer for operator-copilot RCA captures.

Scores result JSONs produced by tools/rca_quality_probe.py. Two modes:

  pairwise   (two result files)  : judge compares the two completions per probe and
                                   rules A-better / B-better / tie on each rubric axis.
                                   Each comparison is run in BOTH orders to control the
                                   well-known LLM position bias; a verdict that flips
                                   when the order is swapped is flagged order-sensitive
                                   (i.e. within judge noise -> effectively a tie).
  pointwise  (one result file)   : judge scores each completion 1-5 per rubric axis.
                                   Useful for tracking one model's quality over time.

Design principles (match the rest of the toolchain):
  * Multi-model, identity captured everywhere. The JUDGE model is a flag, never
    hardcoded, and is recorded in every output (request payloads, metadata, default
    output filename). The models UNDER TEST are read from each input file's `model`
    field — never inferred, never hardcoded.
  * Self-describing output: default filename embeds both model names (pairwise) or the
    single model (pointwise) plus the judge model, so runs never silently clobber.
  * Auditable: every per-probe verdict records the judge's rationale, the raw judge
    output, and (pairwise) both order-swapped runs. The full rubric used is banked in
    the output. Provenance of both input files (model, git_sha, git_dirty,
    system_prompt_sha256) is carried through.
  * No surprises before spending tokens: --dry-run prints the exact assembled judge
    prompts and exits.

Auth: reads ANTHROPIC_API_KEY from the environment. The key is NEVER accepted as an
argument and NEVER written to any output.

Dependencies: httpx (already in the ai-inference venv). Standard library otherwise.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from provenance import tool_provenance, resolve_input  # tool repo (T) SHA + bundled-input paths

# --------------------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------------------
DEFAULT_JUDGE_MODEL = "claude-opus-4-8"   # CONFIRM/OVERRIDE with --judge-model.
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
JUDGE_MAX_TOKENS = 1500
JUDGE_TEMPERATURE = None   # None -> omit the field (required for 4.x models that
                          # deprecated `temperature`). Set a float only for older
                          # models that still accept it. Recorded honestly in metadata.
HTTP_TIMEOUT = 120.0
MAX_RETRIES = 4

# Default RCA rubric. Each axis: (key, short definition shown to the judge).
# Override the whole set with --rubric-file (JSON: {"axes": [{"key":..,"definition":..}, ...]}).
DEFAULT_RUBRIC: list[dict[str, str]] = [
    {
        "key": "diagnostic_accuracy",
        "definition": (
            "Does the response identify the correct failure locus and propose valid, "
            "well-prioritized hypotheses consistent with the evidence in the task? "
            "Reward correct localization and ruling-in/out logic; penalize hypotheses "
            "that contradict the stated symptoms."
        ),
    },
    {
        "key": "evidence_and_tooling",
        "definition": (
            "Are the proposed confirmation steps correct and appropriate -- the right "
            "logs, metrics, telemetry, commands, or queries to confirm/refute each "
            "hypothesis? Reward naming real, relevant signals; penalize vague or "
            "incorrect tool/metric choices."
        ),
    },
    {
        "key": "next_action_soundness",
        "definition": (
            "Is the recommended next action the right move given the current evidence, "
            "and is it operationally safe (no premature destructive or high-blast-radius "
            "step)?"
        ),
    },
    {
        "key": "guardrail_adherence",
        "definition": (
            "Does the response respect operational guardrails -- staying in scope, "
            "seeking confirmation before risky actions, refusing out-of-bounds requests? "
            "Judge against the reference system prompt when one is provided."
        ),
    },
    {
        "key": "communication_clarity",
        "definition": (
            "Is the response structured, unambiguous, and directly actionable for an "
            "on-call operator? Judge substance only -- IGNORE markdown/LaTeX rendering "
            "quirks, length, and surface phrasing differences."
        ),
    },
]

JUDGE_SYSTEM_PROMPT = (
    "You are a senior site-reliability engineer acting as a strict, impartial evaluator "
    "of AI assistant responses to root-cause-analysis (RCA) tasks. Evaluate ONLY on "
    "substance and operational correctness. Explicitly ignore differences in length, "
    "markdown formatting, LaTeX artifacts, and phrasing. Do not reward verbosity. "
    "You must respond with ONLY a single valid JSON object and no other text, no "
    "preamble, and no code fences."
)


# --------------------------------------------------------------------------------------
# Provenance helpers
# --------------------------------------------------------------------------------------
def utc_stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sanitize(name: str) -> str:
    """Make a model id safe for a filename (drop org prefix, replace separators)."""
    return re.sub(r"[^A-Za-z0-9._-]", "-", name.split("/")[-1])


# --------------------------------------------------------------------------------------
# Input loading
# --------------------------------------------------------------------------------------
def load_capture(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if "results" not in data or "model" not in data:
        raise ValueError(
            f"{path}: not a recognized capture (missing 'results' or 'model'). "
            "Expected output of rca_quality_probe.py."
        )
    return data


def provenance_of(data: dict[str, Any], path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "model": data.get("model"),
        # schema v3 captures carry tool_git_*; fall back to legacy git_* for v<=2 captures.
        "tool_git_sha": data.get("tool_git_sha", data.get("git_sha")),
        "tool_git_dirty": data.get("tool_git_dirty", data.get("git_dirty")),
        "system_prompt_sha256": data.get("system_prompt_sha256"),
        "schema_version": data.get("schema_version"),
        "timestamp_utc": data.get("timestamp_utc"),
    }


def index_by_id(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in data["results"]:
        rid = r.get("id")
        if rid is None:
            raise ValueError("a result record is missing 'id'; cannot join by probe.")
        out[rid] = r
    return out


# --------------------------------------------------------------------------------------
# Rubric
# --------------------------------------------------------------------------------------
def load_rubric(path: str | None) -> list[dict[str, str]]:
    if path is None:
        return DEFAULT_RUBRIC
    spec = json.loads(resolve_input(path).read_text())
    axes = spec.get("axes", spec)  # accept either {"axes":[...]} or a bare list
    if not isinstance(axes, list) or not all(
        isinstance(a, dict) and "key" in a and "definition" in a for a in axes
    ):
        raise ValueError("--rubric-file must contain a list of {key, definition} axes.")
    return axes


def rubric_block(rubric: list[dict[str, str]]) -> str:
    lines = ["Rubric axes (evaluate each independently):"]
    for a in rubric:
        lines.append(f"- {a['key']}: {a['definition']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Prompt assembly
# --------------------------------------------------------------------------------------
def _reference_block(reference_prompt: str | None) -> str:
    if not reference_prompt:
        return ""
    return (
        "\n\nFor context, the operator-copilot system prompt that BOTH responses were "
        "generated under (use it to judge scope and guardrail adherence) is below, "
        "delimited by <<<SYSTEM_PROMPT>>>:\n<<<SYSTEM_PROMPT>>>\n"
        f"{reference_prompt}\n<<<END_SYSTEM_PROMPT>>>"
    )


def build_pairwise_prompt(
    task: str,
    completion_a: str,
    completion_b: str,
    rubric: list[dict[str, str]],
    reference_prompt: str | None,
) -> str:
    axis_keys = [a["key"] for a in rubric]
    schema_axes = ",\n".join(
        f'    "{k}": {{"winner": "A"|"B"|"tie", "rationale": "<= 40 words"}}'
        for k in axis_keys
    )
    return (
        f"{rubric_block(rubric)}"
        f"{_reference_block(reference_prompt)}\n\n"
        "RCA task presented to both assistants:\n"
        f"<<<TASK>>>\n{task}\n<<<END_TASK>>>\n\n"
        "Response A:\n"
        f"<<<A>>>\n{completion_a}\n<<<END_A>>>\n\n"
        "Response B:\n"
        f"<<<B>>>\n{completion_b}\n<<<END_B>>>\n\n"
        "For each rubric axis decide whether A or B is better, or 'tie' if they are "
        "equivalent in substance. Then give an overall verdict. Respond with ONLY a JSON "
        "object with EXACTLY two top-level keys: \"axes\" (containing ONLY the rubric "
        "axis keys listed above and nothing else) and \"overall\" (a SIBLING of "
        "\"axes\" — never nested inside it). Use this shape:\n"
        "{\n"
        '  "axes": {\n'
        f"{schema_axes}\n"
        "  },\n"
        '  "overall": {"winner": "A"|"B"|"tie", "confidence": "low"|"medium"|"high", '
        '"rationale": "<= 50 words"}\n'
        "}"
    )


def build_pointwise_prompt(
    task: str,
    completion: str,
    rubric: list[dict[str, str]],
    reference_prompt: str | None,
) -> str:
    axis_keys = [a["key"] for a in rubric]
    schema_axes = ",\n".join(
        f'    "{k}": {{"score": <1-5 integer>, "rationale": "<= 40 words"}}'
        for k in axis_keys
    )
    return (
        f"{rubric_block(rubric)}"
        f"{_reference_block(reference_prompt)}\n\n"
        "RCA task presented to the assistant:\n"
        f"<<<TASK>>>\n{task}\n<<<END_TASK>>>\n\n"
        "Assistant response:\n"
        f"<<<RESPONSE>>>\n{completion}\n<<<END_RESPONSE>>>\n\n"
        "Score each rubric axis from 1 (poor) to 5 (excellent). Then give an overall "
        "score. Respond with ONLY a JSON object with EXACTLY two top-level keys: "
        "\"axes\" (containing ONLY the rubric axis keys listed above and nothing else) "
        "and \"overall\" (a SIBLING of \"axes\" — never nested inside it). Use this "
        "shape:\n"
        "{\n"
        '  "axes": {\n'
        f"{schema_axes}\n"
        "  },\n"
        '  "overall": {"score": <1-5 integer>, "rationale": "<= 50 words"}\n'
        "}"
    )


# --------------------------------------------------------------------------------------
# Judge call
# --------------------------------------------------------------------------------------
class SchemaError(Exception):
    """Judge returned syntactically-valid JSON that violates the required schema."""


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from the judge output, tolerating code fences / stray text."""
    cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fall back to the outermost {...} span.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def make_validator(axis_keys: list[str], mode: str):
    """Return a validator(parsed) -> list[str] of schema problems (empty == valid).

    Enforces the contract the judge must follow, so we never accept malformed output
    that only *happens* to parse (e.g. `overall` nested inside `axes`, or a duplicate
    key resolved by luck). Any returned problems trigger a retry of the judge call.
    """
    expected = set(axis_keys)
    field = "winner" if mode == "pairwise" else "score"

    def _validate(parsed: Any) -> list[str]:
        problems: list[str] = []
        if not isinstance(parsed, dict):
            return ["response is not a JSON object"]
        axes = parsed.get("axes")
        if not isinstance(axes, dict):
            problems.append("missing or non-object 'axes'")
            axes = {}
        got = set(axes.keys())
        extra, missing = got - expected, expected - got
        if extra:
            # This is exactly the `overall`-nested-in-`axes` failure mode.
            problems.append(f"unexpected key(s) in 'axes': {sorted(extra)}")
        if missing:
            problems.append(f"missing axis key(s): {sorted(missing)}")
        for k in expected & got:
            if not isinstance(axes[k], dict) or field not in axes[k]:
                problems.append(f"axis '{k}' missing '{field}'")
        overall = parsed.get("overall")
        if not isinstance(overall, dict) or field not in overall:
            problems.append(f"missing/malformed top-level 'overall' (need '{field}')")
        return problems

    return _validate


def call_judge(
    client: httpx.Client,
    api_key: str,
    judge_model: str,
    user_prompt: str,
    validate=None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Return (parsed_json, usage, raw_text).

    Retries on transient HTTP errors AND on schema violations (when `validate` is
    given). Fails fast on 4xx client errors, surfacing the API's response body —
    a 4xx is permanent, so retrying it just hides the cause.
    """
    payload: dict[str, Any] = {
        "model": judge_model,
        "max_tokens": JUDGE_MAX_TOKENS,
        "system": JUDGE_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if JUDGE_TEMPERATURE is not None:
        payload["temperature"] = JUDGE_TEMPERATURE
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.post(ANTHROPIC_URL, headers=headers, json=payload)
            if resp.status_code in (429, 500, 502, 503, 529):
                raise httpx.HTTPStatusError(
                    f"retryable status {resp.status_code}", request=resp.request, response=resp
                )
            if 400 <= resp.status_code < 500:
                # Permanent client error — do NOT retry; surface the API's reason.
                raise RuntimeError(f"judge API {resp.status_code}: {resp.text}")
            resp.raise_for_status()
            data = resp.json()
            text = "".join(
                b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
            )
            parsed = extract_json(text)
            if validate is not None:
                problems = validate(parsed)
                if problems:
                    raise SchemaError("; ".join(problems))
            return parsed, data.get("usage", {}), text
        except (httpx.HTTPError, json.JSONDecodeError, SchemaError) as e:
            last_err = e
            if isinstance(e, SchemaError):
                print(f"  [retry] judge schema violation (attempt {attempt}): {e}", file=sys.stderr)
            if attempt < MAX_RETRIES:
                time.sleep(min(2 ** attempt, 16))
    raise RuntimeError(f"judge call failed after {MAX_RETRIES} attempts: {last_err}")


def acc_usage(total: dict[str, int], usage: dict[str, Any]) -> None:
    total["input_tokens"] = total.get("input_tokens", 0) + int(usage.get("input_tokens", 0) or 0)
    total["output_tokens"] = total.get("output_tokens", 0) + int(usage.get("output_tokens", 0) or 0)
    total["judge_calls"] = total.get("judge_calls", 0) + 1


# --------------------------------------------------------------------------------------
# Pairwise scoring
# --------------------------------------------------------------------------------------
def _map_winner(raw_winner: str, swapped: bool) -> str:
    """Map the judge's A/B (relative to what it was shown) back to model_a/model_b.

    swapped=False: position A = model_a. swapped=True: position A = model_b.
    """
    w = (raw_winner or "tie").lower()
    if w == "tie":
        return "tie"
    if not swapped:
        return "model_a" if w == "a" else "model_b"
    return "model_b" if w == "a" else "model_a"


def score_pairwise(
    client, api_key, judge_model, cap_a, cap_b, rubric, reference_prompt, limit, dry_run
):
    idx_a, idx_b = index_by_id(cap_a), index_by_id(cap_b)
    common = [rid for rid in idx_a if rid in idx_b]
    if limit:
        common = common[:limit]
    axis_keys = [a["key"] for a in rubric]

    per_probe: list[dict[str, Any]] = []
    usage_total: dict[str, int] = {}
    validate = make_validator(axis_keys, "pairwise")
    # Tallies are model-relative (model_a / model_b / tie / order_sensitive).
    axis_tally = {k: {"model_a": 0, "model_b": 0, "tie": 0, "order_sensitive": 0} for k in axis_keys}
    overall_tally = {"model_a": 0, "model_b": 0, "tie": 0, "order_sensitive": 0}

    for rid in common:
        ra, rb = idx_a[rid], idx_b[rid]
        task = ra.get("user_prompt", "")
        ca, cb = ra.get("completion", ""), rb.get("completion", "")

        # Order 1: position A = model_a.  Order 2: position A = model_b (swapped).
        p1 = build_pairwise_prompt(task, ca, cb, rubric, reference_prompt)
        p2 = build_pairwise_prompt(task, cb, ca, rubric, reference_prompt)

        if dry_run:
            print(f"\n===== probe {rid} :: ORDER 1 (A=model_a) =====\n{p1}")
            print(f"\n===== probe {rid} :: ORDER 2 (A=model_b) =====\n{p2}")
            continue

        j1, u1, raw1 = call_judge(client, api_key, judge_model, p1, validate=validate)
        acc_usage(usage_total, u1)
        j2, u2, raw2 = call_judge(client, api_key, judge_model, p2, validate=validate)
        acc_usage(usage_total, u2)

        axes_result: dict[str, Any] = {}
        for k in axis_keys:
            w1 = _map_winner(j1.get("axes", {}).get(k, {}).get("winner", "tie"), swapped=False)
            w2 = _map_winner(j2.get("axes", {}).get(k, {}).get("winner", "tie"), swapped=True)
            if w1 == w2:
                consensus, order_sensitive = w1, False
            else:
                consensus, order_sensitive = "tie", True
            axis_tally[k][consensus] += 1
            if order_sensitive:
                axis_tally[k]["order_sensitive"] += 1
            axes_result[k] = {
                "consensus": consensus,
                "order_sensitive": order_sensitive,
                "order1_winner": w1,
                "order2_winner": w2,
                "order1_rationale": j1.get("axes", {}).get(k, {}).get("rationale", ""),
                "order2_rationale": j2.get("axes", {}).get(k, {}).get("rationale", ""),
            }

        o1 = _map_winner(j1.get("overall", {}).get("winner", "tie"), swapped=False)
        o2 = _map_winner(j2.get("overall", {}).get("winner", "tie"), swapped=True)
        if o1 == o2:
            o_consensus, o_sensitive = o1, False
        else:
            o_consensus, o_sensitive = "tie", True
        overall_tally[o_consensus] += 1
        if o_sensitive:
            overall_tally["order_sensitive"] += 1

        per_probe.append(
            {
                "id": rid,
                "title": ra.get("title", ""),
                "axes": axes_result,
                "overall": {
                    "consensus": o_consensus,
                    "order_sensitive": o_sensitive,
                    "order1_winner": o1,
                    "order2_winner": o2,
                    "order1_confidence": j1.get("overall", {}).get("confidence", ""),
                    "order2_confidence": j2.get("overall", {}).get("confidence", ""),
                    "order1_rationale": j1.get("overall", {}).get("rationale", ""),
                    "order2_rationale": j2.get("overall", {}).get("rationale", ""),
                },
                "_raw": {"order1": raw1, "order2": raw2},
            }
        )
        print(
            f"  scored {rid:<28} overall={o_consensus}"
            + ("  [order-sensitive]" if o_sensitive else "")
        )

    return {
        "probes_scored": len(per_probe),
        "axis_tally": axis_tally,
        "overall_tally": overall_tally,
        "per_probe": per_probe,
        "usage": usage_total,
    }


# --------------------------------------------------------------------------------------
# Pointwise scoring
# --------------------------------------------------------------------------------------
def score_pointwise(client, api_key, judge_model, cap, rubric, reference_prompt, limit, dry_run):
    records = cap["results"][:limit] if limit else cap["results"]
    axis_keys = [a["key"] for a in rubric]
    per_probe: list[dict[str, Any]] = []
    usage_total: dict[str, int] = {}
    sums = {k: 0.0 for k in axis_keys}
    overall_sum = 0.0
    validate = make_validator(axis_keys, "pointwise")

    for r in records:
        rid = r.get("id")
        prompt = build_pointwise_prompt(
            r.get("user_prompt", ""), r.get("completion", ""), rubric, reference_prompt
        )
        if dry_run:
            print(f"\n===== probe {rid} =====\n{prompt}")
            continue
        j, u, raw = call_judge(client, api_key, judge_model, prompt, validate=validate)
        acc_usage(usage_total, u)
        axes = {}
        for k in axis_keys:
            sc = j.get("axes", {}).get(k, {})
            score = sc.get("score")
            if isinstance(score, (int, float)):
                sums[k] += score
            axes[k] = {"score": score, "rationale": sc.get("rationale", "")}
        ov = j.get("overall", {})
        if isinstance(ov.get("score"), (int, float)):
            overall_sum += ov["score"]
        per_probe.append({"id": rid, "title": r.get("title", ""), "axes": axes,
                          "overall": ov, "_raw": raw})
        print(f"  scored {rid:<28} overall={ov.get('score')}")

    n = max(len(per_probe), 1)
    return {
        "probes_scored": len(per_probe),
        "axis_means": {k: round(sums[k] / n, 3) for k in axis_keys},
        "overall_mean": round(overall_sum / n, 3),
        "per_probe": per_probe,
        "usage": usage_total,
    }


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="LLM-judge quality scorer for operator-copilot RCA captures (multi-model)."
    )
    ap.add_argument("--mode", choices=["pairwise", "pointwise"], default="pairwise")
    ap.add_argument("--a", help="pairwise: result JSON for model A (or the single file for pointwise)")
    ap.add_argument("--b", help="pairwise: result JSON for model B")
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                    help="Judge model id sent to the Anthropic API and recorded everywhere.")
    ap.add_argument("--reference-prompt",
                    help="Path to the operator-copilot system prompt; given to the judge as "
                         "grounds for scope/guardrail scoring. Recommended. Relative paths "
                         "resolve against the CWD then the tool repo (bundled prompts/).")
    ap.add_argument("--rubric-file",
                    help="JSON overriding the default rubric axes. Relative paths resolve "
                         "against the CWD then the tool repo (bundled rubrics/).")
    ap.add_argument("--results-dir", default=".", help="Directory for the output JSON.")
    ap.add_argument("--output", help="Explicit output path (default includes model + judge names).")
    ap.add_argument("--limit", type=int, default=0, help="Score only the first N probes (debug/cost).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the assembled judge prompts and exit without calling the API.")
    args = ap.parse_args()

    if args.mode == "pairwise" and (not args.a or not args.b):
        ap.error("pairwise mode requires --a and --b")
    if args.mode == "pointwise" and not args.a:
        ap.error("pointwise mode requires --a (the single result file)")

    rubric = load_rubric(args.rubric_file)
    reference_prompt = resolve_input(args.reference_prompt).read_text() if args.reference_prompt else None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: ANTHROPIC_API_KEY not set in environment.", file=sys.stderr)
        return 2

    cap_a = load_capture(Path(args.a))
    cap_b = load_capture(Path(args.b)) if args.b else None

    git = tool_provenance()
    stamp = utc_stamp()
    print("=" * 72)
    print("  RCA Quality Judge")
    print("=" * 72)
    print(f"  mode         : {args.mode}")
    print(f"  judge model  : {args.judge_model}  (temp={JUDGE_TEMPERATURE})")
    print(f"  model A      : {cap_a['model']}  [{Path(args.a).name}]")
    if cap_b:
        print(f"  model B      : {cap_b['model']}  [{Path(args.b).name}]")
    print(f"  rubric axes  : {', '.join(a['key'] for a in rubric)}")
    print(f"  reference    : {'yes' if reference_prompt else 'no'}")
    print(f"  tool git_sha : {git['tool_git_sha']}  dirty={git['tool_git_dirty']}")
    if git["tool_git_dirty"] and not args.dry_run:
        print("  WARNING: working tree is dirty -- commit the tool before recording results.")
    print("=" * 72)

    client = None if args.dry_run else httpx.Client(timeout=HTTP_TIMEOUT)
    try:
        if args.mode == "pairwise":
            scored = score_pairwise(client, api_key, args.judge_model, cap_a, cap_b,
                                    rubric, reference_prompt, args.limit, args.dry_run)
        else:
            scored = score_pointwise(client, api_key, args.judge_model, cap_a,
                                     rubric, reference_prompt, args.limit, args.dry_run)
    finally:
        if client:
            client.close()

    if args.dry_run:
        print("\n[dry-run] no API calls made; no output written.")
        return 0

    out = {
        "schema_version": 1,
        "tool": "rca_quality_judge.py",
        "mode": args.mode,
        "timestamp_utc": stamp,
        "judge_model": args.judge_model,
        "judge_temperature": JUDGE_TEMPERATURE,
        "judge_max_tokens": JUDGE_MAX_TOKENS,
        "position_bias_control": "both-orders" if args.mode == "pairwise" else "n/a",
        "rubric": rubric,
        "reference_prompt_used": bool(reference_prompt),
        **git,
        "inputs": {
            "a": provenance_of(cap_a, Path(args.a)),
            **({"b": provenance_of(cap_b, Path(args.b))} if cap_b else {}),
        },
        "summary": {k: v for k, v in scored.items() if k not in ("per_probe",)},
        "per_probe": scored["per_probe"],
    }

    if args.output:
        out_path = Path(args.output)
    else:
        if args.mode == "pairwise":
            fname = (f"judge_pairwise_{sanitize(cap_a['model'])}_vs_"
                     f"{sanitize(cap_b['model'])}_by_{sanitize(args.judge_model)}_{stamp}.json")
        else:
            fname = (f"judge_pointwise_{sanitize(cap_a['model'])}_"
                     f"by_{sanitize(args.judge_model)}_{stamp}.json")
        out_path = Path(args.results_dir) / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print("\n--- summary ---")
    if args.mode == "pairwise":
        ot = scored["overall_tally"]
        print(f"  overall: model_a={ot['model_a']}  model_b={ot['model_b']}  "
              f"tie={ot['tie']}  (order-sensitive: {ot['order_sensitive']})")
        print(f"  model_a = {cap_a['model']}")
        print(f"  model_b = {cap_b['model']}")
    else:
        print(f"  overall mean: {scored['overall_mean']}")
        print(f"  axis means  : {scored['axis_means']}")
    u = scored["usage"]
    print(f"  judge calls : {u.get('judge_calls', 0)}  "
          f"tokens in/out: {u.get('input_tokens', 0)}/{u.get('output_tokens', 0)}")
    print(f"\n  wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())