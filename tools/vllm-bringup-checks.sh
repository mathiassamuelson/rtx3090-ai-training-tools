#!/usr/bin/env bash
# vllm-bringup-checks.sh — post-launch verification gates for a vLLM model serve.
#
# Runs the standard checks we perform after every model bring-up, in order:
#   1. container is Running (docker inspect)
#   2. PHYSICAL GPU placement by PID-join (docker top  x  nvidia-smi compute-apps) — never
#      trusts --gpus intent; maps the container's real PIDs to physical GPU indices
#   3. startup-log scan (--log): hard errors / tracebacks / CUDA-OOM, patch_dense signature,
#      whether quantization is actually active (catches the --hf-overrides shallow-replace
#      trap), and the KV-cache allocation lines
#   4. /v1/models reachable; served model id reported (asserted vs --expected-model if given)
#   5. chat-endpoint smoke: one /v1/chat/completions call (Gemma 4 degenerates on raw
#      /v1/completions), confirming the model is usable and warming it
#
# Exit code: 0 if all CRITICAL checks pass, else 1. WARN-level findings do not fail the run.
#
# Usage:
#   tools/vllm-bringup-checks.sh --name gemma4-12b-native-test --port 8001 \
#       --expected-model google/gemma-4-12B-it-qat-w4a16-ct \
#       --expected-gpus 1 --log /tmp/12b_v0230_native_startup.log
#
#   tools/vllm-bringup-checks.sh --name gemma4-12b-bf16 --port 8002 \
#       --expected-gpus 1,3 --no-smoke      # e.g. while still warming

set -uo pipefail

NAME=""
HOST="127.0.0.1"
PORT="8000"
EXPECTED_MODEL=""
EXPECTED_GPUS=""
LOG=""
SMOKE=1
SMOKE_MAX_TOKENS=16

CRIT_FAIL=0
WARN_COUNT=0

pass() { printf '  [PASS] %s\n' "$*"; }
info() { printf '  [INFO] %s\n' "$*"; }
warn() { printf '  [WARN] %s\n' "$*"; WARN_COUNT=$((WARN_COUNT + 1)); }
fail() { printf '  [FAIL] %s\n' "$*"; CRIT_FAIL=$((CRIT_FAIL + 1)); }
hr()   { printf '%s\n' "------------------------------------------------------------------------"; }

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --name)            NAME="$2"; shift 2;;
    --host)            HOST="$2"; shift 2;;
    --port)            PORT="$2"; shift 2;;
    --expected-model)  EXPECTED_MODEL="$2"; shift 2;;
    --expected-gpus)   EXPECTED_GPUS="$2"; shift 2;;
    --log)             LOG="$2"; shift 2;;
    --smoke)           SMOKE=1; shift;;
    --no-smoke)        SMOKE=0; shift;;
    --smoke-max-tokens) SMOKE_MAX_TOKENS="$2"; shift 2;;
    -h|--help)         usage;;
    *) printf 'unknown arg: %s\n' "$1" >&2; usage;;
  esac
done

[ -n "$NAME" ] || { printf 'ERROR: --name is required\n' >&2; exit 2; }
for bin in docker nvidia-smi curl python3; do
  command -v "$bin" >/dev/null 2>&1 || { printf 'ERROR: missing required tool: %s\n' "$bin" >&2; exit 2; }
done

BASE="http://${HOST}:${PORT}/v1"

printf '========================================================================\n'
printf '  vLLM Bring-up Checks\n'
printf '========================================================================\n'
printf '  container : %s\n' "$NAME"
printf '  endpoint  : %s\n' "$BASE"
printf '  expect mdl: %s\n' "${EXPECTED_MODEL:-(not asserted)}"
printf '  expect gpu: %s\n' "${EXPECTED_GPUS:-(not asserted)}"
printf '  log       : %s\n' "${LOG:-(none)}"
hr

# ---- 1. container running -------------------------------------------------------------
printf '1. Container state\n'
RUNNING="$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null || true)"
if [ "$RUNNING" != "true" ]; then
  fail "container '$NAME' is not running (state=${RUNNING:-absent})"
  hr; printf 'Aborting: cannot check a container that is not up.\n'; exit 1
fi
CID="$(docker inspect -f '{{.Id}}' "$NAME")"
STATUS="$(docker inspect -f '{{.State.Status}}' "$NAME")"
pass "container running (status=$STATUS, id=${CID:0:12})"
hr

# ---- 2. physical GPU placement by PID-join --------------------------------------------
printf '2. GPU placement (PID-join, empirical)\n'
# host PIDs belonging to the container
CONTAINER_PIDS="$(docker top "$NAME" 2>/dev/null | awk 'NR>1{print $2}' | sort -u)"
if [ -z "$CONTAINER_PIDS" ]; then
  warn "docker top returned no PIDs for the container"
fi
# uuid -> physical index map
declare -A UUID2IDX
while IFS=',' read -r idx uuid; do
  idx="$(echo "$idx" | tr -d ' ')"; uuid="$(echo "$uuid" | tr -d ' ')"
  [ -n "$uuid" ] && UUID2IDX["$uuid"]="$idx"
done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null)

# compute-apps: which PID is on which uuid, with mem
FOUND_GPUS=""
ANY_APP=0
while IFS=',' read -r pid uuid mem; do
  pid="$(echo "$pid" | tr -d ' ')"; uuid="$(echo "$uuid" | tr -d ' ')"; mem="$(echo "$mem" | tr -d ' ')"
  [ -n "$pid" ] || continue
  ANY_APP=1
  if echo "$CONTAINER_PIDS" | grep -qx "$pid"; then
    idx="${UUID2IDX[$uuid]:-?}"
    info "pid $pid -> GPU $idx ($uuid, ${mem} MiB)"
    FOUND_GPUS="$FOUND_GPUS $idx"
  fi
done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits 2>/dev/null)

FOUND_GPUS="$(echo "$FOUND_GPUS" | tr ' ' '\n' | grep -v '^$' | sort -un | tr '\n' ',' | sed 's/,$//')"
if [ -z "$FOUND_GPUS" ]; then
  if [ "$ANY_APP" -eq 0 ]; then
    warn "no GPU compute processes visible yet — model may still be loading; re-run shortly"
  else
    fail "container has no GPU compute processes (PIDs did not join any nvidia-smi app)"
  fi
else
  pass "container occupies physical GPU(s): $FOUND_GPUS"
  if [ -n "$EXPECTED_GPUS" ]; then
    want="$(echo "$EXPECTED_GPUS" | tr ',' '\n' | sort -un | tr '\n' ',' | sed 's/,$//')"
    if [ "$FOUND_GPUS" = "$want" ]; then
      pass "placement matches --expected-gpus ($want)"
    else
      fail "placement mismatch: on [$FOUND_GPUS], expected [$want]"
    fi
  fi
fi
hr

# ---- 3. startup-log scan --------------------------------------------------------------
if [ -n "$LOG" ]; then
  printf '3. Startup-log scan (%s)\n' "$LOG"
  if [ ! -f "$LOG" ]; then
    warn "log file not found: $LOG"
  else
    ERR_HITS="$(grep -nEi 'traceback|cuda out of memory|oom-?kill|RuntimeError|AssertionError|failed to|cannot allocate' "$LOG" | head -n 12)"
    if [ -n "$ERR_HITS" ]; then
      warn "hard-error signatures present in log:"
      printf '%s\n' "$ERR_HITS" | sed 's/^/         /'
    else
      pass "no hard-error signatures (traceback / OOM / alloc failure)"
    fi
    if grep -qiE 'patch[._]dense' "$LOG"; then
      warn "'patch_dense' appears in log — verify it is not the ignore-list bug:"
      grep -niE 'patch[._]dense' "$LOG" | head -n 4 | sed 's/^/         /'
    else
      pass "no patch_dense signature"
    fi
    QUANT_HITS="$(grep -niE 'quantization|w4a16|compressed.?tensors|awq|gptq|int4' "$LOG" | head -n 6)"
    if [ -n "$QUANT_HITS" ]; then
      info "quantization lines (confirm quant is ACTIVE, not silently disabled):"
      printf '%s\n' "$QUANT_HITS" | sed 's/^/         /'
    else
      warn "no quantization lines found — if this is a QAT/quantized model, suspect the"
      warn "  --hf-overrides shallow-replace trap (quant silently dropped)"
    fi
    KV_HITS="$(grep -niE 'kv cache|gpu blocks|maximum concurrency|gpu memory' "$LOG" | head -n 6)"
    if [ -n "$KV_HITS" ]; then
      info "KV-cache / capacity lines:"
      printf '%s\n' "$KV_HITS" | sed 's/^/         /'
    fi
  fi
  hr
else
  printf '3. Startup-log scan: skipped (no --log)\n'; hr
fi

# ---- 4. /v1/models --------------------------------------------------------------------
printf '4. /v1/models endpoint\n'
MODELS_JSON="$(curl -sS -m 15 "${BASE}/models" 2>/dev/null || true)"
if [ -z "$MODELS_JSON" ]; then
  fail "no response from ${BASE}/models (server not ready?)"
  SERVED=""
else
  SERVED="$(printf '%s' "$MODELS_JSON" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
    print(d["data"][0]["id"])
except Exception:
    print("")
' 2>/dev/null)"
  if [ -z "$SERVED" ]; then
    fail "could not parse a served model id from /v1/models"
  else
    pass "served model id: $SERVED"
    if [ -n "$EXPECTED_MODEL" ]; then
      if [ "$SERVED" = "$EXPECTED_MODEL" ]; then
        pass "served id matches --expected-model"
      else
        fail "served id '$SERVED' != expected '$EXPECTED_MODEL'"
      fi
    fi
  fi
fi
hr

# ---- 5. chat-endpoint smoke -----------------------------------------------------------
if [ "$SMOKE" -eq 1 ]; then
  printf '5. Chat-endpoint smoke (Gemma 4 requires /v1/chat/completions)\n'
  SMOKE_MODEL="${SERVED:-${EXPECTED_MODEL:-model}}"
  PAYLOAD="$(python3 -c '
import json, sys
print(json.dumps({
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": "Reply with exactly the word: ready"}],
    "temperature": 0, "top_p": 1, "max_tokens": int(sys.argv[2]), "stream": False,
}))' "$SMOKE_MODEL" "$SMOKE_MAX_TOKENS")"
  T0="$(date +%s.%N)"
  SMOKE_RESP="$(curl -sS -m 60 -X POST "${BASE}/chat/completions" \
      -H 'Content-Type: application/json' -d "$PAYLOAD" 2>/dev/null || true)"
  T1="$(date +%s.%N)"
  if [ -z "$SMOKE_RESP" ]; then
    fail "no response from chat endpoint"
  else
    printf '%s' "$SMOKE_RESP" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
    c = d["choices"][0]
    txt = (c["message"]["content"] or "").strip().replace("\n", " ")
    fin = c.get("finish_reason")
    if not txt:
        print("FAIL\tempty assistant message"); sys.exit(0)
    print("PASS\t%r (finish=%s)" % (txt[:120], fin))
except Exception as e:
    print("FAIL\tunparseable chat response: %s" % e)
' | while IFS=$'\t' read -r verdict detail; do
        if [ "$verdict" = "PASS" ]; then pass "chat reply $detail"; else fail "chat smoke: $detail"; fi
      done
    DT="$(python3 -c 'import sys; print("%.2f" % (float(sys.argv[2])-float(sys.argv[1])))' "$T0" "$T1" 2>/dev/null || echo '?')"
    info "chat round-trip: ${DT}s"
  fi
  hr
else
  printf '5. Chat-endpoint smoke: skipped (--no-smoke)\n'; hr
fi

# ---- verdict --------------------------------------------------------------------------
printf 'Summary: %d critical failure(s), %d warning(s)\n' "$CRIT_FAIL" "$WARN_COUNT"
if [ "$CRIT_FAIL" -eq 0 ]; then
  printf 'RESULT: PASS — bring-up checks clear.\n'; exit 0
else
  printf 'RESULT: FAIL — resolve critical findings before capturing.\n'; exit 1
fi
