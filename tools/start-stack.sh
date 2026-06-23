#!/usr/bin/env bash
#
# start-stack.sh — boot-choreography orchestrator for the multi-tier inference layout.
#
# Brings up the full serving stack — two 12B QAT workers + one 31B TP=2 orchestrator —
# in either STAGGERED or SIMULTANEOUS order, probes each tier to time-to-healthy via the
# CHAT endpoint (raw /v1/completions triggers Gemma-4 token-repetition gibberish — never
# use it), verifies GPU placement empirically (nvidia-smi index->uuid->busy join), captures
# steady-state host RSS, and writes a self-describing results JSON.
#
# Multi-model by design: model identities are CLI args with defaults, propagated into the
# probe payloads, console headers, placement records, JSON metadata, and the DEFAULT output
# filename. A different model set, or a different mode, never overwrites a prior result.
#
# Usage:
#   tools/start-stack.sh staggered
#   tools/start-stack.sh simultaneous
#   tools/start-stack.sh teardown
#   tools/start-stack.sh staggered --week week-14        # results -> phase-3.../week-14/results/
#   tools/start-stack.sh staggered --image vllm/vllm-openai:v0.23.0   # override the single image
#   tools/start-stack.sh staggered --host-label my-box   # tag results with a host id (default: omitted)
#   tools/start-stack.sh staggered --model-12b <id> --model-31b <id> --mml-12b 131072 \
#                                  --mml-31b 33024 --util-12b 0.90 --util-31b 0.95
#
# Results path: phase-3-optimization-and-quantization/<week>/results/   (--week; default week-13)
#
# ============================================================================================
# >>> LAUNCHER SEAM — one native launcher (start-vllm.sh) now serves BOTH tiers <<<
#   start-vllm.sh : --model --mode tp --size N --gpus --port --max-model-len --gpu-mem-util \
#                   --image --name
#   Convergence (Week 13 Day 4/5): the gemma4-unified scaffolding launcher (start-12b-qat.sh)
#   and its source patch are RETIRED — v0.23.0 boots the 12B natively (Marlin INT4, no source
#   patch, no --hf-overrides blob). Workers are TP=1 (--mode tp --size 1 --gpus <single id>);
#   orchestrator is TP=2 (--mode tp --size 2 --gpus 0,2). Both tiers run the SAME image, pinned
#   here and threaded to each launcher via --image.
#   Notes: 31B launcher MML default is 131072 (provisional) -> we pass 33024 (Week 11 baseline).
#          Both workers need DISTINCT --name (start-vllm.sh derives NAME=vllm-tp1 for both ->
#          the 2nd docker run collides without distinct names). Util flag is --gpu-mem-util.
# ============================================================================================

set -euo pipefail

# ---- defaults (all overridable on the CLI; none hardcoded downstream) ----------------------
MODEL_12B="google/gemma-4-12B-it-qat-w4a16-ct"
MODEL_31B="RedHatAI/gemma-4-31B-it-FP8-block"
MML_12B=131072
MML_31B=33024
UTIL_12B=0.90           # 12B native (v0.23.0) verified at util 0.90
UTIL_31B=0.95           # MML 33024 needs 0.95: the cudagraph-profiling tax (persists on the
                        # standard runner under v0.23.0) cuts effective util to ~0.9093, so a
                        # nominal 0.90 leaves too little KV to admit a 33024 pool. Week 11
                        # baseline 33024 was characterized at 0.95; Day 4 re-baseline confirmed.
PORT_W1=8001
PORT_W2=8003
PORT_ORCH=8000
GPUS_W1="1"
GPUS_W2="3"
GPUS_ORCH="0,2"
NAME_W1="gemma4-12b-qat-gpu1"   # distinct names required: start-vllm.sh derives NAME=vllm-tp1
NAME_W2="gemma4-12b-qat-gpu3"   # for BOTH workers -> the 2nd docker run collides without these
NAME_ORCH="gemma4-31b-tp2"
LAUNCHER_12B="tools/start-vllm.sh"   # convergence: native launcher serves both tiers (was start-12b-qat.sh)
LAUNCHER_31B="tools/start-vllm.sh"
IMAGE="vllm/vllm-openai:v0.23.0"     # ONE image, both tiers (Week 13 convergence).
                                     # digest: sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f
                                     # pre-session check verifies this tag resolves to that digest.
PROBE_INTERVAL=2
PROBE_TIMEOUT=420
WEEK="week-13"          # phase-3 week subdir for the default results path; override with --week
RESULTS_DIR=""          # resolved below relative to repo root unless overridden
OUT_FILE=""             # default computed from models + mode
HOST_CAPTURE=1          # iostat/free background sampling
TEARDOWN_FIRST=0
HOST_LABEL=""           # optional host identifier recorded in the results JSON. Default empty ->
                        # the "host" field is omitted. Set explicitly (--host-label) when you want
                        # to tag a run; never auto-scraped from the environment (no hostname leak).

usage() { sed -n '2,40p' "$0"; exit "${1:-0}"; }

# ---- arg parse -----------------------------------------------------------------------------
MODE="${1:-}"; shift || true
case "$MODE" in staggered|simultaneous|teardown) ;; ""|-h|--help) usage 0 ;; *) echo "ERROR: unknown mode '$MODE'"; usage 1 ;; esac

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-12b) MODEL_12B="$2"; shift 2 ;;
    --model-31b) MODEL_31B="$2"; shift 2 ;;
    --mml-12b)   MML_12B="$2";   shift 2 ;;
    --mml-31b)   MML_31B="$2";   shift 2 ;;
    --util-12b)  UTIL_12B="$2";  shift 2 ;;
    --util-31b)  UTIL_31B="$2";  shift 2 ;;
    --image)     IMAGE="$2";     shift 2 ;;
    --week)        WEEK="$2";        shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --out)       OUT_FILE="$2";  shift 2 ;;
    --probe-interval) PROBE_INTERVAL="$2"; shift 2 ;;
    --probe-timeout)  PROBE_TIMEOUT="$2";  shift 2 ;;
    --host-label)  HOST_LABEL="$2"; shift 2 ;;
    --no-host-capture) HOST_CAPTURE=0; shift ;;
    --teardown-first)  TEARDOWN_FIRST=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "ERROR: unknown arg '$1'"; usage 1 ;;
  esac
done

# ---- helpers -------------------------------------------------------------------------------
now()     { date +%s.%N; }
elapsed() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.2f", b-a}'; }   # b - a
slugify() { basename "$1" | tr '/:@' '___' | tr -cs 'A-Za-z0-9._-' '-' | sed 's/-*$//'; }
ts_utc()  { date -u +%Y-%m-%dT%H:%M:%SZ; }    # ISO 8601 — for JSON metadata fields
ts_file() { date -u +%Y%m%dT%H%M%SZ; }        # colon-free compact stamp — for default filenames
log()     { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# Anchor to the SCRIPT's own location, NOT cwd. tools/ is one level under the repo root.
# This is invariant to (a) the directory the script is invoked from and (b) where the repo
# is cloned. The prior `git rev-parse --show-toplevel` resolved the toplevel of whichever
# checkout cwd happened to sit in — so running this from inside a sibling repo (e.g. a separate
# stack/config checkout) retargeted RESULTS_DIR, GIT_SHA, and the launcher paths at the wrong
# repo. (BASH_SOURCE
# is reliable here because the script is always invoked as a file, never sourced/piped.)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[[ -z "$RESULTS_DIR" ]] && RESULTS_DIR="$ROOT/phase-3-optimization-and-quantization/$WEEK/results"
mkdir -p "$RESULTS_DIR"

GIT_SHA="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
# Dirty-tree check EXCLUDES the results dir: result files are expected to be uncommitted at
# write time, so a dirty results/ is normal and not a provenance problem. Only changes OUTSIDE
# results/ (code, tools, configs) mean the recorded SHA won't reflect what actually ran. The
# exclude is applied only when results/ sits inside the repo; an external --results-dir isn't
# in the tree anyway, so the plain whole-tree check still applies there.
DIRTY_PATHSPEC=(.)
case "$RESULTS_DIR" in "$ROOT"/*) DIRTY_PATHSPEC=(. ":(exclude)${RESULTS_DIR#"$ROOT"/}") ;; esac
if [[ -n "$(git -C "$ROOT" status --porcelain -- "${DIRTY_PATHSPEC[@]}" 2>/dev/null)" ]]; then
  log "WARNING: working tree is DIRTY outside results/ — recorded git SHA ($GIT_SHA) will not be clean."
  log "         Commit code/tool changes before a measured run (commit-before-running discipline)."
fi

# functional health probe — CHAT endpoint only (raw completions => Gemma-4 gibberish)
probe_healthy() {  # $1=port  $2=model
  local code body="/tmp/stack_probe_$1.json"
  code=$(curl -s -o "$body" -w '%{http_code}' --max-time 10 \
    -X POST "http://127.0.0.1:$1/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$2\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":8,\"temperature\":0}" \
    2>/dev/null) || return 1
  [[ "$code" == "200" ]] && grep -q '"choices"' "$body"
}

# Container lookup by the deterministic per-tier --name (NOT by published port).
# start-vllm.sh launches with --network host and no `-p` mapping, so host-network containers
# publish nothing and a `--filter publish=` query returns empty -> image_digest would be
# UNKNOWN for every tier under the converged single-launcher layout. Exact-matching the name
# is robust across host- and bridge-network launchers and across docker's name-storage quirks.
container_for_name() {  # $1=exact container name
  docker ps --format '{{.ID}} {{.Names}}' | awk -v n="$1" '$2==n{print $1; exit}'
}
image_tag_for_container()    { docker inspect --format '{{.Config.Image}}' "$1" 2>/dev/null; }
image_digest_for_tag()       { docker image inspect --format '{{if .RepoDigests}}{{index .RepoDigests 0}}{{else}}{{.Id}}{{end}}' "$1" 2>/dev/null || echo UNKNOWN; }

# ---- teardown ------------------------------------------------------------------------------
teardown_stack() {
  log "Tearing down vLLM containers (observability stack untouched)..."
  local ids
  ids=$(docker ps --format '{{.ID}} {{.Image}}' | awk '$2 ~ /vllm\/vllm-openai/ {print $1}')
  if [[ -n "$ids" ]]; then docker stop $ids >/dev/null; log "Stopped: $ids"; else log "No vLLM containers resident."; fi
  log "GPU state after teardown:"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
}

if [[ "$MODE" == "teardown" ]]; then teardown_stack; exit 0; fi
[[ "$TEARDOWN_FIRST" == "1" ]] && teardown_stack

# ============================================================================================
# >>> EDITABLE SEAM: launcher invocations. Confirm flag names against the real launcher. <<<
# Each launch runs in the background so the orchestrator can probe regardless of whether the
# launcher blocks (foreground docker run) or detaches.
#
# Both tiers now route through the SAME native launcher (start-vllm.sh):
#   workers      -> --mode tp --size 1 --gpus <single id>   (single-GPU TP=1, x1 link irrelevant)
#   orchestrator -> --mode tp --size 2 --gpus 0,2           (NVLink pair)
# --image threads the converged image to both. No patch mount, no --hf-overrides.
# ============================================================================================
launch_12b() {  # $1=gpu  $2=port  $3=model  $4=mml  $5=util  $6=name
  "$ROOT/$LAUNCHER_12B" \
    --model "$3" \
    --mode tp --size 1 \
    --gpus "$1" \
    --port "$2" \
    --max-model-len "$4" \
    --gpu-mem-util "$5" \
    --image "$IMAGE" \
    --name "$6"
}
launch_31b() {  # $1=gpus  $2=port  $3=model  $4=mml  $5=util  $6=name
  "$ROOT/$LAUNCHER_31B" \
    --model "$3" \
    --mode tp --size 2 \
    --gpus "$1" \
    --port "$2" \
    --max-model-len "$4" \
    --gpu-mem-util "$5" \
    --image "$IMAGE" \
    --name "$6"
}
# ============================================================================================

# ---- service table -------------------------------------------------------------------------
# tier|kind|gpus|port|name
SPECS=(
  "worker1|12b|$GPUS_W1|$PORT_W1|$NAME_W1"
  "worker2|12b|$GPUS_W2|$PORT_W2|$NAME_W2"
  "orchestrator|31b|$GPUS_ORCH|$PORT_ORCH|$NAME_ORCH"
)
declare -A MODEL_OF MML_OF UTIL_OF GPUS_OF PORT_OF NAME_OF LAUNCH_T HEALTHY_T
for spec in "${SPECS[@]}"; do
  IFS='|' read -r tier kind gpus port name <<<"$spec"
  GPUS_OF[$tier]="$gpus"; PORT_OF[$tier]="$port"; NAME_OF[$tier]="$name"
  if [[ "$kind" == "12b" ]]; then
    MODEL_OF[$tier]="$MODEL_12B"; MML_OF[$tier]="$MML_12B"; UTIL_OF[$tier]="$UTIL_12B"
  else
    MODEL_OF[$tier]="$MODEL_31B"; MML_OF[$tier]="$MML_31B"; UTIL_OF[$tier]="$UTIL_31B"
  fi
done

launch_one() {  # $1=tier  $2=kind
  local tier="$1" kind="$2"
  LAUNCH_T[$tier]="$(now)"
  log "LAUNCH  $tier  model=${MODEL_OF[$tier]}  gpus=${GPUS_OF[$tier]}  port=${PORT_OF[$tier]}  mml=${MML_OF[$tier]}  util=${UTIL_OF[$tier]}  name=${NAME_OF[$tier]}"
  if [[ "$kind" == "12b" ]]; then
    launch_12b "${GPUS_OF[$tier]}" "${PORT_OF[$tier]}" "${MODEL_OF[$tier]}" "${MML_OF[$tier]}" "${UTIL_OF[$tier]}" "${NAME_OF[$tier]}" \
      >"$RESULTS_DIR/launch_${tier}.log" 2>&1 &
  else
    launch_31b "${GPUS_OF[$tier]}" "${PORT_OF[$tier]}" "${MODEL_OF[$tier]}" "${MML_OF[$tier]}" "${UTIL_OF[$tier]}" "${NAME_OF[$tier]}" \
      >"$RESULTS_DIR/launch_${tier}.log" 2>&1 &
  fi
}

wait_healthy() {  # poll the given tiers until all healthy or timeout
  local pending=("$@") start; start="$(now)"
  while [[ ${#pending[@]} -gt 0 ]]; do
    local still=()
    for tier in "${pending[@]}"; do
      if probe_healthy "${PORT_OF[$tier]}" "${MODEL_OF[$tier]}"; then
        HEALTHY_T[$tier]="$(now)"
        log "HEALTHY $tier  (t2h=$(elapsed "${LAUNCH_T[$tier]}" "${HEALTHY_T[$tier]}")s)"
      else
        still+=("$tier")
      fi
    done
    pending=("${still[@]}")
    [[ ${#pending[@]} -eq 0 ]] && break
    # proper elapsed>timeout test via awk EXIT CODE.
    # (prior `print (n-s)>TIMEOUT` was awk file-redirection, not a comparison — it never fired.)
    if awk -v s="$start" -v n="$(now)" -v t="$PROBE_TIMEOUT" 'BEGIN{exit !((n-s)>t)}'; then
      log "ERROR: probe timeout (${PROBE_TIMEOUT}s); still unhealthy: ${pending[*]}"
      return 1
    fi
    sleep "$PROBE_INTERVAL"
  done
}

# ---- host-resource capture -----------------------------------------------------------------
CAP_PIDS=()
start_host_capture() {  # $1=stem
  [[ "$HOST_CAPTURE" == "1" ]] || return 0
  if command -v iostat >/dev/null 2>&1; then
    iostat -x 1 >"$RESULTS_DIR/${1}_iostat.log" 2>&1 & CAP_PIDS+=("$!")
  else
    log "NOTE: iostat (sysstat) absent — relying on node-exporter/Prometheus for disk metrics."
  fi
  free -m -s 1 >"$RESULTS_DIR/${1}_free.log" 2>&1 & CAP_PIDS+=("$!")
}
stop_host_capture() { for p in "${CAP_PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap stop_host_capture EXIT

# ---- placement verification (index->uuid->busy join) ---------------------------------------
declare -A GPU_UUID GPU_USEDMB
load_gpu_maps() {
  while IFS=',' read -r idx uuid; do GPU_UUID[$(echo "$idx"|xargs)]="$(echo "$uuid"|xargs)"; done \
    < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
  while IFS=',' read -r idx used; do GPU_USEDMB[$(echo "$idx"|xargs)]="$(echo "$used"|tr -dc '0-9')"; done \
    < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader)
}

# ---- run -----------------------------------------------------------------------------------
# Default STEM carries a UTC timestamp so repeated boots of the same model pair never
# overwrite a prior (possibly committed) result — the Day-3 clobber. An explicit --out
# bypasses this: if you name the file, you own any overwrite.
STEM="boot_choreography_${MODE}_workers-$(slugify "$MODEL_12B")_orch-$(slugify "$MODEL_31B")_$(ts_file)"
[[ -n "$OUT_FILE" ]] && STEM="$(basename "${OUT_FILE%.json}")"
JSON_OUT="$RESULTS_DIR/${STEM}.json"

T0="$(now)"
FREE_START_MB="$(free -m | awk '/^Mem:/{print $7}')"   # 'available' column
start_host_capture "$STEM"
log "MODE=$MODE  t0 set.  free(avail)=${FREE_START_MB}MB  image=$IMAGE  results=$RESULTS_DIR"

if [[ "$MODE" == "staggered" ]]; then
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r tier kind _ _ <<<"$spec"
    launch_one "$tier" "$kind"
    wait_healthy "$tier" || { log "Aborting after $tier failed to come healthy."; break; }
  done
else  # simultaneous
  for spec in "${SPECS[@]}"; do IFS='|' read -r tier kind _ _ <<<"$spec"; launch_one "$tier" "$kind"; done
  wait_healthy worker1 worker2 orchestrator || log "One or more tiers failed to come healthy."
fi

sleep 3   # let RSS settle
load_gpu_maps
FREE_STEADY_MB="$(free -m | awk '/^Mem:/{print $7}')"
SWAP_USED_MB="$(free -m | awk '/^Swap:/{print $3}')"

# per-container steady-state memory
DOCKER_STATS_JSON="$(docker stats --no-stream --format '{"name":"{{.Name}}","mem":"{{.MemUsage}}","perc":"{{.MemPerc}}"}' \
  $(docker ps --format '{{.ID}} {{.Image}}' | awk '$2 ~ /vllm\/vllm-openai/ {print $1}') 2>/dev/null | paste -sd, - || echo "")"

# ---- assemble services.tsv -----------------------------------------------------------------
TSV="$(mktemp)"
for spec in "${SPECS[@]}"; do
  IFS='|' read -r tier kind gpus port name <<<"$spec"
  lt="${LAUNCH_T[$tier]:-}"; ht="${HEALTHY_T[$tier]:-}"
  offset="NA"; t2h="NA"
  [[ -n "$lt" ]] && offset="$(elapsed "$T0" "$lt")"
  [[ -n "$lt" && -n "$ht" ]] && t2h="$(elapsed "$lt" "$ht")"
  cid="$(container_for_name "$name")"
  digest="UNKNOWN"; [[ -n "$cid" ]] && digest="$(image_digest_for_tag "$(image_tag_for_container "$cid")")"
  # placement: every intended GPU index should be busy (used memory well above driver idle)
  pok="true"; obs_uuids=""
  IFS=',' read -ra idxs <<<"$gpus"
  for i in "${idxs[@]}"; do
    used="${GPU_USEDMB[$i]:-0}"; uuid="${GPU_UUID[$i]:-NA}"
    obs_uuids="${obs_uuids:+$obs_uuids;}${i}:${uuid}:${used}MB"
    [[ "${used:-0}" -lt 1000 ]] && pok="false"
  done
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$tier" "${MODEL_OF[$tier]}" "$gpus" "$port" "${MML_OF[$tier]}" "${UTIL_OF[$tier]}" \
    "$([[ $kind == 12b ]] && echo "$LAUNCHER_12B" || echo "$LAUNCHER_31B")" \
    "$offset" "$t2h" "$obs_uuids" "$pok" "$digest" >>"$TSV"
done

# ---- emit JSON (python for correct escaping) -----------------------------------------------
MODE="$MODE" GIT_SHA="$GIT_SHA" TS="$(ts_utc)" IMAGE="$IMAGE" \
FREE_START="$FREE_START_MB" FREE_STEADY="$FREE_STEADY_MB" SWAP="$SWAP_USED_MB" \
DOCKER_STATS="$DOCKER_STATS_JSON" PROBE_INTERVAL="$PROBE_INTERVAL" PROBE_TIMEOUT="$PROBE_TIMEOUT" \
HOST_LABEL="$HOST_LABEL" \
python3 - "$TSV" "$JSON_OUT" <<'PY'
import json, os, sys
tsv, out = sys.argv[1], sys.argv[2]
services = []
with open(tsv) as f:
    for line in f:
        tier, model, gpus, port, mml, util, launcher, offset, t2h, uuids, pok, digest = line.rstrip("\n").split("\t")
        def num(x):
            try: return float(x)
            except: return None
        services.append({
            "tier": tier, "model": model, "gpus_requested": gpus, "port": int(port),
            "max_model_len": int(mml), "gpu_memory_utilization": float(util),
            "launcher": launcher, "launch_offset_s": num(offset), "time_to_healthy_s": num(t2h),
            "placement_observed": uuids, "placement_ok": pok == "true", "image_digest": digest,
        })
stats_raw = os.environ.get("DOCKER_STATS", "").strip()
docker_stats = json.loads("[" + stats_raw + "]") if stats_raw else []
host_label = os.environ.get("HOST_LABEL", "").strip()  # explicit, opt-in; never the real nodename
doc = {
    "experiment": "boot_choreography",
    "mode": os.environ["MODE"],
    "timestamp_utc": os.environ["TS"],
    "git_sha": os.environ["GIT_SHA"],
    "image_requested": os.environ["IMAGE"],
    **({"host": host_label} if host_label else {}),
    "probe": {"endpoint": "/v1/chat/completions",
              "interval_s": float(os.environ["PROBE_INTERVAL"]),
              "timeout_s": float(os.environ["PROBE_TIMEOUT"])},
    "host_mem_mb": {"available_at_start": int(os.environ["FREE_START"]),
                    "available_at_steady": int(os.environ["FREE_STEADY"]),
                    "swap_used_at_steady": int(os.environ["SWAP"])},
    "services": services,
    "steady_state_containers": docker_stats,
}
with open(out, "w") as f:
    json.dump(doc, f, indent=2)
print(f"\nWrote {out}")
PY
rm -f "$TSV"

# ---- summary table -------------------------------------------------------------------------
log "=== boot choreography summary ($MODE) ============================================"
printf '%-14s %-38s %-8s %8s %8s %-7s\n' tier model gpus offset t2h place
python3 - "$JSON_OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for s in d["services"]:
    print(f'{s["tier"]:<14} {s["model"]:<38} {s["gpus_requested"]:<8} '
          f'{(s["launch_offset_s"] or 0):8.1f} {(s["time_to_healthy_s"] or 0):8.1f} '
          f'{"OK" if s["placement_ok"] else "BAD":<7}')
print(f'\nhost available RAM: start={d["host_mem_mb"]["available_at_start"]}MB  '
      f'steady={d["host_mem_mb"]["available_at_steady"]}MB  '
      f'swap_used={d["host_mem_mb"]["swap_used_at_steady"]}MB')
for c in d["steady_state_containers"]:
    print(f'  {c["name"]:<24} mem={c["mem"]:<20} ({c["perc"]})')
PY
log "results JSON: $JSON_OUT"
log "host capture: $RESULTS_DIR/${STEM}_iostat.log , ${STEM}_free.log"
log "NEXT: verify placement_ok=OK on all tiers; if BAD, check CUDA_VISIBLE_DEVICES steering before trusting timings."
