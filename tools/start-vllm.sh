#!/usr/bin/env bash
#
# start-vllm.sh — launch a vLLM OpenAI-compatible server for the Week 11 TP-vs-PP
# experiment. Parameterized over model and parallelism strategy so the same script
# serves TP=2, PP=2, and the PP=4 stretch run rather than hardcoding one config.
#
# Zero-arg default: RedHatAI/gemma-4-31B-it-FP8-block, TP=2, on the NVLink pair (GPUs 0,2).
#
# Usage:
#   ./start-vllm.sh                          # FP8 31B, TP=2, GPUs 0,2 (default)
#   ./start-vllm.sh --mode pp --size 2       # PP=2 on the NVLink pair
#   ./start-vllm.sh --mode pp --size 4 --gpus all   # PP=4 stretch, naive placement (stage i -> GPU i)
#   ./start-vllm.sh --mode pp --size 4 --device-order 0,2,1,3   # PP=4, STEERED: NVLink pair adjacent
#   ./start-vllm.sh --model <hf-id> --mode tp --size 2 --max-model-len 65536
#   ./start-vllm.sh --mode tp --size 2 --profiler-cudagraphs off   # recover CUDA-graph KV tax
#   ./start-vllm.sh ... -- --enforce-eager   # anything after `--` is passed to vLLM verbatim
#   ./start-vllm.sh --help                   # print this header and exit 0
#
# Deterministic stage->GPU placement (--device-order):
#   Docker's `--gpus` device-list ORDER does NOT reliably control in-container CUDA
#   enumeration; with identical GPUs, CUDA's default FASTEST_FIRST resolves ties by
#   bus order, so vLLM lands stage i on physical GPU i regardless of the list. To steer
#   placement deterministically we expose ALL GPUs to the container (`--gpus all`) and
#   then select+order them INSIDE the container with CUDA_VISIBLE_DEVICES, pinning the
#   index basis with CUDA_DEVICE_ORDER=PCI_BUS_ID so the order matches `nvidia-smi`.
#   vLLM assigns PP rank i -> cuda:i, so `--device-order 0,2,1,3` puts physical GPUs
#   0 and 2 (the NVLink pair) on adjacent stages PP0 & PP1 -> the PP0->PP1 boundary
#   becomes the NVLink hop. Confirm where stages actually landed via nvidia-smi
#   uuid-join; never trust the intent line.
#
# CUDA-graph KV tax (--profiler-cudagraphs):
#   Since vLLM v0.21.0, CUDA-graph memory profiling reserves capture memory BEFORE the
#   KV pool, so a nominal --gpu-memory-utilization is effectively lower for KV purposes
#   (boot log reports the equivalent). This is the default ("on") and is the held-constant
#   condition for the Week 11 Days 1-5 baseline; leave it at the default to reproduce
#   those runs. Setting "off" injects VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0, which
#   disables the estimate and returns the reserved memory to the KV pool (recovers the
#   tax). This CHANGES held-constant: only use "off" as a deliberate, named variable in a
#   tax-recovery boot, never as a silent default. One variable per boot.
#
# Ampere notes (RTX 3090, SM 8.6):
#   - FP8 KV cache requires SM 8.9+, so --kv-cache-dtype auto resolves to BF16. Do not
#     force fp8 KV here.
#   - Marlin FP8 weight emulation works fine on SM 8.6.
#
# Deployment note: this is a TEXT-ONLY deployment. --limit-mm-per-prompt zeroes image,
# audio, AND video so vLLM does not reserve an encoder cache budget for modalities we
# never use; that budget is reclaimed into the KV pool. Held constant across all
# TP-vs-PP runs.
#
set -euo pipefail

# ---- Defaults (override via flags) ------------------------------------------------
MODEL="RedHatAI/gemma-4-31B-it-FP8-block"
MODE="tp"                 # tp | pp
SIZE="2"                  # parallel degree
GPUS="0,2"                # comma list of device ids, or the literal "all"
DEVICE_ORDER=""           # optional in-container CUDA_VISIBLE_DEVICES order for deterministic PP stage placement
MAX_MODEL_LEN="131072"    # PROVISIONAL — final value set after Day 2 KV characterization
GPU_MEM_UTIL="0.90"
PROFILER_CUDAGRAPHS="on"  # on (default, baseline) | off (recover CUDA-graph KV tax)
PORT="8000"
IMAGE="vllm/vllm-openai:v0.21.0"
NAME=""                   # container name; default derived below from mode/size
SHM_SIZE="16G"

# ---- Help -------------------------------------------------------------------------
# Render the comment-block header as usage text. Matches vllm-bringup-checks.sh's
# extraction so both scripts present help identically.
usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; }

# ---- Parse flags ------------------------------------------------------------------
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)              MODEL="$2"; shift 2 ;;
    --mode)               MODE="$2"; shift 2 ;;
    --size)               SIZE="$2"; shift 2 ;;
    --gpus)               GPUS="$2"; shift 2 ;;
    --device-order)       DEVICE_ORDER="$2"; shift 2 ;;
    --max-model-len)      MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-mem-util)       GPU_MEM_UTIL="$2"; shift 2 ;;
    --profiler-cudagraphs) PROFILER_CUDAGRAPHS="$2"; shift 2 ;;
    --port)               PORT="$2"; shift 2 ;;
    --image)              IMAGE="$2"; shift 2 ;;
    --name)               NAME="$2"; shift 2 ;;
    -h|--help)            usage; exit 0 ;;
    --)                   shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "[error] unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
done

# ---- Resolve parallelism flag -----------------------------------------------------
case "$MODE" in
  tp) PARALLEL_FLAG=(--tensor-parallel-size "$SIZE") ;;
  pp) PARALLEL_FLAG=(--pipeline-parallel-size "$SIZE") ;;
  *)  echo "[error] --mode must be 'tp' or 'pp' (got '$MODE')" >&2; exit 2 ;;
esac

# ---- Resolve CUDA-graph profiler (KV tax) -----------------------------------------
# "on"  -> default vLLM behavior (estimate enabled); the Days 1-5 baseline.
# "off" -> inject VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 to recover the tax.
PROFILER_ENV_ARGS=()
case "$PROFILER_CUDAGRAPHS" in
  on)  : ;;  # leave vLLM default; no env injected
  off) PROFILER_ENV_ARGS=(-e "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0") ;;
  *)   echo "[error] --profiler-cudagraphs must be 'on' or 'off' (got '$PROFILER_CUDAGRAPHS')" >&2; exit 2 ;;
esac

# ---- Resolve GPU selector + deterministic placement -------------------------------
# Docker wants the literal quoted form '"device=0,2"' for an explicit id list, or the
# bare token 'all' to expose every GPU.
#
# When --device-order is set, we force Docker to expose ALL GPUs and do the actual
# selection+ordering inside the container via CUDA_VISIBLE_DEVICES (+ PCI_BUS_ID), so
# stage placement is deterministic instead of relying on Docker list order.
CUDA_ENV_ARGS=()
if [[ -n "$DEVICE_ORDER" ]]; then
  if [[ "$GPUS" != "all" && "$GPUS" != "0,2" ]]; then
    echo "[warn] --device-order overrides --gpus '${GPUS}'; exposing all GPUs and selecting via CUDA_VISIBLE_DEVICES=${DEVICE_ORDER}." >&2
  fi
  GPU_ARG="all"
  CUDA_ENV_ARGS=(-e "CUDA_DEVICE_ORDER=PCI_BUS_ID" -e "CUDA_VISIBLE_DEVICES=${DEVICE_ORDER}")
elif [[ "$GPUS" == "all" ]]; then
  GPU_ARG="all"
else
  GPU_ARG="\"device=${GPUS}\""
fi

# ---- Container name (descriptive, lets you `docker stop` it) ----------------------
[[ -z "$NAME" ]] && NAME="vllm-${MODE}${SIZE}"

# ---- HF token check (Gemma weights are gated on Hugging Face) ---------------------
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[warn] HF_TOKEN is not set. Gemma weights are gated; the pull may 401." >&2
fi

# ---- Echo resolved config (identity capture) --------------------------------------
echo "=== start-vllm.sh ==="
echo "  model         : ${MODEL}"
echo "  parallelism   : ${MODE}=${SIZE}"
echo "  gpus          : ${GPUS}"
if [[ -n "$DEVICE_ORDER" ]]; then
  echo "  device-order  : ${DEVICE_ORDER}  (in-container CUDA_VISIBLE_DEVICES, PCI_BUS_ID; --gpus forced to all)"
else
  echo "  device-order  : (none — naive: stage i -> physical GPU i)"
fi
echo "  max-model-len : ${MAX_MODEL_LEN}  (provisional until Day 2 characterization)"
echo "  gpu-mem-util  : ${GPU_MEM_UTIL}"
if [[ "$PROFILER_CUDAGRAPHS" == "off" ]]; then
  echo "  cudagraph prof: OFF  (VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 — KV tax recovered; NON-baseline)"
else
  echo "  cudagraph prof: on   (vLLM default; Days 1-5 baseline)"
fi
echo "  image         : ${IMAGE}"
echo "  container     : ${NAME}"
echo "  port          : ${PORT}"
echo "  modalities    : text-only (image/audio/video limited to 0)"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "  extra vllm    : ${EXTRA_ARGS[*]}"
echo "====================="

# ---- Launch -----------------------------------------------------------------------
# Foreground (--rm): Ctrl-C stops and removes the container.
TTY_FLAGS=(-i)
[ -t 1 ] && TTY_FLAGS=(-it)      # interactive terminal -> -it (Week 11 behavior, unchanged)
                                 # non-interactive (orchestrator/nohup/CI) -> -i, no TTY required
exec docker run "${TTY_FLAGS[@]}" --rm \
  --name "${NAME}" \
  --gpus "${GPU_ARG}" \
  --ipc=host --shm-size "${SHM_SIZE}" --network host \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  "${CUDA_ENV_ARGS[@]}" \
  "${PROFILER_ENV_ARGS[@]}" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  "${IMAGE}" \
  --model "${MODEL}" \
  "${PARALLEL_FLAG[@]}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --kv-cache-dtype auto \
  --limit-mm-per-prompt '{"image":0,"audio":0,"video":0}' \
  --host 0.0.0.0 --port "${PORT}" \
  "${EXTRA_ARGS[@]}"
