# rtx3090-ai-training-tools (T)

Evaluation and benchmarking toolchain for the RTX 3090 AI-infrastructure training program.
This is the **tools** repo. The **data** repo — results, journals, captures — is
[`rtx3090-ai-training`](https://github.com/mathiassamuelson/rtx3090-ai-training) (R).

The split: **T holds the code and the bundled eval inputs; R holds the outputs.** Tools live
here, you run them from R, and results land in R. The two were one repo until the Week-14 split;
they are kept separate so a relative `--results-dir` can never write into the tool repo.

## What's here

```
tools/
  rca_quality_probe.py     capture harness — runs RCA probes against a served model, emits a
                           judged-comparison-ready capture JSON
  rca_quality_judge.py     LLM-as-judge — pairwise (both-orders position-bias control) + pointwise
                           scoring via the Anthropic API; schema-validated, retry-on-violation
  worker_contract_check.py deterministic strict-JSON contract conformance checker (no LLM)
  throughput_sweep.py      prefill/decode throughput sweep for OpenAI-compatible endpoints
                           (vLLM, llama.cpp-server); single-request or concurrent waves
  interference_probe.py    cross-tier interference measurement — floods aggressor tiers and
                           measures victim-tier latency degradation through SHARED HOST paths
                           (the per-tier GPUs are disjoint and cannot contend); one direction per
                           run (--victim 31b|12b)
  start-vllm.sh            single-model vLLM launcher with role presets (see below)
  start-stack.sh           full-stack boot choreographer — brings up both 12B workers + the 31B
                           orchestrator (staggered or simultaneous), probes each to time-to-healthy,
                           verifies placement, writes a self-describing results JSON; has a
                           `teardown` verb
  vllm-bringup-checks.sh   post-launch verification gates (container, GPU placement, log scan,
                           /v1/models, chat smoke)
  run-judge.sh             thin wrapper that loads the Anthropic key from a file into the judge's
                           env without it entering shell history
  provenance.py            shared provenance + input-resolution module (see below)
prompts/                   system prompts (orchestrator + worker tiers)
probes/                    RCA probe sets (orchestrator + per-component worker probes)
rubrics/                   judge rubrics
```

The bundled `prompts/`, `probes/`, and `rubrics/` are eval **inputs** and ship here with the
tools — they are not in the data repo.

Two launchers, different scopes: `start-vllm.sh` brings up **one** model/role; `start-stack.sh`
choreographs the **whole three-service stack** at once. They are siblings (start-stack does its own
bring-up, not a loop over start-vllm).

## Setup

T's dependency surface is tiny: the tools are HTTP clients (the serving stack lives in the Docker
image, not the host venv), so the only third-party dependency is `httpx`. Everything else is the
Python standard library; the Anthropic call is hand-rolled over `httpx`, with no SDK.

```bash
python3 -m venv ~/ai-inference        # the materialized venv is gitignored; never committed
. ~/ai-inference/bin/activate
pip install -r requirements.txt        # httpx
```

The judge additionally needs an Anthropic API key — see "Running the judge" below.

## Provenance model — record T's SHA, write into R

The defining constraint of the split: a tool runs from R but is versioned in T, so its provenance
must be **T's git SHA, not the working directory's**. `tools/provenance.py` anchors to its own
`__file__` to resolve the tool repo's checkout regardless of CWD:

- `tool_provenance()` returns T's `{git_sha, git_dirty}`. Every result JSON records this, so a
  capture committed into R is traceable to the exact tool revision that produced it — even though
  the CWD at runtime was R.
- `resolve_input(path)` resolves a bundled eval input (a prompt/probe/rubric) **CWD-first, then
  tool-repo-relative**. Callers pass a short relative path (e.g.
  `prompts/operator-copilot-rca-system-prompt.md`) and it resolves against T without spelling out
  the full checkout path.

Output paths, by contrast, resolve against the **CWD** like any ordinary CLI tool — so a relative
`--results-dir results/...` writes into R, never into T. Provenance (T's SHA) and output location
(R's tree) are deliberately decoupled.

## Run convention

Run tools **from the data repo (R)** so results land in R:

```bash
cd ~/work/rtx3090-ai-training                                   # CWD = R
T=~/work/rtx3090-ai-training-tools                              # T checkout

python3 "$T/tools/throughput_sweep.py" \
    --backend vllm-openai --endpoint http://localhost:8000 \
    --parallelism tp2 \
    --results-dir phase-3-optimization-and-quantization/week-14/results
```

The result filename is self-describing — model name, concurrency, and parallelism tag are folded
in, so runs against different models/configs never silently overwrite. Provenance in the JSON is
T's SHA (via `tool_provenance()`); the file lands under R's `results/` (relative to CWD).

## Running the judge

`run-judge.sh` wraps `rca_quality_judge.py` and loads the Anthropic key from a file into the
judge's process env, so the key never appears on a command line or in `~/.bash_history`:

```bash
# one-time key setup (no history exposure):
mkdir -p ~/.config && ( umask 077; cat > ~/.config/anthropic.key )   # paste key, Enter, Ctrl-D

cd ~/work/rtx3090-ai-training
"$T/tools/run-judge.sh" --mode pairwise --a A.json --b B.json --judge-model <id> \
    --reference-prompt "$T/prompts/operator-copilot-rca-system-prompt.md" \
    --results-dir phase-3-optimization-and-quantization/week-14/results
```

`run-judge.sh --help` prints the wrapper's usage without needing the key or the venv.
`--dry-run` spends no tokens and needs no key. Override the key location with
`ANTHROPIC_KEY_FILE=/path`.

## start-vllm.sh role presets

The launcher carries presets for the production roles (a positional first arg; explicit flags
still override). No role and no `--model` prints usage and exits — there is no silent zero-arg
launch.

```
role           model                               mode  gpus  port   MML       util
orchestrator   google/gemma-4-31B-it-qat-w4a16-ct   TP=2  0,2   8000   131072    0.95
worker1        google/gemma-4-12B-it-qat-w4a16-ct   TP=1   1    8001   131072    0.90
worker2        google/gemma-4-12B-it-qat-w4a16-ct   TP=1   3    8002   131072    0.90
```

```bash
"$T/tools/start-vllm.sh" orchestrator        # 31B-QAT TP=2 on the NVLink pair
"$T/tools/start-vllm.sh" worker1             # 12B-QAT TP=1 on GPU 1
"$T/tools/start-vllm.sh" --help              # full flag list incl. PP / device-order
```

Default image is `vllm/vllm-openai:v0.23.0`, which loads all three production models with no
per-model workarounds. MML 131072 = the models' `max_position_embeddings` validation boundary;
the orchestrator KV ceiling is ~218K at util 0.95 (131K serves at 1.48× concurrency). The
launcher records intent only — **always verify GPU placement empirically** via
`vllm-bringup-checks.sh` (UUID→PID join), never trust the intent echo.
