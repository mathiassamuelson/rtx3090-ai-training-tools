#!/usr/bin/env bash
#
# run-judge.sh — thin wrapper around rca_quality_judge.py that loads the Anthropic
# API key from a file (default ~/.config/anthropic.key) into the environment, then
# execs the judge with all arguments forwarded verbatim.
#
# Why a wrapper: a command run inside a script does NOT enter interactive shell
# history, so the key never appears in ~/.bash_history. The key is read here, never
# typed on a command line, never echoed, and never written to any file by this
# script. It is exported only into the judge's process (via exec) and dies with it.
#
# Usage:
#   ./tools/run-judge.sh --mode pairwise --a A.json --b B.json --judge-model <id> \
#       --reference-prompt prompts/operator-copilot-rca-system-prompt.md \
#       --results-dir <dir>
#   ./tools/run-judge.sh ... --dry-run        # no key needed; prints prompts only
#   ANTHROPIC_KEY_FILE=/other/path ./tools/run-judge.sh ...   # override key location
#
# Key file setup (creates it owner-only, without the key entering history):
#   mkdir -p ~/.config && ( umask 077; cat > ~/.config/anthropic.key )
#   <paste the key, then press Enter and Ctrl-D>
#   chmod 600 ~/.config/anthropic.key   # if not already
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JUDGE="$SCRIPT_DIR/rca_quality_judge.py"

if [[ ! -f "$JUDGE" ]]; then
  echo "[error] judge not found next to wrapper: $JUDGE" >&2
  exit 2
fi

# --dry-run spends no tokens and needs no key; pass straight through.
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    exec python3 "$JUDGE" "$@"
  fi
done

KEYFILE="${ANTHROPIC_KEY_FILE:-$HOME/.config/anthropic.key}"

if [[ ! -f "$KEYFILE" ]]; then
  echo "[error] key file not found: $KEYFILE" >&2
  echo "        create it (no history exposure):" >&2
  echo "          mkdir -p \"\$(dirname \"$KEYFILE\")\" && ( umask 077; cat > \"$KEYFILE\" )" >&2
  echo "          <paste key, Enter, Ctrl-D>" >&2
  exit 2
fi

# Warn (don't fail) if the key file is readable beyond its owner.
perms="$(stat -c '%a' "$KEYFILE" 2>/dev/null || echo '')"
if [[ -n "$perms" && "$perms" != "600" && "$perms" != "400" ]]; then
  echo "[warn] $KEYFILE is mode $perms; tighten with: chmod 600 \"$KEYFILE\"" >&2
fi

# Command substitution strips the trailing newline; the value is never echoed.
ANTHROPIC_API_KEY="$(<"$KEYFILE")"
export ANTHROPIC_API_KEY

if [[ -z "$ANTHROPIC_API_KEY" ]]; then
  echo "[error] key file is empty: $KEYFILE" >&2
  exit 2
fi

exec python3 "$JUDGE" "$@"
