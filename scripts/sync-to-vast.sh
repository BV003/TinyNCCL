#!/usr/bin/env bash
#=============================================================================
# sync-to-vast.sh
#
# Pushes local (Mac) TinyNCCL code to the Vast.ai cloud instance.
#
# Strategy: Mac is the source of truth for code. The cloud box only pulls/
# runs it (never commits). This script copies the whole project directory
# over, optionally wiping the old remote copy first so deletions/new structure
# are reflected.
#
# Usage:
#   ./sync-to-vast.sh                 # full clean overwrite (default)
#   ./sync-to-vast.sh --light         # only copy changed dirs, no remote wipe
#   ./sync-to-vast.sh --build         # after sync, run build.py on the remote
#   ./sync-to-vast.sh --verify        # after sync, run topology.py on the remote
#
# Config via env vars (defaults match docs/sync-to-vast.md):
#   PORT, IP, LOCAL, REMOTE, KEY
#=============================================================================

set -euo pipefail

# ---- Config (override via env vars) ---------------------------------------
PORT="${PORT:-40022}"
IP="${IP:-167.179.138.57}"
LOCAL="${LOCAL:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"  # project root
REMOTE="${REMOTE:-/root/tiny-nccl}"
KEY="${KEY:-$HOME/.ssh/vastai_id}"

SSH_OPTS=(-p "$PORT" -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=20)
REMOTE_URI="root@$IP"

# ---- Flags -----------------------------------------------------------------
LIGHT=0
DO_BUILD=0
DO_VERIFY=0
for arg in "$@"; do
  case "$arg" in
    --light)  LIGHT=1 ;;
    --build)  DO_BUILD=1 ;;
    --verify) DO_VERIFY=1 ;;
    -h|--help)
      sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^#\{0,1\} //'
      exit 0 ;;
    *) echo "[warn] unknown argument: $arg" >&2 ;;
  esac
done

# ---- Preflight --------------------------------------------------------------
[ -f "$KEY" ] || { echo "[err] SSH key not found: $KEY" >&2; exit 1; }
[ -d "$LOCAL" ] || { echo "[err] local project not found: $LOCAL" >&2; exit 1; }

echo "Target: $REMOTE_URI:$REMOTE  (port $PORT)"

# ---- Sync ------------------------------------------------------------------
if [ "$LIGHT" = "1" ]; then
  # Lightweight: only push code dirs, do NOT wipe remote (keeps build artifacts).
  echo "[1/2] scp (light) — pushing src/ scripts/ docs/ system-design/ ..."
  scp "${SSH_OPTS[@]/#-p/-P}" -o StrictHostKeyChecking=no -r \
      "$LOCAL/src" \
      "$LOCAL/scripts" \
      "$LOCAL/docs" \
      "$LOCAL/system-design" \
      "$REMOTE_URI:$REMOTE/"
  echo "[2/2] light sync done. (remote deletions NOT applied; use full sync for that)"
else
  # Full: wipe remote, then copy entire project (incl. .git).
  echo "[1/3] wiping remote $REMOTE_URI:$REMOTE ..."
  ssh "${SSH_OPTS[@]}" "$REMOTE_URI" "rm -rf '$REMOTE'"

  echo "[2/3] scp full project $LOCAL -> $REMOTE_URI:$REMOTE ..."
  scp "${SSH_OPTS[@]/#-p/-P}" -o StrictHostKeyChecking=no -r \
      "$LOCAL" "$REMOTE_URI:$REMOTE"

  echo "[3/3] full sync done."
fi

# ---- Optional remote run -----------------------------------------------------
if [ "$DO_BUILD" = "1" ]; then
  echo "[build] compiling CUDA extension on remote ..."
  ssh "${SSH_OPTS[@]}" "$REMOTE_URI" "cd '$REMOTE' && python scripts/build.py"
fi
if [ "$DO_VERIFY" = "1" ]; then
  echo "[verify] running topology check on remote ..."
  ssh "${SSH_OPTS[@]}" "$REMOTE_URI" "cd '$REMOTE' && python scripts/topology.py"
fi

echo "Done. To compile/run on the box:"
echo "    ssh -p $PORT -i $KEY $REMOTE_URI \"cd $REMOTE && python scripts/build.py\""
