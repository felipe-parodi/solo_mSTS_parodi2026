#!/usr/bin/env bash
# ==========================================================================
# Temporal Split-Half Stability Control for Figure 2
#
# Splits each session chronologically into early/late halves and re-runs
# the full Fig 2 encoding pipeline on each half. Reports per-session,
# per-subject (M2/M1), and pooled stability metrics.
#
# Run from the project root via Git Bash on Windows:
#
#   cd /c/Users/USER\ Parodi/Documents/ext_repos/freemat_os
#   bash scripts/single_fm/run_temporal_split_control.sh
#
# Estimated time: ~1-1.5 hours (GPU + 8 CPUs, 50 perms)
#   5 sessions x 2 halves x ~6-9 min each
# ==========================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PROD_DIR="results/temporal_split_control"
LOGFILE="$PROD_DIR/temporal_split_control.log"
mkdir -p "$PROD_DIR"

SESSIONS="M2_0526 M2_0605 M1_0525 M1_0609 M1_0613"

echo "========================================================================"
echo "  TEMPORAL SPLIT-HALF CONTROL - $(date)"
echo "  Output: $PROD_DIR"
echo "  Log: $LOGFILE"
echo "  Sessions: $SESSIONS"
echo "========================================================================"
echo ""

t0=$(date +%s)

python scripts/single_fm/run_fig2b_temporal_split_control.py \
    --sessions $SESSIONS \
    --n-perm 50 --n-perm-null 0 --n-jobs 8 --use-gpu \
    --output-dir "$PROD_DIR" \
    2>&1 | tee "$LOGFILE"

exit_code=${PIPESTATUS[0]}
t1=$(date +%s)
elapsed=$((t1 - t0))

echo ""
echo "========================================================================"
if [ $exit_code -eq 0 ]; then
    echo "  DONE: completed in ${elapsed}s at $(date)"
else
    echo "  FAIL: exit code $exit_code after ${elapsed}s at $(date)"
fi
echo "========================================================================"
