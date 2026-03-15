#!/usr/bin/env bash
# Overnight batch runner for Fig 2B comprehensive analysis.
#
# All runs use: --time-drift --use-gpu --n-jobs 4
#
# Run from the project root via Git Bash on Windows:
#
#   cd /c/Users/USER\ Parodi/Documents/ext_repos/freemat_os
#   bash scripts/single_fm/run_fig2b_overnight.sh
#
# Priority order (if you need to kill early, most -> least important):
#   Run 1 (M2 prod) > Run 2 (GAM) > Run 3 (bin025) > Run 4 (bin010)
#   > Run 5 (SUA) > Run 6 (4-session)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

OUT_BASE="results/fig2b_overnight"
mkdir -p "$OUT_BASE"

RUNNER="python scripts/single_fm/run_fig2b_all_sessions.py"
HEDY_SESSIONS="--sessions M2_0526 M2_0605"
ALL4_SESSIONS="--sessions M1_0525 M2_0526 M2_0605 M1_0609"
COMMON="--n-jobs 12 --lag-s 1.0 --use-gpu --time-drift --no-floor-filter --save-residuals"

# Track results
declare -a RUN_NAMES=()
declare -a RUN_STATUS=()

run_analysis() {
    local name="$1"
    local logfile="$OUT_BASE/${name}.log"
    shift
    RUN_NAMES+=("$name")

    echo ""
    echo "========================================================================"
    echo "  Starting: $name"
    echo "  Log: $logfile"
    echo "  Time: $(date)"
    echo "========================================================================"

    if $RUNNER "$@" > "$logfile" 2>&1; then
        echo "  DONE: $name succeeded at $(date)"
        RUN_STATUS+=("OK")
    else
        echo "  FAIL: $name failed at $(date) (see $logfile)"
        RUN_STATUS+=("FAIL")
    fi
}

echo "Fig 2B Overnight Batch - Started at $(date)"
echo "Output directory: $OUT_BASE"
echo "Common flags: $COMMON"
echo ""

# -- Run 1: M2 Production (PRIMARY) - full p-values + FDR ----------------
run_analysis "hedy_prod" \
    --output-dir "$OUT_BASE/hedy_prod" \
    $HEDY_SESSIONS \
    --n-perm 100 --n-perm-null 0 \
    $COMMON --bin-s 0.5

# -- Run 2: M2 GAM (nonlinearity test) -----------------------------------
run_analysis "hedy_gam" \
    --output-dir "$OUT_BASE/hedy_gam" \
    $HEDY_SESSIONS \
    --n-perm 100 --n-perm-null 0 \
    $COMMON --bin-s 0.5 --gam

# -- Run 3: M2 Fine Bins (0.25s) -----------------------------------------
run_analysis "hedy_bin025" \
    --output-dir "$OUT_BASE/hedy_bin025" \
    $HEDY_SESSIONS \
    --n-perm 100 --n-perm-null 0 \
    $COMMON --bin-s 0.25

# -- Run 4: M2 Very Fine Bins (0.1s) -------------------------------------
run_analysis "hedy_bin010" \
    --output-dir "$OUT_BASE/hedy_bin010" \
    $HEDY_SESSIONS \
    --n-perm 100 --n-perm-null 0 \
    $COMMON --bin-s 0.1

# -- Run 5: M2 SUA-only (unit quality) -----------------------------------
run_analysis "hedy_sua" \
    --output-dir "$OUT_BASE/hedy_sua" \
    $HEDY_SESSIONS \
    --n-perm 100 --n-perm-null 0 \
    $COMMON --bin-s 0.5 --sua-only

# -- Run 6: 4-Session Production (full dataset) ----------------------------
run_analysis "all4_prod" \
    --output-dir "$OUT_BASE/all4_prod" \
    $ALL4_SESSIONS \
    --n-perm 100 --n-perm-null 0 \
    $COMMON --bin-s 0.5

# -- Summary ----------------------------------------------------------------
echo ""
echo "========================================================================"
echo "  OVERNIGHT BATCH COMPLETE - $(date)"
echo "========================================================================"
echo ""

n_ok=0
n_fail=0
for i in "${!RUN_NAMES[@]}"; do
    status="${RUN_STATUS[$i]}"
    name="${RUN_NAMES[$i]}"
    if [ "$status" = "OK" ]; then
        echo "  [OK]   $name"
        ((n_ok++))
    else
        echo "  [FAIL] $name"
        ((n_fail++))
    fi
done

echo ""
echo "  $n_ok succeeded, $n_fail failed out of ${#RUN_NAMES[@]} runs"
echo "  Logs in: $OUT_BASE/*.log"
echo "  Results in: $OUT_BASE/*/summary/"
