#!/usr/bin/env bash
# ==========================================================================
# Production: Matched subsampling control for Fig 2 POS encoding
#
# Tests whether POS encoding survives when vertical occupancy (com_z)
# distribution is equalized across sessions via block-matched subsampling.
#
# 3 tiers run sequentially:
#   Tier 1a: M2 within-subject (M2_0526 + M2_0605)
#   Tier 1b: M1 within-subject (M1_0609 + M1_0613)
#   Tier 2+3: Cross-subject + random-size control (M2_0526 + M1_0613)
#
# Run from project root via Git Bash on Windows:
#
#   cd /c/Users/USER\ Parodi/Documents/ext_repos/freemat_os
#   bash scripts/single_fm/run_matched_subsample_all.sh
#
# Estimated time: ~3.5 hours
#   Tier 1a M2:   ~1.8h (443+400 units, 50 reps, ~65s/rep)
#   Tier 1b M1:  ~0.8h (~185 units, 50 reps, ~30s/rep)
#   Tier 2+3 Cross: ~1.0h (443+184 units, 50 reps matched + 50 random)
# ==========================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="C:/ProgramData/miniconda3/envs/mucal3/python.exe"
SCRIPT="scripts/single_fm/run_matched_subsample_control.py"

# Timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M)
PROD_DIR="results/matched_subsample_control/production_${TIMESTAMP}"
mkdir -p "$PROD_DIR"

# Common flags: CPU parallel with 8 workers (n_jobs=-1 causes OOM)
COMMON="--n-repeats 50 --n-jobs 8"

# Track results
declare -a RUN_NAMES=()
declare -a RUN_STATUS=()
declare -a RUN_TIMES=()

run_step() {
    local name="$1"
    local logfile="$PROD_DIR/${name}.log"
    shift
    RUN_NAMES+=("$name")

    echo ""
    echo "========================================================================"
    echo "  Starting: $name"
    echo "  Log: $logfile"
    echo "  Time: $(date)"
    echo "========================================================================"

    local t0=$(date +%s)
    if "$@" > "$logfile" 2>&1; then
        local t1=$(date +%s)
        local elapsed=$((t1 - t0))
        echo "  DONE: $name succeeded at $(date) (${elapsed}s)"
        RUN_STATUS+=("OK")
        RUN_TIMES+=("${elapsed}s")
    else
        local t1=$(date +%s)
        local elapsed=$((t1 - t0))
        echo "  FAIL: $name failed at $(date) (${elapsed}s)"
        echo "  -- Last 15 lines of $logfile --"
        tail -15 "$logfile" 2>/dev/null | sed 's/^/  | /'
        echo "  ------------------------------------"
        RUN_STATUS+=("FAIL")
        RUN_TIMES+=("${elapsed}s")
    fi
}

echo "========================================================================"
echo "  MATCHED SUBSAMPLING CONTROL - $(date)"
echo "  Output: $PROD_DIR"
echo "  Repeats: 50, Workers: 8"
echo "========================================================================"
echo ""

# ======================================================================
# TIER 1a: M2 within-subject
# ======================================================================

run_step "tier1a_hedy_within" \
    "$PYTHON" -u "$SCRIPT" \
    --sessions M2_0526 M2_0605 \
    $COMMON \
    --output-dir "$PROD_DIR/hedy_within"

# ======================================================================
# TIER 1b: M1 within-subject
# ======================================================================

run_step "tier1b_hooke_within" \
    "$PYTHON" -u "$SCRIPT" \
    --sessions M1_0609 M1_0613 \
    $COMMON \
    --output-dir "$PROD_DIR/hooke_within"

# ======================================================================
# TIER 2+3: Cross-subject with tuned params + random-size control
# ======================================================================

run_step "tier2_cross_subject" \
    "$PYTHON" -u "$SCRIPT" \
    --sessions M2_0526 M1_0613 \
    --n-hist-bins 10 --block-size-s 10 --cv-folds 5 \
    $COMMON --random-size-control \
    --output-dir "$PROD_DIR/cross_subject"

# ======================================================================
# SUMMARY
# ======================================================================
echo ""
echo "========================================================================"
echo "  MATCHED SUBSAMPLING CONTROL COMPLETE - $(date)"
echo "========================================================================"
echo ""

n_ok=0
n_fail=0
for i in "${!RUN_NAMES[@]}"; do
    status="${RUN_STATUS[$i]}"
    name="${RUN_NAMES[$i]}"
    elapsed="${RUN_TIMES[$i]}"
    if [ "$status" = "OK" ]; then
        echo "  [OK]   $name ($elapsed)"
        ((n_ok++))
    else
        echo "  [FAIL] $name ($elapsed)"
        ((n_fail++))
    fi
done

echo ""
echo "  $n_ok succeeded, $n_fail failed out of ${#RUN_NAMES[@]} steps"
echo "  Results: $PROD_DIR"
echo "  Logs:    $PROD_DIR/*.log"
echo ""
echo "  Output structure:"
echo "    $PROD_DIR/"
echo "    ├── hedy_within/     # Tier 1a: M2 within-subject"
echo "    ├── hooke_within/    # Tier 1b: M1 within-subject"
echo "    ├── cross_subject/   # Tier 2+3: Cross-subject + random"
echo "    └── *.log            # Per-step logs"
echo ""
