#!/usr/bin/env bash
# ==========================================================================
# Production batch runner: 3 Fig 2 controls
#
# Runs all 3 supplementary controls against the frozen Fig 2 production
# baseline with strict config parity (time-drift, no floor filter,
# n_perm_null=0).
#
# Controls:
#   1. RSSI + antenna nuisance (all 5 sessions in one call)
#   2. Matched subsampling (tiered: within-Hedy, within-Hooke, cross)
#   3. Chronological split (all 5 sessions in one call)
#
# Run from the project root via Git Bash on Windows:
#
#   cd /c/Users/USER\ Parodi/Documents/ext_repos/freemat_os
#   bash scripts/single_fm/run_controls_production.sh
#
# Estimated time: ~12-18 hours
#   RSSI/antenna: ~5h (all 5 sessions sequential)
#   Matched subsample: ~5h (3 tiered runs)
#   Temporal split: ~5h (all 5 sessions sequential)
#
# If a session crashes mid-run (RSSI or temporal), rerun with only the
# remaining sessions, then manually merge per-session parquets.
# ==========================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate mucal3

# Timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M)
CTRL_DIR="results/single_fm/controls_production_${TIMESTAMP}"
mkdir -p "$CTRL_DIR"

# ── Configuration ────────────────────────────────────────────────────────
BASELINE_DIR="results/single_fm/production_20260209_2301/all4_bin05"
ALL_SESSIONS="hedy_0526 hedy_0605 hooke_0525 hooke_0609 hooke_0613"
HEDY_SESSIONS="hedy_0526 hedy_0605"
HOOKE_SESSIONS="hooke_0525 hooke_0609 hooke_0613"

N_PERM=100
N_PERM_NULL=0
N_REPEATS=50
N_JOBS=4

# Track results
declare -a RUN_NAMES=()
declare -a RUN_STATUS=()
declare -a RUN_TIMES=()

run_step() {
    local name="$1"
    local logfile="$CTRL_DIR/${name}.log"
    shift
    RUN_NAMES+=("$name")

    echo ""
    echo "========================================================================"
    echo "  Starting: $name"
    echo "  Log: $logfile"
    echo "  Time: $(date)"
    echo "========================================================================"

    local t0
    t0=$(date +%s)
    if "$@" > "$logfile" 2>&1; then
        local t1
        t1=$(date +%s)
        local elapsed=$((t1 - t0))
        echo "  DONE: $name succeeded at $(date) (${elapsed}s)"
        RUN_STATUS+=("OK")
        RUN_TIMES+=("${elapsed}s")
    else
        local t1
        t1=$(date +%s)
        local elapsed=$((t1 - t0))
        echo "  FAIL: $name failed at $(date) (${elapsed}s)"
        echo "  -- Last 15 lines of $logfile --"
        tail -15 "$logfile" 2>/dev/null | sed 's/^/  | /'
        echo "  ----------------------------------"
        RUN_STATUS+=("FAIL")
        RUN_TIMES+=("${elapsed}s")
    fi
}

echo "========================================================================"
echo "  CONTROLS PRODUCTION RUN - $(date)"
echo "  Output: $CTRL_DIR"
echo "  Baseline: $BASELINE_DIR"
echo "  Sessions: $ALL_SESSIONS"
echo "  Config: n_perm=$N_PERM, n_perm_null=$N_PERM_NULL, n_repeats=$N_REPEATS"
echo "========================================================================"
echo ""

# ======================================================================
# CONTROL 1: RSSI + Antenna Nuisance (all 5 sessions, sequential)
# ======================================================================

echo ""
echo "  CONTROL 1: RSSI + Antenna Distance Nuisance"
echo ""

run_step "rssi_all" \
    python scripts/single_fm/run_rssi_control.py \
    --sessions $ALL_SESSIONS \
    --n-perm $N_PERM --n-perm-null $N_PERM_NULL \
    --include-antenna-distance \
    --use-gpu \
    --baseline-dir "$BASELINE_DIR" \
    --output-dir "$CTRL_DIR/rssi_antenna"

# ======================================================================
# CONTROL 2: Matched Subsampling (tiered)
# ======================================================================

echo ""
echo "  CONTROL 2: Matched Subsampling (tiered)"
echo ""

# Tier 1a: Within-subject Hedy
run_step "matched_tier1a_hedy" \
    python scripts/single_fm/run_matched_subsample_control.py \
    --sessions $HEDY_SESSIONS \
    --n-repeats $N_REPEATS \
    --random-size-control \
    --use-gpu \
    --n-jobs $N_JOBS \
    --baseline-dir "$BASELINE_DIR" \
    --output-dir "$CTRL_DIR/matched_subsample/tier1a_hedy_within"

# Tier 1b: Within-subject Hooke
run_step "matched_tier1b_hooke" \
    python scripts/single_fm/run_matched_subsample_control.py \
    --sessions $HOOKE_SESSIONS \
    --n-repeats $N_REPEATS \
    --random-size-control \
    --use-gpu \
    --n-jobs $N_JOBS \
    --baseline-dir "$BASELINE_DIR" \
    --output-dir "$CTRL_DIR/matched_subsample/tier1b_hooke_within"

# Tier 2: Cross-subject (all 5)
run_step "matched_tier2_cross" \
    python scripts/single_fm/run_matched_subsample_control.py \
    --sessions $ALL_SESSIONS \
    --n-repeats $N_REPEATS \
    --random-size-control \
    --use-gpu \
    --n-jobs $N_JOBS \
    --baseline-dir "$BASELINE_DIR" \
    --output-dir "$CTRL_DIR/matched_subsample/tier2_cross_subject"

# ======================================================================
# CONTROL 3: Chronological Split-Half (all 5 sessions, sequential)
# ======================================================================

echo ""
echo "  CONTROL 3: Chronological Split-Half Stability"
echo ""

run_step "split_all" \
    python scripts/single_fm/run_fig2b_temporal_split_control.py \
    --sessions $ALL_SESSIONS \
    --n-perm $N_PERM --n-perm-null $N_PERM_NULL \
    --use-gpu \
    --output-dir "$CTRL_DIR/temporal_split"

# ======================================================================
# SUMMARY
# ======================================================================
echo ""
echo "========================================================================"
echo "  CONTROLS PRODUCTION RUN COMPLETE - $(date)"
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
echo "  Results: $CTRL_DIR"
echo "  Logs:    $CTRL_DIR/*.log"
echo ""
echo "  Output structure:"
echo "    $CTRL_DIR/"
echo "    |- rssi_antenna/               # Control 1 (per-session subdirs + summary)"
echo "    |- matched_subsample/"
echo "    |  |- tier1a_hedy_within/       # Control 2a (Hedy within-subject)"
echo "    |  |- tier1b_hooke_within/      # Control 2b (Hooke within-subject)"
echo "    |  |- tier2_cross_subject/      # Control 2c (cross-subject)"
echo "    |- temporal_split/              # Control 3 (pooled + per-session + per-subject)"
echo "    |- *.log                        # Per-step logs"
echo ""
echo "  If a step failed, rerun the failed session(s) only:"
echo "    # Example: rerun RSSI for sessions that crashed"
echo "    python scripts/single_fm/run_rssi_control.py \\"
echo "      --sessions <failed_sessions> --n-perm 100 --n-perm-null 0 \\"
echo "      --include-antenna-distance --use-gpu \\"
echo "      --baseline-dir $BASELINE_DIR --output-dir $CTRL_DIR/rssi_antenna"
echo ""
