#!/usr/bin/env bash
# ==========================================================================
# Production batch runner: Figs 2, 3, 4 + controls + publication figures
#
# Re-runs the full analysis pipeline with raw 3D pose data (no constraints)
# and cross-session canonical arena alignment (Procrustes to M2_0526).
# Each Fig 2 session runs as its own step for OOM isolation — if one session
# crashes, the rest still complete. A final aggregation step combines results.
# Fig 3 runs all sessions in one call (lighter memory footprint).
# Controls (RSSI+antenna, temporal split) run after core figures.
#
# Run from the project root via Git Bash on Windows:
#
#   cd /c/Users/Felipe\ Parodi/Documents/ext_repos/freemat_os
#   bash scripts/single_fm/run_production_all.sh
#
# Estimated time: ~7-8 hours
#   Fig 2 per session (~30-50min each, 5 sessions)
#   Fig 2 aggregation (~30s)
#   Fig 3 (5 sessions): ~60min
#   Fig 4 (5 sessions): ~30min
#   RSSI + antenna control: ~5h
#   Publication figures: ~2min
#
# Priority order (kill early, most -> least important):
#   Fig 2 > Fig 3 > Fig 4 > Controls > Pub figs
# ==========================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M)
PROD_DIR="results/production_${TIMESTAMP}"
mkdir -p "$PROD_DIR"

# ── Sessions to run ──────────────────────────────────────────────────────
# All 5 sessions. M1_0609 pose data corrected (30hz bbox interp fix, 2026-02-14).
SESSIONS="M1_0525 M2_0526 M2_0605 M1_0609 M1_0613"

# Control scripts use a different naming convention
CTRL_SESSIONS="M1_0525 M2_0526 M2_0605 M1_0609 M1_0613"

# Cross-session arena alignment (Procrustes to M2_0526 reference)
CANONICAL_ALIGN="--canonical-alignment configs/canonical_alignment.json"

# Common flags for Fig 2
FIG2_COMMON="--n-jobs 6 --lag-s 1.0 --use-gpu --time-drift --no-floor-filter --save-residuals"

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
        echo "  ── Last 15 lines of $logfile ──"
        tail -15 "$logfile" 2>/dev/null | sed 's/^/  | /'
        echo "  ──────────────────────────────────"
        RUN_STATUS+=("FAIL")
        RUN_TIMES+=("${elapsed}s")
    fi
}

echo "========================================================================"
echo "  PRODUCTION RUN - $(date)"
echo "  Output: $PROD_DIR"
echo "  Pose data: triangulation_40hz_raw (no constraints, canonical alignment)"
echo "  Sessions: $SESSIONS"
echo "========================================================================"
echo ""

# ══════════════════════════════════════════════════════════════════════════
# FIG 2: Encoding (unique variance) — one session per step for OOM safety
# ══════════════════════════════════════════════════════════════════════════

for sess in $SESSIONS; do
    run_step "fig2_${sess}" \
        python scripts/single_fm/run_fig2b_all_sessions.py \
        --output-dir "$PROD_DIR/all4_bin05" \
        --sessions "$sess" \
        --n-perm 100 --n-perm-null 0 \
        $FIG2_COMMON --bin-s 0.5 \
        $CANONICAL_ALIGN
done

# -- Aggregation: re-run with --skip-existing to combine per-session results --
run_step "fig2_aggregate" \
    python scripts/single_fm/run_fig2b_all_sessions.py \
    --output-dir "$PROD_DIR/all4_bin05" \
    --sessions $SESSIONS \
    --skip-existing \
    --n-perm 100 --n-perm-null 0 \
    $FIG2_COMMON --bin-s 0.5 \
    $CANONICAL_ALIGN

# ══════════════════════════════════════════════════════════════════════════
# FIG 3: Reference frame encoding
# ══════════════════════════════════════════════════════════════════════════

# Fig 3 is lighter on memory — run all sessions in one call
run_step "fig3_all" \
    python scripts/single_fm/run_fig3_all_sessions.py \
    --output-dir "$PROD_DIR/fig3_bin05" \
    --sessions $SESSIONS \
    --bin-s 0.5 --use-gpu --no-floor-filter \
    $CANONICAL_ALIGN

# ══════════════════════════════════════════════════════════════════════════
# FIG 4: Motif decoding (12 motifs, top-10 decoded)
# ══════════════════════════════════════════════════════════════════════════

for sess in $SESSIONS; do
    run_step "fig4_${sess}" \
        python scripts/single_fm/run_fig4_all_sessions.py \
        --output-dir "$PROD_DIR/fig4_all4" \
        --sessions "$sess" \
        --n-perm 100 \
        $CANONICAL_ALIGN
done

# ══════════════════════════════════════════════════════════════════════════
# CONTROL: RSSI + Antenna Distance Nuisance (all sessions, sequential)
# ══════════════════════════════════════════════════════════════════════════

run_step "rssi_antenna_control" \
    python scripts/single_fm/run_rssi_control.py \
    --sessions $CTRL_SESSIONS \
    --n-perm 100 --n-perm-null 0 \
    --include-antenna-distance \
    --use-gpu \
    --baseline-dir "$PROD_DIR/all4_bin05" \
    --output-dir "$PROD_DIR/rssi_control"

# ══════════════════════════════════════════════════════════════════════════
# PUBLICATION FIGURES
# ══════════════════════════════════════════════════════════════════════════

# Uses --prod-dir and --fig4-dir CLI overrides
run_step "pub_figures" \
    python scripts/single_fm/generate_publication_figures.py \
    --prod-dir "$PROD_DIR" \
    --fig4-dir "$PROD_DIR/fig4_all4"

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================================================"
echo "  PRODUCTION RUN COMPLETE - $(date)"
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
echo "    ├── all4_bin05/            # Fig 2 (M2 filtered in fig gen)"
echo "    ├── fig3_bin05/            # Fig 3 (filter by subject in figs)"
echo "    ├── fig4_all4/             # Fig 4 (motif decoding)"
echo "    ├── rssi_control/          # RSSI + antenna distance control"
echo "    └── *.log                  # Per-step logs"
echo ""
echo "  Next steps:"
echo "    1. Check logs for any failures"
echo "    2. Visual review of results/publication_figures/"
echo "    3. Re-run pub figs if needed:"
echo "       python scripts/single_fm/generate_publication_figures.py \\"
echo "         --prod-dir $PROD_DIR --fig4-dir $PROD_DIR/fig4_all4"
echo ""
