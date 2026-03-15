#!/usr/bin/env bash
# =============================================================================
# Overnight Batch: Fig 3 + Fig 2B Production
# =============================================================================
# Runs AFTER arena coordinate frame fix (commit 7cfa394).
# d_wall_horiz was completely broken in all previous runs (r=-0.016 vs correct).
#
# Schedule:
#   Phase 1: Fig 3 Reference Frames (CPU, ~15 min total)
#     - All 4 sessions, 0.5s bins, 5-fold CV, n_jobs=14
#     - 0.25s bins same sessions
#
#   Phase 2: Fig 2B Production (2 GPUs parallel, ~4-8 hours)
#     GPU0: M2 sessions (M2_0526, M2_0605) 0.5s then 0.25s
#     GPU1: All 4 sessions (M1_0525, M2_0526, M2_0605, M1_0609) 0.5s then 0.25s
#     100 shuffle perms, 10-fold CV, save residuals
#
# Usage:
#   bash scripts/single_fm/run_overnight_20260206.sh
#   bash scripts/single_fm/run_overnight_20260206.sh --smoke   # quick test
#   bash scripts/single_fm/run_overnight_20260206.sh --fig3-only
#   bash scripts/single_fm/run_overnight_20260206.sh --fig2b-only
# =============================================================================

set -euo pipefail
export PYTHONUNBUFFERED=1

TIMESTAMP=$(date +%Y%m%d_%H%M)
BASE_OUTPUT="results/production_${TIMESTAMP}"
LOG_DIR="logs/overnight_${TIMESTAMP}"

mkdir -p "${LOG_DIR}"

FIG3_RUNNER="python scripts/single_fm/run_fig3_all_sessions.py"
FIG2B_RUNNER="python scripts/single_fm/run_fig2b_all_sessions.py"

FIG2B_COMMON="--n-perm 100 --n-perm-null 0 --use-gpu --n-jobs 14 --save-residuals --cv-folds 10"
ALL_SESSIONS="M1_0525 M2_0526 M2_0605 M1_0609"
HEDY_SESSIONS="M2_0526 M2_0605"

echo "============================================================"
echo "Overnight Batch: Fig 3 + Fig 2B Production"
echo "Started: $(date)"
echo "Output:  ${BASE_OUTPUT}"
echo "Logs:    ${LOG_DIR}"
echo "============================================================"

# ======================= SMOKE TEST =======================
if [[ "${1:-}" == "--smoke" ]]; then
    echo ""
    echo "[SMOKE] Quick test: Fig 3 + Fig 2B on M2_0526 only"
    echo ""
    SMOKE_OUTPUT="results/smoke_overnight_${TIMESTAMP}"

    # Fig 3 smoke
    echo "[SMOKE] Fig 3: M2_0526, 0.5s bins..."
    ${FIG3_RUNNER} \
        --output-dir "${SMOKE_OUTPUT}/fig3" \
        --sessions M2_0526 \
        --bin-s 0.5 --cv-folds 5 --n-jobs 14 \
        2>&1 | tee "${LOG_DIR}/smoke_fig3.log"

    # Fig 2B smoke
    echo "[SMOKE] Fig 2B: M2_0526, 0.5s bins, 10 perms..."
    CUDA_VISIBLE_DEVICES=0 ${FIG2B_RUNNER} \
        --output-dir "${SMOKE_OUTPUT}/fig2b" \
        --sessions M2_0526 \
        --bin-s 0.5 \
        --n-perm 10 --n-perm-null 0 \
        --use-gpu --n-jobs 14 --save-residuals \
        2>&1 | tee "${LOG_DIR}/smoke_fig2b.log"

    echo ""
    echo "[SMOKE] Complete. Output: ${SMOKE_OUTPUT}"
    exit 0
fi

# ======================= PHASE 1: FIG 3 =======================
run_fig3() {
    echo ""
    echo "============================================================"
    echo "[Phase 1] Fig 3: Reference Frame Encoding"
    echo "============================================================"

    # 0.5s bins - all sessions
    echo "[Fig3] All sessions, 0.5s bins..."
    ${FIG3_RUNNER} \
        --output-dir "${BASE_OUTPUT}/fig3_bin05" \
        --sessions ${ALL_SESSIONS} \
        --bin-s 0.5 --cv-folds 5 --n-jobs 14 \
        2>&1 | tee "${LOG_DIR}/fig3_bin05.log"

    echo "[Fig3] 0.5s complete."

    # 0.25s bins - all sessions
    echo "[Fig3] All sessions, 0.25s bins..."
    ${FIG3_RUNNER} \
        --output-dir "${BASE_OUTPUT}/fig3_bin025" \
        --sessions ${ALL_SESSIONS} \
        --bin-s 0.25 --cv-folds 5 --n-jobs 14 \
        2>&1 | tee "${LOG_DIR}/fig3_bin025.log"

    echo "[Fig3] 0.25s complete."
    echo "[Phase 1] Fig 3 DONE."
}

# ======================= PHASE 2: FIG 2B =======================
run_fig2b_gpu0() {
    echo "[GPU0] Starting M2 sessions..."

    # 0.5s bins
    echo "[GPU0] M2 0.5s bins"
    CUDA_VISIBLE_DEVICES=0 ${FIG2B_RUNNER} \
        --output-dir "${BASE_OUTPUT}/M2_bin05" \
        --sessions ${HEDY_SESSIONS} \
        --bin-s 0.5 \
        ${FIG2B_COMMON} \
        2>&1 | tee "${LOG_DIR}/gpu0_M2_bin05.log"

    # 0.25s bins
    echo "[GPU0] M2 0.25s bins"
    CUDA_VISIBLE_DEVICES=0 ${FIG2B_RUNNER} \
        --output-dir "${BASE_OUTPUT}/hedy_bin025" \
        --sessions ${HEDY_SESSIONS} \
        --bin-s 0.25 \
        ${FIG2B_COMMON} \
        2>&1 | tee "${LOG_DIR}/gpu0_hedy_bin025.log"

    echo "[GPU0] M2 complete."
}

run_fig2b_gpu1() {
    echo "[GPU1] Starting All-4 sessions..."

    # 0.5s bins
    echo "[GPU1] All4 0.5s bins"
    CUDA_VISIBLE_DEVICES=1 ${FIG2B_RUNNER} \
        --output-dir "${BASE_OUTPUT}/all4_bin05" \
        --sessions ${ALL_SESSIONS} \
        --bin-s 0.5 \
        ${FIG2B_COMMON} \
        2>&1 | tee "${LOG_DIR}/gpu1_all4_bin05.log"

    # 0.25s bins
    echo "[GPU1] All4 0.25s bins"
    CUDA_VISIBLE_DEVICES=1 ${FIG2B_RUNNER} \
        --output-dir "${BASE_OUTPUT}/all4_bin025" \
        --sessions ${ALL_SESSIONS} \
        --bin-s 0.25 \
        ${FIG2B_COMMON} \
        2>&1 | tee "${LOG_DIR}/gpu1_all4_bin025.log"

    echo "[GPU1] All-4 complete."
}

run_fig2b() {
    echo ""
    echo "============================================================"
    echo "[Phase 2] Fig 2B: Unique Variance (4-way: VIS|BODY_KIN|HEAD_KIN|3D POS)"
    echo "============================================================"

    run_fig2b_gpu0 &
    GPU0_PID=$!

    run_fig2b_gpu1 &
    GPU1_PID=$!

    echo "GPU0 PID: ${GPU0_PID}"
    echo "GPU1 PID: ${GPU1_PID}"
    echo ""
    echo "Monitor:"
    echo "  tail -f ${LOG_DIR}/gpu0_M2_bin05.log"
    echo "  tail -f ${LOG_DIR}/gpu1_all4_bin05.log"

    wait ${GPU0_PID}
    GPU0_EXIT=$?

    wait ${GPU1_PID}
    GPU1_EXIT=$?

    echo ""
    echo "[Phase 2] Fig 2B DONE."
    echo "GPU0 exit: ${GPU0_EXIT}, GPU1 exit: ${GPU1_EXIT}"
}

# ======================= DISPATCH =======================
case "${1:-all}" in
    --fig3-only)
        run_fig3
        ;;
    --fig2b-only)
        run_fig2b
        ;;
    all|"")
        run_fig3
        run_fig2b
        ;;
    *)
        echo "Usage: $0 [--smoke | --fig3-only | --fig2b-only]"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "OVERNIGHT BATCH COMPLETE"
echo "Finished: $(date)"
echo ""
echo "Outputs:"
echo "  Fig 3 0.5s:    ${BASE_OUTPUT}/fig3_bin05"
echo "  Fig 3 0.25s:   ${BASE_OUTPUT}/fig3_bin025"
echo "  Fig 2B M2:   ${BASE_OUTPUT}/M2_bin05, hedy_bin025"
echo "  Fig 2B All4:   ${BASE_OUTPUT}/all4_bin05, all4_bin025"
echo ""
echo "Logs: ${LOG_DIR}/"
echo "============================================================"
