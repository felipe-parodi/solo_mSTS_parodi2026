#!/usr/bin/env bash
# =============================================================================
# Fig 2B Production Batch Runs
# =============================================================================
# Parallel-pair execution:
#   GPU0: M2 sessions (M2_0526, M2_0605) at 0.5s then 0.25s
#   GPU1: All 4 sessions (M1_0525, M2_0526, M2_0605, M1_0609) at 0.5s then 0.25s
#
# Each run: 100 shuffle perms, time drift ON, save residuals, n_jobs=14
# No null permutations (n_perm_null=0) — broken test skipped for speed
# =============================================================================

set -euo pipefail

# Ensure unbuffered output for progress monitoring
export PYTHONUNBUFFERED=1

# Timestamp for output dirs and logs
TIMESTAMP=$(date +%Y%m%d_%H%M)
BASE_OUTPUT="results/production_${TIMESTAMP}"
LOG_DIR="logs/fig2b_production_${TIMESTAMP}"

# Common flags
COMMON_FLAGS="--n-perm 100 --n-perm-null 0 --use-gpu --n-jobs 14 --save-residuals --cv-folds 10"
RUNNER="python scripts/single_fm/run_fig2b_all_sessions.py"

# Create log directory
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "Fig 2B Production Batch Runs"
echo "Started: $(date)"
echo "Base output: ${BASE_OUTPUT}"
echo "Logs: ${LOG_DIR}"
echo "Common flags: ${COMMON_FLAGS}"
echo "============================================================"

# --- GPU 0: M2 sessions (M2_0526 + M2_0605) ---
run_gpu0() {
    echo "[GPU0] Starting M2 sessions..."

    # 0.5s bins
    echo "[GPU0] Run 1/2: M2 0.5s bins"
    CUDA_VISIBLE_DEVICES=0 ${RUNNER} \
        --output-dir "${BASE_OUTPUT}/M2_bin05" \
        --sessions M2_0526 M2_0605 \
        --bin-s 0.5 \
        ${COMMON_FLAGS} \
        2>&1 | tee "${LOG_DIR}/gpu0_M2_bin05.log"

    echo "[GPU0] Run 1/2 complete."

    # 0.25s bins
    echo "[GPU0] Run 2/2: M2 0.25s bins"
    CUDA_VISIBLE_DEVICES=0 ${RUNNER} \
        --output-dir "${BASE_OUTPUT}/hedy_bin025" \
        --sessions M2_0526 M2_0605 \
        --bin-s 0.25 \
        ${COMMON_FLAGS} \
        2>&1 | tee "${LOG_DIR}/gpu0_hedy_bin025.log"

    echo "[GPU0] All M2 runs complete."
}

# --- GPU 1: All 4 sessions ---
run_gpu1() {
    echo "[GPU1] Starting All-4 sessions..."

    # 0.5s bins
    echo "[GPU1] Run 1/2: All4 0.5s bins"
    CUDA_VISIBLE_DEVICES=1 ${RUNNER} \
        --output-dir "${BASE_OUTPUT}/all4_bin05" \
        --sessions M1_0525 M2_0526 M2_0605 M1_0609 \
        --bin-s 0.5 \
        ${COMMON_FLAGS} \
        2>&1 | tee "${LOG_DIR}/gpu1_all4_bin05.log"

    echo "[GPU1] Run 1/2 complete."

    # 0.25s bins
    echo "[GPU1] Run 2/2: All4 0.25s bins"
    CUDA_VISIBLE_DEVICES=1 ${RUNNER} \
        --output-dir "${BASE_OUTPUT}/all4_bin025" \
        --sessions M1_0525 M2_0526 M2_0605 M1_0609 \
        --bin-s 0.25 \
        ${COMMON_FLAGS} \
        2>&1 | tee "${LOG_DIR}/gpu1_all4_bin025.log"

    echo "[GPU1] All-4 runs complete."
}

# --- Smoke test mode ---
if [[ "${1:-}" == "--smoke" ]]; then
    echo "============================================================"
    echo "SMOKE TEST MODE: 1 session, 10 perms, 0.5s bins"
    echo "============================================================"
    SMOKE_OUTPUT="results/smoke_production_${TIMESTAMP}"
    mkdir -p "${LOG_DIR}"

    CUDA_VISIBLE_DEVICES=0 ${RUNNER} \
        --output-dir "${SMOKE_OUTPUT}" \
        --sessions M2_0526 \
        --bin-s 0.5 \
        --n-perm 10 --n-perm-null 0 \
        --use-gpu --n-jobs 14 --save-residuals \
        2>&1 | tee "${LOG_DIR}/smoke_test.log"

    echo ""
    echo "Smoke test complete. Check output at: ${SMOKE_OUTPUT}"
    echo "Log: ${LOG_DIR}/smoke_test.log"
    exit 0
fi

# --- Production mode: run GPU0 and GPU1 in parallel ---
echo ""
echo "Launching GPU0 (M2) and GPU1 (All4) in parallel..."
echo ""

run_gpu0 &
GPU0_PID=$!

run_gpu1 &
GPU1_PID=$!

echo "GPU0 PID: ${GPU0_PID}"
echo "GPU1 PID: ${GPU1_PID}"
echo ""
echo "Monitor logs:"
echo "  tail -f ${LOG_DIR}/gpu0_M2_bin05.log"
echo "  tail -f ${LOG_DIR}/gpu1_all4_bin05.log"
echo ""

# Wait for both to finish
wait ${GPU0_PID}
GPU0_EXIT=$?

wait ${GPU1_PID}
GPU1_EXIT=$?

echo ""
echo "============================================================"
echo "PRODUCTION BATCH COMPLETE"
echo "Finished: $(date)"
echo "GPU0 exit code: ${GPU0_EXIT}"
echo "GPU1 exit code: ${GPU1_EXIT}"
echo ""
echo "Outputs:"
echo "  M2 0.5s:  ${BASE_OUTPUT}/M2_bin05"
echo "  M2 0.25s: ${BASE_OUTPUT}/hedy_bin025"
echo "  All4 0.5s:  ${BASE_OUTPUT}/all4_bin05"
echo "  All4 0.25s: ${BASE_OUTPUT}/all4_bin025"
echo ""
echo "Logs: ${LOG_DIR}/"
echo "============================================================"
