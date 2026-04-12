#!/usr/bin/env bash
# Run on each benchmark node (invoked via SSH). Starts one Docker container that runs amd_utils/server.sh.
# Environment is supplied by off-repo launchers or scripts/run_dsr1_fp8_mi325x_sglang_disagg_ssh.sh.

set -euo pipefail

: "${NODE_RANK:?}"
: "${JOB_ID:?}"
: "${NODE0_ADDR:?}"
: "${IPADDRS:?}"
: "${HOST_MODEL_DIR:?}"
: "${HOST_REPO:?}"
: "${IMAGE:?}"
: "${IBDEVICES:?}"

# Use single-token DOCKER_BIN (default: docker) + USE_SUDO_FOR_DOCKER (default: 1).
# Do not rely on a multi-word "DOCKER=sudo docker" env value — it breaks ssh/env on some setups.
: "${DOCKER_BIN:=docker}"
: "${USE_SUDO_FOR_DOCKER:=1}"
HOST_LOG_ROOT="${HOST_LOG_ROOT:-/tmp/inferencex_disagg_logs_${JOB_ID}}"
RUN_LOG_HOST="/tmp/run_logs_${JOB_ID}"

mkdir -p "${RUN_LOG_HOST}" "${HOST_LOG_ROOT}"

DOCKER_DEVICES=(--device /dev/kfd)
shopt -s nullglob
for _d in /dev/dri/renderD* /dev/dri/card*; do
  DOCKER_DEVICES+=(--device "${_d}")
done
if [[ -d /dev/infiniband ]]; then
  for _d in /dev/infiniband/*; do
    [[ -e "${_d}" ]] && DOCKER_DEVICES+=(--device "${_d}")
  done
fi
shopt -u nullglob

EXTRA_ARR=()
if [[ -n "${EXTRA_DOCKER_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=(${EXTRA_DOCKER_ARGS})
fi

if [[ "${USE_SUDO_FOR_DOCKER}" == "1" ]]; then
  _dcmd=(sudo "${DOCKER_BIN}")
else
  _dcmd=("${DOCKER_BIN}")
fi
exec "${_dcmd[@]}" run --rm --init \
  --stop-timeout 10 \
  "${DOCKER_DEVICES[@]}" \
  "${EXTRA_ARR[@]}" \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network host \
  --ipc host \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  -v "${HOST_MODEL_DIR}:/models:ro" \
  -v "${HOST_REPO}:/workspace" \
  -v "${RUN_LOG_HOST}:/run_logs" \
  -v "${HOST_LOG_ROOT}:/benchmark_logs" \
  --shm-size "${DOCKER_SHM_SIZE:-128g}" \
  -e SLURM_JOB_ID="${JOB_ID}" \
  -e SLURM_JOB_NODELIST="manual" \
  -e NODE_RANK="${NODE_RANK}" \
  -e NODE0_ADDR="${NODE0_ADDR}" \
  -e NNODES="${NNODES:-2}" \
  -e IPADDRS="${IPADDRS}" \
  -e MODEL_DIR=/models \
  -e MODEL_NAME="${MODEL_NAME}" \
  -e SGLANG_WS_PATH=/workspace/benchmarks/multi_node/amd_utils \
  -e "xP=${xP:-1}" \
  -e "yD=${yD:-1}" \
  -e "GPUS_PER_NODE=${GPUS_PER_NODE:-8}" \
  -e "PREFILL_TP_SIZE=${PREFILL_TP_SIZE}" \
  -e "PREFILL_ENABLE_EP=${PREFILL_ENABLE_EP}" \
  -e "PREFILL_ENABLE_DP=${PREFILL_ENABLE_DP}" \
  -e "DECODE_TP_SIZE=${DECODE_TP_SIZE}" \
  -e "DECODE_ENABLE_EP=${DECODE_ENABLE_EP}" \
  -e "DECODE_ENABLE_DP=${DECODE_ENABLE_DP}" \
  -e "DECODE_MTP_SIZE=${DECODE_MTP_SIZE:-0}" \
  -e "BENCH_INPUT_LEN=${BENCH_INPUT_LEN}" \
  -e "BENCH_OUTPUT_LEN=${BENCH_OUTPUT_LEN}" \
  -e "BENCH_RANDOM_RANGE_RATIO=${BENCH_RANDOM_RANGE_RATIO}" \
  -e "BENCH_REQUEST_RATE=${BENCH_REQUEST_RATE:-inf}" \
  -e "BENCH_NUM_PROMPTS_MULTIPLIER=${BENCH_NUM_PROMPTS_MULTIPLIER:-10}" \
  -e "BENCH_MAX_CONCURRENCY=${BENCH_MAX_CONCURRENCY}" \
  -e "DRY_RUN=${DRY_RUN:-0}" \
  -e "IBDEVICES=${IBDEVICES}" \
  -e "MORI_RDMA_TC=${MORI_RDMA_TC:-}" \
  -e "BENCHMARK_LOGS_DIR=/benchmark_logs" \
  -e "PYTHONDONTWRITEBYTECODE=1" \
  -e "HOST_IP=${HOST_IP:-}" \
  -e "BARRIER_LOCAL_IP=${BARRIER_LOCAL_IP:-}" \
  -e "BARRIER_SYNC_PORT=${BARRIER_SYNC_PORT:-}" \
  -e "SGLANG_PD_PORT=${SGLANG_PD_PORT:-}" \
  -e "ROUTER_PORT=${ROUTER_PORT:-}" \
  -e "REBUILD_LIBBNXT_IN_CONTAINER=${REBUILD_LIBBNXT_IN_CONTAINER:-0}" \
  -e "PATH_TO_BNXT_TAR_PACKAGE=${PATH_TO_BNXT_TAR_PACKAGE:-}" \
  -e "INSTALL_MORI_IN_CONTAINER=${INSTALL_MORI_IN_CONTAINER:-0}" \
  -e "INSTALL_MORI_MODE=${INSTALL_MORI_MODE:-git}" \
  -e "MORI_GIT_URL=${MORI_GIT_URL:-}" \
  -e "MORI_GIT_REF=${MORI_GIT_REF:-}" \
  -e "MORI_GIT_CLONE_DIR=${MORI_GIT_CLONE_DIR:-}" \
  -e "MORI_SOURCE_PATH=${MORI_SOURCE_PATH:-}" \
  -e "INSTALL_MORI_PYTHON_BIN=${INSTALL_MORI_PYTHON_BIN:-}" \
  -e "INSTALL_MORI_NO_BUILD_ISOLATION=${INSTALL_MORI_NO_BUILD_ISOLATION:-0}" \
  "${IMAGE}" \
  bash /workspace/scripts/_disagg_container_entry.sh
