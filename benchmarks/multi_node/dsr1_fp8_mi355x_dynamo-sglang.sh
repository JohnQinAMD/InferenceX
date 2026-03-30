#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Benchmark: DeepSeek-R1 FP8 on MI355X via NVIDIA Dynamo + SGLang
#
# This script sets up and benchmarks Dynamo's disaggregated serving pipeline
# with SGLang on AMD MI355X GPUs using MoRI RDMA for KV cache transfer.
#
# Environment variables (set by InferenceX matrix runner):
#   MODEL           - HuggingFace model path
#   TP              - Tensor parallelism per worker
#   EP              - Expert parallelism (MoE models)
#   CONCURRENCY     - Request concurrency level
#   ISL / OSL       - Input / output sequence lengths
#   NUM_PREFILL     - Number of prefill workers (disagg mode)
#   NUM_DECODE      - Number of decode workers (disagg mode)
#   DISAGG          - "true" for disaggregated mode
#   DISAGG_TRANSFER_BACKEND - Transfer backend (default: mori)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

# Defaults
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-0528}"
MODEL_PREFIX="${MODEL_PREFIX:-dsr1}"
TP="${TP:-8}"
CONCURRENCY="${CONCURRENCY:-8}"
ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
DISAGG="${DISAGG:-false}"
FRAMEWORK="dynamo-sglang"
TRANSFER_BACKEND="${DISAGG_TRANSFER_BACKEND:-mori}"

# AMD-specific environment
export SGLANG_AITER_MLA_PERSIST=False
export SGLANG_USE_AITER=1
export RCCL_MSCCL_ENABLE=0
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

echo "============================================"
echo "  Dynamo + SGLang Benchmark on AMD MI355X"
echo "============================================"
echo "Model:      $MODEL"
echo "Framework:  $FRAMEWORK"
echo "TP:         $TP"
echo "Concurrency: $CONCURRENCY"
echo "ISL/OSL:    ${ISL}/${OSL}"
echo "Disagg:     $DISAGG"
if [ "$DISAGG" = "true" ]; then
    echo "Transfer:   $TRANSFER_BACKEND"
    echo "Prefill:    ${NUM_PREFILL:-1} worker(s)"
    echo "Decode:     ${NUM_DECODE:-1} worker(s)"
fi
echo "============================================"

# Clone Dynamo if not present
DYNAMO_DIR="${DYNAMO_DIR:-/tmp/dynamo}"
if [ ! -d "$DYNAMO_DIR" ]; then
    echo "Cloning Dynamo..."
    git clone --depth 1 https://github.com/ai-dynamo/dynamo.git "$DYNAMO_DIR"
fi

# Build Dynamo (if not already built)
if ! python3 -c "import dynamo" 2>/dev/null; then
    echo "Building Dynamo..."
    cd "$DYNAMO_DIR"
    apt-get update -qq && apt-get install -y -qq build-essential pkg-config libclang-dev protobuf-compiler > /dev/null 2>&1
    command -v rustc &>/dev/null || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.93.1 -q 2>&1 | tail -1
    export PATH=/root/.cargo/bin:$PATH LIBCLANG_PATH=/opt/rocm/lib/llvm/lib
    cd lib/bindings/python && maturin develop --release 2>&1 | tail -3
    cd "$DYNAMO_DIR" && pip install -e . 2>&1 | tail -2
fi

# Start infrastructure
echo "Starting etcd + NATS..."
etcd &
ETCD_PID=$!
nats-server -p 4222 -js &
NATS_PID=$!
sleep 3

export ETCD_ENDPOINTS="http://localhost:2379"
export NATS_SERVER="nats://localhost:4222"

cleanup() {
    echo "Cleaning up..."
    kill $ETCD_PID $NATS_PID 2>/dev/null || true
    pkill -f "dynamo.frontend" 2>/dev/null || true
    pkill -f "dynamo.sglang" 2>/dev/null || true
}
trap cleanup EXIT

# Start Dynamo services
echo "Starting Dynamo frontend..."
python3 -m dynamo.frontend --http-port 8000 &
sleep 2

if [ "$DISAGG" = "true" ]; then
    echo "Starting prefill worker..."
    python3 -m dynamo.sglang \
        --model-path "$MODEL" \
        --tp-size "$TP" \
        --attention-backend aiter \
        --kv-cache-dtype fp8_e4m3 \
        --page-size 16 \
        --disaggregation-mode prefill \
        --disaggregation-transfer-backend "$TRANSFER_BACKEND" &

    echo "Starting decode worker..."
    python3 -m dynamo.sglang \
        --model-path "$MODEL" \
        --tp-size "$TP" \
        --attention-backend aiter \
        --kv-cache-dtype fp8_e4m3 \
        --page-size 16 \
        --disaggregation-mode decode \
        --disaggregation-transfer-backend "$TRANSFER_BACKEND" &
else
    echo "Starting aggregated worker..."
    python3 -m dynamo.sglang \
        --model-path "$MODEL" \
        --tp-size "$TP" \
        --attention-backend aiter \
        --kv-cache-dtype fp8_e4m3 \
        --page-size 16 &
fi

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in $(seq 1 120); do
    if curl -s http://localhost:8000/health | grep -q "ok\|healthy\|200" 2>/dev/null; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 5
done

# Run benchmark
echo "Running benchmark (concurrency=$CONCURRENCY, ISL=$ISL, OSL=$OSL)..."
BENCH_SCRIPT="${REPO_ROOT}/utils/bench_serving/bench_serving.py"
if [ ! -f "$BENCH_SCRIPT" ]; then
    BENCH_SCRIPT="$(python3 -c 'import sglang; import os; print(os.path.join(os.path.dirname(sglang.__file__), "..", "benchmark", "bench_serving.py"))' 2>/dev/null || echo "")"
fi

if [ -f "$BENCH_SCRIPT" ]; then
    python3 "$BENCH_SCRIPT" \
        --backend openai \
        --host localhost \
        --port 8000 \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len "$ISL" \
        --random-output-len "$OSL" \
        --num-prompts "$((CONCURRENCY * 10))" \
        --request-rate "$CONCURRENCY" \
        --output-file "/tmp/benchmark_${MODEL_PREFIX}_${FRAMEWORK}_c${CONCURRENCY}.json"
    echo "Benchmark results saved to /tmp/benchmark_${MODEL_PREFIX}_${FRAMEWORK}_c${CONCURRENCY}.json"
else
    echo "WARNING: bench_serving.py not found, using simple curl benchmark"
    START=$(date +%s%N)
    for i in $(seq 1 $((CONCURRENCY * 2))); do
        curl -s http://localhost:8000/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":16}" \
            -o /dev/null &
    done
    wait
    END=$(date +%s%N)
    ELAPSED=$(( (END - START) / 1000000 ))
    echo "Simple benchmark: $((CONCURRENCY * 2)) requests in ${ELAPSED}ms"
fi

echo "Benchmark complete."
