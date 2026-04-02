#!/bin/bash
# SGLang/MoRI environment setup for multi-node disaggregated serving.
#
# REQUIRED ENVIRONMENT VARIABLES:
#   IBDEVICES - RDMA/InfiniBand device names (e.g., ionic_0,ionic_1,... or mlx5_0,mlx5_1,...)
#               This must be set by the runner script (runners/launch_mi355x-amds.sh)
#
# OPTIONAL ENVIRONMENT VARIABLES:
#   MORI_RDMA_TC - RDMA traffic class (e.g., 96, 104). Set by runner if cluster uses QoS.

set -x
export PYTHONDONTWRITEBYTECODE=1

# Pre-flight: warn if GPU VRAM has residual usage (stale containers hold memory)
_vram=$(amd-smi monitor --gpu 0 2>/dev/null | awk 'NR==2{print $NF}' | cut -d/ -f1 || true)
if [[ -n "$_vram" ]] && (( $(echo "$_vram > 0" | bc -l 2>/dev/null || echo 0) )); then
    echo "[WARN] GPU 0 has ${_vram} GB VRAM in use. Stale containers may cause OOM."
    echo "       Run 'docker rm -f \$(docker ps -aq)' to clean before benchmarking."
fi

# =============================================================================
# Ionic RDMA initialization (only runs on Pensando ionic NIC clusters)
# Handles: ABI fix, IPv4 assignment — required for MoRI RDMA on ionic hardware.
# Clusters using rdma/mlx5 devices skip this entirely.
# =============================================================================
if ls /sys/class/infiniband/ionic_0 &>/dev/null; then
    # --- Step 1: Fix libionic ABI mismatch ---
    # Container libionic rarely matches the host kernel module version.
    # job.slurm bind-mounts /usr/lib/x86_64-linux-gnu to /host_libs:ro
    _abi_warn=$(ibv_devinfo 2>&1 | grep -c "does not support the kernel ABI" || true)
    if [[ "$_abi_warn" -gt 0 ]]; then
        _host_lib=$(ls /host_libs/libionic.so.1.1.* 2>/dev/null | head -1)
        if [[ -n "$_host_lib" ]]; then
            cp "$_host_lib" /usr/lib/x86_64-linux-gnu/libionic.so.1
            ldconfig 2>/dev/null || true
            _devs=$(ibv_devinfo 2>/dev/null | grep -c "hca_id" || true)
            echo "[INFO] Ionic ABI fixed: copied $(basename "$_host_lib") → $_devs devices visible"
        else
            echo "[ERROR] Ionic ABI mismatch but no host libionic found at /host_libs/"
            echo "        Ensure job.slurm has: -v /usr/lib/x86_64-linux-gnu:/host_libs:ro"
        fi
    else
        echo "[INFO] Ionic ABI OK: $(ibv_devinfo 2>/dev/null | grep -c 'hca_id' || echo 0) devices"
    fi

    # --- Step 2: Assign IPv4 to ionic ports (required for RoCE v2 GIDs) ---
    # Without IPv4, ibv_modify_qp fails with EINVAL during QP INIT→RTR.
    _node_id=$(hostname | grep -oE '[0-9]+$' | tail -1 || true)
    _node_id=$((10#${_node_id: -2}))
    [[ "$_node_id" -eq 0 ]] && _node_id=100
    for _i in 0 1 2 3 4 5 6 7; do
        _iface=$(ls /sys/class/infiniband/ionic_$_i/device/net/ 2>/dev/null | head -1)
        [[ -z "$_iface" ]] && continue
        _gid_hex=$(cat /sys/class/infiniband/ionic_$_i/ports/1/gids/1 2>/dev/null | cut -d: -f4)
        [[ -z "$_gid_hex" ]] && continue
        _subnet=$((16#${_gid_hex: -2} + 100))
        [[ $_subnet -gt 254 ]] && _subnet=$((_subnet - 100))
        _existing=$(ip -4 addr show "$_iface" 2>/dev/null | grep "192.168.${_subnet}\." || true)
        if [[ -z "$_existing" ]]; then
            ip addr add "192.168.${_subnet}.${_node_id}/24" dev "$_iface" 2>/dev/null || true
            ip link set "$_iface" up 2>/dev/null || true
        fi
    done
    echo "[INFO] Ionic IPv4 configured (node_id=$_node_id)"
fi

# IBDEVICES configuration
# Prefer IBDEVICES set by runner (runners/launch_mi355x-amds.sh)
# Fall back to hostname detection if not set (for direct script execution)
if [[ -z "${IBDEVICES:-}" ]]; then
    NODENAME=$(hostname -s)
    if [[ $NODENAME == GPU* ]] || [[ $NODENAME == smci355-ccs-aus* ]]; then
        export IBDEVICES=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
    elif [[ $NODENAME == mia1* ]]; then
        export IBDEVICES=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
    else
        echo "ERROR: Unable to detect cluster from hostname $NODENAME and IBDEVICES not set" >&2
        exit 1
    fi
    echo "[INFO] Auto-detected IBDEVICES=$IBDEVICES from hostname $NODENAME"
else
    echo "[INFO] Using IBDEVICES=$IBDEVICES (set by runner or environment)"
fi
export IBDEVICES

# Auto-detect default network interface (portable across clusters)
export GLOO_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}' | head -n 1)
export NCCL_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}' | head -n 1)

set +x

export NCCL_IB_HCA=$IBDEVICES

export SGLANG_USE_AITER=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200

# Disable allocating memory in one pass
export MORI_SHMEM_MODE=ISOLATION
export SGLANG_MORI_FP8_DISP=True

if [[ "$MODEL_NAME" == *mxfp4* ]]; then
export SGLANG_MORI_FP8_DISP=False
fi

export SGLANG_MORI_FP4_DISP=False
export SGLANG_MORI_FP8_COMB=False

# Per-role dispatch token limits (prefill uses higher throughput, decode uses lower)
export MORI_MAX_DISPATCH_TOKENS_PREFILL=16384
if [[ "$MODEL_NAME" == *mxfp4* ]]; then
    export MORI_MAX_DISPATCH_TOKENS_PREFILL=12288
fi
export MORI_MAX_DISPATCH_TOKENS_DECODE=160

# set MTP size=1 when EP16
export SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD=$((MORI_MAX_DISPATCH_TOKENS_DECODE * 2))

export MORI_EP_LAUNCH_CONFIG_MODE=AUTO
export MORI_IO_QP_MAX_SEND_WR=16384
export MORI_IO_QP_MAX_CQE=32768
export MORI_IO_QP_MAX_SGE=4

export MORI_APP_LOG_LEVEL=INFO

# Router logging control:
# 0 (default) keeps noisy per-request access logs out of stdout while still logging to file.
# 1 mirrors router logs to stdout via tee (useful for live debugging).
export SGLANG_ROUTER_STDOUT_LOGS="${SGLANG_ROUTER_STDOUT_LOGS:-0}"

# QoS/DSCP configuration
# Priority order: 1) Set by runner, 2) Detect via nicctl, 3) Detect from hostname
if [[ -n "${MORI_RDMA_TC:-}" ]]; then
    echo "[INFO] Using MORI_RDMA_TC=$MORI_RDMA_TC (set by runner or environment)"
elif command -v nicctl &> /dev/null; then
    ND_PRIO=$(nicctl show qos  2>/dev/null | awk '/PFC no-drop priorities/ {print $NF; exit}')
    ND_DSCP=$(nicctl show qos 2>/dev/null| awk -v p="$ND_PRIO" '
$1 == "DSCP" && $2 == ":" && $NF == p {
    print $3; exit
}')

    if [[ -n "$ND_DSCP" ]] && [[ -n "$ND_PRIO" ]]; then
        TC=$(( 4 * ND_DSCP ))
        export MORI_RDMA_SL=$ND_PRIO
        export MORI_RDMA_TC=$TC
        echo "[INFO] Detected QoS config from nicctl: MORI_RDMA_TC=$MORI_RDMA_TC, MORI_RDMA_SL=$MORI_RDMA_SL"
    else
        echo "[WARN] nicctl available but QoS data unavailable; trying hostname detection."
        # Fall back to hostname-based detection
        NODENAME=$(hostname -s)
        if [[ $NODENAME == GPU* ]] || [[ $NODENAME == smci355-ccs-aus* ]]; then
            export MORI_RDMA_TC=96
            echo "[INFO] Auto-detected MORI_RDMA_TC=$MORI_RDMA_TC from hostname $NODENAME"
        elif [[ $NODENAME == mia1* ]]; then
            export MORI_RDMA_TC=104
            echo "[INFO] Auto-detected MORI_RDMA_TC=$MORI_RDMA_TC from hostname $NODENAME"
        else
            echo "[INFO] Unable to detect MORI_RDMA_TC from hostname. Skipping RDMA QoS configuration."
        fi
    fi
else
    # nicctl not available, try hostname-based detection
    NODENAME=$(hostname -s)
    if [[ $NODENAME == GPU* ]] || [[ $NODENAME == smci355-ccs-aus* ]]; then
        export MORI_RDMA_TC=96
        echo "[INFO] Auto-detected MORI_RDMA_TC=$MORI_RDMA_TC from hostname $NODENAME"
    elif [[ $NODENAME == mia1* ]]; then
        export MORI_RDMA_TC=104
        echo "[INFO] Auto-detected MORI_RDMA_TC=$MORI_RDMA_TC from hostname $NODENAME"
    else
        echo "[INFO] nicctl not found and unable to detect from hostname. Skipping RDMA QoS configuration."
        echo "       This is normal for clusters without QoS or outside Docker containers."
    fi
fi

# FIXME: WA for latest upstream 0305 image
export PYTHONPATH=/sgl-workspace/aiter:${PYTHONPATH}


