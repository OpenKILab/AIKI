#!/bin/bash

# Add parameter check
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <gpu_index> <tensor_parallel_size> [base_port]"
    echo "Example: $0 0 4      (to use 4 GPUs starting from GPU 0 with default port 10022)"
    echo "         $0 0 2 8080 (to use 2 GPUs starting from GPU 0 with port 8080)"
    echo "Supported tensor parallel sizes: 1, 2, 4, 8"
    exit 1
fi

GPU_INDEX=$1
TP_SIZE=$2
# Use default port if not specified
BASE_PORT=${3:-10022}

# Kill existing session if it exists
tmux kill-session -t vllm-qwen 2>/dev/null

NUM_INSTANCES=1

VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Get number of available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)

# 解析CUDA_VISIBLE_DEVICES
if [ -z "$VISIBLE_DEVICES" ]; then
    # 如果CUDA_VISIBLE_DEVICES未设置，使用所有可用的GPU
    AVAILABLE_GPUS=$(seq 0 $((NUM_GPUS-1)) | tr '\n' ',' | sed 's/,$//')
else
    # 使用CUDA_VISIBLE_DEVICES中指定的GPU
    AVAILABLE_GPUS=$VISIBLE_DEVICES
fi

# 将GPU列表转换为数组
IFS=',' read -ra GPU_ARRAY <<< "$AVAILABLE_GPUS"
NUM_AVAILABLE_GPUS=${#GPU_ARRAY[@]}

# Validate tensor parallel size
if [[ ! "$TP_SIZE" =~ ^(1|2|4|8)$ ]]; then
    echo "Error: Tensor parallel size must be 1, 2, 4, or 8"
    exit 1
fi

# Validate GPU index and ensure enough consecutive GPUs are available
if [ $GPU_INDEX -ge $((NUM_AVAILABLE_GPUS-TP_SIZE+1)) ]; then
    echo "Error: GPU index $GPU_INDEX requires $TP_SIZE consecutive GPUs. Available GPUs: 0 to $((NUM_AVAILABLE_GPUS-TP_SIZE))"
    exit 1
fi

# Use the selected GPUs for tensor parallelism
SELECTED_GPU=""
for ((i=0; i<$TP_SIZE; i++)); do
    if [ $i -eq 0 ]; then
        SELECTED_GPU="${GPU_ARRAY[$((GPU_INDEX+i))]}"
    else
        SELECTED_GPU="$SELECTED_GPU,${GPU_ARRAY[$((GPU_INDEX+i))]}"
    fi
done

SESSION_NAME="vllm-qwen"

# Create a new tmux session
tmux new-session -d -s $SESSION_NAME

# Activate conda environment
tmux send-keys -t "$SESSION_NAME" "conda activate aiki" C-m

# Create windows and start servers
for ((i=0; i<$NUM_INSTANCES; i++)); do
    PORT=$((BASE_PORT + i))
    
    # Create new window
    if [ $i -eq 0 ]; then
        # First window already exists, just rename it
        tmux rename-window -t $SESSION_NAME:0 "vllm-$PORT"
    else
        # Create new windows with explicit target
        tmux new-window -t $SESSION_NAME: -n "vllm-$PORT"
    fi

    # Send commands to the window
    tmux send-keys -t "$SESSION_NAME:vllm-$PORT" "proxy_off" C-m
    tmux send-keys -t "$SESSION_NAME:vllm-$PORT" "echo 'Using port: $PORT on GPU $SELECTED_GPU'" C-m
    # 禁用心跳监控
    tmux send-keys -t "$SESSION_NAME:vllm-$PORT" "export TORCH_NCCL_ENABLE_MONITORING=0" C-m
    tmux send-keys -t "$SESSION_NAME:vllm-$PORT" "export VLLM_WORKER_MULTIPROC_METHOD=spawn" C-m    
    tmux send-keys -t "$SESSION_NAME:vllm-$PORT" "CUDA_VISIBLE_DEVICES=$SELECTED_GPU HF_ENDPOINT=https://hf-mirror.com python3 -m vllm.entrypoints.openai.api_server \
        --model /fs-computility/ai-shen/shared/hf-hub/models--google--gemma-3-27b-it/snapshots/dfb98f29ff907e391ceed2be3834ca071ea260f1 \
        --trust-remote-code \
        --tensor-parallel-size $TP_SIZE \
        --served-model-name gemma-3-27b-it \
        --dtype auto \
        --gpu-memory-utilization 0.85 \
        --port $PORT \
        --use-v2-block-manager \
        --api-key sk-123456" C-m 
done

# Attach to the tmux session
tmux attach-session -t $SESSION_NAME