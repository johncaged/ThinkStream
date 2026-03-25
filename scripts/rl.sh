#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=0
NPROC_PER_NODE=8
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NNODES=$NNODES"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=/your/cold/start/ckpt  # Using HuggingFace model ID
# Training hyperparameters
lr=2e-7
batch_size=1
grad_accum_steps=1

# Training entry point
entry_file=thinkstream/train.py

# Dataset configuration (replace with public dataset names)
datasets=stream_rlvr

# Output configuration
run_name="thinkstream-rl"
output_dir=./output/${run_name}

# Training arguments
args="
    grpo \
    --args.train.num_train_epochs 1 \
    --args.train.output_dir ${output_dir} \
    --args.train.deepspeed ${deepspeed} \
    --args.train.per_device_train_batch_size ${batch_size} \
    --args.train.warmup_ratio 0.03 \
    --args.train.learning_rate ${lr} \
    --args.train.gradient_accumulation_steps ${grad_accum_steps} \
    --args.model.name_or_path ${llm} \
    --args.train.save_steps 300 \
    --args.data.dataset_use ${datasets} \
    --args.model.model_type qwen2.5vl \
    --args.train.group_size 8 \
    --args.train.micro_batch_size 8 \
    --args.train.rollout_max_pixels $((192*28*28)) \
    --args.train.rollout_min_pixels $((128*28*28))"

# Launch training
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         --node-rank=${NODE_RANK} \
         --nnodes=${NNODES} \
         ${entry_file} ${args}
