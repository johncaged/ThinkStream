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
llm=Qwen/Qwen2.5-VL-3B-Instruct

# Training hyperparameters
lr=1e-5
batch_size=8
grad_accum_steps=1

# Training entry point
entry_file=thinkstream/train.py

# Dataset configuration
datasets=stream_cold_start

# Output configuration
run_name="thinkstream-cold-start"
output_dir=./output/${run_name}

# DeepSlyme Training arguments
args="
    sft \
    --args.train.deepspeed ${deepspeed} \
    --args.model.name_or_path ${llm} \
    --args.model.model_type qwen2.5vl \
    --args.model.max_length 32768 \
    --args.data.dataset_use ${datasets} \
    --args.data.flatten False \
    --args.data.max_pixels 150528 \
    --args.data.min_pixels 100352 \
    --args.train.bf16 True \
    --args.train.output_dir ${output_dir} \
    --args.train.num_train_epochs 1.0 \
    --args.train.per_device_train_batch_size ${batch_size} \
    --args.train.gradient_accumulation_steps ${grad_accum_steps} \
    --args.train.save_steps 1000 \
    --args.train.learning_rate ${lr} \
    --args.train.weight_decay 0.0 \
    --args.train.warmup_ratio 0.03 \
    --args.train.max_grad_norm 1.0 \
    --args.train.lr_scheduler_type cosine \
    --args.train.torch_empty_cache_steps 1 \
    --args.train.dataloader.num_workers 4
"

# Launch training
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         --node-rank=${NODE_RANK} \
         --nnodes=${NNODES} \
         ${entry_file} ${args}
