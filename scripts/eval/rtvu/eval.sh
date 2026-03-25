#!/bin/bash

# 解析关键字参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)
            CKPT="$2"
            shift 2
            ;;
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --ngpu)
            NGPU="$2"
            shift 2
            ;;
        --min_pixels)
            MIN_PIXELS="$2"
            shift 2
            ;;
        --max_pixels)
            MAX_PIXELS="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --think_budget)
            THINK_BUDGET="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# 设置默认值
NGPU=${NGPU:-8}
MIN_PIXELS=${MIN_PIXELS:-$((100352*2))}
MAX_PIXELS=${MAX_PIXELS:-$((100352*4))}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-30}
THINK_BUDGET=${THINK_BUDGET:-20}

echo "NGPU: ${NGPU}"
echo "PIXELS: ${MIN_PIXELS}-${MAX_PIXELS}"
echo "MAX_NEW_TOKENS: ${MAX_NEW_TOKENS}"
echo "THINK_BUDGET: ${THINK_BUDGET}"

TOKENIZERS_PARALLELISM=false \
torchrun \
--nproc_per_node=${NGPU} thinkstream/eval/rtvu/eval_rtvu.py \
--benchmark_dir /your/benchmark/dir \
--model_path "${CKPT}" \
--model_type "${MODEL_TYPE}" \
--min_pixels "${MIN_PIXELS}" \
--max_pixels "${MAX_PIXELS}" \
--max_new_tokens "${MAX_NEW_TOKENS}" \
--think_budget "${THINK_BUDGET}"
