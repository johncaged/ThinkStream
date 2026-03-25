import os
import json
import logging
from typing import Any
import torch
from transformers import PreTrainedModel, AutoProcessor
from slyme.context import Context, Ref
from slyme.node import Node, node, wrapper, sequential_exec, Auto
from slyme.node import expression
from thinkstream.model import MODEL_CLS

logger = logging.getLogger(__name__)


@node
def build_optimizer_kwargs(
    ctx: Context,
    /,
    *,
    learning_rate: Auto[float],
    adam_beta1: Auto[float],
    adam_beta2: Auto[float],
    adam_epsilon: Auto[float],
    optimizer_kwargs: Ref[dict],
) -> Context:
    return ctx.set(
        optimizer_kwargs,
        {
            "lr": learning_rate,
            "betas": (adam_beta1, adam_beta2),
            "eps": adam_epsilon,
        },
    )


@node
def set_gradient_checkpointing(
    ctx: Context,
    /,
    *,
    model: Auto[PreTrainedModel],
    grad_ckpt_kwargs: Auto[dict],
) -> Context:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=grad_ckpt_kwargs)
    return ctx


@node
def align_special_tokens(
    ctx: Context,
    /,
    *,
    model: Auto[PreTrainedModel],
    tokenizer: Auto[Any],
) -> Context:
    updated_tokens = {}
    updated_tokens["bos_token_id"] = tokenizer.bos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    updated_tokens["pad_token_id"] = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    if len(updated_tokens) > 0:
        logger.warning(
            "The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. "
            "The model config and generation config were aligned accordingly, being updated with the tokenizer's "
            f"values. Updated tokens: {updated_tokens}."
        )
    return ctx


@node
def set_model_train(
    ctx: Context,
    /,
    *,
    model: Auto[PreTrainedModel],
) -> Context:
    model.train()
    return ctx


@node
def model_zero_grad(
    ctx: Context,
    /,
    *,
    model_for_training: Auto[Any],
) -> Context:
    model_for_training.zero_grad()
    return ctx


@wrapper
def check_should_save(
    ctx: Context,
    wrapped: Node,
    call_next,
    /,
    *,
    save_steps: Auto[int],
    state_global_step: Auto[int],
) -> Context:
    if state_global_step > 0 and state_global_step % save_steps == 0:
        return call_next(ctx)
    return ctx


@node
def update_deepspeed_config_by_hidden_size(
    ctx: Context,
    /,
    *,
    deepspeed_config: Auto[dict],
    hidden_size: Auto[int],
) -> Context:
    if not deepspeed_config or "zero_optimization" not in deepspeed_config:
        return ctx
    zero_config = deepspeed_config["zero_optimization"]
    reduce_bucket_size = hidden_size * hidden_size
    stage3_prefetch_bucket_size = int(0.9 * hidden_size * hidden_size)
    stage3_param_persistence_threshold = 10 * hidden_size
    if zero_config.get("reduce_bucket_size") in ["auto", None]:
        zero_config["reduce_bucket_size"] = reduce_bucket_size
    if zero_config.get("stage3_prefetch_bucket_size") in ["auto", None]:
        zero_config["stage3_prefetch_bucket_size"] = stage3_prefetch_bucket_size
    if zero_config.get("stage3_param_persistence_threshold") in ["auto", None]:
        zero_config["stage3_param_persistence_threshold"] = (
            stage3_param_persistence_threshold
        )
    return ctx


@node
def hf_deepspeed_save_model(
    ctx: Context,
    /,
    *,
    model_for_training: Auto[Any],
    model: Auto[PreTrainedModel],
    processor: Auto[Any],
    output_dir: Auto[str],
    process_index: Auto[int],
    state_global_step: Auto[int],
    state_log_history: Auto[
        list
    ],  # Introduced to save log_history into trainer_state.json
) -> Context:
    """
    HF model saving node compatible with ZeRO-3 without relying on the accelerator.
    """
    # Get State Dict: If using ZeRO-3, call deepspeed's consolidate method to gather partitioned parameters.
    if (
        hasattr(model_for_training, "zero_optimization_stage")
        and model_for_training.zero_optimization_stage() == 3
    ):
        state_dict = model_for_training._zero3_consolidated_16bit_state_dict()
    else:
        state_dict = model.state_dict()

    # Only execute write operations on the main process (Rank 0) to avoid conflicts
    if process_index == 0:
        ckpt_dir = os.path.join(output_dir, f"checkpoint-{state_global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Save model weights in HuggingFace format
        model.save_pretrained(
            ckpt_dir,
            state_dict=state_dict,
            safe_serialization=True,
        )

        # Save Processor/Tokenizer if available
        if processor is not None:
            processor.save_pretrained(ckpt_dir)

        # Assemble and save trainer_state.json for training resumption and logging
        trainer_state = {
            "global_step": state_global_step,
            "log_history": state_log_history,
        }
        with open(os.path.join(ckpt_dir, "trainer_state.json"), "w") as f:
            json.dump(trainer_state, f, indent=4)

    return ctx


@expression
def sft_mini_metrics(
    ctx: Context,
    /,
    *,
    step_loss: Auto[torch.Tensor],
) -> dict:
    return {"loss": step_loss}


@expression
def sft_global_metrics(
    ctx: Context,
    /,
    *,
    model_for_training: Auto[Any],
    optimizer: Auto[torch.optim.Optimizer],
) -> dict:
    grad_norm = model_for_training.get_global_grad_norm()
    if hasattr(grad_norm, "item"):
        grad_norm = grad_norm.item()
    lr = optimizer.param_groups[0]["lr"]
    return {
        "grad_norm": grad_norm,
        "learning_rate": lr,
    }


@wrapper
def with_hf_deepspeed_context(
    ctx: Context,
    wrapped: Node,
    call_next,
    /,
    *,
    deepspeed_config: Auto[dict],
) -> Context:
    """For safe ZeRO-3 load"""
    from transformers.integrations import HfDeepSpeedConfig
    from transformers.integrations.deepspeed import (
        set_hf_deepspeed_config,
        unset_hf_deepspeed_config,
    )

    hf_ds_config = HfDeepSpeedConfig(deepspeed_config)
    set_hf_deepspeed_config(hf_ds_config)

    try:
        ctx = call_next(ctx)
    finally:
        unset_hf_deepspeed_config()
        if "HF_DEEPSPEED_CONFIG" in os.environ:
            del os.environ["HF_DEEPSPEED_CONFIG"]

    return ctx


@node
def load_model(
    ctx: Context,
    /,
    *,
    model_name_or_path: Auto[str],
    model_cache_dir: Auto[str],
    bf16: Auto[bool],
    model: Ref[PreTrainedModel],
    model_type: Auto[str],
    deepspeed_config: Auto[dict],
) -> Context:
    attn_implementation = "streaming_attention"
    vision_attn_implementation = "flash_attention_2"
    if model_type not in MODEL_CLS:
        raise ValueError(
            f"Unsupported model_type: {model_type}. Choose from {list(MODEL_CLS.keys())}"
        )

    lmm = MODEL_CLS[model_type].from_pretrained(
        model_name_or_path,
        cache_dir=model_cache_dir,
        attn_implementation=attn_implementation,
        dtype=(torch.bfloat16 if bf16 else None),
    )
    lmm.config.vision_config._attn_implementation = vision_attn_implementation

    return ctx.update({model: lmm})


@node
def configure_model_gradients(
    ctx: Context,
    /,
    *,
    model: Auto[PreTrainedModel],
) -> Context:
    for n, p in model.model.visual.named_parameters():
        p.requires_grad = False
    for n, p in model.model.visual.merger.named_parameters():
        p.requires_grad = True
    for n, p in model.model.language_model.named_parameters():
        p.requires_grad = True
    model.lm_head.requires_grad = True
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    return ctx


@node
def init_processor(
    ctx: Context,
    /,
    *,
    model_name_or_path: Auto[str],
    model_type: Auto[str],
    processor: Ref[Any],
) -> Context:
    lmm_processor = AutoProcessor.from_pretrained(model_name_or_path)
    if model_type == "qwen3vl":
        lmm_processor.tokenizer.add_tokens(["<silent>", "<response>"])
    else:
        lmm_processor.tokenizer.add_tokens(
            ["<silent>", "<response>", "<think>", "</think>"]
        )
    return ctx.set(processor, lmm_processor)


class DataArgs:
    pass


@node
def init_dataset(
    ctx: Context,
    /,
    *,
    processor: Auto[Any],
    model_type: Auto[str],
    train_dataset: Ref[Any],
    data_collator: Ref[Any],
    data_dataset_use: Auto[str],
    data_flatten: Auto[bool],
    data_packing: Auto[bool],
    data_base_interval: Auto[int],
    data_max_pixels: Auto[int],
    data_min_pixels: Auto[int],
    data_video_max_frames: Auto[int],
    data_video_min_frames: Auto[int],
    data_video_max_pixels: Auto[int],
    data_video_min_pixels: Auto[int],
    data_video_fps: Auto[float],
    model_max_length: Auto[int],
    vocab_size: Auto[int],
) -> Context:
    items = dict(
        dataset_use=data_dataset_use,
        data_flatten=data_flatten,
        data_packing=data_packing,
        base_interval=data_base_interval,
        max_pixels=data_max_pixels,
        min_pixels=data_min_pixels,
        video_max_frames=data_video_max_frames,
        video_min_frames=data_video_min_frames,
        video_max_pixels=data_video_max_pixels,
        video_min_pixels=data_video_min_pixels,
        video_fps=data_video_fps,
        model_type=model_type,
        model_max_length=model_max_length,
    )
    data_args = DataArgs()
    for k, v in items.items():
        setattr(data_args, k, v)

    from thinkstream.data.stream_data_processor import make_supervised_data_module

    data_module = make_supervised_data_module(
        processor,
        data_args=data_args,
        vocab_size=vocab_size,
    )
    return ctx.update(
        {
            train_dataset: data_module["train_dataset"],
            data_collator: data_module["data_collator"],
        }
    )


@node
def train_pipeline(
    ctx: Context,
    /,
    *,
    nodes: list[Node],
) -> Context:
    ctx = sequential_exec(ctx, nodes)
    return ctx
