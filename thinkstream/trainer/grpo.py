import os
import re
import types
import math
import logging
import deepspeed
from pathlib import Path
from typing import List, Any, Dict, Optional, Tuple
import torch
from transformers import PreTrainedModel
from slyme.context import Context, Ref
from slyme.node import Node, node, wrapper, Auto, expression
from deepslyme.utils.accelerator import empty_cache

# Import thinkstream specifics
from thinkstream.model.inference import (
    StreamingWindowInferenceEngine,
    streaming_video_chat,
    think_budget_sample,
)
from thinkstream.data.stream_data_processor import (
    SYSTEM_PROMPT,
    QWEN_TEMPLATE_WO_SYSTEM,
    _make_abs_paths,
    build_video_meta,
    process_messages_to_model_inputs,
    pad_and_cat,
    find_assistant_spans,
    compute_position_ids,
    make_raw_data_module,
)
from thinkstream.model.patch import build_video_block_mask
from thinkstream.model import MODEL_CLS, get_text_config, DEFAULT_VIDEO_FLEX_WINDOW_SIZE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reward helper functions (Unchanged from original grpo.py)
# ---------------------------------------------------------------------------

_CHUNK_FORMAT_RE = re.compile(
    r"^<think>.*?</think>(?:<response>.*|<silent>)<\|im_end\|>$",
    re.DOTALL,
)


def _check_chunk_format(text: str) -> bool:
    """Return *True* if a single chunk's generated text matches the format."""
    return _CHUNK_FORMAT_RE.match(text.strip()) is not None


def _compute_format_reward(chunk_texts: List[str]) -> float:
    """Return format reward in [0, 1]: proportion of chunks that match format."""
    if not chunk_texts:
        return 0.0
    correct_count = sum(1 for t in chunk_texts if _check_chunk_format(t))
    return correct_count / len(chunk_texts)


def _collect_think_lengths(
    chunk_results: List[Dict[str, Any]], gen_idx: int, tokenizer: Any
) -> List[int]:
    """Collect token lengths of <think>...</think> spans for one (chunk_results, gen_idx).
    One length per chunk that contains a think block.
    """
    lengths: List[int] = []
    for cr in chunk_results:
        gen_tokens_list = cr.get("generated_tokens", [])
        if gen_idx >= len(gen_tokens_list):
            continue
        gen_tokens = gen_tokens_list[gen_idx]
        if isinstance(gen_tokens, torch.Tensor):
            gen_tokens = gen_tokens.tolist()
        text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
        m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if m:
            think_part = "<think>" + m.group(1) + "</think>"
            think_ids = tokenizer.encode(think_part, add_special_tokens=False)
            lengths.append(len(think_ids))
    return lengths


def _avg_think_len_for_generation(
    chunk_results: List[Dict[str, Any]], gen_idx: int, tokenizer: Any
) -> float:
    """Average number of think tokens (inside <think>...</think>) for one (sample, gen_idx) across chunks."""
    lengths = _collect_think_lengths(chunk_results, gen_idx, tokenizer)
    return sum(lengths) / len(lengths) if lengths else 0.0


def _compute_think_length_factor(
    avg_think_len: float, target_tokens: int, step_window: int = 5
) -> float:
    """
    Compute a discrete, step-wise reward for the thinking token length.
    The reward increases in discrete steps of `step_window`.
    Any length >= (target_tokens - step_window) receives the maximum reward of 1.0.
    """
    if target_tokens <= 0:
        return 1.0
    step_window = max(1, step_window)
    threshold = max(0, target_tokens - step_window)
    if avg_think_len >= threshold:
        return 1.0
    if threshold == 0:
        return 1.0
    step_idx = int(avg_think_len // step_window)
    total_steps = int(threshold // step_window) + 1
    return float(step_idx) / float(total_steps)


def _extract_literal_answer(text: str) -> Optional[str]:
    text = text.strip()
    if re.fullmatch(r"[A-E]", text):
        return text
    if re.fullmatch(r"\([A-E]\)", text):
        return text[1]
    if re.fullmatch(r"[A-E]\.", text):
        return text[0]
    if text.lower() in {"yes", "no"}:
        return text.lower()
    if re.fullmatch(r"[0-9]", text):
        return text
    return None


_RESPONSE_RE = re.compile(r"<response>(.*?)(?:<\|im_end\|>|$)", re.DOTALL)


def _scan_responses_for_answer(
    chunk_results: List[Dict[str, Any]], gen_idx: int, tokenizer: Any
) -> Tuple[Optional[str], Optional[int], int]:
    first_response_chunk_idx: Optional[int] = None
    first_answer: Optional[str] = None
    response_count = 0
    for cr in chunk_results:
        gen_tokens_list = cr.get("generated_tokens", [])
        if gen_idx >= len(gen_tokens_list):
            continue
        gen_tokens = gen_tokens_list[gen_idx]
        if isinstance(gen_tokens, torch.Tensor):
            gen_tokens = gen_tokens.tolist()
        text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
        for m in _RESPONSE_RE.finditer(text):
            response_count += 1
            if first_answer is None:
                answer = _extract_literal_answer(m.group(1))
                if answer is not None:
                    first_answer = answer
                    if first_response_chunk_idx is None:
                        first_response_chunk_idx = cr["chunk_idx"]
    return first_answer, first_response_chunk_idx, response_count


def _compute_time_reward(
    response_chunk_idx: Optional[int],
    gt_chunk_idx: int,
    window: int,
    slack_window: int = 0,
) -> float:
    if response_chunk_idx is None:
        return 0.0
    diff = abs(response_chunk_idx - gt_chunk_idx)
    if diff <= slack_window:
        return 1.0
    if diff <= slack_window + window:
        return 1.0 - (diff - slack_window) / window
    return 0.0


def _compute_correctness_reward(model_answer: Optional[str], gt_content: str) -> float:
    if not model_answer:
        return 0.0
    gt_answer = _extract_literal_answer(gt_content)
    if gt_answer is None:
        return 0.0
    return 1.0 if model_answer == gt_answer else 0.0


def _compute_num_response_reward(
    num_responses: int, step_window: int = 3, max_responses: int = 10
) -> float:
    """
    Compute a discrete, step-wise reward for the number of responses.
    Exactly 1 response yields 1.0.
    Multiple responses decay in intervals of `step_window`, reaching 0.0
    when exceeding `max_responses`.
    """
    if num_responses == 1:
        return 1.0
    if num_responses <= 0 or num_responses > max_responses:
        return 0.0
    step_window = max(1, step_window)
    step_idx = int((num_responses - 2) // step_window) + 1
    max_steps = int((max_responses - 2) // step_window) + 1
    reward = 1.0 - (float(step_idx) / float(max_steps + 1))
    return max(0.0, reward)


# ---------------------------------------------------------------------------
# New Nodes for GRPO adapted for DeepSlyme
# ---------------------------------------------------------------------------


@node
def load_grpo_models(
    ctx: Context,
    /,
    *,
    model_name_or_path: Auto[str],
    model_cache_dir: Auto[str],
    bf16: Auto[bool],
    reference_model: Ref[PreTrainedModel],
    model: Ref[PreTrainedModel],
    model_type: Auto[str],
    model_for_generation: Ref[Any],
    deepspeed_config: Auto[dict],
) -> Context:
    """
    Load the policy model with DeepSpeed config, and clean CPU models for generation and reference.
    """
    from transformers.integrations.deepspeed import (
        set_hf_deepspeed_config,
        unset_hf_deepspeed_config,
    )
    from transformers.integrations import HfDeepSpeedConfig

    if model_type not in MODEL_CLS:
        raise ValueError(f"Unsupported model_type: {model_type}")
    cls = MODEL_CLS[model_type]

    dtype = torch.bfloat16 if bf16 else None
    attn_implementation = "flash_attention_2"
    vision_attn_implementation = "flash_attention_2"

    # 1. Load Policy Model with DeepSpeed context
    hf_ds_config = HfDeepSpeedConfig(deepspeed_config)
    set_hf_deepspeed_config(hf_ds_config)
    try:
        policy_model = cls.from_pretrained(
            model_name_or_path,
            cache_dir=model_cache_dir,
            attn_implementation="streaming_attention",
            dtype=dtype,
        )
        policy_model.config.vision_config._attn_implementation = (
            vision_attn_implementation
        )
    finally:
        unset_hf_deepspeed_config()
        if "HF_DEEPSPEED_CONFIG" in os.environ:
            del os.environ["HF_DEEPSPEED_CONFIG"]

    # 2. Load Generation Model (clean, CPU)
    logger.info("Loading model_for_generation (clean, CPU)...")
    gen_model = cls.from_pretrained(
        model_name_or_path,
        cache_dir=model_cache_dir,
        attn_implementation=attn_implementation,
        dtype=dtype,
    )
    gen_model.config.vision_config._attn_implementation = vision_attn_implementation
    gen_model.config.text_config._attn_implementation = "flash_attention_2_infer"
    gen_model.eval()
    gen_model.requires_grad_(False)
    gen_model.to("cpu")

    # 3. Load Reference Model (clean, frozen, CPU)
    logger.info("Loading reference_model (clean, frozen, CPU)...")
    ref_model = cls.from_pretrained(
        model_name_or_path,
        cache_dir=model_cache_dir,
        attn_implementation=attn_implementation,
        dtype=dtype,
    )
    ref_model.config.vision_config._attn_implementation = vision_attn_implementation
    ref_model.config.text_config._attn_implementation = "streaming_attention"
    ref_model.eval()
    ref_model.requires_grad_(False)
    ref_model.to("cpu")

    return ctx.update(
        {
            model: policy_model,
            model_for_generation: gen_model,
            reference_model: ref_model,
        }
    )


@wrapper
def unwrap_model_for_generation(
    ctx: Context,
    wrapped: Node,
    call_next,
    /,
    *,
    model_for_training: Auto[Any],
    inference_engine: Ref[Any],
    model_for_generation: Auto[Any],
    device: Auto[torch.device],
    state_global_step: Auto[int],
    rollout_last_sync_step: Ref[int],
    rollout_sync_per_step: Auto[int] = 1,
) -> Context:
    """Sync weights from ZeRO-3 model_for_training to CPU model_for_generation before rollout."""
    # With raw DeepSpeed, unwrapped model is accessed via .module
    unwrapped_model = (
        model_for_training.module
        if hasattr(model_for_training, "module")
        else model_for_training
    )

    is_zero3 = (
        hasattr(model_for_training, "zero_optimization_stage")
        and model_for_training.zero_optimization_stage() == 3
    )
    model_for_generation.to(device)
    rollout_last_sync_step_ = ctx.get(rollout_last_sync_step, None)
    if (
        rollout_last_sync_step_ is None
        or state_global_step - rollout_last_sync_step_ >= rollout_sync_per_step
    ):

        def _sync_params():
            train_params = dict(unwrapped_model.named_parameters())
            train_buffers = dict(unwrapped_model.named_buffers())
            with torch.no_grad():
                for name, gen_p in model_for_generation.named_parameters():
                    if name in train_params:
                        gen_p.data.copy_(train_params[name].data)
                    else:
                        logger.warning(
                            "Parameter %s not found in training model.", name
                        )
                for name, gen_b in model_for_generation.named_buffers():
                    if name in train_buffers:
                        gen_b.data.copy_(train_buffers[name].data)

        # NOTE: sync params
        if is_zero3:
            with deepspeed.zero.GatheredParameters(list(unwrapped_model.parameters())):
                _sync_params()
        else:
            _sync_params()
        ctx = ctx.set(rollout_last_sync_step, state_global_step)

    try:
        ctx = call_next(ctx)
    finally:
        model_for_generation.to("cpu")
        ctx = ctx.set(inference_engine, None)
        empty_cache()
    return ctx


@node
def rollout(
    ctx: Context,
    /,
    *,
    step_inputs: Auto[Dict[str, Any]],
    model_for_generation: Auto[Any],
    processor: Auto[Any],
    tokenizer: Auto[Any],
    group_size: Auto[int],
    rollout_data: Ref[Dict[str, Any]],
    inference_engine: Ref[Any],
    model_type: Auto[str],
    rollout_max_new_tokens: Auto[int],
    rollout_max_think_tokens: Auto[int],
    rollout_temperature: Auto[float],
    rollout_top_k: Auto[int],
    rollout_top_p: Auto[float],
    rollout_fpc: Auto[float],
    rollout_max_chunks: Auto[int],
    rollout_min_pixels: Auto[int],
    rollout_max_pixels: Auto[int],
) -> Context:
    """
    GRPO rollout using streaming video inference.

    For each raw sample in the batch, calls ``streaming_video_chat`` to
    generate completions chunk-by-chunk.  Stores per-sample results
    (generated tokens, chunk metadata, raw sample) in ``rollout_data`` for
    downstream reward computation and loss calculation.

    NOTE: This node should be wrapped with ``unwrap_model_for_generation``
    which handles ZeRO-3 parameter gathering and inference engine cleanup.
    """
    inference_engine_ = ctx.get(inference_engine, None)
    if inference_engine_ is None:
        video_token_id = tokenizer.convert_tokens_to_ids(["<|video_pad|>"])[0]
        video_flex_window_size = getattr(
            model_for_generation.config,
            "video_flex_window_size",
            DEFAULT_VIDEO_FLEX_WINDOW_SIZE,
        )
        model_for_generation.config.text_config._attn_implementation = (
            "flash_attention_2_infer"
        )
        text_cfg = get_text_config(model_for_generation.config)
        inference_engine_ = StreamingWindowInferenceEngine(
            model_for_generation,
            batch_size=group_size,
            max_len=16384,
            num_hidden_layers=text_cfg.num_hidden_layers,
            num_key_value_heads=text_cfg.num_key_value_heads,
            head_dim=text_cfg.hidden_size // text_cfg.num_attention_heads,
            vocab_size=text_cfg.vocab_size,
            pad_token_id=model_for_generation.generation_config.pad_token_id,
            eos_token_ids=model_for_generation.generation_config.eos_token_id,
            video_token_id=video_token_id,
            video_flex_window_size=video_flex_window_size,
        )
        ctx = ctx.set(inference_engine, inference_engine_)

    think_end_token_id = tokenizer.convert_tokens_to_ids("</think>")
    sample_kwargs = {
        "think_end_token_id": think_end_token_id,
        "max_think_tokens": rollout_max_think_tokens,
    }

    all_rollout_results: List[Dict[str, Any]] = []
    model_for_generation.eval()

    for raw_sample in step_inputs:
        data_path = raw_sample.get("data_path", "")
        video_path = raw_sample.get("video_path", "")
        abs_video_path = str(_make_abs_paths(Path(data_path), video_path))

        preloaded_video = raw_sample.pop("_preloaded_video", None)

        user_convs = [
            c for c in raw_sample.get("conversations", []) if c.get("role") == "user"
        ]
        queries = [
            {
                "content": c.get("content", ""),
                "timestamp": float(c.get("timestamp", 0.0)),
            }
            for c in user_convs
        ]

        chunk_results: List[Dict[str, Any]] = []
        for result in streaming_video_chat(
            engine=inference_engine_,
            processor=processor,
            video_path=abs_video_path,
            queries=queries,
            num_generations=group_size,
            system_prompt=SYSTEM_PROMPT,
            chat_template_wo_system=QWEN_TEMPLATE_WO_SYSTEM,
            max_new_tokens=rollout_max_new_tokens,
            top_k=rollout_top_k,
            top_p=rollout_top_p,
            temperature=rollout_temperature,
            frames_per_chunk=rollout_fpc,
            max_chunks=rollout_max_chunks,
            min_pixels=rollout_min_pixels,
            max_pixels=rollout_max_pixels,
            sample=think_budget_sample,
            sample_kwargs=sample_kwargs,
            reset_engine=True,
            model_type=model_type,
            preloaded_video=preloaded_video,
            break_on_answer=False,
        ):
            chunk_results.append(result)

        all_rollout_results.append(
            {
                "raw_sample": raw_sample,
                "chunk_results": chunk_results,
                "_preloaded_video": preloaded_video,
            }
        )

    model_for_generation.train()
    return ctx.set(rollout_data, all_rollout_results)


REWARD_DICT_KEYS = ("format", "time", "correctness", "response_efficiency")
DEFAULT_REWARD_WEIGHTS = {
    "format": 0.2,
    "time": 0.2,
    "correctness": 0.4,
    "response_efficiency": 0.2,
}


@node
def calc_rewards(
    ctx: Context,
    /,
    *,
    rollout_data: Auto[Dict[str, Any]],
    rewards: Ref[torch.Tensor],
    rewards_dict: Ref[Dict[str, torch.Tensor]],
    group_size: Auto[int],
    tokenizer: Auto[Any],
    time_reward_window: Auto[int],
    time_reward_slack: Auto[float],
    rollout_max_think_tokens: Auto[int],
) -> Context:
    """Compute per-generation rewards (format + time + correctness + response efficiency).

    ``rollout_data`` is ``List[Dict]`` of length B (one per sample).  Each
    element contains ``raw_sample`` and ``chunk_results``.  ``chunk_results``
    is a list of dicts (one per temporal chunk) with key
    ``generated_tokens: List[torch.Tensor]`` of length G.

    Response efficiency combines a think-length factor (near
    ``rollout_max_think_tokens``) and a response-count decay.

    Sets ``rewards`` (total, shape [B*G]) and ``rewards_dict``: dict of
    component name -> tensor [B*G], e.g. {"format": ..., "time": ..., "correctness": ...}.
    """
    weights = DEFAULT_REWARD_WEIGHTS
    all_rewards, all_fmt, all_time, all_corr, all_response_eff = [], [], [], [], []

    for sample_data in rollout_data:
        raw_sample = sample_data["raw_sample"]
        chunk_results: List[Dict[str, Any]] = sample_data["chunk_results"]

        conversations = raw_sample["conversations"]
        gt_msg = conversations[1]
        gt_timestamp: float = float(gt_msg["timestamp"])
        gt_content: str = gt_msg.get("content", "")

        if chunk_results:
            time_per_chunk = (
                chunk_results[0]["window_end"] - chunk_results[0]["window_start"]
            )
            video_start = chunk_results[0]["window_start"]
        else:
            time_per_chunk = 1.0
            video_start = 0.0

        if time_per_chunk > 0 and chunk_results:
            gt_chunk_idx = int((gt_timestamp - video_start) / time_per_chunk)
            gt_chunk_idx = max(0, min(gt_chunk_idx, len(chunk_results) - 1))
        else:
            gt_chunk_idx = None

        for g in range(group_size):
            chunk_texts: List[str] = []
            for cr in chunk_results:
                tokens = cr["generated_tokens"][g]
                chunk_texts.append(tokenizer.decode(tokens, skip_special_tokens=False))
            model_answer, response_chunk_idx, num_responses = (
                _scan_responses_for_answer(chunk_results, g, tokenizer)
            )

            fmt_r = _compute_format_reward(chunk_texts)
            avg_think_len = _avg_think_len_for_generation(chunk_results, g, tokenizer)
            think_len_rew = _compute_think_length_factor(
                avg_think_len, rollout_max_think_tokens
            )
            num_response_rew = _compute_num_response_reward(num_responses)
            response_eff_r = think_len_rew * num_response_rew

            if gt_chunk_idx is not None:
                slack_window_chunks = (
                    int(time_reward_slack / time_per_chunk) if time_per_chunk > 0 else 0
                )
                time_r = _compute_time_reward(
                    response_chunk_idx,
                    gt_chunk_idx,
                    time_reward_window,
                    slack_window_chunks,
                )
            else:
                time_r = 0.0

            corr_r = _compute_correctness_reward(model_answer, gt_content)

            total_r = (
                weights["format"] * fmt_r
                + weights["time"] * time_r
                + weights["correctness"] * corr_r
                + weights["response_efficiency"] * response_eff_r
            )

            all_rewards.append(total_r)
            all_fmt.append(fmt_r)
            all_time.append(time_r)
            all_corr.append(corr_r)
            all_response_eff.append(response_eff_r)

    rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32)
    rewards_dict_val = {
        "format": torch.tensor(all_fmt, dtype=torch.float32),
        "time": torch.tensor(all_time, dtype=torch.float32),
        "correctness": torch.tensor(all_corr, dtype=torch.float32),
        "response_efficiency": torch.tensor(all_response_eff, dtype=torch.float32),
    }
    return ctx.update({rewards: rewards_tensor, rewards_dict: rewards_dict_val})


def _build_rollout_messages(
    raw_sample, chunk_results, gen_idx, tokenizer, frames_per_chunk
):
    # (Kept identical to original implementation...)
    data_path = raw_sample.get("data_path", "")
    video_path = raw_sample.get("video_path", "")
    abs_video_path = str(_make_abs_paths(Path(data_path), video_path))

    pending_queries = sorted(
        [
            (float(c.get("timestamp", 0.0)), c.get("content", ""))
            for c in raw_sample.get("conversations", [])
            if c.get("role") == "user"
        ],
        key=lambda x: x[0],
    )
    num_chunks = len(chunk_results)
    if num_chunks == 0:
        raise ValueError("No chunk results – cannot build messages.")
    video_chunk_size = chunk_results[0]["window_end"] - chunk_results[0]["window_start"]
    total_start = chunk_results[0]["window_start"]
    total_end = chunk_results[-1]["window_end"]

    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for cr_idx, cr in enumerate(chunk_results):
        w_start, w_end = cr["window_start"], cr["window_end"]
        is_last = cr_idx == num_chunks - 1
        user_content: List[Dict] = [
            {
                "type": "video",
                "video": abs_video_path,
                "video_start": w_start,
                "video_end": w_end,
            }
        ]

        chunk_queries: List[str] = []
        while pending_queries:
            ts = pending_queries[0][0]
            if ts < w_start:
                pending_queries.pop(0)
                continue
            if is_last or ts < w_end:
                chunk_queries.append(pending_queries.pop(0)[1])
            else:
                break
        if chunk_queries:
            user_content.append(
                {"type": "text", "text": "\n" + "\n".join(chunk_queries)}
            )
        messages.append({"role": "user", "content": user_content})

        gen_text = tokenizer.decode(
            cr["generated_tokens"][gen_idx], skip_special_tokens=False
        )
        for _sp in ("<|im_end|>", "<|endoftext|>"):
            if gen_text.endswith(_sp):
                gen_text = gen_text[: -len(_sp)]
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": gen_text}]}
        )

    video_meta = build_video_meta(
        abs_path=abs_video_path,
        total_start=total_start,
        total_end=total_end,
        num_chunks=num_chunks,
        frames_per_chunk=frames_per_chunk,
    )
    return messages, video_meta, video_chunk_size


@node
def build_grpo_inputs(
    ctx: Context,
    /,
    *,
    step_micro_items: Auto[List],
    step_micro_inputs: Ref[Dict[str, Any]],
    rollout_data: Auto[Dict[str, Any]],
    processor: Auto[Any],
    tokenizer: Auto[Any],
    model_type: Auto[str],
    rollout_fpc: Auto[float],
) -> Context:
    """Convert rollout data + raw sample info into tokenised model inputs.

    ``step_micro_items`` coming in is a list of micro-batch item descriptors
    (``{"sample_idx": int, "gen_idx": int}``).  For each descriptor we:

    1. Reconstruct the full chat messages from the rollout's raw sample and
       generated tokens.
    2. Call ``process_messages_to_model_inputs`` (shared with the SFT pipeline)
       to load video frames and tokenise.
    3. Compute MROPE position IDs.
    4. Collate everything into a single batched dict and write it back to
       ``step_micro_inputs`` so that downstream nodes (``prepare_inputs``,
       ``compute_grpo_loss``) receive the expected tensor format.

    Pixel limits are already baked into the processor via
    ``update_processor_pixels`` (called in ``LazyRawDataset.__init__``).
    """
    micro_items = step_micro_items
    all_items = []
    _preloaded_cache = {}

    for item_desc in micro_items:
        sample_idx, gen_idx = item_desc["sample_idx"], item_desc["gen_idx"]
        sample_data = rollout_data[sample_idx]
        messages, video_meta, video_chunk_size = _build_rollout_messages(
            raw_sample=sample_data["raw_sample"],
            chunk_results=sample_data["chunk_results"],
            gen_idx=gen_idx,
            tokenizer=tokenizer,
            frames_per_chunk=int(rollout_fpc),
        )
        if sample_idx not in _preloaded_cache:
            pv = sample_data.get("_preloaded_video")
            _preloaded_cache[sample_idx] = (
                (pv["split_videos"], pv["video_kwargs"], pv["chunk_metadatas"])
                if pv
                else None
            )

        result = process_messages_to_model_inputs(
            messages=messages,
            video_meta=video_meta,
            video_chunk_size=video_chunk_size,
            processor=processor,
            model_type=model_type,
            add_generation_prompt=False,
            preloaded_frames=_preloaded_cache[sample_idx],
        )
        result["position_ids"] = compute_position_ids(result, processor, model_type)
        all_items.append(result)

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids"].squeeze(0) for item in all_items],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    video_masks = torch.nn.utils.rnn.pad_sequence(
        [item["video_mask"].squeeze(0) for item in all_items],
        batch_first=True,
        padding_value=0,
    )
    position_ids = pad_and_cat([item["position_ids"] for item in all_items])
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    completion_mask = torch.zeros_like(input_ids)
    for b in range(input_ids.size(0)):
        for start, end in find_assistant_spans(input_ids[b].tolist(), tokenizer):
            completion_mask[b, start:end] = 1

    videos = [
        item["pixel_values_videos"]
        for item in all_items
        if "pixel_values_videos" in item
    ]
    video_grid_thws = [
        item["video_grid_thw"] for item in all_items if "video_grid_thw" in item
    ]

    return ctx.set(
        step_micro_inputs,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
            "video_mask": video_masks,
            "position_ids": position_ids,
            "pixel_values_videos": torch.cat(videos, dim=0) if videos else None,
            "video_grid_thw": torch.cat(video_grid_thws, dim=0)
            if video_grid_thws
            else None,
        },
    )


@node
def apply_liger_kernel_for_grpo(
    ctx: Context,
    /,
    *,
    model: Auto[PreTrainedModel],
    reference_model: Auto[PreTrainedModel],
    model_type: Auto[str],
) -> Context:
    from liger_kernel.transformers import _apply_liger_kernel_to_instance
    from thinkstream.model.patch import GRPO_LCE_FORWARD

    if model_type not in GRPO_LCE_FORWARD:
        raise ValueError(f"Unsupported model_type for GRPO: {model_type}")
    grpo_forward_fn = GRPO_LCE_FORWARD[model_type]

    for m in [model, reference_model]:
        _apply_liger_kernel_to_instance(model=m, fused_linear_cross_entropy=False)
        m.forward = types.MethodType(grpo_forward_fn, m)
    return ctx


@node
def compute_grpo_loss(
    ctx: Context,
    /,
    *,
    step_micro_inputs: Auto[Dict[str, Any]],
    step_advantages: Auto[torch.Tensor],
    model_for_training: Auto[Any],
    reference_model: Auto[PreTrainedModel],
    step_loss: Ref[torch.Tensor],
    beta: Auto[float],
    device: Auto[torch.device],
) -> Context:
    """Compute GRPO loss via the patched model forward.

    The model's ``forward`` has been replaced by the model-type-specific
    GRPO LCE forward from :data:`thinkstream.model.patch.GRPO_LCE_FORWARD`
    (applied by :func:`apply_liger_kernel_for_grpo`).  When the extra kwarg
    ``advantages`` is provided, that forward uses
    :class:`LigerFusedLinearGRPOLoss` to fuse the ``lm_head`` projection and
    the GRPO loss in memory-efficient chunks – the full ``[B, L, V]`` logits
    tensor is **never** materialised.

    Because the call goes through ``model_for_training`` (the DeepSpeed /
    DDP engine) directly, all distributed training features (gradient
    synchronisation, mixed-precision, ZeRO, …) work normally.
    """
    ref_input, ref_weight, ref_bias = None, None, None
    if beta != 0.0:
        reference_model.to(device)
        video_block_mask = build_video_block_mask(
            reference_model,
            step_micro_inputs.get("video_mask"),
            step_micro_inputs.get("attention_mask"),
        )
        ref_backbone_kwargs = dict(
            input_ids=step_micro_inputs["input_ids"],
            attention_mask=step_micro_inputs["attention_mask"],
            position_ids=step_micro_inputs.get("position_ids"),
            pixel_values_videos=step_micro_inputs.get("pixel_values_videos"),
            video_grid_thw=step_micro_inputs.get("video_grid_thw"),
            video_block_mask=video_block_mask,
            use_cache=False,
            return_dict=True,
        )
        with torch.no_grad():
            ref_out = reference_model.model(**ref_backbone_kwargs)
            ref_input = ref_out.last_hidden_state[:, :-1, :].contiguous()
            ref_weight = reference_model.lm_head.weight.detach().clone()
            if reference_model.lm_head.bias is not None:
                ref_bias = reference_model.lm_head.bias.detach().clone()
        reference_model.to("cpu")
        torch.cuda.empty_cache()

    model_kwargs = dict(
        input_ids=step_micro_inputs["input_ids"],
        attention_mask=step_micro_inputs["attention_mask"],
        position_ids=step_micro_inputs.get("position_ids"),
        pixel_values_videos=step_micro_inputs.get("pixel_values_videos"),
        video_grid_thw=step_micro_inputs.get("video_grid_thw"),
        video_mask=step_micro_inputs.get("video_mask"),
        use_cache=False,
        advantages=step_advantages,
        ref_input=ref_input,
        ref_weight=ref_weight,
        ref_bias=ref_bias,
        completion_mask=step_micro_inputs.get("completion_mask"),
        grpo_beta=beta,
    )
    outputs = model_for_training(**model_kwargs)
    return ctx.set(step_loss, outputs.loss)


def _avg_think_len_per_chunk_micro(micro_items, rollout_data_, tokenizer):
    all_lengths = []
    for item in micro_items:
        chunk_results = rollout_data_[item["sample_idx"]].get("chunk_results", [])
        all_lengths.extend(
            _collect_think_lengths(chunk_results, item["gen_idx"], tokenizer)
        )
    return sum(all_lengths) / len(all_lengths) if all_lengths else 0.0


@expression
def grpo_micro_metrics(
    ctx: Context,
    /,
    *,
    step_loss: Auto[torch.Tensor],
    step_micro_items: Auto[List],
    rollout_data: Auto[Dict[str, Any]],
    tokenizer: Auto[Any],
) -> dict:
    loss_val = step_loss.detach().float().item()
    avg_think = _avg_think_len_per_chunk_micro(
        step_micro_items, rollout_data, tokenizer
    )

    return {
        "loss": loss_val,
        "avg_think_len": avg_think,
    }


@expression
def grpo_global_metrics(
    ctx: Context,
    /,
    *,
    model_for_training: Auto[Any],
    optimizer: Auto[torch.optim.Optimizer],
    rewards: Auto[torch.Tensor],
    rewards_dict: Auto[Dict[str, torch.Tensor]],
    group_size: Auto[int],
) -> dict:
    grad_norm = model_for_training.get_global_grad_norm()
    if hasattr(grad_norm, "item"):
        grad_norm = grad_norm.item()
    lr = optimizer.param_groups[0]["lr"]

    # Global reward mean
    reward_mean = rewards.float().mean().item()

    # Calculate intra-group variance, then inter-group mean
    # rewards shape is [B * G], reshape to [B, G]
    if rewards.numel() > 1 and group_size > 1:
        grouped_rewards = rewards.float().view(-1, group_size)
        # var(dim=1) computes variance within each prompt group
        reward_var = grouped_rewards.var(dim=1).mean().item()
    else:
        reward_var = 0.0

    # Component-wise average rewards
    component_means = {
        f"reward_{k}_mean": v.float().mean().item() for k, v in rewards_dict.items()
    }

    return {
        "grad_norm": grad_norm,
        "learning_rate": lr,
        "reward_mean": reward_mean,
        "reward_var": reward_var,
        **component_means,
    }


@node
def prepare_grpo_micro_batches(
    ctx: Context,
    /,
    *,
    advantages: Auto[torch.Tensor],
    rewards: Auto[torch.Tensor],
    rewards_dict: Auto[Dict[str, torch.Tensor]],
    micro_batch_size: Auto[int],
    group_size: Auto[int],
    step_advantages: Ref[torch.Tensor],
    step_micro_rewards: Ref[torch.Tensor],
    step_micro_rewards_dict: Ref[Dict[str, torch.Tensor]],
    step_micro_items: Ref[List],
    step_micro_batches: Ref[list[dict[Ref, Any]]],
) -> Context:
    total_samples = advantages.shape[0]
    num_micro_batches = math.ceil(total_samples / micro_batch_size)
    micro_batches = []

    for mb_idx in range(num_micro_batches):
        start_idx = mb_idx * micro_batch_size
        end_idx = min(start_idx + micro_batch_size, total_samples)
        micro_items = [
            {
                "sample_idx": flat_idx // group_size,
                "gen_idx": flat_idx % group_size,
            }
            for flat_idx in range(start_idx, end_idx)
        ]

        mb_updates = {
            step_advantages: advantages[start_idx:end_idx],
            step_micro_rewards: rewards[start_idx:end_idx],
            step_micro_rewards_dict: {
                k: v[start_idx:end_idx] for k, v in rewards_dict.items()
            },
            step_micro_items: micro_items,
        }
        micro_batches.append(mb_updates)

    return ctx.set(step_micro_batches, micro_batches)


@node
def init_grpo_refs(ctx: Context, /, *, inference_engine: Ref[Any]) -> Context:
    return ctx.set(inference_engine, None)


class DataArgs:
    pass


@node
def init_grpo_dataset(
    ctx: Context,
    /,
    *,
    processor: Auto[Any],
    train_dataset: Ref[Any],
    data_collator: Ref[Any],
    data_dataset_use: Auto[str],
    rollout_min_pixels: Auto[int],
    rollout_max_pixels: Auto[int],
    rollout_fpc: Auto[float],
    rollout_max_chunks: Auto[int],
    model_type: Auto[str],
) -> Context:
    """
    Initialises a raw (unprocessed) dataset for the GRPO pipeline.

    Unlike the SFT ``init_dataset`` which pre-tokenises every sample, this
    node simply loads the JSON annotations so that the rollout stage can
    perform streaming inference on the raw data.

    Like SFT's ``init_dataset``, the processor's pixel limits are updated
    here via ``update_processor_pixels`` to match the rollout configuration.

    Video loading config is forwarded to ``LazyRawDataset`` so that
    ``__getitem__`` can pre-load frames via DataLoader ``num_workers``.
    """
    vp = processor.video_processor
    data_args = DataArgs()
    items = dict(
        dataset_use=data_dataset_use,
        min_pixels=rollout_min_pixels,
        max_pixels=rollout_max_pixels,
        video_min_pixels=rollout_min_pixels,
        video_max_pixels=rollout_max_pixels,
        video_min_frames=getattr(vp, "min_frames", 4),
        video_max_frames=getattr(vp, "max_frames", 768),
        video_fps=getattr(vp, "fps", 2.0),
    )
    for k, v in items.items():
        setattr(data_args, k, v)

    data_module = make_raw_data_module(
        processor,
        data_args,
        frames_per_chunk=int(rollout_fpc),
        max_chunks=rollout_max_chunks,
        model_type=model_type,
    )
    return ctx.update(
        {
            train_dataset: data_module["train_dataset"],
            data_collator: data_module["data_collator"],
        }
    )


@wrapper
def timer(
    ctx: Context,
    wrapped: Node,
    call_next,
    /,
    *,
    name: str = "",
):
    import time

    start = time.time()
    ctx = call_next(ctx)
    end = time.time()
    print(f"{name}: {end - start}s")
    return ctx
