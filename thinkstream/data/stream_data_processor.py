import json
import math
import random
import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, List, Tuple, Any
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
from qwen_vl_utils import process_vision_info
import transformers
from torchcodec.decoders import VideoDecoder

from . import data_list
from .rope2d import ROPE_INDEX_FN


def _get_video_pixels(processor):
    """Return (min_pixels, max_pixels) from the video processor.

    Qwen2/2.5 VL exposes ``min_pixels`` / ``max_pixels`` directly,
    while Qwen3 VL only provides ``size["shortest_edge"]`` /
    ``size["longest_edge"]``.
    """
    vp = processor.video_processor
    min_px = getattr(vp, "min_pixels", None)
    max_px = getattr(vp, "max_pixels", None)
    if min_px is None or max_px is None:
        size = getattr(vp, "size", {})
        min_px = min_px or size.get("shortest_edge")
        max_px = max_px or size.get("longest_edge")
    if min_px is None or max_px is None:
        raise ValueError(
            "Cannot resolve video pixel limits from processor.video_processor: "
            f"min_pixels={min_px}, max_pixels={max_px}, size={getattr(vp, 'size', 'N/A')}"
        )
    return min_px, max_px


def _resolve_vit_patch_size(processor) -> int:
    """Extract ViT patch size from a HuggingFace processor."""
    return getattr(
        processor.video_processor,
        "patch_size",
        getattr(processor.image_processor, "patch_size", 14),
    )


def load_video_frames(
    video_path: str,
    video_start: float,
    video_end: float,
    total_nframes: int,
    frames_per_chunk: int,
    num_chunks: int,
    *,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    processor=None,
    vit_patch_size: Optional[int] = None,
    model_type: str,
) -> Tuple[List[torch.Tensor], dict, Optional[List[dict]]]:
    """Load video frames via a ghost message and split into per-chunk tensors.

    This is the single entry-point for the "ghost message" video loading
    pattern shared by SFT (``process_messages_to_model_inputs``), GRPO
    (``build_grpo_inputs``), and inference (``streaming_video_chat``).

    Pixel limits are resolved in order of priority:
    1. Explicit *min_pixels* / *max_pixels* arguments (inference path).
    2. Extracted from *processor* via ``_get_video_pixels`` (training path).

    Args:
        video_path:       Absolute filesystem path to the video.
        video_start:      Start time in seconds.
        video_end:        End time in seconds.
        total_nframes:    Total number of frames to extract.
        frames_per_chunk: Frames per temporal chunk.
        num_chunks:       Expected number of chunks.
        min_pixels:       Minimum pixel budget (overrides processor).
        max_pixels:       Maximum pixel budget (overrides processor).
        processor:        HuggingFace processor (required when *min_pixels*
                          or *max_pixels* is ``None``).
        vit_patch_size:   ViT patch size.  When provided, *processor* is not
                          needed for the ``image_patch_size`` kwarg.
        model_type:       ``"qwen2.5vl"`` or ``"qwen3vl"``.

    Returns:
        ``(split_videos, video_kwargs, chunk_metadatas)``

        - *split_videos*:    ``List[Tensor]`` of length *num_chunks*.
        - *video_kwargs*:    Extra kwargs from ``process_vision_info``
                             (e.g. ``fps``).  May be empty.
        - *chunk_metadatas*: Per-chunk Qwen3VL metadata list, or ``None``.
    """
    is_qwen3vl = model_type == "qwen3vl"

    # --- resolve pixel limits ---
    if min_pixels is None or max_pixels is None:
        if processor is None:
            raise ValueError(
                "Either (min_pixels, max_pixels) or processor must be provided."
            )
        _min, _max = _get_video_pixels(processor)
        min_pixels = min_pixels or _min
        max_pixels = max_pixels or _max

    # --- guard: video_start == video_end ---
    if video_end <= video_start:
        video_end = video_start + 1e-3

    # --- ghost message ---
    ghost_message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "video_start": video_start,
                    "video_end": video_end,
                    "nframes": total_nframes,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                }
            ],
        }
    ]

    pvi_kwargs: dict = dict(return_video_kwargs=True)
    if vit_patch_size is not None:
        pvi_kwargs["image_patch_size"] = vit_patch_size
    elif processor is not None:
        pvi_kwargs["image_patch_size"] = _resolve_vit_patch_size(processor)
    if is_qwen3vl:
        pvi_kwargs["return_video_metadata"] = True

    _, video_inputs_list, video_kwargs = process_vision_info(
        ghost_message,
        **pvi_kwargs,
    )

    # --- unpack ---
    if is_qwen3vl:
        big_video_tensor, video_metadata = video_inputs_list[0]
    else:
        big_video_tensor = video_inputs_list[0]
        video_metadata = None

    # --- split ---
    split_videos = list(torch.split(big_video_tensor, frames_per_chunk, dim=0))

    if len(split_videos) > num_chunks:
        split_videos = split_videos[:num_chunks]
    elif len(split_videos) < num_chunks:
        raise RuntimeError(
            f"Video split mismatch: expected {num_chunks}, got {len(split_videos)}"
        )

    # --- per-chunk metadata (Qwen3VL) or replicate fps (Qwen2/2.5VL) ---
    chunk_metadatas: Optional[List[dict]] = None
    if is_qwen3vl and video_metadata is not None:
        all_indices = video_metadata["frames_indices"]
        if isinstance(all_indices, torch.Tensor):
            chunk_idx_splits = list(torch.split(all_indices, frames_per_chunk))
        else:
            chunk_idx_splits = [
                all_indices[i : i + frames_per_chunk]
                for i in range(0, len(all_indices), frames_per_chunk)
            ]
        chunk_idx_splits = chunk_idx_splits[: len(split_videos)]
        chunk_metadatas = [
            {**video_metadata, "frames_indices": ci} for ci in chunk_idx_splits
        ]
    else:
        if "fps" in video_kwargs and isinstance(video_kwargs["fps"], list):
            video_kwargs["fps"] = [video_kwargs["fps"][0]] * len(split_videos)

    return split_videos, video_kwargs, chunk_metadatas


def preload_video(
    video_path: str,
    *,
    video_start: float = 0.0,
    video_end: Optional[float] = None,
    frames_per_chunk: int = 2,
    max_chunks: int = 120,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    processor=None,
    vit_patch_size: Optional[int] = None,
    model_type: str,
) -> Dict[str, Any]:
    """Pre-load video metadata and decoded frames in one shot.

    This combines the ``VideoDecoder`` metadata probe, chunking strategy
    computation, and ``load_video_frames`` call that are otherwise spread
    across ``streaming_video_chat`` and dataset code.  By calling this in
    ``Dataset.__getitem__``, the heavy video I/O can be parallelised via
    DataLoader ``num_workers``.

    Returns a dict suitable for passing as ``preloaded_video`` to
    ``streaming_video_chat`` or extracting ``(split_videos, video_kwargs,
    chunk_metadatas)`` for ``process_messages_to_model_inputs``.
    """
    decoder = VideoDecoder(video_path)
    video_fps = decoder.metadata.average_fps
    total_video_frames = decoder.metadata.num_frames
    if video_end is None:
        video_end = decoder.metadata.duration_seconds
    del decoder

    # When start == end, take a window of 4 frames centered at that time, clamped to video bounds.
    if video_end <= video_start:
        t, max_sec = video_start, (total_video_frames - 1) / video_fps
        half_span = 2.0 / video_fps
        video_start = max(0.0, t - half_span)
        video_end = min(max_sec, t + half_span)

    video_duration = video_end - video_start
    video_chunk_size = max(1.0, float(math.ceil(video_duration / max_chunks)))
    num_iterations = max(1, int(video_duration // video_chunk_size))
    frames_per_chunk = int(frames_per_chunk)

    total_nframes = num_iterations * frames_per_chunk

    # Clamp for low-fps videos (mirrors streaming_video_chat logic).
    _start_frame = math.ceil(max(0.0, video_start) * video_fps)
    _end_frame = min(math.floor(video_end * video_fps), total_video_frames - 1)
    available_frames = max(_end_frame - _start_frame + 1, frames_per_chunk)
    if total_nframes > available_frames:
        num_iterations = max(1, available_frames // frames_per_chunk)
        total_nframes = num_iterations * frames_per_chunk
        video_chunk_size = video_duration / num_iterations

    split_videos, video_kwargs, chunk_metadatas = load_video_frames(
        video_path=video_path,
        video_start=video_start,
        video_end=video_end,
        total_nframes=total_nframes,
        frames_per_chunk=frames_per_chunk,
        num_chunks=num_iterations,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        processor=processor,
        vit_patch_size=vit_patch_size,
        model_type=model_type,
    )

    return {
        "video_path": video_path,
        "video_start": video_start,
        "video_end": video_end,
        "video_fps": video_fps,
        "total_video_frames": total_video_frames,
        "video_duration": video_duration,
        "video_chunk_size": video_chunk_size,
        "num_iterations": num_iterations,
        "frames_per_chunk": frames_per_chunk,
        "total_nframes": total_nframes,
        "split_videos": split_videos,
        "video_kwargs": video_kwargs,
        "chunk_metadatas": chunk_metadatas,
    }


IGNORE_INDEX = -100
FRAMES_PER_CHUNK = 2
DEFAULT_MAX_CHUNKS = 120
DEFAULT_INFERENCE_MIN_PIXELS = 100352 * 2
DEFAULT_INFERENCE_MAX_PIXELS = 100352 * 4


def build_video_meta(
    abs_path: str,
    total_start: float,
    total_end: float,
    num_chunks: int,
    frames_per_chunk: int = FRAMES_PER_CHUNK,
) -> Dict[str, Any]:
    """Build the canonical ``video_meta`` dict consumed by ``load_video_frames``
    and ``process_messages_to_model_inputs``."""
    return {
        "abs_path": abs_path,
        "total_start": total_start,
        "total_end": total_end,
        "num_chunks": num_chunks,
        "frames_per_chunk": frames_per_chunk,
        "total_nframes": num_chunks * frames_per_chunk,
    }


def rank0_print(*args):
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args)
    else:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def _make_abs_paths(base: Path, files: str) -> str:
    return f"{(base / files).resolve()}"


def update_processor_pixels(processor, data_args):
    # --- Image Processor ---
    ip = processor.image_processor
    rank0_print("=== BEFORE IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"ip.size: {ip.size}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    ip.min_pixels = data_args.min_pixels
    ip.max_pixels = data_args.max_pixels
    rank0_print(f"✅ Updated image_processor min_pixels to {data_args.min_pixels}")
    rank0_print(f"✅ Updated image_processor max_pixels to {data_args.max_pixels}")

    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = data_args.min_pixels
        ip.size["longest_edge"] = data_args.max_pixels
        rank0_print(
            f"✅ Updated image_processor size['shortest_edge'] to {data_args.min_pixels}"
        )
        rank0_print(
            f"✅ Updated image_processor size['longest_edge'] to {data_args.max_pixels}"
        )

    rank0_print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    # --- Video Processor ---
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        rank0_print("\n=== BEFORE VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

        vp.min_pixels = data_args.video_min_pixels
        vp.max_pixels = data_args.video_max_pixels
        rank0_print(
            f"✅ Updated video_processor min_pixels to {data_args.video_min_pixels}"
        )
        rank0_print(
            f"✅ Updated video_processor max_pixels to {data_args.video_max_pixels}"
        )

        if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
            vp.min_frames = data_args.video_min_frames
            vp.max_frames = data_args.video_max_frames
            rank0_print(
                f"✅ Updated video_processor min_frames to {data_args.video_min_frames}"
            )
            rank0_print(
                f"✅ Updated video_processor max_frames to {data_args.video_max_frames}"
            )

        if hasattr(vp, "fps"):
            vp.fps = data_args.video_fps
            rank0_print(f"✅ Updated video_processor fps to {data_args.video_fps}")

        if hasattr(vp, "size") and isinstance(vp.size, dict):
            vp.size["shortest_edge"] = data_args.video_min_pixels
            vp.size["longest_edge"] = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
            )
            rank0_print(
                f"✅ Updated Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}"
            )

        rank0_print("=== AFTER VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

    # Vision architecture params that affect visual token count
    ip = processor.image_processor
    vp = getattr(processor, "video_processor", None)
    vit_patch = getattr(vp, "patch_size", getattr(ip, "patch_size", "N/A"))
    vit_merge = getattr(ip, "merge_size", "N/A")
    vit_temporal = getattr(vp, "temporal_patch_size", "N/A") if vp else "N/A"
    factor = (
        (vit_patch * vit_merge)
        if isinstance(vit_patch, int) and isinstance(vit_merge, int)
        else "N/A"
    )
    rank0_print(f"\n=== VISION ARCHITECTURE ===")
    rank0_print(
        f"patch_size={vit_patch}, merge_size={vit_merge}, temporal_patch_size={vit_temporal}, factor={factor}"
    )
    if hasattr(data_args, "model_max_length"):
        rank0_print(f"Before: max_length: ", processor.tokenizer.model_max_length)
        processor.tokenizer.model_max_length = data_args.model_max_length
        rank0_print(f"After: max_length: ", processor.tokenizer.model_max_length)

    return processor


def _get_duration(path: str) -> float:
    return VideoDecoder(path).metadata.duration_seconds


SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "You will see a continuous stream of video chunks. "
    "Based on the user's query and the video content, first output your internal reasoning enclosed in <think>...</think> tags. "
    "Then, if you determine that a response is needed at this moment, output <response> followed by the content. "
    "If no response is needed, output <silent>. "
    "Your generated thoughts and responses should be continuous and fluent across the video chunks."
)

# Chat template without system prompt (for subsequent streaming chunks)
QWEN_TEMPLATE_WO_SYSTEM = (
    "\n{% set image_count = namespace(value=0) %}"
    "{% set video_count = namespace(value=0) %}"
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] is string %}"
    "{{ message['content'] }}<|im_end|>\n"
    "{% else %}"
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
    "{% set image_count.value = image_count.value + 1 %}"
    "{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif content['type'] == 'video' or 'video' in content %}"
    "{% set video_count.value = video_count.value + 1 %}"
    "{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}"
    "<|vision_start|><|video_pad|><|vision_end|>"
    "{% elif 'text' in content %}"
    "{{ content['text'] }}"
    "{% endif %}"
    "{% endfor %}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def _build_messages(
    item: Dict[str, Any], base_path: Path, remaining_video_chunks: int = 3
) -> Dict[str, Any]:
    video_path = item.get("video_path")
    if not video_path:
        raise ValueError("video_path is required")

    abs_video_path = str(_make_abs_paths(base_path, video_path))
    video_duration = _get_duration(abs_video_path)
    video_chunk_size = float(math.ceil(video_duration / DEFAULT_MAX_CHUNKS))

    user_queue = sorted(
        [
            (float(c.get("timestamp", 0.0)), c.get("content", ""))
            for c in item.get("conversations", [])
            if c.get("role") == "user"
        ],
        key=lambda x: x[0],
    )

    assistant_queue = sorted(
        [
            (float(c.get("timestamp", 0.0)), c.get("content", ""))
            for c in item.get("conversations", [])
            if c.get("role") == "assistant"
        ],
        key=lambda x: x[0],
    )

    # [新增] 读取并排序 thoughts 队列
    thoughts_queue = sorted(
        [
            (float(t.get("timestamp", 0.0)), t.get("think", ""))
            for t in item.get("thoughts", [])
        ],
        key=lambda x: x[0],
    )

    max_video_chunks = int(video_duration // video_chunk_size)

    # [修改点 1] 保持原逻辑：只根据 assistant 的最后一条消息来决定视频切分长度
    last_assist_ts = assistant_queue[-1][0] if assistant_queue else 0.0
    last_assist_chunk_idx = int(np.floor(last_assist_ts / video_chunk_size))
    target_stop_idx = min(
        last_assist_chunk_idx + remaining_video_chunks, max_video_chunks - 1
    )

    # [修改点 2] 插入 System Prompt 到 messages 开头
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    video_chunks_count = 0

    for chunk_idx in range(target_stop_idx + 1):
        window_start = chunk_idx * video_chunk_size
        window_end = (chunk_idx + 1) * video_chunk_size
        is_last_chunk = chunk_idx == max_video_chunks - 1

        # --- User Content 构建 ---
        user_content = []
        user_content.append(
            {
                "type": "video",
                "video": abs_video_path,
                "video_start": window_start,
                "video_end": window_end,
            }
        )

        video_chunks_count += 1

        while user_queue and user_queue[0][0] < window_end:
            ts, text = user_queue[0]
            if ts >= window_start:
                user_queue.pop(0)
                user_content.append({"type": "text", "text": "\n" + text})
            else:
                user_queue.pop(0)  # 丢弃过期

        messages.append({"role": "user", "content": user_content})

        # --- Assistant Content 构建 (Thoughts + Response) ---

        # 1. 处理 Thoughts (与 assistant 逻辑一致)
        current_thought_segments = []
        while thoughts_queue:
            ts, content = thoughts_queue[0]

            if not is_last_chunk and ts >= window_end:
                break

            if ts >= window_start:
                thoughts_queue.pop(0)
                current_thought_segments.append(content)
            else:
                thoughts_queue.pop(0)  # 丢弃过期

        # 2. 处理 Response
        current_text_segments = []
        while assistant_queue:
            ts, content = assistant_queue[0]

            if not is_last_chunk and ts >= window_end:
                break

            if ts >= window_start:
                assistant_queue.pop(0)
                current_text_segments.append(content)
            else:
                assistant_queue.pop(0)

        # [修改点 3] 拼接逻辑：<think>...</think> + <response>/<silent>

        # 即使没有思考内容，也要输出空的 <think></think>
        thought_part = "<think>" + " ".join(current_thought_segments) + "</think>"

        if current_text_segments:
            response_part = "<response>" + " ".join(current_text_segments)
        else:
            response_part = "<silent>"

        final_content = thought_part + response_part

        assistant_content = [{"type": "text", "text": final_content}]
        messages.append({"role": "assistant", "content": assistant_content})

    total_covered_duration = min(video_chunks_count * video_chunk_size, video_duration)

    video_meta = build_video_meta(
        abs_path=abs_video_path,
        total_start=0.0,
        total_end=total_covered_duration,
        num_chunks=video_chunks_count,
    )

    return {
        "messages": messages,
        "video_meta": video_meta,
        "video_chunk_size": video_chunk_size,
    }


def process_messages_to_model_inputs(
    messages: List[Dict[str, Any]],
    video_meta: Dict[str, Any],
    video_chunk_size: float,
    processor,
    model_type: str,
    add_generation_prompt: bool = False,
    preloaded_frames: Optional[Tuple[List, dict, Optional[List]]] = None,
) -> Dict:
    """
    Common function: takes pre-built chat messages + video metadata, loads video
    frames, tokenises, and returns model-forward-ready tensors.

    This is used by both SFT (``preprocess_qwen_visual``) and GRPO (after
    rollout, to build training inputs from generated completions).

    Video pixel limits are read from the processor's ``video_processor``
    configuration.  Both ``LazySupervisedDataset`` (SFT) and
    ``LazyRawDataset`` (GRPO) call ``update_processor_pixels`` at init
    time, so the processor always carries the correct limits.

    Args:
        messages:  Full chat-format message list (system + alternating user/assistant).
                   User messages may contain ``{"type": "video", ...}`` entries.
        video_meta:  Dict with keys:
            - abs_path:         str, absolute path to the video file
            - total_start:      float, start time (seconds)
            - total_end:        float, end time (seconds)
            - num_chunks:       int, number of video chunks
            - frames_per_chunk: int, frames per chunk
            - total_nframes:    int, total number of frames
        video_chunk_size: float, seconds per video chunk (needed for RoPE
            computation downstream).
        processor:  HuggingFace processor (e.g. ``AutoProcessor``).
        add_generation_prompt:  Whether to append the generation prompt
            (``<|im_start|>assistant\\n``) at the end.
        preloaded_frames:  Optional pre-loaded ``(split_videos, video_kwargs,
            chunk_metadatas)`` tuple.  When provided, ``load_video_frames``
            is skipped entirely.

    Returns:
        Dict containing at least:
            - input_ids:            [1, L]
            - pixel_values_videos:  video pixel tensors
            - video_grid_thw:       [num_chunks, 3]
            - video_mask:           [1, L]  (True where video tokens appear)
            - video_chunk_size:     float (pass-through for RoPE)
    """
    is_qwen3vl = model_type == "qwen3vl"

    # 1. Load ALL video frames at once via unified loader
    if preloaded_frames is not None:
        split_videos, video_kwargs, chunk_metadatas = preloaded_frames
    else:
        split_videos, video_kwargs, chunk_metadatas = load_video_frames(
            video_path=video_meta["abs_path"],
            video_start=video_meta["total_start"],
            video_end=video_meta["total_end"],
            total_nframes=video_meta["total_nframes"],
            frames_per_chunk=video_meta["frames_per_chunk"],
            num_chunks=video_meta["num_chunks"],
            processor=processor,
            model_type=model_type,
        )

    # 2. Tokenise
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    processor_call_kwargs = dict(
        text=text,
        images=None,
        videos=split_videos,
        return_tensors="pt",
        **video_kwargs,
    )
    if is_qwen3vl:
        processor_call_kwargs["do_resize"] = False
        if chunk_metadatas is not None:
            processor_call_kwargs["video_metadata"] = chunk_metadatas
    full_result = processor(**processor_call_kwargs)

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)
    full_result["input_ids"] = input_ids

    # 4. Video mask
    video_token_id = processor.tokenizer.convert_tokens_to_ids(["<|video_pad|>"])[0]
    assert "video_mask" not in full_result, "Key Conflict!"
    full_result["video_mask"] = input_ids == video_token_id

    # 5. Pass-through for RoPE computation
    full_result["video_chunk_size"] = video_chunk_size

    return full_result


# ---------------------------------------------------------------------------
# Unified helpers (shared across SFT, GRPO, and Eval)
# ---------------------------------------------------------------------------


def find_assistant_spans(
    input_ids_1d: List[int],
    tokenizer,
) -> List[Tuple[int, int]]:
    """Locate all assistant-turn token spans in a flat token sequence.

    Scans *input_ids_1d* for the ``assistant`` token.  For each occurrence the
    assistant content starts 2 positions later (skipping ``assistant\\n``) and
    extends until ``<|im_end|>`` (inclusive) plus one trailing token (the
    ``\\n`` that follows ``<|im_end|>``).

    Returns:
        A list of ``(start, end_exclusive)`` tuples suitable for slicing, e.g.
        ``input_ids[start:end_exclusive]``.
    """
    assistant_id, im_end_id = tokenizer.convert_tokens_to_ids(
        ["assistant", "<|im_end|>"]
    )
    spans: List[Tuple[int, int]] = []
    L = len(input_ids_1d)
    pos = 0
    while pos < L:
        if input_ids_1d[pos] == assistant_id:
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < L and input_ids_1d[ans_end] != im_end_id:
                ans_end += 1
            if ans_end < L:
                spans.append((ans_start, ans_end + 2))
                pos = ans_end
        pos += 1
    return spans


def compute_position_ids(
    processor_output: Dict[str, Any],
    processor,
    model_type: str,
) -> torch.Tensor:
    """Compute MROPE position IDs – the **single** path for all workflows.

    Used by SFT, GRPO training reconstruction, **and** inference
    (``streaming_video_chat``).  Callers must ensure ``video_chunk_size``
    is present in *processor_output* so that ``second_per_grid_ts`` is
    computed identically everywhere.

    Handles ``image_grid_thw`` / ``video_grid_thw`` normalisation,
    ``second_per_grid_ts`` computation from ``video_chunk_size``, and
    dispatches to the correct RoPE function for the given *model_type*.

    The ``video_chunk_size`` key is **consumed** (popped) from
    *processor_output* during this call.

    Returns:
        ``position_ids`` tensor of shape ``(3, B, L)`` for MROPE models.
    """
    input_ids = processor_output["input_ids"]

    # --- image_grid_thw ---
    if "image_grid_thw" in processor_output:
        image_grid_thw = processor_output["image_grid_thw"]
        if not isinstance(image_grid_thw, (list, tuple)):
            image_grid_thw = [image_grid_thw]
        image_grid_thw = torch.cat(image_grid_thw, dim=0)
    else:
        image_grid_thw = None

    # --- video_grid_thw + second_per_grid_ts ---
    if "video_grid_thw" in processor_output:
        video_grid_thw = processor_output["video_grid_thw"]
        if not isinstance(video_grid_thw, (list, tuple)):
            video_grid_thw = [video_grid_thw]
        video_grid_thw = torch.cat(video_grid_thw, dim=0)
        second_per_grid_ts = [
            processor_output.pop("video_chunk_size", 1)
            * processor.video_processor.temporal_patch_size
            / processor.video_processor.fps
        ] * len(video_grid_thw)
    else:
        video_grid_thw = None
        second_per_grid_ts = None

    merge_size = getattr(processor.image_processor, "merge_size", 2)
    rope_fn = ROPE_INDEX_FN[model_type]

    position_ids, _ = rope_fn(
        merge_size,
        input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
    )
    return position_ids


def preprocess_qwen_visual(sources, processor, model_type: str) -> Dict:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")
    source = sources[0]
    base_path = Path(source.get("data_path", ""))

    build_result = _build_messages(source, base_path)
    messages = build_result["messages"]
    video_meta = build_result["video_meta"]
    video_chunk_size = build_result["video_chunk_size"]

    # Use common function for video loading + tokenisation
    full_result = process_messages_to_model_inputs(
        messages=messages,
        video_meta=video_meta,
        video_chunk_size=video_chunk_size,
        processor=processor,
        model_type=model_type,
        add_generation_prompt=False,
    )

    # SFT-specific: create labels (only train on assistant turns)
    input_ids = full_result["input_ids"]
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    for start, end in find_assistant_spans(input_ids[0].tolist(), processor.tokenizer):
        labels[0, start:end] = input_ids[0, start:end]

    full_result["labels"] = labels
    return full_result


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, processor, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type not in ROPE_INDEX_FN:
            raise ValueError(
                f"model_type: {data_args.model_type} not supported. "
                f"Choose from {list(ROPE_INDEX_FN.keys())}"
            )
        self.get_rope_index = ROPE_INDEX_FN[data_args.model_type]

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                rank0_print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                if isinstance(ann, list):
                    for sub_ann in ann:
                        sub_ann["data_path"] = data["data_path"]
                else:
                    ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)  # Randomly shuffle the data for training
        rank0_print("Formatting inputs...Skip in lazy mode")
        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.list_data_dict = list_data_dict
        self.item_fn = self._get_item

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        for attempt_idx in range(num_base_retries):
            try:
                sources = self.list_data_dict[i]
                if isinstance(sources, dict):
                    sources = [sources]
                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        for attempt_idx in range(num_final_retries):
            try:
                random_index = random.randint(0, len(self.list_data_dict) - 1)
                if random_index == i:
                    continue

                sources = self.list_data_dict[random_index]
                if isinstance(sources, dict):
                    sources = [sources]

                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try random #{attempt_idx}] Failed to fetch random sample {random_index}. Exception:",
                    e,
                )
                pass

        try:
            sources = self.list_data_dict[i]
            if isinstance(sources, dict):
                sources = [sources]
            sample = self.item_fn(sources)
            return sample
        except Exception as e:
            raise

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        data_dict = preprocess_qwen_visual(
            sources,
            self.processor,
            model_type=self.model_type,
        )

        seq_len = data_dict["input_ids"][0].size(0)

        data_dict["position_ids"] = compute_position_ids(
            data_dict,
            self.processor,
            self.model_type,
        )
        data_dict["attention_mask"] = [seq_len]

        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    vocab_size: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, video_masks = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "video_mask")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        video_masks = [ids.squeeze(0) for ids in video_masks]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        video_masks = torch.nn.utils.rnn.pad_sequence(
            video_masks, batch_first=True, padding_value=0
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        video_masks = video_masks[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            video_mask=video_masks,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids

        # CE Weight
        response_id = self.tokenizer.convert_tokens_to_ids(["<response>"])[0]
        silent_id = self.tokenizer.convert_tokens_to_ids(["<silent>"])[0]
        n_response = (batch["labels"] == response_id).sum()
        n_silent = (batch["labels"] == silent_id).sum()
        total_n = n_response + n_silent
        ce_weight = torch.ones(self.vocab_size)
        eps = 1e-3
        ce_weight[silent_id] = total_n / (2 * n_silent + eps)
        ce_weight[response_id] = total_n / (2 * n_response + eps)
        batch["ce_weight"] = torch.clamp(ce_weight, 0, 20)
        print(input_ids.shape, batch["ce_weight"][[silent_id, response_id]])
        return batch


def make_supervised_data_module(processor, data_args, vocab_size: int) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(processor, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(
        processor.tokenizer, vocab_size=vocab_size
    )
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


# ---------------------------------------------------------------------------
# Raw (unprocessed) Dataset for GRPO
# ---------------------------------------------------------------------------


class LazyRawDataset(Dataset):
    """
    Dataset that returns raw JSON annotations with optional video preloading.

    Used by the GRPO pipeline: the rollout stage receives raw data and
    performs streaming inference to generate completions.

    When *model_type* is provided, ``__getitem__`` also pre-loads video
    frames via ``preload_video`` so that DataLoader ``num_workers`` can
    parallelise the heavy video I/O.

    Like ``LazySupervisedDataset``, the processor's pixel limits are updated
    via ``update_processor_pixels`` at init time so that downstream callers
    (``process_messages_to_model_inputs``, ``load_video_frames``) always see
    the correct resolution config.
    """

    def __init__(
        self,
        processor,
        data_args,
        *,
        frames_per_chunk: int = FRAMES_PER_CHUNK,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
        model_type: str = "",
    ):
        super().__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"[LazyRawDataset] Loading datasets: {dataset_list}")

        list_data_dict: List[Dict] = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                rank0_print(
                    f"  sampling {len(annotations)} examples from dataset {data}"
                )
            else:
                rank0_print(f"  dataset name: {data}")
            for ann in annotations:
                if isinstance(ann, list):
                    for sub_ann in ann:
                        sub_ann["data_path"] = data["data_path"]
                else:
                    ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        random.shuffle(list_data_dict)
        rank0_print(f"[LazyRawDataset] Total raw samples: {len(list_data_dict)}")

        update_processor_pixels(processor, data_args)
        self.processor = processor
        self.list_data_dict = list_data_dict

        self._do_preload = bool(model_type)
        self._model_type = model_type
        self._frames_per_chunk = frames_per_chunk
        self._max_chunks = max_chunks
        min_px, max_px = _get_video_pixels(processor)
        self._min_pixels = min_px
        self._max_pixels = max_px
        self._vit_patch_size = _resolve_vit_patch_size(processor)

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, Any]:
        item = self.list_data_dict[i]
        if not self._do_preload:
            return item

        data_path = item.get("data_path", "")
        video_path = item.get("video_path", "")
        abs_video_path = str(_make_abs_paths(Path(data_path), video_path))

        preloaded = preload_video(
            abs_video_path,
            frames_per_chunk=self._frames_per_chunk,
            max_chunks=self._max_chunks,
            min_pixels=self._min_pixels,
            max_pixels=self._max_pixels,
            vit_patch_size=self._vit_patch_size,
            model_type=self._model_type,
        )
        item = {**item, "_preloaded_video": preloaded}
        return item


def raw_data_collate_fn(instances: Sequence[Dict]) -> List[Dict]:
    """Collator that passes through raw dicts without any processing."""
    return list(instances)


def make_raw_data_module(
    processor,
    data_args,
    *,
    frames_per_chunk: int = FRAMES_PER_CHUNK,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    model_type: str = "",
) -> Dict:
    """Make dataset and collator that return raw JSON data (for GRPO)."""
    train_dataset = LazyRawDataset(
        processor,
        data_args,
        frames_per_chunk=frames_per_chunk,
        max_chunks=max_chunks,
        model_type=model_type,
    )
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=raw_data_collate_fn,
    )
