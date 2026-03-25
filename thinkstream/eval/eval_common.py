import sys
import contextlib
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _EVAL_DIR.parent
_PROJECT_ROOT = _WORKSPACE_ROOT.parent

for _p in (str(_PROJECT_ROOT), str(_WORKSPACE_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gc
import json
import os
import random
import numpy as np
import torch
import torch.distributed as dist
import tqdm
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoProcessor
from thinkstream.model import MODEL_CLS, DEFAULT_VIDEO_FLEX_WINDOW_SIZE, get_text_config
from thinkstream.model.inference import (
    StreamingWindowInferenceEngine,
    think_budget_sample_restricted,
    streaming_video_chat,
)
from thinkstream.data.stream_data_processor import (
    SYSTEM_PROMPT,
    QWEN_TEMPLATE_WO_SYSTEM,
    FRAMES_PER_CHUNK,
    DEFAULT_MAX_CHUNKS,
    DEFAULT_INFERENCE_MIN_PIXELS,
    DEFAULT_INFERENCE_MAX_PIXELS,
    preload_video,
    _resolve_vit_patch_size,
)

# ─── Constants ───────────────────────────────────────────────────────────────

MAX_NEW_TOKENS = 30
MIN_PIXELS = DEFAULT_INFERENCE_MIN_PIXELS
MAX_PIXELS = DEFAULT_INFERENCE_MAX_PIXELS

# ─── Utility Classes ─────────────────────────────────────────────────────────


class TeeWriter:
    """Write to multiple streams simultaneously."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


class NoPadDistributedSampler(Sampler):
    """Round-robin sampler that does NOT pad/duplicate samples for even division.
    Each rank gets indices[rank::world_size], so ranks may have different lengths."""

    def __init__(self, dataset: Dataset, num_replicas: int, rank: int):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        return (
            len(self.dataset) - self.rank + self.num_replicas - 1
        ) // self.num_replicas


class MCQDataset(Dataset):
    """Generic MCQ dataset that loads JSONL files.

    When *processor* and *model_type* are provided the dataset pre-loads
    video frames in ``__getitem__`` so that DataLoader ``num_workers`` can
    parallelise the heavy video I/O.
    """

    def __init__(
        self,
        path,
        sample=None,
        *,
        processor=None,
        model_type: str = "",
        frames_per_chunk: int = FRAMES_PER_CHUNK,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        slack_time: float = 0.0,
    ):
        lines = open(path).readlines()
        if sample is not None:
            random.seed(42)
            lines = random.sample(lines, sample)

        self.datums = [
            json.loads(line) for line in tqdm.tqdm(lines, desc="Loading data")
        ]
        if self.datums and isinstance(self.datums[0], str):
            self.datums = [
                json.loads(d) for d in tqdm.tqdm(self.datums, desc="Loading data (x2)")
            ]

        self.data_dir = os.path.dirname(path)

        self._do_preload = processor is not None and model_type
        self._model_type = model_type
        self._frames_per_chunk = frames_per_chunk
        self._max_chunks = max_chunks
        self._min_pixels = min_pixels
        self._max_pixels = max_pixels
        self._slack_time = slack_time
        self._vit_patch_size = (
            _resolve_vit_patch_size(processor) if processor is not None else None
        )

    def __len__(self):
        return len(self.datums)

    def __getitem__(self, i):
        datum = self.datums[i]
        if not self._do_preload:
            return i, datum, None

        video_path = os.path.join(self.data_dir, datum["video"])
        try:
            video_end = datum.get("video_end", None)
            if video_end is None:
                raise ValueError(f"video_end is None for datum {datum}")
            original_video_end = video_end
            if self._slack_time > 0:
                original_video_end = video_end
                video_end = video_end + self._slack_time

            preloaded = preload_video(
                video_path,
                video_start=datum.get("video_start", 0.0),
                video_end=video_end,
                frames_per_chunk=self._frames_per_chunk,
                max_chunks=self._max_chunks,
                min_pixels=self._min_pixels,
                max_pixels=self._max_pixels,
                vit_patch_size=self._vit_patch_size,
                model_type=self._model_type,
            )
            if original_video_end is not None:
                preloaded["original_video_end"] = original_video_end
        except Exception:
            preloaded = None
        return i, datum, preloaded


# ─── Setup Helpers ───────────────────────────────────────────────────────────


def setup_distributed():
    """Initialize distributed process group. Returns (local_rank, rank, world_size)."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return local_rank, rank, world_size


def cleanup_distributed(world_size):
    if world_size > 1:
        dist.destroy_process_group()


def load_model_and_processor(
    model_path,
    local_rank=0,
    *,
    model_type: str,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
):
    """Load model + processor and apply standard configuration."""
    if model_type not in MODEL_CLS:
        raise ValueError(
            f"Unsupported model_type: {model_type}. Choose from {list(MODEL_CLS.keys())}"
        )
    model = MODEL_CLS[model_type].from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=f"cuda:{local_rank}",
    )

    processor = AutoProcessor.from_pretrained(model_path, padding_side="left")

    vp = processor.video_processor
    vp.max_pixels = max_pixels
    vp.min_pixels = min_pixels
    vp.size["shortest_edge"] = min_pixels
    vp.size["longest_edge"] = max_pixels

    model.config.text_config._attn_implementation = "flash_attention_2_infer"
    model.eval()

    return model, processor


def add_common_args(parser):
    """Add benchmark-agnostic CLI arguments."""
    parser.add_argument(
        "--benchmark_dir", type=str, required=True, help="Path to benchmark directory."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model directory."
    )
    parser.add_argument("--frames_per_chunk", type=int, default=FRAMES_PER_CHUNK)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument(
        "--sample", type=int, default=None, help="Subsample N items (for debugging)."
    )
    parser.add_argument(
        "--remaining_seconds",
        type=int,
        default=DEFAULT_MAX_CHUNKS,
        help="Max seconds to process.",
    )
    parser.add_argument(
        "--think_budget", type=int, default=20, help="Max thinking tokens budget."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="qwen2.5vl",
        choices=list(MODEL_CLS.keys()),
        help="Model type.",
    )
    parser.add_argument(
        "--slack_time", type=float, default=3.0, help="Slack time window in seconds."
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=MIN_PIXELS,
        help="Minimum number of pixels for video processing.",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=MAX_PIXELS,
        help="Maximum number of pixels for video processing.",
    )
    return parser


# ─── Core Prediction ─────────────────────────────────────────────────────────


def preprocess_logits_for_metrics(logits, labels, strict_option_ids):
    """Extract logits for option tokens at the last non-padding position."""
    return torch.stack(
        [
            logit[(logit[:, 0] != -100).nonzero().squeeze()[-1], strict_option_ids]
            for logit in logits
        ]
    ).argmax(dim=-1)


@torch.inference_mode()
def mcq_predict_streaming(
    model,
    processor,
    benchmark_path: str,
    options: list[str],
    question_prefix: str = "",
    question_postfix: str = "\nPlease select the correct answer.",
    max_len: int = 24576,
    frames_per_chunk: int = FRAMES_PER_CHUNK,
    max_new_tokens: int = MAX_NEW_TOKENS,
    remaining_seconds: int = DEFAULT_MAX_CHUNKS,
    think_budget: int = 20,
    rank: int = 0,
    world_size: int = 1,
    *,
    model_type: str,
    slack_time: float = 3.0,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
):
    """
    Generic streaming MCQ prediction.

    The JSONL file at *benchmark_path* must contain one JSON object per line
    with at least: ``video``, ``question``, ``video_start``, ``video_end``.
    If an ``options`` key is present its values are appended to the question.
    """
    # The sampler picks the top-1 token among restricted_token_ids via argmax
    # over logits, so the relative ranking is all that matters — the exact
    # token variant (with / without prefix space) does not affect results.
    strict_option_ids = [
        processor.tokenizer(opt, add_special_tokens=False).input_ids[-1]
        for opt in options
    ]

    think_end_token_id = processor.tokenizer.convert_tokens_to_ids("</think>")
    silent_token_id = processor.tokenizer.convert_tokens_to_ids("<silent>")
    response_token_id = processor.tokenizer.convert_tokens_to_ids("<response>")
    eos_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    _base_sample_kwargs = {
        "think_end_token_id": think_end_token_id,
        "max_think_tokens": think_budget,
        "eos_token_id": eos_token_id,
        "silent_token_id": silent_token_id,
        "response_token_id": response_token_id,
    }
    # We only use one sample function with restricted token ids.
    # The logic in streaming_video_chat/think_budget_sample_restricted will handle:
    # - is_query_window=False (Observation) -> Silent output
    # - is_query_window=True (Answer) -> Constrained output using restricted_token_ids
    sample_kwargs = {**_base_sample_kwargs, "restricted_token_ids": strict_option_ids}

    dataset = MCQDataset(
        benchmark_path,
        processor=processor,
        model_type=model_type,
        frames_per_chunk=frames_per_chunk,
        max_chunks=remaining_seconds,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        slack_time=slack_time,
    )

    if world_size > 1:
        sampler = NoPadDistributedSampler(dataset, num_replicas=world_size, rank=rank)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=lambda batch: batch[0],
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True,
    )

    video_token_id = processor.tokenizer.convert_tokens_to_ids(["<|video_pad|>"])[0]
    video_flex_window_size = getattr(
        model.config, "video_flex_window_size", DEFAULT_VIDEO_FLEX_WINDOW_SIZE
    )
    text_cfg = get_text_config(model.config)
    engine = StreamingWindowInferenceEngine(
        model,
        batch_size=1,
        max_len=max_len,
        num_hidden_layers=text_cfg.num_hidden_layers,
        num_key_value_heads=text_cfg.num_key_value_heads,
        head_dim=text_cfg.hidden_size // text_cfg.num_attention_heads,
        vocab_size=text_cfg.vocab_size,
        pad_token_id=model.generation_config.pad_token_id,
        eos_token_ids=model.generation_config.eos_token_id,
        video_token_id=video_token_id,
        video_flex_window_size=video_flex_window_size,
    )

    predictions = []
    datums = []
    local_indices = []

    for idx, datum, preloaded in tqdm.tqdm(
        dataloader, desc="Processing samples", disable=(rank != 0)
    ):
        try:
            video_path = os.path.join(dataset.data_dir, datum["video"])

            # Determine query timestamp
            if preloaded is not None and "original_video_end" in preloaded:
                query_ts = preloaded["original_video_end"]
                # Use extended video_end from preloaded
                run_video_end = preloaded["video_end"]
            else:
                # Fallback if no preloading or no slack applied (shouldn't happen if slack_time > 0)
                # But if slack_time=0, we just use datum video_end or None
                base_end = datum.get("video_end")
                if base_end is not None:
                    query_ts = base_end
                    run_video_end = (
                        base_end + slack_time if slack_time > 0 else base_end
                    )
                else:
                    # Critical failure to determine timing
                    raise ValueError(
                        f"Sample {idx}: Cannot determine video_end/query time."
                    )

            if "options" in datum and datum["options"]:
                query = (
                    question_prefix
                    + datum["question"]
                    + "\n"
                    + "\n".join(datum["options"])
                    + question_postfix
                )
            else:
                query = datum["question"]

            for result in streaming_video_chat(
                engine=engine,
                processor=processor,
                video_path=video_path,
                queries=[{"content": query, "timestamp": query_ts}],
                video_start=datum.get("video_start", 0.0),
                video_end=run_video_end,
                frames_per_chunk=frames_per_chunk,
                max_chunks=remaining_seconds,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                max_new_tokens=max_new_tokens,
                system_prompt=SYSTEM_PROMPT,
                chat_template_wo_system=QWEN_TEMPLATE_WO_SYSTEM,
                sample=think_budget_sample_restricted,
                sample_kwargs=sample_kwargs,
                model_type=model_type,
                preloaded_video=preloaded,
                slack_time=slack_time,
                break_on_answer=True,
            ):
                if result["is_answer"]:
                    gen_tokens = result["generated_tokens"][0]
                    # We expect: ... <think>...</think><response> ANSWER <|im_end|>
                    try:
                        # Find position of <response>
                        resp_pos = (
                            (gen_tokens == response_token_id)
                            .nonzero(as_tuple=True)[0][0]
                            .item()
                        )
                        # Answer is the next token
                        ans_token = gen_tokens[resp_pos + 1].item()
                        predicted_idx = strict_option_ids.index(ans_token)
                    except (IndexError, ValueError):
                        # Fallback search if strict structure failed (shouldn't with restricted sampling)
                        print(
                            f"Warning: could not parse answer from {processor.decode(gen_tokens)}"
                        )
                        predicted_idx = random.randint(0, len(options) - 1)

                    print(
                        f"[Answer] {processor.decode(gen_tokens)} -> {options[predicted_idx]}"
                    )
                    predictions.append(predicted_idx)
                    datums.append({**datum, "success": True})
                    local_indices.append(idx)

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            import traceback

            traceback.print_exc()
            predictions.append(random.randint(0, len(options) - 1))
            datums.append({**datum, "success": False})
            local_indices.append(idx)
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    if world_size > 1:
        local_data = list(zip(local_indices, predictions, datums))
        gathered_data = [None] * world_size
        dist.all_gather_object(gathered_data, local_data)

        if rank == 0:
            all_data = []
            for proc_data in gathered_data:
                all_data.extend(proc_data)
            all_data.sort(key=lambda x: x[0])
            return np.array([d[1] for d in all_data]), [d[2] for d in all_data], 0
        else:
            return np.array(predictions), datums, rank
    else:
        return np.array(predictions), datums, 0


# ─── Results I/O ─────────────────────────────────────────────────────────────


def build_results(datums, predictions, options):
    """Merge all original datum fields with the predicted response."""
    return [
        {**datum, "response": options[pred_idx]}
        for datum, pred_idx in zip(datums, predictions)
    ]


def save_results(results, save_json_path, evaluate_fn):
    """Persist full results to JSON and write evaluation summary to both stdout and a .txt file."""
    os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    save_txt_path = save_json_path.replace(".json", ".txt")
    with open(save_txt_path, "w", encoding="utf-8") as f:
        tee = TeeWriter(sys.stdout, f)
        with contextlib.redirect_stdout(tee):
            evaluate_fn(results)
