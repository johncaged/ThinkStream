import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
import torch
import time
import gc
from transformers import AutoProcessor
from thinkstream.model import MODEL_CLS, DEFAULT_VIDEO_FLEX_WINDOW_SIZE
from thinkstream.model.inference import (
    StreamingWindowInferenceEngine,
    streaming_video_chat,
    think_budget_sample,
)
from thinkstream.data.stream_data_processor import (
    SYSTEM_PROMPT,
    QWEN_TEMPLATE_WO_SYSTEM,
)

# ================= 配置参数 =================
MODEL_ID = "/your/model/ckpt"
VIDEO_PATH = "/path/to/a/video.mp4"
NUM_SECONDS = 90
FPS = 2.0
MAX_NEW_TOKENS = 30
MAX_THINK_TOKENS = 20
NUM_GENERATIONS = 1
MODEL_TYPE = "qwen2.5vl"
MIN_PIXELS = 100352
MAX_PIXELS = 150528


def split_think_and_response(text: str):
    """将 <think>...</think> 部分与后面内容分开，返回 (think_parts, response)。"""
    if not text or "</think>" not in text:
        return ([], text.strip() if text else "")
    parts = text.split("</think>")
    # 最后一段是 </think> 之后的「回复」；前面每段要再剥掉 <think> 得到「思考」
    think_parts = []
    for i, block in enumerate(parts[:-1]):
        block = block.strip()
        if block.startswith("<think>"):
            block = block[len("<think>"):].strip()  # 去掉开头的 <think>
        if block:
            think_parts.append(block)
    response = (parts[-1] or "").strip()
    if response.endswith("<|im_end|>"):
        response = response[:-len("<|im_end|>")]
    return (think_parts, response)


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def run_video_streaming_benchmark():
    print(f"\n{'='*20} Testing: Real Video Streaming (2 FPS) {'='*20}")
    clear_gpu()

    # 1. 加载模型与处理器
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    vp = processor.video_processor
    vp.min_pixels = MIN_PIXELS
    vp.max_pixels = MAX_PIXELS
    vp.size["shortest_edge"] = MIN_PIXELS
    vp.size["longest_edge"] = MAX_PIXELS

    model = MODEL_CLS[MODEL_TYPE].from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
    )
    model.config.text_config._attn_implementation = "flash_attention_2_infer"
    model.eval()

    video_token_id = processor.tokenizer.convert_tokens_to_ids(["<|video_pad|>"])[0]
    video_flex_window_size = getattr(
        model.config, "video_flex_window_size", DEFAULT_VIDEO_FLEX_WINDOW_SIZE,
    )
    if MODEL_TYPE == "qwen3vl":
        text_config = model.config.text_config
    else:
        text_config = model.config
    engine = StreamingWindowInferenceEngine(
        model,
        batch_size=NUM_GENERATIONS,
        max_len=16384,
        num_hidden_layers=text_config.num_hidden_layers,
        num_key_value_heads=text_config.num_key_value_heads,
        head_dim=text_config.hidden_size // text_config.num_attention_heads,
        vocab_size=text_config.vocab_size,
        pad_token_id=model.generation_config.pad_token_id,
        eos_token_ids=model.generation_config.eos_token_id,
        video_token_id=video_token_id,
        video_flex_window_size=video_flex_window_size,
    )

    # 2. 思考长度限制（与 GRPO rollout 一致）
    think_end_token_id = processor.tokenizer.convert_tokens_to_ids("</think>")
    sample_kwargs = {
        "think_end_token_id": think_end_token_id,
        "max_think_tokens": MAX_THINK_TOKENS,
    }

    # 3. 使用 streaming_video_chat 统一推理
    queries = [
        {
            "content": "Question/Instruction here",
            "timestamp": 0.0,  # Question time here.
        },
    ]

    total_token_count = 0
    start_time = time.time()

    for result in streaming_video_chat(
        engine=engine,
        processor=processor,
        video_path=VIDEO_PATH,
        queries=queries,
        video_start=0.0,
        video_end=float(NUM_SECONDS),
        frames_per_chunk=int(FPS),
        max_chunks=NUM_SECONDS,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        max_new_tokens=MAX_NEW_TOKENS,
        num_generations=NUM_GENERATIONS,
        system_prompt=SYSTEM_PROMPT,
        chat_template_wo_system=QWEN_TEMPLATE_WO_SYSTEM,
        sample=think_budget_sample,
        sample_kwargs=sample_kwargs,
        model_type=MODEL_TYPE,
        break_on_answer=False,
    ):
        idx = result["chunk_idx"]
        gen_tokens = result["generated_tokens"]
        decoded = processor.batch_decode([gen_tokens[0]])[0]
        think_parts, response = split_think_and_response(decoded)

        print(f"\n[Chunk {idx}] has_query={result['has_query']}")
        if think_parts:
            for i, think in enumerate(think_parts, 1):
                print(f"  [think {i}] {think}")
        if response:
            print(f"  [response] {response}")
        if not think_parts and not response:
            print(f"  (empty) {decoded!r}")
        total_token_count += max(t.shape[0] for t in gen_tokens) * NUM_GENERATIONS

    torch.cuda.synchronize()
    end_time = time.time()

    # 4. 指标计算
    latency = end_time - start_time
    print(f"\nBenchmark Results:")
    print(f"Total Time: {latency:.2f}s for {NUM_SECONDS}s video")
    print(f"Total new: {total_token_count}")
    print(f"Speed: {total_token_count / latency} tokens/s (batch_size={NUM_GENERATIONS})")


if __name__ == "__main__":
    run_video_streaming_benchmark()
