import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_common import (
    add_common_args,
    setup_distributed,
    cleanup_distributed,
    load_model_and_processor,
    mcq_predict_streaming,
    build_results,
    save_results,
)


def evaluate_ovobench_results(results: list):
    task_to_counts = {}
    for result in results:
        task = result["task"]
        if task not in task_to_counts:
            task_to_counts[task] = {"correct": 0, "total": 0}
        task_to_counts[task]["total"] += 1
        if result["response"][: len(result["answer"])] == result["answer"]:
            task_to_counts[task]["correct"] += 1

    rt_accs, bt_accs, fr_accs = [], [], []
    for task, counts in task_to_counts.items():
        acc = counts["correct"] / counts["total"]
        print(f"{task}: {counts['correct']}/{counts['total']}={acc}")
        if task in ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]:
            rt_accs.append(acc)
        elif task in ["EPM", "ASI", "HLD"]:
            bt_accs.append(acc)
        else:
            fr_accs.append(acc)

    if rt_accs:
        print(
            f"Real-Time Visual Perception avg.: {sum(rt_accs)}/{len(rt_accs)}={sum(rt_accs) / len(rt_accs)}"
        )
    if bt_accs:
        print(
            f"Backward Tracing avg.: {sum(bt_accs)}/{len(bt_accs)}={sum(bt_accs) / len(bt_accs)}"
        )
    if fr_accs:
        print(
            f"Forward Tracing avg.: {sum(fr_accs)}/{len(fr_accs)}={sum(fr_accs) / len(fr_accs)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test OVO-Bench with streaming model.")
    add_common_args(parser)
    args = parser.parse_args()

    local_rank, rank, world_size = setup_distributed()

    benchmark_path = os.path.join(args.benchmark_dir, "ovo-bench-formatted.jsonl")
    model, processor = load_model_and_processor(
        args.model_path,
        local_rank,
        model_type=args.model_type,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    options = [
        "No",
        "Yes",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "A",
        "B",
        "C",
        "D",
        "E",
    ]

    predictions, datums, process_index = mcq_predict_streaming(
        model=model,
        processor=processor,
        benchmark_path=benchmark_path,
        options=options,
        frames_per_chunk=args.frames_per_chunk,
        max_new_tokens=args.max_new_tokens,
        remaining_seconds=args.remaining_seconds,
        think_budget=args.think_budget,
        rank=rank,
        world_size=world_size,
        model_type=args.model_type,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        slack_time=args.slack_time,
    )

    if process_index == 0:
        results = build_results(datums, predictions, options)
        filename = f"result_{args.min_pixels}_{args.max_pixels}_{args.remaining_seconds}_{args.think_budget}_{args.max_new_tokens}.json"
        save_path = os.path.join(args.model_path, "eval", "ovo_bench", filename)
        save_results(results, save_path, evaluate_ovobench_results)

    cleanup_distributed(world_size)
