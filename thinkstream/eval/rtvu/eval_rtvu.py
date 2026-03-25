import sys
import os
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


# ─── Evaluation ──────────────────────────────────────────────────────────────


def _acc(counts):
    return counts["correct"] / counts["total"] if counts["total"] else 0.0


def _update(bucket, key, correct):
    bucket.setdefault(key, {"correct": 0, "total": 0})
    bucket[key]["total"] += 1
    if correct:
        bucket[key]["correct"] += 1


def evaluate_rtvu_results(results: list):
    """
    Compute accuracy grouped by:
      1. Overall
      2. Per task_type         (Object Perception, Causal Reasoning, …)
      3. Per frames_required   (single / multiple)
      4. Per temporal_clue_type (Prior / Concurrent)
      5. Per sample_id          (per-video breakdown)
    """
    task_counts = {}
    frames_counts = {}
    temporal_counts = {}
    sample_counts = {}
    total_correct = 0
    total = len(results)

    for r in results:
        correct = r["response"] == r["answer"]
        if correct:
            total_correct += 1

        _update(task_counts, r.get("task_type", r.get("task", "Unknown")), correct)
        _update(frames_counts, r.get("frames_required", "unknown"), correct)
        _update(temporal_counts, r.get("temporal_clue_type", "unknown"), correct)
        _update(sample_counts, r.get("sample_id", "unknown"), correct)

    print("=" * 60)
    print("Real-Time Visual Understanding Evaluation Results")
    print("=" * 60)
    print(
        f"\nOverall: {total_correct}/{total} = {_acc({'correct': total_correct, 'total': total}):.4f}"
    )

    print("\n--- Per Task Type ---")
    for task, c in sorted(task_counts.items()):
        print(f"  {task:<28s}: {c['correct']:>4d}/{c['total']:<4d} = {_acc(c):.4f}")

    print("\n--- Per Frames Required ---")
    for fr, c in sorted(frames_counts.items()):
        print(f"  {fr:<12s}: {c['correct']:>4d}/{c['total']:<4d} = {_acc(c):.4f}")

    print("\n--- Per Temporal Clue Type ---")
    for tc, c in sorted(temporal_counts.items()):
        print(f"  {tc:<12s}: {c['correct']:>4d}/{c['total']:<4d} = {_acc(c):.4f}")

    task_accs = [_acc(c) for c in task_counts.values() if c["total"]]
    if task_accs:
        print(f"\nMean per-task-type accuracy: {sum(task_accs) / len(task_accs):.4f}")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RTVU with streaming model.")
    add_common_args(parser)
    args = parser.parse_args()

    local_rank, rank, world_size = setup_distributed()

    benchmark_path = os.path.join(args.benchmark_dir, "rtvu-formatted.jsonl")
    model, processor = load_model_and_processor(
        args.model_path,
        local_rank,
        model_type=args.model_type,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    options = ["A", "B", "C", "D"]

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
        save_path = os.path.join(args.model_path, "eval", "rtvu", filename)
        save_results(results, save_path, evaluate_rtvu_results)

    cleanup_distributed(world_size)
