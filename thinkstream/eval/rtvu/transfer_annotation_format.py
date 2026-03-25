import csv
import json
import re
import argparse


def parse_timestamp(ts: str) -> float:
    """Convert ``HH:MM:SS`` or ``MM:SS`` to seconds."""
    parts = ts.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return float(ts)


def parse_options(raw: str) -> list:
    """Parse a Python-style list-of-strings literal using json.loads.

    Converts boundary single quotes to double quotes so that interior
    apostrophes (e.g. "player's") are preserved correctly.
    """
    raw = raw.strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    converted = re.sub(r"([\[,])\s*'", r'\1"', raw)
    converted = re.sub(r"'\s*([,\]])", r'"\1', converted)
    try:
        return json.loads(converted)
    except (json.JSONDecodeError, ValueError):
        pass
    items = re.findall(r"""['"](.+?)['"]""", raw)
    if items:
        return items
    raise ValueError(f"Cannot parse options: {raw[:80]}")


def transfer_rtvu(input_path, output_path, video_format="sample_{sample_id}/video.mp4"):
    """
    Convert Real_Time_Visual_Understanding.csv → JSONL for mcq_predict_streaming.

    Original CSV columns:
        question_id, task_type, question, time_stamp, answer, options,
        frames_required, temporal_clue_type

    question_id pattern: ``Real-Time Visual Understanding_sample_{X}_{Y}``
    where X = sample (video) number, Y = question index within that sample.

    Since the CSV has no explicit video column the video filename is derived
    from the sample number via *video_format* (default ``{sample_id}.mp4``).
    """
    rows = []
    with open(input_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    with open(output_path, "w", encoding="utf-8") as out:
        for row in rows:
            qid = row["question_id"]
            # "Real-Time Visual Understanding_sample_1_1" → sample_id="1"
            parts = qid.rsplit("_", 2)
            sample_id = parts[-2]

            options = parse_options(row["options"])
            video_end = parse_timestamp(row["time_stamp"])

            formatted = {
                "id": qid,
                "task": row["task_type"],
                "video": video_format.format(sample_id=sample_id),
                "question": row["question"],
                "options": options,
                "answer": row["answer"],
                "video_start": 0,
                "video_end": video_end,
                "task_type": row["task_type"],
                "frames_required": row["frames_required"],
                "temporal_clue_type": row["temporal_clue_type"],
                "time_stamp": row["time_stamp"],
                "sample_id": sample_id,
            }
            out.write(json.dumps(formatted, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} entries → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Format Real-Time Visual Understanding CSV to JSONL."
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Path to input CSV file."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to save formatted JSONL file.",
    )
    parser.add_argument(
        "--video_format",
        type=str,
        default="sample_{sample_id}/video.mp4",
        help="Video filename template. Use {sample_id} as placeholder (default: '{sample_id}.mp4').",
    )
    args = parser.parse_args()
    transfer_rtvu(args.input, args.output, args.video_format)
