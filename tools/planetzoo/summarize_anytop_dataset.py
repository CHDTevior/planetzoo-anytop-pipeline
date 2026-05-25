"""Summarize AnyTop processed Planet Zoo motion folders."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--length-bin", type=int, default=10)
    return parser.parse_args()


def iter_processed_dirs(root: Path) -> list[Path]:
    if (root / "motions").is_dir() and (root / "cond.npy").is_file():
        return [root]
    return sorted([p for p in root.iterdir() if p.is_dir() and (p / "motions").is_dir() and (p / "cond.npy").is_file()])


def load_cond(cond_path: Path) -> dict[str, Any]:
    return np.load(cond_path, allow_pickle=True).item()


def bin_length(length: int, bin_size: int) -> str:
    start = (length // bin_size) * bin_size
    end = start + bin_size - 1
    return f"{start}-{end}"


def main() -> None:
    args = parse_args()
    root = Path(args.processed_root)
    object_rows: list[dict[str, Any]] = []
    clip_rows: list[dict[str, Any]] = []
    node_counter: Counter[int] = Counter()
    length_counter: Counter[int] = Counter()
    length_bin_counter: Counter[str] = Counter()
    clips_per_object_counter: Counter[int] = Counter()

    for obj_dir in iter_processed_dirs(root):
        cond = load_cond(obj_dir / "cond.npy")
        if len(cond) != 1:
            object_name = obj_dir.name
            cond_entry = next(iter(cond.values()))
        else:
            object_name, cond_entry = next(iter(cond.items()))

        joint_names = cond_entry["joints_names"]
        node_count = len(joint_names)
        motions = sorted((obj_dir / "motions").glob("*.npy"))
        clip_lengths = []
        feature_dims = []
        finite = True

        for motion_path in motions:
            arr = np.load(motion_path)
            clip_lengths.append(int(arr.shape[0]))
            feature_dims.append(int(arr.shape[-1]))
            finite = finite and bool(np.isfinite(arr).all())
            clip_rows.append(
                {
                    "object_name": object_name,
                    "motion_file": motion_path.name,
                    "nodes": int(arr.shape[1]),
                    "length": int(arr.shape[0]),
                    "feature_dim": int(arr.shape[-1]),
                    "finite": bool(np.isfinite(arr).all()),
                    "foot_contact_sum": float(arr[..., 12].sum()) if arr.shape[-1] > 12 else None,
                }
            )
            length_counter[int(arr.shape[0])] += 1
            length_bin_counter[bin_length(int(arr.shape[0]), args.length_bin)] += 1

        node_counter[node_count] += 1
        clips_per_object_counter[len(motions)] += 1
        object_rows.append(
            {
                "object_name": object_name,
                "processed_dir": str(obj_dir),
                "nodes": node_count,
                "clips": len(motions),
                "length_min": min(clip_lengths) if clip_lengths else None,
                "length_max": max(clip_lengths) if clip_lengths else None,
                "feature_dim_values": sorted(set(feature_dims)),
                "finite": finite,
            }
        )

    all_lengths = [row["length"] for row in clip_rows]
    all_nodes = [row["nodes"] for row in object_rows]
    all_clips = [row["clips"] for row in object_rows]
    summary = {
        "processed_root": str(root),
        "total_objects": len(object_rows),
        "total_clips": len(clip_rows),
        "node_count_range": [min(all_nodes), max(all_nodes)] if all_nodes else [None, None],
        "clip_length_range": [min(all_lengths), max(all_lengths)] if all_lengths else [None, None],
        "clips_per_object_range": [min(all_clips), max(all_clips)] if all_clips else [None, None],
        "feature_dim_values": sorted({row["feature_dim"] for row in clip_rows}),
        "node_count_distribution": [
            {"nodes": nodes, "objects": count} for nodes, count in sorted(node_counter.items())
        ],
        "clip_length_distribution": [
            {"length": length, "clips": count} for length, count in sorted(length_counter.items())
        ],
        "clip_length_bins": [
            {"bin": key, "clips": count}
            for key, count in sorted(length_bin_counter.items(), key=lambda item: int(item[0].split("-", 1)[0]))
        ],
        "clips_per_object_distribution": [
            {"clips": clips, "objects": count} for clips, count in sorted(clips_per_object_counter.items())
        ],
        "objects": object_rows,
        "clips": clip_rows,
    }

    output_json = Path(args.output_json) if args.output_json else root / "dataset_summary.json"
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.output_csv:
        with Path(args.output_csv).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "object_name",
                    "processed_dir",
                    "nodes",
                    "clips",
                    "length_min",
                    "length_max",
                    "feature_dim_values",
                    "finite",
                ],
            )
            writer.writeheader()
            writer.writerows(object_rows)

    print(json.dumps({k: summary[k] for k in summary if k not in {"objects", "clips"}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
