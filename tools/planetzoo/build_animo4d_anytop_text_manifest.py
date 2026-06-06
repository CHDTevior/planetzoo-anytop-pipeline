"""Build text manifests from an AniMo4D-aligned AnyTop alignment JSONL.

The alignment manifest is produced by ``build_animo4d_anytop_manifest.py`` and
contains one row per official AniMo4D text file. This helper filters rows with
processed AnyTop motions and writes the text manifest format used by the
Planet Zoo AnyTop packing scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment-manifest", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--by-file-output", default=None)
    parser.add_argument("--missing-output", default=None)
    return parser.parse_args()


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matched: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for row in rows:
        status = row.get("status", "")
        if status != "matched":
            missing.append(row)
            continue
        texts = row.get("texts") or []
        text = texts[0] if texts else ""
        processed_motion = row.get("processed_motion", "")
        processed_bvh = row.get("processed_bvh", "")
        processed_animation = row.get("processed_animation", "")
        item = {
            "id": row.get("id", ""),
            "object_key": row.get("object_key", ""),
            "object_name": row.get("object_name", ""),
            "raw_bvh_stem": row.get("raw_bvh_stem", ""),
            "raw_bvh": row.get("raw_bvh", ""),
            "processed_motion": processed_motion,
            "processed_bvh": processed_bvh,
            "processed_animation": processed_animation,
            "text": text,
            "texts": texts,
            "text_entries": row.get("text_entries", []),
            "caption_count": row.get("caption_count", len(texts)),
            "text_file": row.get("text_file", ""),
            "text_file_name": row.get("text_file_name", ""),
            "text_match_key": row.get("text_file_name", "") or row.get("id", ""),
            "text_match_status": "matched",
            "text_status": "present",
            "annotation_source": "animo4d_official_text",
            "needs_human_review": False,
            "motion_file": Path(processed_motion).name if processed_motion else "",
            "bvh_file": Path(processed_bvh).name if processed_bvh else "",
        }
        matched.append(item)
    return matched, missing


def write_json(path: Path, rows: list[dict[str, Any]], missing: list[dict[str, Any]]) -> None:
    status_counts = Counter(row["text_match_status"] for row in rows)
    object_counts = Counter(row["object_name"] for row in rows)
    payload = {
        "summary": {
            "rows": len(rows),
            "missing_rows": len(missing),
            "status_counts": dict(sorted(status_counts.items())),
            "object_counts": dict(sorted(object_counts.items())),
        },
        "rows": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = [
        "id",
        "object_name",
        "raw_bvh_stem",
        "motion_file",
        "bvh_file",
        "text",
        "caption_count",
        "text_match_status",
        "annotation_source",
        "text_file_name",
        "processed_motion",
        "processed_bvh",
        "raw_bvh",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_by_file(path: Path, rows: list[dict[str, Any]]) -> None:
    payload: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row["motion_file"] or f"{row['id']}.npy"
        payload[key] = {
            "captions": row["texts"],
            "text_status": row["text_status"],
            "annotation_source": row["annotation_source"],
            "needs_human_review": row["needs_human_review"],
            "source_file": row["text_file_name"],
            "object_name": row["object_name"],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_rows = iter_jsonl(Path(args.alignment_manifest))
    matched, missing = build_rows(source_rows)
    write_jsonl(Path(args.output_jsonl), matched)
    write_json(Path(args.output_json), matched, missing)
    write_csv(Path(args.output_csv), matched)
    if args.by_file_output:
        write_by_file(Path(args.by_file_output), matched)
    if args.missing_output:
        write_jsonl(Path(args.missing_output), missing)
    summary = {
        "source_rows": len(source_rows),
        "matched_rows": len(matched),
        "missing_rows": len(missing),
        "output_jsonl": str(Path(args.output_jsonl).resolve()),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
