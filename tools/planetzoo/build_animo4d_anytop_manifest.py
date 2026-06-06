"""Build an AniMo4D-aligned manifest for Planet Zoo AnyTop outputs.

The source of truth is the AniMo4D text directory. Each text file names one
official AniMo4D motion and contains lines in the format:

    species#gender#caption#tokens#f_tag#to_tag

This script maps those official motion ids to raw full-topology BVHs and
processed AnyTop tensors. It intentionally writes one row per official text
file so sample counts can be audited against AniMo4D.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


TEXT_SUFFIX = "_keypoints.json.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-root", required=True, help="AniMo4D text directory.")
    parser.add_argument("--raw-root", default=None, help="Optional raw BVH root with *_ovl/raw_bvhs folders.")
    parser.add_argument("--processed-root", default=None, help="Optional per-object AnyTop output root.")
    parser.add_argument("--output-jsonl", required=True, help="Manifest JSONL output path.")
    parser.add_argument("--summary-json", required=True, help="Summary JSON output path.")
    parser.add_argument("--csv-output", default=None, help="Optional CSV output path.")
    return parser.parse_args()


def expected_raw_stem_from_text_name(name: str) -> str | None:
    if not name.endswith(TEXT_SUFFIX):
        return None
    base = name[: -len(TEXT_SUFFIX)]
    match = re.match(r"^(.+?)__(animation(?:not)?motionextracted[^.]+)\.([A-Za-z0-9]+)_(.+)$", base)
    if not match:
        return None
    animal, anim_group, maniset, action = match.groups()
    return f"{animal}__{anim_group}_{maniset}__{action}"


def object_key_from_raw_stem(raw_stem: str) -> str:
    animal = raw_stem.split("__", 1)[0]
    return "_".join(part.capitalize() for part in animal.split("_"))


def object_name_from_raw_stem(raw_stem: str) -> str:
    return "PZ_" + object_key_from_raw_stem(raw_stem)


def parse_text_lines(path: Path) -> tuple[list[str], list[dict[str, Any]], Counter]:
    captions: list[str] = []
    entries: list[dict[str, Any]] = []
    issues: Counter = Counter()
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split("#")
        if len(parts) < 3:
            issues["bad_text_line"] += 1
            entries.append({"line_no": line_no, "raw": line, "parse_status": "bad_text_line"})
            continue
        species = parts[0].strip()
        gender = parts[1].strip()
        caption = parts[2].strip()
        tokens = parts[3].strip() if len(parts) > 3 else ""
        f_tag = parts[4].strip() if len(parts) > 4 else "0.0"
        to_tag = parts[5].strip() if len(parts) > 5 else "0.0"
        if caption:
            captions.append(caption)
        entries.append(
            {
                "line_no": line_no,
                "species": species,
                "gender": gender,
                "caption": caption,
                "tokens": tokens,
                "f_tag": f_tag,
                "to_tag": to_tag,
                "parse_status": "ok",
            }
        )
    if not captions:
        issues["empty_captions"] += 1
    return captions, entries, issues


def build_raw_index(raw_root: Path | None) -> dict[str, Path]:
    if raw_root is None:
        return {}
    return {path.stem: path for path in raw_root.rglob("*.bvh") if path.is_file()}


def build_processed_index(processed_root: Path | None) -> dict[str, list[dict[str, Path]]]:
    if processed_root is None:
        return {}
    rows: dict[str, list[dict[str, Path]]] = {}
    for motion_path in processed_root.rglob("motions/*.npy"):
        object_name = motion_path.parent.parent.name
        stem = motion_path.stem
        prefix = object_name + "_"
        if not stem.startswith(prefix):
            continue
        tail = stem[len(prefix) :]
        raw_stem = re.sub(r"_[0-9]+$", "", tail)
        bvh_path = motion_path.parent.parent / "bvhs" / f"{stem}.bvh"
        animation_path = motion_path.parent.parent / "animations" / f"{stem}_from_ric.mp4"
        rows.setdefault(raw_stem, []).append(
            {
                "motion_path": motion_path,
                "bvh_path": bvh_path,
                "animation_path": animation_path,
            }
        )
    return rows


def path_str(path: Path | None) -> str:
    return str(path.resolve()) if path is not None else ""


def main() -> None:
    args = parse_args()
    text_root = Path(args.text_root)
    raw_root = Path(args.raw_root) if args.raw_root else None
    processed_root = Path(args.processed_root) if args.processed_root else None
    raw_index = build_raw_index(raw_root)
    processed_index = build_processed_index(processed_root)

    rows: list[dict[str, Any]] = []
    status_counts: Counter = Counter()
    caption_counts: Counter = Counter()
    object_counts: Counter = Counter()
    text_issues: Counter = Counter()
    parse_bad: list[str] = []

    for text_path in sorted(text_root.glob("*.txt")):
        raw_stem = expected_raw_stem_from_text_name(text_path.name)
        if raw_stem is None:
            parse_bad.append(text_path.name)
            status_counts["bad_text_filename"] += 1
            continue
        object_key = object_key_from_raw_stem(raw_stem)
        object_name = object_name_from_raw_stem(raw_stem)
        captions, text_entries, issues = parse_text_lines(text_path)
        text_issues.update(issues)
        processed_matches = sorted(processed_index.get(raw_stem, []), key=lambda item: item["motion_path"].name)
        raw_path = raw_index.get(raw_stem)

        if processed_root is None:
            status = "not_checked_processed"
        elif len(processed_matches) == 1:
            status = "matched"
        elif len(processed_matches) == 0:
            status = "missing_processed"
        else:
            status = "duplicate_processed"
        if raw_root is not None and raw_path is None:
            status = "missing_raw" if status in {"matched", "not_checked_processed"} else f"{status}+missing_raw"

        match = processed_matches[0] if len(processed_matches) == 1 else None
        row = {
            "id": raw_stem,
            "object_key": object_key,
            "object_name": object_name,
            "raw_bvh_stem": raw_stem,
            "text_file": str(text_path.resolve()),
            "text_file_name": text_path.name,
            "texts": captions,
            "text_entries": text_entries,
            "caption_count": len(captions),
            "raw_bvh": path_str(raw_path),
            "processed_motion": path_str(match["motion_path"]) if match else "",
            "processed_bvh": path_str(match["bvh_path"]) if match else "",
            "processed_animation": path_str(match["animation_path"]) if match else "",
            "processed_match_count": len(processed_matches),
            "status": status,
        }
        rows.append(row)
        status_counts[status] += 1
        caption_counts[len(captions)] += 1
        object_counts[object_name] += 1

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "rows": len(rows),
        "text_root": str(text_root.resolve()),
        "raw_root": str(raw_root.resolve()) if raw_root else "",
        "processed_root": str(processed_root.resolve()) if processed_root else "",
        "status_counts": dict(status_counts),
        "caption_count_distribution": dict(sorted(caption_counts.items())),
        "object_count": len(object_counts),
        "object_counts_top20": object_counts.most_common(20),
        "text_issues": dict(text_issues),
        "bad_text_filenames": parse_bad[:20],
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.csv_output:
        csv_path = Path(args.csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "id",
            "object_key",
            "object_name",
            "caption_count",
            "status",
            "raw_bvh",
            "processed_motion",
            "processed_bvh",
            "text_file",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
