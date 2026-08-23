"""Validate exact official AniMo4D caption pairing in a converted corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--unavailable-manifest", required=True, type=Path)
    parser.add_argument("--output-report", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    source_rows = read_jsonl(args.source_manifest)
    source_by_id = {row["official_id"]: row for row in source_rows}
    if len(source_by_id) != len(source_rows):
        raise ValueError("Official source manifest has duplicate IDs")
    unavailable_rows = read_jsonl(args.unavailable_manifest)
    clips = read_jsonl(args.dataset_root / "manifests" / "clips.jsonl")
    clip_by_id: dict[str, dict] = {}
    duplicates: list[str] = []
    errors: list[dict] = []
    for clip in clips:
        official_id = clip.get("official_id")
        if not official_id:
            errors.append({"clip_id": clip.get("clip_id"), "error": "missing_official_id"})
            continue
        if official_id in clip_by_id:
            duplicates.append(official_id)
            continue
        clip_by_id[official_id] = clip
        source = source_by_id.get(official_id)
        if source is None:
            errors.append({"official_id": official_id, "error": "not_in_official_source_manifest"})
            continue
        caption = clip.get("caption") or {}
        if caption.get("annotation_source") != "animo4d_official":
            errors.append({"official_id": official_id, "error": "wrong_annotation_source"})
        if caption.get("text_status") != "present":
            errors.append({"official_id": official_id, "error": "missing_present_text_status"})
        if caption.get("texts") != source.get("texts"):
            errors.append({"official_id": official_id, "error": "caption_texts_mismatch"})
        if caption.get("text_entries") != source.get("text_entries"):
            errors.append({"official_id": official_id, "error": "caption_entries_mismatch"})
    expected_ids = set(source_by_id)
    actual_ids = set(clip_by_id)
    report = {
        "dataset_root": str(args.dataset_root),
        "source_official_actions": len(source_rows),
        "source_unique_official_ids": len(source_by_id),
        "dataset_clips": len(clips),
        "dataset_unique_official_ids": len(actual_ids),
        "unavailable_official_text_rows": len(unavailable_rows),
        "missing_official_ids": sorted(expected_ids - actual_ids)[:100],
        "unexpected_official_ids": sorted(actual_ids - expected_ids)[:100],
        "duplicate_official_ids": sorted(set(duplicates))[:100],
        "error_count": len(errors),
        "errors": errors[:100],
    }
    report["complete"] = not (
        report["missing_official_ids"]
        or report["unexpected_official_ids"]
        or report["duplicate_official_ids"]
        or report["error_count"]
        or len(actual_ids) != len(expected_ids)
    )
    output = args.output_report or args.dataset_root / "reports" / "official_text_pairing_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
