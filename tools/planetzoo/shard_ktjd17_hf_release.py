"""Shard a KTJD-17 release's motion files by rig for Hugging Face limits.

Hugging Face Git repositories allow at most 10,000 files directly below one
directory. The working corpus intentionally keeps a flat motions directory;
this release-only step rewrites it as motions/<rig_id>/<clip_id>.npz and
updates manifest paths without changing any payload values.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path, PurePosixPath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--apply", action="store_true", help="Perform the reshaping; omit for a dry run.")
    return parser.parse_args()


def target_motion_file(record: dict) -> str:
    current = PurePosixPath(record["motion_file"])
    if current.parts[0] != "motions":
        raise ValueError(f"Not a motion path: {current}")
    return (PurePosixPath("motions") / record["rig_id"] / current.name).as_posix()


def main() -> None:
    args = parse_args()
    dataset_root = args.release_root / "data"
    manifest_path = dataset_root / "manifests" / "clips.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]

    destinations: set[str] = set()
    moves: list[tuple[Path, Path, dict]] = []
    per_rig = Counter()
    for record in records:
        target_relative = target_motion_file(record)
        if target_relative in destinations:
            raise ValueError(f"Duplicate release motion path: {target_relative}")
        destinations.add(target_relative)
        source = dataset_root / record["motion_file"]
        target = dataset_root / target_relative
        if source != target and not source.is_file() and not target.is_file():
            raise FileNotFoundError(f"Neither source nor target exists for {record['clip_id']}")
        moves.append((source, target, record))
        per_rig[record["rig_id"]] += 1

    report = {
        "clip_count": len(records),
        "rig_directory_count": len(per_rig),
        "max_files_in_rig_directory": max(per_rig.values()),
        "already_sharded": sum(source == target or target.is_file() for source, target, _ in moves),
        "pending_moves": sum(source != target and source.is_file() for source, target, _ in moves),
    }
    if not args.apply:
        print(json.dumps(report, indent=2))
        return

    for source, target, _ in moves:
        if source == target or target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)

    lines = []
    for _, _, record in moves:
        record["motion_file"] = target_motion_file(record)
        lines.append(json.dumps(record, ensure_ascii=False))
    temporary_manifest = manifest_path.with_suffix(".jsonl.tmp")
    temporary_manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)

    motion_count = sum(1 for _ in (dataset_root / "motions").rglob("*.npz"))
    if motion_count != len(records):
        raise RuntimeError(f"Expected {len(records)} motion files after sharding, found {motion_count}")
    (dataset_root / "reports" / "hf_motion_layout.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
