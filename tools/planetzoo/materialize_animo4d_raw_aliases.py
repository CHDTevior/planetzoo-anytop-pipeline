"""Materialize AniMo4D raw BVH aliases by matching object and action name.

Some Planet Zoo builds expose the same action under a different `.manis` group
than the AniMo4D text filename records. This script fills missing official raw
BVH stems by hardlinking/copying an existing BVH with the same object and action
suffix.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment-manifest", required=True, help="JSONL from build_animo4d_anytop_manifest.py.")
    parser.add_argument("--raw-root", required=True, help="Raw BVH root containing *_ovl/raw_bvhs.")
    parser.add_argument("--output-jsonl", required=True, help="Alias manifest output JSONL.")
    parser.add_argument("--summary-json", required=True, help="Alias summary output JSON.")
    parser.add_argument("--mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def split_raw_stem(stem: str) -> tuple[str, str, str]:
    animal, rest = stem.split("__", 1)
    group, action = rest.split("__", 1)
    return animal, group, action


def object_key_from_animal(animal: str) -> str:
    return "_".join(part.capitalize() for part in animal.split("_"))


def group_score(target_group: str, candidate_group: str) -> tuple[int, int, int, str]:
    if target_group == candidate_group:
        exact = 0
    else:
        exact = 1

    target_kind = target_group.split("_maniset", 1)[0]
    candidate_kind = candidate_group.split("_maniset", 1)[0]
    same_kind = 0 if target_kind == candidate_kind else 1

    def family(kind: str) -> str:
        if kind.startswith("animationnotmotionextracted"):
            return kind.replace("animationnotmotionextracted", "")
        if kind.startswith("animationmotionextracted"):
            return kind.replace("animationmotionextracted", "")
        return kind

    same_family = 0 if family(target_kind) == family(candidate_kind) else 1
    return (exact, same_kind, same_family, candidate_group)


def materialize(src: Path, dst: Path, mode: str, dry_run: bool) -> str:
    if dry_run:
        return "dry_run"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy_fallback"


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    raw_files = [p for p in raw_root.rglob("*.bvh") if "__tpos" not in p.stem.lower()]

    by_action: dict[tuple[str, str], list[tuple[str, Path]]] = defaultdict(list)
    by_stem: dict[str, Path] = {}
    for path in raw_files:
        try:
            animal, group, action = split_raw_stem(path.stem)
        except ValueError:
            continue
        by_action[(animal, action)].append((group, path))
        by_stem[path.stem] = path

    records: list[dict[str, Any]] = []
    counts: Counter = Counter()
    with Path(args.alignment_manifest).open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    for row in rows:
        if row.get("status") != "missing_raw":
            continue
        target_stem = row["raw_bvh_stem"]
        if target_stem in by_stem:
            counts["already_exists_after_manifest"] += 1
            continue
        animal, target_group, action = split_raw_stem(target_stem)
        candidates = by_action.get((animal, action), [])
        if not candidates:
            counts["no_action_alias"] += 1
            continue

        candidates_sorted = sorted(
            candidates,
            key=lambda item: group_score(target_group, item[0]),
        )
        chosen_group, src_path = candidates_sorted[0]
        object_dir = raw_root / f"{object_key_from_animal(animal)}_ovl" / "raw_bvhs"
        dst_path = object_dir / f"{target_stem}.bvh"
        method = materialize(src_path, dst_path, args.mode, args.dry_run)
        counts[method] += 1
        counts["aliased"] += 1
        records.append(
            {
                "target_raw_bvh_stem": target_stem,
                "target_raw_bvh": str(dst_path.resolve()),
                "source_raw_bvh_stem": src_path.stem,
                "source_raw_bvh": str(src_path.resolve()),
                "source_group": chosen_group,
                "candidate_count": len(candidates),
                "materialization": method,
            }
        )
        if not args.dry_run:
            by_stem[target_stem] = dst_path
            by_action[(animal, action)].append((target_group, dst_path))

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "alignment_manifest": str(Path(args.alignment_manifest).resolve()),
        "raw_root": str(raw_root.resolve()),
        "mode": args.mode,
        "dry_run": args.dry_run,
        "counts": dict(counts),
        "records": len(records),
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
