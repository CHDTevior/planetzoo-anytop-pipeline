"""Assemble and validate all per-object official no-IK export manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-manifest", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    tasks = read_jsonl(args.tasks_manifest)
    expected = {task["official_id"]: task for task in tasks}
    object_keys = sorted({task["object_key"] for task in tasks})
    records: list[dict] = []
    errors: list[dict] = []
    seen: set[str] = set()
    tposes: dict[str, dict] = {}
    for object_key in object_keys:
        manifest = args.raw_root / "object_manifests" / f"{object_key}.jsonl"
        if not manifest.is_file():
            errors.append({"object_key": object_key, "error": "missing_object_manifest"})
            continue
        for record in read_jsonl(manifest):
            if record.get("sample_type") == "tpose":
                if record.get("status") == "ok" and Path(record["raw_bvh"]).is_file():
                    tposes[object_key] = record
                else:
                    errors.append({"object_key": object_key, "error": "invalid_tpose", "record": record})
                continue
            official_id = record.get("official_id")
            if official_id not in expected:
                errors.append({"object_key": object_key, "error": "unknown_official_id", "record": record})
                continue
            if official_id in seen:
                errors.append({"object_key": object_key, "error": "duplicate_official_id", "official_id": official_id})
                continue
            seen.add(official_id)
            if record.get("status") not in {"ok", "skipped_existing"}:
                errors.append({"object_key": object_key, "error": "export_error", "record": record})
                continue
            if not Path(record["raw_bvh"]).is_file():
                errors.append({"object_key": object_key, "error": "missing_bvh", "record": record})
                continue
            records.append(record)
    missing = sorted(set(expected) - seen)
    for object_key in object_keys:
        if object_key not in tposes:
            errors.append({"object_key": object_key, "error": "missing_tpose"})
    complete = not errors and not missing and len(records) == len(expected) and len(tposes) == len(object_keys)
    rows = [tposes[key] for key in sorted(tposes)] + sorted(records, key=lambda row: row["official_id"])
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    summary = {
        "expected_official_actions": len(expected),
        "exported_official_actions": len(records),
        "missing_official_actions": len(missing),
        "tposes": len(tposes),
        "expected_tposes": len(object_keys),
        "errors": len(errors),
        "complete": complete,
        "source_resolution_counts": dict(Counter(task["source_resolution"] for task in tasks)),
        "output_manifest": str(args.output_manifest),
        "missing_ids_preview": missing[:20],
        "errors_preview": errors[:20],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
