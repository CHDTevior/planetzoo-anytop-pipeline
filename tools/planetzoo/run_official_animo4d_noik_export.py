"""Run official AniMo4D no-IK BVH export across independent Blender processes."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", required=True, type=Path)
    parser.add_argument("--cobra-tools", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--tasks-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--objects-file", type=Path)
    parser.add_argument("--max-objects", type=int)
    parser.add_argument("--max-actions", type=int)
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--status-name", default="parallel_export_status.jsonl")
    parser.add_argument("--summary-name", default="parallel_export_summary.json")
    return parser.parse_args()


def read_object_keys(path: Path) -> list[str]:
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            keys.add(json.loads(line)["object_key"])
    return sorted(keys)


def complete_summary(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("complete"))
    except json.JSONDecodeError:
        return False


def run_one(object_key: str, args: argparse.Namespace, worker: Path) -> dict:
    started = time.time()
    summary_path = args.output_root / "object_summaries" / f"{object_key}.json"
    if args.skip_complete and complete_summary(summary_path):
        return {"object_key": object_key, "status": "skipped_complete", "seconds": 0.0}
    log = args.output_root / "logs" / f"{object_key}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.blender),
        "--background",
        "--python",
        str(worker),
        "--",
        "--cobra-tools",
        str(args.cobra_tools),
        "--source-root",
        str(args.source_root),
        "--tasks-manifest",
        str(args.tasks_manifest),
        "--output-root",
        str(args.output_root),
        "--object-key",
        object_key,
        "--fps",
        str(args.fps),
    ]
    if args.max_actions is not None:
        command.extend(("--max-actions", str(args.max_actions)))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    log.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8", errors="ignore")
    summary = None
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {
        "object_key": object_key,
        "status": "ok" if result.returncode == 0 and summary and summary.get("complete") else "incomplete_or_error",
        "returncode": result.returncode,
        "summary": summary,
        "log": str(log),
        "seconds": round(time.time() - started, 3),
    }


def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.fps <= 0:
        raise ValueError("--workers and --fps must be positive")
    worker = Path(__file__).with_name("export_official_animo4d_noik_bvhs.py")
    objects = read_object_keys(args.tasks_manifest)
    if args.objects_file:
        requested = {line.strip() for line in args.objects_file.read_text(encoding="utf-8").splitlines() if line.strip()}
        objects = [object_key for object_key in objects if object_key in requested]
    if args.max_objects is not None:
        objects = objects[: args.max_objects]
    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / args.status_name
    records = []
    with status_path.open("w", encoding="utf-8", newline="\n") as output:
        with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            pending = [executor.submit(run_one, object_key, args, worker) for object_key in objects]
            for future in futures.as_completed(pending):
                record = future.result()
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                records.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
    summary = {
        "requested_objects": len(objects),
        "completed_objects": sum(record["status"] in {"ok", "skipped_complete"} for record in records),
        "incomplete_or_error": sum(record["status"] == "incomplete_or_error" for record in records),
        "workers": args.workers,
        "fps": args.fps,
        "status_jsonl": str(status_path),
    }
    (args.output_root / args.summary_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
