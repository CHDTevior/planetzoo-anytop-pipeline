"""Parallel Planet Zoo BVH export with one Blender subprocess per object."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import shutil
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender", required=True)
    parser.add_argument("--cobra-tools", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--objects", nargs="*", default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--only-manis-contains", default=None)
    parser.add_argument(
        "--target-text-root",
        default=None,
        help="Optional AniMo4D text directory. Passed to the Blender exporter to export only target-captioned actions.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-complete", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    value = value.replace("@", "_").replace(".", "_")
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unnamed"


def find_objects(input_root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    return [p.name for p in sorted(input_root.iterdir()) if p.is_dir()]


def expected_raw_stem_from_text_name(name: str) -> str | None:
    if not name.endswith("_keypoints.json.txt"):
        return None
    base = name[: -len("_keypoints.json.txt")]
    match = re.match(r"^(.+?)__(animation(?:not)?motionextracted[^.]+)\.([A-Za-z0-9]+)_(.+)$", base)
    if not match:
        return None
    animal, anim_group, maniset, action = match.groups()
    return f"{animal}__{anim_group}_{maniset}__{action}"


def object_dir_name_from_raw_stem(raw_stem: str) -> str:
    animal = raw_stem.split("__", 1)[0]
    object_key = "_".join(part.capitalize() for part in animal.split("_"))
    return f"{object_key}_ovl"


def target_object_names(text_root: str | None) -> set[str] | None:
    if not text_root:
        return None
    root = Path(text_root)
    files = [root] if root.is_file() else sorted(root.glob("*.txt"))
    objects: set[str] = set()
    for path in files:
        raw_stem = expected_raw_stem_from_text_name(path.name)
        if raw_stem:
            objects.add(object_dir_name_from_raw_stem(raw_stem))
    return objects


def raw_counts(object_out: Path) -> tuple[int, int]:
    raw_bvhs = object_out / "raw_bvhs"
    if not raw_bvhs.is_dir():
        return 0, 0
    bvhs = list(raw_bvhs.glob("*.bvh"))
    motions = [p for p in bvhs if "__tpos" not in p.stem.lower()]
    return len(bvhs), len(motions)


def is_complete(object_out: Path) -> bool:
    bvh_count, motion_count = raw_counts(object_out)
    return bvh_count >= 2 and motion_count >= 1 and (object_out / "export_manifest.jsonl").is_file()


def rebuild_root_manifest(output_root: Path) -> tuple[int, int]:
    manifests = sorted(output_root.glob("*_ovl/export_manifest.jsonl"))
    rows = 0
    with (output_root / "export_manifest.jsonl").open("w", encoding="utf-8") as out_f:
        for manifest in manifests:
            text = manifest.read_text(encoding="utf-8")
            if text and not text.endswith("\n"):
                text += "\n"
            out_f.write(text)
            rows += len([line for line in text.splitlines() if line.strip()])
    summary = {}
    for manifest in manifests:
        object_dir = manifest.parent.name
        object_rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        summary[object_dir] = sum(1 for row in object_rows if row.get("sample_type") == "motion")
    (output_root / "export_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(manifests), rows


def export_one(
    object_name: str,
    args: argparse.Namespace,
    exporter: Path,
    logs_dir: Path,
    status_dir: Path,
) -> dict:
    started = time.time()
    output_root = Path(args.output_root)
    object_out = output_root / safe_name(object_name)
    log_path = logs_dir / f"{safe_name(object_name)}.log"
    status_path = status_dir / f"{safe_name(object_name)}.json"

    if args.skip_complete and is_complete(object_out) and not args.overwrite:
        bvh_count, motion_count = raw_counts(object_out)
        record = {
            "object": object_name,
            "status": "skipped_complete",
            "returncode": 0,
            "seconds": 0.0,
            "log": str(log_path),
            "output_dir": str(object_out),
            "bvh_count": bvh_count,
            "motion_bvh_count": motion_count,
        }
        status_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return record

    if args.overwrite and object_out.exists():
        resolved = object_out.resolve()
        root = output_root.resolve()
        if root not in resolved.parents and resolved != root:
            raise RuntimeError(f"Refusing to remove unexpected path: {resolved}")
        shutil.rmtree(object_out)

    cmd = [
        args.blender,
        "-b",
        "--python",
        str(exporter),
        "--",
        "--cobra-tools",
        args.cobra_tools,
        "--input-root",
        args.input_root,
        "--output-root",
        args.output_root,
        "--objects",
        object_name,
        "--no-root-manifest",
    ]
    if args.max_actions is not None:
        cmd.extend(["--max-actions", str(args.max_actions)])
    if args.only_manis_contains:
        cmd.extend(["--only-manis-contains", args.only_manis_contains])
    if args.target_text_root:
        cmd.extend(["--target-text-root", args.target_text_root])

    with log_path.open("w", encoding="utf-8", errors="ignore") as log_f:
        result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True)

    bvh_count, motion_count = raw_counts(object_out)
    record = {
        "object": object_name,
        "status": "ok" if result.returncode == 0 else "error",
        "returncode": result.returncode,
        "seconds": round(time.time() - started, 3),
        "log": str(log_path),
        "output_dir": str(object_out),
        "bvh_count": bvh_count,
        "motion_bvh_count": motion_count,
    }
    status_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = output_root / "logs"
    status_dir = output_root / "status"
    logs_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    status_jsonl = output_root / "parallel_bvh_export_status.jsonl"
    exporter = Path(__file__).resolve().with_name("planetzoo_fulltopo_bvh_export.py")

    objects = find_objects(input_root, args.objects)
    targets = target_object_names(args.target_text_root)
    if targets is not None:
        before = len(objects)
        objects = [name for name in objects if safe_name(name) in targets or name in targets]
        print(
            json.dumps(
                {"target_text_root": args.target_text_root, "objects_before_filter": before, "objects_after_filter": len(objects)},
                ensure_ascii=False,
            ),
            flush=True,
        )
    objects = objects[args.start_index :]
    if args.max_objects is not None:
        objects = objects[: args.max_objects]

    started = time.time()
    completed = 0
    with status_jsonl.open("a", encoding="utf-8") as status_f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(export_one, object_name, args, exporter, logs_dir, status_dir): object_name
                for object_name in objects
            }
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                object_name = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = {
                        "object": object_name,
                        "status": "exception",
                        "error": repr(exc),
                        "returncode": None,
                        "seconds": None,
                        "log": str(logs_dir / f"{safe_name(object_name)}.log"),
                        "output_dir": str(output_root / safe_name(object_name)),
                        "bvh_count": 0,
                        "motion_bvh_count": 0,
                    }
                record["completed"] = completed
                record["total"] = len(objects)
                status_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                status_f.flush()
                print(json.dumps(record, ensure_ascii=False), flush=True)

    manifest_count, manifest_rows = rebuild_root_manifest(output_root)
    summary = {
        "objects_requested": len(objects),
        "workers": args.workers,
        "seconds": round(time.time() - started, 3),
        "object_manifests": manifest_count,
        "manifest_rows": manifest_rows,
        "root_manifest": str(output_root / "export_manifest.jsonl"),
    }
    (output_root / "parallel_bvh_export_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
