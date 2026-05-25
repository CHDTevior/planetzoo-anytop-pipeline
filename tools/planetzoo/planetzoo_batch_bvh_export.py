"""Run the Planet Zoo BVH exporter with one Blender process per object."""

from __future__ import annotations

import argparse
import json
import re
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
    parser.add_argument("--continue-on-error", action="store_true")
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


def rebuild_root_manifest(output_root: Path) -> int:
    manifests = sorted(output_root.glob("*_ovl/export_manifest.jsonl"))
    count = 0
    with (output_root / "export_manifest.jsonl").open("w", encoding="utf-8") as out_f:
        for manifest in manifests:
            text = manifest.read_text(encoding="utf-8")
            if text and not text.endswith("\n"):
                text += "\n"
            out_f.write(text)
            count += len([line for line in text.splitlines() if line.strip()])
    return count


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "batch_bvh_export_status.jsonl"

    objects = find_objects(input_root, args.objects)
    objects = objects[args.start_index :]
    if args.max_objects is not None:
        objects = objects[: args.max_objects]

    exporter = Path(__file__).resolve().with_name("planetzoo_fulltopo_bvh_export.py")
    with status_path.open("a", encoding="utf-8") as status_f:
        for object_name in objects:
            started = time.time()
            log_path = logs_dir / f"{safe_name(object_name)}.log"
            cmd = [
                args.blender,
                "-b",
                "--python",
                str(exporter),
                "--",
                "--cobra-tools",
                args.cobra_tools,
                "--input-root",
                str(input_root),
                "--output-root",
                str(output_root),
                "--objects",
                object_name,
            ]
            if args.max_actions is not None:
                cmd.extend(["--max-actions", str(args.max_actions)])
            if args.only_manis_contains:
                cmd.extend(["--only-manis-contains", args.only_manis_contains])

            print(f"EXPORT {object_name}")
            with log_path.open("w", encoding="utf-8", errors="ignore") as log_f:
                result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True)

            object_out = output_root / safe_name(object_name)
            raw_bvhs = object_out / "raw_bvhs"
            bvh_count = len(list(raw_bvhs.glob("*.bvh"))) if raw_bvhs.is_dir() else 0
            motion_bvh_count = len([p for p in raw_bvhs.glob("*.bvh") if "__tpos" not in p.stem.lower()]) if raw_bvhs.is_dir() else 0
            record = {
                "object": object_name,
                "status": "ok" if result.returncode == 0 else "error",
                "returncode": result.returncode,
                "seconds": round(time.time() - started, 3),
                "log": str(log_path),
                "output_dir": str(object_out),
                "bvh_count": bvh_count,
                "motion_bvh_count": motion_bvh_count,
            }
            status_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            status_f.flush()
            print(json.dumps(record, ensure_ascii=False))
            rebuild_root_manifest(output_root)
            if result.returncode != 0 and not args.continue_on_error:
                raise SystemExit(result.returncode)

    manifest_rows = rebuild_root_manifest(output_root)
    print(json.dumps({"root_manifest": str(output_root / "export_manifest.jsonl"), "rows": manifest_rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
