"""Parallel AnyTop conversion for raw Planet Zoo BVH object folders."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--face-joints-names", nargs="+", default=["def_c_hips_joint", "def_c_chest_joint"])
    parser.add_argument("--objects", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--skip-animations", action="store_true", help="Skip AnyTop sanity-check MP4 rendering.")
    return parser.parse_args()


def safe_name(value: str) -> str:
    value = value.replace("@", "_").replace(".", "_")
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unnamed"


def object_name_from_raw_dir(raw_object_dir: Path) -> str:
    stem = raw_object_dir.name
    if stem.endswith("_ovl"):
        stem = stem[:-4]
    return "PZ_" + safe_name(stem)


def find_raw_dirs(raw_root: Path, requested: list[str] | None) -> list[Path]:
    if requested:
        return [raw_root / name for name in requested]
    return sorted([p for p in raw_root.iterdir() if p.is_dir() and (p / "raw_bvhs").is_dir()])


def find_tpos(raw_bvhs: Path) -> Path | None:
    candidates = sorted(raw_bvhs.glob("*__tpos.bvh"))
    if not candidates:
        candidates = sorted(raw_bvhs.glob("*tpos*.bvh"))
    return candidates[0] if candidates else None


def motion_bvh_count(raw_bvhs: Path) -> int:
    if not raw_bvhs.is_dir():
        return 0
    return len([p for p in raw_bvhs.glob("*.bvh") if "__tpos" not in p.stem.lower()])


def processed_motion_count(out_dir: Path) -> int:
    motions = out_dir / "motions"
    if not motions.is_dir():
        return 0
    return len(list(motions.glob("*.npy")))


def process_one(raw_object_dir: Path, args: argparse.Namespace, logs_dir: Path, status_dir: Path) -> dict:
    started = time.time()
    object_name = object_name_from_raw_dir(raw_object_dir)
    raw_bvhs = raw_object_dir / "raw_bvhs"
    out_dir = Path(args.output_root) / object_name
    log_path = logs_dir / f"{object_name}.log"
    status_path = status_dir / f"{object_name}.json"

    tpos = find_tpos(raw_bvhs)
    raw_motion_count = motion_bvh_count(raw_bvhs)
    if tpos is None or raw_motion_count == 0:
        record = {
            "raw_object_dir": str(raw_object_dir),
            "object_name": object_name,
            "status": "skipped_no_motion_bvh",
            "raw_motion_bvh_count": raw_motion_count,
            "processed_motion_count": processed_motion_count(out_dir),
            "returncode": 0,
            "seconds": 0.0,
            "log": str(log_path),
            "output_dir": str(out_dir),
        }
        status_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return record

    if args.skip_complete and (out_dir / "cond.npy").is_file() and processed_motion_count(out_dir) >= raw_motion_count and not args.overwrite:
        record = {
            "raw_object_dir": str(raw_object_dir),
            "object_name": object_name,
            "status": "skipped_complete",
            "raw_motion_bvh_count": raw_motion_count,
            "processed_motion_count": processed_motion_count(out_dir),
            "returncode": 0,
            "seconds": 0.0,
            "log": str(log_path),
            "output_dir": str(out_dir),
        }
        status_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return record

    if args.overwrite and out_dir.exists():
        resolved = out_dir.resolve()
        root = Path(args.output_root).resolve()
        if root not in resolved.parents and resolved != root:
            raise RuntimeError(f"Refusing to remove unexpected path: {resolved}")
        shutil.rmtree(out_dir)

    cmd = [
        args.python,
        "-m",
        "utils.process_new_skeleton",
        "--object_name",
        object_name,
        "--bvh_dir",
        str(raw_bvhs),
        "--save_dir",
        str(out_dir),
        "--face_joints_names",
        *args.face_joints_names,
        "--tpos_bvh",
        str(tpos),
    ]
    env = os.environ.copy()
    if args.skip_animations:
        env["ANYTOP_SKIP_ANIMATIONS"] = "1"
    with log_path.open("w", encoding="utf-8", errors="ignore") as log_f:
        result = subprocess.run(cmd, cwd=args.repo_root, stdout=log_f, stderr=subprocess.STDOUT, text=True, env=env)

    record = {
        "raw_object_dir": str(raw_object_dir),
        "object_name": object_name,
        "status": "ok" if result.returncode == 0 else "error",
        "raw_motion_bvh_count": raw_motion_count,
        "processed_motion_count": processed_motion_count(out_dir),
        "returncode": result.returncode,
        "seconds": round(time.time() - started, 3),
        "log": str(log_path),
        "output_dir": str(out_dir),
    }
    status_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = output_root / "logs_anytop"
    status_dir = output_root / "status_anytop"
    logs_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    status_jsonl = output_root / "parallel_anytop_process_status.jsonl"
    raw_dirs = find_raw_dirs(Path(args.raw_root), args.objects)

    started = time.time()
    completed = 0
    with status_jsonl.open("a", encoding="utf-8") as status_f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_one, raw_object_dir, args, logs_dir, status_dir): raw_object_dir
                for raw_object_dir in raw_dirs
            }
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                raw_object_dir = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = {
                        "raw_object_dir": str(raw_object_dir),
                        "object_name": object_name_from_raw_dir(raw_object_dir),
                        "status": "exception",
                        "error": repr(exc),
                        "returncode": None,
                        "seconds": None,
                    }
                record["completed"] = completed
                record["total"] = len(raw_dirs)
                status_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                status_f.flush()
                print(json.dumps(record, ensure_ascii=False), flush=True)

    summary = {
        "objects_requested": len(raw_dirs),
        "workers": args.workers,
        "seconds": round(time.time() - started, 3),
        "status_jsonl": str(status_jsonl),
    }
    (output_root / "parallel_anytop_process_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
