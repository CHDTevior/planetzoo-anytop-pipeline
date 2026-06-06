"""Batch-process Planet Zoo raw BVH exports into AnyTop feature folders."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loaders.truebones.truebones_utils.motion_process import process_skeleton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, help="Directory containing *_ovl/raw_bvhs folders.")
    parser.add_argument("--output-root", required=True, help="Directory where processed AnyTop object folders are written.")
    parser.add_argument(
        "--face-joints-names",
        nargs="+",
        default=["def_c_hips_joint", "def_c_chest_joint"],
        help="Face joints passed to AnyTop. Two joints are treated as a centerline.",
    )
    parser.add_argument("--objects", nargs="*", default=None, help="Optional raw object directory names to process.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of objects to process.")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess objects even when cond.npy already exists.")
    parser.add_argument(
        "--max-clip-frames",
        type=int,
        default=240,
        help="Split BVHs longer than this many source frames. Set <=0 to disable splitting.",
    )
    parser.add_argument(
        "--clip-step-frames",
        type=int,
        default=200,
        help="Step size used when max-clip-frames triggers splitting.",
    )
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


def find_raw_object_dirs(raw_root: Path, requested: list[str] | None) -> list[Path]:
    if requested:
        return [raw_root / name for name in requested]
    dirs = []
    for child in sorted(raw_root.iterdir()):
        if child.is_dir() and (child / "raw_bvhs").is_dir():
            dirs.append(child)
    return dirs


def count_motion_bvhs(raw_bvhs: Path) -> int:
    return len([p for p in raw_bvhs.glob("*.bvh") if "__tpos" not in p.stem.lower()])


def find_tpos(raw_bvhs: Path) -> Path | None:
    candidates = sorted(raw_bvhs.glob("*__tpos.bvh"))
    if not candidates:
        candidates = sorted(raw_bvhs.glob("*tpos*.bvh"))
    return candidates[0] if candidates else None


def count_processed_motions(out_dir: Path) -> int:
    motions = out_dir / "motions"
    if not motions.exists():
        return 0
    return len(list(motions.glob("*.npy")))


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    batch_manifest = output_root / "batch_process_manifest.jsonl"

    raw_dirs = find_raw_object_dirs(raw_root, args.objects)
    if args.limit is not None:
        raw_dirs = raw_dirs[: args.limit]

    with batch_manifest.open("a", encoding="utf-8") as manifest_f:
        for raw_object_dir in raw_dirs:
            object_name = object_name_from_raw_dir(raw_object_dir)
            out_dir = output_root / object_name
            raw_bvhs = raw_object_dir / "raw_bvhs"
            record = {
                "raw_object_dir": str(raw_object_dir),
                "object_name": object_name,
                "output_dir": str(out_dir),
                "face_joints_names": args.face_joints_names,
            }
            try:
                if not raw_bvhs.is_dir():
                    raise FileNotFoundError(f"missing raw_bvhs directory: {raw_bvhs}")
                tpos = find_tpos(raw_bvhs)
                if tpos is None:
                    raise FileNotFoundError(f"missing T-pose BVH in {raw_bvhs}")
                motion_bvh_count = count_motion_bvhs(raw_bvhs)
                if motion_bvh_count == 0:
                    raise RuntimeError(f"no motion BVHs in {raw_bvhs}")

                record["tpos_bvh"] = str(tpos)
                record["raw_motion_bvh_count"] = motion_bvh_count

                if out_dir.exists() and args.overwrite:
                    shutil.rmtree(out_dir)
                if (out_dir / "cond.npy").exists() and not args.overwrite:
                    record["status"] = "skipped_existing"
                else:
                    process_skeleton(
                        object_name,
                        str(raw_bvhs),
                        args.face_joints_names,
                        str(out_dir),
                        str(tpos),
                        max_clip_frames=args.max_clip_frames,
                        clip_step_frames=args.clip_step_frames,
                    )
                    record["status"] = "ok"
                record["processed_motion_count"] = count_processed_motions(out_dir)
            except Exception as exc:
                record["status"] = "error"
                record["error"] = repr(exc)
                record["traceback"] = traceback.format_exc()
                print(f"ERROR {object_name}: {exc}")

            manifest_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest_f.flush()
            print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
