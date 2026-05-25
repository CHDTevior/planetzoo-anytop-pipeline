"""Pack per-object Planet Zoo AnyTop outputs into AnyTop's pooled layout.

The Planet Zoo converter runs AnyTop's new-skeleton path once per skeleton, so
the immediate output is one folder per object. AnyTop's full Truebones dataset
layout pools all motions/BVHs under one root and stores all skeleton conditions
in a single cond.npy. This script performs that lossless packaging step.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np

DATA_SUBDIRS = ("motions", "bvhs", "animations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", required=True, help="Root containing per-object PZ_* AnyTop folders.")
    parser.add_argument("--output-root", required=True, help="Destination root with pooled motions/bvhs/cond.npy.")
    parser.add_argument(
        "--text-manifest",
        default=None,
        help="Optional per-object motion_text_manifest.jsonl to rewrite for the pooled layout.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("auto", "hardlink", "copy", "symlink"),
        default="auto",
        help="How to materialize pooled files. auto tries hardlink then copy.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output root.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect and report without writing files.")
    return parser.parse_args()


def is_object_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and path.name.startswith("PZ_")
        and (path / "cond.npy").is_file()
        and (path / "motions").is_dir()
        and (path / "bvhs").is_dir()
    )


def parse_metadata_frames(path: Path) -> int:
    if not path.is_file():
        return 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.match(r"total frames:\s*(\d+)", line.strip())
        if match:
            return int(match.group(1))
    return 0


def safe_prepare_output(output_root: Path, processed_root: Path, overwrite: bool, dry_run: bool) -> None:
    output_root = output_root.resolve()
    processed_root = processed_root.resolve()
    if output_root == processed_root:
        raise ValueError("output-root must differ from processed-root")
    if processed_root in output_root.parents:
        raise ValueError("output-root must not be nested inside processed-root")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"output root already exists: {output_root}")
        if dry_run:
            return
        shutil.rmtree(output_root)
    if dry_run:
        return
    for subdir in DATA_SUBDIRS:
        (output_root / subdir).mkdir(parents=True, exist_ok=True)


def materialize_file(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"destination already exists: {dst}")
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    if mode == "symlink":
        os.symlink(src, dst)
        return "symlink"
    if mode == "hardlink":
        os.link(src, dst)
        return "hardlink"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def rewrite_text_manifest(src_path: Path, output_root: Path) -> tuple[int, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    with src_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            motion_path = row.get("processed_motion") or row.get("motion_path")
            bvh_path = row.get("processed_bvh") or row.get("bvh_path")
            if motion_path:
                row["processed_motion"] = str(output_root / "motions" / Path(motion_path).name)
            if bvh_path:
                row["processed_bvh"] = str(output_root / "bvhs" / Path(bvh_path).name)
            status = row.get("text_match_status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            rows.append(row)

    jsonl_path = output_root / "motion_text_manifest.jsonl"
    write_jsonl(jsonl_path, rows)
    json_path = output_root / "motion_text_manifest.json"
    json_path.write_text(
        json.dumps({"rows": len(rows), "status_counts": status_counts, "items": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path = output_root / "motion_text_manifest.csv"
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    return len(rows), status_counts


def main() -> None:
    args = parse_args()
    processed_root = Path(args.processed_root)
    output_root = Path(args.output_root)
    object_dirs = sorted(path for path in processed_root.iterdir() if is_object_dir(path))
    if not object_dirs:
        raise RuntimeError(f"no PZ object folders found under {processed_root}")

    safe_prepare_output(output_root, processed_root, args.overwrite, args.dry_run)

    cond: dict[str, Any] = {}
    object_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    max_joints = 0
    total_frames = 0
    link_counts: dict[str, int] = {}

    for object_dir in object_dirs:
        object_cond = np.load(object_dir / "cond.npy", allow_pickle=True).item()
        if len(object_cond) != 1:
            raise ValueError(f"expected one cond key in {object_dir / 'cond.npy'}, got {list(object_cond.keys())}")
        object_name, value = next(iter(object_cond.items()))
        if object_name in cond:
            raise ValueError(f"duplicate object key: {object_name}")
        cond[object_name] = value

        joints = len(value.get("parents", []))
        max_joints = max(max_joints, joints)
        object_frames = parse_metadata_frames(object_dir / "metadata.txt")
        total_frames += object_frames

        counts = {}
        for subdir in DATA_SUBDIRS:
            src_dir = object_dir / subdir
            files = sorted(path for path in src_dir.glob("*") if path.is_file()) if src_dir.is_dir() else []
            counts[subdir] = len(files)
            for src in files:
                dst = output_root / subdir / src.name
                if not args.dry_run:
                    used_mode = materialize_file(src, dst, args.link_mode)
                    link_counts[used_mode] = link_counts.get(used_mode, 0) + 1
                else:
                    used_mode = "dry_run"
                file_rows.append(
                    {
                        "object_name": object_name,
                        "kind": subdir,
                        "source": str(src),
                        "destination": str(dst),
                        "materialization": used_mode,
                    }
                )

        object_rows.append(
            {
                "object_name": object_name,
                "source_dir": str(object_dir),
                "motions": counts["motions"],
                "bvhs": counts["bvhs"],
                "animations": counts["animations"],
                "joints": joints,
                "frames": object_frames,
            }
        )

    if not args.dry_run:
        np.save(output_root / "cond.npy", cond)
        write_jsonl(output_root / "pack_manifest.jsonl", file_rows)
        with (output_root / "object_index.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(object_rows[0].keys()))
            writer.writeheader()
            writer.writerows(object_rows)

        total_clips = sum(row["motions"] for row in object_rows)
        metadata = [
            f"max joints: {max_joints}",
            f"total frames: {total_frames}",
            f"duration: {int(total_frames / 12.5 / 60)}",
            f"~~~~ objects_counts - Total: {total_clips} ~~~~",
        ]
        metadata.extend(f"{row['object_name']}: {row['motions']}" for row in object_rows)
        (output_root / "metadata.txt").write_text("\n".join(metadata) + "\n", encoding="utf-8")

        text_rows = 0
        text_status_counts: dict[str, int] = {}
        if args.text_manifest:
            text_rows, text_status_counts = rewrite_text_manifest(Path(args.text_manifest), output_root)

        summary = {
            "processed_root": str(processed_root),
            "output_root": str(output_root),
            "objects": len(object_rows),
            "motions": total_clips,
            "bvhs": sum(row["bvhs"] for row in object_rows),
            "animations": sum(row["animations"] for row in object_rows),
            "max_joints": max_joints,
            "total_frames": total_frames,
            "link_counts": link_counts,
            "text_manifest_rows": text_rows,
            "text_status_counts": text_status_counts,
        }
        (output_root / "pack_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False))
    else:
        print(
            json.dumps(
                {
                    "processed_root": str(processed_root),
                    "output_root": str(output_root),
                    "objects": len(object_rows),
                    "motions": sum(row["motions"] for row in object_rows),
                    "bvhs": sum(row["bvhs"] for row in object_rows),
                    "animations": sum(row["animations"] for row in object_rows),
                    "max_joints": max_joints,
                    "total_frames": total_frames,
                    "dry_run": True,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
