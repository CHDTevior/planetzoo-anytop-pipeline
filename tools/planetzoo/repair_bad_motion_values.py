"""Repair bad Planet Zoo AnyTop motion values and refresh per-object normalization.

This script is intentionally conservative:
- bad motion/BVH files are moved to a quarantine directory, not deleted;
- original metadata files are copied to a timestamped backup directory;
- cond.npy keeps the same object keys and skeleton fields, but mean/std are
  recomputed from the remaining motion files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


BACKUP_FILENAMES = [
    "cond.npy",
    "motion_texts_by_file_with_codex_drafts.json",
    "motion_texts_by_file_with_codex_drafts_summary.json",
    "motion_texts_by_file_with_animosty4d_matches.json",
    "motion_text_manifest.json",
    "motion_text_manifest.jsonl",
    "motion_text_manifest.csv",
    "motion_text_match_summary.json",
    "pack_manifest.jsonl",
    "pack_summary.json",
    "object_index.csv",
    "metadata.txt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-root", required=True, type=Path)
    parser.add_argument("--quarantine-root", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=100.0)
    parser.add_argument("--std-floor", type=float, default=1e-6)
    parser.add_argument("--backup-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_object_names(root: Path) -> list[str]:
    object_index = root / "object_index.csv"
    if not object_index.is_file():
        return []
    with object_index.open(newline="", encoding="utf-8") as f:
        return sorted([row["object_name"] for row in csv.DictReader(f)], key=len, reverse=True)


def object_from_filename(name: str, object_names: list[str]) -> str:
    for object_name in object_names:
        if name.startswith(f"{object_name}_"):
            return object_name
    return name.rsplit("_", 1)[0]


def scan_bad_motions(root: Path, threshold: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    motion_dir = root / "motions"
    object_names = load_object_names(root)
    bad: list[dict[str, Any]] = []
    total = 0
    per_object = defaultdict(lambda: {"files": 0, "bad": 0, "frames": 0})
    for path in sorted(motion_dir.glob("*.npy")):
        total += 1
        object_name = object_from_filename(path.name, object_names)
        per_object[object_name]["files"] += 1
        try:
            arr = np.load(path, mmap_mode="r")
            per_object[object_name]["frames"] += int(arr.shape[0])
            finite = np.isfinite(arr)
            nonfinite_count = int((~finite).sum())
            try:
                max_abs = float(np.nanmax(np.abs(arr)))
            except ValueError:
                max_abs = math.inf
            is_bad = nonfinite_count > 0 or (not math.isfinite(max_abs)) or max_abs > threshold
            reason = []
            if nonfinite_count > 0:
                reason.append("nonfinite")
            if (not math.isfinite(max_abs)) or max_abs > threshold:
                reason.append(f"abs_gt_{threshold:g}")
            if is_bad:
                per_object[object_name]["bad"] += 1
                bad.append(
                    {
                        "object": object_name,
                        "file": path.name,
                        "path": str(path),
                        "shape": [int(x) for x in arr.shape],
                        "nonfinite_count": nonfinite_count,
                        "max_abs": max_abs,
                        "reason": reason,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            per_object[object_name]["bad"] += 1
            bad.append(
                {
                    "object": object_name,
                    "file": path.name,
                    "path": str(path),
                    "error": repr(exc),
                    "nonfinite_count": -1,
                    "max_abs": math.inf,
                    "reason": ["load_error"],
                }
            )
    summary = {
        "motion_files": total,
        "bad_motion_files": len(bad),
        "bad_objects": sorted({row["object"] for row in bad}),
        "per_object": per_object,
    }
    return bad, summary


def copy_backups(root: Path, backup_root: Path, dry_run: bool) -> list[str]:
    copied = []
    if dry_run:
        return copied
    backup_root.mkdir(parents=True, exist_ok=True)
    for name in BACKUP_FILENAMES:
        src = root / name
        if src.exists():
            dst = backup_root / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(name)
    return copied


def move_bad_files(root: Path, quarantine_root: Path, bad: list[dict[str, Any]], dry_run: bool) -> list[dict[str, Any]]:
    moved = []
    if dry_run:
        return moved
    (quarantine_root / "motions").mkdir(parents=True, exist_ok=True)
    (quarantine_root / "bvhs").mkdir(parents=True, exist_ok=True)
    for row in bad:
        motion_src = root / "motions" / row["file"]
        bvh_name = f"{Path(row['file']).stem}.bvh"
        bvh_src = root / "bvhs" / bvh_name
        record = dict(row)
        if motion_src.exists():
            motion_dst = quarantine_root / "motions" / motion_src.name
            if motion_dst.exists():
                motion_dst.unlink()
            shutil.move(str(motion_src), str(motion_dst))
            record["quarantine_motion"] = str(motion_dst)
        if bvh_src.exists():
            bvh_dst = quarantine_root / "bvhs" / bvh_src.name
            if bvh_dst.exists():
                bvh_dst.unlink()
            shutil.move(str(bvh_src), str(bvh_dst))
            record["quarantine_bvh"] = str(bvh_dst)
        moved.append(record)
    return moved


def get_mean_std(data: np.ndarray, std_floor: float) -> tuple[np.ndarray, np.ndarray]:
    mean = data.mean(axis=0)
    std = data.std(axis=0)

    std[0, :3] = std[0, :3].mean()
    std[0, 3:9] = std[0, 3:9].mean()
    std[0, 9:12] = std[0, 9:12].mean()

    if std.shape[0] > 1:
        std[1:, :3] = std[1:, :3].mean()
        std[1:, 3:9] = std[1:, 3:9].mean()
        std[1:, 9:12] = std[1:, 9:12].mean()

    foot_std = std[:, 12]
    nonzero = foot_std != 0
    if np.any(nonzero):
        foot_std[nonzero] = foot_std[nonzero].mean()
    foot_std[~nonzero] = 1.0
    std[:, 12] = foot_std

    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.nan_to_num(std, nan=std_floor, posinf=std_floor, neginf=std_floor)
    non_foot = np.ones(std.shape, dtype=bool)
    non_foot[:, 12] = False
    std[non_foot & (std < std_floor)] = std_floor
    return mean, std


def recompute_cond(root: Path, std_floor: float, dry_run: bool) -> dict[str, Any]:
    cond_path = root / "cond.npy"
    cond = np.load(cond_path, allow_pickle=True).item()
    object_names = sorted(cond.keys(), key=len, reverse=True)
    by_object: dict[str, list[Path]] = {obj: [] for obj in cond}
    for path in sorted((root / "motions").glob("*.npy")):
        by_object[object_from_filename(path.name, object_names)].append(path)

    report = {}
    for object_name, files in by_object.items():
        if not files:
            report[object_name] = {"files": 0, "updated": False, "error": "no_remaining_motions"}
            continue
        arrays = [np.load(path) for path in files]
        data = np.concatenate(arrays, axis=0)
        mean, std = get_mean_std(data, std_floor=std_floor)
        if not dry_run:
            cond[object_name]["mean"] = mean
            cond[object_name]["std"] = std
        report[object_name] = {
            "files": len(files),
            "frames": int(data.shape[0]),
            "joints": int(data.shape[1]),
            "updated": True,
            "mean_abs_max": float(np.abs(mean).max()),
            "std_min": float(std.min()),
            "std_max": float(std.max()),
        }
    if not dry_run:
        np.save(cond_path, cond)
    return report


def filter_json_dict(path: Path, bad_names: set[str], dry_run: bool) -> int:
    if not path.exists():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return 0
    before = len(data)
    data = {k: v for k, v in data.items() if k not in bad_names}
    if not dry_run:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return before - len(data)


def summarize_caption_dict(path: Path, summary_path: Path, dry_run: bool) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    text_status = Counter()
    annotation_source = Counter()
    caption_lengths = Counter()
    for row in data.values():
        text_status[row.get("text_status", "")] += 1
        annotation_source[row.get("annotation_source", "")] += 1
        caption_lengths[str(len(row.get("captions", [])))] += 1
    summary = {
        "rows": len(data),
        "text_status": dict(sorted(text_status.items())),
        "annotation_source": dict(sorted(annotation_source.items())),
        "caption_lengths": dict(sorted(caption_lengths.items())),
        "fields": sorted(next(iter(data.values())).keys()) if data else [],
        "repair_note": "Updated after filtering abnormal motion values.",
    }
    if not dry_run:
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def row_motion_basename(row: dict[str, Any]) -> str | None:
    for key in ("processed_motion", "source_file"):
        val = row.get(key)
        if val:
            return Path(str(val)).name
    return None


def update_motion_text_manifest(root: Path, bad_names: set[str], dry_run: bool) -> dict[str, Any]:
    path = root / "motion_text_manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    before_rows = len(data["rows"])
    rows = [row for row in data["rows"] if row_motion_basename(row) not in bad_names]
    status_counts = Counter(row.get("text_match_status", "") for row in rows)
    summary = data.get("summary", {})
    summary["rows"] = len(rows)
    summary["status_counts"] = dict(sorted(status_counts.items()))
    summary["object_count"] = len({row.get("object_key", "") for row in rows})
    summary["unique_action_short_count"] = len({row.get("action_short", "") for row in rows})
    data["summary"] = summary
    data["rows"] = rows
    if not dry_run:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    jsonl_path = root / "motion_text_manifest.jsonl"
    if jsonl_path.exists() and not dry_run:
        with jsonl_path.open("w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = root / "motion_text_manifest.csv"
    if csv_path.exists():
        if rows:
            fieldnames = list(rows[0].keys())
        else:
            fieldnames = []
        if not dry_run:
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    return {"rows": len(rows), "removed": before_rows - len(rows)}


def update_pack_manifest(root: Path, bad_names: set[str], dry_run: bool) -> int:
    bad_bvh = {f"{Path(name).stem}.bvh" for name in bad_names}
    path = root / "pack_manifest.jsonl"
    if not path.exists():
        return 0
    kept = []
    removed = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            basename = Path(str(row.get("destination", ""))).name
            if basename in bad_names or basename in bad_bvh:
                removed += 1
                continue
            kept.append(row)
    if not dry_run:
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for row in kept:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return removed


def update_object_index_and_summaries(root: Path, dry_run: bool) -> dict[str, Any]:
    cond = np.load(root / "cond.npy", allow_pickle=True).item()
    old_sources = {}
    object_index_path = root / "object_index.csv"
    if object_index_path.exists():
        with object_index_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                old_sources[row["object_name"]] = row.get("source_dir", "")

    motion_files = sorted((root / "motions").glob("*.npy"))
    bvh_files = sorted((root / "bvhs").glob("*.bvh"))
    object_names = sorted(cond.keys(), key=len, reverse=True)
    counts = {obj: {"motions": 0, "bvhs": 0, "frames": 0, "joints": int(cond[obj]["mean"].shape[0])} for obj in cond}
    for path in motion_files:
        obj = object_from_filename(path.name, object_names)
        arr = np.load(path, mmap_mode="r")
        counts[obj]["motions"] += 1
        counts[obj]["frames"] += int(arr.shape[0])
    for path in bvh_files:
        obj = object_from_filename(path.name.replace(".bvh", ".npy"), object_names)
        if obj in counts:
            counts[obj]["bvhs"] += 1

    rows = []
    for obj in sorted(counts):
        rows.append(
            {
                "object_name": obj,
                "source_dir": old_sources.get(obj, ""),
                "motions": counts[obj]["motions"],
                "bvhs": counts[obj]["bvhs"],
                "animations": 0,
                "joints": counts[obj]["joints"],
                "frames": counts[obj]["frames"],
            }
        )
    if not dry_run:
        with object_index_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    total_motions = sum(row["motions"] for row in rows)
    total_bvhs = sum(row["bvhs"] for row in rows)
    total_frames = sum(row["frames"] for row in rows)
    max_joints = max(row["joints"] for row in rows)

    pack_summary_path = root / "pack_summary.json"
    pack_summary = json.loads(pack_summary_path.read_text(encoding="utf-8")) if pack_summary_path.exists() else {}
    pack_summary.update(
        {
            "objects": len(rows),
            "motions": total_motions,
            "bvhs": total_bvhs,
            "animations": 0,
            "max_joints": max_joints,
            "total_frames": total_frames,
            "text_manifest_rows": total_motions,
            "repair_note": "Filtered abnormal motion values and recomputed cond mean/std.",
        }
    )
    if not dry_run:
        pack_summary_path.write_text(json.dumps(pack_summary, indent=2), encoding="utf-8")

    metadata_lines = [
        f"max joints: {max_joints}",
        f"total frames: {total_frames}",
        f"duration: {int(total_frames / 12.5 / 60)}",
        f"~~~~ objects_counts - Total: {total_motions} ~~~~",
    ]
    metadata_lines += [f"{row['object_name']}: {row['motions']}" for row in rows]
    if not dry_run:
        (root / "metadata.txt").write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")

    return {
        "objects": len(rows),
        "motions": total_motions,
        "bvhs": total_bvhs,
        "max_joints": max_joints,
        "total_frames": total_frames,
    }


def main() -> None:
    args = parse_args()
    root = args.layout_root
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quarantine_root = args.quarantine_root or root.parent / f"quarantine_bad_motion_values_{timestamp}"
    backup_root = args.backup_root or root.parent / f"repair_backup_bad_motion_values_{timestamp}"

    bad, scan_summary = scan_bad_motions(root, threshold=args.threshold)
    bad_names = {row["file"] for row in bad}

    copied = copy_backups(root, backup_root, dry_run=args.dry_run)
    moved = move_bad_files(root, quarantine_root, bad, dry_run=args.dry_run)
    cond_report = recompute_cond(root, std_floor=args.std_floor, dry_run=args.dry_run)

    removed_text_clean = filter_json_dict(root / "motion_texts_by_file_with_codex_drafts.json", bad_names, args.dry_run)
    removed_text_rich = filter_json_dict(root / "motion_texts_by_file_with_animosty4d_matches.json", bad_names, args.dry_run)
    clean_summary = summarize_caption_dict(
        root / "motion_texts_by_file_with_codex_drafts.json",
        root / "motion_texts_by_file_with_codex_drafts_summary.json",
        args.dry_run,
    )
    manifest_summary = update_motion_text_manifest(root, bad_names, args.dry_run)
    pack_manifest_removed = update_pack_manifest(root, bad_names, args.dry_run)
    layout_summary = update_object_index_and_summaries(root, args.dry_run)

    match_summary_path = root / "motion_text_match_summary.json"
    if match_summary_path.exists() and not args.dry_run:
        match_summary = json.loads(match_summary_path.read_text(encoding="utf-8"))
        match_summary["rows"] = clean_summary["rows"]
        match_summary["status_counts"] = clean_summary["text_status"]
        match_summary["object_count"] = layout_summary["objects"]
        match_summary["repair_note"] = "Updated after filtering abnormal motion values."
        match_summary_path.write_text(json.dumps(match_summary, indent=2), encoding="utf-8")

    repair_manifest = {
        "layout_root": str(root),
        "threshold": args.threshold,
        "std_floor": args.std_floor,
        "dry_run": args.dry_run,
        "backup_root": str(backup_root),
        "quarantine_root": str(quarantine_root),
        "backup_files": copied,
        "bad_motion_count": len(bad),
        "bad_objects": scan_summary["bad_objects"],
        "bad_motions": moved if moved else bad,
        "removed_text_clean": removed_text_clean,
        "removed_text_rich": removed_text_rich,
        "pack_manifest_removed_rows": pack_manifest_removed,
        "layout_summary": layout_summary,
        "cond_recomputed_objects": len([r for r in cond_report.values() if r.get("updated")]),
    }
    if not args.dry_run:
        (root / "repair_bad_values_manifest.json").write_text(json.dumps(repair_manifest, indent=2), encoding="utf-8")
        quarantine_root.mkdir(parents=True, exist_ok=True)
        (quarantine_root / "repair_bad_values_manifest.json").write_text(json.dumps(repair_manifest, indent=2), encoding="utf-8")

    print(json.dumps(repair_manifest, indent=2))


if __name__ == "__main__":
    main()
