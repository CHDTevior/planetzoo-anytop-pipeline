"""Validate a KTJD-17 Planet Zoo dataset without relying on training code."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-report", type=Path)
    return parser.parse_args()


def validate_clip(task: tuple[Path, dict, int]) -> dict:
    root, clip, expected_joints = task
    path = root / clip["motion_file"]
    with np.load(path) as data:
        motion = data["motion"]
        heading_valid = data["heading_valid"]
    errors = []
    if motion.ndim != 3 or motion.shape[-1] != 17:
        errors.append(f"motion shape {motion.shape}")
    elif motion.shape[1] != expected_joints:
        errors.append(f"motion joints={motion.shape[1]}, skeleton joints={expected_joints}")
    if heading_valid.shape != (motion.shape[0],):
        errors.append(f"heading_valid shape {heading_valid.shape}")
    if not np.issubdtype(heading_valid.dtype, np.bool_):
        errors.append(f"heading_valid dtype {heading_valid.dtype}")
    if not np.isfinite(motion).all():
        errors.append("non-finite motion")
    if not np.all(motion[~heading_valid, 0, 15:17] == 0.0):
        errors.append("invalid heading rows are nonzero")
    return {
        "clip_id": clip["clip_id"],
        "errors": errors,
        "shape": list(motion.shape),
        "max_abs": float(np.max(np.abs(motion))),
    }


def main() -> None:
    args = parse_args()
    root = args.dataset_root
    clips = [json.loads(line) for line in (root / "manifests" / "clips.jsonl").read_text(encoding="utf-8").splitlines() if line]
    skeletons: dict[str, int] = {}
    skeleton_errors: list[dict] = []
    stats_errors: list[dict] = []
    for path in sorted((root / "skeletons").glob("*.npz")):
        with np.load(path) as data:
            names = data["joint_names"]
            parents = data["parents"]
            positions = data["P_rest_global"]
            rotations = data["R_rest_global"]
        rig_id = path.stem
        skeletons[rig_id] = len(names)
        if len(set(names.tolist())) != len(names):
            skeleton_errors.append({"rig_id": rig_id, "error": "duplicate joint names"})
        if parents.shape != (len(names),) or np.count_nonzero(parents == -1) != 1:
            skeleton_errors.append({"rig_id": rig_id, "error": f"invalid parents {parents.shape}"})
        if positions.shape != (len(names), 3) or rotations.shape != (len(names), 3, 3):
            skeleton_errors.append({"rig_id": rig_id, "error": "invalid rest shape"})
        if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
            skeleton_errors.append({"rig_id": rig_id, "error": "non-finite rest values"})
        stats_path = root / "stats" / f"{rig_id}.npz"
        if not stats_path.is_file():
            stats_errors.append({"rig_id": rig_id, "error": "missing stats"})
        else:
            with np.load(stats_path) as stats:
                for key in ("count", "mean", "std_raw", "std_eff", "supervise_mask"):
                    if key not in stats or stats[key].shape != (len(names), 17):
                        stats_errors.append({"rig_id": rig_id, "error": f"invalid stats {key}"})
                        break
                else:
                    if not np.isfinite(stats["mean"]).all() or not np.isfinite(stats["std_eff"]).all():
                        stats_errors.append({"rig_id": rig_id, "error": "non-finite stats"})
                    elif np.any(stats["std_eff"] <= 0):
                        stats_errors.append({"rig_id": rig_id, "error": "non-positive effective std"})
    missing_rigs = sorted({clip["rig_id"] for clip in clips} - set(skeletons))
    tasks = [(root, clip, skeletons.get(clip["rig_id"], -1)) for clip in clips]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        clip_results = list(executor.map(validate_clip, tasks))
    clip_errors = [result for result in clip_results if result["errors"]]
    motion_files = {path.stem for path in (root / "motions").glob("*.npz")}
    manifest_files = {clip["clip_id"] for clip in clips}
    fps_counts = Counter(int(clip["source_fps"]) for clip in clips)
    report = {
        "dataset_root": str(root),
        "clip_count": len(clips),
        "skeleton_count": len(skeletons),
        "stats_count": len(list((root / "stats").glob("*.npz"))),
        "missing_rigs": missing_rigs,
        "orphan_motion_files": len(motion_files - manifest_files),
        "missing_motion_files": len(manifest_files - motion_files),
        "skeleton_errors": skeleton_errors,
        "stats_errors": stats_errors,
        "clip_error_count": len(clip_errors),
        "clip_errors": clip_errors[:100],
        "source_fps_counts": dict(sorted(fps_counts.items())),
        "stored_frame_min": min(clip["stored_frames"] for clip in clips),
        "stored_frame_max": max(clip["stored_frames"] for clip in clips),
        "joint_min": min(result["shape"][1] for result in clip_results),
        "joint_max": max(result["shape"][1] for result in clip_results),
        "motion_max_abs": max(result["max_abs"] for result in clip_results),
        "fk_error_max": max(clip["fk_position_max"] for clip in clips),
        "fk_error_mean": float(np.mean([clip["fk_position_mae"] for clip in clips])),
    }
    output = args.output_report or root / "reports" / "dataset_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"clip_errors"}}, indent=2))
    if missing_rigs or skeleton_errors or stats_errors or clip_errors or report["orphan_motion_files"] or report["missing_motion_files"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
