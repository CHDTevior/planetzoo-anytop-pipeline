"""Audit no-IK official AniMo4D BVHs for physical pose discontinuities.

Quaternion angle alone is not a reliable failure signal: rapid turns and
short motion transitions can legitimately rotate a parent joint by a large
amount.  This scanner combines that signal with root-relative world-space
joint displacement.  A real upstream decode failure produces both a sudden
local rotation and a large discontinuity in its downstream subtree.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np


PROXIMAL_RE = re.compile(
    r"(frontLegUpr|frontLegLwr|rearLegUpr|rearLegLwr|upperArm|foreArm|shoulder|thigh|shin|calf|humerus|radius|ulna)",
    re.IGNORECASE,
)
EXTREMITY_RE = re.compile(
    r"(toe|claw|finger|pinky|ring|mid|index|thumb|foot|paw|end_site|lip|jaw|tongue|nose|nostril|brow|eyelid|cheek|ear|tail)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, help="Defaults to raw_root/export_manifest.jsonl.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--review-angle-deg", type=float, default=90.0)
    parser.add_argument("--review-subtree-step-scale", type=float, default=0.20)
    parser.add_argument("--severe-angle-deg", type=float, default=150.0)
    parser.add_argument("--severe-subtree-step-scale", type=float, default=0.45)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def descendants(parents: np.ndarray) -> list[list[int]]:
    children: list[list[int]] = [[] for _ in range(len(parents))]
    for joint, parent in enumerate(parents):
        if parent >= 0:
            children[int(parent)].append(joint)
    result: list[list[int]] = []
    for joint in range(len(parents)):
        stack = list(children[joint])
        found: list[int] = []
        while stack:
            child = stack.pop()
            found.append(child)
            stack.extend(children[child])
        result.append(found)
    return result


def quaternion_angle_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
    dot = np.clip(np.abs(np.sum(first * second, axis=-1)), -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def skeleton_scale(positions: np.ndarray) -> float:
    radius = np.linalg.norm(positions[0] - positions[0, 0], axis=-1)
    nonzero = radius[radius > 1e-6]
    if nonzero.size:
        return max(float(np.percentile(nonzero, 90)), 1e-4)
    return 1.0


def scan_one(row: dict, thresholds: dict[str, float]) -> dict:
    import BVH
    from Animation import positions_global

    path = Path(row["raw_bvh"])
    base = {
        "official_id": row.get("official_id"),
        "object_key": row.get("object_key"),
        "raw_bvh_stem": row.get("raw_bvh_stem"),
        "raw_bvh": str(path),
    }
    try:
        anim, names, frame_time = BVH.load(str(path))
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {**base, "status": "error", "error": repr(exc)}
    if len(anim) < 2:
        return {**base, "status": "too_short", "frames": int(len(anim)), "joints": int(len(names))}

    names = list(names)
    all_positions = positions_global(anim).astype(np.float64)
    relative_positions = all_positions - all_positions[:, :1]
    scale = skeleton_scale(all_positions)
    relative_step = np.linalg.norm(np.diff(relative_positions, axis=0), axis=-1) / scale
    rotation_step = quaternion_angle_deg(anim.rotations.qs[1:], anim.rotations.qs[:-1])
    desc = descendants(anim.parents)
    focus = [
        index
        for index, name in enumerate(names)
        if not EXTREMITY_RE.search(name)
        and PROXIMAL_RE.search(name) is not None
    ]
    if not focus:
        focus = [index for index, name in enumerate(names) if not EXTREMITY_RE.search(name) and len(desc[index]) >= 4]
    if not focus:
        focus = list(range(len(names)))

    focus_array = np.asarray(focus, dtype=np.int32)
    angle_arg = np.unravel_index(np.argmax(rotation_step[:, focus_array]), (len(anim) - 1, len(focus_array)))
    angle_frame, angle_joint_column = int(angle_arg[0]), int(angle_arg[1])
    angle_joint = int(focus_array[angle_joint_column])
    max_angle = float(rotation_step[angle_frame, angle_joint])
    affected = [angle_joint, *desc[angle_joint]]
    affected_step = relative_step[angle_frame, affected]
    affected_column = int(np.argmax(affected_step))
    affected_joint = int(affected[affected_column])
    subtree_step = float(affected_step[affected_column])

    spatial_arg = np.unravel_index(np.argmax(relative_step[:, focus_array]), (len(anim) - 1, len(focus_array)))
    spatial_frame, spatial_joint_column = int(spatial_arg[0]), int(spatial_arg[1])
    spatial_joint = int(focus_array[spatial_joint_column])
    max_spatial = float(relative_step[spatial_frame, spatial_joint])

    severe = max_angle >= thresholds["severe_angle_deg"] and subtree_step >= thresholds["severe_subtree_step_scale"]
    review = (
        max_angle >= thresholds["review_angle_deg"]
        and subtree_step >= thresholds["review_subtree_step_scale"]
    )
    flag = "severe" if severe else "review" if review else "clean"
    return {
        **base,
        "status": "ok",
        "flag": flag,
        "frames": int(len(anim)),
        "joints": int(len(names)),
        "frame_time": float(frame_time),
        "skeleton_scale": scale,
        "max_focus_rotation_step_deg": max_angle,
        "rotation_trigger_frame": angle_frame,
        "rotation_trigger_joint": angle_joint,
        "rotation_trigger_joint_name": names[angle_joint],
        "trigger_subtree_max_step_scale": subtree_step,
        "trigger_subtree_joint": affected_joint,
        "trigger_subtree_joint_name": names[affected_joint],
        "max_focus_relative_step_scale": max_spatial,
        "spatial_trigger_frame": spatial_frame,
        "spatial_trigger_joint": spatial_joint,
        "spatial_trigger_joint_name": names[spatial_joint],
        "score": max(max_spatial, (max_angle / 180.0) * subtree_step),
    }


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    manifest = args.manifest or args.raw_root / "export_manifest.jsonl"
    rows = [row for row in read_jsonl(manifest) if row.get("sample_type") == "motion"]
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = {
        "review_angle_deg": args.review_angle_deg,
        "review_subtree_step_scale": args.review_subtree_step_scale,
        "severe_angle_deg": args.severe_angle_deg,
        "severe_subtree_step_scale": args.severe_subtree_step_scale,
    }
    started = time.time()
    results_path = args.output_dir / "noik_physical_qc_all.jsonl"
    flagged_path = args.output_dir / "noik_physical_qc_flagged.jsonl"
    counts: Counter[str] = Counter()
    flagged: list[dict] = []
    with results_path.open("w", encoding="utf-8", newline="\n") as all_output, flagged_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as flagged_output:
        with cf.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(scan_one, row, thresholds) for row in rows]
            for completed, future in enumerate(cf.as_completed(futures), start=1):
                result = future.result()
                all_output.write(json.dumps(result, ensure_ascii=False) + "\n")
                counts[result.get("flag", result.get("status", "unknown"))] += 1
                if result.get("flag") in {"review", "severe"} or result.get("status") == "error":
                    flagged.append(result)
                    flagged_output.write(json.dumps(result, ensure_ascii=False) + "\n")
                if completed % 1000 == 0 or completed == len(rows):
                    print(
                        json.dumps(
                            {"processed": completed, "total": len(rows), "flags": dict(counts), "elapsed_sec": round(time.time() - started, 1)},
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
    flagged.sort(key=lambda item: item.get("score", math.inf), reverse=True)
    summary = {
        "raw_root": str(args.raw_root),
        "manifest_rows": len(rows),
        "thresholds": thresholds,
        "counts": dict(counts),
        "flagged_total": len(flagged),
        "top_flagged": flagged[:100],
        "results_jsonl": str(results_path),
        "flagged_jsonl": str(flagged_path),
        "elapsed_sec": round(time.time() - started, 3),
    }
    (args.output_dir / "noik_physical_qc_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if counts["severe"] or counts["error"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
