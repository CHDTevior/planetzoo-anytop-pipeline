"""Scan final AniMo4D-AnyTop samples for proximal limb rotation spikes.

The scanner operates on the raw BVH paths recorded in the final
motion_text_manifest.jsonl, but counts only samples that are present in the
final AnyTop layout. It is intended to catch shoulder/upper-limb axis flips,
not harmless toe/claw/end-site jitter.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


PROXIMAL_RE = re.compile(
    r"(frontLegUpr|frontLegLwr|rearLegUpr|rearLegLwr|upperArm|foreArm|shoulder|thigh|shin|calf|humerus|radius|ulna)",
    re.IGNORECASE,
)
EXCLUDE_RE = re.compile(
    r"(toe|claw|finger|pinky|ring|mid|index|thumb|foot|paw|end_site|lip|jaw|tongue|nose|nostril|brow|eyelid|cheek|ear|tail)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-descendants", type=int, default=4)
    parser.add_argument("--candidate-threshold", type=float, default=80.0)
    parser.add_argument("--severe-jump-threshold", type=float, default=150.0)
    parser.add_argument("--severe-accel-threshold", type=float, default=120.0)
    parser.add_argument("--borderline-threshold", type=float, default=55.0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_manifest(layout_root: Path, limit: int | None) -> list[dict]:
    rows = []
    manifest = layout_root / "motion_text_manifest.jsonl"
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            raw = row.get("raw_bvh")
            motion = row.get("processed_motion")
            if not raw or not motion:
                continue
            rows.append(
                {
                    "raw_bvh": str(Path(raw)),
                    "object_name": row.get("object_name"),
                    "motion_file": row.get("motion_file"),
                    "processed_motion": motion,
                    "processed_bvh": row.get("processed_bvh"),
                    "text": row.get("text"),
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def q_angle_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = q1 / np.maximum(np.linalg.norm(q1, axis=-1, keepdims=True), 1e-12)
    q2 = q2 / np.maximum(np.linalg.norm(q2, axis=-1, keepdims=True), 1e-12)
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def descendants(parents: np.ndarray) -> list[list[int]]:
    children: list[list[int]] = [[] for _ in range(len(parents))]
    for joint, parent in enumerate(parents):
        if parent >= 0:
            children[int(parent)].append(joint)
    result: list[list[int]] = []
    for joint in range(len(parents)):
        stack = list(children[joint])
        desc: list[int] = []
        while stack:
            child = stack.pop()
            desc.append(child)
            stack.extend(children[child])
        result.append(desc)
    return result


def root_center(positions: np.ndarray) -> np.ndarray:
    result = positions.copy()
    result[..., 0] -= result[:, None, 0, 0]
    result[..., 2] -= result[:, None, 0, 2]
    return result


def scan_one(row: dict, min_descendants: int, borderline_threshold: float, candidate_threshold: float) -> dict:
    import BVH
    from Animation import positions_global

    path = Path(row["raw_bvh"])
    try:
        anim, names, frametime = BVH.load(str(path))
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {**row, "status": "error", "error": repr(exc)}

    q = anim.rotations.qs
    names = list(names)
    if q.shape[0] < 5:
        return {**row, "status": "too_short", "frames": int(q.shape[0]), "joints": int(q.shape[1])}

    mask = np.array(
        [bool(PROXIMAL_RE.search(name)) and not bool(EXCLUDE_RE.search(name)) for name in names],
        dtype=bool,
    )
    if not mask.any():
        return {**row, "status": "no_proximal_joints", "frames": int(q.shape[0]), "joints": int(q.shape[1])}

    desc = descendants(anim.parents)
    proximal_indices = [int(i) for i in np.where(mask)[0] if len(desc[int(i)]) >= min_descendants]
    if not proximal_indices:
        return {**row, "status": "no_proximal_with_descendants", "frames": int(q.shape[0]), "joints": int(q.shape[1])}

    proximal = np.asarray(proximal_indices, dtype=int)
    jump = q_angle_deg(q[1:], q[:-1])
    accel = np.abs(np.diff(jump, axis=0))
    jump_p = jump[:, proximal]
    accel_p = accel[:, proximal]
    jump_arg = np.unravel_index(np.argmax(jump_p), jump_p.shape)
    accel_arg = np.unravel_index(np.argmax(accel_p), accel_p.shape)
    jump_joint = int(proximal[jump_arg[1]])
    accel_joint = int(proximal[accel_arg[1]])
    max_jump = float(jump_p[jump_arg])
    max_accel = float(accel_p[accel_arg])
    trigger_joint = jump_joint if max_jump >= 0.75 * max_accel else accel_joint
    trigger_frame = int(jump_arg[0]) if trigger_joint == jump_joint else int(accel_arg[0])
    trigger_desc = desc[trigger_joint]

    base = {
        **row,
        "status": "ok",
        "raw_bvh_file": path.name,
        "object_dir": path.parent.parent.name,
        "frames": int(q.shape[0]),
        "joints": int(q.shape[1]),
        "frametime": float(frametime),
        "max_proximal_jump_deg": max_jump,
        "max_proximal_jump_delta_frame": int(jump_arg[0]),
        "max_proximal_jump_joint": jump_joint,
        "max_proximal_jump_joint_name": names[jump_joint],
        "max_proximal_accel_deg": max_accel,
        "max_proximal_accel_frame": int(accel_arg[0]),
        "max_proximal_accel_joint": accel_joint,
        "max_proximal_accel_joint_name": names[accel_joint],
        "trigger_joint": trigger_joint,
        "trigger_joint_name": names[trigger_joint],
        "trigger_frame_raw": trigger_frame,
        "trigger_descendant_count": int(len(trigger_desc)),
        "score_base": float(max(max_jump, max_accel)),
    }

    if max(max_jump, max_accel) < borderline_threshold:
        return {**base, "flag": "clean"}

    subtree_jerk_max = 0.0
    subtree_jerk_p95 = 0.0
    if trigger_desc:
        positions = root_center(positions_global(anim))
        jerk = np.linalg.norm(np.diff(positions[:, trigger_desc], n=3, axis=0), axis=-1)
        if jerk.size:
            subtree_jerk_max = float(jerk.max())
            subtree_jerk_p95 = float(np.percentile(jerk, 95))

    score = max(max_jump, max_accel) * math.log1p(max(len(trigger_desc), 1)) * (1.0 + min(subtree_jerk_max, 2.0))
    if max(max_jump, max_accel) >= candidate_threshold:
        flag = "candidate"
    else:
        flag = "borderline"
    return {
        **base,
        "flag": flag,
        "subtree_jerk_max": subtree_jerk_max,
        "subtree_jerk_p95": subtree_jerk_p95,
        "score": float(score),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(args.layout_root, args.limit)
    started = time.time()
    results_path = args.output_dir / "proximal_rotation_qc_all_results.jsonl"
    flags_path = args.output_dir / "proximal_rotation_qc_flagged.jsonl"
    summary_path = args.output_dir / "proximal_rotation_qc_summary.json"

    status_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    object_flag_counts: Counter[str] = Counter()
    joint_flag_counts: Counter[str] = Counter()
    severe_count = 0
    candidate_count = 0
    borderline_count = 0
    processed = 0
    flagged_rows = []
    top_rows = []

    with results_path.open("w", encoding="utf-8") as all_f, flags_path.open("w", encoding="utf-8") as flag_f:
        with cf.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    scan_one,
                    row,
                    args.min_descendants,
                    args.borderline_threshold,
                    args.candidate_threshold,
                )
                for row in rows
            ]
            for future in cf.as_completed(futures):
                result = future.result()
                processed += 1
                all_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                status_counts[result.get("status", "missing")] += 1
                flag = result.get("flag")
                if flag:
                    flag_counts[flag] += 1
                if flag in {"candidate", "borderline"}:
                    if (
                        result.get("max_proximal_jump_deg", 0.0) >= args.severe_jump_threshold
                        or result.get("max_proximal_accel_deg", 0.0) >= args.severe_accel_threshold
                    ):
                        severity = "severe"
                        severe_count += 1
                    elif flag == "candidate":
                        severity = "candidate"
                        candidate_count += 1
                    else:
                        severity = "borderline"
                        borderline_count += 1
                    result["severity"] = severity
                    flag_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    flagged_rows.append(result)
                    object_flag_counts[result.get("object_name") or ""] += 1
                    joint_flag_counts[result.get("trigger_joint_name") or ""] += 1
                    top_rows.append(result)
                    top_rows = sorted(top_rows, key=lambda item: item.get("score", 0.0), reverse=True)[:50]

                if processed % 1000 == 0 or processed == len(rows):
                    elapsed = time.time() - started
                    print(
                        json.dumps(
                            {
                                "processed": processed,
                                "total": len(rows),
                                "elapsed_sec": round(elapsed, 1),
                                "status_counts": dict(status_counts),
                                "flag_counts": dict(flag_counts),
                                "severe": severe_count,
                                "candidate": candidate_count,
                                "borderline": borderline_count,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

    summary = {
        "layout_root": str(args.layout_root),
        "output_dir": str(args.output_dir),
        "manifest_rows": len(rows),
        "workers": args.workers,
        "rules": {
            "proximal_regex": PROXIMAL_RE.pattern,
            "excluded_regex": EXCLUDE_RE.pattern,
            "min_descendants": args.min_descendants,
            "borderline_threshold_deg": args.borderline_threshold,
            "candidate_threshold_deg": args.candidate_threshold,
            "severe_jump_threshold_deg": args.severe_jump_threshold,
            "severe_accel_threshold_deg": args.severe_accel_threshold,
        },
        "status_counts": dict(status_counts),
        "flag_counts": dict(flag_counts),
        "severity_counts": {
            "severe": severe_count,
            "candidate": candidate_count,
            "borderline": borderline_count,
        },
        "flagged_total": severe_count + candidate_count + borderline_count,
        "candidate_or_severe_total": severe_count + candidate_count,
        "flagged_fraction": (severe_count + candidate_count + borderline_count) / max(len(rows), 1),
        "candidate_or_severe_fraction": (severe_count + candidate_count) / max(len(rows), 1),
        "top_objects": object_flag_counts.most_common(30),
        "top_trigger_joints": joint_flag_counts.most_common(30),
        "top_scored": top_rows,
        "all_results_jsonl": str(results_path),
        "flagged_jsonl": str(flags_path),
        "elapsed_sec": round(time.time() - started, 3),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
