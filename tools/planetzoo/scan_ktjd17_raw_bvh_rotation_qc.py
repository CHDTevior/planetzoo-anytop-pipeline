"""Scan freshly exported, IK-disabled BVHs for retained-joint rotation jumps.

This is deliberately a diagnostic rather than a repair/filtering pass.  It
measures the sign-invariant adjacent-frame angle of each *retained* local BVH
quaternion, so a false sign flip between q and -q cannot be reported as a
motion jump.  The records in ``export_manifest.jsonl`` are also checked for
the strict provenance required by the KTJD-17 builder.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MOTION_LIB = REPO_ROOT / "tools" / "planetzoo" / "motion_lib"
if str(MOTION_LIB) not in sys.path:
    sys.path.insert(0, str(MOTION_LIB))


# These are the joints whose abrupt local rotation can redirect a visible limb
# or torso.  Digit/foot-end rotations are reported separately: the source has
# many of those, but they do not explain the historical whole-arm failures.
STRUCTURAL_RE = re.compile(
    r"(shoulder|upperarm|forearm|humerus|radius|ulna|frontlegupr|frontleglwr|rearlegupr|rearleglwr|thigh|shin|calf|hips|pelvis|spine|chest|neck)",
    re.IGNORECASE,
)
EXTREMITY_RE = re.compile(r"(toe|foot|paw|finger|thumb|index|pinky|claw|phalanx|tail|ear|jaw|tongue)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--joint-spec", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--candidate-degrees", type=float, default=90.0)
    return parser.parse_args()


def quaternion_step_degrees(rotations: np.ndarray) -> np.ndarray:
    """Return sign-invariant angular distance for adjacent quaternion frames."""
    norm = rotations / np.maximum(np.linalg.norm(rotations, axis=-1, keepdims=True), 1e-12)
    dot = np.abs(np.sum(norm[1:] * norm[:-1], axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def scan_one(task: tuple[dict, set[str], float]) -> dict:
    row, retained_names, threshold = task
    import BVH  # Imported in the worker so Windows process spawning stays clean.

    try:
        animation, names, frame_time = BVH.load(str(row["raw_bvh"]))
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"clip_id": row["raw_bvh_stem"], "status": "load_error", "error": repr(exc)}
    names = list(names)
    retained = np.asarray([index for index, name in enumerate(names) if name in retained_names], dtype=int)
    if len(animation) < 2 or retained.size == 0:
        return {
            "clip_id": row["raw_bvh_stem"],
            "status": "not_scannable",
            "frames": int(len(animation)),
            "retained_joint_count": int(retained.size),
        }
    steps = quaternion_step_degrees(animation.rotations.qs[:, retained])

    def maximum(indices: np.ndarray, prefix: str) -> dict:
        if indices.size == 0:
            return {f"{prefix}_joint_count": 0, f"{prefix}_step_deg": 0.0, f"{prefix}_candidate": False}
        local = steps[:, indices]
        frame_index, local_index = np.unravel_index(np.argmax(local), local.shape)
        retained_index = int(indices[local_index])
        joint_index = int(retained[retained_index])
        maximum_step = float(local[frame_index, local_index])
        return {
            f"{prefix}_joint_count": int(indices.size),
            f"{prefix}_step_deg": maximum_step,
            f"{prefix}_step_frame": int(frame_index),
            f"{prefix}_step_joint": names[joint_index],
            f"{prefix}_candidate": bool(maximum_step > threshold),
        }

    structural = np.asarray(
        [
            i
            for i, joint_index in enumerate(retained)
            if STRUCTURAL_RE.search(names[int(joint_index)]) and not EXTREMITY_RE.search(names[int(joint_index)])
        ],
        dtype=int,
    )
    return {
        "clip_id": row["raw_bvh_stem"],
        "status": "ok",
        "frames": int(len(animation)),
        "source_fps": int(round(1.0 / frame_time)),
        "retained_joint_count": int(retained.size),
        **maximum(np.arange(retained.size, dtype=int), "all_retained"),
        **maximum(structural, "structural"),
    }


def main() -> None:
    args = parse_args()
    specification = json.loads(args.joint_spec.read_text(encoding="utf-8"))
    retained_by_rig = {
        rig_id: {joint["name"] for joint in rig["joints"]}
        for rig_id, rig in specification["rigs"].items()
    }
    rows = [
        json.loads(line)
        for line in (args.raw_root / "export_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    motions = []
    provenance_errors = []
    for row in rows:
        if row.get("sample_type") != "motion":
            continue
        rig_id = f"PZ_{row['object_key']}"
        if rig_id not in retained_by_rig:
            provenance_errors.append({"clip_id": row.get("raw_bvh_stem"), "error": "rig absent from joint spec"})
            continue
        if not row.get("source_action_verified") or not row.get("ik_disabled_during_export"):
            provenance_errors.append({"clip_id": row.get("raw_bvh_stem"), "error": "unverified or IK-enabled export"})
            continue
        motions.append((row, retained_by_rig[rig_id], args.candidate_degrees))

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(scan_one, motions, chunksize=24):
            results.append(result)

    status_counts = Counter(result["status"] for result in results)
    all_candidates = [result for result in results if result.get("all_retained_candidate")]
    structural_candidates = [result for result in results if result.get("structural_candidate")]
    maxima = [result["all_retained_step_deg"] for result in results if result["status"] == "ok"]
    report = {
        "raw_root": str(args.raw_root),
        "candidate_degrees": args.candidate_degrees,
        "motions_requested": len(motions),
        "status_counts": dict(sorted(status_counts.items())),
        "provenance_errors": provenance_errors,
        "max_all_retained_local_step_deg": max(maxima, default=0.0),
        "all_retained_candidate_count": len(all_candidates),
        "structural_candidate_count": len(structural_candidates),
        "top_all_retained_candidates": sorted(
            all_candidates, key=lambda item: item["all_retained_step_deg"], reverse=True
        )[:100],
        "top_structural_candidates": sorted(
            structural_candidates, key=lambda item: item["structural_step_deg"], reverse=True
        )[:100],
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
