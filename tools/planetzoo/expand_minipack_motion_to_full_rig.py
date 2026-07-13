"""Expand a cleaned AnyTop minipack PZ motion to its full Planet Zoo topology.

The minipack deliberately removes non-essential joints.  It remains a valid
induced body hierarchy, but a full-topology decoder needs the original joint
order and one rot6d token for every child slot.  AnyTop stores the rotation of
joint ``p`` in each child ``c`` where ``parent[c] == p``.  This tool restores
the full rest-pose tensor, inserts the kept channels by joint name, and
broadcasts every retained parent's rot6d to all of its original child slots.

It is intended for skinning / visualisation, not for turning a reduced model
output back into an independently supervised full-topology training target.
Omitted joints remain at their rest pose; their independent facial or terminal
motion cannot be reconstructed from a cleaned minipack sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-path", required=True, type=Path, help="Raw [T, J_min, 13] minipack motion.")
    parser.add_argument("--skeleton-path", required=True, type=Path, help="Matching minipack skeleton.json.")
    condition = parser.add_mutually_exclusive_group(required=True)
    condition.add_argument("--full-cond-path", type=Path, help="Full AniMo4D cond.npy.")
    condition.add_argument("--full-skeleton-path", type=Path, help="One object's full_skeleton.json from the resource package.")
    parser.add_argument("--object-name", required=True, help="Planet Zoo object type, for example PZ_Bengal_Tiger_Male.")
    parser.add_argument("--output-motion", required=True, type=Path, help="Expanded raw [T, J_full, 13] .npy output.")
    parser.add_argument("--output-report", type=Path, help="Optional JSON report describing the mapping.")
    return parser.parse_args()


def children_by_parent(parents: np.ndarray) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for child, parent in enumerate(parents):
        if parent >= 0:
            children.setdefault(int(parent), []).append(child)
    return children


def unique_names(names: list[str], label: str) -> None:
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{label} has duplicate joint names: {duplicates[:8]}")


def load_full_entry(cond_path: Path | None, skeleton_path: Path | None, object_name: str) -> dict:
    if skeleton_path:
        entry = json.loads(skeleton_path.read_text(encoding="utf-8"))
        object_type = entry.get("object_type")
        if object_type and object_type != object_name:
            raise ValueError(f"full_skeleton object_type {object_type!r} does not match {object_name!r}")
        return entry
    cond = np.load(cond_path, allow_pickle=True).item()
    if object_name not in cond:
        raise KeyError(f"{object_name!r} is not present in {cond_path}")
    return cond[object_name]


def expand_motion(motion: np.ndarray, mini: dict, full: dict) -> tuple[np.ndarray, dict]:
    if motion.ndim != 3 or motion.shape[-1] != 13:
        raise ValueError(f"Expected [T, J, 13] motion, got {motion.shape}")
    mini_names = list(mini["joints_names"])
    mini_parents = np.asarray(mini["parents"], dtype=int)
    full_names = list(full["joints_names"])
    full_parents = np.asarray(full["parents"], dtype=int)
    rest = np.asarray(full["tpos_first_frame"], dtype=motion.dtype)
    if motion.shape[1] != len(mini_names) or len(mini_names) != len(mini_parents):
        raise ValueError("The minipack motion and skeleton.json have different joint counts.")
    if rest.shape != (len(full_names), 13) or len(full_names) != len(full_parents):
        raise ValueError("The full cond entry has an invalid tpos_first_frame / hierarchy shape.")
    unique_names(mini_names, "minipack skeleton")
    unique_names(full_names, "full cond skeleton")

    full_index = {name: index for index, name in enumerate(full_names)}
    unknown = [name for name in mini_names if name not in full_index]
    if unknown:
        raise ValueError(f"Minipack joints absent from full skeleton: {unknown[:8]}")
    mini_to_full = np.asarray([full_index[name] for name in mini_names], dtype=int)

    # The reduced topology must preserve every retained parent's original edge.
    # Otherwise a local rotation could not be safely inserted into the full rig.
    incompatible: list[dict] = []
    for mini_child, mini_parent in enumerate(mini_parents):
        full_child = int(mini_to_full[mini_child])
        expected_parent = -1 if mini_parent < 0 else int(mini_to_full[mini_parent])
        actual_parent = int(full_parents[full_child])
        if actual_parent != expected_parent:
            incompatible.append(
                {
                    "joint": mini_names[mini_child],
                    "mini_parent": None if mini_parent < 0 else mini_names[mini_parent],
                    "full_parent": None if actual_parent < 0 else full_names[actual_parent],
                }
            )
    if incompatible:
        raise ValueError(
            "The cleaned skeleton is not an induced subgraph of the full rig; "
            f"first mismatch: {incompatible[0]}"
        )

    frames = motion.shape[0]
    expanded = np.broadcast_to(rest, (frames, *rest.shape)).copy()
    expanded[:, mini_to_full, :] = motion

    mini_children = children_by_parent(mini_parents)
    full_children = children_by_parent(full_parents)
    broadcast_parents = 0
    unavailable_leaf_rotations: list[str] = []
    for full_parent, children in full_children.items():
        name = full_names[full_parent]
        try:
            mini_parent = mini_names.index(name)
        except ValueError:
            continue
        source_children = mini_children.get(mini_parent, [])
        if not source_children:
            # Leaf rotations are not represented by parent-indexed AnyTop
            # rot6d.  Rest pose is the only faithful default.
            unavailable_leaf_rotations.append(name)
            continue
        source = motion[:, source_children[0], 3:9]
        expanded[:, children, 3:9] = source[:, None, :]
        broadcast_parents += 1

    report = {
        "frames": int(frames),
        "minipack_joints": int(len(mini_names)),
        "full_joints": int(len(full_names)),
        "mapped_joints": int(len(mini_to_full)),
        "omitted_full_joints": int(len(full_names) - len(mini_to_full)),
        "induced_parent_graph": True,
        "rot6d_broadcast_parents": int(broadcast_parents),
        "unavailable_retained_leaf_rotation_count": int(len(unavailable_leaf_rotations)),
        "unavailable_retained_leaf_rotations": unavailable_leaf_rotations,
    }
    return expanded, report


def main() -> None:
    args = parse_args()
    motion = np.load(args.motion_path)
    mini = json.loads(args.skeleton_path.read_text(encoding="utf-8"))
    full = load_full_entry(args.full_cond_path, args.full_skeleton_path, args.object_name)
    expanded, report = expand_motion(motion, mini, full)
    args.output_motion.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_motion, expanded.astype(motion.dtype, copy=False))
    report.update(
        {
            "object_name": args.object_name,
            "motion_path": str(args.motion_path),
            "skeleton_path": str(args.skeleton_path),
            "full_condition_path": str(args.full_skeleton_path or args.full_cond_path),
            "output_motion": str(args.output_motion),
        }
    )
    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
