"""Reproduce AniMo4D's position-only MS2/MANIS export for a single action.

AniMo4D's public exporter loads MS2 plus MANIS in Blender with the default
``disable_ik=False`` and writes ``pose_bone.head`` for every frame.  This probe
uses exactly that evaluation route, writes the public JSON shape plus the
subsequent fixed 30-joint ``[F, 30, 3]`` array, and reports root-relative
per-frame position jumps.  It does not modify the source assets.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import bpy
import numpy as np


ANIMO4D_JOINTS = [
    "def_c_root_joint",
    "def_c_chest_joint",
    "def_c_neck1_joint",
    "def_c_head_joint",
    "def_c_jaw_joint",
    "def_eye_joint.L",
    "def_eye_joint.R",
    "def_clavicle_joint.L",
    "def_clavicle_joint.R",
    "def_c_tail1_joint",
    "def_c_anus_joint",
    "def_c_tail2_joint",
    "def_frontLegUpr_joint.L",
    "def_frontLegLwr_joint.L",
    "def_frontFoot_joint.L",
    "def_frontLegLwrAllTwist_joint.L",
    "def_frontLegUpr_joint.R",
    "def_frontLegLwr_joint.R",
    "def_frontFoot_joint.R",
    "def_frontLegLwrAllTwist_joint.R",
    "def_rearLegUpr_joint.L",
    "def_rearLegLwr_joint.L",
    "def_rearFoot_joint.L",
    "def_rearLegLwrAllTwist_joint.L",
    "def_rearLegUprAllTwist_joint.L",
    "def_rearLegUpr_joint.R",
    "def_rearLegLwr_joint.R",
    "def_rearFoot_joint.R",
    "def_rearLegLwrAllTwist_joint.R",
    "def_rearLegUprAllTwist_joint.R",
]


class Reporter:
    def __call__(self, message_type, message) -> None:
        print(f"{message_type}: {message}")

    def show_info(self, message) -> None:
        print(f"INFO: {message}")

    def show_warning(self, message) -> None:
        print(f"WARNING: {message}")

    def show_error(self, message) -> None:
        print(f"ERROR: {message}")


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--cobra-tools", required=True, type=Path)
    parser.add_argument("--ms2-path", required=True, type=Path)
    parser.add_argument("--manis-path", required=True, type=Path)
    parser.add_argument("--action", required=True, help="Exact Blender action name, e.g. caracal_male@walkbase.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--disable-ik",
        action="store_true",
        help=(
            "Disable the Blender IK constraints created by Cobra Tools before "
            "sampling. This is a diagnostic A/B mode; omit it to reproduce "
            "AniMo4D's original IK-enabled exporter."
        ),
    )
    parser.add_argument(
        "--mute-all-constraints",
        action="store_true",
        help=(
            "Mute every imported Blender pose-bone constraint before sampling. "
            "Diagnostic only: this isolates decompressed MANIS F-curves from "
            "all Blender constraint evaluation."
        ),
    )
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)


def register_cobra(cobra_tools: Path) -> None:
    sys.path.insert(0, str(cobra_tools))
    spec = importlib.util.spec_from_file_location(
        "cobra_tools_addon",
        str(cobra_tools / "__init__.py"),
        submodule_search_locations=[str(cobra_tools)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Cobra Tools from {cobra_tools}")
    addon = importlib.util.module_from_spec(spec)
    sys.modules["cobra_tools_addon"] = addon
    spec.loader.exec_module(addon)
    addon.register()


def find_armature() -> bpy.types.Object:
    armature = next((obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"), None)
    if armature is None:
        raise RuntimeError("MS2 import produced no armature.")
    return armature


def mute_all_constraints(armature: bpy.types.Object) -> int:
    count = 0
    for pose_bone in armature.pose.bones:
        for constraint in pose_bone.constraints:
            constraint.mute = True
            count += 1
    return count


def collect_animo4d_positions(armature: bpy.types.Object, action: bpy.types.Action) -> tuple[dict[str, dict[str, list[float]]], np.ndarray, list[int]]:
    armature.animation_data_create()
    armature.animation_data.action = action
    start, end = (int(value) for value in action.frame_range)
    missing = [name for name in ANIMO4D_JOINTS if name not in armature.pose.bones]
    if missing:
        raise RuntimeError(f"This action cannot enter AniMo4D's 30-joint format; missing: {missing}")

    raw: dict[str, dict[str, list[float]]] = {}
    frames = list(range(start, end + 1))
    positions = np.empty((len(frames), len(ANIMO4D_JOINTS), 3), dtype=np.float64)
    for index, frame in enumerate(frames):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        # This intentionally mirrors AniMo_blender_ovl2json.py: bone.head,
        # not tail positions, local rotations, or the evaluated mesh.
        raw[str(frame)] = {bone.name: [float(value) for value in bone.head.xyz] for bone in armature.pose.bones}
        positions[index] = np.asarray([armature.pose.bones[name].head.xyz[:] for name in ANIMO4D_JOINTS], dtype=np.float64)
    return raw, positions, frames


def position_jump_report(positions: np.ndarray, frames: list[int]) -> dict:
    root_relative = positions - positions[:, :1]
    step = np.linalg.norm(np.diff(root_relative, axis=0), axis=-1)
    combined = np.linalg.norm(np.diff(root_relative, axis=0).reshape(len(step), -1), axis=1)
    per_joint = []
    for joint, name in enumerate(ANIMO4D_JOINTS):
        frame_index = int(np.argmax(step[:, joint]))
        per_joint.append(
            {
                "joint_index": joint,
                "joint_name": name,
                "max_root_relative_step": float(step[frame_index, joint]),
                "transition_frames": [frames[frame_index], frames[frame_index + 1]],
            }
        )
    frame_index = int(np.argmax(combined))
    return {
        "metric": "Euclidean displacement of root-relative pose_bone.head positions between adjacent frames",
        "frame_count": int(len(frames)),
        "joint_count": len(ANIMO4D_JOINTS),
        "max_all_joint_step": float(combined[frame_index]),
        "max_all_joint_transition_frames": [frames[frame_index], frames[frame_index + 1]],
        "top_joints": sorted(per_joint, key=lambda row: row["max_root_relative_step"], reverse=True)[:12],
        "per_joint": per_joint,
    }


def main() -> None:
    args = parse_args()
    for path in [args.cobra_tools / "__init__.py", args.ms2_path, args.manis_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clear_scene()
    register_cobra(args.cobra_tools)
    from plugin import import_manis, import_ms2  # pylint: disable=import-outside-toplevel

    import_ms2.load(reporter=Reporter(), filepath=str(args.ms2_path), merge_vertices=False)
    armature = find_armature()
    # AniMo4D's public export uses the default (IK enabled). The optional
    # switch isolates Blender/Cobra IK evaluation without changing the asset.
    import_manis.load(reporter=Reporter(), filepath=str(args.manis_path), disable_ik=args.disable_ik)
    muted_constraint_count = mute_all_constraints(armature) if args.mute_all_constraints else 0
    action = bpy.data.actions.get(args.action)
    if action is None:
        candidates = sorted(item.name for item in bpy.data.actions if item.id_root == "OBJECT")
        raise KeyError(f"Action {args.action!r} was not imported. Candidates: {candidates}")

    raw, positions, frames = collect_animo4d_positions(armature, action)
    raw_path = args.output_dir / "official_keypoints.json"
    npy_path = args.output_dir / "official_positions_30j.npy"
    report_path = args.output_dir / "official_position_probe.json"
    raw_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    np.save(npy_path, positions)
    report = {
        "source": "AniMo4D-compatible Blender pose_bone.head export",
        "ms2_path": str(args.ms2_path),
        "manis_path": str(args.manis_path),
        "action": action.name,
        "ik_disabled": bool(args.disable_ik),
        "all_constraints_muted": bool(args.mute_all_constraints),
        "muted_constraint_count": muted_constraint_count,
        "joint_order": ANIMO4D_JOINTS,
        "json_path": str(raw_path),
        "npy_path": str(npy_path),
        "position_jump_report": position_jump_report(positions, frames),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
