"""Reconstruct a paired motion-extracted MANIS action without smoothing.

Planet Zoo keeps certain moving ``motionextracted`` actions paired with a
co-timed ``notmotionextracted`` ``onspot`` action. This script retains the
moving action's root/trunk translation and copies only independently tracked
limb-branch local rotations from its exact on-spot pair. The branch selector
comes from ``LimbTrackData`` and the MS2 hierarchy, not joint-name heuristics.

The input pair must have passed the strict matching contract in
``build_motionextracted_onspot_pairs.py``. No interpolation, temporal
smoothing, Blender IK, or runtime capture is used.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import bpy
import numpy as np


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cobra-tools", required=True, type=Path)
    parser.add_argument("--ms2-path", required=True, type=Path)
    parser.add_argument("--extracted-manis", required=True, type=Path)
    parser.add_argument("--extracted-action", required=True)
    parser.add_argument("--onspot-manis", required=True, type=Path)
    parser.add_argument("--onspot-action", required=True)
    parser.add_argument("--output-position-npz", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--output-bvh", type=Path, help="Optional standard BVH export of the reconstructed action.")
    parser.add_argument(
        "--rotation-bone-pattern",
        action="append",
        default=[],
        help="Optional regex, repeatable. Copy only matching pose-bone rotations from the on-spot action.",
    )
    parser.add_argument(
        "--use-limb-branches",
        action="store_true",
        help="Select independent limb branches from the extracted MANIS LimbTrackData and MS2 hierarchy.",
    )
    return parser.parse_args(argv)


class Reporter:
    def show_info(self, message): print(f"INFO: {message}")
    def show_warning(self, message): print(f"WARNING: {message}")
    def show_error(self, message): print(f"ERROR: {message}")


def register_cobra(path: Path) -> None:
    sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location(
        "cobra_tools_addon", str(path / "__init__.py"), submodule_search_locations=[str(path)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Cobra Tools from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cobra_tools_addon"] = module
    spec.loader.exec_module(module)
    module.register()


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def copy_curve(source, target) -> None:
    copied = target.fcurves.new(source.data_path, index=source.array_index, action_group=source.group.name if source.group else None)
    copied.extrapolation = source.extrapolation
    copied.keyframe_points.add(len(source.keyframe_points) - 1)
    for dst, src in zip(copied.keyframe_points, source.keyframe_points):
        dst.co = src.co
        dst.handle_left = src.handle_left
        dst.handle_right = src.handle_right
        dst.interpolation = src.interpolation
        dst.handle_left_type = src.handle_left_type
        dst.handle_right_type = src.handle_right_type


def curve_bone_name(curve) -> str | None:
    match = re.fullmatch(r'pose\.bones\["(.+)"\]\.rotation_quaternion', curve.data_path)
    return match.group(1) if match else None


def limb_branch_bones(extracted_manis: Path, action_name: str, rig) -> set[str]:
    """Return limb-only bones, excluding ancestors shared by multiple limbs."""
    from generated.formats.manis import ManisFile  # pylint: disable=import-outside-toplevel
    from plugin.utils.blender_util import bone_name_for_blender  # pylint: disable=import-outside-toplevel

    manis = ManisFile()
    manis.load(str(extracted_manis))
    action = next(info for info in manis.mani_infos if info.name == action_name)
    limbs = getattr(action.keys, "limb_track_data", None)
    if limbs is None:
        raise ValueError(f"{action_name} has no LimbTrackData")
    chains: list[list[str]] = []
    for limb in limbs.limbs:
        if not limb.keys.list_one:
            continue
        depth = int(limb.keys.list_one[0].countb)
        bone = rig.pose.bones.get(bone_name_for_blender(limb.bone))
        if bone is None:
            continue
        chain = []
        for _ in range(depth):
            chain.append(bone.name)
            if bone.parent is None:
                break
            bone = bone.parent
        chains.append(chain)
    if not chains:
        raise ValueError(f"No LimbTrackData bones resolved for {action_name}")
    use_count = {name: sum(name in chain for chain in chains) for chain in chains for name in chain}
    return {name for chain in chains for name in chain if use_count[name] == 1}


def export_bvh(rig, action, output_path: Path) -> None:
    scene = bpy.context.scene
    scene.frame_start = int(action.frame_range[0])
    scene.frame_end = int(action.frame_range[1])
    scene.render.fps = 20
    rig.animation_data.action = action
    bpy.ops.preferences.addon_enable(module="io_anim_bvh")
    bpy.ops.object.select_all(action="DESELECT")
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_anim.bvh(
        filepath=str(output_path),
        frame_start=scene.frame_start,
        frame_end=scene.frame_end,
        global_scale=1.0,
        rotate_mode="NATIVE",
        root_transform_only=False,
    )
    promote_single_child_root(output_path)


def promote_single_child_root(bvh_path: Path) -> bool:
    """Strip Blender's zero-channel wrapper so AnyTop sees a single BVH root."""
    lines = bvh_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 8 or not lines[1].lstrip().startswith("ROOT "):
        return False
    if "CHANNELS 0" not in lines[4]:
        return False
    child_match = re.match(r"\s*JOINT\s+(.+)", lines[5])
    if child_match is None:
        return False
    depth = 0
    child_close = None
    for index in range(5, len(lines)):
        stripped = lines[index].strip()
        if stripped == "{":
            depth += 1
        elif stripped == "}":
            depth -= 1
            if depth == 0:
                child_close = index
                break
    try:
        motion_index = next(index for index, line in enumerate(lines) if line.strip() == "MOTION")
    except StopIteration:
        return False
    if child_close is None or child_close >= motion_index - 1 or lines[motion_index - 1].strip() != "}":
        return False
    fixed = [lines[0], f"ROOT {child_match.group(1)}"]
    fixed.extend(lines[6:child_close])
    for line in lines[child_close + 1 : motion_index - 1]:
        fixed.append("\t" + line if line.strip() else line)
    fixed.append(lines[child_close])
    fixed.extend(lines[motion_index:])
    bvh_path.write_text("\n".join(fixed) + "\n", encoding="utf-8")
    return True


def quat_step_degrees(previous: np.ndarray, current: np.ndarray) -> float:
    dot = abs(float(np.dot(previous, current)))
    return float(math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot)))))


def main() -> None:
    args = parse_args()
    clear_scene()
    register_cobra(args.cobra_tools)
    from plugin import import_manis, import_ms2  # pylint: disable=import-outside-toplevel

    import_ms2.load(reporter=Reporter(), filepath=str(args.ms2_path), merge_vertices=False)
    rig = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE")
    import_manis.load(reporter=Reporter(), filepath=str(args.extracted_manis), disable_ik=True)
    import_manis.load(reporter=Reporter(), filepath=str(args.onspot_manis), disable_ik=True)
    extracted = bpy.data.actions[args.extracted_action]
    onspot = bpy.data.actions[args.onspot_action]
    if extracted.frame_range != onspot.frame_range:
        raise ValueError(f"Frame ranges differ: extracted={tuple(extracted.frame_range)}, onspot={tuple(onspot.frame_range)}")

    hybrid = extracted.copy()
    hybrid.name = f"{args.extracted_action}__onspot_local_rotations"
    patterns = [re.compile(pattern) for pattern in args.rotation_bone_pattern]
    if args.use_limb_branches and patterns:
        raise ValueError("Use either --use-limb-branches or --rotation-bone-pattern, not both")
    selected_limb_bones = limb_branch_bones(args.extracted_manis, args.extracted_action, rig) if args.use_limb_branches else set()

    def should_copy(curve) -> bool:
        bone_name = curve_bone_name(curve)
        if bone_name is None:
            return False
        if selected_limb_bones:
            return bone_name in selected_limb_bones
        return not patterns or any(pattern.search(bone_name) for pattern in patterns)

    for curve in tuple(hybrid.fcurves):
        if should_copy(curve):
            hybrid.fcurves.remove(curve)
    rotation_sources = [curve for curve in onspot.fcurves if should_copy(curve)]
    for curve in rotation_sources:
        copy_curve(curve, hybrid)

    rig.animation_data_create()
    rig.animation_data.action = hybrid
    for bone in rig.pose.bones:
        for constraint in bone.constraints:
            constraint.mute = True

    frames = list(range(int(hybrid.frame_range[0]), int(hybrid.frame_range[1]) + 1))
    names = [bone.name for bone in rig.pose.bones]
    positions = np.empty((len(frames), len(names), 3), dtype=np.float64)
    quats = np.empty((len(frames), len(names), 4), dtype=np.float64)
    for frame_i, frame in enumerate(frames):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        for bone_i, name in enumerate(names):
            bone = rig.pose.bones[name]
            positions[frame_i, bone_i] = bone.matrix.to_translation()[:]
            quats[frame_i, bone_i] = bone.matrix_basis.to_quaternion()[:]
    steps = np.asarray([
        [quat_step_degrees(quats[f, b], quats[f + 1, b]) for b in range(len(names))]
        for f in range(len(frames) - 1)
    ])
    worst_flat = int(np.argmax(steps))
    worst_frame, worst_bone = np.unravel_index(worst_flat, steps.shape)
    parents = np.asarray([names.index(bone.parent.name) if bone.parent else -1 for bone in rig.pose.bones], dtype=np.int32)
    args.output_position_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_position_npz, names=np.asarray(names), parents=parents, frames=np.asarray(frames, dtype=np.int32), positions=positions)
    report = {
        "purpose": "strict paired offline reconstruction; no smoothing, interpolation, or IK",
        "extracted_action": args.extracted_action,
        "onspot_action": args.onspot_action,
        "rotation_curve_source": "onspot",
        "translation_curve_source": "motionextracted",
        "frames": [frames[0], frames[-1]],
        "max_local_step_degrees": float(steps[worst_frame, worst_bone]),
        "max_local_step_transition": [frames[worst_frame], frames[worst_frame + 1]],
        "max_local_step_bone": names[worst_bone],
        "rotation_curve_count": len(rotation_sources),
        "rotation_bone_patterns": args.rotation_bone_pattern,
        "use_limb_branches": bool(args.use_limb_branches),
        "limb_branch_bones": sorted(selected_limb_bones),
        "rotation_bones": sorted({curve_bone_name(curve) for curve in rotation_sources}),
    }
    if args.output_bvh:
        export_bvh(rig, hybrid, args.output_bvh)
        report["output_bvh"] = str(args.output_bvh)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
