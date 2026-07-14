"""Decode an AnyTop motion array back to a Planet Zoo mesh animation.

The AnyTop tensor is intentionally canonical: its root orientation, root
translation, scale and rest-basis differ from the original Planet Zoo rig.
This tool reverses those deterministic preprocessing steps using the object's
original T-pose BVH, writes an inspectable reconstructed raw BVH, then drives
the original weighted MS2 armature with a rest-relative pose bridge.  The
driver preserves the MS2 inverse-bind basis instead of reusing BVH bone rolls.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


REPO_ROOT = Path(__file__).resolve().parents[2]
HML_AVG_BONELEN = 0.2092142857142857


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--cobra-tools", required=True, type=Path)
    parser.add_argument(
        "--motion-lib",
        type=Path,
        default=Path(__file__).with_name("motion_lib"),
        help="Directory containing BVH.py and Quaternions.py (defaults to the bundled motion_lib).",
    )
    parser.add_argument("--ms2-path", required=True, type=Path)
    parser.add_argument("--manis-path", required=True, type=Path, help="Matching game MANIS, used to initialise the native rig state.")
    parser.add_argument("--motion-path", required=True, type=Path)
    condition = parser.add_mutually_exclusive_group(required=True)
    condition.add_argument("--cond-path", type=Path, help="Full AniMo4D cond.npy containing every object entry.")
    condition.add_argument(
        "--full-skeleton-path",
        type=Path,
        help="One object's full_skeleton.json from the skinning resource package.",
    )
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--tpose-bvh", required=True, type=Path)
    parser.add_argument("--raw-template-bvh", required=True, type=Path, help="Matching original action BVH used only for its local bind basis.")
    parser.add_argument("--output-raw-bvh", required=True, type=Path)
    parser.add_argument("--output-blend", required=True, type=Path)
    parser.add_argument("--output-mp4", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--qc-remove-filenames", type=Path)
    parser.add_argument(
        "--face-joints",
        nargs=2,
        default=["def_c_hips_joint", "def_c_chest_joint"],
        metavar=("TAIL", "HEAD"),
    )
    parser.add_argument("--max-frames", type=int, default=72)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--debug-frame-dir", type=Path, help="Optional directory for first/middle/last mesh validation frames.")
    parser.add_argument(
        "--show-world-axes",
        action="store_true",
        help="Render a labelled scene-axis triad (+X red, +Y green, +Z blue).",
    )
    return parser.parse_args(argv)


class Reporter:
    def __call__(self, message_type, message) -> None:
        print(f"{message_type}: {message}")

    def show_info(self, message) -> None:
        print(f"INFO: {message}")

    def show_warning(self, message) -> None:
        print(f"WARNING: {message}")

    def show_error(self, message) -> None:
        print(f"ERROR: {message}")


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for collection in list(bpy.data.collections):
        if collection.name != "Scene Collection":
            bpy.data.collections.remove(collection)


def register_cobra(cobra_tools: Path) -> None:
    sys.path.insert(0, str(cobra_tools))
    spec = importlib.util.spec_from_file_location(
        "cobra_tools_addon",
        str(cobra_tools / "__init__.py"),
        submodule_search_locations=[str(cobra_tools)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Cobra Tools from {cobra_tools}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cobra_tools_addon"] = module
    spec.loader.exec_module(module)
    module.register()


def load_motion_modules(motion_lib: Path):
    sys.path.insert(0, str(motion_lib))
    import Animation  # pylint: disable=import-outside-toplevel
    import BVH  # pylint: disable=import-outside-toplevel
    from Quaternions import Quaternions  # pylint: disable=import-outside-toplevel

    return Animation, BVH, Quaternions


def load_condition(args: argparse.Namespace) -> dict:
    if args.full_skeleton_path:
        entry = json.loads(args.full_skeleton_path.read_text(encoding="utf-8"))
        object_type = entry.get("object_type")
        if object_type and object_type != args.object_name:
            raise ValueError(
                f"full_skeleton object_type {object_type!r} does not match --object-name {args.object_name!r}"
            )
        return {args.object_name: entry}
    cond = np.load(args.cond_path, allow_pickle=True).item()
    if args.object_name not in cond:
        raise KeyError(f"{args.object_name} not present in {args.cond_path}")
    return cond


def rotation_6d_to_matrix_np(cont6d: np.ndarray) -> np.ndarray:
    """NumPy-equivalent of AnyTop's rotation_6d_to_matrix_np."""
    if cont6d.shape[-1] != 6:
        raise ValueError(f"Expected a six-dimensional rotation, got {cont6d.shape}")
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]
    x = x_raw / np.linalg.norm(x_raw, axis=-1, keepdims=True)
    z = np.cross(x, y_raw, axis=-1)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.cross(z, x, axis=-1)
    return np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)


def prune_planetzoo_helpers(anim, names: list[str]):
    excluded = {index for index, name in enumerate(names) if name.lower() == "srb" or name.lower().startswith("srb_")}
    changed = True
    while changed:
        changed = False
        for index, parent in enumerate(anim.parents):
            if parent in excluded and index not in excluded:
                excluded.add(index)
                changed = True
    keep = [index for index in range(len(names)) if index not in excluded]
    return anim[:, keep], [names[index] for index in keep], sorted(excluded)


def safe_normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else fallback.copy()


def yaw_to_face_z(positions: np.ndarray, tail: int, head: int, Quaternions):
    forward = positions[head] - positions[tail]
    forward[1] = 0.0
    forward = safe_normalize(forward, np.array([0.0, 0.0, 1.0]))
    angle = -math.atan2(float(forward[0]), float(forward[2]))
    return Quaternions.from_angle_axis(np.array([angle]), np.array([0.0, 1.0, 0.0]))


def repeat_quaternion(quat, shape, Quaternions):
    return Quaternions(np.tile(quat.qs.reshape(-1, 4)[0], tuple(shape) + (1,)))


def offsets_from_positions(positions: np.ndarray, parents: np.ndarray) -> np.ndarray:
    offsets = positions.copy()
    for joint, parent in enumerate(parents):
        if parent >= 0:
            offsets[..., joint, :] -= positions[..., parent, :]
    return offsets


def positions_from_offsets(offsets: np.ndarray, parents: np.ndarray) -> np.ndarray:
    positions = np.zeros_like(offsets, dtype=float)
    for joint, parent in enumerate(parents):
        positions[joint] = offsets[joint] if parent < 0 else positions[parent] + offsets[joint]
    return positions


def selected_joint_indices(names: list[str], offsets: np.ndarray, tokens: list[str], excluded: list[str]) -> list[int]:
    indices = []
    for index, name in enumerate(names):
        lname = name.lower()
        if index == 0 or float(np.linalg.norm(offsets[index])) <= 1e-8:
            continue
        if any(token in lname for token in excluded):
            continue
        if any(token in lname for token in tokens):
            indices.append(index)
    return indices


def compute_rest_parameters(tpose_anim, names: list[str], parents: np.ndarray, face_names: list[str], Animation, Quaternions):
    tail, head = (names.index(name) for name in face_names)
    global_tpose = Animation.positions_global(tpose_anim)[0]
    yaw = yaw_to_face_z(global_tpose, tail, head, Quaternions)
    roll = Quaternions.from_angle_axis(np.array([-math.pi / 2.0]), np.array([0.0, 0.0, 1.0]))

    processed_rotations = tpose_anim.rotations.copy()
    processed_rotations[:, 0] = roll * yaw * processed_rotations[:, 0]
    processed_positions = tpose_anim.positions.copy()
    processed_positions[:, 0] = roll * (yaw * processed_positions[:, 0])
    processed = Animation.Animation(
        processed_rotations,
        processed_positions,
        tpose_anim.orients.copy(),
        tpose_anim.offsets.copy(),
        parents.copy(),
    )

    root_origin_xz = processed.positions[0, 0] * np.array([1.0, 0.0, 1.0])
    processed.positions[:, 0] -= root_origin_xz
    processed.offsets[0] -= root_origin_xz

    scale_indices = selected_joint_indices(
        names,
        processed.offsets,
        ["spine2", "spine3", "chest_joint", "neck1", "neck2", "head_joint", "frontlegupr_joint", "frontleglwr_joint", "frontfoot_joint", "rearlegupr_joint", "rearleglwr_joint", "rearfoot_joint"],
        ["end_site", "twist", "srb", "breath"],
    )
    lengths = np.linalg.norm(processed.offsets[scale_indices], axis=1) if scale_indices else np.linalg.norm(processed.offsets[1:], axis=1)
    lengths = lengths[lengths > 1e-8]
    if not len(lengths):
        raise RuntimeError("Cannot determine an AnyTop scale factor from this T-pose.")
    scale_factor = HML_AVG_BONELEN / float(np.mean(lengths))
    processed.positions *= scale_factor
    processed.offsets *= scale_factor

    ground_indices = selected_joint_indices(names, processed.offsets, ["toe", "foot", "hoof", "ashi"], [])
    rest_global = Animation.positions_global(processed)
    ground_height = float(rest_global[:, ground_indices, 1].min()) if ground_indices else float(rest_global[..., 1].min())
    processed.positions[:, 0, 1] -= ground_height
    processed.offsets[0, 1] -= ground_height

    canonical_offsets = offsets_from_positions(Animation.positions_global(processed), parents)[0]
    canonical_rest_positions = positions_from_offsets(canonical_offsets, parents)
    rest_yaw = yaw_to_face_z(canonical_rest_positions, tail, head, Quaternions)
    yaw_offsets = canonical_offsets.copy()
    yaw_offsets[1:] = repeat_quaternion(rest_yaw, (len(names) - 1,), Quaternions) * yaw_offsets[1:]

    foot_indices = [index for index, name in enumerate(names) if any(token in name.lower() for token in ["toe", "foot", "hoof", "ashi", "paw"])]
    core_indices = [index for index, name in enumerate(names) if any(token in name.lower() for token in ["hips", "spine", "chest", "neck"])]
    best_roll = -math.pi / 2.0
    if foot_indices and core_indices:
        best_score = -np.inf
        for angle in [-math.pi / 2.0, math.pi / 2.0]:
            candidate = Quaternions.from_angle_axis(np.array([angle]), np.array([0.0, 0.0, 1.0]))
            candidate_offsets = yaw_offsets.copy()
            candidate_offsets[1:] = repeat_quaternion(candidate, (len(names) - 1,), Quaternions) * candidate_offsets[1:]
            candidate_positions = positions_from_offsets(candidate_offsets, parents)
            score = float(candidate_positions[core_indices, 1].mean() - candidate_positions[foot_indices, 1].mean())
            if score > best_score + 1e-8:
                best_score = score
                best_roll = angle
    rest_roll = Quaternions.from_angle_axis(np.array([best_roll]), np.array([0.0, 0.0, 1.0]))
    rest_align = rest_roll * rest_yaw

    return {
        "yaw": yaw,
        "roll": roll,
        "root_origin_xz": root_origin_xz,
        "scale_factor": scale_factor,
        "ground_height": ground_height,
        "tpos_rotations": processed.rotations.copy(),
        "rest_align": rest_align,
        "rest_roll_degrees": math.degrees(best_roll),
        "scale_joint_count": len(scale_indices),
        "ground_joint_count": len(ground_indices),
    }


def decode_feature_rotations(data: np.ndarray, parents: np.ndarray, tpos_rotations, rest_align, Quaternions, rotation_6d_to_matrix_np):
    frames, joints = data.shape[:2]
    feature_rotations = np.tile(tpos_rotations.qs[0], (frames, 1, 1))
    child_for_parent: dict[int, int] = {}
    for child, parent in enumerate(parents):
        if parent >= 0 and parent not in child_for_parent:
            child_for_parent[int(parent)] = child
    for parent, child in child_for_parent.items():
        feature_rotations[:, parent] = Quaternions.from_transforms(rotation_6d_to_matrix_np(data[:, child, 3:9])).qs

    feature = Quaternions(feature_rotations)
    q_frame = repeat_quaternion(rest_align, (frames,), Quaternions)
    q_frame_joint = repeat_quaternion(rest_align, (frames, joints - 1), Quaternions)
    unaligned = feature.copy()
    unaligned[:, 0] = feature[:, 0] * q_frame
    if joints > 1:
        unaligned[:, 1:] = -q_frame_joint * feature[:, 1:] * q_frame_joint

    tpos = Quaternions(np.tile(tpos_rotations.qs[0], (frames, 1, 1)))
    cumulative_tpos = tpos.copy()
    destination = unaligned.copy()
    destination[:, 0] = unaligned[:, 0] * tpos[:, 0]
    for joint, parent in enumerate(parents):
        if parent < 0:
            continue
        cumulative_tpos[:, joint] = cumulative_tpos[:, parent] * tpos[:, joint]
        destination[:, joint] = -cumulative_tpos[:, parent] * unaligned[:, joint] * cumulative_tpos[:, parent] * tpos[:, joint]
    return destination


def recover_root_positions(data: np.ndarray, parameters: dict, action_yaw, Quaternions, rotation_6d_to_matrix_np) -> np.ndarray:
    frames = data.shape[0]
    facing = Quaternions.from_transforms(rotation_6d_to_matrix_np(data[:, 0, 3:9]))
    canonical = np.zeros((frames, 3), dtype=float)
    canonical[1:, [0, 2]] = data[:-1, 0, [9, 11]]
    canonical = -facing * canonical
    canonical = np.cumsum(canonical, axis=0)
    canonical[:, 1] = data[:, 0, 1]

    ungrounded = canonical.copy()
    ungrounded[:, 1] += parameters["ground_height"]
    unscaled = ungrounded / parameters["scale_factor"] + parameters["root_origin_xz"]
    return -action_yaw * (-parameters["roll"] * unscaled)


def reconstruct_raw_animation(data: np.ndarray, cond: dict, object_name: str, tpose_bvh: Path, raw_template_bvh: Path, face_names: list[str], motion_lib: Path):
    Animation, BVH, Quaternions = load_motion_modules(motion_lib)
    tpose_anim, names, frame_time = BVH.load(str(tpose_bvh))
    tpose_anim, names, excluded = prune_planetzoo_helpers(tpose_anim, names)
    raw_template, template_names, _ = BVH.load(str(raw_template_bvh))
    raw_template, template_names, template_excluded = prune_planetzoo_helpers(raw_template, template_names)
    parents = np.asarray(tpose_anim.parents, dtype=int)
    entry = cond[object_name]
    cond_parents = np.asarray(entry["parents"], dtype=int)
    cond_names = list(entry.get("joints_names", names))
    if data.shape[1] != len(names) or len(cond_parents) != len(names):
        raise RuntimeError(f"Joint count mismatch: motion={data.shape[1]}, T-pose={len(names)}, cond={len(cond_parents)}")
    if not np.array_equal(parents, cond_parents):
        raise RuntimeError("The supplied T-pose hierarchy does not match cond.npy.")
    if cond_names != names:
        raise RuntimeError("The supplied T-pose joint order does not match cond.npy.")
    if template_names != names or not np.array_equal(raw_template.parents, parents):
        raise RuntimeError("The supplied raw template BVH does not match the T-pose hierarchy.")

    parameters = compute_rest_parameters(tpose_anim, names, parents, face_names, Animation, Quaternions)
    tail, head = (names.index(name) for name in face_names)
    action_yaw = yaw_to_face_z(Animation.positions_global(raw_template)[0], tail, head, Quaternions)
    destination = decode_feature_rotations(
        data,
        parents,
        parameters["tpos_rotations"],
        parameters["rest_align"],
        Quaternions,
        rotation_6d_to_matrix_np,
    )
    raw_rotations = destination.copy()
    q_yaw = repeat_quaternion(action_yaw, (len(data),), Quaternions)
    q_roll = repeat_quaternion(parameters["roll"], (len(data),), Quaternions)
    raw_rotations[:, 0] = -q_yaw * (-q_roll * destination[:, 0])
    raw_positions = np.tile(raw_template.positions[0], (len(data), 1, 1))
    recovered_root_positions = recover_root_positions(data, parameters, action_yaw, Quaternions, rotation_6d_to_matrix_np)
    template_root_anchor = raw_template.positions[0, 0] - recovered_root_positions[0]
    raw_positions[:, 0] = recovered_root_positions + template_root_anchor
    raw_anim = Animation.Animation(raw_rotations, raw_positions, raw_template.orients.copy(), raw_template.offsets.copy(), parents.copy())
    diagnostics = {
        "decoded_frames": len(data),
        "joints": len(names),
        "pruned_helper_joint_indices": excluded,
        "raw_template_pruned_helper_joint_indices": template_excluded,
        "scale_factor": parameters["scale_factor"],
        "ground_height": parameters["ground_height"],
        "rest_roll_degrees": parameters["rest_roll_degrees"],
        "scale_joint_count": parameters["scale_joint_count"],
        "ground_joint_count": parameters["ground_joint_count"],
        "action_initial_yaw_degrees": math.degrees(2.0 * math.atan2(float(action_yaw.qs[0, 2]), float(action_yaw.qs[0, 0]))),
        "template_root_anchor": [float(value) for value in template_root_anchor],
        "leaf_rotation_tokens_unavailable": int(sum(1 for joint in range(len(names)) if joint not in set(parents[parents >= 0]))),
    }
    return raw_anim, names, frame_time, diagnostics, BVH


def hide_non_lod0_meshes() -> list[bpy.types.Object]:
    meshes = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        keep = "_L0:" in obj.name
        obj.hide_render = not keep
        obj.hide_viewport = not keep
        if keep:
            meshes.append(obj)
    return meshes


def bounds_world(meshes: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    if not points:
        return Vector((-1.0, -1.0, -1.0)), Vector((1.0, 1.0, 1.0))
    return Vector(tuple(min(point[i] for point in points) for i in range(3))), Vector(tuple(max(point[i] for point in points) for i in range(3)))


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def material(name: str, color: tuple[float, float, float, float], roughness: float) -> bpy.types.Material:
    value = bpy.data.materials.new(name)
    value.use_nodes = True
    nodes = value.node_tree.nodes
    links = value.node_tree.links
    nodes.clear()
    node = nodes.new(type="ShaderNodeBsdfPrincipled")
    output = nodes.new(type="ShaderNodeOutputMaterial")
    node.inputs["Base Color"].default_value = color
    node.inputs["Roughness"].default_value = roughness
    links.new(node.outputs["BSDF"], output.inputs["Surface"])
    return value


def add_preview_axis_triad(parent: bpy.types.Object, camera: bpy.types.Object, location: Vector, extent: float) -> None:
    """Add a labelled scene-world triad that follows root translation only."""
    length = extent * 0.28
    shaft_length = length * 0.74
    shaft_radius = max(extent * 0.012, 0.006)
    cone_radius = shaft_radius * 2.7
    cone_length = length - shaft_length
    labels = (("+X", Vector((1.0, 0.0, 0.0)), (0.9, 0.08, 0.06, 1.0)),
              ("+Y", Vector((0.0, 1.0, 0.0)), (0.08, 0.85, 0.16, 1.0)),
              ("+Z", Vector((0.0, 0.0, 1.0)), (0.1, 0.38, 0.95, 1.0)))
    origin_material = material("preview_axis_origin", (0.92, 0.92, 0.95, 1.0), 0.35)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=10, radius=shaft_radius * 1.7, location=location)
    origin = bpy.context.object
    origin.name = "preview_world_axes_origin"
    origin.parent = parent
    origin.location = location
    origin.data.materials.append(origin_material)
    for label, direction, color in labels:
        axis_material = material(f"preview_axis_{label}", color, 0.3)
        shaft_center = location + direction * (shaft_length * 0.5)
        bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=shaft_radius, depth=shaft_length, location=shaft_center)
        shaft = bpy.context.object
        shaft.name = f"preview_world_axis_{label}_shaft"
        shaft.parent = parent
        shaft.location = shaft_center
        shaft.rotation_euler = Vector((0.0, 0.0, 1.0)).rotation_difference(direction).to_euler()
        shaft.data.materials.append(axis_material)
        tip_center = location + direction * (shaft_length + cone_length * 0.5)
        bpy.ops.mesh.primitive_cone_add(
            vertices=16,
            radius1=cone_radius,
            radius2=0.0,
            depth=cone_length,
            location=tip_center,
        )
        tip = bpy.context.object
        tip.name = f"preview_world_axis_{label}_tip"
        tip.parent = parent
        tip.location = tip_center
        tip.rotation_euler = Vector((0.0, 0.0, 1.0)).rotation_difference(direction).to_euler()
        tip.data.materials.append(axis_material)
        bpy.ops.object.text_add(location=location + direction * (length + shaft_radius * 4.0))
        text = bpy.context.object
        text.name = f"preview_world_axis_{label}_label"
        text.parent = parent
        text.location = location + direction * (length + shaft_radius * 4.0)
        text.data.body = label
        text.data.align_x = "CENTER"
        text.data.align_y = "CENTER"
        text.data.size = length * 0.17
        text.data.extrude = shaft_radius * 0.22
        text.data.materials.append(axis_material)
        text.rotation_euler = camera.rotation_euler


def look_at_camera(camera: bpy.types.Object, target: Vector, world_up: Vector) -> None:
    """Orient a camera without a Track To singularity when looking along Y."""
    forward = (target - camera.location).normalized()
    right = forward.cross(world_up).normalized()
    up = right.cross(forward).normalized()
    camera.rotation_euler = Matrix((right, up, -forward)).transposed().to_euler()


def make_preview(target: bpy.types.Object, meshes: list[bpy.types.Object], start: int, end: int, args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    scene.frame_start = start
    scene.frame_end = min(end, start + args.max_frames - 1)
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 720
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.fps = args.fps
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.filepath = str(args.output_mp4)
    if scene.world is None:
        scene.world = bpy.data.worlds.new("skinning_preview_world")
    scene.world.color = (0.035, 0.04, 0.055)
    shade = material("skinning_validation_material", (0.22, 0.07, 0.025, 1.0), 0.68)
    for mesh in meshes:
        mesh.data.materials.clear()
        mesh.data.materials.append(shade)
    scene.frame_set(start)
    minimum, maximum = bounds_world(meshes)
    center = (minimum + maximum) * 0.5
    extent = max((maximum - minimum).length, 0.5)
    # PZ's ``srb`` helper can be a hierarchy root after BVH promotion but has
    # no anatomical meaning and is unsuitable for a camera-follow anchor.
    root = "def_c_root_joint" if "def_c_root_joint" in target.pose.bones else next(
        bone.name for bone in target.data.bones if bone.parent is None
    )
    root_position = target.matrix_world @ target.pose.bones[root].head
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=root_position)
    anchor = bpy.context.object
    anchor.name = "preview_root_follow_anchor"
    constraint = anchor.constraints.new(type="COPY_LOCATION")
    constraint.target = target
    constraint.subtarget = root

    # A root joint is often near the feet or hips.  Keep the camera following
    # it, but aim at the mesh-centre offset captured on the first frame so the
    # preview remains an elevated three-quarter view for every body shape.
    focus_offset = center - root_position
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=root_position)
    focus = bpy.context.object
    focus.name = "preview_mesh_focus"
    focus.parent = anchor
    focus.location = focus_offset

    bpy.ops.object.light_add(type="AREA", location=root_position)
    key = bpy.context.object
    key.parent = anchor
    key.location = Vector((extent, extent * 1.35, -extent * 0.8))
    key.data.energy = 700
    key.data.shape = "DISK"
    key.data.size = extent * 1.4
    look_at(key, center)
    bpy.ops.object.light_add(type="AREA", location=root_position)
    fill = bpy.context.object
    fill.parent = anchor
    fill.location = Vector((-extent, extent * 0.65, extent * 0.9))
    fill.data.energy = 280
    fill.data.size = extent
    look_at(fill, center)

    # This presentation convention has the animal standing on the XZ plane;
    # its anatomical dorsal direction is -Y, so the preview floor is XZ.
    bpy.ops.mesh.primitive_plane_add(
        size=extent * 8.0,
        location=root_position + Vector((0.0, extent * 0.38, 0.0)),
        rotation=(math.pi / 2.0, 0.0, 0.0),
    )
    ground = bpy.context.object
    ground.name = "preview_ground"
    ground.parent = anchor
    ground.location = Vector((0.0, extent * 0.38, 0.0))
    ground.data.materials.append(material("preview_ground", (0.02, 0.026, 0.038, 1.0), 0.95))
    bpy.ops.object.camera_add(location=root_position)
    camera = bpy.context.object
    camera.parent = anchor
    # The requested canonical view is from -Y towards +Y: animals stand on
    # XZ, face +X, and have their anatomical top along -Y.
    camera.location = focus_offset + Vector((0.0, -extent * 3.25, 0.0))
    camera.data.lens = 70
    look_at_camera(camera, focus_offset, Vector((0.0, 0.0, 1.0)))
    scene.camera = camera
    if args.show_world_axes:
        # Place the triad in the lower-left screen quadrant. Its parent only
        # copies root translation, so RGB arrows retain scene-world
        # orientation while the animal moves through the frame.
        bpy.context.view_layer.update()
        camera_rotation = camera.matrix_world.to_quaternion()
        screen_right = camera_rotation @ Vector((1.0, 0.0, 0.0))
        screen_up = camera_rotation @ Vector((0.0, 1.0, 0.0))
        axis_world = focus.matrix_world.translation - screen_right * (extent * 0.53) - screen_up * (extent * 0.45)
        axis_local = anchor.matrix_world.inverted() @ axis_world
        add_preview_axis_triad(
            anchor,
            camera,
            axis_local,
            extent,
        )


def mute_constraints(rig: bpy.types.Object) -> int:
    count = 0
    for pose_bone in rig.pose.bones:
        for pose_bone_constraint in pose_bone.constraints:
            pose_bone_constraint.mute = True
            count += 1
    return count


def parent_first_names(rig: bpy.types.Object, names: list[str]) -> list[str]:
    pending = set(names)
    ordered = []
    while pending:
        ready = [
            name for name in pending
            if rig.data.bones[name].parent is None or rig.data.bones[name].parent.name not in pending
        ]
        if not ready:
            raise RuntimeError("Armature hierarchy contains an unexpected cycle.")
        ordered.extend(sorted(ready))
        pending.difference_update(ready)
    return ordered


def map_rest_relative_pose(source: bpy.types.Object, target: bpy.types.Object, shared: list[str]) -> None:
    """Map BVH global rest deltas onto the MS2 rig in its native local basis.

    ``srb`` only exists as a Planet Zoo export helper.  The AnyTop decoder
    intentionally drops it, so it remains at the MS2 rest pose while the true
    deformation hierarchy is mapped parent-first.
    """
    source_matrices = {name: source.pose.bones[name].matrix.copy() for name in shared}
    desired = {
        name: (
            source_matrices[name]
            @ source.data.bones[name].matrix_local.inverted()
            @ target.data.bones[name].matrix_local
        )
        for name in shared
    }
    shared_set = set(shared)
    for bone in target.data.bones:
        if bone.name not in shared_set:
            target.pose.bones[bone.name].matrix_basis = Matrix.Identity(4)
    for name in parent_first_names(target, shared):
        bone = target.data.bones[name]
        parent = bone.parent
        parent_rest = parent.matrix_local if parent else Matrix.Identity(4)
        parent_pose = desired[parent.name] if parent and parent.name in desired else parent_rest
        rest_local = parent_rest.inverted() @ bone.matrix_local
        destination = target.pose.bones[name]
        destination.rotation_mode = "QUATERNION"
        destination.matrix_basis = rest_local.inverted() @ parent_pose.inverted() @ desired[name]
    bpy.context.view_layer.update()


def embed_driver_note(target: bpy.types.Object, source: bpy.types.Object) -> None:
    text = bpy.data.texts.new("README_enable_anytop_npy_mesh_driver.py")
    text.write(
        "# This scene was rendered with a procedural AnyTop NPY -> BVH -> MS2 pose driver.\n"
        "# Re-run tools/planetzoo/build_planetzoo_anytop_npy_skinning_poc.py for playback.\n"
        f"# Target rig: {target.name}\n# Decoded BVH source: {source.name}\n"
    )


def verify_not_removed(motion_path: Path, removal_list: Path | None) -> bool:
    if removal_list is None:
        return False
    removed = {line.strip() for line in removal_list.read_text(encoding="utf-8").splitlines() if line.strip()}
    if motion_path.name in removed:
        raise RuntimeError(f"Refusing to skin a rotation-QC-removed motion: {motion_path.name}")
    return True


def main() -> None:
    args = parse_args()
    required = [
        args.cobra_tools / "__init__.py",
        args.motion_lib / "BVH.py",
        args.ms2_path,
        args.manis_path,
        args.motion_path,
        args.cond_path or args.full_skeleton_path,
        args.tpose_bvh,
        args.raw_template_bvh,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    qc_checked = verify_not_removed(args.motion_path, args.qc_remove_filenames)
    data = np.load(args.motion_path, allow_pickle=False)
    if data.ndim != 3 or data.shape[-1] != 13:
        raise RuntimeError(f"Expected [frames, joints, 13], got {data.shape}")
    cond = load_condition(args)
    raw_anim, names, frame_time, diagnostics, BVH = reconstruct_raw_animation(
        data, cond, args.object_name, args.tpose_bvh, args.raw_template_bvh, args.face_joints, args.motion_lib
    )
    args.output_raw_bvh.parent.mkdir(parents=True, exist_ok=True)
    BVH.save(str(args.output_raw_bvh), raw_anim, names, frametime=frame_time)

    clear_scene()
    register_cobra(args.cobra_tools)
    from plugin import import_manis, import_ms2  # pylint: disable=import-outside-toplevel

    import_ms2.load(reporter=Reporter(), filepath=str(args.ms2_path), merge_vertices=False)
    target = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE")
    target.name = "game_deform_rig"
    # Loading the matching MANIS initialises the game rig exactly as Cobra
    # Tools expects.  The decoded BVH supplies the actual animation instead.
    import_manis.load(reporter=Reporter(), filepath=str(args.manis_path), disable_ik=False)
    target.animation_data.action = None
    muted_constraints = mute_constraints(target)
    before = {obj.name for obj in bpy.context.scene.objects}
    bpy.ops.import_anim.bvh(filepath=str(args.output_raw_bvh), axis_forward="Y", axis_up="Z")
    source = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE" and obj.name not in before)
    source.name = "decoded_raw_bvh_source"
    if source.animation_data is None or source.animation_data.action is None:
        raise RuntimeError("Decoded BVH import produced no action.")
    # The decoder prunes ``srb``.  Keep it in its native rest state on the
    # target too; it has no mesh vertex group and only carries exporter state.
    shared = [
        bone.name for bone in target.data.bones
        if bone.name in source.pose.bones and bone.name.lower() != "srb"
    ]
    start, end = (int(math.floor(value)) for value in source.animation_data.action.frame_range)
    source.hide_render = True
    # Keep the source armature evaluated in the dependency graph.  Hiding it
    # in the viewport can freeze imported BVH action evaluation in background
    # Blender renders, even though it has no renderable geometry.
    source.hide_viewport = False
    lod0_meshes = hide_non_lod0_meshes()
    make_preview(target, lod0_meshes, start, end, args)
    # The preview plane is useful for interactive framing, but the mesh
    # validation render should remain uncluttered and match the raw-BVH audit.
    ground = bpy.data.objects.get("preview_ground")
    if ground:
        ground.hide_render = True
        ground.hide_viewport = True

    def on_frame_change(_scene):
        map_rest_relative_pose(source, target, shared)

    # BVH evaluation requires a transition onto its first frame; otherwise
    # Blender can retain the rest pose when rendering frame one.
    bpy.context.scene.frame_set(start - 1)
    bpy.context.scene.frame_set(start)
    map_rest_relative_pose(source, target, shared)
    embed_driver_note(target, source)

    for path in [args.output_blend, args.output_mp4, args.output_report]:
        path.parent.mkdir(parents=True, exist_ok=True)
    if args.debug_frame_dir:
        args.debug_frame_dir.mkdir(parents=True, exist_ok=True)
        debug_end = min(end, start + args.max_frames - 1)
        debug_frames = sorted({start, (start + debug_end) // 2, debug_end})
        bpy.context.scene.render.image_settings.file_format = "PNG"
        for frame in debug_frames:
            bpy.context.scene.frame_set(frame - 1)
            bpy.context.scene.frame_set(frame)
            map_rest_relative_pose(source, target, shared)
            bpy.context.scene.render.filepath = str(args.debug_frame_dir / f"frame_{frame:04d}.png")
            bpy.ops.render.render(write_still=True)
        bpy.context.scene.render.image_settings.file_format = "FFMPEG"
        bpy.context.scene.render.filepath = str(args.output_mp4)
        bpy.context.scene.frame_set(start - 1)
        bpy.context.scene.frame_set(start)
        map_rest_relative_pose(source, target, shared)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output_blend))
    bpy.context.scene.frame_set(start - 1)
    bpy.app.handlers.frame_change_post.append(on_frame_change)
    bpy.ops.render.render(animation=True)
    report = {
        "motion_path": str(args.motion_path),
        "object_name": args.object_name,
        "source_ms2": str(args.ms2_path),
        "source_manis": str(args.manis_path),
        "source_tpose_bvh": str(args.tpose_bvh),
        "condition_source": str(args.full_skeleton_path or args.cond_path),
        "raw_template_bvh": str(args.raw_template_bvh),
        "reconstructed_raw_bvh": str(args.output_raw_bvh),
        "output_blend": str(args.output_blend),
        "output_mp4": str(args.output_mp4),
        "qc_removal_list_checked": qc_checked,
        "shared_bones": len(shared),
        "target_bones": len(target.data.bones),
        "muted_constraints": muted_constraints,
        "bvh_import_axes": {"forward": "Y", "up": "Z"},
        "transfer": "source_pose @ inverse(source_rest) @ target_rest, converted to target local matrix_basis parent-first",
        "mode": "procedural_frame_change_post",
        "frames": [start, end],
        "lod0_meshes": [mesh.name for mesh in lod0_meshes],
        "debug_frame_dir": str(args.debug_frame_dir) if args.debug_frame_dir else None,
        "decoder": diagnostics,
    }
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
