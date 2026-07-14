"""Render an MS2 mesh with a procedural raw-BVH deformation driver.

This is the robust Blender 4.5 route for the Planet Zoo hierarchy: on every
frame, map the BVH's global rest-relative pose onto the MS2 deformation rig.
It avoids serialising local F-curves across the deliberately reparented BVH
``srb`` helper while preserving the game's MS2 inverse-bind basis.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--cobra-tools", required=True, type=Path)
    parser.add_argument("--ms2-path", required=True, type=Path)
    parser.add_argument("--manis-path", required=True, type=Path)
    parser.add_argument("--raw-bvh", required=True, type=Path)
    parser.add_argument("--output-blend", required=True, type=Path)
    parser.add_argument("--output-mp4", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--debug-frame-dir", type=Path)
    parser.add_argument("--max-frames", type=int, default=72)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--show-world-axes", action="store_true")
    return parser.parse_args(argv)


def load_helpers():
    path = Path(__file__).with_name("build_planetzoo_anytop_npy_skinning_poc.py")
    spec = importlib.util.spec_from_file_location("planetzoo_runtime_skin_helpers", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_raw_bvh(path: Path, known_objects: set[str]):
    bpy.ops.import_anim.bvh(filepath=str(path), axis_forward="Y", axis_up="Z")
    source = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE" and obj.name not in known_objects)
    if source.animation_data is None or source.animation_data.action is None:
        raise RuntimeError("BVH import did not create an animation action.")
    source.name = "raw_bvh_motion_source"
    return source


def mute_constraints(rig: bpy.types.Object) -> int:
    count = 0
    for pose_bone in rig.pose.bones:
        for constraint in pose_bone.constraints:
            constraint.mute = True
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


def map_pose(source: bpy.types.Object, target: bpy.types.Object, ordered: list[str]) -> None:
    source_matrices = {name: source.pose.bones[name].matrix.copy() for name in ordered}
    desired = {
        name: (
            source_matrices[name]
            @ source.data.bones[name].matrix_local.inverted()
            @ target.data.bones[name].matrix_local
        )
        for name in ordered
    }
    ordered_set = set(ordered)
    for bone in target.data.bones:
        if bone.name not in ordered_set:
            target.pose.bones[bone.name].matrix_basis = Matrix.Identity(4)
    for name in parent_first_names(target, ordered):
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
    text = bpy.data.texts.new("README_enable_raw_bvh_mesh_driver.py")
    text.write(
        "# This scene was rendered with a procedural BVH -> MS2 pose driver.\n"
        "# Re-run tools/planetzoo/drive_ms2_mesh_from_raw_bvh_runtime.py for playback.\n"
        f"# Target rig: {target.name}\n# BVH source: {source.name}\n"
    )


def configure_static_preview(helpers, meshes: list[bpy.types.Object]) -> None:
    """Use a mesh-centred camera: ``srb`` is not a reliable follow target."""
    minimum, maximum = helpers.bounds_world(meshes)
    center = (minimum + maximum) * 0.5
    extent = max((maximum - minimum).length, 0.5)
    camera = bpy.context.scene.camera
    if camera:
        camera.parent = None
        camera.constraints.clear()
        # Match the canonical preview: camera sits on -Y and looks towards
        # +Y while animals stand on the XZ plane.
        camera.location = center + Vector((0.0, -extent * 3.25, 0.0))
        camera.data.lens = 70
        helpers.look_at_camera(camera, center, Vector((0.0, 0.0, 1.0)))
    lights = [obj for obj in bpy.context.scene.objects if obj.type == "LIGHT"]
    if lights:
        lights[0].parent = None
        lights[0].location = center + Vector((extent, extent * 1.35, -extent * 0.8))
        helpers.look_at(lights[0], center)
    if len(lights) > 1:
        lights[1].parent = None
        lights[1].location = center + Vector((-extent, extent * 0.65, extent * 0.9))
        helpers.look_at(lights[1], center)
    anchor = bpy.data.objects.get("preview_root_follow_anchor")
    if anchor:
        anchor.hide_render = True
        anchor.hide_viewport = True


def main() -> None:
    args = parse_args()
    for path in [args.cobra_tools / "__init__.py", args.ms2_path, args.manis_path, args.raw_bvh]:
        if not path.is_file():
            raise FileNotFoundError(path)
    helpers = load_helpers()
    helpers.clear_scene()
    helpers.register_cobra(args.cobra_tools)
    from plugin import import_manis, import_ms2  # pylint: disable=import-outside-toplevel

    import_ms2.load(reporter=helpers.Reporter(), filepath=str(args.ms2_path), merge_vertices=False)
    target = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE")
    # MANIS initialises the pose-bone animation state used by the game rig.
    # Its imported actions are deliberately cleared before BVH driving.
    import_manis.load(reporter=helpers.Reporter(), filepath=str(args.manis_path), disable_ik=False)
    target.animation_data.action = None
    muted = mute_constraints(target)
    known_objects = {obj.name for obj in bpy.context.scene.objects}
    source = import_raw_bvh(args.raw_bvh, known_objects)
    # Match AnyTop's decoder: ``srb`` is an exporter helper, not an animated
    # model joint.  It stays in the MS2 rest pose as the parent of the true
    # deformation hierarchy.
    shared = [
        bone.name for bone in target.data.bones
        if bone.name in source.pose.bones and bone.name.lower() != "srb"
    ]
    # Keep MS2's original bone order for the remaining shared joints.
    ordered = shared
    start, end = (int(math.floor(value)) for value in source.animation_data.action.frame_range)
    meshes = helpers.hide_non_lod0_meshes()
    helpers.make_preview(target, meshes, start, end, args)
    ground = bpy.data.objects.get("preview_ground")
    if ground:
        ground.hide_render = True
        ground.hide_viewport = True

    def on_frame_change(_scene):
        map_pose(source, target, ordered)

    # The BVH importer needs a frame transition to evaluate its first action
    # sample; setting an already-current frame leaves it at its rest pose.
    bpy.context.scene.frame_set(start - 1)
    bpy.context.scene.frame_set(start)
    map_pose(source, target, ordered)
    embed_driver_note(target, source)
    for path in [args.output_blend, args.output_mp4, args.output_report]:
        path.parent.mkdir(parents=True, exist_ok=True)
    if args.debug_frame_dir:
        args.debug_frame_dir.mkdir(parents=True, exist_ok=True)
        debug_frames = sorted({start, (start + min(end, start + args.max_frames - 1)) // 2, min(end, start + args.max_frames - 1)})
        bpy.context.scene.render.image_settings.file_format = "PNG"
        for frame in debug_frames:
            bpy.context.scene.frame_set(frame - 1)
            bpy.context.scene.frame_set(frame)
            map_pose(source, target, ordered)
            bpy.context.scene.render.filepath = str(args.debug_frame_dir / f"frame_{frame:04d}.png")
            bpy.ops.render.render(write_still=True)
        bpy.context.scene.render.image_settings.file_format = "FFMPEG"
        bpy.context.scene.render.filepath = str(args.output_mp4)
        bpy.context.scene.frame_set(start - 1)
        bpy.context.scene.frame_set(start)
        map_pose(source, target, ordered)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output_blend))
    bpy.context.scene.frame_set(start - 1)
    bpy.app.handlers.frame_change_post.append(on_frame_change)
    bpy.ops.render.render(animation=True)
    report = {
        "ms2_path": str(args.ms2_path),
        "manis_path": str(args.manis_path),
        "raw_bvh": str(args.raw_bvh),
        "bvh_import_axes": {"forward": "Y", "up": "Z"},
        "transfer": "source_pose @ inverse(source_rest) @ target_rest",
        "mode": "procedural_frame_change_post",
        "frames": [start, end],
        "shared_bones": len(shared),
        "target_bones": len(target.data.bones),
        "muted_constraints": muted,
        "lod0_meshes": [mesh.name for mesh in meshes],
        "debug_frame_dir": str(args.debug_frame_dir) if args.debug_frame_dir else None,
        "output_blend": str(args.output_blend),
        "output_mp4": str(args.output_mp4),
    }
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
