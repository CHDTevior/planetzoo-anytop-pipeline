"""Assemble several saved Planet Zoo mesh previews into one Blender scene.

The input ``mesh_preview.blend`` files are produced by the verified AnyTop
motion-to-MS2 rendering route.  This utility appends only each target
``game_deform_rig`` and its L0 render meshes, keeps their baked pose, and
places the animals in a compact 2-by-2 inspection layout.  It deliberately
does not append the source camera, BVH armature, lights, debug geometry, or
physics helpers from the individual preview scenes.

Run inside Blender, for example:

    H:/blender4_5/blender.exe -b --python tools/planetzoo/assemble_mesh_preview_scene.py -- \
      --source Tiger H:/.../PZ_Bengal_Tiger_Male/mesh_preview.blend \
      --source SeaLion H:/.../PZ_California_Sea_Lion_Male/mesh_preview.blend \
      --output-blend H:/.../combined_animals_source.blend
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        nargs=2,
        metavar=("LABEL", "BLEND"),
        required=True,
        help="An animal label and a verified mesh_preview.blend. Repeat for each animal.",
    )
    parser.add_argument("--output-blend", required=True, type=Path)
    parser.add_argument("--output-preview", type=Path, help="Optional still render used to inspect the assembled scene.")
    parser.add_argument("--frame", type=int, default=1, help="Timeline frame saved in the assembled scene.")
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)


def append_preview_objects(path: Path) -> list[bpy.types.Object]:
    """Append only the deform rig and its highest-detail render meshes."""
    with bpy.data.libraries.load(str(path), link=False) as (source, destination):
        destination.objects = [
            name
            for name in source.objects
            if name == "game_deform_rig" or "_L0:" in name
        ]
    return [obj for obj in destination.objects if obj is not None]


def bounds_world(meshes: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    if not corners:
        raise RuntimeError("The appended preview did not contain any L0 render mesh.")
    minimum = Vector((min(point.x for point in corners), min(point.y for point in corners), min(point.z for point in corners)))
    maximum = Vector((max(point.x for point in corners), max(point.y for point in corners), max(point.z for point in corners)))
    return minimum, maximum


def look_at_camera(camera: bpy.types.Object, target: Vector, world_up: Vector) -> None:
    """Aim a Blender camera while keeping a chosen world direction screen-up."""
    forward = (target - camera.location).normalized()
    right = forward.cross(world_up)
    if right.length < 1e-8:
        raise RuntimeError("Camera direction is parallel to the requested screen-up direction.")
    right.normalize()
    up = right.cross(forward).normalized()
    camera.rotation_euler = Matrix((right, up, -forward)).transposed().to_euler()


def setup_world() -> None:
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    background = nodes.get("Background") or nodes.new("ShaderNodeBackground")
    output = nodes.get("World Output") or nodes.new("ShaderNodeOutputWorld")
    if not any(link.from_node == background and link.to_node == output for link in links):
        links.new(background.outputs["Background"], output.inputs["Surface"])
    background.inputs["Color"].default_value = (0.025, 0.035, 0.05, 1.0)
    background.inputs["Strength"].default_value = 0.28


def add_area_light(name: str, location: Vector, energy: float, size: float, target: Vector) -> None:
    light_data = bpy.data.lights.new(name, "AREA")
    light_data.energy = energy
    light_data.shape = "DISK"
    light_data.size = size
    light = bpy.data.objects.new(name, light_data)
    bpy.context.scene.collection.objects.link(light)
    light.location = location
    light.rotation_euler = (target - location).to_track_quat("-Z", "Y").to_euler()


def add_floor(minimum: Vector, maximum: Vector) -> None:
    center = (minimum + maximum) * 0.5
    size = max(maximum.x - minimum.x, maximum.z - minimum.z) * 1.35
    bpy.ops.mesh.primitive_plane_add(size=size, location=(center.x, maximum.y + 0.012, center.z), rotation=(math.pi / 2.0, 0.0, 0.0))
    floor = bpy.context.object
    floor.name = "inspection_ground_xz"
    material = bpy.data.materials.new("inspection_ground_material")
    material.diffuse_color = (0.075, 0.09, 0.12, 1.0)
    material.roughness = 0.82
    floor.data.materials.append(material)


def add_camera_and_lights(meshes: list[bpy.types.Object]) -> None:
    minimum, maximum = bounds_world(meshes)
    center = (minimum + maximum) * 0.5
    extent = max(maximum.x - minimum.x, maximum.y - minimum.y, maximum.z - minimum.z, 1.0)

    # Canonical inspection view: +X is toward the observer, -Y projects upward.
    camera_data = bpy.data.cameras.new("combined_preview_camera")
    camera_data.lens = 53
    camera = bpy.data.objects.new("combined_preview_camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    camera.location = center + Vector((extent * 2.9, -extent * 3.2, extent * 1.85))
    look_at_camera(camera, center, Vector((0.0, -1.0, 0.0)))
    bpy.context.scene.camera = camera

    power_scale = extent * extent
    add_area_light("key_light", center + Vector((extent * 1.5, -extent * 2.2, extent * 2.5)), 170.0 * power_scale, extent * 1.2, center)
    add_area_light("fill_light", center + Vector((-extent * 1.7, -extent * 0.8, extent * 0.6)), 90.0 * power_scale, extent, center)
    add_area_light("rim_light", center + Vector((extent * 0.4, extent * 2.4, extent * 1.4)), 120.0 * power_scale, extent * 1.1, center)
    add_floor(minimum, maximum)


def add_scene_note(sources: list[tuple[str, Path]], frame: int) -> None:
    text = bpy.data.texts.new("README_combined_mesh_preview.txt")
    text.write(
        "Combined inspection scene for verified AnyTop-to-Planet-Zoo mesh previews.\n"
        f"Saved inspection frame: {frame}\n\n"
        "Each collection contains only the target game_deform_rig plus its L0 render meshes.\n"
        "The original per-animal mesh_preview.blend files remain the motion sources.\n\n"
        "Input scenes:\n"
        + "".join(f"- {label}: {path}\n" for label, path in sources)
    )


def main() -> None:
    args = parse_args()
    sources = [(label, Path(path).resolve()) for label, path in args.source]
    for _, path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)

    clear_scene()
    scene = bpy.context.scene
    scene.frame_start = args.frame
    scene.frame_end = args.frame
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 1600
    scene.render.resolution_y = 1000
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    setup_world()

    layouts = [(-0.55, 0.52), (0.55, 0.52), (-0.55, -0.52), (0.55, -0.52)]
    loaded_groups: list[tuple[str, bpy.types.Object, list[bpy.types.Object], Vector, Vector]] = []
    for index, (label, path) in enumerate(sources):
        objects = append_preview_objects(path)
        rigs = [obj for obj in objects if obj.type == "ARMATURE"]
        meshes = [obj for obj in objects if obj.type == "MESH"]
        if len(rigs) != 1 or not meshes:
            raise RuntimeError(f"{path}: expected one game_deform_rig and one or more L0 meshes; got rigs={len(rigs)}, meshes={len(meshes)}")
        collection = bpy.data.collections.new(label)
        scene.collection.children.link(collection)
        for obj in objects:
            collection.objects.link(obj)
        root = bpy.data.objects.new(f"{label}_inspection_root", None)
        root.empty_display_type = "PLAIN_AXES"
        root.empty_display_size = 0.35
        collection.objects.link(root)
        rigs[0].parent = root
        minimum, maximum = bounds_world(meshes)
        loaded_groups.append((label, root, meshes, minimum, maximum))

    largest_x = max(maximum.x - minimum.x for _, _, _, minimum, maximum in loaded_groups)
    largest_z = max(maximum.z - minimum.z for _, _, _, minimum, maximum in loaded_groups)
    for index, (_, root, meshes, minimum, maximum) in enumerate(loaded_groups):
        col, row = layouts[index] if index < len(layouts) else (0.0, -1.55 - 1.05 * (index - len(layouts)))
        center = (minimum + maximum) * 0.5
        root.location = Vector((col * (largest_x + 1.3), -maximum.y, row * (largest_z + 1.3))) - Vector((center.x, 0.0, center.z))
        for mesh in meshes:
            mesh.hide_render = False
            mesh.hide_viewport = False

    scene.frame_set(args.frame)
    all_meshes = [mesh for _, _, meshes, _, _ in loaded_groups for mesh in meshes]
    add_camera_and_lights(all_meshes)
    add_scene_note(sources, args.frame)
    args.output_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output_blend))
    if args.output_preview:
        args.output_preview.parent.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(args.output_preview)
        bpy.ops.render.render(write_still=True)
    print(f"ASSEMBLED sources={len(sources)} meshes={len(all_meshes)} output={args.output_blend}")


if __name__ == "__main__":
    main()
