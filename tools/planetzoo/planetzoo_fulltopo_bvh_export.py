"""
Export full-topology Planet Zoo armature animations to raw BVH files.

Run with Blender, for example:
  blender.exe -b --python scripts/planetzoo_fulltopo_bvh_export.py -- \
      --cobra-tools H:/path/to/cobra-tools \
      --input-root H:/AniMo4D_work/01_ovl_extracted \
      --output-root H:/AniMo4D_work/05_fulltopo_raw_bvh \
      --objects Aardvark_Female.ovl Cheetah_Female.ovl \
      --max-actions 3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import traceback
from pathlib import Path

import bpy


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--cobra-tools", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--objects", nargs="*", default=None)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--only-manis-contains", default=None)
    parser.add_argument("--no-root-manifest", action="store_true", help="Skip writing output_root summary/manifest files.")
    return parser.parse_args(argv)


def safe_name(value: str) -> str:
    value = value.replace("@", "_")
    value = value.replace(".", "_")
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unnamed"


def action_short_name(ms2_stem: str, action_name: str) -> str:
    animal_key = ms2_stem.rstrip("_")
    for prefix in (animal_key + "_", animal_key + "@"):
        if action_name.startswith(prefix):
            return action_name[len(prefix) :]
    return action_name


def safe_clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for collection in list(bpy.data.collections):
        if collection.name != "Scene Collection":
            bpy.data.collections.remove(collection)
    for datablock in [
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.textures,
        bpy.data.images,
        bpy.data.armatures,
        bpy.data.actions,
    ]:
        for item in list(datablock):
            if not item.users:
                datablock.remove(item)


class Reporter:
    def __call__(self, message_type, message):
        print(f"{message_type}: {message}")

    def show_info(self, message):
        print(f"INFO: {message}")

    def show_warning(self, message):
        print(f"WARNING: {message}")

    def show_error(self, message):
        print(f"ERROR: {message}")


def find_armature():
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            return obj
    return None


def bone_vector(vec):
    return [float(vec.x), float(vec.y), float(vec.z)]


def write_skeleton_meta(armature, output_path: Path) -> None:
    bones = []
    for idx, bone in enumerate(armature.data.bones):
        bones.append(
            {
                "index": idx,
                "name": bone.name,
                "parent": bone.parent.name if bone.parent else None,
                "parent_index": list(armature.data.bones).index(bone.parent) if bone.parent else -1,
                "head_local": bone_vector(bone.head_local),
                "tail_local": bone_vector(bone.tail_local),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"bones": bones}, indent=2), encoding="utf-8")


def export_bvh(armature, action, output_path: Path) -> None:
    frame_start = int(action.frame_range[0])
    frame_end = int(action.frame_range[1])
    scene = bpy.context.scene
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    scene.render.fps = 20
    if armature.animation_data is None:
        armature.animation_data_create()
    armature.animation_data.action = action
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_anim.bvh(
        filepath=str(output_path),
        frame_start=frame_start,
        frame_end=frame_end,
        global_scale=1.0,
        rotate_mode="NATIVE",
        root_transform_only=False,
    )
    promote_single_child_root(output_path)


def export_rest_bvh(armature, output_path: Path) -> None:
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 2
    scene.render.fps = 20
    if armature.animation_data is not None:
        armature.animation_data.action = None
    pose_position = armature.data.pose_position
    armature.data.pose_position = "REST"
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_anim.bvh(
        filepath=str(output_path),
        frame_start=1,
        frame_end=2,
        global_scale=1.0,
        rotate_mode="NATIVE",
        root_transform_only=False,
    )
    armature.data.pose_position = pose_position
    promote_single_child_root(output_path)


def promote_single_child_root(bvh_path: Path) -> bool:
    """Remove Blender's zero-channel wrapper and keep a single BVH tree.

    Blender exports an armature wrapper like ROOT __0 with zero channels. Some
    Planet Zoo skeletons also put helper bones (for example ``srb``) as siblings
    of the real skeleton root under that wrapper. The Motion BVH loader used by
    AnyTop cannot read zero-channel roots and produces invalid parents when
    multiple top-level nodes remain, so we promote the first child to ROOT and
    reparent any wrapper-level siblings under it.
    """
    lines = bvh_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 8:
        return False
    if not lines[1].lstrip().startswith("ROOT "):
        return False
    if lines[4].strip() != "CHANNELS 0":
        return False
    child_match = re.match(r"\s*JOINT\s+(.+?)\s*$", lines[5])
    if not child_match or lines[6].strip() != "{":
        return False

    depth = 0
    child_close = None
    for idx in range(6, len(lines)):
        stripped = lines[idx].strip()
        if stripped == "{":
            depth += 1
        elif stripped == "}":
            depth -= 1
            if depth == 0:
                child_close = idx
                break
    if child_close is None:
        return False

    try:
        motion_idx = next(idx for idx, line in enumerate(lines) if line.strip() == "MOTION")
    except StopIteration:
        return False
    if child_close >= motion_idx - 1:
        return False
    if lines[motion_idx - 1].strip() != "}":
        return False

    fixed = [lines[0], f"ROOT {child_match.group(1)}"]
    fixed.extend(lines[6:child_close])
    for line in lines[child_close + 1 : motion_idx - 1]:
        fixed.append("\t" + line if line.strip() else line)
    fixed.append(lines[child_close])
    fixed.extend(lines[motion_idx:])
    bvh_path.write_text("\n".join(fixed) + "\n", encoding="utf-8")
    print(f"PROMOTED_ROOT {bvh_path.name}: removed {lines[1].strip()}")
    return True


def process_object_dir(object_dir: Path, output_root: Path, import_ms2, import_manis, reporter, max_actions, only_manis_contains):
    ms2_files = sorted(object_dir.glob("*.ms2"))
    manis_files = sorted(object_dir.glob("*.manis"))
    if not ms2_files or not manis_files:
        print(f"SKIP {object_dir.name}: ms2={len(ms2_files)} manis={len(manis_files)}")
        return 0, output_root / safe_name(object_dir.name) / "export_manifest.jsonl"

    ms2_path = ms2_files[0]
    object_out = output_root / safe_name(object_dir.name)
    object_manifest_path = object_out / "export_manifest.jsonl"
    if object_manifest_path.exists():
        object_manifest_path.unlink()
    exported = 0

    for manis_path in manis_files:
        if only_manis_contains and only_manis_contains.lower() not in manis_path.name.lower():
            continue
        safe_clear_scene()
        print(f"LOAD {object_dir.name} :: {manis_path.name}")
        import_ms2.load(reporter=reporter, filepath=str(ms2_path))
        armature = find_armature()
        if armature is None:
            print(f"NO_ARMATURE {object_dir.name}")
            continue

        write_skeleton_meta(armature, object_out / "skeleton_meta.json")
        ms2_stem = ms2_path.stem
        animal_key = ms2_stem.rstrip("_")
        rest_bvh_name = f"{safe_name(ms2_stem)}__tpos.bvh"
        rest_bvh_path = object_out / "raw_bvhs" / rest_bvh_name
        if not rest_bvh_path.exists():
            export_rest_bvh(armature, rest_bvh_path)
            print(f"EXPORTED {rest_bvh_name}")
            rest_entry = {
                "sample_type": "tpose",
                "object_dir": object_dir.name,
                "object_key": safe_name(object_dir.stem),
                "animal_key": animal_key,
                "ms2_file": ms2_path.name,
                "manis_file": None,
                "action_name": "tpos",
                "action_short": "tpos",
                "source_motion_key": f"{animal_key}@tpos",
                "raw_bvh": str(rest_bvh_path.resolve()),
                "raw_bvh_stem": Path(rest_bvh_name).stem,
            }
            with object_manifest_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rest_entry, ensure_ascii=False) + "\n")

        import_manis.load(reporter=reporter, filepath=str(manis_path))
        action_count_for_manis = 0
        for action in list(bpy.data.actions):
            if action.id_root != "OBJECT":
                continue
            safe_action_name = safe_name(action.name)
            if ms2_stem[:-1] not in safe_action_name:
                continue
            bvh_name = f"{safe_name(ms2_stem)}__{safe_name(manis_path.stem)}__{safe_action_name}.bvh"
            export_bvh(armature, action, object_out / "raw_bvhs" / bvh_name)
            print(f"EXPORTED {bvh_name}")
            short_action = action_short_name(ms2_stem, action.name)
            manifest_entry = {
                "sample_type": "motion",
                "object_dir": object_dir.name,
                "object_key": safe_name(object_dir.stem),
                "animal_key": animal_key,
                "ms2_file": ms2_path.name,
                "manis_file": manis_path.name,
                "action_name": action.name,
                "action_short": short_action,
                "source_motion_key": f"{animal_key}@{short_action}",
                "raw_bvh": str((object_out / "raw_bvhs" / bvh_name).resolve()),
                "raw_bvh_stem": Path(bvh_name).stem,
            }
            with object_manifest_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
            exported += 1
            action_count_for_manis += 1
            if max_actions is not None and exported >= max_actions:
                return exported, object_manifest_path
        if action_count_for_manis == 0:
            print(f"NO_MATCHING_ACTIONS {object_dir.name} :: {manis_path.name}")

    return exported, object_manifest_path


def main() -> None:
    args = parse_args()
    cobra_tools = Path(args.cobra_tools)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    sys.path.insert(0, str(cobra_tools))
    spec = importlib.util.spec_from_file_location(
        "cobra_tools_addon",
        str(cobra_tools / "__init__.py"),
        submodule_search_locations=[str(cobra_tools)],
    )
    cobra_addon = importlib.util.module_from_spec(spec)
    sys.modules["cobra_tools_addon"] = cobra_addon
    spec.loader.exec_module(cobra_addon)
    cobra_addon.register()
    from plugin import import_ms2, import_manis

    bpy.ops.preferences.addon_enable(module="io_anim_bvh")
    reporter = Reporter()

    if args.objects:
        object_dirs = [input_root / name for name in args.objects]
    else:
        object_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])
    if args.max_objects is not None:
        object_dirs = object_dirs[: args.max_objects]

    summary = {}
    manifest_paths = []
    for object_dir in object_dirs:
        try:
            count, manifest_path = process_object_dir(
                object_dir,
                output_root,
                import_ms2,
                import_manis,
                reporter,
                args.max_actions,
                args.only_manis_contains,
            )
            summary[object_dir.name] = count
            if manifest_path.exists():
                manifest_paths.append(manifest_path)
        except Exception:
            print(f"FAILED {object_dir}")
            traceback.print_exc()
    output_root.mkdir(parents=True, exist_ok=True)
    if not args.no_root_manifest:
        (output_root / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with (output_root / "export_manifest.jsonl").open("w", encoding="utf-8") as out_f:
            for manifest_path in manifest_paths:
                out_f.write(manifest_path.read_text(encoding="utf-8"))
    print("SUMMARY", summary)


if __name__ == "__main__":
    main()
