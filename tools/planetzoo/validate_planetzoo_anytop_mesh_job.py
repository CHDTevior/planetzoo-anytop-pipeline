"""Validate one AnyTop NPY -> raw BVH -> Planet Zoo MS2 skinning job.

This is intentionally render-free.  It evaluates the real LOD0 meshes at the
first, middle, and final animation frames, so a 311-topology audit can check
actual skinning without creating hundreds of large preview videos.
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
from mathutils import Matrix


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--cobra-tools", required=True, type=Path)
    parser.add_argument("--motion-lib", required=True, type=Path)
    return parser.parse_args(argv)


def load_skinning_helpers():
    path = Path(__file__).with_name("build_planetzoo_anytop_npy_skinning_poc.py")
    spec = importlib.util.spec_from_file_location("planetzoo_anytop_skinning_helpers", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load skinning helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_raw_bvh(path: Path, known_objects: set[str]) -> bpy.types.Object:
    bpy.ops.import_anim.bvh(filepath=str(path), axis_forward="Y", axis_up="Z")
    source = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE" and obj.name not in known_objects)
    if source.animation_data is None or source.animation_data.action is None:
        raise RuntimeError("Decoded BVH import did not produce an action.")
    source.name = "decoded_raw_bvh_source"
    return source


def desired_global_matrices(source: bpy.types.Object, target: bpy.types.Object, shared: list[str]) -> dict[str, Matrix]:
    return {
        name: (
            source.pose.bones[name].matrix.copy()
            @ source.data.bones[name].matrix_local.inverted()
            @ target.data.bones[name].matrix_local
        )
        for name in shared
    }


def max_matrix_error(target: bpy.types.Object, expected: dict[str, Matrix]) -> float:
    return max(
        max(abs(target.pose.bones[name].matrix[row][col] - matrix[row][col]) for row in range(4) for col in range(4))
        for name, matrix in expected.items()
    )


def lod0_meshes() -> list[bpy.types.Object]:
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and "_L0:" in obj.name]


def mesh_summary(meshes: list[bpy.types.Object], target: bpy.types.Object) -> dict:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    result = {}
    for obj in meshes:
        evaluated = obj.evaluated_get(depsgraph)
        evaluated_mesh = evaluated.to_mesh()
        try:
            values = np.empty(len(evaluated_mesh.vertices) * 3, dtype=np.float64)
            evaluated_mesh.vertices.foreach_get("co", values)
            values = values.reshape((-1, 3))
            finite = bool(np.isfinite(values).all())
            result[obj.name] = {
                "vertices": int(len(values)),
                "finite": finite,
                "min": [float(value) for value in values.min(axis=0)],
                "max": [float(value) for value in values.max(axis=0)],
                "centroid": [float(value) for value in values.mean(axis=0)],
                "has_target_armature_modifier": any(
                    modifier.type == "ARMATURE" and modifier.object == target for modifier in obj.modifiers
                ),
            }
        finally:
            evaluated.to_mesh_clear()
    return result


def deform_bones(meshes: list[bpy.types.Object], target: bpy.types.Object) -> list[str]:
    groups = {group.name for mesh in meshes for group in mesh.vertex_groups}
    return sorted(name for name in groups if name in target.data.bones)


def main() -> None:
    args = parse_args()
    job = json.loads(args.job.read_text(encoding="utf-8"))
    output_path = Path(job["output_report"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    helpers = load_skinning_helpers()

    try:
        required = [
            args.cobra_tools / "__init__.py",
            args.motion_lib / "BVH.py",
            Path(job["motion_path"]),
            Path(job["cond_path"]),
            Path(job["tpose_bvh"]),
            Path(job["raw_template_bvh"]),
            Path(job["ms2_path"]),
            Path(job["manis_path"]),
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

        data = np.load(job["motion_path"], allow_pickle=False)
        if data.ndim != 3 or data.shape[-1] != 13 or not np.isfinite(data).all():
            raise RuntimeError(f"Invalid motion tensor: shape={data.shape}, finite={np.isfinite(data).all()}")
        cond = np.load(job["cond_path"], allow_pickle=True).item()
        raw_anim, names, frame_time, decoder, bvh_module = helpers.reconstruct_raw_animation(
            data,
            cond,
            job["object_name"],
            Path(job["tpose_bvh"]),
            Path(job["raw_template_bvh"]),
            job.get("face_joints", ["def_c_hips_joint", "def_c_chest_joint"]),
            args.motion_lib,
        )
        decoded_path = Path(job["decoded_bvh_path"])
        decoded_path.parent.mkdir(parents=True, exist_ok=True)
        bvh_module.save(str(decoded_path), raw_anim, names, frametime=frame_time)

        helpers.clear_scene()
        helpers.register_cobra(args.cobra_tools)
        from plugin import import_manis, import_ms2  # pylint: disable=import-outside-toplevel

        import_ms2.load(reporter=helpers.Reporter(), filepath=job["ms2_path"], merge_vertices=False)
        target = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE")
        target.name = "game_deform_rig"
        import_manis.load(reporter=helpers.Reporter(), filepath=job["manis_path"], disable_ik=False)
        if target.animation_data:
            target.animation_data.action = None
        muted_constraints = helpers.mute_constraints(target)
        known_objects = {obj.name for obj in bpy.context.scene.objects}
        source = import_raw_bvh(decoded_path, known_objects)
        shared = [
            bone.name
            for bone in target.data.bones
            if bone.name in source.pose.bones and bone.name.lower() != "srb"
        ]
        if not shared:
            raise RuntimeError("No shared non-srb bones between decoded BVH and MS2 rig.")
        meshes = lod0_meshes()
        if not meshes:
            raise RuntimeError("MS2 import produced no LOD0 meshes.")
        deform = deform_bones(meshes, target)
        missing_deform = [name for name in deform if name not in shared and name.lower() != "srb"]
        start, end = (int(math.floor(value)) for value in source.animation_data.action.frame_range)
        frames = sorted({start, (start + end) // 2, end})
        samples = []
        for frame in frames:
            bpy.context.scene.frame_set(frame - 1)
            bpy.context.scene.frame_set(frame)
            expected = desired_global_matrices(source, target, shared)
            helpers.map_rest_relative_pose(source, target, shared)
            matrix_error = max_matrix_error(target, expected)
            geometry = mesh_summary(meshes, target)
            samples.append({"frame": frame, "matrix_max_abs_error": matrix_error, "meshes": geometry})

        all_finite = all(mesh["finite"] for sample in samples for mesh in sample["meshes"].values())
        all_bound = all(mesh["has_target_armature_modifier"] for sample in samples for mesh in sample["meshes"].values())
        max_matrix = max(sample["matrix_max_abs_error"] for sample in samples)
        report = {
            "status": "pass" if all_finite and all_bound and not missing_deform and max_matrix <= 1e-4 else "fail",
            "object_name": job["object_name"],
            "motion_path": job["motion_path"],
            "raw_template_bvh": job["raw_template_bvh"],
            "decoded_bvh_path": str(decoded_path),
            "ms2_path": job["ms2_path"],
            "manis_path": job["manis_path"],
            "frames": frames,
            "motion_shape": [int(value) for value in data.shape],
            "target_bones": len(target.data.bones),
            "shared_bones": len(shared),
            "deform_bones": len(deform),
            "missing_non_srb_deform_bones": missing_deform,
            "muted_constraints": muted_constraints,
            "max_matrix_abs_error": max_matrix,
            "all_mesh_vertices_finite": all_finite,
            "all_lod0_meshes_bound_to_target": all_bound,
            "samples": samples,
            "decoder": decoder,
        }
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({key: report[key] for key in ["status", "object_name", "shared_bones", "deform_bones", "max_matrix_abs_error"]}, indent=2))
        if report["status"] != "pass":
            raise RuntimeError(f"Mesh validation failed: {output_path}")
    except Exception as exc:
        failure = {"status": "error", "object_name": job.get("object_name"), "error": repr(exc)}
        output_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
