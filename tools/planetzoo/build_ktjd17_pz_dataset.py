"""Build the audited Planet Zoo KTJD-17 corpus from freshly exported BVHs.

The exporter input must be produced by ``planetzoo_parallel_bvh_export.py``
with ``--disable-ik --fps 30``.  This script deliberately refuses legacy or
unverified BVHs so a new corpus cannot silently inherit old exports.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MOTION_LIB = REPO_ROOT / "tools" / "planetzoo" / "motion_lib"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MOTION_LIB) not in sys.path:
    sys.path.insert(0, str(MOTION_LIB))

from Animation import Animation, transforms_global, positions_global  # noqa: E402
from data_loaders.truebones.truebones_utils.motion_process import (  # noqa: E402
    get_common_features_from_T_pose,
    get_hml_aligned_anim,
)


CHANNELS = 17
POSITION_SLICE = slice(0, 3)
ROTATION_SLICE = slice(3, 9)
VELOCITY_SLICE = slice(9, 12)
CONTACT_CHANNEL = 12
ROOT_XZ_SLICE = slice(13, 15)
HEADING_SLICE = slice(15, 17)
STD_MIN = 0.05
DEFAULT_FACE_NAMES = ("def_c_hips_joint", "def_c_chest_joint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--joint-spec", type=Path, required=True)
    parser.add_argument("--caption-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-rigs", type=int)
    parser.add_argument("--max-clips-per-rig", type=int)
    parser.add_argument(
        "--allow-missing-captions",
        action="store_true",
        help="For diagnostics only. The production corpus must keep this disabled.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safe_stem(value: str) -> str:
    return "".join(char if char.isalnum() or char in "_-" else "_" for char in value)


def sha256_strings(strings: list[str]) -> str:
    payload = "\n".join(strings).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def cont6d_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Store the first two matrix columns as [c0x,c0y,c0z,c1x,c1y,c1z]."""
    return np.swapaxes(matrix[..., :, :2], -1, -2).reshape(*matrix.shape[:-2], 6)


def matrix_from_cont6d(cont6d: np.ndarray) -> np.ndarray:
    first = cont6d[..., :3]
    second = cont6d[..., 3:6]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-1)


def slerp_quaternions(first: np.ndarray, second: np.ndarray, amount: np.ndarray) -> np.ndarray:
    """Vectorized shortest-path SLERP for local BVH quaternions."""
    second = second.copy()
    dot = np.sum(first * second, axis=-1, keepdims=True)
    second[dot[..., 0] < 0.0] *= -1.0
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    amount = amount[..., None]
    linear = dot > 0.9995
    theta = np.arccos(dot)
    sine = np.sin(theta)
    left = np.sin((1.0 - amount) * theta) / np.maximum(sine, 1e-12)
    right = np.sin(amount * theta) / np.maximum(sine, 1e-12)
    output = left * first + right * second
    linear_output = (1.0 - amount) * first + amount * second
    output = np.where(linear, linear_output, output)
    return output / np.maximum(np.linalg.norm(output, axis=-1, keepdims=True), 1e-12)


def resample_aligned_animation(aligned: Animation, source_fps: int, target_fps: int) -> Animation:
    """Resample local rotations and translations before FK at the target FPS."""
    if source_fps == target_fps:
        return aligned
    duration = (len(aligned) - 1) / float(source_fps)
    target_frames = int(math.floor(duration * target_fps + 1e-8)) + 1
    sample_source = np.arange(target_frames, dtype=np.float64) * source_fps / target_fps
    left = np.floor(sample_source).astype(int)
    right = np.minimum(left + 1, len(aligned) - 1)
    alpha = sample_source - left
    rotations = slerp_quaternions(aligned.rotations.qs[left], aligned.rotations.qs[right], alpha[:, None])
    positions = (1.0 - alpha[:, None, None]) * aligned.positions[left] + alpha[:, None, None] * aligned.positions[right]
    return Animation(type(aligned.rotations)(rotations), positions, aligned.orients.copy(), aligned.offsets.copy(), aligned.parents.copy())


def selected_face_names(names: list[str]) -> tuple[str, str]:
    if all(name in names for name in DEFAULT_FACE_NAMES):
        return DEFAULT_FACE_NAMES
    lower_names = [name.lower() for name in names]
    root = names[0]
    for token in ("chest", "head", "neck", "spine"):
        for index, name in enumerate(lower_names):
            if token in name:
                return root, names[index]
    raise ValueError("Unable to find a stable heading pair in rig")


def contact_indices(names: list[str], selected_full_indices: list[int], common_foot_indices: list[int]) -> list[int]:
    full_to_selected = {full: selected for selected, full in enumerate(selected_full_indices)}
    selected = [full_to_selected[index] for index in common_foot_indices if index in full_to_selected]
    if selected:
        return sorted(set(selected))
    tokens = ("toe", "foot", "phalanx", "hoof", "ashi")
    return [index for index, name in enumerate(names) if any(token in name.lower() for token in tokens)]


class RunningStats:
    def __init__(self, joints: int) -> None:
        self.count = np.zeros((joints, CHANNELS), dtype=np.int64)
        self.sum = np.zeros((joints, CHANNELS), dtype=np.float64)
        self.sumsq = np.zeros((joints, CHANNELS), dtype=np.float64)
        self.minimum = np.full((joints, CHANNELS), np.inf, dtype=np.float64)
        self.maximum = np.full((joints, CHANNELS), -np.inf, dtype=np.float64)

    def _update(self, values: np.ndarray, joint_indices: np.ndarray, channel_slice: slice) -> None:
        if values.size == 0:
            return
        start = channel_slice.start
        stop = channel_slice.stop
        self.count[joint_indices, start:stop] += values.shape[0]
        self.sum[joint_indices, start:stop] += values.sum(axis=0, dtype=np.float64)
        self.sumsq[joint_indices, start:stop] += np.square(values, dtype=np.float64).sum(axis=0, dtype=np.float64)
        self.minimum[joint_indices, start:stop] = np.minimum(
            self.minimum[joint_indices, start:stop], values.min(axis=0)
        )
        self.maximum[joint_indices, start:stop] = np.maximum(
            self.maximum[joint_indices, start:stop], values.max(axis=0)
        )

    def update(self, motion: np.ndarray, heading_valid: np.ndarray) -> None:
        joint_indices = np.arange(motion.shape[1])
        self._update(motion[:, :, :13], joint_indices, slice(0, 13))
        root_index = np.array([0])
        self._update(motion[:, :1, 13:15], root_index, ROOT_XZ_SLICE)
        heading = motion[heading_valid, :1, 15:17]
        self._update(heading, root_index, HEADING_SLICE)

    def finish(self, supervise_mask: np.ndarray) -> dict[str, np.ndarray]:
        count = self.count
        mean = np.zeros_like(self.sum, dtype=np.float32)
        populated = count > 0
        mean[populated] = (self.sum[populated] / count[populated]).astype(np.float32)
        variance = np.zeros_like(self.sum)
        variance[populated] = self.sumsq[populated] / count[populated] - np.square(mean[populated])
        std_raw = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)
        std_eff = std_raw.copy()
        constants = populated & (std_raw <= 1e-8)
        std_eff[constants] = 1.0
        variable = populated & ~constants
        std_eff[variable & (std_eff < STD_MIN)] = STD_MIN
        std_eff[~populated] = 1.0
        minimum = self.minimum.astype(np.float32)
        maximum = self.maximum.astype(np.float32)
        minimum[~populated] = 0.0
        maximum[~populated] = 0.0
        return {
            "count": count,
            "mean": mean,
            "std_raw": std_raw,
            "std_eff": std_eff,
            "minimum": minimum,
            "maximum": maximum,
            "supervise_mask": supervise_mask,
        }


def load_caption_map(path: Path) -> dict[str, dict]:
    captions: dict[str, dict] = {}
    for row in read_jsonl(path):
        stem = row.get("raw_bvh_stem")
        if stem and row.get("text_status") == "present":
            captions[stem.lower()] = {
                "texts": row.get("texts", [row.get("text", "")]),
                "text_entries": row.get("text_entries", []),
                "text_status": row.get("text_status"),
                "annotation_source": row.get("annotation_source"),
            }
    return captions


def grouped_verified_records(raw_root: Path, rig_ids: set[str], captions: dict[str, dict], args: argparse.Namespace) -> dict[str, list[dict]]:
    rows = read_jsonl(raw_root / "export_manifest.jsonl")
    tposes: dict[str, dict] = {}
    motions: dict[str, list[dict]] = {rig: [] for rig in rig_ids}
    for row in rows:
        rig_id = f"PZ_{row.get('object_key', '')}"
        if rig_id not in rig_ids:
            continue
        if int(row.get("export_fps", -1)) != args.fps:
            raise ValueError(f"{row.get('raw_bvh')}: expected export_fps={args.fps}")
        if row.get("sample_type") == "tpose":
            tposes[rig_id] = row
            continue
        if row.get("sample_type") != "motion":
            continue
        if not row.get("source_action_verified") or not row.get("ik_disabled_during_export"):
            raise ValueError(f"Unverified or IK-enabled BVH: {row.get('raw_bvh')}")
        caption = captions.get(str(row.get("raw_bvh_stem", "")).lower())
        if caption is None and not args.allow_missing_captions:
            raise ValueError(f"Missing official AniMo4D caption for {row.get('raw_bvh_stem')}")
        copied = dict(row)
        copied["caption"] = caption
        motions[rig_id].append(copied)
    missing_rest = sorted(rig for rig, rows_ in motions.items() if rows_ and rig not in tposes)
    if missing_rest:
        raise ValueError(f"Missing T-pose BVHs for {len(missing_rest)} rigs: {missing_rest[:5]}")
    for rig_id, records in motions.items():
        records.sort(key=lambda row: row["raw_bvh_stem"])
        if args.max_clips_per_rig is not None:
            del records[args.max_clips_per_rig :]
    return {rig: {"tpose": tposes[rig], "motions": records} for rig, records in motions.items() if records}


def validate_selection(rig_spec: dict, actual_names: list[str], actual_parents: np.ndarray) -> tuple[list[int], list[str], np.ndarray, list[str]]:
    selected = rig_spec["joints"]
    expected_names = [joint["name"] for joint in selected]
    lookup = {name: index for index, name in enumerate(actual_names)}
    missing = [name for name in expected_names if name not in lookup]
    if missing:
        raise ValueError(f"Missing selected joints: {missing[:5]}")
    full_indices = [lookup[name] for name in expected_names]
    parent_indices = np.asarray([joint["parent"] for joint in selected], dtype=np.int32)
    for selected_index, full_index in enumerate(full_indices):
        expected_parent = int(parent_indices[selected_index])
        actual_parent = int(actual_parents[full_index])
        actual_parent_selected = -1 if actual_parent == -1 else full_indices.index(actual_parent)
        if actual_parent_selected != expected_parent:
            raise ValueError(
                f"Parent mismatch for {expected_names[selected_index]}: "
                f"spec={expected_parent}, BVH={actual_parent_selected}"
            )
    descriptions = [joint["description"] for joint in selected]
    sources = [joint["rotation_source_kind"] for joint in selected]
    return full_indices, expected_names, parent_indices, descriptions, sources


def make_skeleton_context(rig_id: str, rig_spec: dict, tpose_path: Path) -> dict:
    raw_features = get_common_features_from_T_pose(str(tpose_path), rig_id, list(DEFAULT_FACE_NAMES))
    (
        root_pose_init_xz,
        scale_factor,
        ground_height,
        offsets,
        common_foot_indices,
        tpos_rots,
        names,
        tpos_anim,
        face_joints,
        rest_align_quat,
    ) = raw_features
    full_indices, joint_names, parents, descriptions, rotation_sources = validate_selection(
        rig_spec, names, tpos_anim.parents
    )
    squared_error: dict[str, float] = {}
    rest_anim, _ = get_hml_aligned_anim(
        tpos_anim,
        rig_id,
        root_pose_init_xz,
        scale_factor,
        ground_height,
        tpos_rots,
        offsets,
        squared_error,
        face_joints=face_joints,
        rest_pose_align_quat=rest_align_quat,
    )
    rest_global = transforms_global(rest_anim)[0]
    rest_position = positions_global(rest_anim)[0, full_indices].astype(np.float64)
    rest_rotation = rest_global[full_indices, :3, :3].astype(np.float64)
    rest_local_rotation = np.empty_like(rest_rotation)
    offset_parent_local = np.zeros_like(rest_position)
    for joint, parent in enumerate(parents):
        if parent < 0:
            rest_local_rotation[joint] = rest_rotation[joint]
            offset_parent_local[joint] = rest_position[joint]
        else:
            rest_local_rotation[joint] = rest_rotation[parent].T @ rest_rotation[joint]
            offset_parent_local[joint] = rest_rotation[parent].T @ (rest_position[joint] - rest_position[parent])
    face_name_pair = selected_face_names(names)
    face_full_indices = [names.index(name) for name in face_name_pair]
    selected_contact = contact_indices(names, full_indices, common_foot_indices)
    supervise_mask = np.zeros((len(joint_names), CHANNELS), dtype=bool)
    supervise_mask[:, :13] = True
    supervise_mask[0, 13:17] = True
    for joint, source in enumerate(rotation_sources):
        if source == "fixed_dof":
            supervise_mask[joint, ROTATION_SLICE] = False
    return {
        "rig_id": rig_id,
        "root_pose_init_xz": root_pose_init_xz,
        "scale_factor": float(scale_factor),
        "ground_height": float(ground_height),
        "offsets": offsets,
        "foot_indices_full": common_foot_indices,
        "tpos_rots": tpos_rots,
        "full_names": names,
        "full_parents": tpos_anim.parents,
        "face_joints_full": face_joints,
        "face_names": face_name_pair,
        "face_full_indices": face_full_indices,
        "rest_align_quat": rest_align_quat,
        "selected_full_indices": full_indices,
        "joint_names": joint_names,
        "parents": parents,
        "descriptions": descriptions,
        "rotation_sources": rotation_sources,
        "rest_position": rest_position,
        "rest_rotation": rest_rotation,
        "rest_local_rotation": rest_local_rotation,
        "offset_parent_local": offset_parent_local,
        "contact_indices": selected_contact,
        "supervise_mask": supervise_mask,
    }


def build_motion(context: dict, raw_bvh: Path, fps: int) -> tuple[np.ndarray, np.ndarray, dict]:
    aligned, _ = get_hml_aligned_anim(
        str(raw_bvh),
        context["rig_id"],
        context["root_pose_init_xz"],
        context["scale_factor"],
        context["ground_height"],
        context["tpos_rots"],
        context["offsets"],
        {},
        face_joints=context["face_joints_full"],
        rest_pose_align_quat=context["rest_align_quat"],
    )
    frame_time = None
    with raw_bvh.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "Frame Time:" in line:
                frame_time = float(line.split(":", 1)[1].strip())
                break
    if frame_time is None or frame_time <= 0:
        raise ValueError(f"Missing valid BVH frame time in {raw_bvh}")
    source_fps = int(round(1.0 / frame_time))
    if not math.isclose(frame_time, 1.0 / source_fps, rel_tol=1e-4, abs_tol=1e-5):
        raise ValueError(f"Unsupported BVH frame time {frame_time} in {raw_bvh}")
    source_frames = len(aligned)
    aligned = resample_aligned_animation(aligned, source_fps, fps)
    source_positions = positions_global(aligned)[:, context["selected_full_indices"]].astype(np.float64)
    source_global = transforms_global(aligned)[:, context["selected_full_indices"], :3, :3].astype(np.float64)
    frames = source_positions.shape[0]
    if frames < 2:
        raise ValueError(f"Need at least two target-rate frames, got {frames}")

    stored = frames - 1
    motion = np.zeros((stored, source_positions.shape[1], CHANNELS), dtype=np.float32)
    smooth_root_xz = source_positions[:stored, 0, (0, 2)]
    motion[:, :, POSITION_SLICE] = source_positions[:stored]
    motion[:, :, 0] -= smooth_root_xz[:, None, 0]
    motion[:, :, 2] -= smooth_root_xz[:, None, 1]
    rotation_delta = source_global[:stored] @ np.swapaxes(context["rest_rotation"], -1, -2)[None]
    motion[:, :, ROTATION_SLICE] = cont6d_from_matrix(rotation_delta)
    motion[:, :, VELOCITY_SLICE] = (source_positions[1:] - source_positions[:-1]) * fps

    contacts = np.zeros((stored, source_positions.shape[1]), dtype=np.float32)
    feet = context["contact_indices"]
    if feet:
        velocity_squared = np.square(source_positions[1:, feet] - source_positions[:-1, feet]).sum(axis=-1)
        ground = source_positions[:, feet, 1].min()
        height = source_positions[1:, feet, 1] - ground
        contacts[:, feet] = ((velocity_squared <= 0.002) & (np.abs(height) <= 0.3)).astype(np.float32)
    motion[:, :, CONTACT_CHANNEL] = contacts
    motion[:, 0, ROOT_XZ_SLICE] = smooth_root_xz

    forward = source_positions[:stored, context["face_full_indices"][1]] - source_positions[:stored, context["face_full_indices"][0]]
    forward[:, 1] = 0.0
    forward_length = np.linalg.norm(forward[:, (0, 2)], axis=-1)
    heading_valid = forward_length > 1e-6
    heading = np.zeros((stored, 2), dtype=np.float32)
    heading[heading_valid, 0] = forward[heading_valid, 2] / forward_length[heading_valid]
    heading[heading_valid, 1] = forward[heading_valid, 0] / forward_length[heading_valid]
    motion[:, 0, HEADING_SLICE] = heading

    decoded = decode_fk_positions(motion, context)
    position_error = np.abs(decoded - source_positions[:stored])
    metrics = {
        "source_frames": int(source_frames),
        "stored_frames": int(stored),
        "frame_time": frame_time,
        "source_fps": source_fps,
        "resampled_to_fps": fps,
        "fk_position_mae": float(position_error.mean()),
        "fk_position_max": float(position_error.max()),
        "heading_valid_frames": int(heading_valid.sum()),
    }
    if not np.isfinite(motion).all():
        raise ValueError("Non-finite KTJD-17 values")
    if metrics["fk_position_max"] > 1e-3:
        raise ValueError(f"FK reconstruction error too large: {metrics['fk_position_max']}")
    return motion, heading_valid, metrics


def decode_fk_positions(motion: np.ndarray, context: dict) -> np.ndarray:
    delta = matrix_from_cont6d(motion[:, :, ROTATION_SLICE].astype(np.float64))
    global_rotation = delta @ context["rest_rotation"][None]
    positions = np.zeros((motion.shape[0], motion.shape[1], 3), dtype=np.float64)
    positions[:, 0] = motion[:, 0, POSITION_SLICE]
    positions[:, 0, 0] += motion[:, 0, 13]
    positions[:, 0, 2] += motion[:, 0, 14]
    for joint, parent in enumerate(context["parents"]):
        if parent >= 0:
            positions[:, joint] = positions[:, parent] + np.einsum(
                "tij,j->ti", global_rotation[:, parent], context["offset_parent_local"][joint]
            )
    return positions


def write_skeleton(output_root: Path, context: dict) -> str:
    path = output_root / "skeletons" / f"{context['rig_id']}.npz"
    np.savez_compressed(
        path,
        joint_names=np.asarray(context["joint_names"], dtype=np.str_),
        joint_descriptions=np.asarray(context["descriptions"], dtype=np.str_),
        parents=context["parents"].astype(np.int32),
        P_rest_global=context["rest_position"].astype(np.float32),
        R_rest_global=context["rest_rotation"].astype(np.float32),
        R_rest_local=context["rest_local_rotation"].astype(np.float32),
        offset_parent_local=context["offset_parent_local"].astype(np.float32),
        rotation_source_kind=np.asarray(context["rotation_sources"], dtype=np.str_),
        contact_joint_indices=np.asarray(context["contact_indices"], dtype=np.int32),
        face_joint_names=np.asarray(context["face_names"], dtype=np.str_),
        joint_order_sha256=np.asarray(sha256_strings(context["joint_names"]), dtype=np.str_),
    )
    return path.relative_to(output_root).as_posix()


def process_rig(task: dict) -> dict:
    output_root = Path(task["output_root"])
    rig_id = task["rig_id"]
    try:
        context = make_skeleton_context(rig_id, task["rig_spec"], Path(task["tpose"]["raw_bvh"]))
        skeleton_file = write_skeleton(output_root, context)
        stats = RunningStats(len(context["joint_names"]))
        written: list[dict] = []
        errors: list[dict] = []
        for source in task["motions"]:
            try:
                motion, heading_valid, metrics = build_motion(context, Path(source["raw_bvh"]), task["fps"])
                clip_id = safe_stem(source["raw_bvh_stem"])
                motion_path = output_root / "motions" / f"{clip_id}.npz"
                if motion_path.exists():
                    raise ValueError(f"Duplicate clip_id: {clip_id}")
                np.savez_compressed(motion_path, motion=motion, heading_valid=heading_valid)
                stats.update(motion, heading_valid)
                source_rel = Path(source["raw_bvh"]).relative_to(Path(task["raw_root"])).as_posix()
                written.append(
                    {
                        "clip_id": clip_id,
                        "rig_id": rig_id,
                        "motion_file": motion_path.relative_to(output_root).as_posix(),
                        "skeleton_file": skeleton_file,
                        "source_raw_bvh": source_rel,
                        "raw_bvh_stem": source["raw_bvh_stem"],
                        "source_action_name": source["action_name"],
                        "source_motion_key": source["source_motion_key"],
                        "source_action_verified": True,
                        "ik_disabled_during_export": True,
                        "fps_target": task["fps"],
                        "caption": source["caption"],
                        **metrics,
                    }
                )
            except Exception as exc:
                errors.append({"raw_bvh": source.get("raw_bvh"), "error": repr(exc), "traceback": traceback.format_exc()})
        stats_path = output_root / "stats" / f"{rig_id}.npz"
        np.savez_compressed(stats_path, **stats.finish(context["supervise_mask"]))
        return {
            "rig_id": rig_id,
            "ok": True,
            "skeleton_file": skeleton_file,
            "stats_file": stats_path.relative_to(output_root).as_posix(),
            "clips": written,
            "errors": errors,
            "joint_count": len(context["joint_names"]),
            "joint_order_sha256": sha256_strings(context["joint_names"]),
        }
    except Exception as exc:
        return {"rig_id": rig_id, "ok": False, "error": repr(exc), "traceback": traceback.format_exc(), "clips": []}


def prepare_output(root: Path) -> None:
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Output root must be new and empty: {root}")
    for directory in ("motions", "skeletons", "stats", "manifests", "reports", "metadata"):
        (root / directory).mkdir(parents=True, exist_ok=True)


def git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or args.workers <= 0:
        raise ValueError("--fps and --workers must be positive")
    joint_spec = json.loads(args.joint_spec.read_text(encoding="utf-8"))
    rig_specs = {rig: value for rig, value in joint_spec["rigs"].items() if rig.startswith("PZ_")}
    rig_ids = sorted(rig_specs)
    if args.max_rigs is not None:
        rig_ids = rig_ids[: args.max_rigs]
    captions = load_caption_map(args.caption_manifest)
    sources = grouped_verified_records(args.raw_root, set(rig_ids), captions, args)
    missing = sorted(set(rig_ids) - set(sources))
    if missing:
        raise ValueError(f"No verified motion records for {len(missing)} requested rigs: {missing[:5]}")
    prepare_output(args.output_root)
    descriptions = {
        rig: [
            {key: joint[key] for key in ("index", "name", "parent", "description", "rotation_source_kind")}
            for joint in rig_specs[rig]["joints"]
        ]
        for rig in sources
    }
    (args.output_root / "metadata" / "joint_descriptions.json").write_text(
        json.dumps(descriptions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    generation = {
        "format": "KTJD-17",
        "version": "20260822-pz-fresh-v1",
        "source": "Planet Zoo Cobra MS2/MANIS",
        "rig_count_requested": len(rig_ids),
        "fps_target": args.fps,
        "coordinate_system": "right-handed, +Y up, +Z viewer-facing; heading measured from +Z",
        "motion_layout": "[q_position(3), rest_delta_6d(6), velocity_xyz(3), contact(1), smooth_root_xz(2), heading_cos_sin(2)]",
        "smooth_root": "root horizontal trajectory; no arbitrary smoothing was applied",
        "temporal_resampling": "BVH local rotations use shortest-path SLERP and local translations use linear interpolation when source FPS differs from fps_target.",
        "contact": "legacy PZ foot criteria: squared frame displacement <= 0.002 and relative foot height <= 0.3",
        "raw_bvh_root": str(args.raw_root),
        "joint_spec": str(args.joint_spec),
        "caption_manifest": str(args.caption_manifest),
        "code_revision": git_revision(),
    }
    tasks = [
        {
            "output_root": str(args.output_root),
            "raw_root": str(args.raw_root),
            "rig_id": rig,
            "rig_spec": rig_specs[rig],
            "tpose": sources[rig]["tpose"],
            "motions": sources[rig]["motions"],
            "fps": args.fps,
        }
        for rig in sorted(sources)
    ]
    results: list[dict] = []
    if args.workers == 1:
        for task in tasks:
            result = process_rig(task)
            results.append(result)
            print(json.dumps({"rig": result["rig_id"], "clips": len(result["clips"]), "ok": result["ok"]}), flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_rig, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps({"rig": result["rig_id"], "clips": len(result["clips"]), "ok": result["ok"]}), flush=True)
    results.sort(key=lambda result: result["rig_id"])
    clips = [clip for result in results for clip in result["clips"]]
    errors = [
        {"rig_id": result["rig_id"], **error}
        for result in results
        for error in result.get("errors", [])
    ]
    failures = [result for result in results if not result["ok"]]
    with (args.output_root / "manifests" / "clips.jsonl").open("w", encoding="utf-8") as handle:
        for clip in clips:
            handle.write(json.dumps(clip, ensure_ascii=False) + "\n")
    (args.output_root / "reports" / "conversion_errors.json").write_text(
        json.dumps(errors + failures, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    generation.update(
        {
            "rig_count_completed": sum(result["ok"] for result in results),
            "clips_written": len(clips),
            "clip_errors": len(errors),
            "rig_failures": len(failures),
        }
    )
    (args.output_root / "generation.json").write_text(
        json.dumps(generation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(generation, indent=2, ensure_ascii=False))
    if errors or failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
