"""Create an isolated preview of canonical (+Y up, +Z forward) rest poses.

This is deliberately a preview tool: it writes selected corrected skeletons and
one corrected clip per rig to a new directory, leaving a release untouched.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.planetzoo.audit_noik_rest_pose_orientation import (  # noqa: E402
    MOTION_COLOR,
    REST_COLOR,
    TEXT_COLOR,
    draw_panel,
    face_indices,
    font,
    forward_vector,
)


CANONICAL_REST_COLOR = (32, 145, 83)
DEFAULT_RIGS = (
    "PZ_Aardvark_Female",
    "PZ_African_Elephant_Female",
    "PZ_Bengal_Tiger_Male",
    "PZ_Caracal_Male",
    "PZ_Grey_Seal_Female",
    "PZ_Nile_Monitor_Male",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rig", action="append", default=[])
    parser.add_argument("--all", action="store_true", help="Render every rig for manual review.")
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Do not write the temporary corrected .npz previews; render only PNGs and the HTML index.",
    )
    parser.add_argument("--panel-width", type=int, default=460)
    parser.add_argument("--panel-height", type=int, default=330)
    return parser.parse_args()


def data_root_from(release_root: Path) -> Path:
    return release_root / "data" if (release_root / "data").is_dir() else release_root


def cont6d_from_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.swapaxes(matrix[..., :, :2], -1, -2).reshape(*matrix.shape[:-2], 6)


def matrix_from_cont6d(cont6d: np.ndarray) -> np.ndarray:
    first = cont6d[..., :3]
    second = cont6d[..., 3:6]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def first_clip_by_rig(manifest_path: Path) -> dict[str, dict]:
    first: dict[str, dict] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            first.setdefault(row["rig_id"], row)
    return first


def accumulate_offsets(offsets: np.ndarray, parents: np.ndarray) -> np.ndarray:
    positions = np.zeros_like(offsets, dtype=np.float64)
    positions[0] = offsets[0]
    for joint in range(1, len(parents)):
        positions[joint] = positions[int(parents[joint])] + offsets[joint]
    return positions


def decode_world_positions(motion: np.ndarray, rotations: np.ndarray, parents: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    positions = np.zeros((motion.shape[0], motion.shape[1], 3), dtype=np.float64)
    positions[:, 0] = motion[:, 0, :3]
    positions[:, 0, 0] += motion[:, 0, 13]
    positions[:, 0, 2] += motion[:, 0, 14]
    for joint in range(1, len(parents)):
        parent = int(parents[joint])
        positions[:, joint] = positions[:, parent] + np.einsum("tij,j->ti", rotations[:, parent], offsets[joint])
    return positions


def support_joint_indices(joint_names: list[str], parents: np.ndarray) -> list[int]:
    candidates = [
        index
        for index, name in enumerate(joint_names)
        if any(token in name.lower() for token in ("toe", "foot", "hoof", "ashi", "paw", "phalanx"))
    ]
    candidate_set = set(candidates)
    return [
        index
        for index in candidates
        if not any(int(parent) == index and child in candidate_set for child, parent in enumerate(parents))
    ]


def canonical_ground_transform(
    positions: np.ndarray, joint_names: list[str], parents: np.ndarray, face: list[int]
) -> tuple[np.ndarray, dict[str, float | int]]:
    support = support_joint_indices(joint_names, parents)
    if len(support) < 3:
        raise ValueError("Need at least three terminal foot/toe joints to define a rest ground plane")
    support_points = positions[support]
    centre = support_points.mean(axis=0)
    _, singular_values, vectors = np.linalg.svd(support_points - centre, full_matrices=False)
    if singular_values[1] <= 1e-8:
        raise ValueError("Rest support joints are collinear")
    normal = vectors[-1]
    upper_body = [
        index
        for index, name in enumerate(joint_names)
        if any(token in name.lower() for token in ("chest", "neck", "head"))
    ]
    if not upper_body:
        upper_body = [
            index
            for index, name in enumerate(joint_names)
            if any(token in name.lower() for token in ("hips", "spine"))
        ]
    upper_point = positions[upper_body].mean(axis=0) if upper_body else positions[0]
    if float(np.dot(upper_point - centre, normal)) < 0.0:
        normal = -normal
    forward = positions[face[1]] - positions[face[0]]
    forward -= normal * np.dot(forward, normal)
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    right = np.cross(normal, forward)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    basis = np.stack((right, normal, forward), axis=1)
    residual = float(np.std((support_points - centre) @ normal))
    return basis.T, {"support_joint_count": len(support), "support_plane_rms": residual}


def apply_global_rotation(positions: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    root = positions[0]
    return (positions - root) @ rotation.T + root


def local_from_global(global_rotation: np.ndarray, parents: np.ndarray) -> np.ndarray:
    local = np.empty_like(global_rotation)
    local[0] = global_rotation[0]
    for joint in range(1, len(parents)):
        parent = int(parents[joint])
        local[joint] = global_rotation[parent].T @ global_rotation[joint]
    return local


def recanonicalize_motion(motion: np.ndarray, correction: np.ndarray) -> np.ndarray:
    corrected = motion.copy()
    delta = matrix_from_cont6d(motion[:, :, 3:9].astype(np.float64))
    corrected_delta = delta @ correction.T
    corrected[:, :, 3:9] = cont6d_from_matrix(corrected_delta).astype(np.float32)
    return corrected


def render_triplet(
    output_path: Path,
    rig_id: str,
    parents: np.ndarray,
    original_rest: np.ndarray,
    corrected_rest: np.ndarray,
    motion_frame: np.ndarray,
    original_forward: np.ndarray,
    corrected_forward: np.ndarray,
    motion_forward: np.ndarray,
    panel_width: int,
    panel_height: int,
) -> None:
    rows = (
        ("CURRENT REST: saved P_rest_global", original_rest, original_forward, REST_COLOR),
        ("CANDIDATE REST: original support plane -> XZ ground", corrected_rest, corrected_forward, CANONICAL_REST_COLOR),
        ("MOTION frame 0: canonical rest + rot6d FK", motion_frame, motion_forward, MOTION_COLOR),
    )
    canvas = Image.new("RGB", (panel_width * 3, panel_height * 3 + 54), (246, 247, 249))
    for row_index, (title, positions, forward, color) in enumerate(rows):
        for column, view in enumerate(("top", "side", "front")):
            panel = draw_panel(
                positions,
                parents,
                forward,
                view,
                title,
                color,
                panel_width,
                panel_height,
            )
            canvas.paste(panel, (column * panel_width, 54 + row_index * panel_height))
    ImageDraw.Draw(canvas).text(
        (16, 15),
        f"{rig_id} | full rest skeleton rotated from original foot plane; q_position stays untouched",
        fill=TEXT_COLOR,
        font=font(23),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_html_index(output_dir: Path, summary: list[dict[str, object]]) -> Path:
    rows = []
    for row in summary:
        rig_id = html.escape(str(row["rig_id"]))
        image = f"gallery/{rig_id}.png"
        rows.append(
            "<a class=\"card\" href=\"{image}\">"
            "<img loading=\"lazy\" src=\"{image}\" alt=\"{rig_id}\">"
            "<span>{rig_id}</span></a>".format(image=image, rig_id=rig_id)
        )
    page = """<!doctype html>
<html><head><meta charset=\"utf-8\"><title>Rest-Pose Review</title>
<style>
body { margin: 20px; font: 15px/1.4 Segoe UI, Arial, sans-serif; color: #222; background: #f4f6f8; }
h1 { margin: 0 0 8px; } p { margin: 0 0 20px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 16px; }
.card { display: block; color: inherit; text-decoration: none; padding: 8px; background: white; border: 1px solid #d9dde2; }
.card:hover { border-color: #3578c5; } img { display: block; width: 100%; height: auto; } span { display: block; margin: 8px 3px 2px; font-weight: 600; }
</style></head><body>
<h1>311-rig Rest-Pose Review</h1>
<p>Orange: stored rest. Green: candidate support-plane canonical rest. Blue: frame 0 decoded from candidate rest plus converted rot6d. Click an image for full resolution.</p>
<main class=\"grid\">{rows}</main></body></html>""".replace("{rows}", "\n".join(rows))
    index = output_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    return index


def main() -> None:
    args = parse_args()
    data_root = data_root_from(args.release_root)
    first_clips = first_clip_by_rig(data_root / "manifests" / "clips.jsonl")
    available = {path.stem for path in (data_root / "skeletons").glob("*.npz")}
    if args.all and args.rig:
        raise ValueError("Use either --all or one or more --rig values, not both")
    rig_ids = sorted(available) if args.all else (args.rig or [rig for rig in DEFAULT_RIGS if rig in available])
    if not rig_ids:
        raise ValueError("No requested preview rigs are present")
    missing = [rig for rig in rig_ids if rig not in available]
    if missing:
        raise ValueError(f"Unknown rigs: {missing}")

    skeleton_out = args.output_dir / "skeletons"
    motion_out = args.output_dir / "motions"
    gallery_out = args.output_dir / "gallery"
    summary: list[dict[str, object]] = []
    for rig_id in rig_ids:
        skeleton_path = data_root / "skeletons" / f"{rig_id}.npz"
        with np.load(skeleton_path, allow_pickle=False) as source:
            payload = {key: source[key].copy() for key in source.files}
        parents = payload["parents"].astype(np.int64)
        offsets = payload["offset_parent_local"].astype(np.float64)
        previous_rest = payload["P_rest_global"].astype(np.float64)
        previous_global_rotation = payload["R_rest_global"].astype(np.float64)
        shared_rotation = previous_global_rotation[0]
        within_rig_error = float(np.abs(previous_global_rotation - shared_rotation).max())
        if within_rig_error > 1e-5:
            raise ValueError(f"{rig_id}: rest global rotations are not one shared basis ({within_rig_error})")

        names = payload["joint_names"].astype(str).tolist()
        faces = payload["face_joint_names"].astype(str).tolist()
        indices = face_indices(names, faces)
        correction, plane_info = canonical_ground_transform(previous_rest, names, parents, indices)
        corrected_rest = apply_global_rotation(previous_rest, correction)
        offsets = offsets.copy()
        offsets[0, 1] -= corrected_rest[:, 1].min()
        corrected_rest[:, 1] -= corrected_rest[:, 1].min()
        corrected_global_rotation = correction[None] @ previous_global_rotation
        payload["offset_parent_local"] = offsets.astype(np.float32)
        payload["P_rest_global"] = corrected_rest.astype(np.float32)
        payload["R_rest_global"] = corrected_global_rotation.astype(np.float32)
        payload["R_rest_local"] = local_from_global(corrected_global_rotation, parents).astype(np.float32)
        if not args.images_only:
            skeleton_out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(skeleton_out / skeleton_path.name, **payload)

        clip = first_clips[rig_id]
        with np.load(data_root / clip["motion_file"], allow_pickle=False) as source:
            motion_payload = {key: source[key].copy() for key in source.files}
        original_motion = motion_payload["motion"]
        corrected_motion = recanonicalize_motion(original_motion, correction)
        motion_payload["motion"] = corrected_motion
        preview_motion_path = motion_out / f"{rig_id}__{Path(clip['motion_file']).stem}.npz"
        if not args.images_only:
            motion_out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(preview_motion_path, **motion_payload)

        corrected_delta = matrix_from_cont6d(corrected_motion[:, :, 3:9].astype(np.float64))
        corrected_global = corrected_delta @ corrected_global_rotation[None]
        recovered = decode_world_positions(corrected_motion, corrected_global, parents, offsets)
        stored_world = original_motion[:, :, :3].astype(np.float64)
        stored_world[:, :, 0] += original_motion[:, 0, 13, None]
        stored_world[:, :, 2] += original_motion[:, 0, 14, None]
        motion_fk_max_error = float(np.abs(recovered - stored_world).max())
        if motion_fk_max_error > 1e-4:
            raise ValueError(f"{rig_id}: corrected FK error too large ({motion_fk_max_error})")

        render_triplet(
            gallery_out / f"{rig_id}.png",
            rig_id,
            parents,
            previous_rest,
            corrected_rest,
            recovered[0],
            forward_vector(previous_rest, indices),
            forward_vector(corrected_rest, indices),
            forward_vector(recovered[0], indices),
            args.panel_width,
            args.panel_height,
        )
        summary.append(
            {
                "rig_id": rig_id,
                "source_motion": clip["motion_file"],
                "preview_motion": None if args.images_only else preview_motion_path.name,
                "shared_old_rest_basis_max_within_rig_error": within_rig_error,
                "corrected_motion_fk_max_abs_error": motion_fk_max_error,
                "corrected_rest_min_y": float(corrected_rest[:, 1].min()),
                "corrected_rest_max_y": float(corrected_rest[:, 1].max()),
                **plane_info,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    index = write_html_index(args.output_dir, summary)
    print(f"HTML review index: {index}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
