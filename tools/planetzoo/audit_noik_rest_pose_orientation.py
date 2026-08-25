"""Audit and visualize rest-pose coordinates in the no-IK AnyTop release.

The release stores a rig-specific rest skeleton and motion clips separately.  This
tool compares every ``P_rest_global`` with the first world-space pose of a clip
from the same rig.  It writes a machine-readable audit for all rigs and a small
gallery with explicit coordinate axes for human review.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REST_COLOR = (214, 96, 28)
MOTION_COLOR = (26, 112, 184)
BONE_COLOR = (48, 48, 48)
GRID_COLOR = (225, 229, 234)
TEXT_COLOR = (30, 30, 30)
AXIS_X = (211, 41, 41)
AXIS_Y = (35, 150, 72)
AXIS_Z = (42, 86, 202)

DEFAULT_RIGS = (
    "PZ_Aardvark_Female",
    "PZ_African_Elephant_Female",
    "PZ_Bengal_Tiger_Male",
    "PZ_Caracal_Male",
    "PZ_Common_Warthog_Female",
    "PZ_Grey_Seal_Female",
    "PZ_Hippopotamus_Female",
    "PZ_Indian_Peafowl_Male",
    "PZ_Indian_Rhinoceros_Male",
    "PZ_Nile_Monitor_Male",
    "PZ_Plains_Zebra_Female",
    "PZ_West_African_Lion_Male",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-root",
        type=Path,
        required=True,
        help="Release root containing data/, or data/ itself.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rig",
        action="append",
        default=[],
        help="Rig to render. May be repeated. Defaults to a representative gallery.",
    )
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--panel-width", type=int, default=460)
    parser.add_argument("--panel-height", type=int, default=390)
    return parser.parse_args()


def data_root_from(release_root: Path) -> Path:
    return release_root / "data" if (release_root / "data").is_dir() else release_root


def font(size: int) -> ImageFont.ImageFont:
    for candidate in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def first_clip_by_rig(manifest_path: Path) -> dict[str, dict]:
    first: dict[str, dict] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            first.setdefault(row["rig_id"], row)
    return first


def face_indices(joint_names: list[str], stored_face_names: list[str]) -> list[int]:
    missing = [name for name in stored_face_names if name not in joint_names]
    if missing:
        raise ValueError(f"Face joints absent from skeleton: {missing}")
    if len(stored_face_names) not in (2, 4):
        raise ValueError(f"Expected 2 or 4 face joints, got {stored_face_names}")
    return [joint_names.index(name) for name in stored_face_names]


def forward_vector(positions: np.ndarray, indices: list[int]) -> np.ndarray:
    if len(indices) == 2:
        vector = positions[indices[1]] - positions[indices[0]]
    else:
        across = (positions[indices[0]] - positions[indices[1]]) + (
            positions[indices[2]] - positions[indices[3]]
        )
        vector = np.cross(np.array([0.0, 1.0, 0.0]), across)
    vector = vector.astype(np.float64, copy=True)
    vector[1] = 0.0
    length = float(np.linalg.norm(vector))
    if length <= 1e-8:
        return np.array([np.nan, np.nan, np.nan])
    return vector / length


def heading_degrees(forward: np.ndarray) -> float:
    if not np.isfinite(forward).all():
        return float("nan")
    return float(np.degrees(np.arctan2(forward[0], forward[2])))


def angle_from_positive_z(degrees: float) -> float:
    return abs((degrees + 180.0) % 360.0 - 180.0)


def axis_label(vector: np.ndarray) -> str:
    labels = ("+X", "+Y", "+Z", "-X", "-Y", "-Z")
    axes = np.vstack((np.eye(3), -np.eye(3)))
    return labels[int(np.argmax(axes @ vector))]


def landmark_indices(joint_names: list[str], tokens: tuple[str, ...]) -> list[int]:
    return [
        index
        for index, name in enumerate(joint_names)
        if any(token in name.lower() for token in tokens)
    ]


def core_to_feet(positions: np.ndarray, joint_names: list[str]) -> np.ndarray:
    feet = landmark_indices(joint_names, ("toe", "foot", "hoof", "ashi", "paw"))
    core = landmark_indices(joint_names, ("hips", "spine", "chest", "neck"))
    if not feet or not core:
        return np.full(3, np.nan, dtype=np.float64)
    return positions[core].mean(axis=0) - positions[feet].mean(axis=0)


def load_world_frame(data_root: Path, motion_file: str) -> np.ndarray:
    with np.load(data_root / motion_file, allow_pickle=False) as payload:
        motion = payload["motion"]
    frame = motion[0, :, :3].astype(np.float64)
    frame[0, 0] += float(motion[0, 0, 13])
    frame[0, 2] += float(motion[0, 0, 14])
    return frame


def rest_fk_error(payload: np.lib.npyio.NpzFile) -> float:
    positions = payload["P_rest_global"].astype(np.float64)
    rotations = payload["R_rest_global"].astype(np.float64)
    offsets = payload["offset_parent_local"].astype(np.float64)
    parents = payload["parents"].astype(np.int64)
    reconstructed = np.empty_like(positions)
    reconstructed[0] = positions[0]
    for joint in range(1, len(parents)):
        parent = int(parents[joint])
        reconstructed[joint] = reconstructed[parent] + rotations[parent] @ offsets[joint]
    return float(np.abs(reconstructed - positions).max())


def orthographic(points: np.ndarray, view: str) -> np.ndarray:
    if view == "top":
        return np.stack((points[:, 0], points[:, 2]), axis=-1)
    if view == "side":
        return np.stack((points[:, 2], points[:, 1]), axis=-1)
    if view == "front":
        return np.stack((points[:, 0], points[:, 1]), axis=-1)
    raise ValueError(view)


def view_title(view: str) -> str:
    return {
        "top": "top: +Y looking down (+Z up, +X right)",
        "side": "side: -X looking in (+Y up, +Z right)",
        "front": "front: -Z looking in (+Y up, +X right)",
    }[view]


def projection_bounds(points: np.ndarray, view: str) -> tuple[float, float, float, float]:
    projected = orthographic(points, view)
    min_x, min_y = projected.min(axis=0)
    max_x, max_y = projected.max(axis=0)
    span = max(max_x - min_x, max_y - min_y, 1e-5)
    pad = span * 0.14
    return min_x - pad, max_x + pad, min_y - pad, max_y + pad


def project(points: np.ndarray, bounds: tuple[float, float, float, float], width: int, height: int) -> np.ndarray:
    min_x, max_x, min_y, max_y = bounds
    scale = min((width - 48) / max(max_x - min_x, 1e-6), (height - 76) / max(max_y - min_y, 1e-6))
    projected = points.copy()
    projected[:, 0] = (projected[:, 0] - (min_x + max_x) * 0.5) * scale + width * 0.5
    projected[:, 1] = height * 0.55 - (projected[:, 1] - (min_y + max_y) * 0.5) * scale
    return projected


def draw_axis_legend(draw: ImageDraw.ImageDraw, view: str, width: int, height: int) -> None:
    origin = (44, height - 36)
    axis_map = {
        "top": (("+X", (1, 0), AXIS_X), ("+Z", (0, -1), AXIS_Z)),
        "side": (("+Z", (1, 0), AXIS_Z), ("+Y", (0, -1), AXIS_Y)),
        "front": (("+X", (1, 0), AXIS_X), ("+Y", (0, -1), AXIS_Y)),
    }
    label_font = font(15)
    for label, direction, color in axis_map[view]:
        endpoint = (origin[0] + direction[0] * 38, origin[1] + direction[1] * 38)
        draw.line((origin, endpoint), fill=color, width=3)
        draw.text((endpoint[0] + 4, endpoint[1] - 14), label, fill=color, font=label_font)


def draw_panel(
    positions: np.ndarray,
    parents: np.ndarray,
    forward: np.ndarray,
    view: str,
    title: str,
    color: tuple[int, int, int],
    width: int,
    height: int,
) -> Image.Image:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    projected = orthographic(positions, view)
    bounds = projection_bounds(positions, view)
    projected_px = project(projected, bounds, width, height)

    min_x, max_x, min_y, max_y = bounds
    for fraction in np.linspace(0.0, 1.0, 6):
        x = min_x + fraction * (max_x - min_x)
        y = min_y + fraction * (max_y - min_y)
        vertical = project(np.array([[x, min_y], [x, max_y]]), bounds, width, height)
        horizontal = project(np.array([[min_x, y], [max_x, y]]), bounds, width, height)
        draw.line([tuple(vertical[0]), tuple(vertical[1])], fill=GRID_COLOR, width=1)
        draw.line([tuple(horizontal[0]), tuple(horizontal[1])], fill=GRID_COLOR, width=1)

    for joint in range(1, len(parents)):
        parent = int(parents[joint])
        if parent >= 0:
            draw.line([tuple(projected_px[parent]), tuple(projected_px[joint])], fill=BONE_COLOR, width=2)
    for joint, point in enumerate(projected_px):
        radius = 3 if joint else 5
        draw.ellipse(
            (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
            fill=color if joint else (20, 20, 20),
        )

    root = positions[0]
    arrow_end = root + forward * max(np.ptp(positions, axis=0).max() * 0.32, 0.1)
    arrow = project(orthographic(np.stack((root, arrow_end)), view), bounds, width, height)
    if np.linalg.norm(arrow[1] - arrow[0]) > 4:
        draw.line([tuple(arrow[0]), tuple(arrow[1])], fill=color, width=5)
        draw.ellipse((arrow[1][0] - 5, arrow[1][1] - 5, arrow[1][0] + 5, arrow[1][1] + 5), fill=color)

    draw.text((14, 12), title, fill=color, font=font(21))
    draw.text((14, 40), view_title(view), fill=TEXT_COLOR, font=font(14))
    draw_axis_legend(draw, view, width, height)
    return image


def render_rig(
    output_path: Path,
    rig_id: str,
    rest: np.ndarray,
    motion: np.ndarray,
    parents: np.ndarray,
    rest_forward: np.ndarray,
    motion_forward: np.ndarray,
    panel_width: int,
    panel_height: int,
) -> None:
    canvas = Image.new("RGB", (panel_width * 3, panel_height * 2 + 54), (246, 247, 249))
    for row, (label, positions, forward, color) in enumerate(
        (("REST: P_rest_global", rest, rest_forward, REST_COLOR), ("MOTION frame 0: world position channels", motion, motion_forward, MOTION_COLOR))
    ):
        for column, view in enumerate(("top", "side", "front")):
            panel = draw_panel(
                positions,
                parents,
                forward,
                view,
                label,
                color,
                panel_width,
                panel_height,
            )
            canvas.paste(panel, (column * panel_width, 54 + row * panel_height))
    title = f"{rig_id} | orange = saved rest skeleton | blue = first stored motion frame"
    ImageDraw.Draw(canvas).text((16, 15), title, fill=TEXT_COLOR, font=font(24))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def select_gallery_rigs(available: list[str], requested: list[str], count: int) -> list[str]:
    lookup = set(available)
    if requested:
        missing = [rig for rig in requested if rig not in lookup]
        if missing:
            raise ValueError(f"Unknown requested rigs: {missing}")
        return requested
    selected = [rig for rig in DEFAULT_RIGS if rig in lookup]
    if len(selected) < count:
        remaining = [rig for rig in available if rig not in selected]
        needed = count - len(selected)
        if needed > 0 and remaining:
            indices = np.linspace(0, len(remaining) - 1, min(needed, len(remaining)), dtype=int)
            selected.extend(remaining[index] for index in indices)
    return selected[:count]


def main() -> None:
    args = parse_args()
    data_root = data_root_from(args.release_root)
    skeleton_dir = data_root / "skeletons"
    first_clips = first_clip_by_rig(data_root / "manifests" / "clips.jsonl")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audits: list[dict[str, object]] = []
    cached: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for skeleton_path in sorted(skeleton_dir.glob("*.npz")):
        rig_id = skeleton_path.stem
        if rig_id not in first_clips:
            raise ValueError(f"No motion clip for {rig_id}")
        with np.load(skeleton_path, allow_pickle=False) as payload:
            names = payload["joint_names"].astype(str).tolist()
            parents = payload["parents"].astype(np.int64)
            rest = payload["P_rest_global"].astype(np.float64)
            faces = payload["face_joint_names"].astype(str).tolist()
            fk_error = rest_fk_error(payload)
        motion = load_world_frame(data_root, str(first_clips[rig_id]["motion_file"]))
        indices = face_indices(names, faces)
        rest_forward = forward_vector(rest, indices)
        motion_forward = forward_vector(motion, indices)
        rest_core_feet = core_to_feet(rest, names)
        motion_core_feet = core_to_feet(motion, names)
        audits.append(
            {
                "rig_id": rig_id,
                "joint_count": len(names),
                "motion_file": first_clips[rig_id]["motion_file"],
                "rest_forward_x": float(rest_forward[0]),
                "rest_forward_y": float(rest_forward[1]),
                "rest_forward_z": float(rest_forward[2]),
                "rest_heading_deg_from_plus_z": heading_degrees(rest_forward),
                "rest_heading_error_from_plus_z_deg": angle_from_positive_z(heading_degrees(rest_forward)),
                "rest_forward_nearest_axis": axis_label(rest_forward),
                "motion_forward_x": float(motion_forward[0]),
                "motion_forward_y": float(motion_forward[1]),
                "motion_forward_z": float(motion_forward[2]),
                "motion_heading_deg_from_plus_z": heading_degrees(motion_forward),
                "motion_heading_error_from_plus_z_deg": angle_from_positive_z(heading_degrees(motion_forward)),
                "motion_forward_nearest_axis": axis_label(motion_forward),
                "rest_core_minus_feet_x": float(rest_core_feet[0]),
                "rest_core_minus_feet_y": float(rest_core_feet[1]),
                "rest_core_minus_feet_z": float(rest_core_feet[2]),
                "rest_up_nearest_axis": axis_label(rest_core_feet) if np.isfinite(rest_core_feet).all() else "unknown",
                "motion_core_minus_feet_x": float(motion_core_feet[0]),
                "motion_core_minus_feet_y": float(motion_core_feet[1]),
                "motion_core_minus_feet_z": float(motion_core_feet[2]),
                "motion_up_nearest_axis": axis_label(motion_core_feet) if np.isfinite(motion_core_feet).all() else "unknown",
                "rest_fk_max_abs_error": fk_error,
            }
        )
        cached[rig_id] = (rest, motion, parents, rest_forward, motion_forward)

    csv_path = args.output_dir / "rest_pose_orientation_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audits[0]))
        writer.writeheader()
        writer.writerows(audits)

    rest_errors = np.asarray([row["rest_heading_error_from_plus_z_deg"] for row in audits], dtype=float)
    motion_errors = np.asarray([row["motion_heading_error_from_plus_z_deg"] for row in audits], dtype=float)
    summary = {
        "rig_count": len(audits),
        "coordinate_contract": "right-handed, +Y up, +Z forward",
        "rest_heading_error_from_plus_z_deg": {
            "min": float(rest_errors.min()),
            "median": float(np.median(rest_errors)),
            "max": float(rest_errors.max()),
            "within_5_deg": int((rest_errors <= 5.0).sum()),
        },
        "motion_frame0_heading_error_from_plus_z_deg": {
            "min": float(motion_errors.min()),
            "median": float(np.median(motion_errors)),
            "max": float(motion_errors.max()),
            "within_5_deg": int((motion_errors <= 5.0).sum()),
        },
        "rest_forward_nearest_axis_counts": {
            label: sum(row["rest_forward_nearest_axis"] == label for row in audits)
            for label in ("+X", "+Y", "+Z", "-X", "-Y", "-Z")
        },
        "motion_forward_nearest_axis_counts": {
            label: sum(row["motion_forward_nearest_axis"] == label for row in audits)
            for label in ("+X", "+Y", "+Z", "-X", "-Y", "-Z")
        },
        "rest_fk_max_abs_error": float(max(row["rest_fk_max_abs_error"] for row in audits)),
    }
    (args.output_dir / "rest_pose_orientation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    gallery_dir = args.output_dir / "gallery"
    gallery_rigs = select_gallery_rigs(sorted(cached), args.rig, args.sample_count)
    for rig_id in gallery_rigs:
        rest, motion, parents, rest_forward, motion_forward = cached[rig_id]
        render_rig(
            gallery_dir / f"{rig_id}.png",
            rig_id,
            rest,
            motion,
            parents,
            rest_forward,
            motion_forward,
            args.panel_width,
            args.panel_height,
        )
    (args.output_dir / "gallery_manifest.json").write_text(
        json.dumps({"rigs": gallery_rigs, "legend": "Orange is P_rest_global; blue is first stored world-position frame."}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Rendered {len(gallery_rigs)} rigs to {gallery_dir}")


if __name__ == "__main__":
    main()
