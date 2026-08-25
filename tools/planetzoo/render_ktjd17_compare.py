"""Render KTJD-17 q-position versus rest-delta-6D FK comparison GIFs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BLUE = (31, 112, 178)
ORANGE = (205, 83, 42)
AXIS_X = (210, 34, 34)
AXIS_Y = (31, 150, 55)
AXIS_Z = (30, 80, 210)


def matrix_from_cont6d(cont6d: np.ndarray) -> np.ndarray:
    first = cont6d[..., :3]
    second = cont6d[..., 3:6]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def decode_fk_positions(motion: np.ndarray, context: dict) -> np.ndarray:
    """Decode the release's per-joint rest-delta rot6d without repo dependencies."""
    rotations = matrix_from_cont6d(motion[:, :, 3:9].astype(np.float64))
    rotations = rotations @ context["rest_rotation"][None]
    parents = context["parents"]
    offsets = context["offset_parent_local"]
    positions = np.zeros((len(motion), len(parents), 3), dtype=np.float64)
    positions[:, 0] = motion[:, 0, :3]
    positions[:, 0, 0] += motion[:, 0, 13]
    positions[:, 0, 2] += motion[:, 0, 14]
    for joint in range(1, len(parents)):
        parent = int(parents[joint])
        positions[:, joint] = positions[:, parent] + np.einsum("tij,j->ti", rotations[:, parent], offsets[joint])
    return positions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--clip-id", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=72)
    parser.add_argument("--gif-fps", type=float, default=12.0)
    parser.add_argument("--width", type=int, default=880)
    parser.add_argument("--height", type=int, default=700)
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def sample_frames(length: int, maximum: int) -> list[int]:
    return sorted(set(np.linspace(0, length - 1, min(length, maximum), dtype=int).tolist()))


def view_uv(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Camera looks from -Y toward +Y; +Y remains screen-up after projection.
    return points[..., 0] - 0.36 * points[..., 2], points[..., 1] - 0.22 * points[..., 2]


def make_projection(series: list[np.ndarray], width: int, height: int) -> tuple[float, float, float]:
    all_points = np.concatenate([points.reshape(-1, 3) for points in series], axis=0)
    u, v = view_uv(all_points)
    span_u = max(float(u.max() - u.min()), 1e-6)
    span_v = max(float(v.max() - v.min()), 1e-6)
    scale = min(width * 0.80 / span_u, height * 0.76 / span_v)
    return scale, float((u.max() + u.min()) * 0.5), float((v.max() + v.min()) * 0.5)


def project(points: np.ndarray, transform: tuple[float, float, float], width: int, height: int) -> np.ndarray:
    scale, u_mid, v_mid = transform
    u, v = view_uv(points)
    return np.stack((width * 0.5 + (u - u_mid) * scale, height * 0.54 - (v - v_mid) * scale), axis=-1)


def draw_axes(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    origin = (int(width * 0.10), int(height * 0.85))
    length = int(min(width, height) * 0.14)
    draw.line((origin, (origin[0] + length, origin[1])), fill=AXIS_X, width=4)
    draw.line((origin, (origin[0], origin[1] - length)), fill=AXIS_Y, width=4)
    draw.line((origin, (origin[0] - int(length * 0.46), origin[1] + int(length * 0.65))), fill=AXIS_Z, width=4)
    label = load_font(18)
    draw.text((origin[0] + length + 5, origin[1] - 12), "+X", fill=AXIS_X, font=label)
    draw.text((origin[0] + 4, origin[1] - length - 22), "+Y", fill=AXIS_Y, font=label)
    draw.text((origin[0] - int(length * 0.46) - 32, origin[1] + int(length * 0.65) - 9), "+Z", fill=AXIS_Z, font=label)


def draw_skeleton(
    draw: ImageDraw.ImageDraw,
    positions: np.ndarray,
    parents: np.ndarray,
    projection: tuple[float, float, float],
    width: int,
    height: int,
    color: tuple[int, int, int],
    trail: np.ndarray,
) -> None:
    if len(trail) > 1:
        trail_2d = project(trail, projection, width, height)
        draw.line([tuple(point) for point in trail_2d], fill=(125, 125, 125), width=2)
    points = project(positions, projection, width, height)
    for joint, parent in enumerate(parents):
        if parent >= 0:
            draw.line([tuple(points[parent]), tuple(points[joint])], fill=color, width=3)
    root = points[0]
    draw.ellipse((root[0] - 5, root[1] - 5, root[0] + 5, root[1] + 5), fill=(20, 20, 20))


def panel(
    series: np.ndarray,
    parents: np.ndarray,
    frame: int,
    projection: tuple[float, float, float],
    width: int,
    height: int,
    title: str,
    color: tuple[int, int, int],
    axes: bool,
) -> Image.Image:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw_skeleton(draw, series[frame], parents, projection, width, height, color, series[: frame + 1, 0])
    if axes:
        draw_axes(draw, width, height)
    draw.text((18, 16), title, fill=color, font=load_font(23))
    return image


def load_context(root: Path, clip_id: str) -> tuple[np.ndarray, np.ndarray, dict, str]:
    motion_data = np.load(root / "motions" / f"{clip_id}.npz")
    motion = motion_data["motion"]
    manifest_path = root / "manifests" / "clips.jsonl"
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        item = __import__("json").loads(line)
        if item["clip_id"] == clip_id:
            skeleton = np.load(root / item["skeleton_file"])
            context = {
                "parents": skeleton["parents"].astype(int),
                "rest_rotation": skeleton["R_rest_global"].astype(float),
                "offset_parent_local": skeleton["offset_parent_local"].astype(float),
            }
            return motion, motion_data["heading_valid"], context, item["rig_id"]
    raise KeyError(f"Clip absent from manifest: {clip_id}")


def render_one(args: argparse.Namespace, clip_id: str) -> dict:
    motion, heading_valid, context, rig_id = load_context(args.dataset_root, clip_id)
    q_position = motion[:, :, :3].astype(float)
    q_position[:, :, 0] += motion[:, 0:1, 13]
    q_position[:, :, 2] += motion[:, 0:1, 14]
    rot_fk = decode_fk_positions(motion, context)
    error = np.linalg.norm(q_position - rot_fk, axis=-1)
    indices = sample_frames(len(motion), args.max_frames)
    root_trajectory = q_position[:, 0].copy()
    q_view = q_position.copy()
    fk_view = rot_fk.copy()
    for series in (q_view, fk_view):
        series[..., 0] -= root_trajectory[:, None, 0]
        series[..., 2] -= root_trajectory[:, None, 2]
    projection = make_projection([q_view[indices], fk_view[indices]], args.width, args.height)
    frames = []
    for frame_index in indices:
        left = panel(
            q_view,
            context["parents"],
            frame_index,
            projection,
            args.width,
            args.height,
            "world position (root XZ centered)",
            BLUE,
            True,
        )
        right = panel(
            fk_view,
            context["parents"],
            frame_index,
            projection,
            args.width,
            args.height,
            "canonical rest + rot6d FK",
            ORANGE,
            False,
        )
        frame = Image.new("RGB", (args.width * 2, args.height), "white")
        frame.paste(left, (0, 0))
        frame.paste(right, (args.width, 0))
        frames.append(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{clip_id}_qpos_vs_rest_delta6d_fk.gif"
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=int(1000 / args.gif_fps), loop=0)
    return {
        "clip_id": clip_id,
        "rig_id": rig_id,
        "frames": int(len(motion)),
        "heading_valid_frames": int(heading_valid.sum()),
        "fk_qposition_mae": float(error.mean()),
        "fk_qposition_max": float(error.max()),
        "gif": str(output),
    }


def main() -> None:
    args = parse_args()
    results = [render_one(args, clip_id) for clip_id in args.clip_id]
    print(__import__("json").dumps(results, indent=2))


if __name__ == "__main__":
    main()
