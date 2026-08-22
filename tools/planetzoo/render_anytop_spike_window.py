"""Render a short AnyTop [T, J, 13] window around a suspected pose spike.

The left panel reconstructs the stored RIC position channels (0:3).  The
right panel reconstructs the stored 6D rotations (3:9) with FK.  Rendering
both is useful for separating an upstream source-pose error from a later
AnyTop conversion or FK error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


RIC_COLOR = (35, 112, 180)
FK_COLOR = (210, 83, 45)
HIGHLIGHT_COLOR = (218, 41, 41)
AXIS_X = (210, 34, 34)
AXIS_Y = (30, 150, 55)
AXIS_Z = (30, 80, 210)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-root", required=True, type=Path)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--motion", required=True, type=Path)
    parser.add_argument("--joint", required=True)
    parser.add_argument("--frame-start", required=True, type=int)
    parser.add_argument("--frame-end", required=True, type=int)
    parser.add_argument("--output-gif", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=760)
    parser.add_argument("--fps", type=float, default=2.0)
    return parser.parse_args()


def font(size: int) -> ImageFont.ImageFont:
    for candidate in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    valid = norm > 1e-8
    return np.where(valid, vector / np.maximum(norm, 1e-8), fallback)


def rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    first = normalize(rotation_6d[..., :3], np.array([1.0, 0.0, 0.0]))
    third = normalize(np.cross(first, rotation_6d[..., 3:6]), np.array([0.0, 0.0, 1.0]))
    second = np.cross(third, first)
    return np.stack([first, second, third], axis=-1)


def apply_inverse_rotation(rotation: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.einsum("...ji,...j->...i", rotation, points)


def apply_rotation(rotation: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.einsum("...ij,...j->...i", rotation, points)


def recover_root(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    root_rotation = rotation_6d_to_matrix(data[:, 0, 3:9])
    root_position = np.zeros((len(data), 3), dtype=np.float64)
    root_position[1:, [0, 2]] = data[:-1, 0, [9, 11]]
    root_position = apply_inverse_rotation(root_rotation, root_position)
    root_position = np.cumsum(root_position, axis=0)
    root_position[:, 1] = data[:, 0, 1]
    return root_rotation, root_position


def recover_ric_positions(data: np.ndarray) -> np.ndarray:
    root_rotation, root_position = recover_root(data)
    relative = data[:, 1:, :3].astype(np.float64, copy=True)
    relative = apply_inverse_rotation(root_rotation[:, None], relative)
    relative[..., 0] += root_position[:, None, 0]
    relative[..., 2] += root_position[:, None, 2]
    return np.concatenate([root_position[:, None], relative], axis=1)


def recover_rot6d_fk(data: np.ndarray, parents: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    root_rotation, root_position = recover_root(data)
    child_tokens = rotation_6d_to_matrix(data[:, 1:, 3:9])
    frames, joints = data.shape[:2]
    local_rotation = np.broadcast_to(np.eye(3), (frames, joints, 3, 3)).copy()
    local_rotation[:, 0] = root_rotation
    # AnyTop stores a parent's local rotation on each of its children.  The
    # duplicate tokens agree; this loop deliberately follows the decoder.
    for child, parent in enumerate(parents[1:], start=1):
        local_rotation[:, parent] = child_tokens[:, child - 1]

    global_rotation = np.empty_like(local_rotation)
    positions = np.empty((frames, joints, 3), dtype=np.float64)
    global_rotation[:, 0] = local_rotation[:, 0]
    positions[:, 0] = root_position
    for joint, parent in enumerate(parents[1:], start=1):
        global_rotation[:, joint] = np.matmul(global_rotation[:, parent], local_rotation[:, joint])
        positions[:, joint] = positions[:, parent] + apply_rotation(global_rotation[:, parent], offsets[joint])
    return positions


def descendants(parents: np.ndarray, root: int) -> set[int]:
    result = {root}
    changed = True
    while changed:
        changed = False
        for joint, parent in enumerate(parents):
            if parent in result and joint not in result:
                result.add(joint)
                changed = True
    return result


def view_uv(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Same fixed AnyTop visual convention as render_rot6d_pose_compare.py:
    # +Y is screen-up, +X is right, and +Z projects down-left.
    return points[..., 0] - 0.36 * points[..., 2], points[..., 1] - 0.22 * points[..., 2]


def make_transform(point_sets: list[np.ndarray], width: int, height: int) -> tuple[float, float, float]:
    points = np.concatenate([item.reshape(-1, 3) for item in point_sets], axis=0)
    u, v = view_uv(points)
    span_u = max(float(u.max() - u.min()), 1e-6)
    span_v = max(float(v.max() - v.min()), 1e-6)
    scale = min(width * 0.82 / span_u, height * 0.76 / span_v)
    return scale, float((u.min() + u.max()) * 0.5), float((v.min() + v.max()) * 0.5)


def project(points: np.ndarray, transform: tuple[float, float, float], width: int, height: int) -> np.ndarray:
    scale, mid_u, mid_v = transform
    u, v = view_uv(points)
    return np.stack([width * 0.5 + (u - mid_u) * scale, height * 0.55 - (v - mid_v) * scale], axis=-1)


def draw_axes(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    origin = (int(width * 0.10), int(height * 0.84))
    length = int(min(width, height) * 0.13)
    draw.line([origin, (origin[0] + length, origin[1])], fill=AXIS_X, width=4)
    draw.line([origin, (origin[0], origin[1] - length)], fill=AXIS_Y, width=4)
    draw.line([origin, (origin[0] - int(length * 0.45), origin[1] + int(length * 0.65))], fill=AXIS_Z, width=4)
    label = font(18)
    draw.text((origin[0] + length + 5, origin[1] - 10), "+X", fill=AXIS_X, font=label)
    draw.text((origin[0] + 5, origin[1] - length - 22), "+Y", fill=AXIS_Y, font=label)
    draw.text((origin[0] - int(length * 0.45) - 30, origin[1] + int(length * 0.65) - 8), "+Z", fill=AXIS_Z, font=label)


def render_panel(
    positions: np.ndarray,
    parents: np.ndarray,
    frame: int,
    target: int,
    affected: set[int],
    transform: tuple[float, float, float],
    size: tuple[int, int],
    title: str,
    color: tuple[int, int, int],
) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    projected = project(positions[frame], transform, width, height)
    for joint, parent in enumerate(parents[1:], start=1):
        edge_color = HIGHLIGHT_COLOR if joint in affected else color
        draw.line([tuple(projected[parent]), tuple(projected[joint])], fill=edge_color, width=4 if joint in affected else 3)
    for joint in affected:
        x, y = projected[joint]
        radius = 7 if joint == target else 4
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=HIGHLIGHT_COLOR)
    x, y = projected[0]
    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(18, 18, 18))
    draw_axes(draw, width, height)
    draw.text((18, 16), title, fill=color, font=font(24))
    return image


def resolve_motion(layout_root: Path, value: Path) -> Path:
    if value.is_file():
        return value
    candidate = layout_root / "motions" / value
    if candidate.suffix != ".npy":
        candidate = candidate.with_suffix(".npy")
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(value)


def main() -> None:
    args = parse_args()
    motion_path = resolve_motion(args.layout_root, args.motion)
    cond = np.load(args.layout_root / "cond.npy", allow_pickle=True).item()
    entry = cond[args.object_name]
    names = list(entry["joints_names"])
    parents = np.asarray(entry["parents"], dtype=np.int64)
    offsets = np.asarray(entry["offsets"], dtype=np.float64)
    if args.joint not in names:
        raise KeyError(f"{args.joint!r} is not in {args.object_name}")
    target = names.index(args.joint)
    data = np.load(motion_path, allow_pickle=False)
    if data.ndim != 3 or data.shape[1:] != (len(parents), 13):
        raise ValueError(f"Expected [T, {len(parents)}, 13], got {list(data.shape)}")
    start = max(0, args.frame_start)
    end = min(len(data) - 1, args.frame_end)
    if end < start:
        raise ValueError("The requested frame window is empty.")

    ric = recover_ric_positions(data)
    fk = recover_rot6d_fk(data, parents, offsets)
    affected = descendants(parents, target)
    roots = ric[:, 0].copy()
    for series in (ric, fk):
        series[..., 0] -= roots[:, None, 0]
        series[..., 2] -= roots[:, None, 2]
    frames = list(range(start, end + 1))
    transform = make_transform([ric[frames], fk[frames]], args.width, args.height)
    images = []
    error = np.linalg.norm(ric - fk, axis=-1)
    for frame in frames:
        left = render_panel(ric, parents, frame, target, affected, transform, (args.width, args.height), "RIC positions 0:3", RIC_COLOR)
        right = render_panel(fk, parents, frame, target, affected, transform, (args.width, args.height), "rot6d FK 3:9", FK_COLOR)
        image = Image.new("RGB", (args.width * 2, args.height), "white")
        image.paste(left, (0, 0))
        image.paste(right, (args.width, 0))
        draw = ImageDraw.Draw(image)
        step = 0.0 if frame == 0 else float(np.linalg.norm(ric[frame, target] - ric[frame - 1, target]))
        text = (
            f"{args.object_name} | frame {frame} | red: {args.joint} + descendants | "
            f"target world step={step:.5f} | RIC-FK max={float(error[frame].max()):.2e}"
        )
        draw.rectangle((0, args.height - 40, image.width, args.height), fill=(250, 250, 250))
        draw.text((18, args.height - 33), text, fill=(20, 20, 20), font=font(20))
        images.append(image)

    args.output_gif.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        args.output_gif,
        save_all=True,
        append_images=images[1:],
        duration=int(round(1000.0 / max(args.fps, 1e-6))),
        loop=0,
        optimize=False,
    )
    report = {
        "motion": str(motion_path),
        "object_name": args.object_name,
        "shape": [int(value) for value in data.shape],
        "joint": args.joint,
        "joint_index": target,
        "highlighted_joint_count": len(affected),
        "frames": frames,
        "max_ric_fk_error": float(error.max()),
        "window_max_ric_fk_error": float(error[frames].max()),
        "target_world_step_by_frame": {
            str(frame): 0.0 if frame == 0 else float(np.linalg.norm(ric[frame, target] - ric[frame - 1, target]))
            for frame in frames
        },
        "gif": str(args.output_gif),
    }
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
