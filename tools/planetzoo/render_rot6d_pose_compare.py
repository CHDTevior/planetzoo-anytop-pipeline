"""Render large pose-vs-rot6d-FK comparison GIFs for pooled AnyTop data."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loaders.truebones.truebones_utils.motion_process import (  # noqa: E402
    recover_from_bvh_ric_np,
    recover_from_bvh_rot_np,
)

POSE_COLOR = (35, 112, 180)
ROTFK_COLOR = (210, 83, 45)
ROOT_COLOR = (18, 18, 18)
TRAIL_COLOR = (128, 128, 128)
GROUND_COLOR = (224, 224, 224)
AXIS_X = (210, 34, 34)
AXIS_Y = (30, 150, 55)
AXIS_Z = (30, 80, 210)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-root", required=True, type=Path)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--include-regex", default="")
    parser.add_argument("--motion-name", action="append", default=[])
    parser.add_argument("--max-motions", type=int, default=None)
    parser.add_argument("--cell-width", type=int, default=900)
    parser.add_argument("--cell-height", type=int, default=760)
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--pad", type=float, default=0.06)
    parser.add_argument("--zoom", type=float, default=1.15)
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument("--joint-radius", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--apply-root-cancel", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return re.sub(r"_+", "_", value) or "motion"


def action_label(path: Path, object_name: str) -> str:
    stem = path.stem
    prefix = object_name + "_"
    if stem.startswith(prefix):
        stem = stem[len(prefix) :]
    parts = stem.split("__")
    tail = parts[-1] if parts else stem
    return tail.rsplit("_", 1)[0]


def load_cond(layout_root: Path, object_name: str) -> tuple[np.ndarray, np.ndarray]:
    cond = np.load(layout_root / "cond.npy", allow_pickle=True).item()
    if object_name not in cond:
        raise KeyError(f"{object_name} not found in {layout_root / 'cond.npy'}")
    entry = cond[object_name]
    return np.asarray(entry["parents"], dtype=int), np.asarray(entry["offsets"], dtype=float)


def select_motions(args: argparse.Namespace) -> list[Path]:
    motions_dir = args.layout_root / "motions"
    if args.motion_name:
        selected = []
        for name in args.motion_name:
            path = motions_dir / name
            if not path.suffix:
                path = path.with_suffix(".npy")
            selected.append(path)
    else:
        selected = sorted(motions_dir.glob(f"{args.object_name}*.npy"))
        if args.include_regex:
            pattern = re.compile(args.include_regex, re.IGNORECASE)
            selected = [path for path in selected if pattern.search(path.name)]
    selected = [path for path in selected if path.is_file()]
    if args.max_motions is not None:
        selected = selected[: args.max_motions]
    return selected


def sample_indices(length: int, max_frames: int) -> list[int]:
    if length <= 0:
        return []
    if length <= max_frames:
        return list(range(length))
    return sorted(set(int(round(v)) for v in np.linspace(0, length - 1, max_frames)))


def normalize_ground(positions: np.ndarray) -> np.ndarray:
    result = positions.copy()
    result[..., 1] -= result[..., 1].min()
    return result


def view_uv(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    return x - 0.36 * z, y - 0.22 * z


def compute_transform(point_sets: list[np.ndarray], size: tuple[int, int], pad: float, zoom: float) -> tuple[float, float, float]:
    points = np.concatenate([points.reshape(-1, 3) for points in point_sets if points.size], axis=0)
    u, v = view_uv(points)
    u_min, u_max = float(u.min()), float(u.max())
    v_min, v_max = float(v.min()), float(v.max())
    width, height = size
    u_span = max(u_max - u_min, 1e-6)
    v_span = max(v_max - v_min, 1e-6)
    base_scale = min(width * (1.0 - 2.0 * pad) / u_span, height * (1.0 - 2.0 * pad) / v_span)
    return base_scale * zoom, (u_min + u_max) * 0.5, (v_min + v_max) * 0.5


def project(points: np.ndarray, transform: tuple[float, float, float], size: tuple[int, int]) -> np.ndarray:
    scale, u_mid, v_mid = transform
    width, height = size
    u, v = view_uv(points)
    px = width * 0.5 + (u - u_mid) * scale
    py = height * 0.54 - (v - v_mid) * scale
    return np.stack([px, py], axis=-1)


def font(size: int) -> ImageFont.ImageFont:
    for candidate in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def draw_axes(draw: ImageDraw.ImageDraw, size: tuple[int, int]) -> None:
    width, height = size
    ox = int(width * 0.10)
    oy = int(height * 0.82)
    length = int(min(width, height) * 0.14)
    draw.line([(ox, oy), (ox + length, oy)], fill=AXIS_X, width=4)
    draw.line([(ox, oy), (ox, oy - length)], fill=AXIS_Y, width=4)
    draw.line([(ox, oy), (ox - int(0.45 * length), oy + int(0.65 * length))], fill=AXIS_Z, width=4)
    label_font = font(18)
    draw.text((ox + length + 6, oy - 10), "+X", fill=AXIS_X, font=label_font)
    draw.text((ox + 5, oy - length - 22), "+Y", fill=AXIS_Y, font=label_font)
    draw.text((ox - int(0.45 * length) - 30, oy + int(0.65 * length) - 7), "+Z", fill=AXIS_Z, font=label_font)


def draw_ground(draw: ImageDraw.ImageDraw, transform: tuple[float, float, float], size: tuple[int, int], radius: float) -> None:
    for value in np.linspace(-radius, radius, 7):
        for line in [
            np.array([[-radius, 0.0, value], [radius, 0.0, value]], dtype=float),
            np.array([[value, 0.0, -radius], [value, 0.0, radius]], dtype=float),
        ]:
            pts = project(line, transform, size)
            draw.line([tuple(pts[0]), tuple(pts[1])], fill=GROUND_COLOR, width=1)


def draw_skeleton(
    draw: ImageDraw.ImageDraw,
    positions: np.ndarray,
    parents: np.ndarray,
    transform: tuple[float, float, float],
    size: tuple[int, int],
    color: tuple[int, int, int],
    line_width: int,
    joint_radius: int,
    trail: np.ndarray | None,
) -> None:
    if trail is not None and len(trail) > 1:
        trail_2d = project(trail, transform, size)
        draw.line([tuple(point) for point in trail_2d], fill=TRAIL_COLOR, width=max(2, line_width - 1))
    pts = project(positions, transform, size)
    for joint in range(1, len(parents)):
        parent = int(parents[joint])
        if parent >= 0:
            draw.line([tuple(pts[parent]), tuple(pts[joint])], fill=color, width=line_width)
    root = tuple(pts[0])
    r = joint_radius
    draw.ellipse((root[0] - r, root[1] - r, root[0] + r, root[1] + r), fill=ROOT_COLOR)


def render_panel(
    positions: np.ndarray,
    parents: np.ndarray,
    frame_index: int,
    transform: tuple[float, float, float],
    size: tuple[int, int],
    title: str,
    color: tuple[int, int, int],
    line_width: int,
    joint_radius: int,
    axes: bool,
) -> Image.Image:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    root = positions[:, 0].copy()
    radius = max(2.0, float(np.linalg.norm(positions.reshape(-1, 3)[:, [0, 2]], axis=-1).max()) * 1.05)
    draw_ground(draw, transform, size, radius=radius)
    frame = positions[frame_index]
    trail = root.copy()
    trail[:, 1] = 0.0
    draw_skeleton(draw, frame, parents, transform, size, color, line_width, joint_radius, trail)
    if axes:
        draw_axes(draw, size)
    title_font = font(24)
    draw.text((18, 16), title, fill=color, font=title_font)
    return image


def make_frame(
    pose: np.ndarray,
    rotfk: np.ndarray,
    parents: np.ndarray,
    frame_index: int,
    transform: tuple[float, float, float],
    args: argparse.Namespace,
    label: str,
) -> Image.Image:
    cell_size = (args.cell_width, args.cell_height)
    pose_series = pose.copy()
    rotfk_series = rotfk.copy()
    root = pose_series[frame_index, 0].copy()
    for series in [pose_series, rotfk_series]:
        series[..., 0] -= root[0]
        series[..., 2] -= root[2]
    pose_panel = render_panel(
        pose_series,
        parents,
        frame_index,
        transform,
        cell_size,
        f"pose 0:3 | {label}",
        POSE_COLOR,
        args.line_width,
        args.joint_radius,
        axes=True,
    )
    rot_panel = render_panel(
        rotfk_series,
        parents,
        frame_index,
        transform,
        cell_size,
        "rot6d-FK 3:9",
        ROTFK_COLOR,
        args.line_width,
        args.joint_radius,
        axes=False,
    )
    image = Image.new("RGB", (args.cell_width * 2, args.cell_height), "white")
    image.paste(pose_panel, (0, 0))
    image.paste(rot_panel, (args.cell_width, 0))
    return image


def render_motion(path: Path, parents: np.ndarray, offsets: np.ndarray, args: argparse.Namespace) -> dict:
    data = np.load(path, allow_pickle=True)
    pose = normalize_ground(recover_from_bvh_ric_np(data))
    rotfk, _ = recover_from_bvh_rot_np(data, parents, offsets, apply_root_cancel=args.apply_root_cancel)
    rotfk = normalize_ground(rotfk)
    errors = np.linalg.norm(pose - rotfk, axis=-1)

    indices = sample_indices(len(pose), args.max_frames)
    roots = pose[:, 0].copy()
    pose_centered = pose.copy()
    rotfk_centered = rotfk.copy()
    for series in [pose_centered, rotfk_centered]:
        series[..., 0] -= roots[:, None, 0]
        series[..., 2] -= roots[:, None, 2]
    transform = compute_transform(
        [pose_centered[index] for index in indices] + [rotfk_centered[index] for index in indices],
        (args.cell_width, args.cell_height),
        args.pad,
        args.zoom,
    )

    label = action_label(path, args.object_name)
    out_path = args.output_dir / f"{args.object_name}_{safe_name(label)}_pose_vs_rot6d_fk_large.gif"
    if out_path.exists() and args.skip_existing:
        return {"motion": path.name, "output": str(out_path), "skipped": True}

    frames = [make_frame(pose, rotfk, parents, idx, transform, args, label) for idx in indices]
    duration_ms = int(round(1000.0 / max(args.fps, 1e-6)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return {
        "motion": path.name,
        "output": str(out_path),
        "shape": [int(v) for v in data.shape],
        "rendered_frames": len(frames),
        "mean_error": float(errors.mean()),
        "p95_error": float(np.percentile(errors, 95)),
        "max_error": float(errors.max()),
    }


def main() -> None:
    args = parse_args()
    parents, offsets = load_cond(args.layout_root, args.object_name)
    motions = select_motions(args)
    if not motions:
        raise FileNotFoundError("No motions matched the requested filters.")
    results = [render_motion(path, parents, offsets, args) for path in motions]
    summary = {
        "layout_root": str(args.layout_root),
        "object_name": args.object_name,
        "output_dir": str(args.output_dir),
        "motion_count": len(results),
        "cell_width": args.cell_width,
        "cell_height": args.cell_height,
        "zoom": args.zoom,
        "max_frames": args.max_frames,
        "results": results,
    }
    summary_path = args.output_dir / f"{args.object_name}_pose_vs_rot6d_fk_large_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
