"""Render compact VLM preview images for AnyTop-format Planet Zoo data."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np


EDGE_COLOR = (210, 83, 45)
ROOT_COLOR = (35, 112, 180)
TRAIL_COLOR = (145, 145, 145)
GROUND_COLOR = (226, 226, 226)
AXIS_X = (210, 34, 34)
AXIS_Y = (30, 150, 55)
AXIS_Z = (30, 80, 210)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--motion-manifest", default=None)
    parser.add_argument("--objects", nargs="*", default=None)
    parser.add_argument("--max-actions-per-object", type=int, default=None)
    parser.add_argument("--frames-per-action", type=int, default=8)
    parser.add_argument("--cell-size", type=int, default=240)
    parser.add_argument("--rest-size", type=int, default=512)
    parser.add_argument("--quality", type=int, default=88)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-rest", action="store_true")
    parser.add_argument("--no-actions", action="store_true")
    return parser.parse_args()


def safe_name(value: str, max_len: int = 90) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    value = re.sub(r"_+", "_", value)
    return (value[:max_len].rstrip("_") or "unnamed")


def positions_from_offsets(offsets: np.ndarray, parents: np.ndarray) -> np.ndarray:
    positions = np.zeros_like(offsets, dtype=float)
    for joint, parent in enumerate(parents):
        if parent >= 0:
            positions[joint] = positions[int(parent)] + offsets[joint]
        else:
            positions[joint] = offsets[joint]
    return positions


def normalize_ground(positions: np.ndarray) -> np.ndarray:
    positions = positions.copy()
    positions[..., 1] -= positions[..., 1].min()
    return positions


def view_uv(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    return x - 0.36 * z, y - 0.22 * z


def compute_transform(point_sets: list[np.ndarray], size: tuple[int, int], pad: float = 0.10) -> tuple[float, float, float]:
    points = np.concatenate([p.reshape(-1, 3) for p in point_sets if p.size], axis=0)
    u, v = view_uv(points)
    u_min, u_max = float(u.min()), float(u.max())
    v_min, v_max = float(v.min()), float(v.max())
    u_span = max(u_max - u_min, 1e-6)
    v_span = max(v_max - v_min, 1e-6)
    width, height = size
    scale = min(width * (1.0 - 2.0 * pad) / u_span, height * (1.0 - 2.0 * pad) / v_span)
    return scale, (u_min + u_max) * 0.5, (v_min + v_max) * 0.5


def project(points: np.ndarray, transform: tuple[float, float, float], size: tuple[int, int]) -> np.ndarray:
    scale, u_mid, v_mid = transform
    width, height = size
    u, v = view_uv(points)
    px = width * 0.5 + (u - u_mid) * scale
    py = height * 0.5 - (v - v_mid) * scale
    return np.stack([px, py], axis=-1)


def draw_axes(draw: ImageDraw.ImageDraw, size: tuple[int, int]) -> None:
    width, height = size
    ox = int(width * 0.12)
    oy = int(height * 0.82)
    length = int(min(width, height) * 0.17)
    draw.line([(ox, oy), (ox + length, oy)], fill=AXIS_X, width=3)
    draw.line([(ox, oy), (ox, oy - length)], fill=AXIS_Y, width=3)
    draw.line([(ox, oy), (ox - int(0.45 * length), oy + int(0.65 * length))], fill=AXIS_Z, width=3)
    draw.text((ox + length + 3, oy - 8), "+X", fill=AXIS_X)
    draw.text((ox + 3, oy - length - 14), "+Y", fill=AXIS_Y)
    draw.text((ox - int(0.45 * length) - 20, oy + int(0.65 * length) - 4), "+Z", fill=AXIS_Z)


def draw_ground(draw: ImageDraw.ImageDraw, transform: tuple[float, float, float], size: tuple[int, int], radius: float = 2.0) -> None:
    lines = []
    for v in np.linspace(-radius, radius, 5):
        lines.append(np.array([[-radius, 0.0, v], [radius, 0.0, v]], dtype=float))
        lines.append(np.array([[v, 0.0, -radius], [v, 0.0, radius]], dtype=float))
    for line in lines:
        pts = project(line, transform, size)
        draw.line([tuple(pts[0]), tuple(pts[1])], fill=GROUND_COLOR, width=1)


def draw_skeleton(
    draw: ImageDraw.ImageDraw,
    positions: np.ndarray,
    parents: np.ndarray,
    transform: tuple[float, float, float],
    size: tuple[int, int],
    trail: np.ndarray | None = None,
    axes: bool = False,
) -> None:
    draw_ground(draw, transform, size)
    if trail is not None and len(trail) > 1:
        trail_2d = project(trail, transform, size)
        draw.line([tuple(p) for p in trail_2d], fill=TRAIL_COLOR, width=2)
    pts = project(positions, transform, size)
    for joint in range(1, len(parents)):
        parent = int(parents[joint])
        if parent >= 0:
            draw.line([tuple(pts[parent]), tuple(pts[joint])], fill=EDGE_COLOR, width=2)
    root = tuple(pts[0])
    draw.ellipse((root[0] - 3, root[1] - 3, root[0] + 3, root[1] + 3), fill=ROOT_COLOR)
    if axes:
        draw_axes(draw, size)


def save_image(image: Image.Image, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=quality, optimize=True)


def render_rest(rest_positions: np.ndarray, parents: np.ndarray, out_path: Path, size: int, quality: int, overwrite: bool) -> None:
    if out_path.exists() and not overwrite:
        return
    rest = normalize_ground(rest_positions)
    rest[:, 0] -= rest[0, 0]
    rest[:, 2] -= rest[0, 2]
    canvas_size = (size, size)
    transform = compute_transform([rest], canvas_size)
    image = Image.new("RGB", canvas_size, "white")
    draw = ImageDraw.Draw(image)
    draw_skeleton(draw, rest, parents, transform, canvas_size, axes=True)
    save_image(image, out_path, quality)


def sample_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        return []
    if count <= 1:
        return [0]
    return sorted(set(int(round(v)) for v in np.linspace(0, length - 1, count)))


def render_action(
    motion_path: Path,
    parents: np.ndarray,
    out_path: Path,
    frames_per_action: int,
    cell_size: int,
    quality: int,
    overwrite: bool,
) -> dict:
    motion = np.load(motion_path, allow_pickle=True)
    positions = recover_from_bvh_ric_np(motion)
    positions = normalize_ground(positions)
    indices = sample_indices(len(positions), frames_per_action)
    if not indices:
        raise ValueError(f"Empty motion: {motion_path}")

    roots = positions[:, 0].copy()
    sampled_frames = []
    sampled_trails = []
    for idx in indices:
        frame = positions[idx].copy()
        frame[:, 0] -= roots[idx, 0]
        frame[:, 2] -= roots[idx, 2]
        trail = roots.copy()
        trail[:, 0] -= roots[idx, 0]
        trail[:, 1] = 0.0
        trail[:, 2] -= roots[idx, 2]
        sampled_frames.append(frame)
        sampled_trails.append(trail)

    if not out_path.exists() or overwrite:
        transform = compute_transform(sampled_frames + sampled_trails, (cell_size, cell_size))
        image = Image.new("RGB", (cell_size * len(indices), cell_size), "white")
        for panel, (frame, trail) in enumerate(zip(sampled_frames, sampled_trails)):
            cell = Image.new("RGB", (cell_size, cell_size), "white")
            draw = ImageDraw.Draw(cell)
            draw_skeleton(draw, frame, parents, transform, (cell_size, cell_size), trail=trail, axes=(panel == 0))
            image.paste(cell, (panel * cell_size, 0))
        save_image(image, out_path, quality)

    return {
        "frames": int(len(positions)),
        "sampled_indices": indices,
        "motion_shape": [int(v) for v in motion.shape],
    }


def load_motion_metadata(path: Path | None) -> dict[str, dict]:
    if path is None or not path.is_file():
        return {}
    meta = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            motion = row.get("processed_motion")
            if motion:
                meta[str(Path(motion).resolve()).lower()] = row
    return meta


def object_dirs(processed_root: Path, requested: list[str] | None) -> list[Path]:
    if requested:
        return [processed_root / name for name in requested]
    return sorted([p for p in processed_root.glob("PZ_*") if (p / "cond.npy").is_file()])


def preview_name(index: int, motion_path: Path, metadata: dict | None) -> str:
    action = ""
    if metadata:
        action = metadata.get("action_short") or metadata.get("action_name") or ""
    if not action:
        action = motion_path.stem
    digest = hashlib.md5(str(motion_path).encode("utf-8")).hexdigest()[:10]
    return f"{index:05d}_{safe_name(action, 80)}_{digest}.jpg"


def process_object(args_tuple: tuple) -> dict:
    (
        object_dir,
        output_root,
        object_meta,
        max_actions,
        frames_per_action,
        cell_size,
        rest_size,
        quality,
        overwrite,
        render_rest_flag,
        render_actions_flag,
    ) = args_tuple
    object_dir = Path(object_dir)
    output_root = Path(output_root)
    object_name = object_dir.name
    cond = np.load(object_dir / "cond.npy", allow_pickle=True).item()
    _, data = next(iter(cond.items()))
    parents = np.asarray(data["parents"], dtype=int)
    offsets = np.asarray(data["offsets"], dtype=float)
    rest_positions = positions_from_offsets(offsets, parents)

    rows = []
    rest_path = output_root / "rest" / f"{object_name}.jpg"
    if render_rest_flag:
        render_rest(rest_positions, parents, rest_path, rest_size, quality, overwrite)
        rows.append(
            {
                "sample_type": "rest",
                "object_name": object_name,
                "nodes": int(len(parents)),
                "preview_path": str(rest_path),
                "cond_path": str(object_dir / "cond.npy"),
            }
        )

    action_count = 0
    if render_actions_flag:
        motion_paths = sorted((object_dir / "motions").glob("*.npy"))
        if max_actions is not None:
            motion_paths = motion_paths[:max_actions]
        action_dir = output_root / "actions" / object_name
        for idx, motion_path in enumerate(motion_paths, start=1):
            metadata = object_meta.get(str(motion_path.resolve()).lower(), {})
            out_path = action_dir / preview_name(idx, motion_path, metadata)
            stats = render_action(motion_path, parents, out_path, frames_per_action, cell_size, quality, overwrite)
            row = {
                "sample_type": "motion",
                "object_name": object_name,
                "nodes": int(len(parents)),
                "motion_path": str(motion_path),
                "preview_path": str(out_path),
                "frames": stats["frames"],
                "sampled_indices": stats["sampled_indices"],
                "motion_shape": stats["motion_shape"],
                "cond_path": str(object_dir / "cond.npy"),
            }
            for key in [
                "action_name",
                "action_short",
                "source_motion_key",
                "raw_bvh",
                "processed_bvh",
                "text",
                "text_match_status",
                "animal_key",
            ]:
                if key in metadata:
                    row[key] = metadata[key]
            rows.append(row)
            action_count += 1

    return {
        "object_name": object_name,
        "rest_count": 1 if render_rest_flag else 0,
        "action_count": action_count,
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    processed_root = Path(args.processed_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    motion_manifest = Path(args.motion_manifest) if args.motion_manifest else processed_root / "motion_text_manifest.jsonl"
    metadata = load_motion_metadata(motion_manifest)

    dirs = object_dirs(processed_root, args.objects)
    grouped_meta: dict[str, dict[str, dict]] = {}
    for obj_dir in dirs:
        grouped_meta[obj_dir.name] = {}
    for motion_key, row in metadata.items():
        object_name = Path(row.get("processed_motion", "")).parents[1].name if row.get("processed_motion") else ""
        if object_name in grouped_meta:
            grouped_meta[object_name][motion_key] = row

    manifest_jsonl = output_root / "vlm_preview_manifest.jsonl"
    manifest_csv = output_root / "vlm_preview_manifest.csv"
    worker_args = [
        (
            str(obj_dir),
            str(output_root),
            grouped_meta.get(obj_dir.name, {}),
            args.max_actions_per_object,
            args.frames_per_action,
            args.cell_size,
            args.rest_size,
            args.quality,
            args.overwrite,
            not args.no_rest,
            not args.no_actions,
        )
        for obj_dir in dirs
    ]

    total_actions = 0
    total_rest = 0
    fieldnames = [
        "sample_type",
        "object_name",
        "nodes",
        "preview_path",
        "motion_path",
        "frames",
        "sampled_indices",
        "motion_shape",
        "action_short",
        "action_name",
        "source_motion_key",
        "animal_key",
        "text",
        "text_match_status",
        "raw_bvh",
        "processed_bvh",
        "cond_path",
    ]
    with manifest_jsonl.open("w", encoding="utf-8") as jsonl_f, manifest_csv.open("w", encoding="utf-8", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_object, item) for item in worker_args]
            for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                result = future.result()
                total_actions += result["action_count"]
                total_rest += result["rest_count"]
                for row in result["rows"]:
                    jsonl_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    writer.writerow(row)
                jsonl_f.flush()
                csv_f.flush()
                print(
                    json.dumps(
                        {
                            "completed_objects": completed,
                            "total_objects": len(futures),
                            "object_name": result["object_name"],
                            "actions": result["action_count"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    summary = {
        "processed_root": str(processed_root),
        "output_root": str(output_root),
        "objects": len(dirs),
        "rest_previews": total_rest,
        "action_previews": total_actions,
        "frames_per_action": args.frames_per_action,
        "cell_size": args.cell_size,
        "rest_size": args.rest_size,
        "manifest_jsonl": str(manifest_jsonl),
        "manifest_csv": str(manifest_csv),
    }
    (output_root / "vlm_preview_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
