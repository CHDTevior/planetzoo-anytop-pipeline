"""One-command launcher: cleaned AnyTop minipack motion -> Planet Zoo MS2 preview.

Run with ordinary Python, not Blender Python.  It expands a raw minipack
prediction to the matching full rig and launches Blender's verified
NPY -> raw-BVH -> rest-relative MS2 skinning route.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", required=True, type=Path)
    parser.add_argument("--cobra-tools", required=True, type=Path)
    parser.add_argument("--motion-path", required=True, type=Path, help="Raw, de-normalised [T, J_min, 13] prediction.")
    parser.add_argument("--skeleton-path", required=True, type=Path, help="Matching minipack skeleton.json.")
    parser.add_argument("--resource-root", required=True, type=Path, help="Root of anytop13_planetzoo_skinning_resources_v1.")
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--debug-frame-dir", action="store_true")
    parser.add_argument("--show-world-axes", action="store_true")
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def main() -> None:
    args = parse_args()
    rig = args.resource_root / "rigs" / args.object_name
    paths = {
        "full_skeleton": rig / "full_skeleton.json",
        "ms2": rig / "model.ms2",
        "manis": rig / "reference_action.manis",
        "tpose": rig / "tpose.bvh",
        "reference_action": rig / "reference_action.bvh",
    }
    for path in [args.blender, args.cobra_tools / "__init__.py", args.motion_path, args.skeleton_path, *paths.values()]:
        require_file(path)
    args.output_root.mkdir(parents=True, exist_ok=True)
    expanded = args.output_root / "expanded_full_motion.npy"
    expand_report = args.output_root / "expanded_full_motion.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "expand_minipack_motion_to_full_rig.py"),
            "--motion-path",
            str(args.motion_path),
            "--skeleton-path",
            str(args.skeleton_path),
            "--full-skeleton-path",
            str(paths["full_skeleton"]),
            "--object-name",
            args.object_name,
            "--output-motion",
            str(expanded),
            "--output-report",
            str(expand_report),
        ],
        check=True,
    )
    command = [
        str(args.blender),
        "--background",
        "--python",
        str(SCRIPT_DIR / "build_planetzoo_anytop_npy_skinning_poc.py"),
        "--",
        "--cobra-tools",
        str(args.cobra_tools),
        "--ms2-path",
        str(paths["ms2"]),
        "--manis-path",
        str(paths["manis"]),
        "--motion-path",
        str(expanded),
        "--full-skeleton-path",
        str(paths["full_skeleton"]),
        "--object-name",
        args.object_name,
        "--tpose-bvh",
        str(paths["tpose"]),
        "--raw-template-bvh",
        str(paths["reference_action"]),
        "--output-raw-bvh",
        str(args.output_root / "decoded_raw.bvh"),
        "--output-blend",
        str(args.output_root / "mesh_preview.blend"),
        "--output-mp4",
        str(args.output_root / "mesh_preview.mp4"),
        "--output-report",
        str(args.output_root / "mesh_preview.json"),
        "--max-frames",
        str(args.max_frames),
        "--fps",
        str(args.fps),
    ]
    if args.debug_frame_dir:
        command.extend(["--debug-frame-dir", str(args.output_root / "frames")])
    if args.show_world_axes:
        command.append("--show-world-axes")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
