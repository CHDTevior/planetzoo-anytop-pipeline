"""Write a corrected no-IK release with canonical rest poses.

The original no-IK release has a rig-wide constant basis in ``R_rest_global``.
That basis was unintentionally retained in ``P_rest_global`` while the stored
offsets already describe the intended +Y-up/+Z-forward rest skeleton.  This
tool writes a *new* release: it replaces that rest basis by identity, converts
the rot6d channel to the equivalent source-global rotations, and recomputes
the per-rig normalization statistics.  Position, velocity, contact, root and
heading channels are copied exactly.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np


CHANNELS = 17
STD_MIN = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="Original release root, or its data/ directory.")
    parser.add_argument("--output-root", type=Path, required=True, help="New, empty release root (never modifies input).")
    parser.add_argument("--workers", type=int, default=4, help="Independent rig workers. Use 1 for lowest RAM usage.")
    parser.add_argument("--rig", action="append", default=[], help="Optional rig ID. May be repeated for a small test run.")
    return parser.parse_args()


def data_root_from(root: Path) -> tuple[Path, bool]:
    has_release_wrapper = (root / "data").is_dir()
    return (root / "data" if has_release_wrapper else root), has_release_wrapper


def cont6d_from_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.swapaxes(matrix[..., :, :2], -1, -2).reshape(*matrix.shape[:-2], 6)


def matrix_from_cont6d(cont6d: np.ndarray) -> np.ndarray:
    first = cont6d[..., :3]
    second = cont6d[..., 3:6]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
    return np.stack((first, second, np.cross(first, second)), axis=-1)


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


class RunningStats:
    def __init__(self, joints: int) -> None:
        self.count = np.zeros((joints, CHANNELS), dtype=np.int64)
        self.sum = np.zeros((joints, CHANNELS), dtype=np.float64)
        self.sumsq = np.zeros((joints, CHANNELS), dtype=np.float64)
        self.minimum = np.full((joints, CHANNELS), np.inf, dtype=np.float64)
        self.maximum = np.full((joints, CHANNELS), -np.inf, dtype=np.float64)

    def update(self, motion: np.ndarray, heading_valid: np.ndarray) -> None:
        joints = np.arange(motion.shape[1])
        self._update(motion[:, :, :13], joints, slice(0, 13))
        self._update(motion[:, :1, 13:15], np.array([0]), slice(13, 15))
        self._update(motion[heading_valid, :1, 15:17], np.array([0]), slice(15, 17))

    def _update(self, values: np.ndarray, joint_indices: np.ndarray, channels: slice) -> None:
        if values.size == 0:
            return
        start, stop = channels.start, channels.stop
        self.count[joint_indices, start:stop] += values.shape[0]
        self.sum[joint_indices, start:stop] += values.sum(axis=0, dtype=np.float64)
        self.sumsq[joint_indices, start:stop] += np.square(values, dtype=np.float64).sum(axis=0, dtype=np.float64)
        self.minimum[joint_indices, start:stop] = np.minimum(self.minimum[joint_indices, start:stop], values.min(axis=0))
        self.maximum[joint_indices, start:stop] = np.maximum(self.maximum[joint_indices, start:stop], values.max(axis=0))

    def finish(self, supervise_mask: np.ndarray) -> dict[str, np.ndarray]:
        populated = self.count > 0
        mean = np.zeros_like(self.sum, dtype=np.float32)
        mean[populated] = (self.sum[populated] / self.count[populated]).astype(np.float32)
        variance = np.zeros_like(self.sum)
        variance[populated] = self.sumsq[populated] / self.count[populated] - np.square(mean[populated])
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
            "count": self.count,
            "mean": mean,
            "std_raw": std_raw,
            "std_eff": std_eff,
            "minimum": minimum,
            "maximum": maximum,
            "supervise_mask": supervise_mask,
        }


def write_canonical_skeleton(source_path: Path, source_stats_path: Path, output_path: Path) -> dict[str, object]:
    with np.load(source_path, allow_pickle=False) as source:
        payload = {key: source[key].copy() for key in source.files}
    parents = payload["parents"].astype(np.int64)
    offsets = payload["offset_parent_local"].astype(np.float64)
    old_global = payload["R_rest_global"].astype(np.float64)
    shared_basis = old_global[0]
    max_within_rig_error = float(np.abs(old_global - shared_basis).max())
    if max_within_rig_error > 1e-5:
        raise ValueError(f"{source_path.name}: R_rest_global is not rig-wide constant ({max_within_rig_error})")
    corrected_rest = accumulate_offsets(offsets, parents)
    offsets = offsets.copy()
    offsets[0, 1] -= corrected_rest[:, 1].min()
    corrected_rest = accumulate_offsets(offsets, parents)
    identity = np.broadcast_to(np.eye(3, dtype=np.float32), old_global.shape).copy()
    payload["offset_parent_local"] = offsets.astype(np.float32)
    payload["P_rest_global"] = corrected_rest.astype(np.float32)
    payload["R_rest_global"] = identity
    payload["R_rest_local"] = identity
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    with np.load(source_stats_path, allow_pickle=False) as source_stats:
        supervise_mask = source_stats["supervise_mask"].copy()
    return {
        "parents": parents,
        "offsets": offsets,
        "shared_basis": shared_basis,
        "supervise_mask": supervise_mask,
        "joint_count": len(parents),
        "max_within_rig_error": max_within_rig_error,
        "corrected_rest_min_y": float(corrected_rest[:, 1].min()),
        "corrected_rest_max_y": float(corrected_rest[:, 1].max()),
    }


def process_rig(task: dict[str, object]) -> dict[str, object]:
    rig_id = str(task["rig_id"])
    input_data = Path(str(task["input_data"]))
    output_data = Path(str(task["output_data"]))
    shared_basis = np.asarray(task["shared_basis"], dtype=np.float64)
    parents = np.asarray(task["parents"], dtype=np.int64)
    offsets = np.asarray(task["offsets"], dtype=np.float64)
    supervise_mask = np.asarray(task["supervise_mask"], dtype=bool)
    rows = list(task["rows"])
    stats = RunningStats(len(parents))
    max_fk_error = 0.0
    for index, row in enumerate(rows):
        source_path = input_data / str(row["motion_file"])
        output_path = output_data / str(row["motion_file"])
        with np.load(source_path, allow_pickle=False) as source:
            payload = {key: source[key].copy() for key in source.files}
        original_motion = payload["motion"]
        delta = matrix_from_cont6d(original_motion[:, :, 3:9].astype(np.float64))
        corrected_rotation = delta @ shared_basis[None, None]
        corrected_motion = original_motion.copy()
        corrected_motion[:, :, 3:9] = cont6d_from_matrix(corrected_rotation).astype(np.float32)
        payload["motion"] = corrected_motion
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **payload)
        stats.update(corrected_motion, payload["heading_valid"])
        if index == 0:
            recovered = decode_world_positions(corrected_motion, corrected_rotation, parents, offsets)
            expected = original_motion[:, :, :3].astype(np.float64)
            expected[:, :, 0] += original_motion[:, 0, 13, None]
            expected[:, :, 2] += original_motion[:, 0, 14, None]
            max_fk_error = float(np.abs(recovered - expected).max())
            if max_fk_error > 1e-4:
                raise ValueError(f"{rig_id}: corrected FK error too large ({max_fk_error})")
    np.savez_compressed(output_data / "stats" / f"{rig_id}.npz", **stats.finish(supervise_mask))
    return {"rig_id": rig_id, "clip_count": len(rows), "fk_max_abs_error_first_clip": max_fk_error}


def load_manifest(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            grouped.setdefault(str(row["rig_id"]), []).append(row)
    return grouped


def copy_static_files(input_data: Path, output_data: Path) -> None:
    for name in ("metadata", "manifests", "reports"):
        source = input_data / name
        if source.is_dir():
            shutil.copytree(source, output_data / name, dirs_exist_ok=True)


def prepare_output(input_root: Path, output_root: Path, has_wrapper: bool) -> Path:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root must be new and empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    if has_wrapper:
        for source in input_root.iterdir():
            if source.is_file() and source.suffix.lower() in {".md", ".txt"}:
                shutil.copy2(source, output_root / source.name)
    output_data = output_root / "data" if has_wrapper else output_root
    for name in ("motions", "skeletons", "stats", "metadata", "manifests", "reports"):
        (output_data / name).mkdir(parents=True, exist_ok=True)
    return output_data


def write_generation(input_data: Path, output_data: Path) -> None:
    source_path = input_data / "generation.json"
    generation = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else {}
    generation["version"] = str(generation.get("version", "noik-release")) + "-canonical-rest-v1"
    generation["rest_pose_recanonicalization"] = {
        "input_rest_basis": "per-rig R_rest_global constant shared by every joint",
        "output_rest_basis": "identity for every joint",
        "output_rest_position": "forward accumulation of offset_parent_local",
        "rot6d_3_9": "old_delta_rotation @ old_rig_rest_basis",
        "unchanged_channels": "q_position 0:3, velocity 9:12, contact 12, root_xz 13:15, heading 15:17",
        "coordinate_system": "right-handed, +Y up, +Z viewer-facing",
    }
    (output_data / "generation.json").write_text(json.dumps(generation, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    input_data, has_wrapper = data_root_from(args.input_root)
    if not (input_data / "motions").is_dir() or not (input_data / "skeletons").is_dir():
        raise ValueError(f"Not a supported no-IK data root: {input_data}")
    output_data = prepare_output(args.input_root, args.output_root, has_wrapper)
    copy_static_files(input_data, output_data)
    write_generation(input_data, output_data)
    grouped = load_manifest(input_data / "manifests" / "clips.jsonl")
    available = sorted(path.stem for path in (input_data / "skeletons").glob("*.npz"))
    requested = set(args.rig)
    unknown = sorted(requested - set(available))
    if unknown:
        raise ValueError(f"Unknown rigs: {unknown}")
    rig_ids = [rig for rig in available if not requested or rig in requested]
    if requested:
        included_motion_files = {row["motion_file"] for rig in rig_ids for row in grouped[rig]}
        manifest_out = output_data / "manifests" / "clips.jsonl"
        with manifest_out.open("w", encoding="utf-8") as handle:
            for rig_id in rig_ids:
                for row in grouped[rig_id]:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        del included_motion_files

    tasks: list[dict[str, object]] = []
    skeleton_reports: dict[str, dict[str, object]] = {}
    for rig_id in rig_ids:
        report = write_canonical_skeleton(
            input_data / "skeletons" / f"{rig_id}.npz",
            input_data / "stats" / f"{rig_id}.npz",
            output_data / "skeletons" / f"{rig_id}.npz",
        )
        skeleton_reports[rig_id] = report
        tasks.append(
            {
                "rig_id": rig_id,
                "input_data": str(input_data),
                "output_data": str(output_data),
                "shared_basis": report["shared_basis"],
                "parents": report["parents"],
                "offsets": report["offsets"],
                "supervise_mask": report["supervise_mask"],
                "rows": grouped[rig_id],
            }
        )

    results: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_rig, task): task["rig_id"] for task in tasks}
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(f"[{index}/{len(tasks)}] {result['rig_id']}: {result['clip_count']} clips, FK max={result['fk_max_abs_error_first_clip']:.3e}", flush=True)

    results.sort(key=lambda row: str(row["rig_id"]))
    rows = []
    for result in results:
        skeleton = skeleton_reports[str(result["rig_id"])]
        rows.append(
            {
                **result,
                "rest_basis_within_rig_max_abs_error": skeleton["max_within_rig_error"],
                "corrected_rest_min_y": skeleton["corrected_rest_min_y"],
                "corrected_rest_max_y": skeleton["corrected_rest_max_y"],
            }
        )
    report_dir = output_data / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "rest_pose_recanonicalization_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "format": "KTJD-17 canonical-rest-v1",
        "input_data": str(input_data),
        "rig_count": len(rows),
        "clip_count": int(sum(int(row["clip_count"]) for row in rows)),
        "max_first_clip_fk_error": float(max(float(row["fk_max_abs_error_first_clip"]) for row in rows)),
        "max_old_rest_basis_within_rig_error": float(max(float(row["rest_basis_within_rig_max_abs_error"]) for row in rows)),
        "output_data": str(output_data),
    }
    (report_dir / "rest_pose_recanonicalization_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
