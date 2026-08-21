"""Run AniMo4D's public position-to-feature logic on one 30-joint export.

The released notebook stores a per-species/sex target skeleton template that
is not distributed in the code repository.  This probe uses the input clip's
first frame as the target template, retaining the same topology, IK, axis
rotations, feature construction, and RIC recovery while avoiding a guessed
cross-species scale.  That scale is global and cannot remove a temporal jump.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# The HumanML3D dependency pinned by AniMo4D still imports the NumPy 1.x
# alias. Keep this compatibility shim local to the probe rather than changing
# the cloned reference implementation.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]


KINEMATIC_CHAIN = [
    [0, 20, 21, 22, 23, 24],
    [0, 25, 26, 27, 28, 29],
    [0, 9, 10, 11],
    [0, 1, 2, 3, 4],
    [1, 7, 12, 13, 14, 15],
    [1, 8, 16, 17, 18, 19],
    [3, 5],
    [3, 6],
]
RAW_OFFSETS = np.array(
    [
        [0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [-1, 0, 0], [1, 0, 0],
        [1, 0, 0], [-1, 0, 0], [0, 0, -1], [0, 0, -1], [0, 0, -1], [0, -1, 0], [0, -1, 0],
        [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [1, 0, 0],
        [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [-1, 0, 0], [0, -1, 0], [0, -1, 0],
        [0, -1, 0], [0, -1, 0],
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-positions", required=True, type=Path)
    parser.add_argument("--humanml-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def rotate_pos(positions: np.ndarray) -> np.ndarray:
    rotated = Rotation.from_rotvec(np.array([-np.pi / 2, 0, 0])).apply(positions.reshape(-1, 3)).reshape(positions.shape)
    return Rotation.from_rotvec(np.array([0, -np.pi / 2, 0])).apply(rotated.reshape(-1, 3)).reshape(positions.shape)


def jump_stats(positions: np.ndarray) -> dict:
    root_relative = positions - positions[:, :1]
    per_joint = np.linalg.norm(np.diff(root_relative, axis=0), axis=-1)
    aggregate = np.linalg.norm(np.diff(root_relative, axis=0).reshape(len(per_joint), -1), axis=1)
    index = int(np.argmax(aggregate))
    return {
        "max_all_joint_step": float(aggregate[index]),
        "max_transition_index": [index, index + 1],
        "median_all_joint_step": float(np.median(aggregate)),
        "p95_all_joint_step": float(np.percentile(aggregate, 95)),
        "max_over_median": float(aggregate[index] / max(np.median(aggregate), 1e-12)),
        "max_over_p95": float(aggregate[index] / max(np.percentile(aggregate, 95), 1e-12)),
        "per_joint_max": [float(value) for value in per_joint.max(axis=0)],
    }


def main() -> None:
    args = parse_args()
    if not args.input_positions.is_file():
        raise FileNotFoundError(args.input_positions)
    if not (args.humanml_root / "common" / "skeleton.py").is_file():
        raise FileNotFoundError(args.humanml_root / "common" / "skeleton.py")
    sys.path.insert(0, str(args.humanml_root))
    from common.quaternion import qbetween_np, qinv_np, qmul_np, qrot_np, quaternion_to_cont6d_np  # pylint: disable=import-outside-toplevel
    from common.skeleton import Skeleton  # pylint: disable=import-outside-toplevel

    source = np.load(args.input_positions).astype(np.float64)
    if source.ndim != 3 or source.shape[1:] != (30, 3):
        raise ValueError(f"Expected [F, 30, 3], got {source.shape}")

    # Port of uniform_skeleton(), using first-frame offsets as the unavailable
    # category template. This makes the scale ratio exactly one.
    skeleton = Skeleton(torch.from_numpy(RAW_OFFSETS), KINEMATIC_CHAIN, "cpu")
    target_offsets = skeleton.get_offsets_joints(torch.from_numpy(source[0])).numpy()
    quat_uniform = skeleton.inverse_kinematics_np(source, [25, 20, 8, 7])
    skeleton.set_offset(torch.from_numpy(target_offsets))
    positions = skeleton.forward_kinematics_np(quat_uniform, source[:, 0])

    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height
    positions -= positions[0, 0] * np.array([1, 0, 1])
    positions = rotate_pos(positions)
    positions[:, :, 1] -= positions.min(axis=0).min(axis=0)[1]
    global_positions = positions.copy()

    # Port of get_cont6d_params() and the RIC branch of process_file().
    quat_params = skeleton.inverse_kinematics_np(positions, [25, 20, 8, 7], smooth_forward=True)
    cont6d = quaternion_to_cont6d_np(quat_params)
    r_rot = quat_params[:, 0].copy()
    local_positions = positions.copy()
    local_positions[..., 0] -= local_positions[:, 0:1, 0]
    local_positions[..., 2] -= local_positions[:, 0:1, 2]
    local_positions = qrot_np(np.repeat(r_rot[:, None], 30, axis=1), local_positions)
    root_y = local_positions[:, 0, 1:2]
    velocity = qrot_np(r_rot[1:], positions[1:, 0] - positions[:-1, 0])
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
    root_data = np.concatenate([np.arcsin(r_velocity[:, 2:3]), velocity[:, [0, 2]], root_y[:-1]], axis=-1)
    ric_data = local_positions[:, 1:].reshape(len(local_positions), -1)
    rot_data = cont6d[:, 1:].reshape(len(cont6d), -1)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], 30, axis=1), global_positions[1:] - global_positions[:-1]).reshape(len(global_positions) - 1, -1)
    # The original foot values are included in its flat feature vector. They
    # are not needed to test positional continuity, but retain its layout.
    feet_l_ids, feet_r_ids = [15, 24], [19, 29]
    feet_l = (np.sum((positions[1:, feet_l_ids] - positions[:-1, feet_l_ids]) ** 2, axis=-1) < 1e-8).astype(np.float32)
    feet_r = (np.sum((positions[1:, feet_r_ids] - positions[:-1, feet_r_ids]) ** 2, axis=-1) < 1e-8).astype(np.float32)
    feature = np.concatenate([root_data, ric_data[:-1], rot_data[:-1], local_vel, feet_l, feet_r], axis=-1)

    # Exact public recover_from_ric() algebra, expressed with NumPy.
    root_angle = np.zeros(len(feature), dtype=np.float64)
    root_angle[1:] = np.cumsum(feature[:-1, 0])
    root_quat = np.zeros((len(feature), 4), dtype=np.float64)
    root_quat[:, 0] = np.cos(root_angle)
    root_quat[:, 2] = np.sin(root_angle)
    root_pos = np.zeros((len(feature), 3), dtype=np.float64)
    root_pos[1:, [0, 2]] = feature[:-1, 1:3]
    root_pos = qrot_np(qinv_np(root_quat), root_pos)
    root_pos = np.cumsum(root_pos, axis=0)
    root_pos[:, 1] = feature[:, 3]
    reconstructed = feature[:, 4 : 4 + 29 * 3].reshape(len(feature), 29, 3)
    reconstructed = qrot_np(np.repeat(qinv_np(root_quat)[:, None], 29, axis=1), reconstructed)
    reconstructed[..., 0] += root_pos[:, None, 0]
    reconstructed[..., 2] += root_pos[:, None, 2]
    reconstructed = np.concatenate([root_pos[:, None], reconstructed], axis=1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "animo4d_notebook_equivalent_feature.npy", feature)
    np.save(args.output_dir / "animo4d_notebook_equivalent_recovered_positions.npy", reconstructed)
    report = {
        "source": "AniMo4D public notebook algorithm with source-first-frame template",
        "template_mode": "source_first_frame_equivalent; official category template was not published in the repository",
        "input_shape": [int(value) for value in source.shape],
        "feature_shape": [int(value) for value in feature.shape],
        "recovered_position_shape": [int(value) for value in reconstructed.shape],
        "input_position_jump": jump_stats(source),
        "recovered_position_jump": jump_stats(reconstructed),
    }
    (args.output_dir / "animo4d_notebook_equivalent_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
