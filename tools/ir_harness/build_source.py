"""build_source.py — the IR driver.

Given a SourceAdapter, run the UNIVERSAL back-end: collect the adapter's prepared [T,J,13] clips,
compute per-object-type mean/std over the train split, write the dataset layout, run the acceptance
gates, and REFUSE to accept the dataset if any hard gate is red. Source-agnostic — the per-source
encoding lives in the adapter (see source_adapter.py); everything here is identical for every source.

Outputs under <out_root>/ (spec §"What you produce"):
    motions/<motion_id>.npy      float32 [T,J,13] RAW, ORIGINAL joint order
    cond.npy                     dict{object_type -> {parents,offsets,joints_names,tpos_first_frame,mean,std,
                                                       foot_joint_idx,gate_c_mode}}
    motion_texts_by_file.json    {motion_id -> {primary_caption, captions:[...]}}
    motion_object_types.json     {motion_id -> object_type}   (explicit map; no prefix guessing)
    splits/{train,val}.txt
    _gate_report.json            Gates A/B/C/E results (Gate D is human-judged, see README)

`build()` raises AcceptanceError on a red hard gate (A/B/C/E) unless strict=False — a red gate must
never silently ship. It always writes _gate_report.json first so the failure is inspectable.

Usage (an agent wires up its adapter, then):
    from tools.ir_harness.build_source import build
    build(MyAdapter(), out_root="/path/out", max_joints=144)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.ir_harness.gates import (  # noqa: E402
    gate_a_shape_finite, gate_b_ric_equivalence, gate_c_fk_selfcheck, gate_e_layout,
)
from tools.ir_harness.source_adapter import SourceAdapter, Clip  # noqa: E402

_CHANNEL_GROUPS = [(0, 3), (3, 9), (9, 12)]   # pos / rot6d / vel; contact (12) handled separately


class AcceptanceError(RuntimeError):
    """Raised when a hard gate (A/B/C/E) is red — the dataset is NOT accepted."""


def compute_mean_std(train_motions: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-object-type mean/std over ALL train frames — a FAITHFUL port of
    data_loaders/truebones/truebones_utils/motion_process.py get_mean_std (spec §4).

    mean = per-(joint,channel) mean. std starts per-(joint,channel), then is group-homogenized
    EXACTLY as the reference:
      - root joint 0: each of its pos(0:3)/rot(3:9)/vel(9:12) blocks -> the block's own scalar mean;
      - ALL non-root joints (1:) SHARE one scalar per block = the mean over every non-root joint AND
        every channel in that block (not per-joint — this is the reference's behaviour, and what the
        existing merged codebook was normalized with);
      - contact (12): the non-zero contact-stds are homogenized to their mean; zeros -> 1.0.
    No epsilon is baked in here — the loader adds +1e-6 at normalize time (spec §4). Never reuse a
    source's own flat stats; recompute per object_type over its own train clips.
    """
    frames = np.concatenate([m.reshape(-1, m.shape[1], 13) for m in train_motions], axis=0)  # [N,J,13]
    mean = frames.mean(axis=0).astype(np.float32)                                             # [J,13]
    std = frames.std(axis=0).astype(np.float32)
    for lo, hi in _CHANNEL_GROUPS:
        std[0, lo:hi] = std[0, lo:hi].mean()        # root block -> its own scalar
        std[1:, lo:hi] = std[1:, lo:hi].mean()      # every non-root joint shares one block scalar
    nz = std[:, 12] != 0.0
    if nz.any():
        std[nz, 12] = std[nz, 12].mean()            # non-zero contact-stds homogenized
    std[std[:, 12] == 0.0, 12] = 1.0                # zero contact-std -> 1.0
    return mean, std


def _excursion(m: np.ndarray) -> float:
    """Temporal root-rotation excursion = summed |Δ(root ch3:9)| over frames. Measures MOTION, so a
    static (even 90°) pose scores ~0 and a real turn scores high — the correct Gate-C clip selector."""
    return float(np.abs(np.diff(m[:, 0, 3:9], axis=0)).sum()) if m.shape[0] > 1 else 0.0


def build(adapter: SourceAdapter, out_root: str, *, max_joints: int, strict: bool = True,
          overwrite: bool = False, log=print) -> dict:
    out = Path(out_root)
    _ARTIFACTS = ("cond.npy", "motion_texts_by_file.json", "motion_object_types.json",
                  "splits", "_gate_report.json", "_ACCEPTED")
    existing = [a for a in _ARTIFACTS if (out / a).exists()] + \
               (["motions/*.npy"] if (out / "motions").is_dir() and any((out / "motions").glob("*.npy")) else [])
    if existing:
        if not overwrite:
            raise AcceptanceError(f"{out} already holds dataset artifacts {existing}; pass overwrite=True "
                                  f"to replace (prevents mixing stale files into a new build)")
        for a in ("motions", "splits"):
            shutil.rmtree(out / a, ignore_errors=True)
        for a in _ARTIFACTS:
            if a not in ("splits",):
                (out / a).unlink(missing_ok=True)
    (out / "_ACCEPTED").unlink(missing_ok=True)            # never leave a stale acceptance marker
    (out / "motions").mkdir(parents=True, exist_ok=True)
    (out / "splits").mkdir(parents=True, exist_ok=True)

    cond: dict = {}
    texts: dict = {}
    mot_ot: dict = {}                                       # motion_id -> object_type manifest
    split_ids = {"train": [], "val": []}
    seen_ids: set = set()
    gate_a_fail, gate_c_fail, gate_b_fail = [], [], []      # per-CLIP failures (every clip is checked)
    c_sample, n_clips, n_b_run, n_no_frames = {}, 0, 0, 0   # c_sample: one representative Gate-C per type

    for ot in adapter.iter_object_types():
        topo = adapter.topology(ot)
        topo.validate()
        J = len(topo.parents)
        if J > max_joints:
            raise AcceptanceError(f"{ot}: J={J} exceeds max_joints={max_joints} (loader would skip it)")
        train_motions, best_ex = [], -1.0
        for c in adapter.iter_clips(ot):
            c.validate(J)
            if c.object_type != ot:
                raise AcceptanceError(f"clip {c.motion_id}: object_type {c.object_type!r} != {ot!r}")
            if c.motion_id in seen_ids:
                raise AcceptanceError(f"duplicate motion_id {c.motion_id!r} (would overwrite)")
            seen_ids.add(c.motion_id)
            m = np.asarray(c.motion, dtype=np.float32)
            np.save(out / "motions" / f"{c.motion_id}.npy", m)
            texts[c.motion_id] = {"primary_caption": (c.captions[0] if c.captions else ""),
                                  "captions": list(c.captions)}
            mot_ot[c.motion_id] = ot
            split_ids[c.split].append(c.motion_id)         # c.validate guarantees split in {train,val}
            if c.split == "train":
                train_motions.append(m)
            # ---- per-CLIP gates: A (integrity) + C (FK==RIC) on EVERY clip; B where source_world given ----
            if c.source_frames is None:
                n_no_frames += 1                            # T=F-1 check skipped for this clip — surfaced below
            a = gate_a_shape_finite(m, c.source_frames, foot_joint_idx=topo.foot_joint_idx, parents=topo.parents)
            if not a["pass"]:
                gate_a_fail.append({"motion_id": c.motion_id, **a})
            cres = gate_c_fk_selfcheck(m.astype(np.float64), topo.parents, topo.offsets, mode=topo.gate_c_mode)
            if not cres["pass"]:
                gate_c_fail.append({"motion_id": c.motion_id, **cres})
            ex = _excursion(m)
            if ex > best_ex:                                # keep the highest-excursion clip as the report sample
                best_ex = ex; c_sample[ot] = {"motion_id": c.motion_id, **cres}
            if c.source_world is not None:
                n_b_run += 1
                bres = gate_b_ric_equivalence(m.astype(np.float64), source_world=np.asarray(c.source_world, np.float64))
                if not bres["pass"]:
                    gate_b_fail.append({"motion_id": c.motion_id, **bres})
            n_clips += 1
        if not train_motions:
            raise AcceptanceError(f"{ot}: no TRAIN clips — normalization stats are undefined "
                                  f"(supply train clips or externally-provided train stats)")
        mean, std = compute_mean_std(train_motions)
        if not (np.isfinite(mean).all() and np.isfinite(std).all()):
            raise AcceptanceError(f"{ot}: mean/std non-finite (degenerate/empty train frames)")
        cond[ot] = {"parents": np.asarray(topo.parents, np.int32),
                    "offsets": np.asarray(topo.offsets, np.float32),
                    "joints_names": list(topo.joints_names),
                    "tpos_first_frame": np.asarray(topo.tpos_first_frame, np.float32),
                    "mean": mean, "std": std, "object_type": ot,
                    "foot_joint_idx": (None if topo.foot_joint_idx is None else list(topo.foot_joint_idx)),
                    "gate_c_mode": topo.gate_c_mode,
                    "scale_ref_joint_idx": (None if topo.scale_ref_joint_idx is None
                                            else list(topo.scale_ref_joint_idx))}
        log(f"[build] {ot}: {len(train_motions)} train (J={J}); Gate-C on all its clips")

    np.save(out / "cond.npy", cond, allow_pickle=True)
    (out / "motion_texts_by_file.json").write_text(json.dumps(texts, indent=1))
    (out / "motion_object_types.json").write_text(json.dumps(mot_ot, indent=1))
    for sp, ids in split_ids.items():
        (out / "splits" / f"{sp}.txt").write_text("\n".join(ids) + ("\n" if ids else ""))
    log(f"[build] wrote {n_clips} clips, {len(cond)} object_types -> {out}")
    if n_no_frames:
        log(f"[build] WARNING: {n_no_frames}/{n_clips} clips lacked source_frames -> Gate-A T=F-1 check skipped for them")

    gate_e = gate_e_layout(str(out), mot_ot)
    a_pass = not gate_a_fail
    c_pass = not gate_c_fail
    b_pass = not gate_b_fail                                 # waived-everywhere counts as pass (see note)
    overall = bool(a_pass and c_pass and b_pass and gate_e["pass"])
    report = {"source": adapter.name, "out_root": str(out), "n_clips": n_clips,
              "n_object_types": len(cond), "max_joints": max_joints,
              "gate_A_pass": a_pass, "gate_A_failures": gate_a_fail,
              "gate_A_clips_without_T_check": n_no_frames,
              "gate_B_pass": b_pass, "gate_B_n_checked": n_b_run, "gate_B_failures": gate_b_fail,
              "gate_C_pass": c_pass, "gate_C_n_checked": n_clips, "gate_C_failures": gate_c_fail,
              "gate_C_sample_per_type": c_sample,
              "gate_E": gate_e, "gate_E_pass": gate_e["pass"],
              "gate_D": ("MANUAL: render GT-vs-converted GIFs (skeleton + mesh) and eye-judge — "
                         "visual QA is authoritative and outranks every scalar gate (spec §8, Gate D). "
                         "Use the repo's render tools; FK world positions via _ref_recover.recover_from_bvh_rot_np."),
              "overall_hard_pass": overall}
    (out / "_gate_report.json").write_text(json.dumps(report, indent=1, default=float))
    if overall:
        (out / "_ACCEPTED").write_text("all hard gates (A/B/C/E) green; Gate D (visual QA) still required\n")
    log(f"[build] gate report -> {out / '_gate_report.json'}  "
        f"(A={a_pass} B={b_pass}[{n_b_run} checked] C={c_pass}[{n_clips} checked] E={gate_e['pass']} "
        f"=> overall={overall}{' ACCEPTED' if overall else ''})")
    if strict and not overall:
        raise AcceptanceError(f"dataset NOT accepted (no _ACCEPTED marker written): "
                              f"A={a_pass} B={b_pass} C={c_pass} E={gate_e['pass']}. "
                              f"See {out / '_gate_report.json'}. Gate D (visual QA) still required before training.")
    return report
