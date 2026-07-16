"""SourceAdapter — the per-source contract an agent implements to enter the AnyTop-13 IR.

Read docs/ANYTOP13_INTERMEDIATE_REPRESENTATION.md first. This file is the *interface*: an agent
subclasses `SourceAdapter` for a new source, implementing ONLY the per-source concerns (spec §7).
The universal back-end (per-object mean/std normalization, dataset layout, acceptance gates) is
done for you by build_source.py — do not re-implement it here.

DIVISION OF LABOUR
  Adapter (you, per source) : source assets -> the encoded 13ch motion + primary cond fields, in
                              the source's ORIGINAL joint order, obeying every §7 per-source rule
                              (Y-up/+Z, HML_AVG_BONELEN scale, 20fps, T=F-1, per-parent rot6d,
                              contact on ch12, root facing-6D + XZ-velocity, twist DOF pinned).
  Driver (build_source.py)  : collect clips -> compute per-object mean/std over the train split ->
                              write motions/ + cond.npy + captions + splits -> run Gates A/B/C/E,
                              and REFUSE to accept the dataset if any hard gate is red.

WHY THE ADAPTER OWNS THE 13ch ENCODING
  The source->13ch step IS the per-source part and it differs fundamentally by source family:
    - BVH/game rigs (Planet Zoo): run the existing BVH-export + `utils.process_new_skeleton`
      (`motion_process.process_object`) pipeline — it already emits [T,J,13] + cond. Your adapter
      is a thin wrapper that invokes it (see BvhPipelineAdapter below) and reads back the results.
    - pose-based mocap (AMASS/SMPL, HumanML3D): re-encode world positions + native rotations into
      the per-parent rot6d via per-joint Kabsch against AnyTop's rest basis (the algorithm in
      spec §2/§7; the training repo's scripts/convert_humanml3d_to_anytop13.py is one reference
      implementation but is NOT vendored in this repo).
  Either way the adapter's OUTPUT is uniform, so the driver + gates are source-agnostic.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

GATE_C_MODES = ("clean", "inferred_twist")   # tolerance regime for the FK==RIC self-check (spec §8)


def check_index_list(idx, J: int, name: str):
    """Validate a joint-index declaration (foot_joint_idx / scale_ref_joint_idx). Single source of
    truth shared by Topology.validate and Gate E so both handle every malformed shape identically.

    Returns (kind, payload):
      "none"  -> idx is None                       (payload None)
      "empty" -> a valid empty 1-D sequence        (payload None)
      "valid" -> a 1-D int array in [0,J)          (payload the int ndarray)
      "error" -> anything malformed                (payload a human reason string)
    Rejects: non-sequence scalars, ragged/inhomogeneous lists, object-dtype, 0-D/2-D arrays,
    non-integer dtype, out-of-range indices.
    """
    if idx is None:
        return "none", None
    if not isinstance(idx, (list, tuple, np.ndarray)):
        return "error", f"{name} must be a list of ints, got {type(idx).__name__}"
    try:
        a = np.asarray(idx)
    except (ValueError, TypeError):
        return "error", f"{name} is not a valid index array (ragged/inhomogeneous)"
    if a.dtype == object:
        return "error", f"{name} is not a valid index array (ragged/object dtype)"
    if a.ndim != 1:
        return "error", f"{name} must be 1-D, got shape {a.shape}"
    if a.size == 0:
        return "empty", None
    if not np.issubdtype(a.dtype, np.integer) or ((a < 0) | (a >= J)).any():
        return "error", f"{name} must be ints in [0,{J}) (got {a.tolist()})"
    return "valid", a.astype(int)


@dataclass
class Topology:
    """Primary cond fields for one object_type, in the source's ORIGINAL joint order (spec §6).

    Everything else the graph model needs (adjacency/geodesic/graph_dist/joint_relations/
    skeleton_features/name_hashes/new_to_old_perm) is DERIVED by the training loader — do not ship it.

    Gate policy is declared HERE, per topology (never inferred from the object_type string):
      gate_c_mode : "clean" (FK==RIC abs L2 < 1e-4) or "inferred_twist" (rel ≤ 0.5% bbox, for
                    sources that infer axial twist from positions, e.g. SMPL-from-positions humans).
      foot_joint_idx : indices where ch12 contact may be non-zero; contact MUST be 0 on all others.
                       None = unknown (the feet-location check is skipped, binary check still runs).
    """
    object_type: str
    parents: np.ndarray          # [J] int; single root (-1) at INDEX 0; connected single tree
    offsets: np.ndarray          # [J,3] float; rest-pose bone vector (child rel. to parent)
    joints_names: list[str]      # [J]; drive name_hashes + left/right/center side tag
    tpos_first_frame: np.ndarray  # [J,13]; rest/T-pose as a 13ch row (parity/render)
    foot_joint_idx: Optional[list[int]] = None
    gate_c_mode: str = "clean"
    scale_ref_joint_idx: Optional[list[int]] = None   # the joints whose mean rest-bone-length the adapter
                                                      # scaled to HML_AVG_BONELEN (motion_process.scale's
                                                      # scale_joint_indices). Declaring it lets Gate E verify
                                                      # scale EXACTLY; None -> only a gross scale band is checked.

    def validate(self) -> None:
        """Full topology contract (raises ValueError — NOT assert, so `python -O` can't strip it)."""
        p = np.asarray(self.parents)
        if p.ndim != 1:
            raise ValueError(f"{self.object_type}: parents must be 1-D, got shape {p.shape}")
        J = p.shape[0]
        if J < 1:
            raise ValueError(f"{self.object_type}: empty parents")
        if not np.issubdtype(p.dtype, np.integer):
            if not np.all(p == np.round(p)):
                raise ValueError(f"{self.object_type}: parents has non-integral values")
            p = p.astype(np.int64)
        roots = np.flatnonzero(p == -1)
        if roots.size != 1:
            raise ValueError(f"{self.object_type}: need exactly one root (-1), got {roots.size}")
        # IR mandates the root at index 0 in ORIGINAL order: the 13ch root special-casing (ch1=height,
        # ch3:9=facing, ch9/ch11=root vel) and get_mean_std's root-block (Std[0]) both key on joint 0.
        if p[0] != -1:
            raise ValueError(f"{self.object_type}: root must be joint 0 (parents[0]==-1), got parents[0]={int(p[0])}")
        if ((p < -1) | (p >= J)).any():
            raise ValueError(f"{self.object_type}: parent index out of range [-1,{J})")
        if (p == np.arange(J)).any():
            raise ValueError(f"{self.object_type}: a joint is its own parent")
        # connected + acyclic: every node reaches root 0 within J hops
        for j in range(J):
            steps, cur = 0, j
            while cur != -1 and steps <= J:
                cur, steps = int(p[cur]), steps + 1
            if cur != -1:
                raise ValueError(f"{self.object_type}: joint {j} not connected to root 0 (cycle or forest)")
        if np.asarray(self.offsets).shape != (J, 3):
            raise ValueError(f"{self.object_type}: offsets {np.asarray(self.offsets).shape} != ({J},3)")
        if len(self.joints_names) != J:
            raise ValueError(f"{self.object_type}: {len(self.joints_names)} names != {J}")
        if np.asarray(self.tpos_first_frame).shape != (J, 13):
            raise ValueError(f"{self.object_type}: tpos {np.asarray(self.tpos_first_frame).shape} != ({J},13)")
        for fld in ("offsets", "tpos_first_frame"):
            if not np.isfinite(np.asarray(getattr(self, fld), np.float64)).all():
                raise ValueError(f"{self.object_type}: {fld} contains non-finite values")
        if self.gate_c_mode not in GATE_C_MODES:
            raise ValueError(f"{self.object_type}: gate_c_mode {self.gate_c_mode!r} not in {GATE_C_MODES}")
        for fld in ("foot_joint_idx", "scale_ref_joint_idx"):
            kind, payload = check_index_list(getattr(self, fld), J, fld)   # none/empty/valid all OK here
            if kind == "error":
                raise ValueError(f"{self.object_type}: {payload}")


@dataclass
class Clip:
    """One prepared motion clip in the IR."""
    motion_id: str               # unique, safe basename (no path sep / no '.npy' / non-empty)
    object_type: str             # must match a Topology.object_type
    motion: np.ndarray           # [T,J,13] float32, RAW (un-normalized), ORIGINAL joint order, T=F-1>0
    captions: list[str] = field(default_factory=list)   # [] allowed (never dropped for lacking text)
    split: str = "train"         # "train" | "val" (any other value is REJECTED, not coerced)
    source_frames: Optional[int] = None                  # F; RECOMMENDED — enables the Gate-A T==F-1
                                                         # check (skipped, and counted in the report, if omitted)
    source_world: Optional[np.ndarray] = None            # [T,J,3] source-official world recovery for Gate B

    def validate(self, J: int) -> None:
        if not isinstance(self.motion_id, str) or not self.motion_id:
            raise ValueError(f"empty/invalid motion_id: {self.motion_id!r}")
        if any(c in self.motion_id for c in ("/", "\\", "\0")) or self.motion_id.endswith(".npy") \
                or self.motion_id in (".", ".."):
            raise ValueError(f"unsafe motion_id (path sep / '.npy' / dots): {self.motion_id!r}")
        m = np.asarray(self.motion)
        if m.ndim != 3 or m.shape[1:] != (J, 13):
            raise ValueError(f"{self.motion_id}: motion {m.shape} != [T,{J},13]")
        if m.shape[0] < 1:
            raise ValueError(f"{self.motion_id}: zero-frame clip (T={m.shape[0]})")
        if self.split not in ("train", "val"):
            raise ValueError(f"{self.motion_id}: split {self.split!r} not in ('train','val')")
        if not isinstance(self.captions, list) or any(not isinstance(x, str) for x in self.captions):
            raise ValueError(f"{self.motion_id}: captions must be a list[str] (a bare string splits into chars)")
        if self.source_world is not None:
            sw = np.asarray(self.source_world)
            if sw.shape != (m.shape[0], J, 3):
                raise ValueError(f"{self.motion_id}: source_world {sw.shape} != {(m.shape[0], J, 3)}")
            if not np.isfinite(sw).all():
                raise ValueError(f"{self.motion_id}: source_world contains non-finite values")


class SourceAdapter(abc.ABC):
    """Implement one subclass per new data source. Only these methods are yours."""

    #: short, stable source name (used in report + logging)
    name: str = "unnamed_source"

    @abc.abstractmethod
    def iter_object_types(self) -> Iterator[str]:
        """Yield every object_type (distinct skeleton topology) this source contributes."""

    @abc.abstractmethod
    def topology(self, object_type: str) -> Topology:
        """Return the primary cond fields for `object_type` (ORIGINAL joint order). See §6/§7."""

    @abc.abstractmethod
    def iter_clips(self, object_type: str) -> Iterator[Clip]:
        """Yield the prepared [T,J,13] clips for `object_type`.

        You OWN the source->13ch encoding here (§7): apply the Y-up/+Z transform, rescale to
        HML_AVG_BONELEN, 20fps resample + drop last frame (T=F-1), per-parent rot6d re-encoded via
        Kabsch so FK(rot)==RIC (§5), root facing-6D + XZ velocity into ch9/ch11, contact into ch12
        of foot joints (0 elsewhere), twist DOF pinned. Attach captions + split + source_frames.
        Output in the object_type's ORIGINAL joint order (the driver/loader handle the FK-reorder).
        """

    def source_recover_fn(self, object_type: str):
        """DEPRECATED shim retained for back-compat. Prefer attaching `Clip.source_world` (the
        source's OWN world recovery [T,J,3]) so Gate B is a genuinely independent comparison.
        If provided, must be fn(motion[T,J,13]) -> world[T,J,3]; return None to skip."""
        return None


# --------------------------------------------------------------------------- worked-example stubs
class BvhPipelineAdapter(SourceAdapter):
    """TEMPLATE STUB (not a runnable example) for a BVH/game-rig source (like Planet Zoo): delegate
    the source->13ch step to the existing exporter + `motion_process.process_object`, then read back
    motions/ + cond.

    An agent fills in `iter_object_types`/`topology`/`iter_clips` to (1) run
    tools/planetzoo/planetzoo_fulltopo_bvh_export.py for the new rig, (2) run
    utils.process_new_skeleton, (3) load the produced motions/*.npy + cond.npy (already [T,J,13] in
    original order) and yield them. See tools/planetzoo/*.py for the exact invocation.

    NOTE: motion_process.process_object pulls heavy top-level imports (BVH/Animation/InverseKinematics/
    torch); InverseKinematics.py is NOT vendored in this repo, so this path needs the full training-repo
    environment. The gate runner deliberately avoids that dependency (see gates.py / _ref_recover.py).
    """
    name = "bvh_pipeline"

    def iter_object_types(self):  # pragma: no cover - template
        raise NotImplementedError("run the BVH export + process_new_skeleton, then enumerate cond keys")

    def topology(self, object_type):  # pragma: no cover - template
        raise NotImplementedError("read parents/offsets/joints_names/tpos_first_frame from the produced cond.npy")

    def iter_clips(self, object_type):  # pragma: no cover - template
        raise NotImplementedError("load motions/*.npy for this object_type (already [T,J,13], original order)")


class PoseSourceAdapter(SourceAdapter):
    """TEMPLATE STUB (not a runnable example) for a pose-based mocap source (AMASS/SMPL/HumanML3D):
    you have world joint positions (+ optionally native local rotations). Re-encode into per-parent
    rot6d via per-joint Kabsch against AnyTop's rest basis so FK(rot)==RIC — implement the algorithm
    in spec §2/§7 (the training repo's scripts/convert_humanml3d_to_anytop13.py is one reference,
    not vendored here).

    Fill in: the fixed topology (parents/offsets/names, gate_c_mode="inferred_twist" if twist is
    inferred from positions), and iter_clips that resamples to 20fps, builds RIC(ch0:3)+root
    (ch1/9/11 & root ch3:9)+contact(ch12), Kabsch-reencodes rotations, and attaches source_world.
    """
    name = "pose_source"

    def iter_object_types(self):  # pragma: no cover - template
        raise NotImplementedError("usually one topology, e.g. 'HML3D_Human'")

    def topology(self, object_type):  # pragma: no cover - template
        raise NotImplementedError("fixed SMPL/mocap parents+offsets+names; set gate_c_mode='inferred_twist'")

    def iter_clips(self, object_type):  # pragma: no cover - template
        raise NotImplementedError("resample 20fps, drop last frame, Kabsch-reencode rot6d, pack RIC/root/contact")
