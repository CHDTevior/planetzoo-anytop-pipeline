"""AnyTop-13 IR acceptance gates (A / B / C / E).

Runnable, source-agnostic verification that a prepared dataset conforms to the AnyTop-13
Intermediate Representation (see docs/ANYTOP13_INTERMEDIATE_REPRESENTATION.md §8). Gate D
(visual QA) is human/agent-judged and lives outside this module (see the README).

Gates operate on the RAW, ORIGINAL-order motions + cond that build_source.py emits. Gate C
(FK vs RIC self-consistency) is the load-bearing invariant: it proves the per-parent rot6d
channels can drive skinning. It BFS-reorders internally so it is correct regardless of the
source's joint order, checks the multi-child rotation-duplication invariant the skinning decoder
relies on, and must be run on LARGE-rotation clips (the double-root-rotation footgun is invisible
on idle poses).

CLI:
    python -m tools.ir_harness.gates --data_root <out_root> [--object_type OT] [--large_rotation]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root, so `tools.` resolves
# Reference decoders. We use the self-contained pure-numpy port in _ref_recover.py rather than
# importing data_loaders/.../motion_process.py: that module's top-level imports pull in
# BVH/Animation/InverseKinematics/torch/matplotlib (and InverseKinematics.py is not vendored in
# this repo), which would make the gate-runner un-runnable. _ref_recover.py reproduces the SAME
# recover_from_bvh_rot_np/recover_from_bvh_ric_np conventions (validated FK==RIC to ~1e-7 vs the
# training repo, apply_root_cancel OFF) with only numpy — see its module docstring.
from tools.ir_harness._ref_recover import (  # noqa: E402
    recover_from_bvh_rot_np,      # FK path: (data[T,J,13], parents[J], offsets[J,3]) -> world [T,J,3]
    recover_from_bvh_ric_np,      # RIC path: (data[T,J,13]) -> world [T,J,3]
)
from tools.ir_harness.source_adapter import check_index_list  # noqa: E402  (shared idx validator)

ABS_VALUE_CEILING = 25.0          # matches repair_bad_motion_values quarantine scale (~22.5)
GATE_C_CLEAN = 1e-4               # FK==RIC selfcheck (mean L2) for clean/native-rotation data
GATE_C_BBOX_TOL = 0.005           # 0.5% of body-bbox for inferred-twist (human) sources
MULTI_CHILD_TOL = 1e-4            # max spread of a parent's rot6d across its child slots (encoder dups it)
ROT6D_DEGENERATE_TOL = 1e-6       # a rot6d row whose two 3-vectors are ~parallel/zero is un-orthonormalizable
HML_AVG_BONELEN = 0.2092142857142857   # shared metric scale anchor (spec §7); mean bone length target


def _bbox_diag(world: np.ndarray) -> float:
    """Mean per-frame bounding-box diagonal, a scale for relative errors."""
    lo = world.min(axis=1)          # [T,3]
    hi = world.max(axis=1)
    return float(np.linalg.norm(hi - lo, axis=-1).mean()) + 1e-8


def bfs_reorder(parents: np.ndarray):
    """Return (perm, new_parents) putting the tree in incremental order (root=0, parents[j]<j),
    matching the training loader's FK-BFS. `perm[new]=old`. build_source ships ORIGINAL order, which
    need NOT satisfy parents[j]<j, so Gate C reorders internally before running the FK recovery."""
    p = np.asarray(parents).astype(np.int64)
    J = p.shape[0]
    root = int(np.flatnonzero(p == -1)[0])
    children: dict[int, list[int]] = {}
    for j in range(J):
        if p[j] != -1:
            children.setdefault(int(p[j]), []).append(j)
    order, queue = [], [root]
    while queue:
        cur = queue.pop(0)
        order.append(cur)
        queue.extend(sorted(children.get(cur, [])))
    if len(order) != J:
        raise ValueError("bfs_reorder: not all joints reachable from root (forest/cycle)")
    perm = np.asarray(order, dtype=np.int64)          # perm[new]=old
    old_to_new = np.empty(J, dtype=np.int64)
    old_to_new[perm] = np.arange(J)
    new_parents = np.array([-1 if p[o] == -1 else int(old_to_new[p[o]]) for o in perm], dtype=np.int64)
    return perm, new_parents


def _multi_child_consistency(motion_fk_order: np.ndarray, parents_fk: np.ndarray) -> float:
    """Max spread of ch3:9 across child slots sharing a parent (in FK order). The encoder writes the
    SAME parent rotation into every child slot; FK is last-child-wins while skinning is
    first-child-wins, so if these disagree the two decoders diverge (codex P0-1). Clean data -> ~0."""
    p = np.asarray(parents_fk)
    worst = 0.0
    for pp in np.unique(p[p >= 0]):
        slots = np.flatnonzero(p == pp)
        if slots.size >= 2:
            block = motion_fk_order[:, slots, 3:9]     # [T, k, 6]
            worst = max(worst, float(np.abs(block - block[:, :1, :]).max()))
    return worst


def gate_a_shape_finite(motion: np.ndarray, source_frames: int | None = None,
                        foot_joint_idx=None, parents=None) -> dict:
    """Per-CLIP integrity (runs on EVERY clip): shape [T,J,13] float, finite, T=F-1 (HARD when
    source_frames given), T>0, abs-range sane, contact binary + zero off declared feet, rot6d rows
    non-degenerate, and — when `parents` is given — the multi-child rot6d duplication invariant
    (every child slot of a parent must carry the SAME rot6d, else FK/skinning decoders diverge; this
    is a per-clip property, so it belongs here, not only in the per-topology Gate C probe)."""
    ok, issues = True, []
    if motion.ndim != 3 or motion.shape[-1] != 13:
        return {"gate": "A", "pass": False, "abs_max": 0.0, "issues": [f"shape {motion.shape} is not [T,J,13]"]}
    if motion.shape[0] < 1:
        ok = False; issues.append(f"T={motion.shape[0]} < 1 (empty clip)")
    if motion.dtype not in (np.float32, np.float64):
        issues.append(f"dtype {motion.dtype} (expected float32)")     # warn: build casts to float32
    if not np.isfinite(motion).all():
        ok = False; issues.append(f"{int((~np.isfinite(motion)).sum())} non-finite values")
    amax = float(np.abs(motion).max()) if motion.size else 0.0
    if amax > ABS_VALUE_CEILING:
        ok = False; issues.append(f"abs-max {amax:.3f} > ceiling {ABS_VALUE_CEILING}")
    if source_frames is not None and motion.shape[0] != source_frames - 1:
        ok = False; issues.append(f"T={motion.shape[0]} != F-1={source_frames - 1} (velocity/contact frame-diff)")
    # contact channel (ch12): binary, and zero off declared feet
    if motion.size:
        contact = motion[..., 12]
        if not np.isin(np.unique(np.round(contact, 6)), (0.0, 1.0)).all():
            ok = False; issues.append("ch12 contact is not binary {0,1}")
        if foot_joint_idx is not None:
            kind, feet = check_index_list(foot_joint_idx, motion.shape[1], "foot_joint_idx")
            if kind in ("valid", "empty"):
                # "empty" declares NO foot joints -> contact must be 0 on EVERY joint (not skipped)
                feet_arr = feet if kind == "valid" else np.empty(0, int)
                non_feet = np.setdiff1d(np.arange(motion.shape[1]), feet_arr)
                if non_feet.size and float(np.abs(contact[:, non_feet]).max()) > 1e-6:
                    ok = False; issues.append("ch12 contact non-zero on a non-foot joint")
            elif kind == "error":
                ok = False; issues.append(f"foot_joint_idx malformed: {feet}")
        # rot6d non-degeneracy (HARD): each row's two 3-vectors must be non-zero and non-parallel,
        # else Gram-Schmidt can't form a rotation. Valid encoded data has unit, orthogonal rows
        # (norms==1, |cross|==1); an all-zero or collapsed row is broken.
        a, b = motion[..., 3:6], motion[..., 6:9]
        na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
        cross = np.linalg.norm(np.cross(a, b), axis=-1)
        if float(np.min(na)) < 1e-6 or float(np.min(nb)) < 1e-6:
            ok = False; issues.append("a rot6d row has a zero-norm vector (ch3:9 broken/all-zero)")
        elif float(np.min(cross)) < ROT6D_DEGENERATE_TOL:
            ok = False; issues.append("a rot6d row is degenerate (two vectors ~parallel) — check ch3:9")
        if parents is not None:
            spread = _multi_child_consistency(motion, np.asarray(parents))  # order-agnostic (groups by parent)
            if spread > MULTI_CHILD_TOL:
                ok = False
                issues.append(f"child slots of one parent carry different rot6d (spread {spread:.3g}) "
                              f"=> FK(last-child) vs skinning(first-child) diverge")
    return {"gate": "A", "pass": ok, "abs_max": amax, "issues": issues}


def gate_b_ric_equivalence(motion: np.ndarray, source_world: np.ndarray | None = None,
                           source_recover_fn=None, tol: float = 1e-4) -> dict:
    """The source's OWN official world recovery must match recover_from_bvh_ric_np(ch0:3).

    Prefer `source_world` [T,J,3] (a genuinely independent recovery from the source's own data).
    `source_recover_fn(motion)->world` is a deprecated shim (it only sees the converted IR motion).
    Absent both, Gate B is WAIVED (pass=True, waived flag) — an explicit waiver, surfaced in the report.
    """
    if source_world is None and source_recover_fn is None:
        return {"gate": "B", "pass": True, "waived": True, "note": "no source-official recovery provided"}
    a = np.asarray(recover_from_bvh_ric_np(np.asarray(motion, np.float64)), np.float64)[..., :3]
    b = np.asarray(source_world if source_world is not None else source_recover_fn(motion), np.float64)[..., :3]
    if a.shape != b.shape:
        return {"gate": "B", "pass": False, "waived": False, "issues": [f"shape mismatch {a.shape} vs {b.shape}"]}
    err = float(np.linalg.norm(a - b, axis=-1).mean())
    return {"gate": "B", "pass": err <= tol, "waived": False, "mean_l2": err, "tol": tol}


def gate_c_fk_selfcheck(motion: np.ndarray, parents: np.ndarray, offsets: np.ndarray,
                        inferred_twist: bool = False, mode: str | None = None) -> dict:
    """THE invariant: FK(ch3:9) and RIC(ch0:3) recover the same world positions.

    Reorders motion/parents/offsets to FK-BFS order internally (so it is correct for ANY input joint
    order), runs the FK vs RIC recovery, and also checks the multi-child rotation-duplication
    invariant the skinning decoder depends on. A large selfcheck means a broken rot6d convention
    (spec §2) or a genuinely inconsistent encode -- NOT a model defect.
    """
    if mode is not None:
        inferred_twist = (mode == "inferred_twist")
    m = np.asarray(motion, np.float64)
    perm, new_parents = bfs_reorder(parents)
    m_fk = m[:, perm, :]
    off_fk = np.asarray(offsets, np.float64)[perm]
    fk = np.asarray(recover_from_bvh_rot_np(m_fk, new_parents, off_fk), np.float64)[..., :3]
    ric = np.asarray(recover_from_bvh_ric_np(m_fk), np.float64)[..., :3]
    mean_l2 = float(np.linalg.norm(fk - ric, axis=-1).mean())
    bbox = _bbox_diag(ric)
    rel = mean_l2 / bbox
    multi_child = _multi_child_consistency(m_fk, new_parents)
    fk_ok = (rel <= GATE_C_BBOX_TOL) if inferred_twist else (mean_l2 <= GATE_C_CLEAN)
    child_ok = multi_child <= MULTI_CHILD_TOL
    passed = bool(fk_ok and child_ok)
    hint = "" if passed else (
        "large FK-vs-RIC => check joint order (spec §3) or rot6d convention (spec §2)" if not fk_ok
        else "child slots of one parent carry different rot6d => FK(last-child) vs skinning(first-child) will diverge")
    return {"gate": "C", "pass": passed, "mean_l2": mean_l2, "bbox_diag": bbox, "rel_bbox": rel,
            "multi_child_spread": multi_child, "fk_pass": bool(fk_ok), "child_pass": bool(child_ok),
            "mode": "inferred_twist" if inferred_twist else "clean", "hint": hint}


def gate_e_layout(data_root: str, motion_object_types: dict | None = None) -> dict:
    """Structural loader contract the training loader depends on (NOT the full loader ingest — that
    runs in the training repo; see spec §8 note). Fails loudly on anything the loader would choke on.

    Verified: cond.npy loads & is NON-empty; each object_type has the required keys with finite
    mean/std and a single-root parents; motions/ non-empty; EVERY motion maps to a known object_type;
    splits present, non-empty union, train/val disjoint; bone-length scale within a sane band.
    """
    root = Path(data_root); issues = []; scale_unverified = []
    cond_p = root / "cond.npy"
    if not cond_p.exists():
        return {"gate": "E", "pass": False, "issues": ["missing cond.npy"]}
    cond = np.load(cond_p, allow_pickle=True).item()
    if not cond:
        return {"gate": "E", "pass": False, "issues": ["cond.npy is empty (no object_types)"]}
    req = {"parents", "offsets", "joints_names", "tpos_first_frame", "mean", "std"}
    for ot, rec in cond.items():
        missing = req - set(rec.keys())
        if missing:
            issues.append(f"{ot}: cond missing {sorted(missing)}"); continue
        p = np.asarray(rec["parents"])
        if p.ndim != 1 or p.shape[0] < 1:
            issues.append(f"{ot}: parents must be a non-empty 1-D array"); continue
        J = p.shape[0]
        if (p == -1).sum() != 1 or p[0] != -1:
            issues.append(f"{ot}: parents must have a single root at index 0")
        # primary-field shapes + finiteness
        shape_ok = True
        for k, want in (("offsets", (J, 3)), ("tpos_first_frame", (J, 13)),
                        ("mean", (J, 13)), ("std", (J, 13))):
            arr = np.asarray(rec[k])
            if arr.shape != want:
                issues.append(f"{ot}: {k} shape {arr.shape} != {want}"); shape_ok = False
            elif not np.isfinite(np.asarray(arr, np.float64)).all():
                issues.append(f"{ot}: {k} contains non-finite values")
        if len(rec["joints_names"]) != J:
            issues.append(f"{ot}: {len(rec['joints_names'])} joints_names != {J}")
        std = np.asarray(rec["std"], np.float64)
        if std.size and std.min() < 0:
            issues.append(f"{ot}: std has negative values")
        if shape_ok:
            off = np.asarray(rec["offsets"], np.float64)
            # validate the declaration structure for ALL J (so a malformed idx is caught even at J=1);
            # compute the actual scale ratio only when there are bones (J>1). Same helper as Topology.
            kind, payload = check_index_list(rec.get("scale_ref_joint_idx"), J, "scale_ref_joint_idx")
            valid_idx = payload if kind == "valid" else None
            undeclared = kind in ("none", "empty")
            if kind == "error":
                issues.append(f"{ot}: {payload}")
            fkind, fpay = check_index_list(rec.get("foot_joint_idx"), J, "foot_joint_idx")
            if fkind == "error":
                issues.append(f"{ot}: {fpay}")
            if J > 1 and valid_idx is not None:
                # adapter declared the joints it scaled to HML -> verify EXACTLY (mean(subset) == HML)
                blen = np.linalg.norm(off[valid_idx], axis=-1)
                blen = blen[blen > 1e-8]
                mean_blen = float(blen.mean()) if blen.size else 0.0
                ratio = mean_blen / HML_AVG_BONELEN if HML_AVG_BONELEN else 0.0
                if not (0.98 <= ratio <= 1.02):          # exact: scale() sets mean(subset)==HML by construction
                    issues.append(f"{ot}: mean scale-ref bone length {mean_blen:.4f} is {ratio:.3f}x "
                                  f"HML_AVG_BONELEN (must be ~1.0 — declared scale_ref_joint_idx were "
                                  f"supposed to be rescaled to HML)")
            elif J > 1 and undeclared:
                # No declared subset. The reference `scale()` normalizes a per-source reference-bone
                # subset to HML; without that subset mean-over-ALL bones legitimately spans ~0.39x–1.42x
                # across the existing 382 object_types, so only a GROSS band is enforceable (a uniform
                # mis-scale is also caught by Gate D). Surfaced in `scale_unverified` so it is not silent.
                blen = np.linalg.norm(off[1:], axis=-1)
                mean_blen = float(blen[blen > 0].mean()) if (blen > 0).any() else 0.0
                ratio = mean_blen / HML_AVG_BONELEN if HML_AVG_BONELEN else 0.0
                scale_unverified.append(ot)
                if not (0.3 <= ratio <= 3.0):
                    issues.append(f"{ot}: mean bone length {mean_blen:.4f} is {ratio:.2f}x HML_AVG_BONELEN "
                                  f"(gross scale error; declare scale_ref_joint_idx for an exact check)")
    mots = sorted((root / "motions").glob("*.npy")) if (root / "motions").is_dir() else []
    if not mots:
        issues.append("no motions/*.npy")
    # every motion must map to a known object_type. If a manifest is supplied it is AUTHORITATIVE:
    # every motion must appear in it (no silent prefix fallback, which can map Wolfish->Wolf).
    stems = {p.stem for p in mots}
    unmapped = []
    for st in stems:
        if motion_object_types is not None:
            ot = motion_object_types.get(st)
        else:
            ot = _object_type_for(st + ".npy", cond)
        if ot is None or ot not in cond:
            unmapped.append(st)
    if unmapped:
        issues.append(f"{len(unmapped)} motion(s) map to no object_type, e.g. {sorted(unmapped)[:3]}")
    # splits: present, non-empty, disjoint, cover EVERY motion exactly once, no dangling ids
    split_sets = {}
    for sp in ("train", "val"):
        f = root / "splits" / f"{sp}.txt"
        split_sets[sp] = set(f.read_text().split() if f.exists() else [])
    union = split_sets["train"] | split_sets["val"]
    if not union:
        issues.append("splits/{train,val}.txt empty or missing")
    inter = split_sets["train"] & split_sets["val"]
    if inter:
        issues.append(f"{len(inter)} id(s) in BOTH train and val, e.g. {sorted(inter)[:3]}")
    if union - stems:
        issues.append(f"{len(union - stems)} split id(s) with no motion file, e.g. {sorted(union - stems)[:3]}")
    if stems - union:
        issues.append(f"{len(stems - union)} motion(s) in NO split, e.g. {sorted(stems - union)[:3]}")
    out = {"gate": "E", "pass": not issues, "n_object_types": len(cond), "n_motions": len(mots),
           "issues": issues}
    if scale_unverified:
        out["scale_unverified"] = (f"{len(scale_unverified)} object_type(s) had no scale_ref_joint_idx: "
                                   f"scale only gross-checked (rely on Gate D), e.g. {scale_unverified[:3]}")
    return out


# back-compat alias (older callers)
gate_e_loader_smoke = gate_e_layout


# ------------------------------------------------------------------ CLI regression mode
def _object_type_for(name: str, cond: dict) -> str | None:
    return next((k for k in sorted(cond, key=len, reverse=True) if name.startswith(k)), None)


def _run_cli(data_root: str, object_type: str | None, large_rotation: bool, limit: int | None,
             allow_unmarked: bool = False) -> dict:
    root = Path(data_root)
    cond = np.load(root / "cond.npy", allow_pickle=True).item()
    manifest_p = root / "motion_object_types.json"
    manifest = json.loads(manifest_p.read_text()) if manifest_p.exists() else None
    report = {"data_root": data_root, "E": gate_e_layout(data_root, manifest), "clips": []}
    mots = sorted((root / "motions").glob("*.npy"))
    n_total = len(mots)

    def excursion(p):
        m = np.load(p); return float(np.abs(np.diff(m[:, 0, 3:9], axis=0)).sum()) if m.ndim == 3 and m.shape[0] > 1 else 0.0
    if large_rotation:
        mots = sorted(mots, key=excursion, reverse=True)
    sampled = limit is not None and limit < n_total
    if sampled:
        mots = mots[:limit]
    unmatched = 0
    for p in mots:
        m = np.load(p)
        ot = object_type or (manifest or {}).get(p.stem) or _object_type_for(p.name, cond)
        if ot is None or ot not in cond:
            report["clips"].append({"clip": p.name, "error": "no cond match"}); unmatched += 1; continue
        rec = cond[ot]
        pj = np.asarray(rec["parents"])
        if pj.ndim != 1 or pj.shape[0] < 1:                 # malformed topology record (Gate E flags it too)
            report["clips"].append({"clip": p.name, "object_type": ot,
                                    "error": "topology parents malformed"}); unmatched += 1; continue
        J = pj.shape[0]
        if m.ndim != 3 or m.shape[1:] != (J, 13):          # J must match the topology, else gates are meaningless
            report["clips"].append({"clip": p.name, "object_type": ot,
                                    "error": f"shape {m.shape} != [T,{J},13]"}); unmatched += 1; continue
        a = gate_a_shape_finite(m, foot_joint_idx=rec.get("foot_joint_idx"), parents=rec["parents"])
        # only run FK/RIC recovery on shape-valid clips (avoids a traceback on malformed data)
        c = (gate_c_fk_selfcheck(m, rec["parents"], rec["offsets"], mode=rec.get("gate_c_mode", "clean"))
             if a["pass"] else {"gate": "C", "pass": False, "skipped": "Gate A failed"})
        report["clips"].append({"clip": p.name, "object_type": ot, "A": a, "C": c})
    a_checks = [c["A"]["pass"] for c in report["clips"] if "A" in c]
    c_checks = [c["C"]["pass"] for c in report["clips"] if "C" in c]
    accepted = (root / "_ACCEPTED").exists()               # build's acceptance marker (Gate B lives there)
    report["n_checked"] = len(a_checks)
    report["n_total"] = n_total
    report["sampled"] = sampled
    report["ACCEPTED_marker"] = accepted
    if sampled:
        report["WARNING"] = f"SAMPLED {len(mots)} of {n_total} motions — NOT a full check (drop --limit for all)"
    if not accepted:
        report.setdefault("WARNING", "no _ACCEPTED marker — build_source did not accept this dataset "
                          "(a legacy dataset, or Gate B/other failed). CLI re-checks A/C/E only (Gate B "
                          "needs source_world). Pass --allow_unmarked to accept an A/C/E-only verdict.")
    # any unmatched clip, empty check, sampling, or missing acceptance marker => FAIL (not vacuous True)
    ok = unmatched == 0 and not sampled and (accepted or allow_unmarked)
    report["A_all_pass"] = bool(a_checks) and all(a_checks) and ok
    report["C_all_pass"] = bool(c_checks) and all(c_checks) and ok
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--object_type", default=None, help="force one object_type (else manifest / longest-prefix)")
    ap.add_argument("--large_rotation", action="store_true", help="order the report by root-rotation excursion")
    ap.add_argument("--limit", type=int, default=None, help="check only the first N motions (SAMPLED; never passes)")
    ap.add_argument("--allow_unmarked", action="store_true",
                    help="accept an A/C/E-only verdict on a dataset lacking build's _ACCEPTED marker (e.g. legacy data)")
    args = ap.parse_args()
    report = _run_cli(args.data_root, args.object_type, args.large_rotation, args.limit, args.allow_unmarked)
    print(json.dumps(report, indent=1, default=float))
    ok = report["E"]["pass"] and report.get("A_all_pass") and report.get("C_all_pass")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
