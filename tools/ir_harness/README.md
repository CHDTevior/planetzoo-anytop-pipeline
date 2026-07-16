# AnyTop-13 IR Harness — adding a new data source

An agent-runnable scaffold for turning **any** new motion source (a game rig, a mocap set, another
animal library) into the shared **AnyTop-13 Intermediate Representation** so it can join the same
Graph-VQVAE → Graph-CodeFlow codebook. Read
[`docs/ANYTOP13_INTERMEDIATE_REPRESENTATION.md`](../../docs/ANYTOP13_INTERMEDIATE_REPRESENTATION.md)
**first** — this harness is only the mechanics; that doc is the contract.

> **Who runs this.** This is written for an **agent** to execute end-to-end: implement one
> `SourceAdapter` subclass for the new source, run the driver, and the harness emits the IR and runs
> the acceptance gates. The model, the graph fields, the normalization, and the FK conventions are
> **not** yours to touch — the harness derives/enforces them.

---

## What you produce (the only outputs)

```
<out_root>/
  motions/<motion_id>.npy        # float32 [T, J, 13]   (RAW, ORIGINAL joint order)
  cond.npy                       # dict{object_type -> cond record}   (ORIGINAL joint order)
  motion_texts_by_file.json      # {motion_id -> {primary_caption, captions:[...]}}
  motion_object_types.json       # {motion_id -> object_type}   (explicit map; no prefix guessing)
  splits/{train,val}.txt         # motion_id per line
  _gate_report.json              # Gates A/B/C/E + overall_hard_pass (written by the harness)
  _ACCEPTED                      # written ONLY when all hard gates pass — CONSUMERS MUST require it
```

`build_source.build(...)` **raises `AcceptanceError` on any red hard gate** (A/B/C/E) unless you pass
`strict=False`; it writes `_gate_report.json` first so the failure is inspectable. It writes the
`_ACCEPTED` marker **only** when `overall_hard_pass` is true (and clears any stale marker at the
start) — **a downstream consumer must refuse a dataset that has no `_ACCEPTED` file** (plus do the
manual Gate D). It also refuses a `<out_root>` that already holds any artifact unless `overwrite=True`
(so stale files can't leak into a new build).

`cond` record per `object_type`: `parents [J] int` (single-root `-1`, connected), `offsets [J,3]`,
`joints_names [J] str`, `tpos_first_frame [J,13]`, `mean [J,13]`, `std [J,13]` (plus `foot_joint_idx`,
`gate_c_mode` used by the gates). Everything else (adjacency / geodesic / graph_dist / joint_relations
/ skeleton_features / name_hashes / `new_to_old_perm`) is **derived by the training loader** — do
**not** ship it. See spec §6.

Joint order of everything you write is the source's **ORIGINAL** order (the loader applies the
FK-reorder itself). See spec §3.

---

## The agent workflow

1. **Understand the source.** Rig topology (joints, parents, rest offsets), coordinate frame /
   up-axis, unit scale, fps, how rotations are stored, which joints are feet, where captions come
   from. Fill in the `SourceAdapter` docstring answers.

2. **Implement one `SourceAdapter` subclass** (`source_adapter.py`). This is where the **per-source**
   work lives, including the source→13ch **encoding** — the source→13ch step *is* the part that
   differs fundamentally by source family (BVH rig vs. pose-based mocap), so the adapter owns it
   (spec §7). You implement exactly four methods:
   - `iter_object_types()` → every distinct skeleton topology this source contributes.
   - `topology(object_type)` → a `Topology` (`parents`, `offsets`, `joints_names`,
     `tpos_first_frame`) in the source's **original** joint order.
   - `iter_clips(object_type)` → `Clip`s whose `.motion` is already **`[T,J,13]`** — you apply the
     Y-up/+Z transform, `HML_AVG_BONELEN` rescale, 20fps resample + drop-last (`T=F−1`), the
     **per-parent rot6d** re-encoded via Kabsch so FK==RIC, root facing-6D + XZ-velocity into
     `ch9/ch11`, contact into `ch12`, twist pinned. Set `Clip.source_frames` (for the Gate-A `T=F−1`
     check) and, when the source has an official world recovery, `Clip.source_world [T,J,3]` (for the
     independent Gate B). Two **template stubs** — `BvhPipelineAdapter` (delegate to the existing BVH
     exporter + `motion_process.process_object`, which already emits `[T,J,13]`) and
     `PoseSourceAdapter` (implement the Kabsch re-encoding of spec §2/§7) — show the two families;
     they raise `NotImplementedError` until you fill them in.
   - `topology()` also declares the per-topology gate policy: `foot_joint_idx` (contact must be 0 off
     these) and `gate_c_mode` (`"clean"` or `"inferred_twist"`).

3. **Run the driver** (`build_source.build(adapter, out_root, max_joints=…)`). It does **only** the
   source-agnostic back-end — never any encoding: it collects your `Clip`s, computes per-`object_type`
   `mean/std` over the **train** split, writes `motions/` + `cond.npy` + `motion_texts_by_file.json` +
   `motion_object_types.json` + `splits/`, runs the gates, and **raises on a red hard gate**.
   `mean/std` mirrors `motion_process.get_mean_std` exactly: root blocks collapse to their own scalar,
   **all non-root joints share one scalar per pos/rot/vel block**, contact non-zeros→their mean /
   zeros→1.0, no `1e-6` baked into the stored std (the loader adds `+1e-6` at normalize time). Every
   convention that must match the existing data was already applied by **your adapter**; the driver
   just packages + normalizes + verifies.

4. **Gates A/B/C/E run automatically**, write `_gate_report.json`, and a red hard gate **raises
   `AcceptanceError`** — the dataset is not accepted. Gate D (visual) is **not** automated: render
   GT-vs-converted GIFs with the repo's render tools and eyeball them yourself (visual QA is
   authoritative; a leg-flip passes every scalar gate). The report carries a `gate_D` reminder, not a
   rendered result — you must still do Gate D by hand before training.

5. **Merge (optional).** Because normalization is per-skeleton, the union is a concatenation:
   `cond.npy` records are merged, `motions/` pooled, splits concatenated. No re-normalization. To add
   the source to the **CodeFlow backbone** you must then re-run VQVAE token export (spec §6 backbone
   note) — the harness stops at the VQVAE-ready dataset.

---

## The acceptance gates (spec §8)

| gate | scope | checks | pass |
|---|---|---|---|
| **A** per-clip integrity | **every clip** | `[T,J,13]`, finite, `T=F−1` (hard when `source_frames` set) & `T>0`, abs-range sane, contact binary + 0 off `foot_joint_idx`, rot6d non-degenerate, **multi-child rot6d duplication** consistent | hard |
| **B** RIC equivalence | **every clip** with `source_world` | source's own world recovery (`Clip.source_world`) `==` `recover_from_bvh_ric_np(ch0:3)`; **waived** (flagged) if no source recovery | `mean_l2≤1e-4` |
| **C** FK self-consistency | **every clip** | BFS-reorders internally, then `recover_from_bvh_rot_np` (FK) vs RIC agree | `L2<1e-4` clean / ≤0.5% bbox `inferred_twist` |
| **D** visual QA | manual | GT-vs-converted GIF (skeleton + mesh if skinnable), side view — rendered **manually**, not by `build_source.py` | **human/agent eye — authoritative** |
| **E** layout contract | dataset | `cond.npy` loads & non-empty; every `object_type` has required keys + finite `mean/std`/`offsets`/`tpos` + correct shapes + single-root-at-0 `parents` + gross bone-length scale; every motion maps to a topology; every motion in **exactly one** split (disjoint + full coverage) | hard |

`build_source` runs Gates A **and** C on **every** clip (per-clip coverage; the report keeps one
representative Gate-C per type in `gate_C_sample_per_type`) and Gate B on every clip carrying
`source_world`. The standalone `gates.py` CLI re-checks A/C/E from disk only — it cannot re-run Gate B
(no `source_world` on disk), so it reads the build's `_ACCEPTED` marker and, without it, refuses to
report pass unless you pass `--allow_unmarked` (for legacy datasets). Full training-loader ingest
(instantiating `AnyTopDataset`, deriving the graph fields) happens in the **training repo** — Gate E
checks only the on-disk contract that loader depends on.

`gates.py` implements A/B/C/E as functions and can be run standalone against an existing dataset as a
regression check:

```bash
python -m tools.ir_harness.gates --data_root <out_root> --object_type <OT> --large_rotation
```

---

## Non-negotiables (why they exist — see spec)

- **Per-parent rot6d** (spec §2): child slot `j` stores `parent[j]`'s rotation. Your adapter must
  emit this layout; never store per-joint-own rotations. Gate C catches a wrong convention.
- **Joint order** (spec §3): write ORIGINAL order; never pre-apply the FK permutation.
- **No `apply_root_cancel`** (spec §5): the double-root-rotation correction is removed; keep it off.
- **Re-encode rotations via Kabsch** into AnyTop's rest basis (spec §7): a direct gather of the
  source's native rotations is broken (up to ~89° bone-basis error). Your adapter does this (the
  `PoseSourceAdapter` stub points at the algorithm in spec §2/§7; it is a template, not an
  implementation); the driver only verifies the result via Gate C.
- **Recompute `mean/std`** on the train split per object_type (spec §4/§8): the driver does this —
  never ship the source's own flat stats in `cond`.
- **Pin the twist DOF** at encoding when inferring rotation from positions (spec §10): an unconstrained
  axial DOF is a fine gauge but an unlearnable target.

---

## Files

- `source_adapter.py` — the `SourceAdapter` abstract base (the four methods you implement per source,
  incl. the source→13ch encoding) + `Topology`/`Clip` dataclasses + `BvhPipelineAdapter` /
  `PoseSourceAdapter` template stubs pointing at the two reference paths.
- `build_source.py` — the driver: collect adapter `Clip`s → per-object `mean/std` →
  `motions/`+`cond.npy`+captions+splits → Gates A/B/C/E + `_gate_report.json`. **Does no encoding.**
- `gates.py` — Gates A/B/C/E as runnable functions (+ a CLI regression mode).
- `_ref_recover.py` — vendored pure-numpy FK/RIC recovery (`recover_from_bvh_rot_np` /
  `recover_from_bvh_ric_np`) that Gates B/C use, so the gate runner needs only numpy.

*Reference implementations to mirror:* the Planet-Zoo path
(`tools/planetzoo/planetzoo_fulltopo_bvh_export.py` → `utils.process_new_skeleton` →
`data_loaders/truebones/truebones_utils/motion_process.py`, all **in this repo**) already produces
this IR from BVH. The HumanML3D/SMPL Kabsch converter lives in the **training repo** as
`scripts/convert_humanml3d_to_anytop13.py` and is **not vendored here** — for a pose-based source,
implement the Kabsch re-encoding described in spec §2/§7 rather than importing that file.
