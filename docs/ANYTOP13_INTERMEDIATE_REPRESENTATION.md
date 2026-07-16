# AnyTop-13 Intermediate Representation (IR)

A source-agnostic, channel-level specification of the motion representation used by the
Graph-VQVAE → Graph-CodeFlow pipeline, plus the exact contract a **new data source** must
satisfy to enter it. This is the reference every data-prep adapter (and every agent running
the harness in [`tools/ir_harness/`](../tools/ir_harness/README.md)) must conform to.

> **Why an IR.** Today the pipeline already merges two structurally different sources into
> **one** shared codebook: Planet Zoo animals (311 topologies, native BVH rotations with real
> axial twist) and HumanML3D/AMASS humans (22-joint SMPL). They coexist because they share this
> exact 13-channel layout, the per-parent rot6d convention, and the per-skeleton graph fields —
> **only the front-end conversion is per-source**. This doc pins that boundary so future sources
> (other games, other mocap, other rigs) can be added by writing one adapter, not by touching the
> model.

Scope: this repo (`planetzoo-anytop-pipeline`) owns **data acquisition & preparation** — turning
raw source assets into the IR. The FK/graph *consumption* conventions the IR must obey are stated
here as a contract; the reference encoder/decoder in this repo is
`data_loaders/truebones/truebones_utils/motion_process.py`.

---

## 1. The tensor: `motion` = `float32 [T, J, 13]`

One clip is `[T, J, 13]`: `T` frames, `J` joints, 13 channels per joint. **`T = F − 1`** where `F`
is the source frame count — the last frame is dropped because velocity and contact are frame
differences (`motion_process.py get_motion`). Stored **un-normalized** (raw physical-ish units, see
§4). Joint order is the source's **original** order in `motions/*.npy` — see the order footgun in §3.

### 1.1 Channel map

| ch | non-root joint `j≥1` | root joint `j=0` |
|----|----------------------|------------------|
| `0:3` | **RIC position** — joint xyz relative to the root origin, expressed in the per-frame root-facing (yaw-cancelled) frame | root RIFKE state; **only `ch1` is meaningful = root height (Y)**. `ch0`,`ch2` are unused by every decoder |
| `3:9` | **6D rotation** (Zhou et al. 2019) of this joint's **PARENT** — see §2 | per-frame **root facing** rotation (6D), used to integrate root travel |
| `9:12` | per-joint local velocity (auxiliary; see §1.2) | `ch9`,`ch11` = root **X,Z linear velocity** (per-frame displacement, root-local); `ch10` unused |
| `12` | binary **foot-contact** flag `∈{0,1}` (meaningful on feet, `0` elsewhere) | normally `0` (not hard-wired: a source whose contact-joint selection includes the root — e.g. legless snakes — can set it; the IR contract keeps it `0` on the root unless the source deliberately selects it) |

Block boundaries `0:3 | 3:9 | 9:12 | 12` are verbatim in `docs/PLANETZOO_ANYTOP_USAGE.md:292-301`.

### 1.2 What each channel actually drives (verified from code)

Two decoders exist: **FK/skinning is rotation-driven; the RIC path is position-driven** (it reads the
`ch0:3` positions). The load-bearing *generative* signal is the rotations, but the two are not both
rotation-driven:

- **FK / skinning** (`recover_from_bvh_rot_np`; skinning `decode_feature_rotations`) reads **only**:
  non-root `ch3:9` (per-parent rotations) + root `ch1`,`ch3:9`,`ch9`,`ch11` + the skeleton's
  `offsets`/`parents`. It **ignores** all `ch0:3` (non-root RIC), `ch10`, non-root `ch9:12`, and `ch12`.
- **RIC path** (`recover_from_bvh_ric_np` in this repo's `motion_process.py`; the training loader's
  equivalent method is `_recover_world_positions`) — used to build the world-position training target
  and as a cross-check — reads non-root `ch0:3` + the same root state.

So the **load-bearing signal** is: the 6D rotations `ch3:9` **plus the root state — three scalars
(`ch1` height, `ch9`/`ch11` XZ velocity) and the root facing-6D (`ch3:9` of joint 0)**. `ch0:3`
(non-root) are load-bearing only for the RIC path. Truly inert in every decoder: **root `ch0`/`ch2`,
and `ch10` (velocity-Y) for all joints**. `ch12` is unused by FK/RIC/skinning but is consumed by the
remove-joints augmentation and exposed as `foot_contact_per_joint`.

> **Design implication for a new source.** You must get `ch3:9` (rotations, per-parent) and the root
> state (`ch1`, `ch9`, `ch11`, root `ch3:9`) exactly right; `ch0:3` must be self-consistent with them
> (§5 invariant). `ch10`, root `ch0`/`ch2` may be zero. `ch12` must be a real contact on the feet.

---

## 2. The per-parent rot6d convention (the #1 semantic subtlety)

**For a non-root child slot `j`, `motion[:, j, 3:9]` stores the local rotation of `j`'s PARENT
(`parent[j]`), NOT joint `j`'s own rotation.** The root slot stores the per-frame facing rotation.

- **Encode** (`motion_process.py get_bvh_cont6d_params`): `for j,p in enumerate(parents[1:],1):
  cont_6d_reordered[:, j] = cont_6d[:, p]`; slot 0 = 6D of the frame facing quaternion.
- **Decode** (`recover_from_bvh_rot_np`): `for j,p in enumerate(parents[1:],1):
  rot_q[:, p] = all_q_hml[:, j]` — the token at child slot `j` is written back into parent slot `p`.

Consequences:
- A **leaf / end-effector** joint (nobody's parent) has **no slot for its own rotation** → FK/skinning
  leave leaves at rest/identity. This is correct: a leaf's own orientation is unobservable here.
- A parent with **multiple children** has its single rotation duplicated across those child slots. On
  clean data all agree; on noisy/generated data the two decoders differ at branch points (FK is
  last-child-wins; the skinning POC is deterministic first-child-wins via a `child_for_parent` dict).

> **Footgun:** naively treating `motion[:, j, 3:9]` as "joint `j`'s rotation" applies every rotation to
> the wrong bone (off-by-one up the chain). Always use the repo's `recover_from_bvh_rot_np`.

---

## 3. Joint order — the `new_to_old_perm` footgun

There are **two joint orders** and mixing them silently produces **fake leg-flips** (looks like a
model bug; it is a data-plumbing bug — cost us an entire debugging session, see the note at the end).

- **ORIGINAL order** — `motions/*.npy`, `cond.npy` (all arrays), and the exported minipack
  `skeleton.json` are in the source's original joint order.
- **FK/BFS order** — the training loader BFS-reorders each topology so `parents[0]==-1` and
  `parents[j] < j`, stashing `new_to_old_perm` (`new_to_old_perm[new_idx] = old_idx`). At
  `__getitem__` it reindexes the raw clip: `raw_motion = raw_motion[:, cond["new_to_old_perm"], :]`.
  **Everything the model sees or exports is in this NEW order**; the normalized cond cache
  (`_cond_normalized_J*.pkl`) is also in NEW order.

**Conversion:** `inv = np.argsort(new_to_old_perm); motion_old = motion_new[:, inv, :]` (NEW→ORIGINAL),
or `motion_new = motion_old[:, new_to_old_perm, :]` (ORIGINAL→NEW). Any FK / skinning / analysis must
use **one** order consistently for the motion **and** the `parents`/`offsets`/skeleton.

> Rule of thumb: **model output / gen export is NEW order; the skinning pipeline + `cond.npy` are
> ORIGINAL order.** Convert the export with `argsort(new_to_old_perm)` before skinning (the reference
> exporter now does this).

The FK-reorder is source-agnostic but **requires exactly one root (`parent==-1`) and a connected,
single-rooted tree**, else the loader raises.

---

## 4. Normalization

Per-**object-type**, per-**joint**, per-**channel** mean/std, stored as `[J,13]` in the cond record:

```
std_safe = std + 1e-6          # _STD_FLOOR
normed   = (raw - mean) / std_safe
```

- The model consumes the **normalized** view; FK/RIC/skinning consume **raw** — de-normalize first:
  `raw = normed * (std + 1e-6) + mean` (with the FK-ordered `mean`/`std`), **then** apply the joint
  permutation if you need ORIGINAL order (§3). Doing only one of the two leaves a mismatch.
- Std is **channel-group-homogenized** at build time (`motion_process.py get_mean_std`): the **root**
  joint's pos(`0:3`)/rot(`3:9`)/vel(`9:12`) blocks each collapse to that block's own scalar mean, while
  **all non-root joints share a single scalar per block** (the mean over every non-root joint *and*
  every channel in the block) — non-root joints are *not* homogenized independently. Contact (`12`):
  non-zero stds → their mean; zero → `1.0`. No `1e-6` is baked into the stored std — it is added at
  normalize time (`std + 1e-6` above).
- Because normalization is **self-contained per skeleton**, **merging sources needs no
  re-normalization** — the union `cond` is just a concatenation of records. Never reuse another
  skeleton's stats.

---

## 5. World recovery (FK) and the self-consistency invariant

Two independent recoveries of world joint positions `[T, J, 3]` from the **raw** 13ch:

- **FK path** `recover_from_bvh_rot_np(data, parents, offsets)` — 6D rotations `ch3:9` → rotation
  matrices → local→global bone-chain matmul on `offsets`; root translation from the root state (`ch1`,`ch9`,`ch11`,facing).
- **RIC path** `recover_from_bvh_ric_np(data)` (training-loader equivalent: `_recover_world_positions`)
  — non-root `ch0:3` inverse-rotated by the per-frame root facing + integrated root XZ.

**Root recovery (shared by both, and by the skinning `recover_root_positions`):** facing = `ch3:9` of
joint 0; cumulative XZ = `cumsum` of the inverse-facing-rotated `(ch9, ch11)` per-frame velocities;
height = `ch1` (taken directly, not integrated). No fps divide — `ch9/ch11` are per-frame deltas.

**THE INVARIANT (Gate C below):** on GT data in a **consistent joint order**, FK and RIC agree to
`L2 ≈ 0` (`selfcheck < 1e-4` for clean data). This is the single strongest data-integrity check: it
proves the rotation channels can drive skinning. A large selfcheck ⇒ either wrong joint order (§3) or a
broken rot6d convention (§2).

> **Double-root-rotation footgun (removed).** Stock AnyTop/SALAD `recover_from_bvh_rot_np` re-applies
> `rot_q[:,0] = -r_rot_quat * rot_q[:,0]` after the reindex, which **double-applies** the root yaw
> (the reordered root-child token already carries facing). This repo removed it (`apply_root_cancel`
> defaults **False**; `True` reproduces the buggy official behavior for debugging only). Evidence:
> with the correction FK-vs-RIC `absL1 = 0.65` (~2× rotation on turn clips); without it `= 0.0000`.
> **The bug is INVISIBLE on near-idle clips** — any regression check must use large-root-rotation
> clips (turns, circle-flies), never idle poses.

---

## 6. Per-skeleton graph / `cond` fields (what the graph model consumes)

Motion is per-clip; **everything below is static per-topology** and lives in the `cond` record. A new
source must supply the **primary** set; the **derived** set is computed for you by the loader.

> **Which loader.** "The loader" here means the **Graph-VQVAE training repo's `AnyTopDataset`** (the
> consumer of this IR), which does the FK-BFS reorder (§3) and derives the graph fields below. That
> module lives in the training repo, **not** in this `planetzoo-anytop-pipeline` repo — this repo's
> legacy `data_loaders/truebones` loader is a *different*, older loader and is **not** the IR consumer.
> The harness's Gate E therefore checks the on-disk contract only; the full ingest is verified in the
> training repo.

**Primary (a source MUST provide, per object_type):**

| field | shape | role |
|-------|-------|------|
| `parents` | `[J]` int | topology; **must be single-root & (after FK-reorder) `parents[j]<j`**. Everything graph-structural is derived from this. |
| `offsets` | `[J,3]` | rest-pose bone vectors; feed `skeleton_features` **and** the FK decoder (`rest_offsets`). |
| `joints_names` | `[J]` str | drive `name_hashes` + the left/right/center side heuristic in `skeleton_features`. |
| `tpos_first_frame` | `[J,13]` | rest/T-pose as a 13ch row (passthrough for parity/render; not load-bearing for the graph model). |
| `mean`,`std` | `[J,13]` | per-skeleton normalization (§4); **required**. |

**Derived by the loader from `parents`/`offsets`/`joints_names` (a source need NOT ship these; they
are recomputed):** `skeleton_features [J,9]` (`norm_offsets(3)+bone(1)+depth(1)+degree(1)+side_onehot(3)`
— the **only** static field projected into the initial joint token), `adjacency [J,J]`,
`geodesic_dist [J,J]` (true Floyd hops, unclamped), `joints_graph_dist [J,J]` (hop distance **clamped
at 5** — the Graphormer hop-bucket bias, distinct from `geodesic_dist`), `joint_relations [J,J]`
(6-class edge type: self/parent/child/sibling/no_relation/end_effector), `name_hashes [J]`
(`md5(name)%1024`). *`joint_relations`/`joints_graph_dist` present in `cond.npy` are only shape-checked
then discarded and re-derived.*

**Which field drives which mechanism (active v4b272 graphormer VQVAE):** node feature =
`skeleton_features`; joint-level attention bias = `joints_graph_dist` (hop-bucket) + `joint_relations`
(edge-type); graph pooling/coarsening = `adjacency` + `geodesic_dist`; FK decode = `offsets`+`parents`;
denorm = `mean`/`std`. `name_hashes` is **present-but-off** (`use_name_embed` defaults False in the
v4b272 VQVAE trainer). `tpos_first_frame`/`kinematic_chains` are passthrough.

> **Backbone note.** The CodeFlow backbone trains on **frozen VQVAE `z_q` tokens** and does not read
> `cond` at train time — the per-skeleton graph state is baked into the exported token cache. To add a
> new skeleton to the backbone: (1) add it to `cond` so the VQVAE can encode it, (2) **re-run token
> export**. You cannot swap skeletons at backbone inference without re-export.

> `d_model % n_heads == 0`; `skeleton_features` is fixed at **9** dims and motion at **13** ch — a new
> source must match these exactly. `max_joints` padding target must be `≥` the source's largest
> skeleton (larger skeletons are silently skipped at dataset build).

---

## 7. Universal vs per-source (the IR boundary)

**Universal core (identical for every source — do NOT re-implement per source):**
the `[T,J,13]` layout + root special-casing; the **per-parent rot6d** convention; **all** graph fields
(pure functions of `parents`/`offsets`/`names`); the FK-BFS ordering + `new_to_old_perm`; per-object
`mean/std` normalization; padding/masking; the caption/T5 pipeline (keyed by `motion_id`).

**Per-source adapter (what a new source must implement):**

| adapter concern | what to do |
|---|---|
| topology | define `parents`/`offsets`/`joints_names` for the source rig (single-root, connected) |
| coordinate frame / up-axis | transform to **Y-up, +Z-facing**. (PZ needs Z=−90° roll + hips→chest yaw + an orientation-selection test; HumanML3D is already Y-up/+Z.) |
| unit scale | rescale so **mean bone length = `HML_AVG_BONELEN = 0.2092142857142857`** (the shared metric anchor) |
| root origin + ground | subtract T-pose initial root XZ; shift so feet touch `Y=0` |
| fps / temporal | resample to **20 fps**; output `T = F−1` (drop last frame) |
| rot6d re-encoding | re-express rotations into AnyTop's rest basis via per-joint Kabsch on world positions so FK(rot) reproduces RIC (**direct gather of native rotations is BROKEN** — up to ~89° bone-basis error); pin the single-child twist DOF deterministically (or carry real twist consistently) |
| root convention | reconstruct root facing-6D + repack root XZ velocity into `ch9`/`ch11`; integrate any angular-Y-velocity source to a yaw |
| contact | select the source's foot joints; produce/threshold `ch12` on them, `0` elsewhere |
| helper/control-joint pruning | drop non-anatomical joints (IK/twist/control) before conversion |
| captions | emit `motion_texts_by_file.json` keyed by `motion_id` |

---

## 8. The new-source **contract** (checklist an adapter must satisfy)

1. Emit `motions/*.npy` `[T,J,13]` with the exact channel map (§1) and root special-casing. The root
   must be **joint 0** in original order (`parents[0]==-1`) — the root special-casing and normalization
   both key on index 0 (Gate C / the harness enforce this).
2. Emit a `cond` record: `parents` (single-root `-1` at index 0, connected), `offsets [J,3]`, `joints_names [J]`,
   `tpos_first_frame [J,13]`, and **freshly recomputed `mean`/`std [J,13]` over the train split**
   (never reuse the source's own flat Mean/Std).
3. Y-up / +Z-facing (§7).
4. Rescale so the **mean of the reference-bone subset** = `HML_AVG_BONELEN` (the reference
   `motion_process.scale`; the subset is `scale_joint_indices`, per-source). Declare that subset as
   `Topology.scale_ref_joint_idx` so Gate E can verify the scale **exactly**; if you leave it `None`,
   Gate E can only enforce a **gross** band (mean-over-all-bones ~0.3×–3×), because the reference
   subset is not otherwise recoverable from `cond` (across the existing 382 object_types, mean-all-bone
   spans ~0.39×–1.42× exactly for this reason).
5. 20 fps, `T = F−1`.
6. `ch3:9` in **per-parent** convention such that FK(rot) reproduces RIC (§5), twist DOF pinned.
7. `ch12` contact on chosen foot joints, `0` elsewhere.
8. Root facing-6D (`ch3:9`, joint 0) + root XZ velocity repacked into `ch9`/`ch11`.
9. Captions JSON + train/val splits.
10. **Pass Gates A/B/C/E** (the driver, called as `build(adapter, out_root, max_joints=…)`, raises
    `AcceptanceError` on any red hard gate) **and** the mandatory manual Gate D (visual QA) before training.

### Verification gates (visual QA is mandatory and outranks metrics)

- **Gate A — per-clip integrity (every clip).** Shape `[T,J,13]`; finite; `T=F−1` (hard when
  `source_frames` is supplied) and `T>0`; abs-value sanity; contact binary `∈{0,1}` and `0` off the
  declared `foot_joint_idx`; rot6d rows non-degenerate; and the **multi-child rot6d duplication**
  invariant (all child slots of a parent carry the *same* rot6d, else FK/skinning decoders diverge).
- **Gate B — RIC-recovery equivalence (independent).** The source's own official world recovery,
  supplied as `Clip.source_world [T,J,3]`, vs `recover_from_bvh_ric_np(ch0:3)` must match to the gate
  tolerance (`mean_l2 ≤ 1e-4`; HumanML3D achieved ~0). Run on **every** clip that carries
  `source_world`. If the source has no official recovery, Gate B is **waived** (explicitly flagged in
  the report), not silently passed. Validates the RIC packing.
- **Gate C — FK-route self-consistency (THE invariant).** For **every clip** (`build_source` runs it
  per-clip; the report keeps one representative per type), `recover_from_bvh_rot_np` (FK) vs the RIC
  path agree to `L2 ≈ 0` (`< 1e-4` clean; ≤0.5% bbox for `gate_c_mode="inferred_twist"`). The gate
  BFS-reorders internally so it is correct for any input joint order. Validates `offsets` + the
  per-parent rot6d convention. The double-root footgun is invisible on idle clips, so a
  **large-root-rotation** clip must be present per topology (§5); the tolerance regime is declared per
  topology, never inferred from the object_type name.
- **Gate D — visual QA (mandatory, manual).** Side-by-side GT-vs-converted **GIFs/videos** (skeleton
  and, for a skinnable rig, the mesh), rendered by hand with the repo's render tools. Visual
  correctness **outranks** any metric — a leg-flip / curl passes every scalar check. Render from a
  clear side view; confirm all limbs plant and no joint inverts.
- **Gate E — layout contract (structural).** `cond.npy` loads & is non-empty; every `object_type` has
  the required keys with finite `mean/std`, a single-root `parents`, and a sane bone-length scale;
  every motion maps to a known topology; splits present, non-empty, and train/val **disjoint**. This is
  the on-disk contract only — the **full** loader ingest (instantiating `AnyTopDataset`, deriving the
  graph fields, plus per-channel + temporal-2nd-difference distribution matching against a known-good
  slice) runs in the **training repo**, not in this harness.

---

## 9. How the current (Planet Zoo) data was obtained

End-to-end, so a new game/mocap source can mirror the stages. Full detail in
[`docs/PLANETZOO_ANYTOP_PIPELINE.md`](PLANETZOO_ANYTOP_PIPELINE.md),
[`docs/ANIMO4D_ANYTOP_PIPELINE.md`](ANIMO4D_ANYTOP_PIPELINE.md),
[`docs/ANIMO4D_ANYTOP_DATA_LINEAGE.md`](ANIMO4D_ANYTOP_DATA_LINEAGE.md).

| stage | tool | in → out |
|---|---|---|
| 0. extract assets | cobra-tools (external) | Steam install → `01_ovl_extracted/<Object>.ovl` (`.ms2` mesh + `.manis` anims) |
| 1. BVH export (in Blender) | `tools/planetzoo/planetzoo_fulltopo_bvh_export.py` (+ `*_batch_/*_parallel_` wrappers, one fresh Blender per object) | `.ms2`/`.manis` → full-topology `raw_bvhs/*.bvh` @20fps + `*__tpos.bvh` + `export_manifest.jsonl`; a wrapper ROOT is stripped so AnyTop sees one single-root tree |
| 2. AnyTop 13ch convert | `tools/planetzoo/planetzoo_parallel_anytop_process.py` → `utils.process_new_skeleton` → `motion_process.py` | BVH → `motions/*.npy [T,J,13]` + per-object `cond.npy`. Does: prune `srb`/twist helpers → yaw to +Z → roll to Y-up → origin/ground/scale → **rot6d encode (per-parent)** + RIC + local-vel + contact |
| 3. align to text | `tools/planetzoo/build_animo4d_anytop_manifest.py` | text index ↔ raw stem ↔ processed npy → matched/missing status |
| 4. attach captions | `build_animo4d_anytop_text_manifest.py` (official text) / `build_planetzoo_text_manifest.py` (generic) | per-npy captions; **missing text kept as empty string, motions never dropped** |
| 5. pack | `tools/planetzoo/pack_planetzoo_anytop_dataset.py` | per-object folders → one pooled layout + merged `cond.npy`; **hardlink = lossless** |
| 6. repair/audit | `tools/planetzoo/repair_bad_motion_values.py` | quarantine out-of-range npy + recompute `mean/std` |

`expand_minipack_motion_to_full_rig.py` is a **skinning/viz** utility (maps a reduced minipack motion
to the full rig by joint name, broadcasts each parent's rot6d to its child slots, freezes omitted/leaf
joints at rest) — **not** a valid full-topology training target.

---

## 10. Pitfalls already hit (carry into every new source)

- **Joint-order mismatch → fake leg-flip.** Model output is NEW order; skinning + `cond.npy` are
  ORIGINAL order. Convert with `argsort(new_to_old_perm)` (§3). Diagnose by re-running Gate C in the
  motion's own order — GT is clean there.
- **Double-root-rotation.** Never re-add `apply_root_cancel`; test on turn clips, not idle (§5).
- **Direct rotation gather is broken across sources.** Re-encode via Kabsch into AnyTop's rest basis (§7).
- **Inferring a high-DOF target from low-DOF input** (e.g. axial twist from positions) yields an
  unconstrained DOF that is a fine gauge but an **unlearnable target** unless pinned to a canonical
  value at encoding — fix it in the data, not the loss.
- **Water/behaviour clips look wrong on flat ground.** For a land skinning **demo**, pick
  locomotion clips (walk/run/trot); a `walktoswim`/`drink` clip genuinely hunches the body (GT does
  too) — that is the motion, not a bug.
- **Metrics miss geometry.** R-precision/FID/freeze-detector can all pass while a limb inverts. Gate D
  (visual) is mandatory and authoritative.

---

*Companion: the agent-runnable data-prep harness — [`tools/ir_harness/README.md`](../tools/ir_harness/README.md).*
