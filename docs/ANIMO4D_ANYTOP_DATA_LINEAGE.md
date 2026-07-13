# AniMo4D to AnyTop Data Lineage

This note records the exact path from a Planet Zoo game action to the latest
`AniMo4D_Anytop/02_anytop_layout` dataset, plus the validated bridge from an
AnyTop NPY back to the original Planet Zoo MS2 mesh.

## Final Dataset

`H:/AniMo4D_work/AniMo4D_Anytop/02_anytop_layout` contains:

- 77,894 motions and 77,894 processed BVHs;
- 311 object topologies;
- 19 to 299 feature frames per motion;
- 143 to 344 joints per topology;
- `motions/<sample>.npy` with shape `(T, J, 13)`;
- `cond.npy` with one condition record per object;
- official AniMo4D text rows for all retained motions.

The final integrity audit is
`H:/AniMo4D_work/AniMo4D_Anytop/02_anytop_layout/final_integrity_audit.json`.
It reports no non-finite motion values, no retained motion above absolute value
11.6253, and refreshed normalization statistics for all 311 objects.

## Conversion Stages

### 1. Planet Zoo assets to raw BVH

Inputs for an object are a weighted `.ms2` mesh plus one or more `.manis`
animation groups from `01_ovl_extracted`.

`tools/planetzoo/planetzoo_fulltopo_bvh_export.py` does the following with
Cobra Tools in Blender:

1. Imports the MS2, which creates the game armature and mesh.
2. Imports each MANIS group, which creates Blender actions such as
   `aardvark_female@walkbase`.
3. Exports every action as a full-topology BVH with native rotation channels.
4. Exports a static rest BVH named `<animal>__tpos.bvh`.
5. Removes Blender's zero-channel wrapper root so AnyTop's BVH loader has one
   valid root hierarchy.
6. Writes `export_manifest.jsonl`, which is the authoritative link from an
   MS2, MANIS action, and raw BVH filename.

For the audit sample, the exported raw action is:

`00_raw_bvh_target/Aardvark_Female_ovl/raw_bvhs/aardvark_female__animationmotionextractedlocomotion_maniset787b80e__aardvark_female_walkbase.bvh`

The copy in `00_raw_bvh_target` has the same SHA256 as the corresponding
full export under `05_fulltopo_raw_bvh_full`.

### 2. Raw BVH to per-object AnyTop representation

`tools/planetzoo/planetzoo_parallel_anytop_process.py` invokes
`utils.process_new_skeleton.py`, which invokes
`data_loaders/truebones/truebones_utils/motion_process.py` once per object.

The rest BVH supplies parameters shared by every action of that object:

1. Remove `srb` helper joints and their descendants. The rest of the topology
   is retained.
2. Use hips-to-chest as a two-joint body centerline to compute a yaw that faces
   the initial animal direction toward canonical `+Z`.
3. Apply the Planet Zoo global `Z=-90` degree roll. This establishes AnyTop's
   `Y up` convention for these game assets.
4. Subtract the T-pose initial root `XZ` location.
5. Scale using the mean length of selected non-helper torso and limb bones so
   that the reference length is `HML_AVG_BONELEN = 0.2092142857142857`.
6. Shift the skeleton vertically so the selected feet/toes/hooves touch
   `Y=0` in the shared rest-derived ground plane.
7. Compute canonical offsets from the processed T-pose and rotate the
   non-root offsets into a rest basis. The Planet Zoo selection tests
   `Z=-90` and `Z=+90` and retains the orientation where core joints are above
   foot joints.

Every action then repeats its own initial-yaw, global-roll, root-origin,
scale, and ground transforms using the rest-derived origin, scale, and ground
parameters. Its rotations are expressed relative to the processed T-pose and
then in the aligned rest basis.

The per-object output contains:

- `motions/*.npy`: AnyTop features;
- `bvhs/*.bvh`: the processed full-frame animation, before the final velocity
  frame is dropped;
- `cond.npy`: hierarchy, rest offsets, joint names, topology relations,
  T-pose feature, and per-object mean/std.

### 3. The `(T, J, 13)` motion tensor

For an input animation with `F` BVH frames, the tensor has `T=F-1` frames,
because velocity and foot contact use a frame difference.

Each joint has 13 channels:

- `0:3`: root-invariant joint position. The root has `[0, root_height, 0]`;
  other joints are in the per-frame root-facing coordinate system.
- `3:9`: 6D rotation. Slot zero stores the per-frame facing rotation. For
  each child joint `j`, its token stores the local rotation of `parent(j)`;
  this parent-indexed convention is the original AnyTop representation.
- `9:12`: frame-to-frame global-joint velocity rotated into the root-facing
  coordinate system. Root X/Z velocity reconstructs accumulated root travel.
- `12`: binary foot-contact indicator on selected feet/toes/hooves, zero for
  all other joints.

`cond.npy` contains these keys for every object: `object_type`, `parents`,
`offsets`, `joints_names`, `tpos_first_frame`, `kinematic_chains`,
`joint_relations`, `joints_graph_dist`, `mean`, and `std`.

### 4. Pooled final layout and value audit

`tools/planetzoo/pack_planetzoo_anytop_dataset.py` materializes the per-object
files in `02_anytop_layout`. The current run used hard links, so packing did
not alter the bytes of the motion or processed-BVH files. The matching raw and
text manifests identify every retained official AniMo4D action.

`tools/planetzoo/repair_bad_motion_values.py` subsequently scanned the pooled
layout with threshold 22.53, would quarantine a failing NPY and its matching
processed BVH, and recomputes `cond.mean/std` from retained files. The latest
run found zero bad motion files; it only refreshed normalization fields.

## Audited Aardvark Walkbase Example

Sample:

`PZ_Aardvark_Female_aardvark_female__animationmotionextractedlocomotion_maniset787b80e__aardvark_female_walkbase_39`

| Stage | Frames / shape | Root trajectory |
| --- | --- | --- |
| Raw game BVH | 31 frames, 226 joints before helper pruning | `X: -0.230 -> 0.621` |
| Processed AnyTop BVH | 31 frames, 224 joints | `Z: 0.425 -> 1.950` |
| Final NPY | `(30, 224, 13)` | canonical root travel is reconstructed from channels `9:12` |
| NPY decoded raw BVH | 30 frames, 224 joints | `X: -0.230 -> 0.593` |

The shortened decoded trajectory is expected because the last raw frame has
no feature row. Its three inverse rotation stages agree with the forward
converter to mean angular error `7.23e-05` radians. The decoded BVH is thus a
valid motion-data inverse, not a guessed retarget.

Visual comparison artifacts, with no ground plane or game mesh, are:

- `H:/AniMo4D_work/AniMo4D_Anytop/skinning_poc_20260713/aardvark_walkbase_raw_vs_decoded_skeleton_audit.blend`
- `H:/AniMo4D_work/AniMo4D_Anytop/skinning_poc_20260713/aardvark_walkbase_raw_vs_decoded_skeleton_audit.mp4`

Orange is the raw MANIS-exported BVH and cyan is the NPY-decoded raw BVH.
Both use the same fixed camera and their own XYZ markers.

## Game Mesh Skinning

The final decoded raw BVH can drive its original weighted MS2 mesh. The valid
route is implemented in
`tools/planetzoo/build_planetzoo_anytop_npy_skinning_poc.py`:

1. Import the original MS2 and matching MANIS to initialise the native rig.
2. Clear the MANIS action and mute constraints for the decoded-BVH render.
3. Import the reconstructed raw BVH with `forward=Y`, `up=Z`.
4. For every shared non-`srb` bone, map the source global rest delta onto the
   target's MS2 rest matrix.
5. Convert the desired global matrices into `matrix_basis` in target
   parent-first order, then render through a frame-change handler.

The per-frame bridge is necessary. A naive direct `PoseBone.matrix` assignment
or F-curve bake compounds rotations around the exporter helper hierarchy and
distorts the mesh. The bridge was tested against a native Aardvark MS2+MANIS
action with per-mesh maximum vertex error below `1.2e-4`; the NPY-decoded
walkbase then produced the corresponding animated mesh at frames 1, 15, and
30.

AnyTop does not encode MANIS-specific IK/control channels. Therefore native
MS2+MANIS is still the ground truth for a retained game action, while this
bridge is the correct way to skin the bone transforms available from AnyTop
and generated NPY motions.
