# AniMo4D-AnyTop Skinning Notes

This note describes how to attach a processed AniMo4D-AnyTop motion to its
original Planet Zoo mesh. It is intentionally separate from the AnyTop model
format: skinning needs game-derived assets that are not distributed with the
processed dataset.

## Required Assets Per Sample

For an existing AniMo4D-AnyTop sample, keep these files together:

- `motions/<sample>.npy`: the AnyTop `(N, J, 13)` motion.
- `cond.npy`: object skeleton condition, including `parents`, `offsets`, and
  `joints_names`.
- original action BVH: the matching raw BVH under
  `00_raw_bvh_target/<object>_ovl/raw_bvhs/`.
- original T-pose BVH: `<animal>__tpos.bvh` in the same raw folder.
- the matching `.ms2` mesh and `.manis` group under `01_ovl_extracted`.

The exact correspondence comes from the raw export manifest and the processed
alignment manifest. Do not use an MS2/BVH pair from an older extraction pass
with the current `02_anytop_layout`; skeleton export details can differ.

## Saved Forward Conversion

The current layout was created by these saved scripts:

1. `tools/planetzoo/planetzoo_fulltopo_bvh_export.py`
   imports MS2 plus MANIS with Cobra Tools and exports raw BVHs and a T-pose.
2. `tools/planetzoo/planetzoo_parallel_anytop_process.py`
   calls `utils/process_new_skeleton.py` for every object.
3. `data_loaders/truebones/truebones_utils/motion_process.py`
   applies the Planet Zoo AnyTop conversion: action-specific initial yaw,
   `Z=-90` global roll, root-origin normalization, scale/ground normalization,
   rest-basis alignment, position/6D/velocity/contact extraction.
4. `tools/planetzoo/pack_planetzoo_anytop_dataset.py`
   pools per-object outputs into `02_anytop_layout` without modifying motion
   values.

See `docs/ANIMO4D_ANYTOP_PIPELINE.md` for the full extraction commands.

## Verified NPY To Raw-BVH Inverse

`tools/planetzoo/build_planetzoo_anytop_npy_skinning_poc.py` reverses the
deterministic AnyTop preprocessing and writes an inspectable raw BVH before it
attempts a Blender preview.

The inverse uses the T-pose only to recover the stored preprocessing parameters.
It must use the matching raw action BVH as the local-basis template because the
forward converter computes initial yaw per action. The initial absolute root
translation is intentionally discarded by AnyTop; the tool anchors the recovered
trajectory at the raw template's first root position for existing samples.

The Aardvark Female `walkbase` validation used:

- motion: `PZ_Aardvark_Female_..._walkbase_39.npy`
- raw template: `aardvark_female__...__aardvark_female_walkbase.bvh`
- T-pose: `aardvark_female__tpos.bvh`

Its three rotation inverse stages agree with the saved forward converter to a
mean angular error of `7.23e-05` radians. The reconstructed raw BVH matches
the native MANIS action with mean joint-position error `0.01038` and P95
`0.04787`, comparable to the original BVH export/import error.

## Mesh-Driving Route

`tools/planetzoo/build_planetzoo_anytop_npy_skinning_poc.py` is the working
NPY-to-Mesh route. It first writes a reconstructed raw BVH, then imports the
matching MS2 and MANIS and evaluates this rest-relative bridge on every frame:

```text
desired_global = source_pose_global @ inverse(source_rest_global) @ target_rest_global
target_local_basis = inverse(rest_local) @ inverse(parent_desired_global) @ desired_global
```

The target hierarchy is evaluated parent-first. `srb` is kept at its MS2 rest
pose because AnyTop deliberately prunes it and it has no mesh vertex group.
This is important: directly assigning `PoseBone.matrix` or baking those values
reapplies rotations through the reparented exporter helper and causes the
large, incorrect double rotations seen in earlier attempts.

The script needs `--manis-path` to initialise the native rig, but clears its
action and mutes its constraints before applying the decoded BVH. It preserves
the reconstructed raw BVH, a Blender scene, MP4, JSON report, and optional
first/middle/last mesh frames. The saved Blender scene contains the assets and
frame-one pose; rerun the tool for procedural playback or animation rendering.

MP4 previews use a mesh-centred, elevated three-quarter camera. The camera and
its mesh-centre focus both follow the anatomical root, avoiding the foot-level
upward view that a raw root-joint target produces for jumping or low-bodied
animals.

For the Aardvark Female `walkbase` audit, raw BVH through this bridge matched
the native MS2+MANIS mesh at per-mesh mean vertex error around `1e-5` and
maximum error below `1.2e-4`. The same bridge was then applied to the NPY
decoded BVH and visually matched the decoded motion at frames 1, 15, and 30.

MANIS-only controls and IK influences are not encoded in the AnyTop tensor, so
direct native MS2+MANIS playback remains the ground truth for an existing
original action. The bridge is the valid mesh-driving route for the bone
transforms that AnyTop can represent, including model-generated NPY motions.

### Cleaned Minipack Inputs

The 311-species `anytop13-animal-minipack` stores a reduced, topology-specific
body skeleton. Before passing a generated `[T, J_min, 13]` tensor to the
full-topology BVH / MS2 bridge, run
`tools/planetzoo/expand_minipack_motion_to_full_rig.py`. It restores the
object's full rest tensor, maps retained joints by name, and broadcasts the
per-parent rot6d token to every original child slot. The reduced skeleton must
be an induced subgraph of the full rig; the script checks this explicitly.
The tool and the mesh renderer accept either the development `cond.npy` or the
per-object `full_skeleton.json` included with the skinning asset package.

This preserves the body animation represented by the minipack. Omitted face,
ear, tongue and other leaf/detail joints intentionally stay at rest and follow
their animated ancestors. Their independent motion is not recoverable from a
reduced `[T, J_min, 13]` output.

## Validation Tools

- `tools/planetzoo/validate_anytop_npy_inverse.py`: compares every inverse
  rotation stage to the stored forward conversion.
- `tools/planetzoo/probe_anytop_native_joint_alignment.py`: compares a BVH to
  its original MANIS action by joint positions.
- `tools/planetzoo/render_planetzoo_native_action.py`: renders the original
  weighted MS2 plus MANIS action as an importer/control check.
- `tools/planetzoo/evaluate_planetzoo_retarget.py`: evaluates candidate
  Blender retarget formulas against a native action before using one for a
  generated motion.
- `tools/planetzoo/drive_ms2_mesh_from_raw_bvh_runtime.py`: validates the
  same procedural MS2 bridge on a raw BVH before using it for decoded NPY.
