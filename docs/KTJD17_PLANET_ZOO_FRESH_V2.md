# KTJD-17 Planet Zoo fresh export

This document describes the reproducible source-to-dataset path for the
fresh Planet Zoo corpus built against the 2026-08-22 KTJD-17 protocol.  It is
intentionally separate from the older AnyTop-13 releases: the motion payload
is `[T, J, 17]`, joint selection and descriptions come from the supplied
PZ-H312 specification, and every source action is freshly exported.

## Result

The completed local corpus is rooted at
`H:\AniMo4D_work\KTJD17_PZ_fresh_v2`:

| Item | Value |
| --- | --- |
| Planet Zoo rigs | 311 |
| verified source actions / output clips | 26,865 / 26,865 |
| output frame range | 19--299 |
| retained joints per rig | 34--102 |
| target rate | 30 fps |
| FK reconstruction maximum error | `3.77e-6` |

The `01_raw_bvh_30fps` directory is the locally retained, freshly exported
BVH source.  The publishable processed data is
`02_ktjd17_pz_dataset_v2`; it deliberately does **not** include proprietary
game meshes, MS2 files, MANIS files, or raw BVHs.

## Extraction

1. `write_ktjd17_pz_object_list.py` derives the exact 311 `*.ovl` object
   directories from the joint specification and verifies that each exists in
   the extracted Planet Zoo asset tree.
2. `planetzoo_parallel_bvh_export.py` launches one isolated Blender/Cobra
   export process per object.  Its exporter imports the MS2 armature, imports
   only MANIS actions declared by the package, permanently disables pose
   constraints/IK, writes one T-pose BVH plus one BVH per action, and records
   source action provenance in `export_manifest.jsonl`.
3. The production invocation includes `--disable-ik --fps 30`.  Blender can
   retain native action timing in a BVH header (the verified sources include
   24, 29, 30 and 31 fps); the dataset builder is responsible for the final
   30 fps sample grid.

Example commands, with local paths adjusted to the host:

```powershell
$py = 'H:\codex_project1\.codex-tmp\venvs\cobra\Scripts\python.exe'
$repo = 'H:\codex_project1\.codex-tmp\planetzoo-anytop-pipeline-upload'

& $py "$repo\tools\planetzoo\write_ktjd17_pz_object_list.py" `
  --joint-spec C:\path\to\20260822_pzh312_joint_names_descriptions.json `
  --input-root H:\path\to\extracted\objects `
  --output H:\work\pzh312_objects.txt

& $py "$repo\tools\planetzoo\planetzoo_parallel_bvh_export.py" `
  --blender H:\blender4_5\blender.exe `
  --cobra-tools H:\path\to\cobra-tools `
  --input-root H:\path\to\extracted\objects `
  --output-root H:\work\01_raw_bvh_30fps `
  --objects-file H:\work\pzh312_objects.txt --disable-ik --fps 30 --workers 8
```

## KTJD-17 encoding

`build_ktjd17_pz_dataset.py` refuses an action unless its source manifest
states `source_action_verified=true` and `ik_disabled_during_export=true`.
For every rig it uses the protocol-selected joint order and descriptions from
the supplied JSON, obtains the canonical rest pose from the freshly exported
T-pose BVH, and uses the established Planet Zoo scale/basis normalization.

When a source BVH rate differs from 30 fps, local quaternions are resampled
on the target timeline using shortest-path SLERP and local translations are
linearly interpolated; forward kinematics is then evaluated.  The stored
sequence is one frame shorter because finite-difference velocity/contact
features need a neighbor frame.

Coordinates are right handed: `+Y` is up and `+Z` faces the viewer.  Heading
is measured from `+Z`.  Every compressed motion NPZ contains:

| Channels | Meaning |
| --- | --- |
| `0:3` | all-joint `q_position`: world XZ minus root horizontal trajectory, absolute Y |
| `3:9` | `rest_delta_6d`: global rotation relative to global rest pose, first two columns |
| `9:12` | world-space forward velocity, units/second |
| `12` | binary foot-contact flag |
| `13:15` | root-only horizontal trajectory (`smooth_root_xz`) |
| `15:17` | root-only heading `[cos(theta), sin(theta)]`; zero when invalid |

`skeletons/<rig>.npz` contains selected names/descriptions, parents, local
offsets, and both local/global rest rotations.  `stats/<rig>.npz` contains the
protocol mean/std/count and supervise mask.  The caption copied into each
`manifests/clips.jsonl` record is the matching official AniMo4D annotation;
no new VLM caption is generated.

## Validation and visual QC

Run the structural validation after a build:

```powershell
& $py "$repo\tools\planetzoo\validate_ktjd17_pz_dataset.py" `
  --dataset-root H:\work\02_ktjd17_pz_dataset_v2 --workers 8 `
  --output-report H:\work\02_ktjd17_pz_dataset_v2\reports\dataset_validation.json
```

The completed corpus has no missing/orphan files, non-finite values, shape
errors, or invalid heading rows.  FK reconstructed from `rest_delta_6d` and
the skeleton matches `q_position + smooth_root_xz` with mean error
`2.60e-8` and maximum error `3.77e-6`.

Use `render_ktjd17_compare.py` to render two synchronized skeletons: the
left path reconstructs world positions from q-position, the right path uses
only rest-delta-6D FK.  Both draw the coordinate axes.

```powershell
& $py "$repo\tools\planetzoo\render_ktjd17_compare.py" `
  --dataset-root H:\work\02_ktjd17_pz_dataset_v2 `
  --clip-id saiga_female__animationmotionextractedlocomotion_maniset65e14583__saiga_female_walkbaseturnl `
  --output-dir H:\work\visual_checks
```

`scan_ktjd17_raw_bvh_rotation_qc.py` is a transparent source-quality report,
not a data repair stage.  It measures sign-invariant adjacent-frame local
quaternion angles in the freshly exported IK-disabled BVHs.  On this export,
all 26,865 BVHs load with verified provenance.  The report marks 1,146 clips
with a step above 90 degrees in any retained joint and 90 in a structural
(torso/proximal limb) joint.  Many are axial roll changes at feet/end joints;
the report must accompany downstream users who want to exclude these
conservative candidates rather than silently altering game motion.
