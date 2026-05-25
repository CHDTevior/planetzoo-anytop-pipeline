# Planet Zoo to AnyTop Processing Notes

This document records the current local pipeline for extracting Planet Zoo
animation assets and converting them into the AnyTop processed-motion format.
It is intentionally specific to the working setup used during the Aardvark
Female demo, while keeping the commands easy to adapt for larger batches.

## Local Paths Used

- Planet Zoo install: `G:/Steam/steamapps/common/Planet Zoo`
- Work root: `H:/AniMo4D_work`
- Extracted OVL assets: `H:/AniMo4D_work/01_ovl_extracted`
- Raw full-topology BVH demo: `H:/AniMo4D_work/05_fulltopo_raw_bvh_demo7`
- Processed AnyTop demo:
  `H:/AniMo4D_work/06_anytop_demo/PZ_Aardvark_Female_demo13_rollz_neg90_contact`
- Blender: `H:/blender4_5/blender.exe`
- CobraTools:
  `H:/codex_project1/.codex-tmp/AniMo/data_generation/export_json/cobra-tools`
- Python environment:
  `H:/codex_project1/.codex-tmp/venvs/cobra/Scripts/python.exe`

## 1. Extract Local Game Assets

The first stage is to extract Planet Zoo OVL assets from the local Steam
install into a separate work directory. The current work directory keeps the
game install read-only and stores all intermediate data under `H:/AniMo4D_work`.

Current extracted input directory:

```text
H:/AniMo4D_work/01_ovl_extracted
```

This directory is the input for the BVH export step. Keep it outside git; it
contains game-derived assets and can be very large.

## 2. Export Full-Topology Raw BVH Files

Use Blender plus CobraTools to load the extracted Planet Zoo assets and export
raw BVH files. The export keeps full topology at this stage so the raw BVHs can
be inspected in Blender before AnyTop conversion.

Script:

```text
tools/planetzoo/planetzoo_fulltopo_bvh_export.py
```

Example command for the current demo:

```powershell
H:/blender4_5/blender.exe -b --python tools/planetzoo/planetzoo_fulltopo_bvh_export.py -- `
  --cobra-tools H:/codex_project1/.codex-tmp/AniMo/data_generation/export_json/cobra-tools `
  --input-root H:/AniMo4D_work/01_ovl_extracted `
  --output-root H:/AniMo4D_work/05_fulltopo_raw_bvh_demo7 `
  --objects Aardvark_Female.ovl `
  --max-actions 2 `
  --only-manis-contains locomotion
```

For larger batches, prefer one Blender process per object:

```powershell
H:/codex_project1/.codex-tmp/venvs/cobra/Scripts/python.exe tools/planetzoo/planetzoo_batch_bvh_export.py `
  --blender H:/blender4_5/blender.exe `
  --cobra-tools H:/codex_project1/.codex-tmp/AniMo/data_generation/export_json/cobra-tools `
  --input-root H:/AniMo4D_work/01_ovl_extracted `
  --output-root H:/AniMo4D_work/05_fulltopo_raw_bvh `
  --max-actions 3 `
  --only-manis-contains locomotion `
  --continue-on-error
```

Running a fresh Blender process for each object avoids cross-object importer
state and makes failures resumable. Logs are written under
`H:/AniMo4D_work/05_fulltopo_raw_bvh/logs/`.

Important outputs:

- `raw_bvhs/*.bvh`: raw animation BVHs
- `raw_bvhs/*__tpos.bvh`: static rest/T-pose BVH
- `export_manifest.jsonl`: stable mapping from source game action to raw BVH

For the current demo, the relevant raw BVHs are:

```text
H:/AniMo4D_work/05_fulltopo_raw_bvh_demo7/Aardvark_Female_ovl/raw_bvhs/aardvark_female__tpos.bvh
H:/AniMo4D_work/05_fulltopo_raw_bvh_demo7/Aardvark_Female_ovl/raw_bvhs/aardvark_female__animationmotionextractedlocomotion_maniset787b80e__aardvark_female_runbase.bvh
H:/AniMo4D_work/05_fulltopo_raw_bvh_demo7/Aardvark_Female_ovl/raw_bvhs/aardvark_female__animationmotionextractedlocomotion_maniset787b80e__aardvark_female_runbaseturnl.bvh
```

## 3. Convert BVH to AnyTop Format

The conversion uses AnyTop's official new-skeleton pipeline, with local
Planet-Zoo-specific adjustments in
`data_loaders/truebones/truebones_utils/motion_process.py`.

Example command for the current demo:

```powershell
H:/codex_project1/.codex-tmp/venvs/cobra/Scripts/python.exe -m utils.process_new_skeleton `
  --object_name PZ_Aardvark_Female `
  --bvh_dir H:/AniMo4D_work/05_fulltopo_raw_bvh_demo7/Aardvark_Female_ovl/raw_bvhs `
  --save_dir H:/AniMo4D_work/06_anytop_demo/PZ_Aardvark_Female_demo13_rollz_neg90_contact `
  --face_joints_names def_c_hips_joint def_c_chest_joint `
  --tpos_bvh H:/AniMo4D_work/05_fulltopo_raw_bvh_demo7/Aardvark_Female_ovl/raw_bvhs/aardvark_female__tpos.bvh
```

Batch conversion can be run with:

```powershell
H:/codex_project1/.codex-tmp/venvs/cobra/Scripts/python.exe tools/planetzoo/planetzoo_batch_anytop_process.py `
  --raw-root H:/AniMo4D_work/05_fulltopo_raw_bvh `
  --output-root H:/AniMo4D_work/06_anytop_processed `
  --face-joints-names def_c_hips_joint def_c_chest_joint
```

The batch script writes `batch_process_manifest.jsonl` under the output root and
records each object's raw directory, T-pose BVH, status, and processed clip
count.

Current Planet Zoo adjustments:

- Raw BVH export keeps full topology.
- AnyTop conversion prunes only the Planet Zoo `srb` helper branch from the
  processed dataset because it is a disconnected helper chain.
- Face direction uses a two-joint centerline:
  `def_c_hips_joint -> def_c_chest_joint`.
- The skeleton is yaw-aligned to face AnyTop `+Z`.
- A fixed Planet Zoo global roll of `Z = -90 degrees` is applied after yaw
  alignment. This was verified visually against the AnyTop camera/rendering
  convention.
- Initial root `XZ` translation is moved to the origin.
- The skeleton is scaled against AnyTop's `HML_AVG_BONELEN`, using stable
  Planet Zoo body/limb reference joints rather than tiny helper/twist joints.
- Grounding uses Planet Zoo foot/toe/hoof/ashi joints when available.
- Foot contact uses the official velocity/height contact representation, with
  a Planet-Zoo-specific relative ground-height check.
- T-pose IK is skipped when zero-length helper/twist offsets would make the IK
  result invalid.

## 4. Processed Representation

The processed `motions/*.npy` files match AnyTop's official processed-motion
format:

```text
shape = (frames - 1, joints, 13)
```

The last dimension is:

```text
0:3    local/root-invariant joint position
3:9    local joint rotation in 6D representation
9:12   local joint velocity
12:13  foot contact flag
```

The sequence is one frame shorter than the raw BVH because velocity and foot
contact depend on adjacent frames.

For the current Aardvark Female demo:

```text
runbase      -> (12, 224, 13)
runbaseturnl -> (12, 224, 13)
```

The accompanying `cond.npy` follows AnyTop's expected object condition schema:

```text
tpos_first_frame
joint_relations
joints_graph_dist
object_type
parents
offsets
joints_names
kinematic_chains
mean
std
```

The number of joints may differ per object. For this demo, `J = 224`.

## 5. Text Matching Manifest

Because the filtering rules now differ from AniMo's original release pipeline,
text is matched after processing. Missing captions are kept as empty strings
rather than dropping the motion.

Script:

```text
tools/planetzoo/build_planetzoo_text_manifest.py
```

Example command:

```powershell
H:/codex_project1/.codex-tmp/venvs/cobra/Scripts/python.exe tools/planetzoo/build_planetzoo_text_manifest.py `
  --processed-dir H:/AniMo4D_work/06_anytop_demo/PZ_Aardvark_Female_demo13_rollz_neg90_contact `
  --export-manifest H:/AniMo4D_work/05_fulltopo_raw_bvh_demo7/Aardvark_Female_ovl/export_manifest.jsonl `
  --csv-output H:/AniMo4D_work/06_anytop_demo/PZ_Aardvark_Female_demo13_rollz_neg90_contact/motion_text_manifest.csv
```

Current demo manifest result:

```text
rows = 2
matched_text = 0
missing_text = 2
```

## 6. Dataset Statistics

After conversion, summarize AnyTop-format outputs with:

```powershell
H:/codex_project1/.codex-tmp/venvs/cobra/Scripts/python.exe tools/planetzoo/summarize_anytop_dataset.py `
  --processed-root H:/AniMo4D_work/06_anytop_processed `
  --output-json H:/AniMo4D_work/06_anytop_processed/dataset_summary.json `
  --output-csv H:/AniMo4D_work/06_anytop_processed/dataset_summary_objects.csv
```

The summary reports total objects, total clips, node-count range, clip-length
range, clips-per-object range, exact node-count distribution, exact/binned
length distribution, and per-object rows.

## 7. Visual Sanity Checks

Use the AnyTop recovered-motion renderer as a sanity check. The camera itself is
not part of the dataset and will not affect training or dataset merging.

Current verified GIF outputs:

```text
H:/AniMo4D_work/06_anytop_demo/PZ_Aardvark_Female_demo13_rollz_neg90_contact/anytop_plain_gif/runbase_anytop.gif
H:/AniMo4D_work/06_anytop_demo/PZ_Aardvark_Female_demo13_rollz_neg90_contact/anytop_plain_gif/turn_left_anytop.gif
```

The final verified transform is the one with Planet Zoo global roll
`Z = -90 degrees`.

## 8. What Should Be Versioned

Version in git:

- Planet Zoo export scripts
- AnyTop preprocessing patches
- Pipeline documentation
- Small validation utilities or manifests that do not contain game assets

Do not version in git:

- Raw extracted Planet Zoo game assets
- Raw BVH batches derived from game assets
- Processed `.npy`, `.mp4`, `.gif`, or large dataset outputs
- Local Python environments
