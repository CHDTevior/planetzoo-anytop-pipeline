# AniMo4D AnyTop Official No-IK Dataset

## Scope

This release contains the complete locally reconstructable AniMo4D official-caption Planet Zoo corpus in the AnyTop-compatible `KTJD-17` representation.

- 311 Planet Zoo rigs
- 77,894 motion clips
- 77,894 exact official AniMo4D caption records
- 311 rest skeletons and per-rig normalization statistics
- 30 FPS target rate

The 78,149 original official text entries contain 255 entries whose source action is not present in the supplied local extracted game assets. Those entries are recorded in `official_actions_unavailable.jsonl`; they are not fabricated as motions.

## Source And Processing

1. Official AniMo4D text files were indexed and mapped to the actual MANIS source file and Blender action for every available official ID.
2. Planet Zoo OVL/MS2/MANIS assets were imported through Cobra with inverse-kinematics constraints disabled before baking/export. No temporal smoothing or rotation repair was applied.
3. Each action was exported as a full-topology BVH together with a per-rig rest/T-pose BVH and skeleton metadata.
4. A physical jump scan evaluated proximal-joint quaternion steps together with the corresponding child-subtree world-position movement. Results: 77,794 `clean`, 100 `review`, 0 `severe`, 0 `error`. Review clips are fast semantic transitions such as pounces, climbs, turns, and births; none show the prior IK-driven limb-flip signature.
5. BVHs were converted to AnyTop-compatible `KTJD-17`; each clip was checked by reconstructing positions via local 6D forward kinematics.

The dataset uses a right-handed coordinate system: `+Y` up, `+Z` viewer-facing, with heading measured from `+Z`. The imported Planet Zoo rest basis is normalized by the pipeline's `roll_z = -90 degrees` convention.

## Motion Layout

Every compressed motion file stores `motion` with shape `[T, J, 17]`:

| Channels | Meaning |
| --- | --- |
| `0:3` | root-relative joint position (`q_position`) |
| `3:9` | local rest-delta rotation in 6D representation |
| `9:12` | local joint velocity |
| `12` | foot-contact flag |
| `13:15` | root horizontal trajectory (`smooth_root_xz`) |
| `15:17` | heading cosine and sine |

`T` is 19 through 299 after velocity/contact derivation. Full topology is retained; this release has 34 to 102 joints depending on the rig. `skeletons/<rig>.json` provides joint names, parent indices, rest offsets, and kinematic chains. The `rest_delta_6d` channels plus rest offsets are sufficient to recover the complete skeletal pose by forward kinematics.

## Layout

```text
data/
  motions/<rig>/<clip_id>.npz
  skeletons/<rig>.json
  stats/<rig>.npz
  manifests/clips.jsonl
  manifests/generation.json
  reports/conversion_errors.json
```

The release root additionally contains:

- `official_action_sources.jsonl`: official caption to verified source action mapping
- `official_actions_unavailable.jsonl`: the 255 text rows without local source assets
- `export_manifest.jsonl` and `export_manifest_summary.json`: raw BVH completeness record
- `noik_physical_qc_all.jsonl`, `noik_physical_qc_flagged.jsonl`, and `noik_physical_qc_summary.json`: no-IK physical QC record
- `validation_ktjd17.json`: final structural and numerical validation
- `validation_text_pairing.json`: exact official caption pairing validation

Raw full-topology BVHs are retained separately under `H:\\a4full_raw` in the local build environment. `export_manifest.jsonl` records the build-local BVH path, file name, and action provenance.

## Reproducibility

The implementation is in the `main` branch of `CHDTevior/planetzoo-anytop-pipeline`, commit `de05f4e`.

Primary commands:

```powershell
python tools/planetzoo/build_official_animo4d_action_manifest.py ...
python tools/planetzoo/run_official_animo4d_noik_export.py ...
python tools/planetzoo/scan_official_noik_bvh_qc.py ...
python tools/planetzoo/build_ktjd17_pz_dataset.py ...
python tools/planetzoo/validate_ktjd17_pz_dataset.py ...
python tools/planetzoo/validate_official_animo4d_text_pairing.py ...
```

Do not re-enable IK or apply generic smoothing while recreating the raw export. That would change the native local rotations and can reintroduce the limb-flip artifact this release avoids.
