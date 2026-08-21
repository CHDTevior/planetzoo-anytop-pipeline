# AniMo4D Position-Only Continuity Probe

This probe reproduces the public AniMo4D extraction route for one Planet Zoo
action. It is useful when a full-topology raw BVH shows a questionable limb
rotation and we need to determine whether the same event survives in AniMo4D's
original position-only representation.

## What It Reproduces

`probe_animo4d_official_position_export.py` matches the public
`AniMo_blender_ovl2json.py` behavior:

- imports the MS2 and MANIS in Blender through Cobra Tools;
- leaves the default `disable_ik=False` setting in place;
- reads `pose_bone.head` for every action frame;
- emits `official_keypoints.json` and the same fixed 30-joint `[F, 30, 3]`
  order used by AniMo4D's JSON-to-NPY conversion.

`probe_animo4d_position_preprocess.py` ports the public notebook's
position-to-feature algorithm and emits the flat 359-channel feature plus its
RIC-recovered positions. The per-species/sex template pickle is not released
with the source repository, so this probe uses the clip's first-frame offsets
as its target template. That removes only the category-wide scale change; it
does not remove or manufacture a temporal discontinuity.

## Run

```powershell
H:\blender4_5\blender.exe -b --python tools\planetzoo\probe_animo4d_official_position_export.py -- `
  --cobra-tools H:\path\to\cobra-tools `
  --ms2-path H:\assets\Caracal_Male.ovl\caracal_male_.ms2 `
  --manis-path H:\assets\Caracal_Male.ovl\animationmotionextractedlocomotion.manisetc936aba9.manis `
  --action caracal_male@walkbase `
  --output-dir H:\probe\Caracal_Male_walkbase

H:\path\to\python.exe tools\planetzoo\probe_animo4d_position_preprocess.py `
  --input-positions H:\probe\Caracal_Male_walkbase\official_positions_30j.npy `
  --humanml-root H:\path\to\HumanML3D `
  --output-dir H:\probe\Caracal_Male_walkbase
```

Read `official_position_probe.json` and
`animo4d_notebook_equivalent_report.json`. A persistent maximum transition in
both reports means the event is present before AnyTop conversion and survives
AniMo4D's public position-to-feature path.

## Pre-BVH Jitter Policy

Append `--disable-ik` to the first command for a diagnostic A/B export. It
keeps the decoded MANIS bone transforms but disables only the Blender IK
constraints created by Cobra Tools. This is not part of the historical
AniMo4D route. `--mute-all-constraints` is strictly diagnostic: it establishes
whether a discontinuity is already present in the decoded F-curves.

The full-topology raw-BVH exporter also accepts `--disable-ik` and records
`ik_disabled_during_export` in each motion manifest entry. Export IK-on and
IK-off candidates to separate roots, then retain only a temporally clean,
visually validated candidate.

If an outlier survives both modes, it is already present in Cobra's decoded
compressed-MANIS F-curves. Do not use interpolation as a substitute for source
motion. Mark the clip decoder-risk, retain the asset paths and importer log,
and exclude it until the decoder itself can be corrected.
