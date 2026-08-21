# Planet Zoo Motion-Extracted Offline Recovery

## Purpose

This is an offline recovery path for Planet Zoo MANIS assets whose direct
Cobra decode contains abrupt proximal-limb rotation jumps. It uses only
locally extracted asset files: MANIS clips and the character's MS2 skeleton.
It does not launch or hook the game, capture memory, use Blender IK, smooth
curves, or interpolate frames.

The recovered BVH can then enter the repository's normal AnyTop preprocessing
path.

## Asset Contract

The recovery is intentionally conservative. A moving clip is accepted only
when the extraction contains one and exactly one paired on-spot donor with:

- the same character OVL;
- the same MANIS category;
- the same canonical action name after removing `onspot` and a trailing
  `turnl` or `turnr` suffix;
- the same frame count;
- a moving `animationmotionextracted*` action with raw dtype `38`;
- a donor `animationnotmotionextracted*` action with raw dtype `36`.

No candidate is inferred for missing or ambiguous donors. Those actions must
remain outside this recovery branch until the unresolved MANIS dtype-38 limb
track semantics are independently decoded.

## Reconstruction

1. Import the character MS2 plus the exact target and donor MANIS files with
   Cobra's importer in `disable_ik=True` mode.
2. Keep the target action's root and trunk channels unchanged. Its
   `LimbTrackData.float_0` and `vec_0` carry the locomotion trajectory.
3. Use each target limb's `LimbTrackData.list_one[0].countb` to walk upward
   through the MS2 parent hierarchy from that limb's leaf bone.
4. Copy donor local quaternion curves only for bones belonging to exactly one
   such limb chain. Shared ancestors, including root/hips/spine/chest, stay
   on the moving target action.
5. Export standard BVH and remove Blender's zero-channel wrapper root so the
   AnyTop BVH reader sees the character skeleton root directly.

This is an asset-authored pairing recovery, not a generic filter for bad
frames.

## Commands

Build the auditable manifest from the locally extracted OVL directories:

```powershell
$py = 'H:\codex_project1\.codex-tmp\venvs\cobra\Scripts\python.exe'
$repo = 'H:\codex_project1\.codex-tmp\planetzoo-anytop-pipeline-upload'
& $py "$repo\tools\planetzoo\build_motionextracted_onspot_pairs.py" `
  --cobra-tools 'H:\codex_project1\.codex-tmp\cobra-tools-latest-decoder-lab' `
  --input-root 'H:\AniMo4D_work\01_ovl_extracted' `
  --output-manifest 'H:\AniMo4D_work\motionextracted_onspot_pairs.jsonl' `
  --output-summary 'H:\AniMo4D_work\motionextracted_onspot_pairs_summary.json'
```

Preview exactly what would be reconstructed for one character. `--dry-run`
is useful before a large export:

```powershell
& $py "$repo\tools\planetzoo\run_motionextracted_onspot_reconstruction.py" `
  --pairs-manifest 'H:\AniMo4D_work\motionextracted_onspot_pairs.jsonl' `
  --input-root 'H:\AniMo4D_work\01_ovl_extracted' `
  --cobra-tools 'H:\codex_project1\.codex-tmp\cobra-tools-latest-decoder-lab' `
  --blender 'H:\blender4_5\blender.exe' `
  --output-root 'H:\AniMo4D_work\motionextracted_onspot_recovered' `
  --owner 'Grey_Seal_Female' --limit 1 --dry-run
```

Remove `--dry-run` to write one reconstructed BVH, report and sampled world
positions per manifest record. Each record runs in its own Blender process;
the JSONL status log provides a resumable audit trail. Add `--skip-existing`
to resume a completed output directory.

## Validation Performed

Two unrelated topologies passed the whole path, including standard BVH export,
BVH re-import and AnyTop conversion:

| Clip | AnyTop shape | max pose-vs-rot6d-FK position error |
| --- | ---: | ---: |
| Grey Seal female `standtowalkturnl` | `76 x 288 x 13` | `3.39e-8` |
| Caracal male `walkbase` | `37 x 273 x 13` | `5.13e-9` |

The one-frame difference is expected: AnyTop features contain velocity and
foot-contact terms, so an input BVH of `N` frames becomes `N - 1` rows.

On the Grey Seal target that originally showed the failure, the largest local
rotation step fell from `131.72 degrees` on a front foot to `42.81 degrees`
after replacing only the independently tracked limb branches. Root, hips,
spine and chest retained the moving action's channels.

## Scope

The full local audit found `26,064` strict dtype-38/dtype-36 pairs among
`79,417` parsed actions (`25,993` locomotion, `7` behaviour, `64` fighting).
It also found `12,116` dtype-38 actions without a clean donor and `2,113`
with ambiguous donors. The latter two groups are deliberately not exported by
this method.
