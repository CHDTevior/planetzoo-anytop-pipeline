# Planet Zoo Motion-Extracted Offline Recovery

## Purpose

This is an offline recovery path for Planet Zoo MANIS assets whose direct
Cobra decode contains abrupt proximal-limb rotation jumps. It uses only
locally extracted asset files: MANIS clips and the character's MS2 skeleton.
It does not launch or hook the game, capture memory, use Blender IK, smooth
curves, or interpolate frames.

The recovered BVH can then enter the repository's normal AnyTop preprocessing
path.

## Export Provenance Guard

Before applying the paired recovery, regenerate raw BVHs with
`planetzoo_fulltopo_bvh_export.py`. Cobra assigns imported Blender Actions a
fake user and stashes them in NLA tracks. The historical exporter only removed
unused actions, so an Action could survive into the next MANIS import and be
written with the later MANIS file's name.

The exporter now removes all Actions in its isolated Blender process before
every MANIS import, parses the MANIS-declared action names directly, and
exports only Actions whose name is declared by that exact file. Each output
manifest row records `source_action_verified: true` and
`manis_declared_action_count` for an auditable count check.

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

For those actions, use the **direct no-IK decode** mode instead of substituting
an approximate donor. It imports exactly one MS2 and one MANIS file in an
isolated Blender process, permanently disables importer IK, and exports the
declared Action untouched. This is a diagnostic/export path, not a repair.

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

Use `--target-action` to test one or more exact actions rather than processing
every strict pair for an owner:

```powershell
& $py "$repo\tools\planetzoo\run_motionextracted_onspot_reconstruction.py" `
  --pairs-manifest 'H:\AniMo4D_work\motionextracted_onspot_pairs.jsonl' `
  --input-root 'H:\AniMo4D_work\01_ovl_extracted' `
  --cobra-tools 'H:\codex_project1\.codex-tmp\cobra-tools-latest-decoder-lab' `
  --blender 'H:\blender4_5\blender.exe' `
  --output-root 'H:\AniMo4D_work\motionextracted_onspot_recovered' `
  --owner 'Grey_Seal_Female' `
  --target-action 'grey_seal_female@walktodrinktroughturnr'
```

Audit legacy QC labels without a strict on-spot pair, then produce a direct
no-IK BVH for each correctly resolved MANIS Action:

```powershell
& $py "$repo\tools\planetzoo\build_legacy_qc_direct_targets.py" `
  --audit 'H:\AniMo4D_work\AniMo4D_Anytop\decoder_trials\legacy_flagged_provenance_audit.json' `
  --pairs-manifest 'H:\AniMo4D_work\motionextracted_onspot_pairs.jsonl' `
  --input-root 'H:\AniMo4D_work\01_ovl_extracted' `
  --output-jsonl 'H:\AniMo4D_work\legacy_qc_direct_targets.jsonl' `
  --output-summary 'H:\AniMo4D_work\legacy_qc_direct_targets_summary.json'

& $py "$repo\tools\planetzoo\run_direct_noik_reconstruction.py" `
  --targets-manifest 'H:\AniMo4D_work\legacy_qc_direct_targets.jsonl' `
  --input-root 'H:\AniMo4D_work\01_ovl_extracted' `
  --cobra-tools 'H:\codex_project1\.codex-tmp\cobra-tools-latest-decoder-lab' `
  --blender 'H:\blender4_5\blender.exe' `
  --output-root 'H:\AniMo4D_work\legacy_qc_direct_noik' `
  --workers 8

& $py "$repo\tools\planetzoo\scan_reconstruction_bvhs.py" `
  --reconstruction-status 'H:\AniMo4D_work\legacy_qc_direct_noik\reconstruction_status.jsonl' `
  --output-dir 'H:\AniMo4D_work\legacy_qc_direct_noik\proximal_qc'
```

`stage_reconstructed_bvhs_for_anytop.py` and
`render_reconstructed_anytop_qc.py` turn any selected QC set into paired
AnyTop RIC-versus-Rot6D-FK GIFs. The generated `index.html` is the review
surface; the red chain is the scanner's trigger bone and descendants.

Audit an existing flagged visual-QC set against the original MANIS declarations
before deciding whether it is an exporter collision or a real decode issue:

```powershell
& $py "$repo\tools\planetzoo\audit_flagged_motion_provenance.py" `
  --cobra-tools 'H:\codex_project1\.codex-tmp\cobra-tools-latest-decoder-lab' `
  --source-root 'H:\AniMo4D_work\01_ovl_extracted' `
  --layout-manifest 'H:\AniMo4D_work\AniMo4D_Anytop\02_anytop_layout\motion_text_manifest.jsonl' `
  --render-summary 'H:\AniMo4D_work\AniMo4D_Anytop\visual_checks\proximal_rotation_qc_100_20260608\flagged_anytop_qc_render_summary.json' `
  --render-summary 'H:\AniMo4D_work\AniMo4D_Anytop\visual_checks\proximal_rotation_qc_borderline_100_20260608\flagged_anytop_qc_render_summary.json' `
  --output 'H:\AniMo4D_work\AniMo4D_Anytop\decoder_trials\legacy_flagged_provenance_audit.json'
```

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

The historical 200-sample proximal-QC set was also audited directly against
the source MANIS data. `123/200` legacy rows named a MANIS file that did not
declare that action; the action was declared by another MANIS in the same
character directory. The 200 rows collapse to 87 real source actions because
the exporter emitted the same action under up to three incorrect source names.
Of those 87 actions, 28 have strict on-spot donors and 59 do not. This means
the provenance guard applies to every fresh raw-BVH export, while paired
recovery intentionally covers only the 28 actions for which the asset contract
is satisfied.

### Full 87-action Follow-up

The complete 87-action follow-up used source-declared MANIS provenance and
`disable_ik=True` throughout:

| Branch | Actions | Clean | Borderline | Candidate | Severe |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strict on-spot local-rotation recovery | 28 | 20 | 7 | 1 | 0 |
| Direct no-IK decode (no strict donor) | 59 | 8 | 44 | 5 | 2 |
| Total | 87 | 28 | 51 | 6 | 2 |

The direct branch's 51 flagged actions were all converted to AnyTop and
rendered. Every paired RIC-versus-Rot6D-FK check remained self-consistent: the
largest position error was `7.0e-9`. Thus these flags are present in the
source offline Cobra/MANIS pose decode, rather than introduced by the AnyTop
13-channel representation. The 2 severe plus 5 candidate clips are the
high-confidence manual-review set; `borderline` is deliberately retained as a
separate, less certain threshold bucket.

## Scope

The full local audit found `26,064` strict dtype-38/dtype-36 pairs among
`79,417` parsed actions (`25,993` locomotion, `7` behaviour, `64` fighting).
It also found `12,116` dtype-38 actions without a clean donor and `2,113`
with ambiguous donors. The latter two groups are deliberately not exported by
the paired-recovery method. A full data rebuild should instead start from the
provenance-guarded raw-BVH export, use paired recovery only where the strict
contract exists, direct no-IK decode otherwise, and run the same QC before
admitting a motion to the final AnyTop layout.
