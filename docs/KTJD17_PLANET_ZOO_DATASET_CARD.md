---
language:
- en
tags:
- motion
- animal-motion
- skeleton
- planet-zoo
- any-topology
license: other
---

# KTJD-17 Planet Zoo Fresh

`KTJD17-PlanetZoo-Fresh` is a 311-rig, 26,865-clip animal motion corpus in the
KTJD-17 `[T, J, 17]` representation.  Every clip originates from a fresh,
provenance-tracked, IK-disabled export of a declared Planet Zoo MANIS action.
It is the KTJD-17 successor to the older AnyTop-13 processing layout.

## Contents

```text
data/
  motions/<clip_id>.npz       # motion[T,J,17] float32, heading_valid[T] bool
  skeletons/<rig_id>.npz      # joint names/descriptions, parents, rest transforms
  stats/<rig_id>.npz          # mean/std/count/min/max/supervise_mask
  manifests/clips.jsonl       # source provenance and official AniMo4D captions
  metadata/joint_descriptions.json
  reports/dataset_validation.json
  reports/raw_bvh_rotation_qc.json
  generation.json
protocol/
  20260822_ktjd17_data_representation_protocol.md
  20260822_pzh312_joint_names_descriptions.json
tools/                         # encoder, validator, raw-BVH QC, visualizer
visual_checks/                 # q-position vs rest-delta-6D FK examples
```

The archive does not redistribute raw BVH, MS2, MANIS, texture, mesh, or skin
weight files from Planet Zoo.  Those files remain local to the authorized game
asset extraction environment.

## Representation

The coordinate system is right handed, `+Y` up, with `+Z` viewer-facing.
Heading is measured from `+Z`; the target rate is 30 fps.  For each frame and
selected joint, channels are:

| Channels | Value |
| --- | --- |
| `0:3` | q-position: world XZ minus root horizontal trajectory, absolute Y |
| `3:9` | global rest-relative 6D rotation |
| `9:12` | world forward velocity in units per second |
| `12` | contact flag |
| `13:15` | root-only horizontal trajectory |
| `15:17` | root-only heading `[cos, sin]`, zero if invalid |

The per-rig skeleton file is necessary for FK.  `q_position` decoded with the
root trajectory and positions decoded from `rest_delta_6d` FK agree to float
precision; the included validation reports a maximum discrepancy of
`3.77e-6` across all clips.

## Captions

`manifests/clips.jsonl` stores the official AniMo4D caption selected by exact
fresh raw-BVH stem.  All 26,865 clips have a matched official text annotation;
no captions were generated or semantically rewritten for this release.

## Quality notes

The full structural check has no missing/orphan files, non-finite values,
invalid heading entries, or FK reconstruction failures.  The raw-BVH report
is intentionally included instead of applying an undocumented smoothing or
repair pass: 1,146 clips contain an adjacent local rotation step above 90
degrees in any retained joint, and 90 contain one in a proximal/structural
joint.  Many are axial rolls at end joints, but downstream work that requires
the most conservative subset can use the included QC report to exclude them.

## Reproduction

The exact pipeline and complete commands are documented in
[`docs/KTJD17_PLANET_ZOO_FRESH_V2.md`](https://github.com/CHDTevior/planetzoo-anytop-pipeline/blob/main/docs/KTJD17_PLANET_ZOO_FRESH_V2.md).
The scripts bundled in `tools/` are copied from that repository revision.
