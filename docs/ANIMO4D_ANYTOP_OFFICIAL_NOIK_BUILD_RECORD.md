# AniMo4D AnyTop Official No-IK Build Record

Build completed on 2026-08-23.

## Delivered Corpus

- Release root: `H:\AniMo4D_work\releases\AniMo4D_AnyTop_Official_NoIK_v1`
- Encoded corpus: `data/`
- Raw no-IK BVHs retained: `H:\a4full_raw`
- Source code revision for the encoder/exporter: `de05f4e`

| Item | Result |
| --- | ---: |
| Planet Zoo rigs | 311 |
| Official captions in local source text directory | 78,149 |
| Official captions with locally available source actions | 77,894 |
| Encoded KTJD-17 clips | 77,894 |
| Raw motion BVHs | 77,894 |
| Raw rest BVHs | 311 |
| Per-rig skeleton files | 311 |
| Per-rig normalization files | 311 |
| Missing / duplicate official IDs in output | 0 / 0 |
| Non-finite values | 0 |
| FK reconstruction maximum position error | `3.764338e-06` |
| Physical-QC severe / error | 0 / 0 |
| Physical-QC clean / review | 77,794 / 100 |

The 255 official captions without an available local source action are preserved in `manifests/official_actions_unavailable.jsonl`; they were not represented by dummy motions.

## Joint Mapping And Selection

The encoded corpus uses the supplied canonical joint-spec file:

```text
H:\AniMo4D_work\releases\KTJD17_PlanetZoo_Fresh_v1\protocol\20260822_pzh312_joint_names_descriptions.json
```

SHA-256: `E9BF34078B6DE6E5769B399521C5D24E8ECA84ED5C28B503E875F526FF3F1E9C`.

For each `PZ_*` rig, its `rigs[rig_id].joints` list is authoritative. It fixes:

- joint name and output order;
- parent index in the selected topology;
- semantic joint description used by AnyTop;
- `rotation_source_kind` metadata.

The encoder requires every selected joint to exist in the no-IK BVH and verifies the complete selected parent relation before it writes any motion. Each output skeleton stores the ordered `joint_names`, `parents`, `offset_parent_local`, rest rotations, contact joints, and a `joint_order_sha256` checksum.

The raw BVHs remain full-topology source assets. The training `data/` is intentionally the topology subset specified by the supplied canonical spec, not an ad hoc 30-joint AniMo filter and not a name-heuristic reduction. The resulting selected topology is 34 to 102 joints per rig. The two uniform zero-length import helpers `srb` and `srb_end_site` are removed before this strict mapping check.

## Filtering Policy

1. Keep only the 77,894 official text IDs for which a source MANIS/Blender action is available locally.
2. Re-export every retained action with IK disabled. No smoothing or rotation repair is applied.
3. Scan all clips using proximal quaternion steps plus child-subtree spatial displacement.
4. Do not remove the 100 review clips: they are high-dynamic semantic transitions such as pounces, climbs, turns, or births and no clip met the severe structural-flip condition.
5. Refuse any clip with a source-action mismatch, an IK-enabled export marker, a skeleton mismatch, a non-finite value, a missing file, or a 6D-FK consistency failure.

The exact checks and reports are included in the release `manifests/`, `qc/`, `validation_ktjd17.json`, and `validation_text_pairing.json`.
