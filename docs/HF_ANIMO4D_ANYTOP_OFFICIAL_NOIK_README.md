---
pretty_name: AniMo4D AnyTop Official No-IK
tags:
  - motion
  - animal-motion
  - animation
  - bvh
  - anytop
  - planet-zoo
license: other
---

# AniMo4D AnyTop Official No-IK

Private archival release of the complete locally reconstructable AniMo4D official-caption Planet Zoo corpus, exported with IK disabled and encoded in AnyTop-compatible `KTJD-17` format.

## Files

| Archive | Size | Contents |
| --- | ---: | --- |
| `AniMo4D_AnyTop_Official_NoIK_v1_processed.tar.zst` | 17.19 GiB | 77,894 `[T,J,17]` motions, 311 skeleton files, 311 per-rig normalization files, exact official captions, QC reports, validation reports, and documentation. |
| `AniMo4D_AnyTop_Official_NoIK_v1_raw_bvhs.tar.zst` | 11.67 GiB | 77,894 full-topology no-IK motion BVHs, 311 rest BVHs, skeleton metadata, export manifest, and QC artifacts. |
| `BUILD_RECORD.md` | - | Exact corpus counts, joint-selection policy, and acceptance checks. |

## Integrity

```text
6E17FA2C3F211A1AD7F1F71E257D9477BFCF52F6F9B883DD8E4812276D00257C  AniMo4D_AnyTop_Official_NoIK_v1_processed.tar.zst
2EB399A9E8BB1416461DA8A697CDCC4D44618FBD9D1D5F0E12E87A173F2BCFEF  AniMo4D_AnyTop_Official_NoIK_v1_raw_bvhs.tar.zst
```

## Corpus Summary

- 311 Planet Zoo rigs
- 77,894 exact official AniMo4D caption-to-motion pairs
- 77,794 physical-QC clean clips, 100 high-dynamic review clips, 0 severe structural jumps, 0 QC errors
- 0 missing/duplicate official IDs, 0 non-finite values, 0 encoding failures
- Local 6D FK reconstruction maximum position error: `3.764338e-06`

The original official text inventory includes 255 captions for which the supplied local game assets have no corresponding source action. They are recorded in the processed archive and are not represented by fabricated motions.

## Extract

```bash
tar --use-compress-program=unzstd -xf AniMo4D_AnyTop_Official_NoIK_v1_processed.tar.zst
tar --use-compress-program=unzstd -xf AniMo4D_AnyTop_Official_NoIK_v1_raw_bvhs.tar.zst
```

On systems where `tar` automatically selects zstd based on file extension, `tar -xf <archive>` is sufficient.

## Representation And Joint Mapping

Each motion stores `[T,J,17]` KTJD-17 features. `T` ranges from 19 to 299 and `J` from 34 to 102. Joint selection is defined by the canonical supplied `20260822_pzh312_joint_names_descriptions.json` specification (SHA-256 `E9BF34078B6DE6E5769B399521C5D24E8ECA84ED5C28B503E875F526FF3F1E9C`). Raw BVHs remain full-topology; the encoded data uses exactly the per-rig joint list in that specification.

The generation code and detailed methodology are available in [CHDTevior/planetzoo-anytop-pipeline](https://github.com/CHDTevior/planetzoo-anytop-pipeline).
