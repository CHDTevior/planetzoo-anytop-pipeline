# Canonical Rest-Pose Repair

This procedure creates a new AnyTop-format release with static skeletons in the
same coordinate system as the motion clips:

- right-handed coordinates;
- `+Y` is up;
- `+Z` is the canonical forward direction;
- the lowest rest-pose joint is placed on `Y=0`.

It is for `AniMo4D_AnyTop_Official_NoIK_v1` and relies only on Python and
NumPy. The input directory is never modified.

## Why This Is Needed

The original release contains a rig-wide constant matrix in
`skeletons/<rig>.npz:R_rest_global`, while its saved static skeleton is in a
different global basis from its motion world coordinates. Treating that root
basis as a ground transform is wrong: it can make a four-legged animal stand
with its front and rear feet at different heights.

For every rig, the repair does the following:

1. Finds terminal toe/foot/hoof joints and fits their original rest-pose
   support plane.
2. Uses the plane normal as the canonical `+Y` direction and the hips-to-chest
   vector projected onto that plane as canonical `+Z`.
3. Applies this one rigid rotation `C` to the full rest skeleton, preserving
   the original foot plane, then translates the root so the lowest point is
   at `Y=0`.
4. Converts every `motion[..., 3:9]` rest-delta rotation `D` to `D @ C.T`, and
   transforms `R_rest_global` by `C`; their product, and therefore every FK
   world position, remains unchanged.
5. Recomputes each rig's `stats/<rig>.npz` from the rewritten motion files.

Channels `0:3`, `9:12`, `12`, `13:15`, and `15:17` are copied unchanged. The
first clip of every rig is checked by FK against the original world-position
channels. Every corrected skeleton is also checked for finite values, static
FK reconstruction, a proper right-handed transform (determinant `+1`), exact
`+Z` hips-to-chest heading, and `min(Y)=0`.

The terminal toe/foot plane is an *orientation estimate*, not a deformation
target. Paws, claws, flippers, and aquatic rigs can have terminal points at
different heights in the source asset. The procedure keeps those differences:
it never edits individual joints or bone lengths merely to make all terminals
coplanar.

## Requirements

```bash
python -m pip install numpy
```

Use Python 3.10 or later. No Blender, PyTorch, or game assets are required.

## Full Server Run

Assume the original release is at `/data/AniMo4D_AnyTop_Official_NoIK_v1` and
the output directory does not exist yet:

```bash
git clone https://github.com/CHDTevior/planetzoo-anytop-pipeline.git
cd planetzoo-anytop-pipeline

python tools/planetzoo/recanonicalize_noik_rest_pose_release.py \
  --input-root /data/AniMo4D_AnyTop_Official_NoIK_v1 \
  --output-root /data/AniMo4D_AnyTop_Official_NoIK_v1_canonical_rest \
  --workers 8
```

The output root must be new and empty. Leave room for the full second processed
release while it runs. Start with `--workers 4` if server RAM or storage I/O is
limited.

On Windows PowerShell the equivalent is:

```powershell
python .\tools\planetzoo\recanonicalize_noik_rest_pose_release.py `
  --input-root 'H:\AniMo4D_work\releases\AniMo4D_AnyTop_Official_NoIK_v1' `
  --output-root 'H:\AniMo4D_work\releases\AniMo4D_AnyTop_Official_NoIK_v1_canonical_rest' `
  --workers 4
```

## Small Validation Run

Run two rigs before committing to the full conversion:

```bash
python tools/planetzoo/recanonicalize_noik_rest_pose_release.py \
  --input-root /data/AniMo4D_AnyTop_Official_NoIK_v1 \
  --output-root /data/noik_canonical_rest_smoketest \
  --workers 1 \
  --rig PZ_Aardvark_Female \
  --rig PZ_Reticulated_Giraffe_Female
```

The command prints one FK error per rig. Values around `1e-7` are float32
roundoff and expected. The summary is written to:

```text
<output>/data/reports/rest_pose_recanonicalization_summary.json
<output>/data/reports/rest_pose_recanonicalization_report.csv
```

## Visual Preview

For a few rig-level before/after figures without rewriting the full corpus:

```bash
python tools/planetzoo/preview_recanonicalize_noik_rest_pose.py \
  --release-root /data/AniMo4D_AnyTop_Official_NoIK_v1 \
  --output-dir /data/rest_preview \
  --rig PZ_Reticulated_Giraffe_Female
```

The generated image places current rest, corrected rest, and motion frame zero
in rows, each with top, side, and front views labelled with coordinate axes.
