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
`skeletons/<rig>.npz:R_rest_global`. Its `P_rest_global` was FKed with that
matrix, which makes the displayed rest skeleton lie in the wrong basis even
though the animated motion coordinates are `+Y`-up / `+Z`-forward.

For every rig, the repair does the following:

1. Confirms all joints share one old rest-basis matrix `B`.
2. Replaces `R_rest_global` and `R_rest_local` by identity matrices.
3. Rebuilds `P_rest_global` by accumulating `offset_parent_local`, then moves
   the root upward so the rest skeleton's lowest point is at `Y=0`.
4. Converts `motion[..., 3:9]` from old rest-delta rotation `D` to the
   equivalent canonical rotation `D @ B`.
5. Recomputes each rig's `stats/<rig>.npz` from the rewritten motion files.

Channels `0:3`, `9:12`, `12`, `13:15`, and `15:17` are copied unchanged. The
first clip of every rig is checked by FK against the original world-position
channels.

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
