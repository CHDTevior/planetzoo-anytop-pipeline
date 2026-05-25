# Planet Zoo AnyTop Usage Guide

This guide shows the practical commands for reproducing the Planet Zoo to
AnyTop dataset pipeline. It assumes the current Windows workstation layout.

## Paths

```text
Repo:
H:/codex_project1/.codex-tmp/planetzoo-anytop-pipeline-upload

Planet Zoo install:
G:/Steam/steamapps/common/Planet Zoo

Extracted OVL assets:
H:/AniMo4D_work/01_ovl_extracted

Blender:
H:/blender4_5/blender.exe

CobraTools:
H:/codex_project1/.codex-tmp/AniMo/data_generation/export_json/cobra-tools

Python:
H:/codex_project1/.codex-tmp/venvs/cobra/Scripts/python.exe
```

## Outputs

Use separate output roots for demo runs and full runs.

```text
Raw full-topology BVH:
H:/AniMo4D_work/05_fulltopo_raw_bvh_full

AnyTop processed dataset:
H:/AniMo4D_work/06_anytop_processed_full
```

Important files produced by the full run:

```text
H:/AniMo4D_work/05_fulltopo_raw_bvh_full/export_manifest.jsonl
H:/AniMo4D_work/05_fulltopo_raw_bvh_full/parallel_bvh_export_status.jsonl
H:/AniMo4D_work/06_anytop_processed_full/parallel_anytop_process_status.jsonl
H:/AniMo4D_work/06_anytop_processed_full/dataset_summary.json
H:/AniMo4D_work/06_anytop_processed_full/dataset_summary_objects.csv
H:/AniMo4D_work/06_anytop_processed_full/motion_text_manifest.json
H:/AniMo4D_work/06_anytop_processed_full/motion_text_manifest.jsonl
H:/AniMo4D_work/06_anytop_processed_full/motion_text_manifest.csv
H:/AniMo4D_work/06_anytop_processed_full/full_pipeline_report.json
```

## Step 1. Export Full-Topology BVH

Run from the repo root:

```powershell
H:/codex_project1/.codex-tmp/venvs/cobra/Scripts/python.exe tools/planetzoo/planetzoo_parallel_bvh_export.py `
  --blender H:/blender4_5/blender.exe `
  --cobra-tools H:/codex_project1/.codex-tmp/AniMo/data_generation/export_json/cobra-tools `
  --input-root H:/AniMo4D_work/01_ovl_extracted `
  --output-root H:/AniMo4D_work/05_fulltopo_raw_bvh_full `
  --workers 4 `
  --only-manis-contains locomotion `
  --overwrite
```

Notes:

- Each worker launches a separate Blender process.
- Do not use `--max-actions` for the full dataset.
- Use `--skip-complete` instead of `--overwrite` to resume without deleting
  completed object folders.
- Per-object logs are written to:

```text
H:/AniMo4D_work/05_fulltopo_raw_bvh_full/logs
H:/AniMo4D_work/05_fulltopo_raw_bvh_full/status
```

## Step 2. Convert BVH to AnyTop Format

Run from the repo root:

The Planet Zoo conversion path aligns motion to AnyTop's `Y up`, `+Z forward`
convention and bakes the processed BVH rest basis so Blender Edit Mode shows an
upright rest pose while preserving animated global joint positions. The rest
basis uses a per-object roll choice, so do not reuse older fixed-roll outputs.

```powershell
H:/codex_project1/.codex-tmp/venvs/cobra/Scripts/python.exe tools/planetzoo/planetzoo_parallel_anytop_process.py `
  --raw-root H:/AniMo4D_work/05_fulltopo_raw_bvh_full `
  --output-root H:/AniMo4D_work/06_anytop_processed_full_autoroll `
  --python H:/codex_project1/.codex-tmp/venvs/cobra/Scripts/python.exe `
  --repo-root H:/codex_project1/.codex-tmp/planetzoo-anytop-pipeline-upload `
  --workers 6 `
  --overwrite `
  --skip-animations
```

Notes:

- `--skip-animations` skips AnyTop MP4 sanity renders.
- The converter still writes `motions/*.npy`, processed `bvhs/*.bvh`, and
  `cond.npy`.
- Use `--skip-complete` instead of `--overwrite` to resume a partial run.
- Per-object logs are written to:

```text
H:/AniMo4D_work/06_anytop_processed_full_autoroll/logs_anytop
H:/AniMo4D_work/06_anytop_processed_full_autoroll/status_anytop
```

## Step 3. Build Text Manifest

Without external captions:

```powershell
H:/codex_project1/.codex-tmp/venvs/cobra/Scripts/python.exe tools/planetzoo/build_planetzoo_text_manifest.py `
  --processed-dir H:/AniMo4D_work/06_anytop_processed_full `
  --export-manifest H:/AniMo4D_work/05_fulltopo_raw_bvh_full/export_manifest.jsonl `
  --output H:/AniMo4D_work/06_anytop_processed_full/motion_text_manifest.jsonl `
  --json-output H:/AniMo4D_work/06_anytop_processed_full/motion_text_manifest.json `
  --csv-output H:/AniMo4D_work/06_anytop_processed_full/motion_text_manifest.csv
```

With external captions, add:

```powershell
  --text-root H:/path/to/texts_or_manifest
```

Missing captions are intentionally kept as empty text. The motion row is not
dropped.

## Step 4. Summarize Dataset

Fast summary without reading every array fully:

```powershell
H:/codex_project1/.codex-tmp/venvs/cobra/Scripts/python.exe tools/planetzoo/summarize_anytop_dataset.py `
  --processed-root H:/AniMo4D_work/06_anytop_processed_full `
  --output-json H:/AniMo4D_work/06_anytop_processed_full/dataset_summary.json `
  --output-csv H:/AniMo4D_work/06_anytop_processed_full/dataset_summary_objects.csv
```

For a slower validation pass that checks every value is finite:

```powershell
H:/codex_project1/.codex-tmp/venvs/cobra/Scripts/python.exe tools/planetzoo/summarize_anytop_dataset.py `
  --processed-root H:/AniMo4D_work/06_anytop_processed_full `
  --output-json H:/AniMo4D_work/06_anytop_processed_full/dataset_summary_finite.json `
  --check-finite
```

## Current Full-Run Result

The current full run produced:

```text
OVL scanned: 556
Raw export ok: 523
Raw export error: 33
Zero-motion objects: 83
Raw motion BVH files: 81685

AnyTop processed objects: 473
AnyTop processed clips: 82035
Feature dimension: 13
Node count range: 88-411
Clip length range: 2-237
Clips per object range: 23-314
```

The processed feature layout follows AnyTop:

```text
shape = (frames - 1, joints, 13)

0:3    local/root-invariant joint position
3:9    local joint rotation in 6D representation
9:12   local joint velocity
12:13  foot contact flag
```

## Step 6. Render VLM Preview Images

For captioning or visual QA, render compact JPEG previews instead of full MP4s.
Each object gets one rest-pose image. Each motion gets one storyboard image with
8 sampled frames, a small axis marker, and the root trajectory.

```powershell
H:/codex_project1/.codex-tmp/venvs/cobra/Scripts/python.exe tools/planetzoo/render_anytop_vlm_previews.py `
  --processed-root H:/AniMo4D_work/06_anytop_processed_full_autoroll `
  --output-root H:/AniMo4D_work/07_vlm_previews_autoroll `
  --workers 10 `
  --frames-per-action 8 `
  --cell-size 220 `
  --rest-size 520 `
  --quality 88
```

Current preview output:

```text
Rest previews: 473
Action previews: 82035
Total JPEGs: 82508
Approx size: 3.5 GB
Manifest: H:/AniMo4D_work/07_vlm_previews_autoroll/vlm_preview_manifest.jsonl
CSV: H:/AniMo4D_work/07_vlm_previews_autoroll/vlm_preview_manifest.csv
```

## Retry Individual Objects

BVH export retry:

```powershell
H:/codex_project1/.codex-tmp/venvs/cobra/Scripts/python.exe tools/planetzoo/planetzoo_parallel_bvh_export.py `
  --blender H:/blender4_5/blender.exe `
  --cobra-tools H:/codex_project1/.codex-tmp/AniMo/data_generation/export_json/cobra-tools `
  --input-root H:/AniMo4D_work/01_ovl_extracted `
  --output-root H:/AniMo4D_work/05_fulltopo_raw_bvh_full `
  --objects Red_Panda_Female.ovl `
  --workers 1 `
  --only-manis-contains locomotion `
  --overwrite
```

AnyTop retry:

```powershell
H:/codex_project1/.codex-tmp/venvs/cobra/Scripts/python.exe tools/planetzoo/planetzoo_parallel_anytop_process.py `
  --raw-root H:/AniMo4D_work/05_fulltopo_raw_bvh_full `
  --output-root H:/AniMo4D_work/06_anytop_processed_full `
  --objects Red_Panda_Female_ovl `
  --python H:/codex_project1/.codex-tmp/venvs/cobra/Scripts/python.exe `
  --repo-root H:/codex_project1/.codex-tmp/planetzoo-anytop-pipeline-upload `
  --workers 1 `
  --overwrite `
  --skip-animations
```

## What Not To Commit

Do not commit extracted game assets or generated datasets:

```text
H:/AniMo4D_work/01_ovl_extracted
H:/AniMo4D_work/05_fulltopo_raw_bvh_full
H:/AniMo4D_work/06_anytop_processed_full_autoroll
H:/AniMo4D_work/07_vlm_previews_autoroll
```

Only commit scripts, docs, and small metadata examples.
