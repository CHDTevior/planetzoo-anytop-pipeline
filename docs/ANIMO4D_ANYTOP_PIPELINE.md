# AniMo4D_Anytop Pipeline

This pipeline builds a Planet Zoo motion dataset that is aligned to the
AniMo4D text/sample list while preserving full AnyTop topology.

## Dataset Contract

- Source of truth: `H:/AniMo4D_work/texts`
- Target sample count: one motion per AniMo4D text file where the local Planet
  Zoo build exposes the exact action
- Current source manifest count: 78,149 motions
- Target object count: 311 Planet Zoo species/sex/age objects
- Text format: `species#gender#caption#tokens#f_tag#to_tag`
- Motion format: AnyTop `(N, J, 13)` plus `cond.npy`
- BVH intermediates are retained for inspection.

Unlike the earlier full Planet Zoo dataset, this version does not use visual
caption generation and does not include extra Planet Zoo actions that are not
present in the AniMo4D text directory.

Current processed result:

- Official AniMo4D text rows: `78,149`
- Exact raw/processed matches: `77,894`
- Missing rows: `255`
- Objects: `311`
- Node count range before pooled packing: `143-344`
- Clip length range after AnyTop velocity/contact processing: `19-299`
- Feature dimension: `13`
- No fuzzy action substitution was applied.

## Step 1: Build Source Manifest

```powershell
C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe `
  tools/planetzoo/build_animo4d_anytop_manifest.py `
  --text-root H:/AniMo4D_work/texts `
  --output-jsonl H:/AniMo4D_work/AniMo4D_Anytop/manifests/source_text_manifest.jsonl `
  --summary-json H:/AniMo4D_work/AniMo4D_Anytop/manifests/source_text_manifest_summary.json `
  --csv-output H:/AniMo4D_work/AniMo4D_Anytop/manifests/source_text_manifest.csv
```

Expected summary:

- `rows`: `78149`
- `object_count`: `311`
- `text_issues`: `{}`

## Step 2: Export Target Raw BVHs

The exporter reads the same AniMo4D text directory and exports only actions
whose raw BVH stem is present in that text manifest.

```powershell
H:/codex_project1/.codex-tmp/venvs/cobra/Scripts/python.exe `
  tools/planetzoo/planetzoo_parallel_bvh_export.py `
  --blender H:/blender4_5/blender.exe `
  --cobra-tools H:/codex_project1/.codex-tmp/AniMo/data_generation/export_json/cobra-tools `
  --input-root H:/AniMo4D_work/01_ovl_extracted `
  --output-root H:/AniMo4D_work/AniMo4D_Anytop/00_raw_bvh_target `
  --workers 4 `
  --target-text-root H:/AniMo4D_work/texts
```

Progress files:

```text
H:/AniMo4D_work/AniMo4D_Anytop/00_raw_bvh_target/parallel_bvh_export_status.jsonl
H:/AniMo4D_work/AniMo4D_Anytop/00_raw_bvh_target/logs
H:/AniMo4D_work/AniMo4D_Anytop/00_raw_bvh_target/status
```

If the first pass leaves missing raw rows, build the missing-object list from
the raw alignment manifest and rerun only those objects. The exporter catches
per-`.manis` Python importer failures and continues with the remaining `.manis`
files for that object.

```powershell
H:/codex_project1/.codex-tmp/venvs/cobra/Scripts/python.exe `
  tools/planetzoo/planetzoo_parallel_bvh_export.py `
  --blender H:/blender4_5/blender.exe `
  --cobra-tools H:/codex_project1/.codex-tmp/AniMo/data_generation/export_json/cobra-tools `
  --input-root H:/AniMo4D_work/01_ovl_extracted `
  --output-root H:/AniMo4D_work/AniMo4D_Anytop/00_raw_bvh_target `
  --workers 2 `
  --target-text-root H:/AniMo4D_work/texts `
  --objects <missing object names>
```

Some official AniMo4D text keys record a different `.manis` group than the
current local Planet Zoo build exposes for the same action. After raw export,
materialize deterministic aliases by matching the same object and action suffix:

```powershell
C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe `
  tools/planetzoo/materialize_animo4d_raw_aliases.py `
  --alignment-manifest H:/AniMo4D_work/AniMo4D_Anytop/manifests/raw_alignment_manifest.jsonl `
  --raw-root H:/AniMo4D_work/AniMo4D_Anytop/00_raw_bvh_target `
  --output-jsonl H:/AniMo4D_work/AniMo4D_Anytop/manifests/raw_alias_manifest.jsonl `
  --summary-json H:/AniMo4D_work/AniMo4D_Anytop/manifests/raw_alias_summary.json `
  --mode hardlink
```

For objects where Blender crashes after loading several `.manis` files, rerun
one group at a time with `--only-manis-contains`. This recovered the Quokka,
Dromedary Camel juvenile, and Capuchin Monkey male gaps caused by sequential
import crashes.

## Step 3: Convert to AnyTop Without Splitting

AnyTop's original new-skeleton preprocessing splits BVHs longer than 240 source
frames into 200-frame chunks. AniMo4D_Anytop disables that split so the sample
count remains exactly aligned with AniMo4D text files.

```powershell
$env:ANYTOP_SKIP_ANIMATIONS='1'
H:/codex_project1/.codex-tmp/venvs/cobra/Scripts/python.exe `
  tools/planetzoo/planetzoo_parallel_anytop_process.py `
  --raw-root H:/AniMo4D_work/AniMo4D_Anytop/00_raw_bvh_target `
  --output-root H:/AniMo4D_work/AniMo4D_Anytop/01_anytop_target `
  --python H:/codex_project1/.codex-tmp/venvs/cobra/Scripts/python.exe `
  --repo-root H:/codex_project1/.codex-tmp/planetzoo-anytop-pipeline-upload `
  --workers 6 `
  --skip-animations `
  --max-clip-frames 0
```

## Step 4: Audit Processed Alignment

```powershell
C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe `
  tools/planetzoo/build_animo4d_anytop_manifest.py `
  --text-root H:/AniMo4D_work/texts `
  --raw-root H:/AniMo4D_work/AniMo4D_Anytop/00_raw_bvh_target `
  --processed-root H:/AniMo4D_work/AniMo4D_Anytop/01_anytop_target `
  --output-jsonl H:/AniMo4D_work/AniMo4D_Anytop/manifests/processed_alignment_manifest.jsonl `
  --summary-json H:/AniMo4D_work/AniMo4D_Anytop/manifests/processed_alignment_summary.json `
  --csv-output H:/AniMo4D_work/AniMo4D_Anytop/manifests/processed_alignment_manifest.csv
```

Ideal final condition:

- `status_counts.matched == 78149`
- no `missing_raw`
- no `missing_processed`
- no `duplicate_processed`

Current local result:

- `status_counts.matched == 77894`
- `status_counts.missing_processed+missing_raw == 255`
- The missing rows are recorded in:
  `H:/AniMo4D_work/AniMo4D_Anytop/manifests/missing_processed_current.jsonl`
- Remaining missing rows have no exact same-object action after deterministic
  same-action aliasing.

## Step 5: Build Text Manifest

```powershell
C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe `
  tools/planetzoo/build_animo4d_anytop_text_manifest.py `
  --alignment-manifest H:/AniMo4D_work/AniMo4D_Anytop/manifests/processed_alignment_current_manifest.jsonl `
  --output-jsonl H:/AniMo4D_work/AniMo4D_Anytop/01_anytop_target/motion_text_manifest.jsonl `
  --output-json H:/AniMo4D_work/AniMo4D_Anytop/01_anytop_target/motion_text_manifest.json `
  --output-csv H:/AniMo4D_work/AniMo4D_Anytop/01_anytop_target/motion_text_manifest.csv `
  --by-file-output H:/AniMo4D_work/AniMo4D_Anytop/01_anytop_target/motion_texts_by_file_with_animo4d_official.json `
  --missing-output H:/AniMo4D_work/AniMo4D_Anytop/manifests/missing_processed_current.jsonl
```

Result:

- `matched_rows`: `77,894`
- `missing_rows`: `255`
- `annotation_source`: `animo4d_official_text`

## Step 6: Pack Pooled AnyTop Layout

```powershell
H:/codex_project1/.codex-tmp/venvs/cobra/Scripts/python.exe `
  tools/planetzoo/pack_planetzoo_anytop_dataset.py `
  --processed-root H:/AniMo4D_work/AniMo4D_Anytop/01_anytop_target `
  --output-root H:/AniMo4D_work/AniMo4D_Anytop/02_anytop_layout `
  --text-manifest H:/AniMo4D_work/AniMo4D_Anytop/01_anytop_target/motion_text_manifest.jsonl `
  --link-mode auto
```

Packed result:

- `motions`: `77,894`
- `bvhs`: `77,894`
- `objects`: `311`
- `max_joints`: `344`
- `total_frames`: `6,179,510`

## Step 7: Value Repair And Audit

```powershell
H:/codex_project1/.codex-tmp/venvs/cobra/Scripts/python.exe `
  tools/planetzoo/repair_bad_motion_values.py `
  --layout-root H:/AniMo4D_work/AniMo4D_Anytop/02_anytop_layout `
  --threshold 22.53 `
  --std-floor 1e-6
```

Final audit:

- `bad_motion_count`: `0`
- motion non-finite files: `0`
- max absolute motion value: `11.6253`
- `cond.std_min`: `1e-6`
- `cond.std_max`: `1.0`
- final audit file:
  `H:/AniMo4D_work/AniMo4D_Anytop/02_anytop_layout/final_integrity_audit.json`

## Demo Validation

The current single-object validation used `Aardvark_Female.ovl`:

- official text motions: 96
- target raw motion BVHs: 96
- processed AnyTop motions with `--max-clip-frames 0`: 96
