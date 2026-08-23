"""Export one official AniMo4D rig's resolved MANIS Actions with IK disabled.

Run this script once per object in separate Blender processes.  The task
manifest is produced by ``build_official_animo4d_action_manifest.py`` and
contains the post-import Blender Action identifier required for a faithful
reproduction of AniMo4D's historical filenames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import bpy


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.planetzoo.planetzoo_fulltopo_bvh_export import (  # noqa: E402
    Reporter,
    clear_imported_actions,
    declared_action_names,
    export_bvh,
    export_rest_bvh,
    find_armature,
    import_ms2_armature_only,
    safe_clear_scene,
    write_skeleton_meta,
)


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cobra-tools", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--tasks-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--object-key", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-actions", type=int)
    return parser.parse_args(argv)


def short_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:20]


def safe_dirname(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def read_tasks(path: Path, object_key: str) -> list[dict]:
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("object_key") != object_key:
            continue
        if not row.get("source_file_exists"):
            raise ValueError(f"Task has no source file: {row['official_id']}")
        if not row.get("source_relative_file") or not row.get("blender_action"):
            raise ValueError(f"Task cannot resolve a Blender Action: {row['official_id']}")
        tasks.append(row)
    if not tasks:
        raise ValueError(f"No tasks for {object_key}")
    return sorted(tasks, key=lambda row: row["official_id"])


def register_cobra(cobra_tools: Path) -> None:
    import importlib.util

    sys.path.insert(0, str(cobra_tools))
    spec = importlib.util.spec_from_file_location(
        "cobra_tools_addon",
        str(cobra_tools / "__init__.py"),
        submodule_search_locations=[str(cobra_tools)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Cobra Tools from {cobra_tools}")
    addon = importlib.util.module_from_spec(spec)
    sys.modules["cobra_tools_addon"] = addon
    spec.loader.exec_module(addon)
    addon.register()


def action_record(task: dict, bvh: Path, status: str, fps: int, error: str | None = None) -> dict:
    record = {
        "sample_type": "motion",
        "status": status,
        "official_id": task["official_id"],
        "object_key": task["object_key"],
        "object_name": task["object_name"],
        "raw_bvh_stem": task["official_raw_bvh_stem"],
        "raw_bvh": str(bvh.resolve()),
        "raw_bvh_file": bvh.name,
        "source_relative_file": task["source_relative_file"],
        "source_declared_action": task.get("declared_action"),
        "source_blender_action": task["blender_action"],
        "source_resolution": task["source_resolution"],
        "source_action_verified": True,
        "ik_disabled_during_export": True,
        "export_fps": fps,
        "clip_storage_id": bvh.stem,
    }
    if error is not None:
        record["error"] = error
    return record


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    for path in (args.cobra_tools / "__init__.py", args.tasks_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    tasks = read_tasks(args.tasks_manifest, args.object_key)
    object_dir = args.source_root / f"{args.object_key}.ovl"
    ms2_files = sorted(object_dir.glob("*.ms2"))
    if len(ms2_files) != 1:
        raise RuntimeError(f"Expected exactly one MS2 in {object_dir}, found {len(ms2_files)}")
    for task in tasks:
        source = args.source_root / task["source_relative_file"]
        if source.parent != object_dir or not source.is_file():
            raise FileNotFoundError(f"Invalid task source for {task['official_id']}: {source}")

    object_dirname = safe_dirname(args.object_key)
    bvh_dir = args.output_root / "bvhs" / object_dirname
    rest_path = args.output_root / "rests" / f"{object_dirname}.bvh"
    object_manifest = args.output_root / "object_manifests" / f"{object_dirname}.jsonl"
    object_summary = args.output_root / "object_summaries" / f"{object_dirname}.json"
    skeleton_meta = args.output_root / "skeleton_meta" / f"{object_dirname}.json"
    for path in (bvh_dir, rest_path.parent, object_manifest.parent, object_summary.parent, skeleton_meta.parent):
        path.mkdir(parents=True, exist_ok=True)

    register_cobra(args.cobra_tools)
    from plugin import import_manis  # pylint: disable=import-outside-toplevel

    safe_clear_scene()
    reporter = Reporter()
    import_ms2_armature_only(reporter=reporter, filepath=str(ms2_files[0]))
    armature = find_armature()
    if armature is None:
        raise RuntimeError(f"No armature after MS2 import: {ms2_files[0]}")
    write_skeleton_meta(armature, skeleton_meta)
    if not rest_path.is_file():
        export_rest_bvh(armature, rest_path, args.fps)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for task in tasks:
        grouped[task["source_relative_file"]].append(task)

    records: list[dict] = [
        {
            "sample_type": "tpose",
            "status": "ok",
            "object_key": args.object_key,
            "raw_bvh_stem": f"{args.object_key.lower()}__tpos",
            "raw_bvh": str(rest_path.resolve()),
            "raw_bvh_file": rest_path.name,
            "source_action_verified": True,
            "ik_disabled_during_export": True,
            "export_fps": args.fps,
        }
    ]
    exported = skipped = errors = processed = 0
    for relative_file, source_tasks in sorted(grouped.items()):
        clear_imported_actions(armature)
        source_path = args.source_root / relative_file
        declared = declared_action_names(source_path)
        try:
            import_manis.load(reporter=reporter, filepath=str(source_path), disable_ik=True)
        except Exception as exc:
            message = f"MANIS import failed: {exc!r}"
            for task in source_tasks:
                bvh = bvh_dir / f"{short_id(task['official_id'])}.bvh"
                records.append(action_record(task, bvh, "error", args.fps, message))
                errors += 1
            traceback.print_exc()
            continue
        actions = {action.name: action for action in bpy.data.actions if action.id_root == "OBJECT"}
        for task in source_tasks:
            if args.max_actions is not None and processed >= args.max_actions:
                break
            processed += 1
            bvh = bvh_dir / f"{short_id(task['official_id'])}.bvh"
            if bvh.is_file() and bvh.stat().st_size > 0:
                records.append(action_record(task, bvh, "skipped_existing", args.fps))
                skipped += 1
                continue
            action = actions.get(task["blender_action"])
            if action is None:
                available = sorted(actions)
                records.append(
                    action_record(
                        task,
                        bvh,
                        "error",
                        args.fps,
                        f"Blender Action missing; expected={task['blender_action']!r}; declared={task.get('declared_action')!r}; available={available}",
                    )
                )
                errors += 1
                continue
            try:
                export_bvh(armature, action, bvh, args.fps)
                records.append(action_record(task, bvh, "ok", args.fps))
                exported += 1
            except Exception as exc:
                records.append(action_record(task, bvh, "error", args.fps, repr(exc)))
                errors += 1
                traceback.print_exc()
        if args.max_actions is not None and processed >= args.max_actions:
            break

    object_manifest.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8"
    )
    summary = {
        "object_key": args.object_key,
        "requested": len(tasks),
        "processed": processed,
        "exported": exported,
        "skipped_existing": skipped,
        "errors": errors,
        "complete": processed == len(tasks) and errors == 0,
        "object_manifest": str(object_manifest),
        "rest_bvh": str(rest_path),
        "skeleton_meta": str(skeleton_meta),
        "fps": args.fps,
        "ik_disabled": True,
    }
    object_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
