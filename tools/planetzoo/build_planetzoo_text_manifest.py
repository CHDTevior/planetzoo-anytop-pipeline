"""Build a processed-motion manifest and optionally attach text captions.

The Planet Zoo export step writes ``export_manifest.jsonl`` with stable source
keys for each raw BVH. AnyTop then writes processed files named like:

    <object_name>_<raw_bvh_stem>_<counter>.npy

This script joins those two worlds and matches text from an optional directory
or file. Missing text is intentionally left as an empty string.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


TEXT_SUFFIXES = {".txt", ".json", ".jsonl"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True, help="AnyTop output directory, or a root containing per-object AnyTop directories.")
    parser.add_argument("--export-manifest", required=True, help="JSONL manifest written by planetzoo_fulltopo_bvh_export.py.")
    parser.add_argument("--text-root", default=None, help="Optional AniMo/custom text file or directory.")
    parser.add_argument("--output", default=None, help="Output JSONL path. Defaults to <processed-dir>/motion_text_manifest.jsonl.")
    parser.add_argument("--json-output", default=None, help="Optional regular JSON manifest with summary and rows.")
    parser.add_argument("--csv-output", default=None, help="Optional CSV copy for quick spreadsheet inspection.")
    return parser.parse_args()


def norm_key(value: str | None) -> str:
    if not value:
        return ""
    value = Path(str(value)).stem if any(sep in str(value) for sep in ("/", "\\")) else str(value)
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def add_text(text_index: dict[str, dict[str, Any]], key: str, texts: list[str], source: str) -> None:
    clean = [text.strip() for text in texts if text and text.strip()]
    if not clean:
        return
    normalized = norm_key(key)
    if not normalized:
        return
    bucket = text_index.setdefault(normalized, {"texts": [], "sources": []})
    for text in clean:
        if text not in bucket["texts"]:
            bucket["texts"].append(text)
    if source not in bucket["sources"]:
        bucket["sources"].append(source)


def text_from_humanml_line(line: str) -> str:
    return line.split("#", 1)[0].strip()


def load_txt_texts(path: Path, text_index: dict[str, dict[str, Any]]) -> None:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    texts = [text_from_humanml_line(line) for line in lines if line.strip()]
    add_text(text_index, path.stem, texts, str(path))


def load_json_record(record: Any, fallback_key: str, source: str, text_index: dict[str, dict[str, Any]]) -> None:
    if isinstance(record, str):
        add_text(text_index, fallback_key, [record], source)
        return
    if not isinstance(record, dict):
        return

    key = (
        record.get("id")
        or record.get("motion_id")
        or record.get("motion")
        or record.get("motion_name")
        or record.get("file")
        or record.get("file_name")
        or record.get("name")
        or fallback_key
    )
    raw_texts = record.get("texts") or record.get("captions") or record.get("caption") or record.get("text") or record.get("description")
    if isinstance(raw_texts, str):
        texts = [raw_texts]
    elif isinstance(raw_texts, list):
        texts = [str(item) for item in raw_texts]
    else:
        texts = []
    add_text(text_index, str(key), texts, source)


def load_json_texts(path: Path, text_index: dict[str, dict[str, Any]]) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str):
                add_text(text_index, key, [value], str(path))
            elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                add_text(text_index, key, value, str(path))
            else:
                load_json_record(value, key, str(path), text_index)
    elif isinstance(data, list):
        for idx, record in enumerate(data):
            load_json_record(record, f"{path.stem}_{idx}", str(path), text_index)


def load_jsonl_texts(path: Path, text_index: dict[str, dict[str, Any]]) -> None:
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        load_json_record(json.loads(line), f"{path.stem}_{idx}", str(path), text_index)


def load_text_index(text_root: str | None) -> dict[str, dict[str, Any]]:
    text_index: dict[str, dict[str, Any]] = {}
    if not text_root:
        return text_index
    root = Path(text_root)
    files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES]
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            load_txt_texts(path, text_index)
        elif suffix == ".json":
            load_json_texts(path, text_index)
        elif suffix == ".jsonl":
            load_jsonl_texts(path, text_index)
    return text_index


def candidate_keys(entry: dict[str, Any], processed_stem: str) -> list[str]:
    keys = [
        processed_stem,
        entry.get("raw_bvh_stem"),
        entry.get("source_motion_key"),
        entry.get("action_name"),
        entry.get("action_short"),
    ]
    animal_key = entry.get("animal_key")
    action_short = entry.get("action_short")
    if animal_key and action_short:
        keys.extend([f"{animal_key}_{action_short}", f"{animal_key}@{action_short}"])
    return [key for key in keys if key]


def build_export_index(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        entry["raw_bvh_stem"]: entry
        for entry in entries
        if entry.get("sample_type", "motion") == "motion" and entry.get("raw_bvh_stem")
    }


def find_export_entry(processed_stem: str, processed_object_name: str, export_index: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    prefix = processed_object_name + "_"
    if processed_stem.startswith(prefix):
        tail = processed_stem[len(prefix):]
        stem_parts = tail.rsplit("_", 1)
        if len(stem_parts) == 2 and stem_parts[1].isdigit():
            direct = export_index.get(stem_parts[0])
            if direct is not None:
                return direct
    stem_parts = processed_stem.rsplit("_", 1)
    if len(stem_parts) == 2 and stem_parts[1].isdigit():
        processed_without_counter = stem_parts[0]
        for raw_stem in export_index:
            if processed_without_counter.endswith(raw_stem):
                return export_index[raw_stem]
    for raw_stem, entry in export_index.items():
        if raw_stem in processed_stem:
            return entry
    return None


def build_manifest(processed_dir: Path, export_index: dict[str, dict[str, Any]], text_index: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    motions_dir = processed_dir / "motions"
    for motion_path in sorted(motions_dir.glob("*.npy")):
        processed_stem = motion_path.stem
        entry = find_export_entry(processed_stem, processed_dir.name, export_index)
        texts: list[str] = []
        text_sources: list[str] = []
        text_match_key = ""

        if entry is not None:
            for key in candidate_keys(entry, processed_stem):
                match = text_index.get(norm_key(key))
                if match:
                    texts = match["texts"]
                    text_sources = match["sources"]
                    text_match_key = key
                    break

        status = "matched" if texts else "missing_text"
        if entry is None:
            status = "missing_export_manifest"

        row = {
            "processed_motion": str(motion_path.resolve()),
            "processed_bvh": str((processed_dir / "bvhs" / f"{processed_stem}.bvh").resolve()),
            "processed_animation": str((processed_dir / "animations" / f"{processed_stem}_from_ric.mp4").resolve()),
            "text": texts[0] if texts else "",
            "texts": texts,
            "text_source": text_sources[0] if text_sources else "",
            "text_sources": text_sources,
            "text_match_key": text_match_key,
            "text_match_status": status,
        }
        if entry:
            row.update(entry)
        rows.append(row)
    return rows


def iter_processed_dirs(processed_dir: Path) -> list[Path]:
    if (processed_dir / "motions").is_dir():
        return [processed_dir]
    return sorted([p for p in processed_dir.iterdir() if p.is_dir() and (p / "motions").is_dir()])


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status_counts: dict[str, int] = {}
    object_counts: dict[str, int] = {}
    for row in rows:
        status = row.get("text_match_status", "")
        status_counts[status] = status_counts.get(status, 0) + 1
        object_name = row.get("object_name") or row.get("animal_key") or Path(row["processed_motion"]).parent.parent.name
        object_counts[object_name] = object_counts.get(object_name, 0) + 1
    payload = {
        "summary": {
            "rows": len(rows),
            "status_counts": status_counts,
            "object_counts": object_counts,
        },
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "processed_motion",
        "raw_bvh",
        "object_dir",
        "animal_key",
        "action_name",
        "action_short",
        "source_motion_key",
        "text",
        "text_match_status",
        "text_match_key",
        "text_source",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    output = Path(args.output) if args.output else processed_dir / "motion_text_manifest.jsonl"
    export_entries = read_jsonl(Path(args.export_manifest))
    export_index = build_export_index(export_entries)
    text_index = load_text_index(args.text_root)
    rows: list[dict[str, Any]] = []
    for cur_processed_dir in iter_processed_dirs(processed_dir):
        rows.extend(build_manifest(cur_processed_dir, export_index, text_index))
    write_jsonl(output, rows)
    if args.json_output:
        write_json(Path(args.json_output), rows)
    if args.csv_output:
        write_csv(Path(args.csv_output), rows)
    matched = sum(1 for row in rows if row["text_match_status"] == "matched")
    print(f"WROTE {output} rows={len(rows)} matched_text={matched} missing_text={len(rows) - matched}")


if __name__ == "__main__":
    main()
