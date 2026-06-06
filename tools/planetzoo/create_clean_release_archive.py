"""Create a clean Planet Zoo AnyTop release archive.

The script packages only the final pooled AnyTop layout files intended for
training/release. It excludes raw game assets, per-skeleton intermediates,
quarantine folders, backups, and stale temporary audit files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
from pathlib import Path
from typing import Iterable

import zstandard as zstd


INCLUDE_FILES = [
    "README.md",
    "cond.npy",
    "metadata.txt",
    "object_index.csv",
    "pack_manifest.jsonl",
    "pack_summary.json",
    "motion_texts_by_file_with_codex_drafts.json",
    "motion_texts_by_file_with_codex_drafts_summary.json",
    "motion_texts_by_file_with_animosty4d_matches.json",
    "motion_text_manifest.json",
    "motion_text_manifest.jsonl",
    "motion_text_manifest.csv",
    "motion_text_match_summary.json",
    "repair_bad_values_manifest.json",
    "post_repair_value_audit.json",
    "final_release_integrity_audit.json",
    "removed_risk_files_manifest.json",
]

INCLUDE_DIRS = [
    "motions",
    "bvhs",
    "animations",
    "caption_gif_qa_random100",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--package-name", default="AnyTopo_AniMo_PlanetZoo_v1_final_81994")
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--part-size-gib", type=float, default=2.0)
    parser.add_argument("--progress-log", type=Path, default=None)
    return parser.parse_args()


def iter_release_files(layout_root: Path) -> Iterable[Path]:
    for name in INCLUDE_FILES:
        path = layout_root / name
        if path.exists():
            yield path
    for dirname in INCLUDE_DIRS:
        dpath = layout_root / dirname
        if not dpath.exists():
            continue
        for path in sorted(dpath.rglob("*")):
            yield path


def log_progress(path: Path | None, message: dict) -> None:
    if path is None:
        return
    message = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **message}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(message, ensure_ascii=False) + "\n")


def format_gib(num_bytes: int) -> float:
    return round(num_bytes / (1024**3), 4)


def add_directory_entry(tar: tarfile.TarFile, layout_root: Path, package_name: str, directory: Path) -> None:
    arcname = Path(package_name) / directory.relative_to(layout_root)
    info = tarfile.TarInfo(str(arcname).replace("\\", "/"))
    info.type = tarfile.DIRTYPE
    info.mtime = int(directory.stat().st_mtime)
    tar.addfile(info)


def create_archive(layout_root: Path, output_dir: Path, package_name: str, compression_level: int, progress_log: Path | None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"{package_name}.tar.zst"
    if archive_path.exists():
        archive_path.unlink()

    files = [path for path in iter_release_files(layout_root) if path.is_file()]
    dirs = [layout_root / dirname for dirname in INCLUDE_DIRS if (layout_root / dirname).exists()]
    total_bytes = sum(path.stat().st_size for path in files)
    log_progress(
        progress_log,
        {
            "phase": "archive_start",
            "archive": str(archive_path),
            "file_count": len(files),
            "input_bytes": total_bytes,
            "input_gib": format_gib(total_bytes),
        },
    )

    written_bytes = 0
    next_log_bytes = 1 * 1024**3
    cctx = zstd.ZstdCompressor(level=compression_level, threads=-1)
    with archive_path.open("wb") as raw:
        with cctx.stream_writer(raw) as compressor:
            with tarfile.open(fileobj=compressor, mode="w|") as tar:
                for directory in dirs:
                    add_directory_entry(tar, layout_root, package_name, directory)
                for idx, path in enumerate(files, 1):
                    arcname = Path(package_name) / path.relative_to(layout_root)
                    tar.add(path, arcname=str(arcname).replace("\\", "/"), recursive=False)
                    written_bytes += path.stat().st_size
                    if written_bytes >= next_log_bytes or idx == len(files):
                        log_progress(
                            progress_log,
                            {
                                "phase": "archiving",
                                "files_done": idx,
                                "files_total": len(files),
                                "input_done_gib": format_gib(written_bytes),
                                "input_total_gib": format_gib(total_bytes),
                                "archive_size_gib": format_gib(archive_path.stat().st_size) if archive_path.exists() else 0,
                                "last_file": str(path.relative_to(layout_root)),
                            },
                        )
                        next_log_bytes = written_bytes + 1 * 1024**3

    log_progress(
        progress_log,
        {
            "phase": "archive_complete",
            "archive": str(archive_path),
            "archive_bytes": archive_path.stat().st_size,
            "archive_gib": format_gib(archive_path.stat().st_size),
        },
    )
    return archive_path


def split_and_checksum(archive_path: Path, part_size: int, progress_log: Path | None) -> dict:
    sha_archive = hashlib.sha256()
    parts = []
    checksums = []
    total_size = archive_path.stat().st_size
    done = 0
    part_idx = 1

    log_progress(
        progress_log,
        {
            "phase": "split_start",
            "archive": str(archive_path),
            "archive_bytes": total_size,
            "part_size_bytes": part_size,
        },
    )

    with archive_path.open("rb") as src:
        while True:
            chunk = src.read(part_size)
            if not chunk:
                break
            part_name = f"{archive_path.name}.part{part_idx:02d}"
            part_path = archive_path.with_name(part_name)
            if part_path.exists():
                part_path.unlink()
            part_sha = hashlib.sha256()
            with part_path.open("wb") as dst:
                dst.write(chunk)
            sha_archive.update(chunk)
            part_sha.update(chunk)
            part_digest = part_sha.hexdigest()
            parts.append(
                {
                    "file": part_name,
                    "bytes": len(chunk),
                    "size_gib": format_gib(len(chunk)),
                    "sha256": part_digest,
                }
            )
            checksums.append(f"{part_digest}  {part_name}")
            done += len(chunk)
            log_progress(
                progress_log,
                {
                    "phase": "splitting",
                    "part": part_idx,
                    "done_gib": format_gib(done),
                    "total_gib": format_gib(total_size),
                    "part_file": part_name,
                },
            )
            part_idx += 1

    archive_digest = sha_archive.hexdigest()
    checksums.insert(0, f"{archive_digest}  {archive_path.name}")
    manifest = {
        "archive": archive_path.name,
        "archive_bytes": total_size,
        "archive_size_gib": format_gib(total_size),
        "archive_sha256": archive_digest,
        "part_size_gib": round(part_size / (1024**3), 4),
        "parts": parts,
    }
    (archive_path.parent / "parts_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (archive_path.parent / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    log_progress(progress_log, {"phase": "split_complete", "parts": len(parts), "archive_sha256": archive_digest})
    return manifest


def main() -> None:
    args = parse_args()
    layout_root = args.layout_root.resolve()
    output_dir = args.output_dir.resolve()
    progress_log = args.progress_log or output_dir / "archive_progress.jsonl"
    if progress_log.exists():
        progress_log.unlink()

    archive = create_archive(
        layout_root=layout_root,
        output_dir=output_dir,
        package_name=args.package_name,
        compression_level=args.compression_level,
        progress_log=progress_log,
    )
    manifest = split_and_checksum(
        archive,
        part_size=int(args.part_size_gib * 1024**3),
        progress_log=progress_log,
    )
    print(json.dumps({"archive": str(archive), "manifest": manifest, "progress_log": str(progress_log)}, indent=2))


if __name__ == "__main__":
    main()
