"""Build and run resumable per-topology AnyTop-to-MS2 mesh validation jobs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender", required=True, type=Path)
    parser.add_argument("--cobra-tools", required=True, type=Path)
    parser.add_argument("--motion-lib", required=True, type=Path)
    parser.add_argument("--layout-root", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--extracted-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--object", dest="objects", action="append", help="Restrict validation to one object; repeatable.")
    parser.add_argument("--resume", action="store_true", help="Skip objects with an existing passing report.")
    parser.add_argument("--build-jobs-only", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def raw_stem_from_motion(motion: Path, object_name: str) -> str | None:
    prefix = object_name + "_"
    if not motion.stem.startswith(prefix):
        return None
    stem, separator, suffix = motion.stem[len(prefix) :].rpartition("_")
    return stem if separator and suffix.isdigit() else None


def motion_priority(path: Path) -> tuple[int, str]:
    name = path.name.lower()
    preferred = ("walkbase", "runbase", "walk", "run", "turn", "swimbase", "idle")
    return (0 if any(token in name for token in preferred) else 1, name)


def infer_native_assets(raw_object_dir: Path, extracted_root: Path, raw_template: Path) -> dict | None:
    """Recover the one missing historical manifest from the raw BVH stem."""
    parts = raw_template.stem.split("__")
    if len(parts) < 3 or "_maniset" not in parts[1]:
        return None
    object_dir = raw_object_dir.name.removesuffix("_ovl") + ".ovl"
    source_dir = extracted_root / object_dir
    ms2_files = sorted(source_dir.glob("*.ms2"))
    manis_name = parts[1].replace("_maniset", ".maniset", 1) + ".manis"
    if len(ms2_files) != 1 or not (source_dir / manis_name).is_file():
        return None
    return {
        "object_dir": object_dir,
        "ms2_file": ms2_files[0].name,
        "manis_file": manis_name,
        "mapping_origin": "inferred_from_raw_bvh_stem",
    }


def resolve_job(row: dict, args: argparse.Namespace) -> tuple[dict | None, dict | None]:
    object_name = row["object_name"]
    raw_object_dir = args.raw_root / (object_name.removeprefix("PZ_") + "_ovl")
    raw_bvhs = raw_object_dir / "raw_bvhs"
    manifest_path = raw_object_dir / "export_manifest.jsonl"
    tpose_candidates = sorted(raw_bvhs.glob("*__tpos.bvh"))
    if not raw_bvhs.is_dir() or not tpose_candidates:
        return None, {"object_name": object_name, "reason": "missing_raw_manifest_or_tpose", "raw_object_dir": str(raw_object_dir)}
    manifest = (
        {
            Path(item["raw_bvh"]).name: item
            for item in read_jsonl(manifest_path)
            if item.get("sample_type") == "motion"
        }
        if manifest_path.is_file()
        else {}
    )
    motions = sorted((args.layout_root / "motions").glob(object_name + "_*.npy"), key=motion_priority)
    for motion in motions:
        raw_stem = raw_stem_from_motion(motion, object_name)
        raw_template = raw_bvhs / f"{raw_stem}.bvh" if raw_stem else None
        source = manifest.get(raw_template.name) if raw_template else None
        if source is None and raw_template:
            source = infer_native_assets(raw_object_dir, args.extracted_root, raw_template)
        if source is None:
            continue
        ms2 = args.extracted_root / source["object_dir"] / source["ms2_file"]
        manis = args.extracted_root / source["object_dir"] / source["manis_file"]
        if not ms2.is_file() or not manis.is_file() or not raw_template.is_file():
            continue
        output_name = safe_name(object_name)
        return {
            "object_name": object_name,
            "motion_path": str(motion.resolve()),
            "cond_path": str((args.layout_root / "cond.npy").resolve()),
            "tpose_bvh": str(tpose_candidates[0].resolve()),
            "raw_template_bvh": str(raw_template.resolve()),
            "ms2_path": str(ms2.resolve()),
            "manis_path": str(manis.resolve()),
            "asset_mapping_origin": source.get("mapping_origin", "export_manifest"),
            "decoded_bvh_path": str((args.output_root / "decoded_bvhs" / f"{output_name}.bvh").resolve()),
            "output_report": str((args.output_root / "reports" / f"{output_name}.json").resolve()),
        }, None
    return None, {"object_name": object_name, "reason": "no_layout_motion_with_complete_native_assets", "motions_seen": len(motions)}


def run_job(job_path: Path, args: argparse.Namespace, log_path: Path) -> dict:
    command = [
        str(args.blender),
        "--background",
        "--python",
        str(Path(__file__).with_name("validate_planetzoo_anytop_mesh_job.py")),
        "--",
        "--job",
        str(job_path),
        "--cobra-tools",
        str(args.cobra_tools),
        "--motion-lib",
        str(args.motion_lib),
    ]
    environment = os.environ.copy()
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment, check=False)
    log_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
    job = json.loads(job_path.read_text(encoding="utf-8"))
    report_path = Path(job["output_report"])
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {"status": "error", "object_name": job["object_name"], "error": "worker did not write report"}
    report["process_returncode"] = completed.returncode
    report["log_path"] = str(log_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    for path in [args.blender, args.cobra_tools / "__init__.py", args.motion_lib / "BVH.py", args.layout_root / "cond.npy"]:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for name in ["jobs", "reports", "logs", "decoded_bvhs"]:
        (args.output_root / name).mkdir(exist_ok=True)
    with (args.layout_root / "object_index.csv").open("r", encoding="utf-8", newline="") as file:
        objects = list(csv.DictReader(file))
    if args.objects:
        allowed = set(args.objects)
        objects = [row for row in objects if row["object_name"] in allowed]

    jobs, unresolved = [], []
    for row in objects:
        job, issue = resolve_job(row, args)
        if issue:
            unresolved.append(issue)
            continue
        job_path = args.output_root / "jobs" / f"{safe_name(job['object_name'])}.json"
        job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
        report_path = Path(job["output_report"])
        if args.resume and report_path.is_file():
            existing = json.loads(report_path.read_text(encoding="utf-8"))
            if existing.get("status") == "pass":
                continue
        jobs.append(job_path)
    (args.output_root / "unresolved.json").write_text(json.dumps(unresolved, indent=2), encoding="utf-8")
    build_summary = {"requested_objects": len(objects), "resolved_jobs": len(jobs), "unresolved": len(unresolved), "output_root": str(args.output_root)}
    print(json.dumps(build_summary, indent=2), flush=True)
    if args.build_jobs_only:
        return

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(run_job, job_path, args, args.output_root / "logs" / f"{job_path.stem}.log"): job_path
            for job_path in jobs
        }
        for index, future in enumerate(as_completed(futures), 1):
            report = future.result()
            results.append(report)
            print(f"[{index}/{len(jobs)}] {report.get('status')} {report.get('object_name')}", flush=True)
    statuses = {}
    for report in results:
        statuses[report.get("status", "error")] = statuses.get(report.get("status", "error"), 0) + 1
    summary = {**build_summary, "completed": len(results), "statuses": statuses}
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if unresolved or statuses.get("error") or statuses.get("fail"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
