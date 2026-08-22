"""Render all staged reconstruction QC clips as AnyTop RIC-versus-FK GIFs."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged-manifest", required=True, type=Path)
    parser.add_argument("--anytop-output-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--window-radius", type=int, default=5)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=600)
    return parser.parse_args()


def sequence_key(path: Path) -> tuple[int, str]:
    match = re.search(r"_(\d+)$", path.stem)
    return (int(match.group(1)) if match else 0, path.name)


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    if len(value) > 96:
        value = f"{value[:83]}_{hashlib.sha1(value.encode('utf-8')).hexdigest()[:12]}"
    return value or "unnamed"


def build_jobs(records: list[dict], anytop_root: Path, staged_root: Path) -> list[dict]:
    by_owner: dict[str, list[dict]] = {}
    for record in records:
        by_owner.setdefault(record["owner"], []).append(record)
    jobs = []
    for owner, owner_records in by_owner.items():
        staged_bvhs = sorted(
            (staged_root / f"{owner}_ovl" / "raw_bvhs").glob("*.bvh"), key=lambda path: path.name.lower()
        )
        staged_bvhs = [path for path in staged_bvhs if "tpos" not in path.stem.lower()]
        object_name = f"PZ_{owner}"
        motions = sorted((anytop_root / object_name / "motions").glob("*.npy"), key=sequence_key)
        lookup = {Path(record["staged_bvh"]).name: record for record in owner_records}
        if len(staged_bvhs) != len(motions):
            raise RuntimeError(f"{owner}: staged BVH count {len(staged_bvhs)} != AnyTop motion count {len(motions)}")
        for bvh, motion in zip(staged_bvhs, motions):
            record = lookup.get(bvh.name)
            if record is None:
                raise KeyError(f"{owner}: no manifest record for {bvh.name}")
            jobs.append({**record, "object_name": object_name, "layout_root": str(anytop_root / object_name), "motion": str(motion)})
    return jobs


def render_one(index: int, total: int, job: dict, args: argparse.Namespace, renderer: Path) -> dict:
    stem = safe_name(f"{index:03d}_{job['owner']}_{job['action_name']}")
    gif = args.output_dir / "gifs" / f"{stem}.gif"
    report = args.output_dir / "reports" / f"{stem}.json"
    frame = max(0, int(job["trigger_frame_raw"]) - 1)
    command = [
        args.python, str(renderer),
        "--layout-root", job["layout_root"],
        "--object-name", job["object_name"],
        "--motion", job["motion"],
        "--joint", job["trigger_joint_name"],
        "--frame-start", str(max(0, frame - args.window_radius)),
        "--frame-end", str(frame + args.window_radius),
        "--output-gif", str(gif),
        "--output-report", str(report),
        "--width", str(args.width),
        "--height", str(args.height),
    ]
    gif.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    item = {**job, "index": index, "gif": str(gif), "report": str(report), "returncode": result.returncode}
    if result.returncode == 0 and gif.is_file() and report.is_file():
        item["status"] = "ok"
        item["anytop_report"] = json.loads(report.read_text(encoding="utf-8"))
    else:
        item.update(status="error", stdout=result.stdout, stderr=result.stderr)
    return item


def write_index(output_dir: Path, results: list[dict]) -> None:
    figures = []
    for item in results:
        label = f"{item['index']:03d} | {item['severity']} | {item['object_name']} | {item['action_name']} | {item['trigger_joint_name']}"
        if item["status"] == "ok":
            figures.append(f"<figure><img src=\"{Path(item['gif']).relative_to(output_dir).as_posix()}\"><figcaption>{label}</figcaption></figure>")
        else:
            figures.append(f"<figure><figcaption>FAILED: {label}</figcaption></figure>")
    html = """<!doctype html><html><head><meta charset=\"utf-8\"><title>Reconstructed AnyTop QC</title>
<style>body{font-family:Arial;margin:18px;background:#f5f5f5}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:16px}figure{margin:0;background:#fff;padding:10px;border:1px solid #ddd}img{width:100%;display:block}figcaption{margin-top:8px}</style>
</head><body><h1>Direct no-IK reconstruction QC</h1><p>Left: AnyTop RIC position channels. Right: AnyTop rot6d FK. Red: scanner trigger chain.</p><div class=\"grid\">"""
    html += "".join(figures) + "</div></body></html>"
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = [json.loads(line) for line in args.staged_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    jobs = build_jobs(records, args.anytop_output_root, args.staged_manifest.parent)
    jobs.sort(key=lambda item: ({"severe": 2, "candidate": 1, "borderline": 0}[item["severity"]], item["owner"], item["action_name"]), reverse=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    renderer = Path(__file__).with_name("render_anytop_spike_window.py")
    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(render_one, index, len(jobs), job, args, renderer) for index, job in enumerate(jobs, 1)]
        results = [future.result() for future in cf.as_completed(futures)]
    results.sort(key=lambda item: item["index"])
    summary = {
        "requested": len(jobs),
        "ok": sum(item["status"] == "ok" for item in results),
        "errors": sum(item["status"] == "error" for item in results),
        "results": results,
    }
    (args.output_dir / "render_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_index(args.output_dir, results)
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
