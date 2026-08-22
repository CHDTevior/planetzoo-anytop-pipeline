"""Directly decode MANIS actions in isolated Blender processes with IK off."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-manifest", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--cobra-tools", required=True, type=Path)
    parser.add_argument("--blender", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return re.sub(r"_+", "_", value) or "unnamed"


def find_ms2(input_root: Path, owner: str) -> Path:
    candidates = sorted((input_root / f"{owner}.ovl").glob("*.ms2"))
    preferred = [path for path in candidates if path.stem.lower().rstrip("_") == owner.lower()]
    if len(preferred) == 1:
        return preferred[0]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"could not resolve exactly one MS2 for {owner}: {candidates}")


def output_stem(target: dict) -> str:
    stem = safe_name("__".join((target["owner"], Path(target["source_path"]).stem, target["action_name"])))
    if len(stem) > 96:
        stem = f"{stem[:83]}_{hashlib.sha1(stem.encode('utf-8')).hexdigest()[:12]}"
    return stem


def reconstruct_one(index: int, total: int, target: dict, args: argparse.Namespace, evaluator: Path) -> dict:
    started = time.time()
    stem = output_stem(target)
    owner_dir = args.output_root / safe_name(target["owner"])
    bvh = owner_dir / "bvhs" / f"{stem}.bvh"
    report = owner_dir / "reports" / f"{stem}.json"
    positions = owner_dir / "positions" / f"{stem}.npz"
    record = {**target, "index": index, "total": total, "bvh": str(bvh), "report": str(report), "positions": str(positions)}
    try:
        if args.skip_existing and all(path.is_file() for path in (bvh, report, positions)):
            record["status"] = "skipped_existing"
        else:
            for path in (bvh, report, positions):
                path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                str(args.blender), "--background", "--python", str(evaluator), "--",
                "--cobra-tools", str(args.cobra_tools),
                "--ms2-path", str(find_ms2(args.input_root, target["owner"])),
                "--extracted-manis", str(args.input_root / Path(target["source_path"])),
                "--extracted-action", target["action_name"],
                "--output-position-npz", str(positions),
                "--output-report", str(report),
                "--output-bvh", str(bvh),
            ]
            log = owner_dir / "logs" / f"{stem}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            log.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8", errors="ignore")
            record.update(command=command, log=str(log), returncode=result.returncode)
            record["status"] = "ok" if result.returncode == 0 and all(path.is_file() for path in (bvh, report, positions)) else "error"
    except Exception as exc:
        record.update(status="error", error=repr(exc))
    record["seconds"] = round(time.time() - started, 3)
    return record


def main() -> None:
    args = parse_args()
    targets = [json.loads(line) for line in args.targets_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        targets = targets[:args.limit]
    evaluator = Path(__file__).with_name("evaluate_motionextracted_onspot_hybrid.py")
    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / "reconstruction_status.jsonl"
    records = []
    with status_path.open("w", encoding="utf-8", newline="\n") as handle:
        with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [
                executor.submit(reconstruct_one, index, len(targets), target, args, evaluator)
                for index, target in enumerate(targets, start=1)
            ]
            for future in cf.as_completed(futures):
                record = future.result()
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                records.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
    summary = {
        "requested": len(targets),
        "ok": sum(record["status"] == "ok" for record in records),
        "errors": sum(record["status"] == "error" for record in records),
        "status_jsonl": str(status_path),
    }
    (args.output_root / "reconstruction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
