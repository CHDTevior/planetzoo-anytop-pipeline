"""Run strict offline motion-extracted reconstruction pairs through Blender.

The input JSONL must be created by ``build_motionextracted_onspot_pairs.py``.
Each entry is handled in an isolated Blender process so an import failure is
recorded per clip and cannot contaminate another character's actions.
"""

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
    parser.add_argument("--pairs-manifest", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--cobra-tools", required=True, type=Path)
    parser.add_argument("--blender", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--owner", action="append", default=[], help="Optional owner name, repeatable.")
    parser.add_argument(
        "--target-action",
        action="append",
        default=[],
        help="Optional exact motion-extracted action name, repeatable.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum clips to reconstruct.")
    parser.add_argument("--workers", type=int, default=1, help="Independent Blender processes to run concurrently.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return re.sub(r"_+", "_", value) or "unnamed"


def find_ms2(input_root: Path, owner: str) -> Path:
    owner_dir = input_root / f"{owner}.ovl"
    candidates = sorted(owner_dir.glob("*.ms2"))
    owner_lower = owner.lower()
    preferred = [path for path in candidates if path.stem.lower().rstrip("_") == owner_lower]
    if len(preferred) == 1:
        return preferred[0]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"could not resolve exactly one MS2 for {owner}: {candidates}")


def load_pairs(path: Path, owners: set[str], target_actions: set[str]) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if owners and row["owner"] not in owners:
            continue
        target = row["target"]
        if target_actions and target["action_name"] not in target_actions:
            continue
        donor = row["donor"]
        if not (
            target["source_kind"] == "motion_extracted"
            and target["dtype_raw"] == 38
            and donor["source_kind"] == "not_motion_extracted"
            and donor["dtype_raw"] == 36
            and target["frame_count"] == donor["frame_count"]
        ):
            raise ValueError(f"manifest row violates strict reconstruction contract: {target['action_name']}")
        rows.append(row)
    return rows


def output_stem(row: dict) -> str:
    target = row["target"]
    stem = safe_name("__".join((row["owner"], target["category"], Path(target["source_path"]).stem, target["action_name"])))
    # Blender's embedded Python may still observe MAX_PATH on Windows. Keep
    # per-clip filenames short enough for deep output roots while retaining a
    # deterministic suffix that prevents collisions between long action names.
    if len(stem) > 96:
        digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:12]
        stem = f"{stem[:83]}_{digest}"
    return stem


def reconstruct_one(index: int, total: int, row: dict, args: argparse.Namespace, runner: Path) -> dict:
    started = time.time()
    target = row["target"]
    donor = row["donor"]
    stem = output_stem(row)
    owner_dir = args.output_root / safe_name(row["owner"])
    bvh_path = owner_dir / "bvhs" / f"{stem}.bvh"
    report_path = owner_dir / "reports" / f"{stem}.json"
    positions_path = owner_dir / "positions" / f"{stem}.npz"
    record = {
        "index": index,
        "total": total,
        "owner": row["owner"],
        "target_action": target["action_name"],
        "donor_action": donor["action_name"],
        "target_source_path": target["source_path"],
        "donor_source_path": donor["source_path"],
        "bvh": str(bvh_path),
        "report": str(report_path),
        "positions": str(positions_path),
    }
    try:
        if args.skip_existing and bvh_path.is_file() and report_path.is_file() and positions_path.is_file():
            record["status"] = "skipped_existing"
        else:
            ms2_path = find_ms2(args.input_root, row["owner"])
            cmd = [
                str(args.blender), "--background", "--python", str(runner), "--",
                "--cobra-tools", str(args.cobra_tools),
                "--ms2-path", str(ms2_path),
                "--extracted-manis", str(args.input_root / Path(target["source_path"])),
                "--extracted-action", target["action_name"],
                "--onspot-manis", str(args.input_root / Path(donor["source_path"])),
                "--onspot-action", donor["action_name"],
                "--use-limb-branches",
                "--output-position-npz", str(positions_path),
                "--output-report", str(report_path),
                "--output-bvh", str(bvh_path),
            ]
            record["command"] = cmd
            if args.dry_run:
                record["status"] = "dry_run"
            else:
                for output_path in (bvh_path, report_path, positions_path):
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                log_path = owner_dir / "logs" / f"{stem}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                result = subprocess.run(cmd, text=True, capture_output=True, check=False)
                log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8", errors="ignore")
                record["log"] = str(log_path)
                record["returncode"] = result.returncode
                record["status"] = "ok" if result.returncode == 0 and bvh_path.is_file() and report_path.is_file() else "error"
    except Exception as exc:
        record["status"] = "error"
        record["error"] = repr(exc)
    record["seconds"] = round(time.time() - started, 3)
    return record


def main() -> None:
    args = parse_args()
    runner = Path(__file__).with_name("evaluate_motionextracted_onspot_hybrid.py")
    pairs = load_pairs(args.pairs_manifest, set(args.owner), set(args.target_action))
    if args.limit is not None:
        pairs = pairs[: args.limit]

    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / "reconstruction_status.jsonl"
    summary_path = args.output_root / "reconstruction_summary.json"
    records = []
    with status_path.open("a", encoding="utf-8", newline="\n") as status_file:
        with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [
                executor.submit(reconstruct_one, index, len(pairs), row, args, runner)
                for index, row in enumerate(pairs, start=1)
            ]
            for future in cf.as_completed(futures):
                record = future.result()
                status_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                status_file.flush()
                records.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)

    summary = {
        "pairs_requested": len(pairs),
        "ok": sum(record["status"] == "ok" for record in records),
        "skipped_existing": sum(record["status"] == "skipped_existing" for record in records),
        "dry_run": sum(record["status"] == "dry_run" for record in records),
        "errors": sum(record["status"] == "error" for record in records),
        "status_jsonl": str(status_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
