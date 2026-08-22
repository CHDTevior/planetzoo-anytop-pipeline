"""Run the proximal-jump QC on BVHs produced by offline reconstruction."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.planetzoo.scan_raw_bvh_proximal_rotation_qc import scan_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction-status", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-descendants", type=int, default=4)
    parser.add_argument("--borderline-threshold", type=float, default=55.0)
    parser.add_argument("--candidate-threshold", type=float, default=80.0)
    parser.add_argument("--severe-jump-threshold", type=float, default=150.0)
    parser.add_argument("--severe-accel-threshold", type=float, default=120.0)
    return parser.parse_args()


def severity(result: dict, args: argparse.Namespace) -> str | None:
    if result.get("flag") not in {"candidate", "borderline"}:
        return None
    if (
        result.get("max_proximal_jump_deg", 0.0) >= args.severe_jump_threshold
        or result.get("max_proximal_accel_deg", 0.0) >= args.severe_accel_threshold
    ):
        return "severe"
    return "candidate" if result["flag"] == "candidate" else "borderline"


def main() -> None:
    args = parse_args()
    records = [
        json.loads(line)
        for line in args.reconstruction_status.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_path = args.output_dir / "proximal_rotation_qc_all_results.jsonl"
    flagged_path = args.output_dir / "proximal_rotation_qc_flagged.jsonl"
    results = []
    with all_path.open("w", encoding="utf-8", newline="\n") as all_file, flagged_path.open("w", encoding="utf-8", newline="\n") as flagged_file:
        for record in records:
            if record.get("status") not in {"ok", "skipped_existing"}:
                result = {**record, "status": "reconstruction_error"}
            else:
                row = {
                    "raw_bvh": record["bvh"],
                    "object_name": f"PZ_{record['owner']}",
                    "motion_file": Path(record["bvh"]).name,
                    "processed_motion": None,
                    "reconstruction_owner": record["owner"],
                    "reconstruction_action": record["action_name"],
                    "source_path": record["source_path"],
                    "reconstruction_bvh": record["bvh"],
                    "legacy_label_count": record["legacy_label_count"],
                }
                result = scan_one(row, args.min_descendants, args.borderline_threshold, args.candidate_threshold)
                level = severity(result, args)
                if level is not None:
                    result["severity"] = level
            all_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            if result.get("severity"):
                flagged_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            results.append(result)

    severity_counts = Counter(result.get("severity", "clean") for result in results)
    summary = {
        "reconstruction_status": str(args.reconstruction_status),
        "samples": len(results),
        "status_counts": dict(Counter(result.get("status", "missing") for result in results)),
        "flag_counts": dict(Counter(result.get("flag", "none") for result in results)),
        "severity_counts": dict(severity_counts),
        "flagged_total": sum(value for key, value in severity_counts.items() if key != "clean"),
        "all_results_jsonl": str(all_path),
        "flagged_jsonl": str(flagged_path),
    }
    (args.output_dir / "proximal_rotation_qc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
