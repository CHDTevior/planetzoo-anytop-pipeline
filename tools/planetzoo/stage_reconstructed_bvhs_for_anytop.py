"""Stage selected reconstructed BVHs with their original T-pose for AnyTop."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flagged-jsonl", required=True, type=Path)
    parser.add_argument("--source-raw-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--severities", default="severe,candidate")
    return parser.parse_args()


def rank(record: dict) -> tuple[int, float]:
    return ({"severe": 2, "candidate": 1, "borderline": 0}[record["severity"]], float(record.get("score", 0.0)))


def find_tpose(raw_bvhs: Path) -> Path:
    candidates = sorted(raw_bvhs.glob("*__tpos.bvh")) or sorted(raw_bvhs.glob("*tpos*.bvh"))
    if not candidates:
        raise FileNotFoundError(f"No T-pose BVH under {raw_bvhs}")
    return candidates[0]


def main() -> None:
    args = parse_args()
    selected_severities = {value.strip() for value in args.severities.split(",") if value.strip()}
    records = [
        json.loads(line)
        for line in args.flagged_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("severity") in selected_severities
    ]
    records.sort(key=rank, reverse=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    copied_tposes: set[str] = set()
    for record in records:
        owner = record["reconstruction_owner"]
        destination = args.output_root / f"{owner}_ovl" / "raw_bvhs"
        destination.mkdir(parents=True, exist_ok=True)
        source_raw_bvhs = args.source_raw_root / f"{owner}_ovl" / "raw_bvhs"
        tpose = find_tpose(source_raw_bvhs)
        if owner not in copied_tposes:
            shutil.copy2(tpose, destination / tpose.name)
            copied_tposes.add(owner)
        source_bvh = Path(record["reconstruction_bvh"])
        target_bvh = destination / source_bvh.name
        shutil.copy2(source_bvh, target_bvh)
        manifest.append({
            "owner": owner,
            "severity": record["severity"],
            "action_name": record["reconstruction_action"],
            "trigger_joint_name": record["trigger_joint_name"],
            "trigger_frame_raw": record["trigger_frame_raw"],
            "source_bvh": str(source_bvh),
            "tpose_source": str(tpose),
            "staged_bvh": str(target_bvh),
        })
    manifest_path = args.output_root / "staged_candidates_manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in manifest), encoding="utf-8")
    summary = {
        "severities": sorted(selected_severities),
        "motions": len(manifest),
        "owners": len(copied_tposes),
        "manifest": str(manifest_path),
    }
    (args.output_root / "staged_candidates_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
