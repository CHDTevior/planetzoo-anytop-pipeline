"""Resolve old QC labels to correct MANIS actions lacking a strict on-spot donor.

The legacy raw-BVH export could attach an Action from a different MANIS file to
the current file name.  This tool uses the provenance audit rather than those
file names, deduplicates labels that point to the same true Action, and removes
only actions covered by the exact motionextracted/onspot pair manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--pairs-manifest", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    return parser.parse_args()


def posix_relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def action_source(row: dict, input_root: Path) -> tuple[str, str]:
    """Return the MANIS-relative path and exact declared Action name."""
    if row["status"] == "declared_in_named_manis":
        return posix_relative(Path(row["named_manis"]), input_root), row["expected_action"]
    alternate = row["alternate_manis"][0]
    return (
        f"{row['object_key']}.ovl/{alternate['manis_file']}",
        alternate["action_name"],
    )


def severity_rank(severity: str) -> int:
    return {"severe": 2, "borderline": 1}.get(severity, 0)


def main() -> None:
    args = parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    paired_actions = set()
    for line in args.pairs_manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            pair = json.loads(line)
            target = pair["target"]
            paired_actions.add((pair["owner"], Path(target["source_path"]).as_posix(), target["action_name"]))

    grouped: dict[tuple[str, str, str], dict] = {}
    for row in audit["rows"]:
        source_path, action_name = action_source(row, args.input_root)
        key = (row["object_key"], source_path, action_name)
        current = grouped.get(key)
        if current is None:
            grouped[key] = {
                "owner": row["object_key"],
                "source_path": source_path,
                "action_name": action_name,
                "severity": row["severity"],
                "max_legacy_score": float(row["score"]),
                "max_legacy_proximal_jump_deg": float(row["max_proximal_jump_deg"]),
                "trigger_joints": [row["trigger_joint_name"]],
                "legacy_motion_files": [row["motion_file"]],
                "legacy_label_count": 1,
            }
            continue
        current["legacy_label_count"] += 1
        current["legacy_motion_files"].append(row["motion_file"])
        if row["trigger_joint_name"] not in current["trigger_joints"]:
            current["trigger_joints"].append(row["trigger_joint_name"])
        if severity_rank(row["severity"]) > severity_rank(current["severity"]):
            current["severity"] = row["severity"]
        current["max_legacy_score"] = max(current["max_legacy_score"], float(row["score"]))
        current["max_legacy_proximal_jump_deg"] = max(
            current["max_legacy_proximal_jump_deg"], float(row["max_proximal_jump_deg"])
        )

    direct = [
        record for key, record in grouped.items()
        if key not in paired_actions
    ]
    direct.sort(
        key=lambda record: (
            -severity_rank(record["severity"]),
            -record["max_legacy_score"],
            record["owner"],
            record["action_name"],
        )
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in direct), encoding="utf-8"
    )
    summary = {
        "legacy_labels": len(audit["rows"]),
        "unique_true_actions": len(grouped),
        "strict_pair_covered_actions": len(grouped) - len(direct),
        "direct_noik_actions": len(direct),
        "severity_counts": {
            severity: sum(record["severity"] == severity for record in direct)
            for severity in ("severe", "borderline")
        },
        "output_jsonl": str(args.output_jsonl),
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
