"""Audit whether legacy flagged motions were exported from their named MANIS.

The legacy exporter could retain fake-user Blender Actions between MANIS files.
This tool compares every rendered QC sample against the action names declared in
the original MANIS files, and reports whether an action was instead declared by
another MANIS in the same object directory.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


RAW_STEM_RE = re.compile(
    r"^(?P<animal>.+?)__(?P<group>animation(?:not)?motionextracted[a-z_]+)_"
    r"maniset(?P<maniset>[0-9a-f]+)__(?P<action>.+)$"
)


def safe_name(value: str) -> str:
    value = value.replace("@", "_")
    value = value.replace(".", "_")
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cobra-tools", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--layout-manifest", required=True, type=Path)
    parser.add_argument("--render-summary", required=True, type=Path, action="append")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_layout_index(path: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            motion_file = row.get("motion_file")
            if motion_file:
                index[motion_file] = row
    return index


def load_declared_actions(path: Path) -> list[dict]:
    from generated.formats.manis import ManisFile

    manis = ManisFile()
    manis.load(path)
    return [
        {
            "name": info.name,
            "safe_name": safe_name(info.name),
            "dtype_raw": int(info.dtype._value),
            "frame_count": int(info.frame_count),
        }
        for info in manis.mani_infos
    ]


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.cobra_tools))
    layout_index = load_layout_index(args.layout_manifest)
    declared_cache: dict[Path, list[dict]] = {}

    def declared(path: Path) -> list[dict]:
        if path not in declared_cache:
            declared_cache[path] = load_declared_actions(path)
        return declared_cache[path]

    rows: list[dict] = []
    for summary_path in args.render_summary:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for report in summary["reports"]:
            legacy = layout_index.get(report["motion_file"])
            if legacy is None:
                rows.append({
                    "motion_file": report["motion_file"],
                    "severity": report["severity"],
                    "status": "missing_layout_manifest_row",
                })
                continue
            raw_stem = legacy["raw_bvh_stem"]
            match = RAW_STEM_RE.match(raw_stem)
            if match is None:
                rows.append({
                    "motion_file": report["motion_file"],
                    "severity": report["severity"],
                    "raw_bvh_stem": raw_stem,
                    "status": "unparseable_raw_stem",
                })
                continue
            named_manis = (
                args.source_root
                / f"{legacy['object_key']}.ovl"
                / f"{match['group']}.maniset{match['maniset']}.manis"
            )
            safe_action_name = match["action"]
            record = {
                "motion_file": report["motion_file"],
                "severity": report["severity"],
                "score": report["score"],
                "trigger_joint_name": report["trigger_joint_name"],
                "max_proximal_jump_deg": report["max_proximal_jump_deg"],
                "object_key": legacy["object_key"],
                "raw_bvh_stem": raw_stem,
                "named_manis": str(named_manis),
                "expected_safe_action_name": safe_action_name,
            }
            if not named_manis.exists():
                record["status"] = "named_manis_missing"
                rows.append(record)
                continue
            named_actions = [
                action for action in declared(named_manis) if action["safe_name"] == safe_action_name
            ]
            if named_actions:
                record["expected_action"] = named_actions[0]["name"]
                record["declared_dtype_raw"] = named_actions[0]["dtype_raw"]
                record["declared_frame_count"] = named_actions[0]["frame_count"]
                record["status"] = "declared_in_named_manis"
                rows.append(record)
                continue
            alternate_manis = []
            for candidate in sorted(named_manis.parent.glob("*.manis")):
                if candidate == named_manis:
                    continue
                matches = [
                    action for action in declared(candidate) if action["safe_name"] == safe_action_name
                ]
                if matches:
                    alternate_manis.append(
                        {
                            "manis_file": candidate.name,
                            "action_name": matches[0]["name"],
                            "dtype_raw": matches[0]["dtype_raw"],
                            "frame_count": matches[0]["frame_count"],
                        }
                    )
            record["alternate_manis"] = alternate_manis
            record["status"] = (
                "declared_in_other_manis" if alternate_manis else "action_not_declared_for_object"
            )
            rows.append(record)

    counts = Counter(row["status"] for row in rows)
    severity_counts = {
        severity: dict(Counter(row["status"] for row in rows if row.get("severity") == severity))
        for severity in sorted({row.get("severity") for row in rows if row.get("severity")})
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "sample_count": len(rows),
                "status_counts": dict(counts),
                "severity_status_counts": severity_counts,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"sample_count": len(rows), "status_counts": dict(counts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
