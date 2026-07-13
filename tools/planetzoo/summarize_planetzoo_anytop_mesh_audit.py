"""Summarize all per-topology AnyTop-to-MS2 mesh validation reports."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument("--layout-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.audit_root / "full_topology_mesh_skinning_report.json"
    with (args.layout_root / "object_index.csv").open("r", encoding="utf-8", newline="") as file:
        expected = [row["object_name"] for row in csv.DictReader(file)]
    reports = {}
    for path in (args.audit_root / "reports").glob("*.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("object_name"):
            reports[report["object_name"]] = report

    missing = sorted(set(expected) - set(reports))
    statuses = Counter(report.get("status", "error") for report in reports.values())
    passing = [report for report in reports.values() if report.get("status") == "pass"]
    matrix_errors = [float(report["max_matrix_abs_error"]) for report in passing]
    shared_bones = [int(report["shared_bones"]) for report in passing]
    target_bones = [int(report["target_bones"]) for report in passing]
    mesh_vertices = [
        int(mesh["vertices"])
        for report in passing
        for sample in report.get("samples", [])[:1]
        for mesh in sample.get("meshes", {}).values()
    ]
    resources = [
        {
            "object_name": report["object_name"],
            "ms2_path": report["ms2_path"],
            "manis_path": report["manis_path"],
            "raw_template_bvh": report["raw_template_bvh"],
            "tpose_bvh": json.loads((args.audit_root / "jobs" / f"{report['object_name']}.json").read_text(encoding="utf-8"))["tpose_bvh"],
            "validation_report": str(args.audit_root / "reports" / f"{report['object_name']}.json"),
        }
        for report in sorted(passing, key=lambda value: value["object_name"])
    ]
    resource_manifest = args.audit_root / "validated_topology_resources.jsonl"
    resource_manifest.write_text("".join(json.dumps(item) + "\n" for item in resources), encoding="utf-8")
    summary = {
        "expected_topologies": len(expected),
        "reported_topologies": len(reports),
        "missing_reports": missing,
        "statuses": dict(statuses),
        "all_topologies_passed": len(passing) == len(expected) and not missing and not (set(statuses) - {"pass"}),
        "max_bone_matrix_abs_error": max(matrix_errors) if matrix_errors else None,
        "shared_bones_range": [min(shared_bones), max(shared_bones)] if shared_bones else None,
        "target_bones_range": [min(target_bones), max(target_bones)] if target_bones else None,
        "lod0_mesh_vertex_count_range": [min(mesh_vertices), max(mesh_vertices)] if mesh_vertices else None,
        "validated_topology_resources": str(resource_manifest),
    }
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["all_topologies_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
