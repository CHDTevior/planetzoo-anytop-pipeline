"""Materialise one verified Planet Zoo skinning rig package per minipack species.

The package contains only the files needed to visualise a generated AnyTop
motion: one MS2 mesh, its matching MANIS, a T-pose BVH, a reference action BVH,
and the object's full skeleton condition.  It does not copy the full extracted
game archive or the AniMo4D training motions.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np


ASSET_FILES = {
    "ms2_path": "model.ms2",
    "manis_path": "reference_action.manis",
    "tpose_bvh": "tpose.bvh",
    "raw_template_bvh": "reference_action.bvh",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-index", required=True, type=Path, help="validated_topology_resources.jsonl from the mesh audit.")
    parser.add_argument("--full-cond-path", required=True, type=Path, help="Full AniMo4D cond.npy.")
    parser.add_argument("--minipack-root", required=True, type=Path, help="Downloaded anytop13-animal-minipack root.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", choices=["hardlink", "copy"], default="hardlink")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def json_compatible(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    return value


def materialise(source: Path, destination: Path, mode: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        if destination.is_file() and destination.stat().st_size == source.stat().st_size:
            return
        raise FileExistsError(f"Refusing to overwrite a different file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            # Cross-volume roots cannot hardlink.  Copying retains the same
            # package layout and is the safe fallback.
            pass
    shutil.copy2(source, destination)


def write_readme(root: Path, count: int) -> None:
    root.joinpath("README.md").write_text(
        "# AnyTop-13 Planet Zoo Skinning Resources\n\n"
        f"This private package contains {count} verified Planet Zoo rig resources for visualising "
        "generated motions from `Tevior/anytop13-animal-minipack`.\n\n"
        "Each `rigs/<object_type>/` directory contains:\n\n"
        "- `model.ms2`: original weighted mesh.\n"
        "- `reference_action.manis`: matching game animation container, used only to initialise the rig.\n"
        "- `tpose.bvh`: exported rest skeleton.\n"
        "- `reference_action.bvh`: matching action template for the deterministic AnyTop inverse.\n"
        "- `full_skeleton.json`: full joint order, parent graph, offsets, chains and rest tensor.\n"
        "- `asset_manifest.json`: portable file mapping for this rig.\n\n"
        "Use the companion `planetzoo-anytop-pipeline` GitHub repository. First expand a raw minipack "
        "prediction with `expand_minipack_motion_to_full_rig.py`, then run "
        "`build_planetzoo_anytop_npy_skinning_poc.py` against the five files above.\n\n"
        "## Licence / access\n\n"
        "These resources are derived from locally extracted commercial Planet Zoo game assets. Keep this "
        "repository private and grant access only to users authorised to use those assets. Do not republish "
        "the mesh, skin weights or original animation files publicly.\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    species_table = json.loads((args.minipack_root / "SPECIES_TABLE.json").read_text(encoding="utf-8"))
    expected = [row["object_type"] for row in species_table["species"]]
    if len(expected) != len(set(expected)):
        raise ValueError("The minipack species table contains duplicate object types.")
    index = {row["object_name"]: row for row in read_jsonl(args.resource_index)}
    missing = sorted(set(expected) - set(index))
    if missing:
        raise ValueError(f"The resource index lacks {len(missing)} minipack objects, first: {missing[:8]}")
    cond = np.load(args.full_cond_path, allow_pickle=True).item()
    missing_cond = sorted(set(expected) - set(cond))
    if missing_cond:
        raise ValueError(f"The full cond lacks {len(missing_cond)} objects, first: {missing_cond[:8]}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for object_name in expected:
        source = index[object_name]
        rig_dir = args.output_root / "rigs" / object_name
        file_map: dict[str, str] = {}
        for source_key, target_name in ASSET_FILES.items():
            materialise(Path(source[source_key]), rig_dir / target_name, args.mode)
            file_map[source_key.removesuffix("_path")] = target_name

        entry = json_compatible(dict(cond[object_name]))
        entry["object_type"] = object_name
        skeleton_path = rig_dir / "full_skeleton.json"
        skeleton_path.write_text(json.dumps(entry, indent=2), encoding="utf-8")
        rig_manifest = {
            "object_name": object_name,
            "rig_dir": f"rigs/{object_name}",
            "files": {**file_map, "full_skeleton": "full_skeleton.json"},
            "asset_mapping_origin": source.get("asset_mapping_origin", "mesh_skinning_audit"),
            "source_filenames": {key: Path(source[key]).name for key in ASSET_FILES},
        }
        (rig_dir / "asset_manifest.json").write_text(json.dumps(rig_manifest, indent=2), encoding="utf-8")
        manifest.append(rig_manifest)

    (args.output_root / "rig_manifest.jsonl").write_text(
        "\n".join(json.dumps(row) for row in manifest) + "\n", encoding="utf-8"
    )
    write_readme(args.output_root, len(manifest))
    total_bytes = sum(path.stat().st_size for path in args.output_root.rglob("*") if path.is_file())
    print(
        json.dumps(
            {
                "package_root": str(args.output_root),
                "species": len(manifest),
                "mode": args.mode,
                "bytes": total_bytes,
                "gigabytes": round(total_bytes / (1024**3), 3),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
