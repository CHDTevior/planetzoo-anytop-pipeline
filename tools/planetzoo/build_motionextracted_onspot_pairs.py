"""Build strict MANIS pairs for offline motion-extracted locomotion recovery.

Planet Zoo stores a moving ``animationmotionextracted*`` clip alongside a
co-timed ``animationnotmotionextracted*onspot`` clip in the same asset OVL.
For the dtype-38 locomotion mode, the latter supplies stable local limb
rotations while the former supplies root/trunk motion.  This script only emits
pairs whose owner, category, canonical action name, frame count and raw dtype
match that contract; it never guesses a semantic match.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SOURCE_RE = re.compile(
    r"animation(?P<not>not)?motionextracted(?P<category>behaviour|fighting|locomotion)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cobra-tools", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--owner", action="append", default=[], help="Optional .ovl directory name, repeatable.")
    return parser.parse_args()


def canonical_action(local_name: str) -> str:
    name = local_name.lower()
    name = re.sub(r"onspot$", "", name)
    return re.sub(r"turn[lr](?:\d{3})?$", "", name)


def scan_file(path: Path, input_root: Path) -> tuple[list[dict], dict | None]:
    match = SOURCE_RE.search(path.name)
    if match is None:
        return [], None
    try:
        from generated.formats.manis import ManisFile  # pylint: disable=import-outside-toplevel

        manis = ManisFile()
        manis.load(path)
        owner = path.parent.name.removesuffix(".ovl")
        common = {
            "owner": owner,
            "source_path": path.relative_to(input_root).as_posix(),
            "source_kind": "not_motion_extracted" if match.group("not") else "motion_extracted",
            "category": match.group("category").lower(),
        }
        rows = []
        for info in manis.mani_infos:
            local = info.name.rsplit("@", 1)[-1]
            rows.append(
                {
                    **common,
                    "action_name": info.name,
                    "local_action": local,
                    "canonical_action": canonical_action(local),
                    "frame_count": int(info.frame_count),
                    "dtype_raw": int(info.dtype._value),
                }
            )
        return rows, None
    except Exception as exc:  # Keep the resulting manifest auditable.
        return [], {"source_path": path.relative_to(input_root).as_posix(), "error": repr(exc)}


def main() -> None:
    args = parse_args()
    import sys

    sys.path.insert(0, str(args.cobra_tools))
    logging.disable(logging.CRITICAL)
    paths = sorted(args.input_root.rglob("*.manis"))
    if args.owner:
        allowed = {name.removesuffix(".ovl") for name in args.owner}
        paths = [path for path in paths if path.parent.name.removesuffix(".ovl") in allowed]

    rows: list[dict] = []
    errors: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(scan_file, path, args.input_root) for path in paths]
        for future in as_completed(futures):
            found, error = future.result()
            rows.extend(found)
            if error:
                errors.append(error)

    donors: dict[tuple[str, str, str, int], list[dict]] = {}
    for row in rows:
        if row["source_kind"] == "not_motion_extracted" and row["dtype_raw"] == 36:
            key = (row["owner"], row["category"], row["canonical_action"], row["frame_count"])
            donors.setdefault(key, []).append(row)

    pairs: list[dict] = []
    skipped: dict[str, int] = {}
    for target in rows:
        if target["source_kind"] != "motion_extracted" or target["dtype_raw"] != 38:
            continue
        key = (target["owner"], target["category"], target["canonical_action"], target["frame_count"])
        matches = donors.get(key, [])
        if len(matches) != 1:
            reason = "missing_clean_onspot" if not matches else "ambiguous_clean_onspot"
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        donor = matches[0]
        pairs.append(
            {
                "owner": target["owner"],
                "category": target["category"],
                "canonical_action": target["canonical_action"],
                "frame_count": target["frame_count"],
                "target": target,
                "donor": donor,
                "reconstruction": {
                    "root_and_trunk": "target motion_extracted action",
                    "independent_limb_branches": "donor not_motion_extracted/onspot action",
                    "selector": "LimbTrackData countb + MS2 parent hierarchy",
                },
            }
        )

    pairs.sort(key=lambda row: (row["owner"], row["category"], row["target"]["action_name"]))
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    summary = {
        "input_root": str(args.input_root),
        "scanned_manis_files": len(paths),
        "scanned_actions": len(rows),
        "parse_errors": errors,
        "strict_dtype38_dtype36_pairs": len(pairs),
        "pairs_by_category": {
            category: sum(pair["category"] == category for pair in pairs)
            for category in ("locomotion", "behaviour", "fighting")
        },
        "skipped_dtype38_motionextracted": skipped,
        "path_format": "relative to input_root, POSIX separators",
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
