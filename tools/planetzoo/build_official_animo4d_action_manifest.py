"""Resolve every official AniMo4D caption to a declared source MANIS action.

The historical AniMo4D exporter could leave Blender Actions from a previous
MANIS file in memory.  Its raw-BVH filename is therefore not sufficient proof
of the Action that was sampled.  This tool makes the correspondence explicit
before any new no-IK export is attempted.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


RAW_STEM_RE = re.compile(
    r"^(?P<animal>.+?)__(?P<group>animation(?:not)?motionextracted[a-z_]*)_"
    r"maniset(?P<maniset>[0-9a-f]+)__(?P<action>.+)$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-manifest", required=True, type=Path)
    parser.add_argument("--mani-inventory", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--raw-alias-manifest", type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--unavailable-output-jsonl", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    return parser.parse_args()


def normalized_owner(value: str) -> str:
    return value.removeprefix("PZ_").removesuffix(".ovl").lower()


def parse_stem(stem: str) -> dict[str, str] | None:
    match = RAW_STEM_RE.fullmatch(stem)
    if match is None:
        return None
    return match.groupdict()


def action_from_stem(owner: str, action_token: str) -> str | None:
    prefix = f"{owner}_"
    if not action_token.lower().startswith(prefix):
        return None
    return f"{owner}@{action_token[len(prefix):]}".lower()


def manis_filename(parts: dict[str, str]) -> str:
    return f"{parts['group']}.maniset{parts['maniset']}.manis".lower()


def load_inventory(path: Path) -> dict[tuple[str, str], list[dict]]:
    index: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = (normalized_owner(record["object"]), record["action"].lower())
        index[key].append(record)
    for candidates in index.values():
        candidates.sort(key=lambda item: item["relative_file"].lower())
    return index


def load_alias_groups(path: Path | None) -> dict[str, dict[str, str]]:
    aliases: dict[str, dict[str, str]] = {}
    if path is None:
        return aliases
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        target = parse_stem(record["target_raw_bvh_stem"])
        source = parse_stem(record["source_raw_bvh_stem"])
        if target is not None and source is not None:
            aliases[record["target_raw_bvh_stem"].lower()] = source
    return aliases


def choose_candidate(
    candidates: list[dict],
    owner: str,
    target: dict[str, str],
    alias_source: dict[str, str] | None,
) -> tuple[dict | None, str]:
    if not candidates:
        return None, "unresolved_action_not_declared"

    target_file = manis_filename(target)
    named = [item for item in candidates if Path(item["relative_file"]).name.lower() == target_file]
    if len(named) == 1:
        return named[0], "declared_in_named_manis"

    if alias_source is not None:
        alias_file = manis_filename(alias_source)
        aliased = [item for item in candidates if Path(item["relative_file"]).name.lower() == alias_file]
        if len(aliased) == 1:
            return aliased[0], "declared_in_recorded_raw_alias"

    same_kind = [
        item
        for item in candidates
        if Path(item["relative_file"]).name.lower().startswith(f"{target['group'].lower()}.maniset")
    ]
    if len(same_kind) == 1:
        return same_kind[0], "declared_in_same_group_kind"
    if len(candidates) == 1:
        return candidates[0], "unique_declared_action"
    return candidates[0], "ambiguous_deterministic_first"


def main() -> None:
    args = parse_args()
    inventory = load_inventory(args.mani_inventory)
    alias_groups = load_alias_groups(args.raw_alias_manifest)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.unavailable_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    reason_counts: Counter[str] = Counter()
    source_exists_counts: Counter[bool] = Counter()
    matched = unavailable = parse_errors = unresolved = ambiguous = 0

    with args.output_jsonl.open("w", encoding="utf-8", newline="\n") as output, args.unavailable_output_jsonl.open(
        "w", encoding="utf-8", newline="\n"
    ) as unavailable_output:
        for line in args.official_manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            official = json.loads(line)
            if official.get("status") != "matched":
                unavailable += 1
                unavailable_output.write(json.dumps(official, ensure_ascii=False) + "\n")
                continue

            matched += 1
            target = parse_stem(official["raw_bvh_stem"])
            owner = normalized_owner(official["object_name"])
            blender_action = action_from_stem(owner, target["action"]) if target else None
            candidates = inventory.get((owner, blender_action), []) if blender_action else []
            alias_source = alias_groups.get(official["raw_bvh_stem"].lower())
            selected, reason = choose_candidate(candidates, owner, target, alias_source) if target else (None, "unparseable_raw_stem")
            # Blender limits Action identifiers.  AniMo4D's old exporter named
            # BVHs from that post-import identifier, so a few official names
            # end in ``turn``/``onspo`` while the MANIS declaration is longer.
            # The named MANIS still gives an exact source file; retain the
            # Blender name as the export key instead of guessing a full suffix.
            truncated_source_path = (
                args.source_root / f"{official['object_key']}.ovl" / manis_filename(target)
                if target is not None
                else None
            )
            if selected is None and truncated_source_path is not None and truncated_source_path.is_file():
                reason = "blender_action_name_truncated"
            reason_counts[reason] += 1
            parse_errors += int(target is None or blender_action is None)
            unresolved += int(selected is None and reason != "blender_action_name_truncated")
            ambiguous += int(reason == "ambiguous_deterministic_first")
            source_path = args.source_root / selected["relative_file"] if selected else truncated_source_path
            source_exists = bool(source_path and source_path.is_file())
            source_exists_counts[source_exists] += 1
            record = {
                "official_id": official["id"],
                "object_key": official["object_key"],
                "object_name": official["object_name"],
                "official_raw_bvh_stem": official["raw_bvh_stem"],
                "official_text_file": official["text_file"],
                "texts": official["texts"],
                "text_entries": official["text_entries"],
                "official_caption_count": official["caption_count"],
                "text_status": "present",
                "annotation_source": "animo4d_official",
                "target_group": target["group"] if target else None,
                "target_maniset": target["maniset"] if target else None,
                "target_action_token": target["action"] if target else None,
                "declared_action": selected["action"] if selected else None,
                "blender_action": blender_action,
                "source_resolution": reason,
                "source_candidate_count": len(candidates),
                "source_relative_file": (
                    selected["relative_file"]
                    if selected
                    else truncated_source_path.relative_to(args.source_root).as_posix()
                    if truncated_source_path is not None
                    else None
                ),
                "source_path": str(source_path) if source_path else None,
                "source_file_exists": source_exists,
                "source_action_frames": selected.get("frames") if selected else None,
                "source_compression": selected.get("compression") if selected else None,
                "raw_alias_source_group": alias_source["group"] if alias_source else None,
                "raw_alias_source_maniset": alias_source["maniset"] if alias_source else None,
                "candidate_relative_files": [candidate["relative_file"] for candidate in candidates],
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "purpose": "official AniMo4D caption to declared MANIS Action resolution",
        "official_manifest": str(args.official_manifest),
        "mani_inventory": str(args.mani_inventory),
        "source_root": str(args.source_root),
        "raw_alias_manifest": str(args.raw_alias_manifest) if args.raw_alias_manifest else None,
        "matched_official_rows": matched,
        "unavailable_official_rows": unavailable,
        "parse_errors": parse_errors,
        "unresolved": unresolved,
        "ambiguous_deterministic_first": ambiguous,
        "source_resolution_counts": dict(reason_counts),
        "source_file_exists_counts": {str(key).lower(): value for key, value in source_exists_counts.items()},
        "output_jsonl": str(args.output_jsonl),
        "unavailable_output_jsonl": str(args.unavailable_output_jsonl),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
