"""Match AniMosity4D text captions to Planet Zoo AnyTop motions.

The AniMosity4D text dump stores one small txt file per source keypoints file,
with lines like:

    animal#sex#caption#token_tags#start#end

This script attaches those captions to the Planet Zoo AnyTop motion manifest and
also writes a by-file JSON dictionary shaped like the prior AnyTop annotation
file used for caption work.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TEXT_NAME_SUFFIXES = (
    "_keypoints.json.txt",
    "_keypoints.txt",
    ".json.txt",
    ".txt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-manifest", required=True, help="Existing Planet Zoo motion_text_manifest.jsonl.")
    parser.add_argument("--texts-root", required=True, help="Directory containing AniMosity4D *.json.txt caption files.")
    parser.add_argument("--vlm-preview-manifest", default=None, help="Optional VLM preview manifest for preview_path fields.")
    parser.add_argument("--pooled-root", default=None, help="Optional pooled AnyTop layout root for rewriting motion/BVH paths.")
    parser.add_argument("--manifest-output", default=None, help="Output JSONL manifest with attached captions.")
    parser.add_argument("--manifest-json-output", default=None, help="Optional regular JSON copy with summary and rows.")
    parser.add_argument("--manifest-csv-output", default=None, help="Optional CSV copy for inspection.")
    parser.add_argument("--by-file-output", required=True, help="Output by-file JSON dictionary.")
    parser.add_argument("--summary-output", default=None, help="Optional summary JSON path.")
    parser.add_argument(
        "--draft-missing",
        action="store_true",
        help="Generate conservative Codex draft captions from action names when no collected text matches.",
    )
    return parser.parse_args()


def norm_key(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def clean_caption(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def article_for(phrase: str) -> str:
    return "An" if phrase[:1].lower() in {"a", "e", "i", "o", "u"} else "A"


def animal_phrase(animal_key: str | None) -> str:
    if not animal_key:
        return "An animal"
    parts = [part for part in str(animal_key).split("_") if part]
    descriptor = ""
    if parts and parts[-1] in {"male", "female", "juvenile"}:
        descriptor = parts.pop()
    species = " ".join(parts) if parts else "animal"
    phrase = f"{descriptor} {species}".strip()
    return f"{article_for(phrase)} {phrase}"


def split_compact_action(value: str) -> str:
    text = value.replace("_", " ")
    text = re.sub(r"(\d+)", r" \1 ", text)
    replacements = [
        ("deepswim", "deep swim"),
        ("divein", "dive in"),
        ("diveout", "dive out"),
        ("climbidlevertical", "climb idle vertical"),
        ("climbbeambase", "climb beam base"),
        ("climbdownbase", "climb down base"),
        ("climbupbase", "climb up base"),
        ("climbbeam", "climb beam"),
        ("climbdown", "climb down"),
        ("climbup", "climb up"),
        ("jumponspot", "jump on spot"),
        ("walkonspot", "walk on spot"),
        ("climbonspot", "climb on spot"),
        ("treadwater", "tread water"),
        ("drinktrough", "drink trough"),
        ("herorespond", "hero respond"),
        ("herocall", "hero call"),
        ("chaseoff", "chase off"),
        ("reacttodie", "react to die"),
        ("victorytaunt", "victory taunt"),
        ("standidle", "stand idle"),
        ("restloop", "rest loop"),
        ("eatloop", "eat loop"),
        ("drinkloop", "drink loop"),
        ("walkbase", "walk base"),
        ("runbase", "run base"),
        ("swimbase", "swim base"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_turn(action: str) -> tuple[str, str, str]:
    match = re.search(r"turn([lr])(\d+)?$", action)
    if not match:
        return action, "", ""
    direction = "left" if match.group(1) == "l" else "right"
    degrees = match.group(2) or ""
    return action[: match.start()], direction, degrees


def strip_onspot(action: str) -> tuple[str, bool]:
    if action.endswith("onspot"):
        return action[: -len("onspot")], True
    return action, False


def state_phrase(state: str) -> str:
    state = re.sub(r"\d+$", "", state.replace("_", ""))
    state, onspot = strip_onspot(state)
    mapping = {
        "stand": "standing",
        "standidle": "standing idly",
        "idle": "idling",
        "run": "running",
        "runbase": "running forward",
        "walk": "walking",
        "walkbase": "walking forward",
        "swim": "swimming",
        "swimbase": "swimming forward",
        "swimtreadwater": "treading water",
        "jump": "jumping",
        "jumpin": "jumping in",
        "jumpmid": "mid-jump",
        "jumpout": "jumping out",
        "drink": "drinking",
        "drinkloop": "drinking",
        "drinktrough": "drinking from a trough",
        "drinktroughloop": "drinking from a trough",
        "eat": "eating",
        "eatloop": "eating",
        "graze": "grazing",
        "rest": "resting",
        "restloop": "resting",
        "sleep": "sleeping",
        "fight": "fighting",
        "climb": "climbing",
        "climbbase": "climbing",
        "climbidle": "idling on a climb",
        "climbidlevertical": "idling vertically on a climb",
        "climbbeam": "climbing on a beam",
        "climbbeambase": "moving along a climb beam",
        "climbdown": "climbing down",
        "climbdownbase": "climbing down",
        "climbup": "climbing up",
        "climbupbase": "climbing up",
        "climbjumpin": "jumping into a climb",
        "climbjumpout": "jumping out of a climb",
        "hangdown": "hanging down",
        "deepswim": "deep swimming",
        "deepswimdivein": "diving into deep water",
        "deepswimdiveout": "surfacing from deep water",
        "deepswimidle": "idling in deep water",
        "deepswimvariant": "deep swimming",
        "burrow": "burrowing",
    }
    phrase = mapping.get(state)
    if phrase is None:
        if state.endswith("base"):
            phrase = f"{split_compact_action(state[:-4])}ing forward"
        else:
            phrase = split_compact_action(state)
    if onspot and "in place" not in phrase:
        phrase = f"{phrase} in place"
    return phrase


def predicate_for_action(action: str) -> str:
    action = action.replace("_", "")
    action, onspot = strip_onspot(action)
    mapping = {
        "run": "runs",
        "runbase": "runs forward",
        "walk": "walks",
        "walkbase": "walks forward",
        "swim": "swims",
        "swimbase": "swims forward",
        "swimtreadwater": "treads water",
        "stand": "stands",
        "standidle": "stands idly",
        "jumpin": "jumps in",
        "jumpmid": "continues through a jump",
        "jumpout": "jumps out",
        "jumptoswim": "jumps into a swim",
        "drinkloop": "drinks",
        "drinktroughloop": "drinks from a trough",
        "eatloop": "eats",
        "graze": "grazes",
        "grazeloop": "grazes",
        "shake": "shakes its body",
        "standroar": "stands and roars",
        "standpoop": "defecates while standing",
        "standpreen": "preens while standing",
        "restpreen": "preens while resting",
        "restloop": "rests",
        "sleeptorest": "wakes from sleep into a resting pose",
        "resttosleep": "settles from resting into sleep",
        "standdie": "collapses from standing",
        "restdie": "collapses from a resting pose",
        "fightattack": "attacks during a fight",
        "fightchaseoff": "chases another animal off",
        "fightflee": "flees from a fight",
        "fightreact": "reacts during a fight",
        "fightreacttodie": "reacts to a fight and collapses",
        "fighttaunt": "taunts during a fight",
        "fighttauntreact": "reacts to a fight taunt",
        "fightvictorytaunt": "performs a victory taunt after fighting",
        "fightidle": "idles in a fighting stance",
        "matingritual": "performs a mating ritual",
        "matingcourtship": "performs a courtship display",
        "standgimmick": "performs a standing display",
        "standherocall": "performs a standing call",
        "standherorespond": "responds while standing",
        "climb": "climbs",
        "climbbase": "climbs",
        "climbbeam": "climbs on a beam",
        "climbbeambase": "moves along a climb beam",
        "climbdown": "climbs down",
        "climbdownbase": "climbs down",
        "climbup": "climbs up",
        "climbupbase": "climbs up",
        "climbidle": "idles on a climb",
        "climbidlevertical": "idles vertically on a climb",
        "climbjumpin": "jumps into a climb",
        "climbjumpout": "jumps out of a climb",
        "hangdown": "hangs down",
        "hangdownloop": "hangs down",
        "deepswim": "swims underwater",
        "deepswimdivein": "dives into deep water",
        "deepswimdiveout": "surfaces from deep water",
        "deepswimidle": "idles in deep water",
        "deepswimvariant": "swims underwater",
    }
    trimmed = re.sub(r"\d+$", "", action)
    predicate = mapping.get(action) or mapping.get(trimmed)
    if predicate is None:
        predicate = f"performs a {split_compact_action(action)} motion"
    if onspot and "in place" not in predicate:
        predicate = f"{predicate} in place"
    return predicate


def draft_caption_from_action(row: dict[str, Any]) -> str:
    action = str(row.get("action_short") or "").lower().replace("_", "")
    subject = animal_phrase(row.get("animal_key"))
    if not action:
        return clean_caption(f"{subject} performs an animal motion.")

    body, direction, degrees = strip_turn(action)
    body, onspot = strip_onspot(body)
    if onspot:
        body = body + "onspot"

    if "to" in body:
        src, dst = body.split("to", 1)
        caption = f"{subject} transitions from {state_phrase(src)} to {state_phrase(dst)}"
    elif direction and body in {"stand", "standidle", ""}:
        caption = f"{subject} turns {direction}"
    else:
        caption = f"{subject} {predicate_for_action(body)}"

    if direction and "turns " not in caption:
        caption += f" while turning {direction}"
    if degrees:
        degrees_value = int(degrees)
        if degrees_value:
            caption += f" about {degrees_value} degrees"
    return clean_caption(caption)


def source_key_from_text_file(path: Path) -> str:
    name = path.name
    lower = name.lower()
    for suffix in TEXT_NAME_SUFFIXES:
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def parse_text_file(path: Path) -> dict[str, Any] | None:
    captions: list[str] = []
    source_animals: list[str] = []
    source_sexes: list[str] = []
    tagged_lines: list[str] = []

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("#")
        if len(parts) >= 3:
            source_animals.append(parts[0].strip())
            source_sexes.append(parts[1].strip())
            caption = parts[2].strip()
            if len(parts) >= 4 and parts[3].strip():
                tagged_lines.append(parts[3].strip())
        else:
            caption = line
        caption = clean_caption(caption)
        if caption and caption not in captions:
            captions.append(caption)

    if not captions:
        return None
    return {
        "source_key": source_key_from_text_file(path),
        "captions": captions,
        "source_animals": sorted({item for item in source_animals if item}),
        "source_sexes": sorted({item for item in source_sexes if item}),
        "tagged_lines": tagged_lines,
        "text_source": str(path.resolve()),
    }


def load_text_index(texts_root: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for path in sorted(texts_root.rglob("*.txt")):
        record = parse_text_file(path)
        if record is None:
            continue
        index[norm_key(record["source_key"])] = record
    return index


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_preview_index(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    index: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("sample_type") != "motion":
                continue
            motion_path = row.get("motion_path")
            preview_path = row.get("preview_path")
            if motion_path and preview_path:
                index[Path(motion_path).name] = preview_path
    return index


def candidate_keys(row: dict[str, Any]) -> list[str]:
    keys = [
        row.get("raw_bvh_stem"),
        Path(row.get("raw_bvh", "")).stem if row.get("raw_bvh") else "",
        row.get("source_motion_key"),
        row.get("action_name"),
        row.get("action_short"),
    ]
    animal_key = row.get("animal_key")
    action_short = row.get("action_short")
    if animal_key and action_short:
        keys.extend([f"{animal_key}_{action_short}", f"{animal_key}@{action_short}"])
    return [str(key) for key in keys if key]


def find_text_match(row: dict[str, Any], text_index: dict[str, dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    for key in candidate_keys(row):
        match = text_index.get(norm_key(key))
        if match:
            return match, key
    return None, ""


def rewrite_to_pooled(row: dict[str, Any], pooled_root: Path | None) -> dict[str, Any]:
    if pooled_root is None:
        return dict(row)
    out = dict(row)
    motion_name = Path(row["processed_motion"]).name
    stem = Path(motion_name).stem
    out["processed_motion"] = str((pooled_root / "motions" / motion_name).resolve())
    out["processed_bvh"] = str((pooled_root / "bvhs" / f"{stem}.bvh").resolve())
    out["processed_animation"] = str((pooled_root / "animations" / f"{stem}_from_ric.mp4").resolve())
    return out


def segment_indices(rows: list[dict[str, Any]]) -> tuple[dict[int, int], dict[str, int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        key = row.get("raw_bvh_stem") or Path(row.get("processed_motion", "")).stem
        grouped[str(key)].append(idx)

    index_by_row: dict[int, int] = {}
    count_by_source: dict[str, int] = {}
    for key, indices in grouped.items():
        count_by_source[key] = len(indices)
        for seg_idx, row_idx in enumerate(indices):
            index_by_row[row_idx] = seg_idx
    return index_by_row, count_by_source


def build_outputs(
    rows: list[dict[str, Any]],
    text_index: dict[str, dict[str, Any]],
    preview_index: dict[str, str],
    pooled_root: Path | None,
    draft_missing: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    seg_index_by_row, seg_count_by_source = segment_indices(rows)
    output_rows: list[dict[str, Any]] = []
    by_file: dict[str, dict[str, Any]] = {}
    status_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    matched_text_sources: Counter[str] = Counter()

    for row_idx, original_row in enumerate(rows):
        row = rewrite_to_pooled(original_row, pooled_root)
        match, match_key = find_text_match(original_row, text_index)
        motion_name = Path(row["processed_motion"]).name
        source_key = str(original_row.get("raw_bvh_stem") or Path(original_row.get("raw_bvh", "")).stem)

        if match:
            captions = list(match["captions"])
            primary_caption = captions[0]
            text_source = match["text_source"]
            text_sources = [text_source]
            text_status = "present"
            match_status = "matched_text"
            annotation_source = "animosty4d_text"
            needs_human_review = False
            matched_text_sources[text_source] += 1
        elif draft_missing:
            primary_caption = draft_caption_from_action(original_row)
            captions = [primary_caption] if primary_caption else []
            text_source = ""
            text_sources = []
            text_status = "codex_draft"
            match_status = "matched_source_missing_text"
            annotation_source = "codex_draft_from_filename_and_preview"
            needs_human_review = True
        else:
            captions = []
            primary_caption = ""
            text_source = ""
            text_sources = []
            text_status = "missing"
            match_status = "missing_text"
            annotation_source = "codex_blank_pending"
            needs_human_review = True

        preview_path = preview_index.get(Path(original_row["processed_motion"]).name, "")
        row.update(
            {
                "text": primary_caption,
                "texts": captions,
                "primary_caption": primary_caption,
                "captions": captions,
                "text_source": text_source,
                "text_sources": text_sources,
                "text_match_key": match_key,
                "text_match_status": match_status,
                "annotation_source": annotation_source,
                "text_status": text_status,
                "needs_human_review": needs_human_review,
                "vlm_preview_path": preview_path,
            }
        )
        output_rows.append(row)

        by_file[motion_name] = {
            "annotation_source": annotation_source,
            "captions": captions,
            "match_status": match_status,
            "needs_human_review": needs_human_review,
            "primary_caption": primary_caption,
            "segment_count_for_source_file": seg_count_by_source.get(source_key, 1),
            "segment_index": seg_index_by_row.get(row_idx, 0),
            "source_file": Path(original_row.get("raw_bvh", "")).name,
            "source_motion_id": original_row.get("source_motion_key", ""),
            "text_status": text_status,
            "text_source": text_source,
            "text_match_key": match_key,
            "processed_motion": row["processed_motion"],
            "processed_bvh": row["processed_bvh"],
            "raw_bvh": original_row.get("raw_bvh", ""),
            "animal_key": original_row.get("animal_key", ""),
            "action_short": original_row.get("action_short", ""),
            "vlm_preview_path": preview_path,
        }

        status_counts[match_status] += 1
        action_counts[str(original_row.get("action_short", ""))] += 1
        object_counts[str(original_row.get("object_name") or Path(original_row["processed_motion"]).parent.parent.name)] += 1

    summary = {
        "rows": len(output_rows),
        "status_counts": dict(status_counts),
        "matched_unique_text_files": len(matched_text_sources),
        "text_index_size": len(text_index),
        "object_count": len(object_counts),
        "unique_action_short_count": len(action_counts),
        "draft_missing": draft_missing,
    }
    return output_rows, by_file, summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "processed_motion",
        "processed_bvh",
        "raw_bvh",
        "animal_key",
        "action_short",
        "source_motion_key",
        "primary_caption",
        "text_match_status",
        "text_match_key",
        "text_source",
        "needs_human_review",
        "vlm_preview_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = load_jsonl(Path(args.motion_manifest))
    text_index = load_text_index(Path(args.texts_root))
    preview_index = load_preview_index(Path(args.vlm_preview_manifest) if args.vlm_preview_manifest else None)
    pooled_root = Path(args.pooled_root) if args.pooled_root else None

    output_rows, by_file, summary = build_outputs(rows, text_index, preview_index, pooled_root, args.draft_missing)

    if args.manifest_output:
        write_jsonl(Path(args.manifest_output), output_rows)
    if args.manifest_json_output:
        path = Path(args.manifest_json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"summary": summary, "rows": output_rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.manifest_csv_output:
        write_csv(Path(args.manifest_csv_output), output_rows)

    by_file_path = Path(args.by_file_output)
    by_file_path.parent.mkdir(parents=True, exist_ok=True)
    by_file_path.write_text(json.dumps(by_file, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.summary_output:
        path = Path(args.summary_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
