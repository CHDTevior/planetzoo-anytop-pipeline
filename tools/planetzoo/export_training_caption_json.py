"""Export a clean AnyTop-style caption JSON for Planet Zoo training.

This consumes the richer working annotation JSON and writes a compact by-file
dictionary matching the original AnyTop caption format:

    npy_file_name -> {
        annotation_source, captions, match_status, needs_human_review,
        primary_caption, segment_count_for_source_file, segment_index,
        source_file, source_motion_id, text_status
    }

No absolute Windows paths are written.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.planetzoo.build_planetzoo_annotation_json import (
    animal_phrase,
    article_for,
    clean_caption,
    draft_caption_from_action,
    split_compact_action,
    state_phrase,
    strip_onspot,
    strip_turn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True, help="Working by-file JSON with action/preview metadata.")
    parser.add_argument("--output-json", required=True, help="Clean AnyTop-style caption JSON to write.")
    parser.add_argument("--summary-output", default=None, help="Optional summary JSON path.")
    parser.add_argument("--caption-count", type=int, default=5, help="Target captions per sample.")
    parser.add_argument("--max-captions", type=int, default=6, help="Maximum captions per sample.")
    parser.add_argument("--objects", nargs="*", default=None, help="Optional PZ object names to export.")
    parser.add_argument("--objects-file", default=None, help="Optional JSON list or text file of PZ object names.")
    parser.add_argument("--limit", type=int, default=None, help="Optional pilot sample size.")
    parser.add_argument("--audit-output", default=None, help="Optional JSON audit report for caption corrections/rejections.")
    parser.add_argument("--seed", type=int, default=20260525)
    return parser.parse_args()


def strip_terminal_punctuation(text: str) -> str:
    return re.sub(r"[.!?]+$", "", text.strip())


def display_species_name(species: str) -> str:
    species = species.lower().replace("_", " ")
    replacements = {
        "bairds tapir": "Baird's tapir",
        "thomsons gazelle": "Thomson's gazelle",
        "ring tailed lemur": "ring-tailed lemur",
        "sussex chicken": "Sussex chicken",
    }
    return replacements.get(species, species)


def subject_forms(subject_key: str | None) -> list[str]:
    specific = animal_phrase(subject_key)
    if not subject_key:
        return [specific, "The animal", "An animal"]

    parts = [part for part in str(subject_key).split("_") if part]
    descriptor = ""
    if parts and parts[-1] in {"male", "female", "juvenile"}:
        descriptor = parts.pop()
    species = display_species_name(" ".join(parts) if parts else "animal")
    if descriptor:
        specific = f"{article_for(descriptor)} {descriptor} {species}"
    specific_the = f"The {descriptor + ' ' if descriptor else ''}{species}".strip()
    lower_species = species.lower()
    generic = "A quadruped"
    if any(token in lower_species for token in ["alligator", "crocodile", "caiman", "gharial"]):
        generic = "A crocodilian"
    elif any(token in lower_species for token in ["snake", "cobra", "anaconda", "boa"]):
        generic = "A snake"
    elif any(token in lower_species for token in ["tortoise", "turtle", "iguana", "lizard", "monitor"]):
        generic = "A reptile"
    elif any(token in lower_species for token in ["flamingo", "penguin", "ostrich", "crane", "swan", "peafowl", "chicken", "duck", "goose"]):
        generic = "A bird"
    elif any(token in lower_species for token in ["gorilla", "orangutan", "bonobo", "chimpanzee", "lemur", "gibbon", "monkey", "macaque", "saki"]):
        generic = "A primate"
    elif any(token in lower_species for token in ["seal", "otter", "sea lion", "walrus", "beaver"]):
        generic = "A semi-aquatic mammal"

    forms = [specific, specific_the, f"The {species}"]
    forms.extend(["An animal", generic])
    return forms


def subject_descriptor_and_species(subject_key: str | None) -> tuple[str, str]:
    if not subject_key:
        return "", "animal"
    parts = [part for part in str(subject_key).split("_") if part]
    descriptor = ""
    if parts and parts[-1] in {"male", "female", "juvenile"}:
        descriptor = parts.pop()
    species = display_species_name(" ".join(parts) if parts else "animal")
    return descriptor, species


def canonical_subject(subject_key: str | None, definite: bool = False) -> str:
    descriptor, species = subject_descriptor_and_species(subject_key)
    if not descriptor:
        return "The animal" if definite else "An animal"
    phrase = f"{descriptor} {species}"
    return f"The {phrase}" if definite else f"{article_for(descriptor)} {phrase}"


VERB_START_PATTERN = (
    r"accelerates|attacks|begins|bends|climbs|collapses|continues|dives|drinks|"
    r"eats|flies|grazes|hangs|idles|jumps|lands|lies|lowers|moves|performs|"
    r"pounces|preens|reacts|rests|runs|settles|shakes|slides|stands|starts|"
    r"surfaces|swims|takes|transitions|trots|turns|walks|wakes"
)


INITIAL_SUBJECT_RE = re.compile(
    rf"^(?P<article>A|An|The)\s+(?P<descriptor>male|female|juvenile)\s+"
    rf"(?P<noun>.+?)\s+(?P<verb>(?:{VERB_START_PATTERN})\b.*)$",
    flags=re.IGNORECASE,
)


def extract_predicate(caption: str, subject_key: str | None) -> str:
    text = strip_terminal_punctuation(clean_caption(caption))
    forms = sorted(subject_forms(subject_key), key=len, reverse=True)
    for subject in forms:
        if text.lower().startswith(subject.lower() + " "):
            return text[len(subject):].strip()
    if text.lower().startswith("the motion shows "):
        return text[len("the motion shows "):].strip()
    return text


def normalize_caption_text(text: str) -> str:
    text = re.sub(r"\bring tailed lemur\b", "ring-tailed lemur", text, flags=re.IGNORECASE)
    text = re.sub(r"\bbairds tapir\b", "Baird's tapir", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthomsons gazelle\b", "Thomson's gazelle", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsussex chicken\b", "Sussex chicken", text, flags=re.IGNORECASE)
    text = re.sub(r"\bgracefully\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\babout 89 degrees\b", "roughly a quarter-turn", text, flags=re.IGNORECASE)
    text = re.sub(r"\babout 90 degrees\b", "roughly a quarter-turn", text, flags=re.IGNORECASE)
    text = re.sub(r"\babout 180 degrees\b", "roughly a half-turn", text, flags=re.IGNORECASE)
    text = re.sub(r"\bruns stands\b", "runs, stands", text)
    text = re.sub(r"\bwalks stands\b", "walks, stands", text)
    text = re.sub(r"\bswims stands\b", "swims, stands", text)
    text = re.sub(r"\bwalks to a point stands still\b", "walks to a stop and stands still", text, flags=re.IGNORECASE)
    text = re.sub(r"\bswims to a point treads water\b", "swims to a point and treads water", text, flags=re.IGNORECASE)
    text = re.sub(r"\bswims to a point and then treads water\b", "swims forward, then transitions toward treading water", text, flags=re.IGNORECASE)
    text = re.sub(r"\bswims to a point and treads water\b", "swims forward, then transitions toward treading water", text, flags=re.IGNORECASE)
    text = re.sub(r"\bjumps into (the )?(tall grass|grass|water)\b", "begins a jump", text, flags=re.IGNORECASE)
    text = re.sub(r"\bjumps into (the )?enclosure\b", "begins a jump", text, flags=re.IGNORECASE)
    text = re.sub(r"\bjumps in\b", "begins a jump", text, flags=re.IGNORECASE)
    text = re.sub(r"\bjumps out\b", "lands from a jump", text, flags=re.IGNORECASE)
    text = re.sub(r"\brushes forward\b", "runs quickly", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(runs?|walks?|trots?) forward in place\b", r"\1 in place", text, flags=re.IGNORECASE)
    text = re.sub(r"\btrots forward on the spot\b", "trots in place", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(walks?|runs?) through (the )?(savannah|grassland|habitat|environment)\b", r"\1 forward", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(walks?|runs?|trots?) (steadily )?across (the )?(savannah|grassland|habitat|environment)\b", r"\1 forward", text, flags=re.IGNORECASE)
    text = re.sub(r"\bswims across (the )?water\b", "swims forward", text, flags=re.IGNORECASE)
    text = re.sub(r"\bswims? to (the )?water\b", "swims forward", text, flags=re.IGNORECASE)
    text = re.sub(r"\bswims? on (the )?water\b", "treads water", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(walks?|runs?) (to|toward|towards) (the )?water to drink\b", r"\1 toward a drinking pose", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(walks?|runs?) (to|toward|towards) (a |the )?waterhole( to drink)?\b", r"\1 toward a drinking pose", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(walks?|runs?) (to|toward|towards) (a |the )?water source( to [a-z ]+)?\b", r"\1 toward a drinking pose", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(walks?|runs?) (to|toward|towards) (a |the )?drink trough\b", r"\1 toward a trough-drinking pose", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(walks?|runs?) (to|toward|towards) drink\b", r"\1 toward a drinking pose", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(walks?|runs?) to a spot (and then |to )drink\b", r"\1 into a drinking pose in place", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(walks?|runs?) (to|toward|towards) (the )?(food|feeding area) to eat\b", r"\1 toward an eating pose", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(walks?|runs?) (to|toward|towards) (the )?(food|feeding area)\b", r"\1 toward an eating pose", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(walks?|runs?) to find food\b", r"\1 toward an eating pose", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(walks?|runs?) (to|toward|towards) eat\b", r"\1 toward an eating pose", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return clean_caption(text)


def light_normalize_existing_caption(text: str) -> str:
    text = re.sub(r"\bring tailed lemur\b", "ring-tailed lemur", text, flags=re.IGNORECASE)
    text = re.sub(r"\bbairds tapir\b", "Baird's tapir", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthomsons gazelle\b", "Thomson's gazelle", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsussex chicken\b", "Sussex chicken", text, flags=re.IGNORECASE)
    text = re.sub(r"\bjumps in in place\b", "jumps in place", text, flags=re.IGNORECASE)
    text = re.sub(r"\bbegins a jump in in place\b", "begins a jump in place", text, flags=re.IGNORECASE)
    text = re.sub(r"\bswimlowing forward\b", "swimming low in the water", text, flags=re.IGNORECASE)
    text = re.sub(r"\bswimlowtread water\b", "treading water low in the water", text, flags=re.IGNORECASE)
    text = re.sub(r"\babout 89 degrees\b", "roughly a quarter-turn", text, flags=re.IGNORECASE)
    text = re.sub(r"\babout 90 degrees\b", "roughly a quarter-turn", text, flags=re.IGNORECASE)
    text = re.sub(r"\babout 180 degrees\b", "roughly a half-turn", text, flags=re.IGNORECASE)
    text = re.sub(r"\bruns stands still and then turns\b", "runs to a standstill, then turns", text, flags=re.IGNORECASE)
    text = re.sub(r"\bruns stands still turns\b", "runs to a standstill, then turns", text, flags=re.IGNORECASE)
    text = re.sub(r"\bruns stands still\b", "runs to a standstill", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwalks stands still and then turns\b", "walks to a standstill, then turns", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwalks stands still turns\b", "walks to a standstill, then turns", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwalks stands still\b", "walks to a standstill", text, flags=re.IGNORECASE)
    text = re.sub(r"\bswims stands\b", "swims, stands", text)
    text = re.sub(r"\bwalks to a point stands still\b", "walks to a stop and stands still", text, flags=re.IGNORECASE)
    text = re.sub(r"\bswims to a point treads water\b", "swims to a point and treads water", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwalks to the Eaton spot\b", "walks to an eating spot", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwalks to Eaton spot\b", "walks to an eating spot", text, flags=re.IGNORECASE)
    text = re.sub(r"\bEaton spot\b", "eating spot", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(runs?|walks?|trots?) forward in place\b", r"\1 in place", text, flags=re.IGNORECASE)
    text = re.sub(r"\btrots forward on the spot\b", "trots in place", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return clean_caption(text)


def rewrite_caption_subject(text: str, subject_key: str | None) -> str:
    if not subject_key:
        return text
    match = INITIAL_SUBJECT_RE.match(text.strip())
    if match:
        target_descriptor, _species = subject_descriptor_and_species(subject_key)
        found_descriptor = match.group("descriptor").lower()
        if target_descriptor and found_descriptor != target_descriptor:
            subject = canonical_subject(subject_key, definite=match.group("article").lower() == "the")
            return f"{subject} {match.group('verb')}"

    parts = [part for part in str(subject_key).split("_") if part]
    descriptor = ""
    if parts and parts[-1] in {"male", "female", "juvenile"}:
        descriptor = parts.pop()
    if not descriptor or not parts:
        return text
    species = display_species_name(" ".join(parts))
    species_variants = {
        species,
        species.replace("-", " "),
        species.replace("'", ""),
        species.replace("'s", "s"),
    }
    specific = f"{article_for(descriptor)} {descriptor} {species}"
    specific_the = f"The {descriptor} {species}"
    for variant in sorted(species_variants, key=len, reverse=True):
        escaped = re.escape(variant)
        text = re.sub(rf"^(A|An)\s+(male|female|juvenile)\s+{escaped}\b", specific, text, flags=re.IGNORECASE)
        text = re.sub(rf"^The\s+(male|female|juvenile)\s+{escaped}\b", specific_the, text, flags=re.IGNORECASE)
    return text


def caption_conflict_reasons(text: str, entry: dict[str, Any], subject_key: str | None) -> list[str]:
    reasons: list[str] = []
    target_descriptor, _species = subject_descriptor_and_species(subject_key)
    lower = text.lower()
    if target_descriptor:
        wrong_descriptors = [
            value
            for value in ("male", "female", "juvenile")
            if value != target_descriptor and re.search(rf"\b{value}\b", lower)
        ]
        if wrong_descriptors:
            reasons.append("unresolved_descriptor_" + "_".join(wrong_descriptors))

    action = str(entry.get("action_short") or "").lower()
    if "adult male" in lower or "adult female" in lower or "become an adult" in lower:
        if "adult" not in action:
            reasons.append("adult_transition_text_does_not_match_action")
    if re.search(r"\bperforms an? [a-z0-9_ ]+ motion\b", lower):
        reasons.append("raw_action_fallback_caption")
    malformed_fragments = [
        "jumpou to nspot",
        "slidebasefas to nspot",
        "swimtread wateronspo",
        "swim baseonspo",
        "wateronspo",
        "baseonspo",
    ]
    if any(fragment in lower for fragment in malformed_fragments):
        reasons.append("malformed_action_tokenization")
    return reasons


def review_existing_caption(
    text: str,
    entry: dict[str, Any],
    subject_key: str | None,
) -> tuple[str, list[str], bool]:
    original = str(text).strip()
    if not original:
        return "", [], True
    corrected = rewrite_caption_subject(original, subject_key)
    corrected = light_normalize_existing_caption(corrected)
    reasons: list[str] = []
    if corrected != original:
        reasons.append("caption_text_corrected")
    conflict_reasons = caption_conflict_reasons(corrected, entry, subject_key)
    reasons.extend(conflict_reasons)
    return corrected, reasons, bool(conflict_reasons)


def is_high_risk_action(action_short: str | None) -> bool:
    action = (action_short or "").lower()
    high_risk_tokens = [
        "eat",
        "drink",
        "trough",
        "graze",
        "swimtreadwater",
        "enrichment",
        "idle",
        "jumpin",
        "jumpout",
    ]
    return any(token in action for token in high_risk_tokens)


def add_caption(captions: list[str], text: str, subject_key: str | None = None) -> None:
    text = rewrite_caption_subject(text, subject_key)
    text = normalize_caption_text(text)
    if not text:
        return
    if re.search(r"\bperforms an? [a-z0-9_]+ motion\b", text.lower()):
        return
    normalized = re.sub(r"[^a-z0-9]+", "", text.lower())
    if all(re.sub(r"[^a-z0-9]+", "", item.lower()) != normalized for item in captions):
        captions.append(text)


def add_preserved_caption(captions: list[str], text: str) -> None:
    text = str(text).strip()
    if not text:
        return
    normalized = re.sub(r"[^a-z0-9]+", "", text.lower())
    if all(re.sub(r"[^a-z0-9]+", "", item.lower()) != normalized for item in captions):
        captions.append(text)


def motion_show_caption(subject_key: str | None, predicate: str) -> str:
    subject = animal_phrase(subject_key).removeprefix("A ").removeprefix("An ")
    return f"The motion shows {subject} {predicate}"


def fallback_predicate_for_action(action_short: str | None) -> str:
    action = str(action_short or "").lower().replace("_", "")
    action = re.sub(r"onspo$", "onspot", action)
    body, direction, degrees = strip_turn(action)
    body, onspot = strip_onspot(body)

    if body == "swimlowbase":
        predicate = "swims low in the water"
    elif body == "jumpout":
        predicate = "lands from a jump"
    elif body == "jumpin":
        predicate = "begins a jump"
    elif re.fullmatch(r"jumpdown\d+x\d+m", body):
        predicate = "jumps down"
    elif re.fullmatch(r"jumpup\d+x\d+m", body):
        predicate = "jumps up"
    elif re.fullmatch(r"jump\d+m", body):
        predicate = "jumps forward"
    elif re.fullmatch(r"testtransition\d*", body):
        predicate = "performs a test transition"
    elif body == "swimlowtreadwater":
        predicate = "treads water low in the water"
    elif body == "slidebasefast":
        predicate = "slides forward quickly"
    elif body == "slidebaseslow":
        predicate = "slides forward slowly"
    elif body == "slidebase":
        predicate = "slides forward"
    elif body == "walkbasewingsout":
        predicate = "walks forward with wings out"
    elif "to" in body:
        src, dst = body.split("to", 1)
        predicate = f"transitions from {state_phrase(src)} to {state_phrase(dst)}"
    else:
        readable = split_compact_action(body)
        predicate = f"performs a {readable} action" if readable else "performs an animal movement"

    if onspot and "in place" not in predicate:
        predicate += " in place"
    if direction and "turns " not in predicate and "turning " not in predicate:
        predicate += f" while turning {direction}"
    if degrees:
        degrees_value = int(degrees)
        if 85 <= degrees_value <= 95:
            predicate += " roughly a quarter-turn"
        elif 175 <= degrees_value <= 185:
            predicate += " roughly a half-turn"
    return predicate


def generic_paraphrases(
    key: str,
    primary: str,
    entry: dict[str, Any],
    target_count: int,
    max_count: int,
    subject_key: str | None,
    audit_records: list[dict[str, Any]] | None = None,
) -> list[str]:
    action_short = entry.get("action_short")
    captions: list[str] = []
    draft = draft_caption_from_action({"animal_key": subject_key or entry.get("animal_key"), "action_short": action_short})

    seen_inputs: set[str] = set()
    for text in list(entry.get("captions") or []) + [primary]:
        input_norm = re.sub(r"[^a-z0-9]+", "", str(text).lower())
        if not input_norm or input_norm in seen_inputs:
            continue
        seen_inputs.add(input_norm)
        corrected, reasons, rejected = review_existing_caption(text, entry, subject_key)
        if audit_records is not None and reasons:
            audit_records.append(
                {
                    "file": key,
                    "action_short": entry.get("action_short", ""),
                    "text_status": entry.get("text_status", ""),
                    "original_caption": str(text).strip(),
                    "reviewed_caption": corrected,
                    "rejected_from_training_json": rejected,
                    "reasons": reasons,
                }
            )
        if not rejected:
            add_preserved_caption(captions, corrected)
    add_caption(captions, draft, subject_key)

    predicate_source = normalize_caption_text(draft) if draft else (captions[0] if captions else primary)
    predicate = extract_predicate(predicate_source, subject_key)
    forms = subject_forms(subject_key)
    for subject in forms:
        add_caption(captions, f"{subject} {predicate}", subject_key)
        if len(captions) >= target_count:
            break

    if len(captions) < target_count:
        add_caption(captions, motion_show_caption(subject_key, predicate), subject_key)

    if len(captions) < target_count and action_short:
        action_predicate = extract_predicate(draft, subject_key)
        for subject in forms:
            add_caption(captions, f"{subject} {action_predicate}", subject_key)
            if len(captions) >= target_count:
                break

    if len(captions) < target_count:
        fallback_predicate = fallback_predicate_for_action(action_short)
        for subject in forms:
            add_caption(captions, f"{subject} {fallback_predicate}", subject_key)
            if len(captions) >= target_count:
                break
        if len(captions) < target_count:
            add_caption(captions, motion_show_caption(subject_key, fallback_predicate), subject_key)

    if not captions:
        add_caption(captions, f"{canonical_subject(subject_key)} performs an animal movement", subject_key)

    return captions[:max_count]


def subject_key_from_object_name(object_name: str) -> str:
    if object_name.startswith("PZ_"):
        object_name = object_name[3:]
    return object_name.lower()


def clean_record(
    key: str,
    entry: dict[str, Any],
    caption_count: int,
    max_captions: int,
    audit_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    text_status = entry.get("text_status")
    is_present = text_status == "present"
    primary = entry.get("primary_caption") or (entry.get("captions") or [""])[0]
    object_name = object_name_from_record(key, entry)
    subject_key = subject_key_from_object_name(object_name) if object_name else entry.get("animal_key")
    captions = generic_paraphrases(key, primary, entry, caption_count, max_captions, subject_key, audit_records)
    primary_caption = captions[0] if captions else ""

    return {
        "annotation_source": "collected" if is_present else "codex_draft_from_contact_sheets_and_action_name",
        "captions": captions,
        "match_status": "matched_text" if is_present else entry.get("match_status", "matched_source_missing_text"),
        "needs_human_review": False if is_present else True,
        "primary_caption": primary_caption,
        "segment_count_for_source_file": int(entry.get("segment_count_for_source_file", 1)),
        "segment_index": int(entry.get("segment_index", 0)),
        "source_file": Path(entry.get("source_file", "")).name,
        "source_motion_id": entry.get("source_motion_id", ""),
        "text_status": "present" if is_present else "codex_draft",
    }


def action_bucket(action: str) -> str:
    action = action.lower()
    if "fight" in action or "attack" in action:
        return "fight"
    if "climb" in action or "hang" in action:
        return "climb"
    if "swim" in action or "dive" in action:
        return "swim"
    if "run" in action:
        return "run"
    if "walk" in action:
        return "walk"
    if "turn" in action:
        return "turn"
    if "jump" in action:
        return "jump"
    if "drink" in action or "eat" in action or "graze" in action:
        return "food"
    if "rest" in action or "sleep" in action or "idle" in action:
        return "rest"
    return "other"


def object_name_from_record(key: str, entry: dict[str, Any]) -> str:
    source_stem = Path(entry.get("source_file", "")).stem
    key_stem = Path(key).stem
    if source_stem:
        marker = "_" + source_stem + "_"
        pos = key_stem.find(marker)
        if pos > 0:
            return key_stem[:pos]
    match = re.match(r"^(PZ_.+?)_[a-z0-9]+_", key_stem, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def load_object_filter(args: argparse.Namespace) -> set[str] | None:
    values: list[str] = []
    if args.objects:
        values.extend(args.objects)
    if args.objects_file:
        path = Path(args.objects_file)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            loaded = json.loads(text)
            if isinstance(loaded, list):
                values.extend(str(item) for item in loaded)
            else:
                raise ValueError("--objects-file JSON must contain a list")
        else:
            values.extend(line.strip() for line in text.splitlines() if line.strip())
    clean = {value.strip() for value in values if value.strip()}
    return clean or None


def select_pilot(data: dict[str, dict[str, Any]], limit: int, seed: int) -> dict[str, dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for key, entry in data.items():
        status = entry.get("text_status", "")
        bucket = action_bucket(str(entry.get("action_short", key)))
        groups[(status, bucket)].append((key, entry))

    for values in groups.values():
        rng.shuffle(values)

    selected: list[tuple[str, dict[str, Any]]] = []
    group_keys = sorted(groups)
    while len(selected) < limit and group_keys:
        progressed = False
        for group_key in group_keys:
            values = groups[group_key]
            if values:
                selected.append(values.pop())
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    return dict(selected)


def summarize(output: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(output),
        "text_status": dict(Counter(row["text_status"] for row in output.values())),
        "annotation_source": dict(Counter(row["annotation_source"] for row in output.values())),
        "caption_lengths": dict(Counter(len(row["captions"]) for row in output.values())),
        "fields": sorted(set().union(*(row.keys() for row in output.values()))) if output else [],
    }


def summarize_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    for record in records:
        reason_counts.update(record.get("reasons", []))
    return {
        "records": len(records),
        "rejected": sum(1 for record in records if record.get("rejected_from_training_json")),
        "corrected": sum(
            1
            for record in records
            if record.get("reviewed_caption") and record.get("reviewed_caption") != record.get("original_caption")
        ),
        "reasons": dict(reason_counts),
        "examples": records[:50],
    }


def main() -> None:
    args = parse_args()
    data: dict[str, dict[str, Any]] = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    object_filter = load_object_filter(args)
    if object_filter is not None:
        data = {
            key: entry
            for key, entry in data.items()
            if object_name_from_record(key, entry) in object_filter
        }
    if args.limit is not None:
        data = select_pilot(data, args.limit, args.seed)

    audit_records: list[dict[str, Any]] = []
    output = {
        key: clean_record(key, entry, args.caption_count, args.max_captions, audit_records)
        for key, entry in data.items()
    }
    summary = summarize(output)
    summary["caption_review"] = summarize_audit(audit_records)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.audit_output:
        audit_path = Path(args.audit_output)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_payload = {
            "summary": summarize_audit(audit_records),
            "records": audit_records,
        }
        audit_path.write_text(json.dumps(audit_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
