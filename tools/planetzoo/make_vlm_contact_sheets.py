"""Build paged contact sheets for Planet Zoo VLM/action caption review."""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.planetzoo.export_training_caption_json import object_name_from_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-json", required=True, help="Rich working JSON with vlm_preview_path.")
    parser.add_argument("--output-dir", required=True, help="Directory for contact sheet JPGs.")
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--cell-width", type=int, default=560)
    parser.add_argument("--image-height", type=int, default=92)
    parser.add_argument("--label-height", type=int, default=76)
    parser.add_argument("--header-height", type=int, default=42)
    parser.add_argument("--max-items-per-sheet", type=int, default=90)
    parser.add_argument("--quality", type=int, default=88)
    parser.add_argument("--objects-file", default=None, help="Optional JSON list or newline text file of object names.")
    return parser.parse_args()


def load_object_filter(path_text: str | None) -> set[str] | None:
    if not path_text:
        return None
    path = Path(path_text)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        loaded = json.loads(text)
        if not isinstance(loaded, list):
            raise ValueError("--objects-file JSON must contain a list")
        values = [str(item).strip() for item in loaded]
    else:
        values = [line.strip() for line in text.splitlines()]
    result = {value for value in values if value}
    return result or None


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "unknown"


def font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def resize_strip(path: Path, width: int, height: int) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height), "white")
        x = (width - image.width) // 2
        y = (height - image.height) // 2
        canvas.paste(image, (x, y))
        return canvas


def wrap_line(draw: ImageDraw.ImageDraw, text: str, face: ImageFont.ImageFont, width: int) -> list[str]:
    if not text:
        return []
    avg = max(1, draw.textlength("abcdefghijklmnopqrstuvwxyz", font=face) / 26.0)
    chars = max(18, int(width / avg))
    lines: list[str] = []
    for raw_line in text.splitlines():
        lines.extend(textwrap.wrap(raw_line, width=chars, max_lines=2, placeholder="..."))
    return lines[:2]


def sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, str]:
    entry = item[1]
    preview = Path(str(entry.get("vlm_preview_path", ""))).name
    match = re.match(r"^(\d+)_", preview)
    index = int(match.group(1)) if match else int(entry.get("segment_index", 0))
    return index, item[0]


def make_sheet(
    object_name: str,
    rows: list[tuple[str, dict[str, Any]]],
    page_index: int,
    page_count: int,
    args: argparse.Namespace,
) -> Image.Image:
    cols = max(1, args.columns)
    cell_width = args.cell_width
    cell_height = args.image_height + args.label_height
    body_rows = (len(rows) + cols - 1) // cols
    width = cols * cell_width
    height = args.header_height + body_rows * cell_height

    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    header_font = font(20)
    title_font = font(15)
    caption_font = font(12)
    muted = (82, 82, 82)
    line = (218, 218, 218)
    header = f"{object_name}    page {page_index + 1}/{page_count}    motions {len(rows)}"
    draw.rectangle([0, 0, width, args.header_height], fill=(246, 246, 246))
    draw.text((8, 9), header, fill=(20, 20, 20), font=header_font)

    for local_index, (key, entry) in enumerate(rows):
        col = local_index % cols
        row = local_index // cols
        x0 = col * cell_width
        y0 = args.header_height + row * cell_height
        draw.rectangle([x0, y0, x0 + cell_width - 1, y0 + cell_height - 1], outline=line)

        preview_path = Path(str(entry.get("vlm_preview_path", "")))
        if preview_path.exists():
            strip = resize_strip(preview_path, cell_width, args.image_height)
            sheet.paste(strip, (x0, y0))
        else:
            draw.rectangle([x0, y0, x0 + cell_width, y0 + args.image_height], fill=(250, 238, 238))
            draw.text((x0 + 8, y0 + 32), "missing preview", fill=(150, 40, 40), font=title_font)

        label_y = y0 + args.image_height + 5
        preview_stem = Path(str(entry.get("vlm_preview_path", ""))).stem
        display_index = preview_stem.split("_", 1)[0] if preview_stem else f"{local_index + 1:03d}"
        action = str(entry.get("action_short") or entry.get("source_motion_id") or Path(key).stem)
        status = str(entry.get("text_status", ""))
        first = f"{display_index} {action} [{status}]"
        draw.text((x0 + 8, label_y), first[:80], fill=(10, 10, 10), font=title_font)

        caption = str(entry.get("primary_caption") or ((entry.get("captions") or [""])[0]))
        for offset, wrapped in enumerate(wrap_line(draw, caption, caption_font, cell_width - 16)):
            draw.text((x0 + 8, label_y + 22 + offset * 17), wrapped, fill=muted, font=caption_font)

    return sheet


def main() -> None:
    args = parse_args()
    data: dict[str, dict[str, Any]] = json.loads(Path(args.annotation_json).read_text(encoding="utf-8"))
    object_filter = load_object_filter(args.objects_file)

    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for key, entry in data.items():
        object_name = object_name_from_record(key, entry)
        if object_filter is not None and object_name not in object_filter:
            continue
        grouped[object_name].append((key, entry))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for object_name in sorted(grouped):
        rows = sorted(grouped[object_name], key=sort_key)
        chunks = [
            rows[index : index + args.max_items_per_sheet]
            for index in range(0, len(rows), args.max_items_per_sheet)
        ]
        for page_index, chunk in enumerate(chunks):
            suffix = f"_p{page_index + 1:02d}" if len(chunks) > 1 else ""
            output_path = output_dir / f"{safe_name(object_name)}{suffix}.jpg"
            sheet = make_sheet(object_name, chunk, page_index, len(chunks), args)
            sheet.save(output_path, quality=args.quality, optimize=True)
            manifest.append(
                {
                    "object_name": object_name,
                    "page": page_index + 1,
                    "pages": len(chunks),
                    "rows": len(chunk),
                    "output_path": str(output_path),
                }
            )

    manifest_path = output_dir / "contact_sheet_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"objects": len(grouped), "sheets": len(manifest), "manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
