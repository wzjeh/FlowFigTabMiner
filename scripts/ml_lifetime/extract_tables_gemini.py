#!/usr/bin/env python3
"""
Extract table data from cropped SI images using Gemini 2.5 Pro API.

Usage:
    # Test on a single image
    python scripts/ml_lifetime/extract_tables_gemini.py --test

    # Process one paper's folder
    python scripts/ml_lifetime/extract_tables_gemini.py --only Usutani_2007

    # Process all
    python scripts/ml_lifetime/extract_tables_gemini.py

Requires:
    pip install google-genai Pillow
    export GEMINI_API_KEY=<your-key>
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except ImportError:
    sys.exit("Missing dependency: pip install google-genai")

# ── paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
SI_DIR = ROOT / "data" / "ml_lifetime" / "papers" / "SI"
PROMPT_FILE = ROOT / "data" / "ml_lifetime" / "vlm_table_prompt.txt"
OUTPUT_DIR = ROOT / "data" / "ml_lifetime" / "gemini_table_extractions"

MODEL = "gemini-2.5-pro"


def load_prompt() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8")


def extract_one_image(client, prompt: str, image_path: Path, paper_folder: str, model_name: str = MODEL) -> dict:
    """Send one image to Gemini and return parsed JSON."""
    img_bytes = image_path.read_bytes()
    suffix = image_path.suffix.lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "tif": "image/tiff", "tiff": "image/tiff", "bmp": "image/bmp"}.get(suffix, "image/png")

    # Add context from filename and folder name
    context = (
        f"Paper: {paper_folder}\n"
        f"Image filename: {image_path.name}\n\n"
    )

    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=img_bytes, mime_type=mime),
                    types.Part.from_text(text=context + prompt),
                ],
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=8192,
        ),
    )

    raw = response.text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON object in the response (Gemini sometimes leaks thinking text)
        match = re.search(r'\{[\s\S]*"image_type"[\s\S]*\}', raw)
        if match:
            # Find the balanced braces
            candidate = match.group(0)
            # Try progressively shorter suffixes to find valid JSON
            depth = 0
            end = -1
            for i, ch in enumerate(candidate):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > 0:
                try:
                    data = json.loads(candidate[:end])
                except json.JSONDecodeError:
                    data = {"_raw_response": raw, "_parse_error": True}
            else:
                data = {"_raw_response": raw, "_parse_error": True}
        else:
            data = {"_raw_response": raw, "_parse_error": True}

    data["_source_paper"] = paper_folder
    data["_source_image"] = image_path.name
    return data


def process_folder(client, prompt: str, folder: Path, output_dir: Path, model_name: str = MODEL):
    """Process all images in one paper's SI folder."""
    images = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    )
    if not images:
        print(f"  [skip] No images in {folder.name}")
        return

    paper_out = output_dir / folder.name
    paper_out.mkdir(parents=True, exist_ok=True)

    results = []
    for i, img in enumerate(images, 1):
        print(f"  [{i}/{len(images)}] {img.name[:60]}...", end="", flush=True)
        try:
            data = extract_one_image(client, prompt, img, folder.name, model_name)
            is_ok = "_parse_error" not in data
            print(f" {'OK' if is_ok else 'PARSE_ERROR'}")
        except Exception as e:
            data = {"_error": str(e), "_source_paper": folder.name, "_source_image": img.name}
            print(f" ERROR: {e}")

        results.append(data)

        # Save individual JSON
        safe_name = re.sub(r"[^\w\-.]", "_", img.stem)[:80]
        (paper_out / f"{safe_name}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Rate limit: 1.5s between calls
        if i < len(images):
            time.sleep(1.5)

    # Save combined JSON for this paper
    (paper_out / "_all_tables.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  -> Saved {len(results)} extractions to {paper_out.name}/")


def main():
    parser = argparse.ArgumentParser(description="Extract SI tables via Gemini VLM")
    parser.add_argument("--test", action="store_true", help="Test on one image only")
    parser.add_argument("--only", type=str, help="Process only folders matching this substring")
    parser.add_argument("--first-n", type=int, default=0, help="Process only the first N folders")
    parser.add_argument("--model", type=str, default=MODEL, help=f"Gemini model (default: {MODEL})")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")

    client = genai.Client(api_key=api_key)
    prompt = load_prompt()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_name = args.model

    # Gather folders
    folders = sorted(
        d for d in SI_DIR.iterdir()
        if d.is_dir() and any(d.glob("*.png")) or any(d.glob("*.jpg"))
    )

    if args.only:
        folders = [f for f in folders if args.only.lower() in f.name.lower()]
    if args.first_n > 0:
        folders = folders[:args.first_n]

    if args.test:
        # Pick first image from first folder
        folder = folders[0] if folders else None
        if not folder:
            sys.exit("No folders found")
        img = next((p for p in sorted(folder.iterdir()) if p.suffix.lower() == ".png"), None)
        if not img:
            sys.exit(f"No PNG in {folder}")
        print(f"=== TEST MODE ===")
        print(f"Paper: {folder.name}")
        print(f"Image: {img.name}")
        print()
        data = extract_one_image(client, prompt, img, folder.name, model_name)
        print(json.dumps(data, ensure_ascii=False, indent=2))
        # Also save
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        test_out = OUTPUT_DIR / "_test_result.json"
        test_out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved to {test_out}")
        return

    print(f"Processing {len(folders)} paper folders, {sum(len(list(f.glob('*.png'))) + len(list(f.glob('*.jpg'))) for f in folders)} total images")
    print(f"Output: {OUTPUT_DIR}\n")

    for i, folder in enumerate(folders, 1):
        # Skip folders that already have _all_tables.json
        done_marker = OUTPUT_DIR / folder.name / "_all_tables.json"
        if done_marker.exists():
            print(f"[{i}/{len(folders)}] {folder.name} — already done, skipping")
            continue
        print(f"[{i}/{len(folders)}] {folder.name}")
        process_folder(client, prompt, folder, OUTPUT_DIR, model_name)
        print()

    print("=== ALL DONE ===")


if __name__ == "__main__":
    main()
