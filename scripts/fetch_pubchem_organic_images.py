#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont

PUG_REST_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

PROPERTIES = [
    "MolecularFormula",
    "MolecularWeight",
    "IUPACName",
    "CanonicalSMILES",
    "IsomericSMILES",
    "XLogP",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "HeavyAtomCount",
]


def esearch_cids(term: str, target_pool: int, sleep_s: float) -> List[int]:
    params = {
        "db": "pccompound",
        "term": term,
        "retmax": min(target_pool, 10000),
        "retstart": 0,
        "retmode": "json",
    }

    resp = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=40)
    resp.raise_for_status()
    data = resp.json()

    count = int(data["esearchresult"].get("count", 0))
    ids = [int(x) for x in data["esearchresult"].get("idlist", []) if x.isdigit()]

    retstart = len(ids)
    while len(ids) < target_pool and retstart < count:
        params["retstart"] = retstart
        params["retmax"] = min(10000, target_pool - len(ids))
        resp = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=40)
        resp.raise_for_status()
        data = resp.json()
        new_ids = [int(x) for x in data["esearchresult"].get("idlist", []) if x.isdigit()]
        if not new_ids:
            break
        ids.extend(new_ids)
        retstart += len(new_ids)
        time.sleep(sleep_s)

    return ids


def fetch_properties(cid: int) -> Optional[Dict[str, str]]:
    props = ",".join(PROPERTIES)
    url = f"{PUG_REST_BASE}/compound/cid/{cid}/property/{props}/JSON"
    try:
        resp = requests.get(url, timeout=40)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception:
        return None

    if "Fault" in data:
        return None
    try:
        props = data["PropertyTable"]["Properties"][0]
    except Exception:
        return None

    out = {}
    for key in PROPERTIES:
        if key in props:
            out[key] = str(props[key])
    return out


def is_organic(props: Dict[str, str]) -> bool:
    formula = props.get("MolecularFormula", "")
    if "C" not in formula:
        return False
    return True


def download_structure_png(cid: int, out_path: str) -> bool:
    url = f"{PUG_REST_BASE}/compound/cid/{cid}/record/PNG?record_type=2d"
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            return False
        if not resp.content:
            return False
    except Exception:
        return False

    with open(out_path, "wb") as f:
        f.write(resp.content)
    return True


def render_table_image(cid: int, props: Dict[str, str], out_path: str) -> None:
    rows = [("CID", str(cid))] + [(k, props.get(k, "")) for k in PROPERTIES]

    font = ImageFont.load_default()
    padding = 16
    col1_width = max(font.getlength(r[0]) for r in rows) + 20
    col2_width = 640
    row_height = 18

    width = int(col1_width + col2_width + padding * 2)
    height = int(row_height * len(rows) + padding * 2 + 10)

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    x0 = padding
    y0 = padding

    for i, (k, v) in enumerate(rows):
        y = y0 + i * row_height
        draw.text((x0, y), k, fill="black", font=font)
        draw.text((x0 + col1_width, y), v, fill="black", font=font)

    for i in range(len(rows) + 1):
        y = y0 + i * row_height - 2
        draw.line((x0 - 6, y, width - padding + 6, y), fill="gray", width=1)
    draw.line((x0 + col1_width - 6, y0 - 6, x0 + col1_width - 6, y0 + len(rows) * row_height), fill="gray", width=1)

    img.save(out_path, format="PNG")

def render_combo_image(struct_paths: List[str], out_path: str, min_count: int, max_count: int) -> bool:
    if not struct_paths:
        return False

    count = random.randint(min_count, max_count)
    chosen = random.sample(struct_paths, min(count, len(struct_paths)))

    canvas_w = 1200
    canvas_h = 700
    padding = 20
    top_line_y = padding + 10
    bottom_line_y = canvas_h - padding - 10

    combo = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(combo)
    draw.line((padding, top_line_y, canvas_w - padding, top_line_y), fill="black", width=2)
    draw.line((padding, bottom_line_y, canvas_w - padding, bottom_line_y), fill="black", width=2)

    placed = []
    max_attempts = 40
    for path in chosen:
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue

        scale = random.uniform(0.35, 0.6)
        new_w = max(1, int(img.width * scale))
        new_h = max(1, int(img.height * scale))
        img = img.resize((new_w, new_h))

        for _ in range(max_attempts):
            x = random.randint(padding, canvas_w - padding - new_w)
            y = random.randint(top_line_y + 8, bottom_line_y - new_h - 8)
            bbox = (x, y, x + new_w, y + new_h)

            overlap = False
            for (px0, py0, px1, py1) in placed:
                if not (bbox[2] < px0 or bbox[0] > px1 or bbox[3] < py0 or bbox[1] > py1):
                    overlap = True
                    break
            if overlap:
                continue

            combo.paste(img, (x, y))
            placed.append(bbox)
            break

    if not placed:
        return False

    combo.save(out_path, format="PNG")
    return True

def ensure_dirs(base_dir: str) -> Tuple[str, str]:
    structures_dir = os.path.join(base_dir, "structures")
    tables_dir = os.path.join(base_dir, "tables")
    os.makedirs(structures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    return structures_dir, tables_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download organic molecule structure images and table images from PubChem.")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--count", type=int, default=1000, help="Total images to download")
    parser.add_argument("--table-ratio", type=float, default=0.5, help="Fraction of table images")
    parser.add_argument("--combo", action="store_true", help="Generate table+structure combo images")
    parser.add_argument("--combo-min", type=int, default=1, help="Minimum molecules per combo")
    parser.add_argument("--combo-max", type=int, default=10, help="Maximum molecules per combo")
    parser.add_argument("--sleep", type=float, default=0.2, help="Seconds to sleep between requests")
    parser.add_argument("--term", default="C[Element]", help="ESearch term for pccompound")
    args = parser.parse_args()

    total_count = max(1, args.count)
    table_count = int(total_count * args.table_ratio)
    structure_count = total_count - table_count

    target_pool = max(total_count * 6, 2000)
    print(f"Fetching CIDs via ESearch term: {args.term}")
    candidates = esearch_cids(args.term, target_pool=target_pool, sleep_s=args.sleep)
    if len(candidates) < total_count:
        raise SystemExit(f"Only found {len(candidates)} candidate CIDs; try a different --term.")

    random.shuffle(candidates)

    structures_dir, tables_dir = ensure_dirs(args.out)
    combos_dir = os.path.join(args.out, "combos")
    if args.combo:
        os.makedirs(combos_dir, exist_ok=True)
    metadata_path = os.path.join(args.out, "metadata.csv")

    downloaded_structures = 0
    downloaded_tables = 0
    structure_pool: List[str] = []

    with open(metadata_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["cid", "type", "file", "formula", "name"])

        for cid in candidates:
            if downloaded_structures >= structure_count and downloaded_tables >= table_count:
                break

            props = fetch_properties(cid)
            time.sleep(args.sleep)
            if not props or not is_organic(props):
                continue

            formula = props.get("MolecularFormula", "")
            name = props.get("IUPACName", "")

            struct_path = os.path.join(structures_dir, f"cid_{cid}.png")
            if downloaded_structures < structure_count:
                if download_structure_png(cid, struct_path):
                    writer.writerow([cid, "structure", struct_path, formula, name])
                    downloaded_structures += 1
                    structure_pool.append(struct_path)
                time.sleep(args.sleep)

            if downloaded_tables < table_count:
                table_path = os.path.join(tables_dir, f"cid_{cid}.png")
                render_table_image(cid, props, table_path)
                writer.writerow([cid, "table", table_path, formula, name])
                downloaded_tables += 1
            if args.combo:
                if not os.path.exists(struct_path):
                    if download_structure_png(cid, struct_path):
                        writer.writerow([cid, "structure", struct_path, formula, name])
                        downloaded_structures += 1
                        structure_pool.append(struct_path)
                        time.sleep(args.sleep)
                combo_path = os.path.join(combos_dir, f"combo_{cid}.png")
                pool = structure_pool[-200:] if len(structure_pool) > 200 else structure_pool
                if render_combo_image(pool, combo_path, args.combo_min, args.combo_max):
                    writer.writerow([cid, "combo", combo_path, formula, name])

            if (downloaded_structures + downloaded_tables) % 50 == 0:
                print(f"Progress: structures={downloaded_structures}, tables={downloaded_tables}")

    print("Done.")
    print(f"Structures: {downloaded_structures}")
    print(f"Tables: {downloaded_tables}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
