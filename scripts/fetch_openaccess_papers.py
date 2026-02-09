#!/usr/bin/env python3
"""
Fetch open-access PDFs from PMC OA subset, ChemRxiv (via Europe PMC preprints), and arXiv.
Stores PDFs under data/papers/{pmc,chemrxiv,arxiv} and writes metadata files.
"""
from __future__ import annotations

import csv
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

BASE_DIR = "/Users/zhaowenyuan/Projects/FlowFigTabMiner/data/papers"
TARGET_TOTAL = 120
TARGET_PMC = 70
TARGET_CHEMRXIV = 25
TARGET_ARXIV = 25
REQUEST_TIMEOUT = 20

KEYWORDS = [
    "substrate scope flow chemistry",
    "optimization of reaction conditions catalyst",
    "organolithium-based synthesis",
    "flow hydrogenation Pd catalyst table",
    "library synthesis flow table",
]

EUROPE_PMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPE_PMC_FIELDS = "https://www.ebi.ac.uk/europepmc/webservices/rest/fields"
PMC_OA_API = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_DATE_FROM = "201001010000"
ARXIV_DATE_TO = "202612312359"

UA = "FlowFigTabMiner/1.0 (mailto:local)"


def http_get(url: str, params: dict | None = None, timeout: int = REQUEST_TIMEOUT) -> bytes:
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def ensure_dirs() -> None:
    for sub in ("pmc", "chemrxiv", "arxiv"):
        os.makedirs(os.path.join(BASE_DIR, sub), exist_ok=True)


def sanitize_filename(text: str, max_len: int = 120) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text[:max_len].strip("_")


def load_epmc_fields() -> set[str]:
    try:
        print("Loading Europe PMC fields...")
        data = http_get(EUROPE_PMC_FIELDS)
        js = json.loads(data.decode("utf-8"))
        fields = {f.get("name") for f in js.get("fields", []) if f.get("name")}
        print(f"Fields loaded: {len(fields)}")
        return fields
    except Exception as e:
        print(f"Failed to load fields: {e}")
        return set()


def epmc_search(query: str, page_size: int = 100, max_records: int = 200) -> list[dict]:
    results: list[dict] = []
    cursor = "*"
    while len(results) < max_records:
        params = {
            "query": query,
            "format": "json",
            "pageSize": str(page_size),
            "cursorMark": cursor,
        }
        data = http_get(EUROPE_PMC_SEARCH, params=params)
        js = json.loads(data.decode("utf-8"))
        items = js.get("resultList", {}).get("result", [])
        if not items:
            break
        results.extend(items)
        next_cursor = js.get("nextCursorMark")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        time.sleep(0.5)
    return results[:max_records]


def get_pmc_pdf_url(pmcid: str) -> str | None:
    try:
        data = http_get(PMC_OA_API, params={"id": pmcid})
        root = ET.fromstring(data)
        for link in root.findall(".//link"):
            if link.get("format") == "pdf":
                href = link.get("href")
                if not href:
                    continue
                if href.startswith("ftp://"):
                    return href.replace("ftp://", "https://")
                return href
    except Exception:
        return None
    return None


def pick_fulltext_pdf_url(fulltext_list: dict | None) -> str | None:
    if not fulltext_list:
        return None
    urls = fulltext_list.get("fullTextUrl", [])
    for item in urls:
        if item.get("documentStyle", "").lower() == "pdf":
            return item.get("url")
    for item in urls:
        url = item.get("url", "")
        if url.lower().endswith(".pdf"):
            return url
    return None


def is_article_or_preprint(item: dict) -> bool:
    pt = item.get("pubTypeList", {}).get("pubType")
    if isinstance(pt, list):
        types = " ".join(pt).lower()
    elif isinstance(pt, str):
        types = pt.lower()
    else:
        types = ""
    if "preprint" in types:
        return True
    if "journal article" in types or "article" in types:
        return True
    # fallback: if no type for PMC, accept
    if item.get("source") == "PMC":
        return True
    return False


def save_file(url: str, path: str) -> bool:
    try:
        if url.startswith("ftp://"):
            url = url.replace("ftp://", "https://")
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = resp.read()
        with open(path, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"Download failed: {url} ({e})")
        return False


def fetch_pmc(target: int, keywords: list[str]) -> list[dict]:
    collected = []
    seen = set()
    for kw in keywords:
        print(f"PMC query: {kw}")
        query = f"({kw}) AND SRC:PMC AND OPEN_ACCESS:Y AND FIRST_PDATE:[2010-01-01 TO 2026-12-31]"
        items = epmc_search(query, max_records=150)
        for item in items:
            if len(collected) >= target:
                return collected
            if not is_article_or_preprint(item):
                continue
            pmcid = item.get("pmcid") or item.get("id")
            if not pmcid or not pmcid.startswith("PMC"):
                continue
            if pmcid in seen:
                continue
            pdf_url = get_pmc_pdf_url(pmcid)
            if not pdf_url:
                continue
            title = item.get("title") or ""
            year = item.get("pubYear") or ""
            filename = sanitize_filename(f"{pmcid}_{year}") + ".pdf"
            out_path = os.path.join(BASE_DIR, "pmc", filename)
            if os.path.exists(out_path):
                seen.add(pmcid)
                continue
            ok = save_file(pdf_url, out_path)
            if not ok:
                continue
            seen.add(pmcid)
            collected.append({
                "source": "PMC",
                "id": pmcid,
                "title": title,
                "year": year,
                "pdf_url": pdf_url,
                "keyword": kw,
                "path": out_path,
            })
            print(f"PMC downloaded: {pmcid} ({len(collected)}/{target})")
            time.sleep(0.5)
    return collected


def fetch_chemrxiv(target: int, keywords: list[str], fields: set[str]) -> list[dict]:
    collected = []
    seen = set()
    server_filter = ""
    if "SERVER" in fields:
        server_filter = " AND SERVER:ChemRxiv"
    for kw in keywords:
        print(f"ChemRxiv query: {kw}")
        query = (
            f"({kw}) AND SRC:PPR AND OPEN_ACCESS:Y AND FIRST_PDATE:[2010-01-01 TO 2026-12-31]"
            f"{server_filter}"
        )
        items = epmc_search(query, max_records=200)
        for item in items:
            if len(collected) >= target:
                return collected
            if not is_article_or_preprint(item):
                continue
            title = item.get("title") or ""
            journal = item.get("journalTitle") or ""
            if "chemrxiv" not in journal.lower() and server_filter == "":
                continue
            doc_id = item.get("id")
            if not doc_id or doc_id in seen:
                continue
            pdf_url = pick_fulltext_pdf_url(item.get("fullTextUrlList"))
            if not pdf_url:
                continue
            year = item.get("pubYear") or ""
            filename = sanitize_filename(f"{doc_id}_{year}") + ".pdf"
            out_path = os.path.join(BASE_DIR, "chemrxiv", filename)
            if os.path.exists(out_path):
                seen.add(doc_id)
                continue
            ok = save_file(pdf_url, out_path)
            if not ok:
                continue
            seen.add(doc_id)
            collected.append({
                "source": "ChemRxiv",
                "id": doc_id,
                "title": title,
                "year": year,
                "pdf_url": pdf_url,
                "keyword": kw,
                "path": out_path,
            })
            print(f"ChemRxiv downloaded: {doc_id} ({len(collected)}/{target})")
            time.sleep(0.5)
    return collected


def arxiv_search(query: str, start: int = 0, max_results: int = 50) -> list[dict]:
    q = f"all:{query} AND submittedDate:[{ARXIV_DATE_FROM} TO {ARXIV_DATE_TO}]"
    params = {
        "search_query": q,
        "start": str(start),
        "max_results": str(max_results),
    }
    data = http_get(ARXIV_API, params=params)
    root = ET.fromstring(data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = []
    for entry in root.findall("atom:entry", ns):
        entry_id = entry.findtext("atom:id", default="", namespaces=ns)
        title = entry.findtext("atom:title", default="", namespaces=ns).strip().replace("\n", " ")
        published = entry.findtext("atom:published", default="", namespaces=ns)
        year = published[:4] if published else ""
        if not entry_id:
            continue
        arxiv_id = entry_id.rsplit("/", 1)[-1]
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        entries.append({
            "id": arxiv_id,
            "title": title,
            "year": year,
            "pdf_url": pdf_url,
        })
    return entries


def fetch_arxiv(target: int, keywords: list[str]) -> list[dict]:
    collected = []
    seen = set()
    for kw in keywords:
        print(f"arXiv query: {kw}")
        # use a compact query for arXiv
        query = " ".join([f'"{w}"' if " " in w else w for w in kw.split()])
        start = 0
        while len(collected) < target and start < 200:
            entries = arxiv_search(query, start=start, max_results=50)
            if not entries:
                break
            for item in entries:
                if len(collected) >= target:
                    return collected
                arxiv_id = item["id"]
                if arxiv_id in seen:
                    continue
                year = item["year"]
                if year and int(year) < 2010:
                    continue
                filename = sanitize_filename(f"{arxiv_id}_{year}") + ".pdf"
                out_path = os.path.join(BASE_DIR, "arxiv", filename)
                if os.path.exists(out_path):
                    seen.add(arxiv_id)
                    continue
                ok = save_file(item["pdf_url"], out_path)
                if not ok:
                    continue
                seen.add(arxiv_id)
                collected.append({
                    "source": "arXiv",
                    "id": arxiv_id,
                    "title": item["title"],
                    "year": year,
                    "pdf_url": item["pdf_url"],
                    "keyword": kw,
                    "path": out_path,
                })
                print(f"arXiv downloaded: {arxiv_id} ({len(collected)}/{target})")
                time.sleep(0.5)
            start += 50
    return collected


def write_metadata(rows: list[dict]) -> None:
    jsonl_path = os.path.join(BASE_DIR, "metadata.jsonl")
    csv_path = os.path.join(BASE_DIR, "metadata.csv")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source", "id", "title", "year", "pdf_url", "keyword", "path"
        ])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ensure_dirs()
    fields = load_epmc_fields()

    all_rows = []

    print("Starting PMC downloads...")
    pmc_rows = fetch_pmc(TARGET_PMC, KEYWORDS)
    all_rows.extend(pmc_rows)

    print("Starting ChemRxiv downloads...")
    chem_rows = fetch_chemrxiv(TARGET_CHEMRXIV, KEYWORDS, fields)
    all_rows.extend(chem_rows)

    print("Starting arXiv downloads...")
    arxiv_rows = fetch_arxiv(TARGET_ARXIV, KEYWORDS)
    all_rows.extend(arxiv_rows)

    # If we still have fewer than 100, try to top up from PMC
    if len(all_rows) < 100:
        print("Topping up from PMC...")
        extra = fetch_pmc(100 - len(all_rows), KEYWORDS)
        all_rows.extend(extra)

    write_metadata(all_rows)
    print(f"Downloaded {len(all_rows)} PDFs into {BASE_DIR}")


if __name__ == "__main__":
    main()
