#!/usr/bin/env python3
"""Download Supporting Information PDFs for all papers in data/ml_lifetime/papers/.

Strategy per publisher:
- ACS (10.1021): scrape suppl page for PDF links
- Wiley (10.1002): scrape article page for SI links
- RSC (10.1039): scrape article page for ESI link
- Others: resolve DOI, try to find SI links on landing page
"""

import os, re, sys, time, json
import requests
from pathlib import Path
from urllib.parse import urljoin

PAPERS_DIR = Path("data/ml_lifetime/papers")
SI_DIR = PAPERS_DIR / "SI"
SI_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Complete DOI mapping (extracted from PDFs + manual corrections)
DOI_MAP = {
    "Asai et al. 2011 - Switching reaction pathways of Benzo[b]thiophen-3-yll": "10.1246/cl.2011.393",
    "Asai et al. 2012 - Practical synthesis of photochromic diarylethenes in integrated flow microreactor systems": "10.1002/cssc.201100376",
    "CHLiIF-carbenoid_2020_Nagaki": "10.1002/anie.202003831",
    "Degennaro et al. 2016 - A direct and sustainable synthesis of tertiary butyl esters enabled by flow microreactors": "10.1039/c6cc04588j",
    "Kim et al. 2011 - A flow-microreactor approach to protecting-group-free synthesis using organolithium compounds": "10.1038/ncomms1264",
    "Miyagishi et al. 2024 - Expanding the scope of C-glycoside synthesis from unstable organolithium reagents using flow microreactors": "10.1021/acs.orglett.4c01698",
    "Musci et al. 2020 - Flow microreactor technology for taming highly react": "10.1021/acs.orglett.0c01085",
    "Nagaki et al. 2007 - Integrated micro flow synthesis based on sequential Br-Li exchange reactions of p-, m-, and o-dibromobenzenes": "10.1002/asia.200700231",
    "Nagaki et al. 2008 - Aryllithium compounds bearing alkoxycarbonyl groups - generation and reactions using a microflow system": "10.1002/anie.200803205",
    "Nagaki et al. 2009 - Generation and reactions of α-silyloxiranyllithium in a microreactor": "10.1246/cl.2009.486",
    "Nagaki et al. 2009 - Generations and reactions ofN-(t-butylsulfonyl)aziridinyllithiums using microreactors": "10.1246/cl.2009.1060",
    "Nagaki et al. 2009 - Nitro-substituted aryl lithium compounds in microreactor synthesis": "10.1002/anie.200904316",
    "Nagaki et al. 2009 - Synthesis of unsymmetrically substituted biaryls vi": "10.3762/bjoc.5.16",
    "Nagaki et al. 2010 - A flow microreactor system enables organolithium reactions without protecting alkoxycarbonyl groups": "10.1002/chem.201000876",
    "Nagaki et al. 2010 - Cross-coupling in a flow microreactor - space integration of lithiation and Murahashi coupling": "10.1002/anie.201002763",
    "Nagaki et al. 2010 - Generation and reaction of cyano-substituted aryllithium compounds using microreactors": "10.1039/b919325c",
    "Nagaki et al. 2010 - Generation and reactions of oxiranyllithiums by use of a flow microreactor system": "10.1002/chem.201000815",
    "Nagaki et al. 2011 - Flash synthesis of TAC-101 and its analogues from 1,3,5-tribromobenzene using integrated flow microreactor systems": "10.1039/c1ra00377a",
    "Nagaki et al. 2011 - Flow microreactor synthesis of disubstituted pyridi": "10.1039/c0gc00852d",
    "Nagaki et al. 2011 - Homocoupling of aryl halides in flow - Space integration of lithiation and FeCl(3) promoted homocoupling": "10.3762/bjoc.7.122",
    "Nagaki et al. 2011 - Perfluoroalkylation in flow microreactors - generation of perfluoroalkyllithiums in the presence and absence of electrophiles": "10.1039/c1ob06350b",
    "Nagaki et al. 2012 - Generation and reactions of vinyllithiums using flow microreactor systems": "10.1556/JFC-D-12-00004",
    "Nagaki et al. 2013 - Generation and reactions of pyridyllithiums via Br - Li exchange reactions using continuous flow microreactor systems": "10.1071/CH12440",
    "Nagaki et al. 2014 - Three-component coupling based on flash chemistry": "10.1021/ja5071762",
    "Nagaki et al. 2015 - Benzyllithiums bearing aldehyde carbonyl groups": "10.1039/c5ob00958h",
    "Nagaki et al. 2016 - Integration of borylation of aryllithiums and Suzuki": "10.1039/c5cy02098k",
    "Nagaki et al. 2019 - Alkyllithium compounds bearing electrophilic functional groups": "10.1002/anie.201814088",
    "Nagaki et al. 2019 - Generation and reaction of functional alkyllithiu": "10.1002/chem.201902867",
    "Okamoto et al. 2026 - Reductive generation of anionic C1 carbenoid spe": "10.1002/asia.70525",
    "PhLi-borylation_2016_Nagaki": "10.3390/catal9030300",
    "Sun et al. 2020 - Practical and rapid construction of 2-pyridyl ketone library in continuous flow": "10.1007/s41981-020-00120-7",
    "Usutani et al. 2007 - Generation and reactions of o-bromophenyllithium": "10.1021/ja074330h",
}


def match_pdf_to_doi(fname):
    """Match a PDF filename to its DOI using prefix matching."""
    stem = fname.replace(".pdf", "")
    for key, doi in DOI_MAP.items():
        if stem.startswith(key) or key.startswith(stem[:40]):
            return doi
    return None


def safe_filename(doi):
    """Convert DOI to safe filename."""
    return doi.replace("/", "_").replace(".", "-") + "_SI.pdf"


def download_acs_si(doi, session):
    """Download SI from ACS (pubs.acs.org)."""
    # ACS suppl page lists SI files
    suppl_url = f"https://pubs.acs.org/doi/suppl/{doi}"
    resp = session.get(suppl_url, headers=HEADERS, timeout=30, allow_redirects=True)
    if resp.status_code != 200:
        return None, f"suppl page {resp.status_code}"

    # Find PDF links in suppl page
    # Pattern: /doi/suppl/10.1021/xxx/suppl_file/xxx.pdf
    pdf_links = re.findall(r'href="(/doi/suppl/[^"]+\.pdf)"', resp.text)
    if not pdf_links:
        # Also try direct SI PDF pattern
        pdf_links = re.findall(r'href="([^"]+suppl_file[^"]+\.pdf)"', resp.text)

    if not pdf_links:
        return None, "no SI PDF links found on suppl page"

    # Download the first (usually main) SI PDF
    pdf_url = urljoin("https://pubs.acs.org", pdf_links[0])
    pdf_resp = session.get(pdf_url, headers=HEADERS, timeout=60, allow_redirects=True)
    if pdf_resp.status_code == 200 and len(pdf_resp.content) > 1000:
        return pdf_resp.content, None
    return None, f"SI PDF download failed: {pdf_resp.status_code}"


def download_wiley_si(doi, session):
    """Download SI from Wiley (onlinelibrary.wiley.com)."""
    # Resolve DOI to get the actual article URL
    article_url = f"https://onlinelibrary.wiley.com/doi/{doi}"
    resp = session.get(article_url, headers=HEADERS, timeout=30, allow_redirects=True)
    if resp.status_code != 200:
        return None, f"article page {resp.status_code}"

    # Find SI links - Wiley uses various patterns
    # Pattern 1: /action/downloadSupplement
    si_links = re.findall(r'href="(/action/downloadSupplement\?doi=[^"]+)"', resp.text)
    # Pattern 2: direct PDF link with "supporting" or "suppl" in filename
    si_links += re.findall(r'href="([^"]*(?:support|suppl|supp)[^"]*\.pdf)"', resp.text, re.IGNORECASE)

    if not si_links:
        return None, "no SI links found on article page"

    pdf_url = urljoin(resp.url, si_links[0])
    pdf_resp = session.get(pdf_url, headers=HEADERS, timeout=60, allow_redirects=True)
    if pdf_resp.status_code == 200 and len(pdf_resp.content) > 1000:
        return pdf_resp.content, None
    return None, f"SI download failed: {pdf_resp.status_code}"


def download_rsc_si(doi, session):
    """Download SI from RSC (pubs.rsc.org)."""
    article_url = f"https://pubs.rsc.org/en/content/articlelanding/{doi}"
    resp = session.get(article_url, headers=HEADERS, timeout=30, allow_redirects=True)
    if resp.status_code != 200:
        # Try DOI redirect
        resp = session.get(f"https://doi.org/{doi}", headers=HEADERS, timeout=30, allow_redirects=True)
        if resp.status_code != 200:
            return None, f"article page {resp.status_code}"

    # RSC ESI pattern: /suppdata/...
    si_links = re.findall(r'href="([^"]*suppdata[^"]*\.pdf)"', resp.text, re.IGNORECASE)
    if not si_links:
        # Try "Electronic supplementary information" link
        si_links = re.findall(r'href="(/en/content/articlepdf[^"]*)"', resp.text)

    if not si_links:
        return None, "no ESI links found"

    pdf_url = urljoin(resp.url, si_links[0])
    pdf_resp = session.get(pdf_url, headers=HEADERS, timeout=60, allow_redirects=True)
    if pdf_resp.status_code == 200 and len(pdf_resp.content) > 1000:
        return pdf_resp.content, None
    return None, f"ESI download failed: {pdf_resp.status_code}"


def download_generic_si(doi, session):
    """Try to download SI by resolving DOI and looking for links."""
    # Resolve DOI
    resp = session.get(f"https://doi.org/{doi}", headers=HEADERS, timeout=30, allow_redirects=True)
    if resp.status_code != 200:
        return None, f"DOI resolve failed: {resp.status_code}"

    landing_url = resp.url

    # Look for SI/supporting links
    patterns = [
        r'href="([^"]*(?:support|suppl|supplementar|ESI|SI)[^"]*\.pdf)"',
        r'href="([^"]*(?:support|suppl|supplementar)[^"]*)"',
    ]

    for pattern in patterns:
        links = re.findall(pattern, resp.text, re.IGNORECASE)
        for link in links:
            if any(skip in link.lower() for skip in ['javascript', '#', 'mailto']):
                continue
            pdf_url = urljoin(landing_url, link)
            if pdf_url.endswith('.pdf') or 'suppl' in pdf_url.lower():
                try:
                    pdf_resp = session.get(pdf_url, headers=HEADERS, timeout=60, allow_redirects=True)
                    content_type = pdf_resp.headers.get('Content-Type', '')
                    if pdf_resp.status_code == 200 and (
                        'pdf' in content_type or len(pdf_resp.content) > 10000
                    ):
                        return pdf_resp.content, None
                except:
                    continue

    return None, f"no SI found on {landing_url[:60]}"


def download_si(doi, session):
    """Route to publisher-specific downloader."""
    if doi.startswith("10.1021/"):
        return download_acs_si(doi, session)
    elif doi.startswith("10.1002/"):
        return download_wiley_si(doi, session)
    elif doi.startswith("10.1039/"):
        return download_rsc_si(doi, session)
    else:
        return download_generic_si(doi, session)


def main():
    session = requests.Session()

    results = {"success": [], "failed": []}

    # Build fname -> doi mapping
    pdf_files = sorted(f for f in os.listdir(PAPERS_DIR) if f.endswith(".pdf"))

    for fname in pdf_files:
        doi = match_pdf_to_doi(fname)
        if not doi:
            print(f"  SKIP  {fname[:60]} — no DOI")
            results["failed"].append({"file": fname, "reason": "no DOI mapping"})
            continue

        # Check if already downloaded
        outname = safe_filename(doi)
        outpath = SI_DIR / outname
        if outpath.exists():
            print(f"  EXIST {fname[:60]} — {outname}")
            results["success"].append({"file": fname, "doi": doi, "si_file": outname})
            continue

        print(f"  FETCH {fname[:55]} DOI={doi}")
        try:
            content, error = download_si(doi, session)
            if content:
                outpath.write_bytes(content)
                size_kb = len(content) / 1024
                print(f"     OK  {outname} ({size_kb:.0f} KB)")
                results["success"].append({"file": fname, "doi": doi, "si_file": outname})
            else:
                print(f"     FAIL {error}")
                results["failed"].append({"file": fname, "doi": doi, "reason": error})
        except Exception as e:
            print(f"     ERROR {e}")
            results["failed"].append({"file": fname, "doi": doi, "reason": str(e)})

        time.sleep(2)  # polite delay between requests

    # Summary
    print(f"\n{'='*60}")
    print(f"Downloaded: {len(results['success'])}/{len(pdf_files)}")
    print(f"Failed: {len(results['failed'])}/{len(pdf_files)}")
    if results["failed"]:
        print("\nFailed papers:")
        for f in results["failed"]:
            print(f"  - {f['file'][:60]}")
            print(f"    DOI: {f.get('doi', 'N/A')}  Reason: {f['reason']}")

    # Save results
    with open(SI_DIR / "download_log.json", "w") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
