"""
01_scrape_talks.py
==================
Scrape all General Conference talks (1971–END_YEAR) from
churchofjesuschrist.org and save each talk as a JSON file under
data/talks/{year}_{month:02d}_{slug}.json.

Strategy
--------
1. Build the list of (year, month) conference pairs.
2. For each conference, fetch the index page via the content API to get
   a list of talk URIs.
3. For each talk URI, fetch the talk page and parse out speaker, title,
   and full paragraph text.
4. Cache raw API responses in data/raw/ so re-runs are fast.

Run
---
    python 01_scrape_talks.py
"""

import json
import re
import time
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    BASE_URL, CONTENT_API, LANG,
    START_YEAR, END_YEAR, MONTHS,
    REQUEST_DELAY, REQUEST_TIMEOUT, MAX_RETRIES,
    RAW_DIR, TALKS_DIR,
)

# ── HTTP session ──────────────────────────────────────────────────────────────
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (academic research bot; "
        "contact: research@example.com)"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "application/json, text/html",
})


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _get_json(url: str, params: dict) -> dict:
    """Fetch JSON from the church content API."""
    resp = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _get_html(url: str) -> str:
    """Fetch raw HTML."""
    resp = SESSION.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.text


# ── Conference index ──────────────────────────────────────────────────────────

def conference_uri(year: int, month: int) -> str:
    return f"/general-conference/{year}/{month:02d}"


def cached_raw_path(slug: str) -> Path:
    """Path for a cached raw API JSON response."""
    return RAW_DIR / f"{slug}.json"


def fetch_conference_index(year: int, month: int) -> list[dict]:
    """
    Return a list of talk metadata dicts for one conference.
    Each dict has at least: uri, title, speaker.
    """
    uri = conference_uri(year, month)
    cache_path = cached_raw_path(f"index_{year}_{month:02d}")

    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
    else:
        try:
            data = _get_json(CONTENT_API, {"lang": LANG, "uri": uri})
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as exc:
            print(f"  [warn] Could not fetch index {year}/{month:02d}: {exc}")
            return []
        time.sleep(REQUEST_DELAY)

    return _parse_conference_index(data, year, month)


def _parse_conference_index(data: dict, year: int, month: int) -> list[dict]:
    """
    Extract talk URIs from the conference index API response.
    The API returns HTML in data['content']['body'] with links like:
      /study/general-conference/YYYY/MM/slug?lang=eng
    We strip the /study prefix and ?lang=... suffix, then skip session
    overview pages (they have no talk slug: only 4 slash-parts after strip).
    """
    body_html = data.get("content", {}).get("body", "")
    if not body_html:
        return []

    soup = BeautifulSoup(body_html, "lxml")
    items: list[dict] = []
    seen: set[str] = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].split("?")[0]   # strip ?lang=eng
        # Normalise: remove /study prefix if present
        if href.startswith("/study/"):
            href = href[len("/study"):]

        # Valid individual talk URIs: /general-conference/YYYY/MM/slug  (5 parts)
        parts = href.strip("/").split("/")
        if len(parts) != 4 or parts[0] != "general-conference":
            continue
        if href in seen:
            continue
        seen.add(href)

        # Speaker in <p class="primaryMeta">, title in <p class="title">
        speaker_el = a_tag.select_one(".primaryMeta")
        title_el   = a_tag.select_one(".title")
        speaker = speaker_el.get_text(" ", strip=True) if speaker_el else ""
        title   = title_el.get_text(" ", strip=True)   if title_el   else ""

        items.append({
            "uri":     href,
            "title":   title,
            "speaker": speaker,
            "year":    year,
            "month":   month,
        })

    return items


# ── Individual talk ───────────────────────────────────────────────────────────

def fetch_talk(meta: dict) -> dict | None:
    """
    Fetch and parse a single talk. Returns an enriched metadata dict
    with a 'text' key containing the full talk as a list of paragraphs,
    or None on failure.
    """
    uri   = meta["uri"]
    slug  = uri.strip("/").replace("/", "_")
    cache = cached_raw_path(f"talk_{slug}")
    out   = TALKS_DIR / f"{slug}.json"

    if out.exists():
        return None  # already processed

    if cache.exists():
        with open(cache, encoding="utf-8") as f:
            raw = json.load(f)
    else:
        try:
            raw = _get_json(CONTENT_API, {"lang": LANG, "uri": uri})
            with open(cache, "w", encoding="utf-8") as f:
                json.dump(raw, f)
        except Exception as exc:
            print(f"  [warn] API failed for {uri}: {exc}. Trying HTML…")
            raw = None
        time.sleep(REQUEST_DELAY)

    # Try content API path first
    paragraphs = []
    speaker    = meta.get("speaker", "")
    title      = meta.get("title", "")

    if raw:
        paragraphs, speaker, title = _parse_api_talk(raw, speaker, title)

    # Fall back to HTML scraping if API gave no body text
    if not paragraphs:
        full_url = BASE_URL + uri + "?lang=" + LANG
        try:
            html = _get_html(full_url)
            paragraphs, speaker, title = _parse_html_talk(html, speaker, title)
            time.sleep(REQUEST_DELAY)
        except Exception as exc:
            print(f"  [warn] HTML fallback failed for {uri}: {exc}")
            return None

    if not paragraphs:
        return None

    talk = {
        "uri":        uri,
        "year":       meta["year"],
        "month":      meta["month"],
        "speaker":    speaker,
        "title":      title,
        "paragraphs": paragraphs,
        "text":       " ".join(paragraphs),
        "word_count": sum(len(p.split()) for p in paragraphs),
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(talk, f, ensure_ascii=False, indent=2)

    return talk


def _parse_api_talk(
    data: dict,
    default_speaker: str,
    default_title: str,
) -> tuple[list[str], str, str]:
    """Extract paragraphs, speaker, title from a content-API response."""
    paragraphs: list[str] = []
    speaker = default_speaker
    title   = default_title

    # Locate the HTML body block inside the API response
    def _walk(node):
        nonlocal speaker, title
        if isinstance(node, list):
            for child in node:
                _walk(child)
        elif isinstance(node, dict):
            # Some API versions put speaker/title at top level
            if node.get("type") == "speaker" and node.get("text"):
                speaker = node["text"]
            if node.get("type") in ("head", "title") and node.get("text") and not title:
                title = node["text"]
            if node.get("type") == "p" and node.get("text"):
                paragraphs.append(node["text"].strip())
            # Recurse into known children keys
            for key in ("body", "content", "children", "blocks", "items",
                        "meta", "header", "footer", "sections"):
                if key in node:
                    _walk(node[key])

    _walk(data)

    # If the API returned an HTML content block, parse it
    if not paragraphs:
        html_body = _find_html_body(data)
        if html_body:
            soup = BeautifulSoup(html_body, "lxml")
            paragraphs = _extract_paragraphs(soup)
            if not speaker:
                speaker = _find_speaker_in_soup(soup)
            if not title:
                title = _find_title_in_soup(soup)

    return paragraphs, speaker, title


def _find_html_body(node, depth: int = 0) -> str | None:
    """Recursively search for an HTML string inside the API response."""
    if depth > 15:
        return None
    if isinstance(node, str) and "<p" in node:
        return node
    if isinstance(node, list):
        for child in node:
            result = _find_html_body(child, depth + 1)
            if result:
                return result
    elif isinstance(node, dict):
        for v in node.values():
            result = _find_html_body(v, depth + 1)
            if result:
                return result
    return None


def _extract_paragraphs(soup: BeautifulSoup) -> list[str]:
    """Pull clean paragraph text from a BeautifulSoup object."""
    # Remove footnotes / references
    for tag in soup.select(".marker, .footnote, .copyright, nav, header, footer"):
        tag.decompose()

    paras = []
    for p in soup.find_all("p"):
        txt = p.get_text(" ", strip=True)
        # Skip very short or boilerplate paragraphs
        if len(txt) > 40:
            paras.append(txt)
    return paras


# Regex to strip leading "By " / "By Elder " / "By President " etc.
_BY_PREFIX = re.compile(r"^By\s+", re.IGNORECASE)


def _find_speaker_in_soup(soup: BeautifulSoup) -> str:
    for sel in (".author-name", ".byline", "[class*='speaker']", "[class*='author']", "p.author"):
        el = soup.select_one(sel)
        if el:
            raw = el.get_text(" ", strip=True)
            # author-name contains the full calling string; take the first sentence
            # and strip the leading "By " prefix
            first_line = raw.split("\n")[0].strip()
            return _BY_PREFIX.sub("", first_line).strip()
    return ""


def _find_title_in_soup(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    return h1.get_text(" ", strip=True) if h1 else ""


def _parse_html_talk(
    html: str,
    default_speaker: str,
    default_title: str,
) -> tuple[list[str], str, str]:
    """Parse a talk page fetched as raw HTML."""
    soup      = BeautifulSoup(html, "lxml")
    speaker   = _find_speaker_in_soup(soup) or default_speaker
    title     = _find_title_in_soup(soup)   or default_title
    paragraphs = _extract_paragraphs(soup)
    return paragraphs, speaker, title


# ── Main ──────────────────────────────────────────────────────────────────────

def build_conference_list() -> list[tuple[int, int]]:
    conferences = []
    for year in range(START_YEAR, END_YEAR + 1):
        for month in MONTHS:
            conferences.append((year, month))
    return conferences


def main():
    conferences = build_conference_list()
    print(f"Scraping {len(conferences)} conferences ({START_YEAR}–{END_YEAR})…")

    total_talks = 0
    total_new   = 0

    for year, month in tqdm(conferences, desc="Conferences", unit="conf"):
        talk_metas = fetch_conference_index(year, month)
        if not talk_metas:
            continue

        total_talks += len(talk_metas)

        for meta in tqdm(
            talk_metas,
            desc=f"  {year}/{month:02d}",
            leave=False,
            unit="talk",
        ):
            result = fetch_talk(meta)
            if result is not None:
                total_new += 1

    # Summary
    existing = len(list(TALKS_DIR.glob("*.json")))
    print(f"\nDone. {existing} talks on disk ({total_new} new this run).")


if __name__ == "__main__":
    main()
