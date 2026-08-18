"""Fetch per-stock news from the keyless Eastmoney search API.

It reads the existing ``*_extended.json`` connector snapshots under
``runtime/stock-profiles/`` and fills each ``news`` field from the public
Eastmoney search endpoint. No API key is used.

Usage:
    python tools/fetch_stock_news.py --limit 6
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROFILE_DIR = ROOT / "runtime/stock-profiles"
SEARCH_URL = "https://search-api-web.eastmoney.com/search/jsonp"
JSONP_PATTERN = re.compile(r"^[^(]*\((.*)\)\s*$", flags=re.DOTALL)
TIMEOUT_SECONDS = 20


def build_search_query(keyword: str, *, page_size: int) -> dict:
    """Build the Eastmoney cmsArticleWebOld search request payload."""
    return {
        "uid": "",
        "keyword": keyword,
        "type": ["cmsArticleWebOld"],
        "client": "web",
        "clientType": "web",
        "clientVersion": "curr",
        "param": {
            "cmsArticleWebOld": {
                "searchScope": "default",
                "sort": "time",
                "pageIndex": 1,
                "pageSize": page_size,
                "preTag": "<em>",
                "postTag": "</em>",
            }
        },
    }


def fetch_search_results(keyword: str, *, page_size: int = 20) -> list[dict]:
    """Return raw article records for ``keyword``."""
    payload = json.dumps(build_search_query(keyword, page_size=page_size), ensure_ascii=False)
    url = f"{SEARCH_URL}?cb=jQuery1124&param={urllib.parse.quote(payload)}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://so.eastmoney.com/",
        },
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        text = response.read().decode("utf-8", errors="replace")
    match = JSONP_PATTERN.match(text.strip())
    if not match:
        raise ValueError(f"unexpected JSONP response: {text[:120]!r}")
    document = json.loads(match.group(1))
    return document.get("result", {}).get("cmsArticleWebOld", [])


def clean_markup(value: str) -> str:
    """Remove search-highlight tags from a title or snippet."""
    return re.sub(r"</?em>", "", value).strip()


def normalize_news(articles: list[dict], *, keywords: set[str], limit: int) -> list[dict]:
    """Convert raw Eastmoney articles into dashboard news records."""
    out: list[dict] = []
    seen_titles: set[str] = set()
    for article in articles:
        title = clean_markup(str(article.get("title", "")))
        content = clean_markup(str(article.get("content", "")))
        haystack = f"{title}\n{content}"
        if not any(keyword and keyword in haystack for keyword in keywords):
            continue
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)
        raw_date = str(article.get("date", ""))
        out.append(
            {
                "title": title,
                "source": str(article.get("mediaName", "")),
                "time": raw_date[:10],
                "url": str(article.get("url", "")),
            }
        )
        if len(out) >= limit:
            break
    return out


def search_name_for(symbol: str, name: str) -> str:
    """Prefer the plain company name for search relevance."""
    cleaned = re.sub(r"^(XD|XR|DR)", "", name).strip()
    if cleaned:
        return cleaned
    return symbol.split(".")[0]


def update_profile(path: Path, *, limit: int) -> tuple[str, str, int]:
    """Fetch news for one extended snapshot and write it back in place."""
    document = json.loads(path.read_text(encoding="utf-8"))
    symbol = str(document.get("symbol", ""))
    name = str(document.get("name", ""))
    code = symbol.split(".")[0] if symbol else ""
    query_name = search_name_for(symbol, name)
    keywords = {query_name, code, name}

    articles = fetch_search_results(query_name)
    news = normalize_news(articles, keywords=keywords, limit=limit)
    if not news:
        print(f"  {code} {name}: no matching news", flush=True)
        return symbol, name, 0

    document["news"] = news
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  {code} {name}: {len(news)} news items", flush=True)
    return symbol, name, len(news)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch public dashboard stock news.")
    parser.add_argument("--limit", type=int, default=6, help="news items per stock")
    parser.add_argument("--symbol", action="append", default=None,
                        help="only update this symbol code, repeatable")
    args = parser.parse_args()

    paths = sorted(PROFILE_DIR.glob("*_extended.json"))
    if args.symbol:
        allowed = {code.lower() for code in args.symbol}
        paths = [p for p in paths if p.stem.removesuffix("_extended").lower() in allowed]
    if not paths:
        print("没有找到 *_extended.json 快照", file=sys.stderr)
        return 2

    total = 0
    updated = 0
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path.name}", flush=True)
        for attempt in range(2):
            try:
                _symbol, _name, count = update_profile(path, limit=args.limit)
                total += count
                if count:
                    updated += 1
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 0:
                    time.sleep(1.0)
                    continue
                print(f"  拉取失败: {exc}", file=sys.stderr, flush=True)
        if index < len(paths):
            time.sleep(0.3)

    print(f"完成: {updated}/{len(paths)} 只更新，共 {total} 条新闻", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
