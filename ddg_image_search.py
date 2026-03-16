import httpx
import re


def _get_vqd(query: str) -> str | None:
    """Fetch the vqd token needed for DuckDuckGo image API."""
    url = "https://duckduckgo.com/"
    params = {"q": query}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    try:
        resp = httpx.get(url, params=params, headers=headers, timeout=10.0)
        if resp.status_code != 200:
            return None
        # vqd appears in JS as: vqd='3-12345678901234567890123456789012'; or vqd="..."
        match = re.search(r"vqd='([0-9-]+)'", resp.text)
        if not match:
            match = re.search(r"vqd=\"([0-9-]+)\"", resp.text)
        if match:
            return match.group(1)
    except Exception:
        pass
    return None


def search_images(query: str, max_results: int = 6) -> list[dict]:
    """Return a list of image results from DuckDuckGo.

    Each item is a dict containing:
      - image_url: URL of the image
      - source_url: the page containing the image
      - title: title or snippet

    Note: This is a best-effort scraper and may break if DuckDuckGo changes.
    """
    vqd = _get_vqd(query)
    if not vqd:
        return []

    params = {
        "l": "us-en",
        "o": "json",
        "q": query,
        "vqd": vqd,
        "f": ",,,",
        "p": "1",
        "v7exp": "a",
    }
    headers = {
        # A minimal, generic User-Agent seems to avoid DuckDuckGo blocking.
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Referer": "https://duckduckgo.com/",
    }

    try:
        # DuckDuckGo's image endpoint expects cookies from a prior visit to the main page.
        # Using a session (Client) ensures we get those cookies first.
        with httpx.Client(headers={"User-Agent": headers["User-Agent"], "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}, timeout=10.0) as client:
            client.get("https://duckduckgo.com/", params={"q": query})
            resp = client.get("https://duckduckgo.com/i.js", params=params, headers=headers)

        if resp.status_code != 200:
            return []
        data = resp.json()
        results = []
        for item in data.get("results", [])[:max_results]:
            image_url = item.get("image") or item.get("thumbnail")
            source_url = item.get("url")
            title = item.get("title") or item.get("subtitle") or ""
            if image_url:
                results.append({
                    "image_url": image_url,
                    "source_url": source_url,
                    "title": title,
                })
        return results
    except Exception:
        return []
