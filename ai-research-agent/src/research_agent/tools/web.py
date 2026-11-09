"""Web tools: DuckDuckGo search (no API key needed) and a guarded page reader."""
from __future__ import annotations

import ipaddress
import re
import socket
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

USER_AGENT = "Mozilla/5.0 (compatible; AI-Research-Agent/1.0; +https://github.com/)"
MAX_BYTES = 1_500_000


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Return [{title, url, snippet}] from DuckDuckGo."""
    if not query or not query.strip():
        raise ValueError("Query must not be empty")
    try:
        from ddgs import DDGS
    except ImportError:  # legacy package name
        from duckduckgo_search import DDGS  # type: ignore
    try:
        raw = DDGS().text(query, max_results=max_results)
    except Exception as exc:
        raise RuntimeError(f"Web search failed: {exc}") from exc
    results = []
    for item in raw or []:
        url = item.get("href") or item.get("url")
        if not url:
            continue
        results.append({"title": (item.get("title") or url).strip(), "url": url,
                        "snippet": (item.get("body") or "").strip()})
    return results


def assert_public_url(url: str) -> None:
    """Reject non-http(s) URLs and hosts that resolve to private/loopback/link-local addresses (SSRF guard)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Only http(s) URLs are allowed")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve host: {parsed.hostname}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            raise ValueError("Refusing to fetch a private or local address")


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "head", "template", "iframe"}
    _BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article", "ul", "ol"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip_depth:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str]:
    parser = _TextExtractor()
    parser.feed(html)
    text = re.sub(r"[ \t\f\v]+", " ", "".join(parser.parts))
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return parser.title.strip(), text


def read_webpage(url: str, max_chars: int = 6000, timeout: float = 15.0) -> dict:
    """Fetch a public web page and return its readable text."""
    import httpx

    current = url
    with httpx.Client(follow_redirects=False, timeout=timeout, headers={"User-Agent": USER_AGENT}) as client:
        for _ in range(4):
            assert_public_url(current)
            with client.stream("GET", current) as resp:
                if resp.is_redirect and resp.headers.get("location"):
                    current = urljoin(current, resp.headers["location"])
                    continue
                resp.raise_for_status()
                ctype = resp.headers.get("content-type", "").lower()
                if "pdf" in ctype:
                    raise ValueError("PDF URLs are not supported by read_webpage")
                if ctype and "html" not in ctype and "text" not in ctype:
                    raise ValueError(f"Unsupported content type: {ctype}")
                body = bytearray()
                for part in resp.iter_bytes():
                    body.extend(part)
                    if len(body) > MAX_BYTES:
                        break
                html = bytes(body).decode(resp.encoding or "utf-8", errors="replace")
                title, text = html_to_text(html) if "html" in ctype or "<html" in html[:2000].lower() else ("", html)
                return {"url": current, "title": title or current, "text": text[:max_chars],
                        "truncated": len(text) > max_chars}
    raise ValueError("Too many redirects")
