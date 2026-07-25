"""Web scraper for collecting training data from URLs.

Extracts clean text from web pages, respecting robots.txt and rate limits.
Returns HuggingFace-compatible Dataset objects.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.robotparser
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Tag
from datasets import Dataset

logger = logging.getLogger(__name__)

# Elements whose content is typically non-article text
_STRIP_TAGS = {
    "script",
    "style",
    "nav",
    "footer",
    "header",
    "aside",
    "form",
    "button",
    "iframe",
    "noscript",
    "svg",
    "figure",
    "figcaption",
    "menu",
    "menuitem",
}

# Minimum text length for a scraped page to be considered valid
_MIN_TEXT_LENGTH = 100

# Default user-agent
_USER_AGENT = "NightmareNet-Scraper/0.2.0 (+https://github.com/Adit-Jain-srm/NightmareNet)"


class WebScraper:
    """Scrapes clean text from a list of URLs.

    Respects robots.txt, applies rate limiting, and produces a HuggingFace
    Dataset with a single ``text`` column.

    Args:
        delay: Seconds to wait between requests (default 1.0).
        timeout: HTTP request timeout in seconds (default 15).
        max_retries: Number of retries on transient failures (default 2).
        user_agent: User-Agent string sent with each request.
        respect_robots: Whether to check robots.txt before scraping.
        min_text_length: Minimum character count for a page to be included.
    """

    def __init__(
        self,
        delay: float = 1.0,
        timeout: int = 15,
        max_retries: int = 2,
        user_agent: str = _USER_AGENT,
        respect_robots: bool = True,
        min_text_length: int = _MIN_TEXT_LENGTH,
    ) -> None:
        if delay < 0:
            raise ValueError(f"delay must be >= 0, got {delay}")
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent
        self.respect_robots = respect_robots
        self.min_text_length = min_text_length
        self._robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}

    # ------------------------------------------------------------------
    # robots.txt
    # ------------------------------------------------------------------

    def _can_fetch(self, url: str) -> bool:
        """Check whether robots.txt permits scraping *url*."""
        if not self.respect_robots:
            return True
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._robots_cache:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f"{origin}/robots.txt")
            try:
                rp.read()
            except Exception:
                logger.debug("Could not read robots.txt for %s; allowing.", origin)
                return True
            self._robots_cache[origin] = rp
        return self._robots_cache[origin].can_fetch(self.user_agent, url)

    # ------------------------------------------------------------------
    # HTML → text
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(html: str) -> str:
        """Strip HTML to clean, paragraph-separated plain text."""
        soup = BeautifulSoup(html, "html.parser")

        # Remove non-content elements
        for tag in soup.find_all(_STRIP_TAGS):
            tag.decompose()

        # Prefer <article> or <main> if present
        main = soup.find("article") or soup.find("main") or soup.find("body") or soup
        tags = ["p", "li", "h1", "h2", "h3", "h4", "blockquote"]
        paragraphs = main.find_all(tags) if isinstance(main, Tag) else []

        texts: list[str] = []
        for p in paragraphs:
            t = p.get_text(separator=" ", strip=True)
            # Collapse whitespace
            t = re.sub(r"\s+", " ", t).strip()
            if len(t) > 20:  # skip tiny fragments
                texts.append(t)

        return "\n\n".join(texts)

    # ------------------------------------------------------------------
    # Fetch single URL
    # ------------------------------------------------------------------

    def _fetch_url(self, url: str) -> Optional[str]:
        """Fetch a single URL, returning clean text or None on failure."""
        if not self._can_fetch(url):
            logger.warning("robots.txt disallows scraping %s; skipping.", url)
            return None

        headers = {"User-Agent": self.user_agent}
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(url, headers=headers, timeout=self.timeout)
                resp.raise_for_status()
                text = self._extract_text(resp.text)
                if len(text) < self.min_text_length:
                    logger.warning(
                        "Page %s yielded only %d chars (min=%d); skipping.",
                        url,
                        len(text),
                        self.min_text_length,
                    )
                    return None
                return text
            except requests.RequestException as exc:
                last_exc = exc
                wait = 2**attempt
                logger.debug(
                    "Attempt %d/%d failed for %s: %s — retrying in %ds",
                    attempt,
                    self.max_retries,
                    url,
                    exc,
                    wait,
                )
                time.sleep(wait)

        logger.error("Failed to fetch %s after %d retries: %s", url, self.max_retries, last_exc)
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape(self, urls: list[str], chunk_size: int = 512) -> Dataset:
        """Scrape URLs and return a HuggingFace Dataset.

        Each page is split into overlapping chunks of roughly *chunk_size*
        characters (split on paragraph boundaries) so the resulting dataset
        is suitable for language-model training.

        Args:
            urls: List of URLs to scrape.
            chunk_size: Target character count per sample.

        Returns:
            A ``datasets.Dataset`` with a ``text`` column.
        """
        if not urls:
            raise ValueError("urls list must not be empty")

        all_chunks: list[str] = []

        for i, url in enumerate(urls):
            logger.info("[%d/%d] Scraping %s", i + 1, len(urls), url)
            text = self._fetch_url(url)
            if text is None:
                continue

            # Split into paragraph-level chunks
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            current_chunk: list[str] = []
            current_len = 0

            for para in paragraphs:
                if current_len + len(para) > chunk_size and current_chunk:
                    all_chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                current_chunk.append(para)
                current_len += len(para)

            # Flush remaining
            if current_chunk:
                all_chunks.append("\n".join(current_chunk))

            # Rate limiting
            if i < len(urls) - 1:
                time.sleep(self.delay)

        if not all_chunks:
            raise RuntimeError(
                f"No text could be extracted from any of the {len(urls)} URLs. "
                "Check your URLs and network connection."
            )

        logger.info(
            "Scraping complete: %d URLs → %d text chunks.",
            len(urls),
            len(all_chunks),
        )
        return Dataset.from_dict({"text": all_chunks})
