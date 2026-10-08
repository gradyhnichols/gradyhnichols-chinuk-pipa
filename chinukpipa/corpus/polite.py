"""A deliberately slow, well-behaved HTTP client shared by the corpus modules.

Digitised-heritage sites are run by small teams.  The corpus download modules
therefore go through :class:`PoliteSession`, which

* identifies itself honestly (``User-Agent`` names the project and, if you
  supply one, a contact address -- set ``CHINUKPIPA_CONTACT``);
* fetches and obeys ``robots.txt`` for each host, including ``Crawl-delay``;
* never sends requests to one host closer together than ``min_interval``
  seconds (default 1.0; a larger robots ``Crawl-delay`` wins);
* retries transient failures (connection errors, HTTP 429/5xx) with
  exponential backoff plus jitter, honouring ``Retry-After``.

``robots.txt`` is honoured by default (``honor_robots=True``); turning it off
is only for hosts whose robots file is unreachable and whose terms you have
checked.  Whether a given *use* is permitted by a site's Terms of Service is a
separate question that this module cannot answer: read the terms first.
"""

from __future__ import annotations

import logging
import os
import random
import time
import urllib.parse
import urllib.robotparser
from typing import Iterable

import requests

log = logging.getLogger(__name__)

PRODUCT_TOKEN = "chinuk-pipa"
"""``User-Agent`` product token; this is what robots.txt rules are matched on."""

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class RobotsDisallowed(RuntimeError):
    """Raised when robots.txt forbids fetching a URL for our user agent."""


class HttpError(RuntimeError):
    """Raised for a non-success HTTP status once retries are exhausted."""

    def __init__(self, url: str, status: int | None, message: str = "") -> None:
        super().__init__(f"{status or 'network error'} for {url} {message}".strip())
        self.url = url
        self.status = status


def default_user_agent(contact: str | None = None) -> str:
    """Build an honest ``User-Agent`` string.

    >>> default_user_agent("me@example.org").startswith("chinuk-pipa/")
    True
    """
    from chinukpipa import __version__

    contact = contact or os.environ.get("CHINUKPIPA_CONTACT") or "contact-not-set"
    return f"{PRODUCT_TOKEN}/{__version__} (research toolkit; Duployan shorthand corpus; {contact})"


class PoliteSession:
    """``requests`` wrapper that rate-limits, obeys robots.txt and retries.

    Parameters
    ----------
    min_interval:
        Minimum seconds between two requests to the same host.  Values below
        1.0 are raised to 1.0.
    max_retries:
        Retries after the first attempt for transient errors.
    backoff:
        Base of the exponential backoff, in seconds (``backoff * 2**attempt``).
    timeout:
        ``(connect, read)`` timeouts in seconds.
    honor_robots:
        Keep ``True`` unless you are fetching from a host you operate.
    """

    def __init__(
        self,
        *,
        min_interval: float = 1.0,
        max_retries: int = 5,
        backoff: float = 2.0,
        max_backoff: float = 300.0,
        timeout: tuple[float, float] = (15.0, 180.0),
        user_agent: str | None = None,
        contact: str | None = None,
        honor_robots: bool = True,
    ) -> None:
        self.min_interval = max(1.0, float(min_interval))
        self.max_retries = max_retries
        self.backoff = backoff
        self.max_backoff = max_backoff
        self.timeout = timeout
        self.honor_robots = honor_robots
        self.user_agent = user_agent or default_user_agent(contact)
        self._http = requests.Session()
        self._http.headers["User-Agent"] = self.user_agent
        self._last: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}

    # ------------------------------------------------------------------ robots
    def _robots_for(self, url: str) -> urllib.robotparser.RobotFileParser:
        parts = urllib.parse.urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin in self._robots:
            return self._robots[origin]
        rp = urllib.robotparser.RobotFileParser()
        robots_url = origin + "/robots.txt"
        try:
            self._sleep_for(parts.netloc)
            resp = self._http.get(robots_url, timeout=self.timeout)
            self._last[parts.netloc] = time.monotonic()
            if resp.status_code >= 500:
                log.warning("robots.txt on %s returned %s; assuming disallow-all", origin, resp.status_code)
                rp.parse(["User-agent: *", "Disallow: /"])
            elif resp.status_code >= 400:
                # RFC 9309 section 2.3.1.3: "unavailable" robots.txt means no restrictions.
                rp.parse([])
            else:
                rp.parse(resp.text.splitlines())
        except requests.RequestException as exc:
            log.warning("could not fetch %s (%s); assuming disallow-all", robots_url, exc)
            rp.parse(["User-agent: *", "Disallow: /"])
        self._robots[origin] = rp
        return rp

    def crawl_delay(self, url: str) -> float:
        """Robots ``Crawl-delay`` for ``url``'s host (0 if none)."""
        if not self.honor_robots:
            return 0.0
        delay = self._robots_for(url).crawl_delay(PRODUCT_TOKEN)
        return float(delay) if delay else 0.0

    def check_allowed(self, url: str) -> None:
        """Raise :class:`RobotsDisallowed` if robots.txt forbids ``url``."""
        if self.honor_robots and not self._robots_for(url).can_fetch(PRODUCT_TOKEN, url):
            raise RobotsDisallowed(f"robots.txt disallows {url} for {PRODUCT_TOKEN}")

    # ---------------------------------------------------------------- pacing
    def effective_interval(self, url: str) -> float:
        """Seconds enforced between requests to ``url``'s host."""
        return max(self.min_interval, self.crawl_delay(url))

    def _sleep_for(self, host: str, interval: float | None = None) -> None:
        interval = self.min_interval if interval is None else interval
        wait = self._last.get(host, 0.0) + interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)

    # --------------------------------------------------------------- requests
    def get(
        self,
        url: str,
        *,
        stream: bool = False,
        accept_statuses: Iterable[int] = (200,),
        headers: dict[str, str] | None = None,
    ) -> requests.Response:
        """GET ``url`` politely.  Returns the response (caller closes it if ``stream``).

        Raises :class:`RobotsDisallowed` or :class:`HttpError`.
        """
        self.check_allowed(url)
        host = urllib.parse.urlsplit(url).netloc
        interval = self.effective_interval(url)
        accept = frozenset(accept_statuses)
        last_error: Exception | None = None
        status: int | None = None
        for attempt in range(self.max_retries + 1):
            self._sleep_for(host, interval)
            try:
                resp = self._http.get(url, stream=stream, timeout=self.timeout, headers=headers)
                self._last[host] = time.monotonic()
            except requests.RequestException as exc:
                self._last[host] = time.monotonic()
                last_error, status = exc, None
                delay = self._delay(attempt, None)
                log.warning("GET %s failed (%s); retry %d/%d in %.0fs", url, exc, attempt + 1, self.max_retries, delay)
                time.sleep(delay)
                continue
            if resp.status_code in accept:
                return resp
            status = resp.status_code
            retry_after = resp.headers.get("Retry-After")
            resp.close()
            if status in RETRY_STATUSES and attempt < self.max_retries:
                delay = self._delay(attempt, retry_after)
                log.warning("GET %s -> %s; retry %d/%d in %.0fs", url, status, attempt + 1, self.max_retries, delay)
                time.sleep(delay)
                continue
            raise HttpError(url, status)
        raise HttpError(url, status, f"after {self.max_retries} retries ({last_error})")

    def _delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(self.max_backoff, max(0.0, float(retry_after)))
            except ValueError:
                pass
        base = min(self.max_backoff, self.backoff * (2**attempt))
        return base * (0.75 + 0.5 * random.random())

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "PoliteSession":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
