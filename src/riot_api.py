"""Rate-limited, cached client for the Riot Games API."""
import json
import logging
import time
from collections import deque

import requests

from src.config import (API_KEY, MATCHES_PER_PLAYER, PLATFORM, QUEUE, QUEUE_ID,
                        RATE_LIMITS, RAW_DIR, REGION)

log = logging.getLogger(__name__)


class RateLimiter:
    """Sliding-window limiter enforcing several (max_calls, seconds) limits at once."""

    def __init__(self, limits=RATE_LIMITS):
        self.limits = [(n, s, deque()) for n, s in limits]

    def wait(self) -> None:
        while True:
            now = time.monotonic()
            sleep_for = 0.0
            for n, s, calls in self.limits:
                while calls and now - calls[0] >= s:  # drop calls outside the window
                    calls.popleft()
                if len(calls) >= n:
                    sleep_for = max(sleep_for, s - (now - calls[0]) + 0.05)
            if sleep_for <= 0:
                for _, _, calls in self.limits:
                    calls.append(now)
                return
            time.sleep(sleep_for)


_session = requests.Session()
_session.headers.update({"X-Riot-Token": API_KEY})  # key in header, not URL
_limiter = RateLimiter()


def riot_get(url: str, params: dict | None = None, max_retries: int = 5):
    """GET a Riot endpoint with rate limiting and retries. Returns None on 404."""
    for attempt in range(max_retries):
        _limiter.wait()
        try:
            r = _session.get(url, params=params, timeout=10)
        except requests.RequestException as e:
            log.warning("Network error (%s), attempt %d", e, attempt + 1)
            time.sleep(2 ** attempt)
            continue

        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 10))
            log.warning("429 rate limited; sleeping %ss", wait)
            time.sleep(wait)
        elif r.status_code == 404:
            return None
        elif r.status_code in (401, 403):
            raise PermissionError(f"{r.status_code}: API key invalid/expired - regenerate it.")
        elif 500 <= r.status_code < 600:
            time.sleep(2 ** attempt)
        else:
            r.raise_for_status()
    raise RuntimeError(f"Failed after {max_retries} retries: {url}")


def cached_get(subdir: str, name: str, url: str, params: dict | None = None, refresh: bool = False):
    """Return cached JSON from data/raw/<subdir>/<name>.json, else fetch and cache it."""
    path = RAW_DIR / subdir / f"{name}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    data = riot_get(url, params)
    if data is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    return data


# ---------- Endpoints ----------

def get_challenger_league(refresh: bool = False) -> dict:
    """league-v4: full NA Challenger ladder for ranked solo/duo."""
    url = f"https://{PLATFORM}.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/{QUEUE}"
    return cached_get("league", "challenger", url, refresh=refresh)


def get_match_ids(puuid: str, count: int = MATCHES_PER_PLAYER, refresh: bool = False) -> list[str]:
    """match-v5: most recent ranked solo match IDs for a player (max 100)."""
    url = f"https://{REGION}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
    params = {"queue": QUEUE_ID, "start": 0, "count": count}
    return cached_get("match_ids", f"{puuid}_{count}", url, params, refresh=refresh) or []


def get_match(match_id: str) -> dict | None:
    """match-v5: full match details (matches never change, so always cached)."""
    url = f"https://{REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    return cached_get("matches", match_id, url)
