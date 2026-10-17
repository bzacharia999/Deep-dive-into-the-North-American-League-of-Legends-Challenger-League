---
name: coding-best-practices
description: >-
  Use this skill when writing or refactoring any Python code in this project,
  especially code that calls the Riot Games API (league-v4, account-v1, match-v5).
  Covers project structure, reusable functions, error handling, logging, secrets,
  caching, and rate-limited API access.
---

# Coding Best Practices

## Core Principles
1. **DRY / reusable functions** – if code is written twice, make it a function in `src/`. Notebooks import from `src/`, they don't define core logic.
2. **Small, single-purpose functions** with type hints and a one-line docstring.
3. **Fail loudly, recover deliberately** – catch *specific* exceptions, log them, never bare `except:` or silent `pass`.
4. **No secrets in code** – load the API key from `.env`; `.env` is git-ignored.
5. **Constants at the top** (UPPER_CASE), no magic numbers/strings scattered around.
6. **Use `logging`, not `print`**, in `src/` modules.
7. **Cache raw API responses** to disk so reruns don't re-spend rate limit.
8. **PEP 8** naming (`snake_case` functions, `PascalCase` classes); format with `black`/`ruff`.
9. **Inspect environment and console first** – look at the current environment and prints to the console before suggesting any console log checks.

## Suggested Layout
```text
src/
  config.py      # constants, paths, env loading
  riot_api.py    # rate-limited client (below)
  collect.py     # pull top-100 players + matches
  features.py    # cleaning / feature engineering
  viz.py         # see data-visualization skill
  model.py       # see ml-modeling skill
data/raw/  data/processed/  figures/  notebooks/
```

## Config & Secrets
```python
# src/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("RIOT_API_KEY")
if not API_KEY:
    raise EnvironmentError("RIOT_API_KEY missing from .env")

PLATFORM = "na1"          # league-v4, summoner-v4
REGION = "americas"       # account-v1, match-v5
QUEUE = "RANKED_SOLO_5x5"
QUEUE_ID = 420            # ranked solo in match-v5
TOP_N = 100
RAW_DIR = Path("data/raw"); RAW_DIR.mkdir(parents=True, exist_ok=True)
```

## Riot API: Rate Limits to Respect
- **Dev key**: 20 req / 1 s **and** 100 req / 2 min (both apply). Dev keys expire every 24 h.
- Each endpoint also has its own **method limit**; read headers `X-App-Rate-Limit`, `X-Method-Rate-Limit`.
- On **429**, sleep for the `Retry-After` header seconds, then retry. Never hammer.
- Retry **5xx** with exponential backoff; **don't retry** 400/401/403/404 (fix the request / renew key).
- Send key via `X-Riot-Token` header, not the URL query string (keeps it out of logs).
- Routing: `na1` for league-v4; `americas` for account-v1 and match-v5.

## Rate-Limited Client Template
```python
# src/riot_api.py
import json, logging, time
from collections import deque
import requests
from src.config import API_KEY, RAW_DIR

log = logging.getLogger(__name__)

class RateLimiter:
    """Sliding-window limiter enforcing multiple (max_calls, seconds) limits."""
    def __init__(self, limits=((20, 1), (100, 120))):
        self.limits = [(n, s, deque()) for n, s in limits]

    def wait(self):
        while True:
            now = time.monotonic()
            sleep_for = 0.0
            for n, s, calls in self.limits:
                while calls and now - calls[0] >= s:
                    calls.popleft()
                if len(calls) >= n:
                    sleep_for = max(sleep_for, s - (now - calls[0]) + 0.05)
            if sleep_for <= 0:
                for _, _, calls in self.limits:
                    calls.append(now)
                return
            time.sleep(sleep_for)

_session = requests.Session()
_session.headers.update({"X-Riot-Token": API_KEY})
_limiter = RateLimiter()

def riot_get(url: str, params: dict | None = None, max_retries: int = 5) -> dict | list | None:
    """GET a Riot endpoint with rate limiting, retries, and backoff. Returns None on 404."""
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
            raise PermissionError(f"{r.status_code}: API key invalid/expired – regenerate it.")
        elif 500 <= r.status_code < 600:
            time.sleep(2 ** attempt)
        else:
            r.raise_for_status()
    raise RuntimeError(f"Failed after {max_retries} retries: {url}")

def cached_get(cache_name: str, url: str, params: dict | None = None):
    """Return cached JSON if present, else fetch and cache it."""
    path = RAW_DIR / f"{cache_name}.json"
    if path.exists():
        return json.loads(path.read_text())
    data = riot_get(url, params)
    if data is not None:
        path.write_text(json.dumps(data))
    return data
```

## Usage: Top 100 NA Players
```python
from src.config import PLATFORM, REGION, QUEUE, QUEUE_ID, TOP_N
from src.riot_api import cached_get

def get_top_players(n: int = TOP_N) -> list[dict]:
    url = f"https://{PLATFORM}.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/{QUEUE}"
    league = cached_get("challenger_league", url)
    return sorted(league["entries"], key=lambda e: e["leaguePoints"], reverse=True)[:n]

def get_match_ids(puuid: str, count: int = 20) -> list[str]:
    url = f"https://{REGION}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
    return cached_get(f"ids_{puuid}", url, {"queue": QUEUE_ID, "count": count}) or []

def get_match(match_id: str) -> dict | None:
    url = f"https://{REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    return cached_get(f"match_{match_id}", url)
```

## Error-Handling Pattern for Batch Loops
```python
results, failed = [], []
for p in players:
    try:
        results.extend(get_match_ids(p["puuid"]))
    except (RuntimeError, KeyError) as e:
        log.error("Skipping %s: %s", p.get("puuid"), e)
        failed.append(p["puuid"])
log.info("Done: %d ok, %d failed", len(results), len(failed))
```
A single bad player must never kill a long collection run; record failures and retry them later.

## Verification
- `git status` never shows `.env`.
- Re-running collection uses cached files (near-instant, no new API calls).
- Logs show no repeated 429s.

## Environment & Console Inspection
- **Inspect current environment and console output first**: Look at the current environment (env variables, process state, configuration) and existing prints to the console before suggesting any console log checks or asking the user to inspect console logs.
- **Avoid speculative or redundant log checks**: Do not ask to check console logs or suggest adding debug print/log statements if the required information is already available in the environment or active console output.
