"""Project-wide constants, paths, and secrets."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# .env currently stores the key as `riot_api`; RIOT_API_KEY accepted as a fallback
API_KEY = os.getenv("riot_api") or os.getenv("RIOT_API_KEY")
if not API_KEY:
    raise EnvironmentError("Riot API key missing: add `riot_api=RGAPI-...` to .env")

# Routing
PLATFORM = "na1"        # league-v4
REGION = "americas"     # match-v5, account-v1

# Ranked Solo/Duo
QUEUE = "RANKED_SOLO_5x5"
QUEUE_ID = 420

TOP_N = 100             # players
MATCHES_PER_PLAYER = 10  # max 100 per match-ids request; each match costs 1 extra request

# Rate limits for a development key: (max_calls, window_seconds)
RATE_LIMITS = ((20, 1), (100, 120))

# Paths
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
for _d in (RAW_DIR, PROCESSED_DIR):
    _d.mkdir(parents=True, exist_ok=True)
