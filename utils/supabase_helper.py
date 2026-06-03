"""
supabase_helper.py
Handles all Supabase interactions:
  - Create tables (if not exist) via REST API
  - Insert prediction records
  - Fetch history
"""
import json
import urllib.request
import urllib.error
from datetime import datetime

SUPABASE_URL = "https://ysousrfyfkpldtfvtihx.supabase.co"
SUPABASE_API_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlzb3VzcmZ5ZmtwbGR0ZnZ0aWh4Iiwicm9sZSI6ImFub24i"
    "LCJpYXQiOjE3ODA0NjU5OTYsImV4cCI6MjA5NjA0MTk5Nn0."
    "8fJ1IE107yUgXIWS3OxAqMCtCz5ugO6KunLsoNUuEqM"
)

HEADERS = {
    "apikey": SUPABASE_API_KEY,
    "Authorization": f"Bearer {SUPABASE_API_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation"
}


def _request(method: str, endpoint: str, data: dict = None) -> dict:
    """Low-level HTTP helper using stdlib (no httpx/requests needed)."""
    url = f"{SUPABASE_URL}/rest/v1/{endpoint}"
    body = json.dumps(data).encode("utf-8") if data else None
    req = urllib.request.Request(url, data=body, headers=HEADERS, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8")
        return {"error": err, "status": e.code}
    except Exception as ex:
        return {"error": str(ex)}


# ──────────────────────────────────────────────
#  TABLE SCHEMAS (created via Supabase SQL editor
#  or auto-upserted here via REST if they exist)
# ──────────────────────────────────────────────
# predictions table columns:
#   id (auto), company_name, founded_year, category, country,
#   funding_round, funding_usd_m, num_funding_rounds, num_employees,
#   num_investors, num_milestones, num_relationships,
#   top_tier_investor, revenue_proxy, market_size, has_patent,
#   founder_experience, predicted_status, confidence,
#   key_insights, recommendations, created_at


def insert_prediction(record: dict) -> dict:
    """Insert a prediction record into Supabase."""
    record["created_at"] = datetime.utcnow().isoformat()
    result = _request("POST", "predictions", record)
    return result


def fetch_predictions(limit: int = 50) -> list:
    """Fetch recent predictions from Supabase."""
    result = _request("GET", f"predictions?order=created_at.desc&limit={limit}")
    if isinstance(result, list):
        return result
    return []


def test_connection() -> bool:
    """Test if Supabase connection is alive."""
    result = _request("GET", "predictions?limit=1")
    # If error key not in result OR result is a list → connected
    if isinstance(result, list):
        return True
    if isinstance(result, dict) and "error" not in result:
        return True
    # Table might not exist yet — that's still "connected"
    if isinstance(result, dict) and result.get("status") in (404, 406):
        return True
    return False


# ── Schema SQL (run once in Supabase SQL editor) ──────────────────
SCHEMA_SQL = """
-- Run this once in the Supabase SQL Editor

CREATE TABLE IF NOT EXISTS predictions (
  id                  BIGSERIAL PRIMARY KEY,
  company_name        TEXT,
  founded_year        INT,
  category            TEXT,
  country             TEXT,
  funding_round       TEXT,
  funding_usd_m       FLOAT,
  num_funding_rounds  INT,
  num_employees       INT,
  num_investors       INT,
  num_milestones      INT,
  num_relationships   INT,
  top_tier_investor   BOOLEAN,
  revenue_proxy       FLOAT,
  market_size         INT,
  has_patent          BOOLEAN,
  founder_experience  INT,
  predicted_status    TEXT,
  confidence          FLOAT,
  key_insights        TEXT,
  recommendations     TEXT,
  created_at          TIMESTAMPTZ DEFAULT now()
);
"""

if __name__ == "__main__":
    print("Testing Supabase connection...")
    ok = test_connection()
    print("Connection:", "✓ OK" if ok else "✗ FAILED")
    print("\nSchema SQL to run in Supabase:")
    print(SCHEMA_SQL)
