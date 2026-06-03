-- ─────────────────────────────────────────────────────────────
--  STARTUP SURVIVAL PREDICTOR — Supabase Schema
--  Run this ONCE in the Supabase SQL Editor
--  Dashboard → SQL Editor → New query → paste → Run
-- ─────────────────────────────────────────────────────────────

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
  top_tier_investor   BOOLEAN DEFAULT FALSE,
  revenue_proxy       FLOAT,
  market_size         INT,
  has_patent          BOOLEAN DEFAULT FALSE,
  founder_experience  INT,
  predicted_status    TEXT,
  confidence          FLOAT,
  key_insights        TEXT,
  recommendations     TEXT,
  created_at          TIMESTAMPTZ DEFAULT now()
);

-- Enable Row Level Security (optional but recommended)
-- ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

-- Allow anon reads and inserts (for the app)
-- CREATE POLICY "Allow all" ON predictions FOR ALL USING (true) WITH CHECK (true);

-- Index for faster queries
CREATE INDEX IF NOT EXISTS idx_predictions_status
  ON predictions(predicted_status);

CREATE INDEX IF NOT EXISTS idx_predictions_created
  ON predictions(created_at DESC);

-- Verify
SELECT 'Table predictions created successfully!' AS message;
