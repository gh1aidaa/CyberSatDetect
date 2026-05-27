-- =========================
-- TELEMETRY DATABASE
-- =========================
\c telemetry_db;

CREATE TABLE runs (
  id BIGSERIAL PRIMARY KEY,
  run_id TEXT UNIQUE NOT NULL,
  user_id TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  status TEXT DEFAULT 'PENDING',
  anom_rate DOUBLE PRECISION,
  results_csv_path TEXT
);

CREATE TABLE telemetry_files (
  id BIGSERIAL PRIMARY KEY,
  run_id TEXT REFERENCES runs(run_id) ON DELETE CASCADE,
  user_file_id TEXT NOT NULL,
  original_filename TEXT,
  raw_path TEXT NOT NULL,
  clean_path TEXT NOT NULL,
  shape_raw JSONB,
  shape_clean JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE telemetry_segments (
  id BIGSERIAL PRIMARY KEY,
  telemetry_file_id BIGINT REFERENCES telemetry_files(id) ON DELETE CASCADE,
  segment_index INT NOT NULL,
  segment_path TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (telemetry_file_id, segment_index)
);

-- =========================
-- NORMAL CASES STORE
-- =========================
\c normal_store_db;

CREATE TABLE normal_segments (
  id BIGSERIAL PRIMARY KEY,
  source_run_id TEXT,
  user_file_id TEXT,
  segment_path TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =========================
-- ANOMALY CASES STORE
-- =========================
\c anomaly_store_db;

CREATE TABLE anomaly_results (
  id BIGSERIAL PRIMARY KEY,
  run_id TEXT,
  user_file_id TEXT,
  num_test_windows INT,
  n_predicted_anoms INT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE anomaly_sequences (
  id BIGSERIAL PRIMARY KEY,
  anomaly_result_id BIGINT REFERENCES anomaly_results(id) ON DELETE CASCADE,
  start_idx INT,
  end_idx INT
);

CREATE TABLE anomaly_scores (
  id BIGSERIAL PRIMARY KEY,
  anomaly_result_id BIGINT REFERENCES anomaly_results(id) ON DELETE CASCADE,
  start_idx INT,
  end_idx INT,
  score DOUBLE PRECISION
);

CREATE TABLE anomaly_segments (
  id BIGSERIAL PRIMARY KEY,
  run_id TEXT,
  user_file_id TEXT,
  segment_path TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =========================
-- MODEL REGISTRY
-- =========================
\c model_registry_db;

CREATE TABLE model_versions (
  id BIGSERIAL PRIMARY KEY,
  use_id TEXT UNIQUE NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  created_by TEXT,
  approved BOOLEAN DEFAULT FALSE,
  approved_by TEXT,
  approved_at TIMESTAMPTZ,
  notes TEXT
);

CREATE TABLE model_artifacts (
  id BIGSERIAL PRIMARY KEY,
  use_id TEXT REFERENCES model_versions(use_id) ON DELETE CASCADE,
  model_type TEXT,
  path TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (use_id, model_type)
);

-- =========================
-- SYSTEM LOGS
-- =========================
\c system_logs_db;

CREATE TABLE system_error_logs (
  id BIGSERIAL PRIMARY KEY,
  level TEXT,
  source TEXT,
  message TEXT,
  meta JSONB,
  run_id TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =========================
-- UI LOGS
-- =========================
\c ui_logs_db;

CREATE TABLE ui_error_logs (
  id BIGSERIAL PRIMARY KEY,
  level TEXT,
  page TEXT,
  message TEXT,
  meta JSONB,
  session_id TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =========================
-- REPORT ARCHIVE
-- =========================
\c report_archive_db;

CREATE TABLE reports (
  id BIGSERIAL PRIMARY KEY,
  run_id TEXT,
  report_type TEXT,
  path TEXT,
  meta JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =========================
-- USER ACCOUNTS
-- =========================
\c user_accounts_db;

CREATE TABLE users (
  id BIGSERIAL PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  role TEXT DEFAULT 'USER',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  last_login TIMESTAMPTZ
);

-- =========================
-- SESSION LOGS
-- =========================
\c session_logs_db;

CREATE TABLE sessions (
  id BIGSERIAL PRIMARY KEY,
  user_id BIGINT,
  token_hash TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  expires_at TIMESTAMPTZ,
  ip TEXT,
  user_agent TEXT
);

CREATE TABLE session_events (
  id BIGSERIAL PRIMARY KEY,
  session_id BIGINT REFERENCES sessions(id) ON DELETE CASCADE,
  event_type TEXT,
  meta JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
