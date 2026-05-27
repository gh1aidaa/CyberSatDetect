from sqlalchemy import create_engine

# =====================================================
# 1️⃣ Telemetry Database
# =====================================================
TELEMETRY_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/telemetry_db"
telemetry_engine = create_engine(TELEMETRY_DB_URL, echo=False, future=True)

# =====================================================
# 2️⃣ Normal Cases Store
# =====================================================
NORMAL_STORE_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/normal_store_db"
normal_store_engine = create_engine(NORMAL_STORE_DB_URL, echo=False, future=True)

# =====================================================
# 3️⃣ Anomaly Store Database
# =====================================================
ANOMALY_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/anomaly_store_db"
anomaly_engine = create_engine(ANOMALY_DB_URL, echo=False, future=True)

# =====================================================
# 4️⃣ Model Registry
# =====================================================
MODEL_REGISTRY_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/model_registry_db"
model_registry_engine = create_engine(MODEL_REGISTRY_DB_URL, echo=False, future=True)

# =====================================================
# 5️⃣ System Error Logs
# =====================================================
SYSTEM_LOGS_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/system_logs_db"
system_logs_engine = create_engine(SYSTEM_LOGS_DB_URL, echo=False, future=True)

# =====================================================
# 6️⃣ UI Error Logs
# =====================================================
UI_LOGS_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/ui_logs_db"
ui_logs_engine = create_engine(UI_LOGS_DB_URL, echo=False, future=True)

# =====================================================
# 7️⃣ Report Archive
# =====================================================
REPORT_ARCHIVE_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/report_archive_db"
report_archive_engine = create_engine(REPORT_ARCHIVE_DB_URL, echo=False, future=True)

# =====================================================
# 8️⃣ User Accounts
# =====================================================
USER_ACCOUNTS_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/user_accounts_db"
user_accounts_engine = create_engine(USER_ACCOUNTS_DB_URL, echo=False, future=True)

# =====================================================
# 9️⃣ Session Logs
# =====================================================
SESSION_LOGS_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/session_logs_db"
session_logs_engine = create_engine(SESSION_LOGS_DB_URL, echo=False, future=True)