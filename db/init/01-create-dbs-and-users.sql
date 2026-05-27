-- ===============================
-- Create Databases
-- ===============================
CREATE DATABASE telemetry_db;
CREATE DATABASE normal_store_db;
CREATE DATABASE anomaly_store_db;
CREATE DATABASE model_registry_db;
CREATE DATABASE system_logs_db;
CREATE DATABASE ui_logs_db;
CREATE DATABASE report_archive_db;
CREATE DATABASE user_accounts_db;
CREATE DATABASE session_logs_db;

-- ===============================
-- Create Users (Roles)
-- ===============================
CREATE USER telemetry_rw WITH PASSWORD 'telemetry_rw_pw';
CREATE USER anomaly_rw   WITH PASSWORD 'anomaly_rw_pw';
CREATE USER reports_rw   WITH PASSWORD 'reports_rw_pw';
CREATE USER syslogs_w    WITH PASSWORD 'syslogs_w_pw';
CREATE USER uilogs_w     WITH PASSWORD 'uilogs_w_pw';

CREATE USER accounts_rw  WITH PASSWORD 'accounts_rw_pw';
CREATE USER sessions_rw  WITH PASSWORD 'sessions_rw_pw';

CREATE USER normal_ro    WITH PASSWORD 'normal_ro_pw';
CREATE USER model_admin  WITH PASSWORD 'model_admin_pw';
