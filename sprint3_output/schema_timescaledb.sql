-- schema_timescaledb.sql — Sprint 3 Team 3 | Topic M6 | Team SG03
-- Run: psql -U postgres -d m6_thermal_tsdb -f sprint3_output/team3/schema_timescaledb.sql

CREATE TABLE IF NOT EXISTS attention_maps (
    ts          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    patient_id  SMALLINT     NOT NULL,
    window_id   INTEGER      NOT NULL,
    timestep    SMALLINT     NOT NULL,
    attn_weight FLOAT        NOT NULL,
    recon_error FLOAT,
    FOREIGN KEY (patient_id, window_id) REFERENCES windows_tsdb(patient_id, window_id)
);

SELECT create_hypertable('attention_maps', 'ts',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_attn_tsdb_patient_window ON attention_maps (patient_id, window_id);
