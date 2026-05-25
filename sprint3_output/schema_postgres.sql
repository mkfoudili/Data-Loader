-- schema_postgres.sql — Sprint 3 Team 3 | Topic M6 | Team SG03
-- Run: psql -U postgres -d m6_thermal -f sprint3_output/team3/schema_postgres.sql

CREATE TABLE IF NOT EXISTS attention_maps (
    id          BIGSERIAL    PRIMARY KEY,
    patient_id  SMALLINT     NOT NULL REFERENCES subjects(patient_id),
    window_id   INTEGER      NOT NULL,
    timestep    SMALLINT     NOT NULL,
    attn_weight FLOAT        NOT NULL,
    recon_error FLOAT,
    created_at  TIMESTAMPTZ  DEFAULT NOW(),
    FOREIGN KEY (patient_id, window_id) REFERENCES windows(patient_id, window_id)
);

CREATE INDEX IF NOT EXISTS idx_attn_patient_window ON attention_maps (patient_id, window_id);
CREATE INDEX IF NOT EXISTS idx_attn_timestep       ON attention_maps (patient_id, timestep);

-- Sample dump (50 windows, 2 patients)
INSERT INTO windows (window_id, patient_id, segment_id, window_index, window_start, window_end, label, anomaly_ratio, is_interpolated)
SELECT i, 0, 0, i,
    '2024-01-01'::timestamptz + (i * INTERVAL '30 seconds'),
    '2024-01-01'::timestamptz + (i * INTERVAL '30 seconds') + INTERVAL '60 seconds',
    0, 0.0, FALSE
FROM generate_series(0, 24) i
ON CONFLICT DO NOTHING;

INSERT INTO windows (window_id, patient_id, segment_id, window_index, window_start, window_end, label, anomaly_ratio, is_interpolated)
SELECT i, 1, 0, i,
    '2024-01-01'::timestamptz + (i * INTERVAL '30 seconds'),
    '2024-01-01'::timestamptz + (i * INTERVAL '30 seconds') + INTERVAL '60 seconds',
    CASE WHEN i IN (3,8,14,19,23) THEN 1 ELSE 0 END,
    CASE WHEN i IN (3,8,14,19,23) THEN 0.85 ELSE 0.0 END,
    FALSE
FROM generate_series(0, 24) i
ON CONFLICT DO NOTHING;
