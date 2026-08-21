from src.database.connection import get_connection

c = get_connection()

c.execute("""
CREATE TABLE IF NOT EXISTS model_monitoring_history (
    id BIGINT,
    run_date DATE,

    evaluated_forecasts INTEGER,

    overall_mae DOUBLE,
    overall_rmse DOUBLE,
    overall_bias DOUBLE,
    overall_within10 DOUBLE,
    overall_within20 DOUBLE,

    mae_24h DOUBLE,
    mae_48h DOUBLE,
    mae_72h DOUBLE,

    baseline_mae DOUBLE,
    baseline_rmse DOUBLE,
    baseline_improvement_pct DOUBLE,

    recent_mean_aqi DOUBLE,
    historical_mean_aqi DOUBLE,
    mean_aqi_change_pct DOUBLE,

    recent_std_aqi DOUBLE,
    historical_std_aqi DOUBLE,
    std_aqi_change_pct DOUBLE,

    drift_status VARCHAR,
    model_status VARCHAR,
    recommendation VARCHAR,

    created_at TIMESTAMP
)
""")

c.close()

print("model_monitoring_history table created successfully.")
