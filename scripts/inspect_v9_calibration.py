import json

file = "models/forecast/calibration_v9/calibration_parameters.json"

with open(file, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=== V9 CALIBRATION ===")

print("\nParameters:")
for k, v in data["selected_parameters"].items():
    print(f"{k}: {v}")

print("\nTable sizes:")
for name in [
    "city_bias",
    "horizon_bias",
    "regime_bias",
    "city_horizon_bias",
    "city_regime_bias",
]:
    table = data.get(name, {})
    print(f"{name}: {len(table)}")

print("\nCity biases:")
for k, v in data.get("city_bias", {}).items():
    print(f"{k}: {v:+.4f}")

print("\nHorizon biases:")
for k, v in data.get("horizon_bias", {}).items():
    print(f"{k}: {v:+.4f}")

print("\nRegime biases:")
for k, v in data.get("regime_bias", {}).items():
    print(f"{k}: {v:+.4f}")
