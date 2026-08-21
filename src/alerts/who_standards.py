# WHO Global Air Quality Guidelines (2021). Values are the 24-hour mean
# limits in micrograms per cubic metre (matching Open-Meteo's units),
# except O3 which uses the 8-hour peak-season guideline as a practical
# reference point, and CO which WHO states as 4 mg/m3 (converted to
# 4000 ug/m3 here to stay in consistent units with the other pollutants).
WHO_24H_LIMITS_UGM3 = {
    "pm25": 15,
    "pm10": 45,
    "no2": 25,
    "so2": 40,
    "o3": 100,
    "co": 4000,
}

POLLUTANT_LABELS = {
    "pm25": "PM2.5",
    "pm10": "PM10",
    "no2": "NO\u2082",
    "so2": "SO\u2082",
    "o3": "O\u2083",
    "co": "CO",
}


def compare_to_who(pollutants):
    """
    pollutants: dict with any of pm25/pm10/no2/so2/o3/co (values in ug/m3).
    Returns a list of {pollutant, label, value, who_limit, ratio, exceeds}
    for whichever keys are present -- missing pollutants are skipped
    rather than raising, since not every reading has every field.
    """
    results = []
    for key, limit in WHO_24H_LIMITS_UGM3.items():
        value = pollutants.get(key)
        if value is None:
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            continue

        ratio = round(value / limit, 2) if limit else None
        results.append({
            "pollutant": key,
            "label": POLLUTANT_LABELS.get(key, key),
            "value": round(value, 1),
            "who_limit": limit,
            "ratio": ratio,
            "exceeds": ratio is not None and ratio > 1,
        })

    return results
