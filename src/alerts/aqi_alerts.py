# Standard AQI category breakpoints (US EPA scale, which AQICN's
# reported "aqi" value follows). Centralized here so app.py and
# dashboard.py never disagree on what counts as "hazardous".

def get_alert(aqi):
    if aqi is None:
        return {"level": "unknown", "message": "No data available."}

    if aqi <= 50:
        return {"level": "good", "message": "Air quality is good."}
    if aqi <= 100:
        return {"level": "moderate", "message": "Air quality is acceptable."}
    if aqi <= 150:
        return {"level": "unhealthy_sensitive", "message": "Unhealthy for sensitive groups."}
    if aqi <= 200:
        return {"level": "unhealthy", "message": "Unhealthy -- consider limiting outdoor activity."}
    if aqi <= 300:
        return {"level": "very_unhealthy", "message": "Very unhealthy -- avoid prolonged outdoor exertion."}
    return {"level": "hazardous", "message": "Hazardous -- stay indoors if possible."}
