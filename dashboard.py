import streamlit as st
import requests
import pandas as pd

from src.config import CITIES

st.set_page_config(page_title="AQI Forecast", page_icon="🌫️")
st.title("AQI Forecast Dashboard")
st.caption("3-day Air Quality Index forecast, powered by a serverless-style ML pipeline.")

API_URL = "http://localhost:5000"

city = st.selectbox("City", CITIES)

if st.button("Get 3-day forecast"):
    with st.spinner("Fetching forecast..."):
        try:
            r = requests.get(f"{API_URL}/predict/{city}", timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            st.error(
                f"Could not reach the prediction API at {API_URL}. "
                f"Make sure `python app.py` is running in another terminal. ({e})"
            )
            st.stop()

    if "error" in data:
        st.error(data["error"])
        st.stop()

    st.caption(f"Latest data as of {data['as_of']}")

    cols = st.columns(3)
    chart_rows = []

    for i, entry in enumerate(data["forecast"]):
        with cols[i]:
            if "error" in entry:
                st.warning(entry["error"])
                continue

            st.metric(f"Day +{entry['horizon']}", entry["predicted_aqi"])
            st.caption(f"Model: {entry['model_used']}")

            level = entry["alert"]["level"]
            message = entry["alert"]["message"]

            if level in ("hazardous", "very_unhealthy", "unhealthy"):
                st.error(message)
            elif level in ("unhealthy_sensitive", "moderate"):
                st.warning(message)
            else:
                st.success(message)

            chart_rows.append({"Day": f"+{entry['horizon']}d", "Predicted AQI": entry["predicted_aqi"]})

    if chart_rows:
        st.subheader("Forecast trend")
        chart_df = pd.DataFrame(chart_rows).set_index("Day")
        st.line_chart(chart_df)

    if data.get("explanation"):
        st.subheader("Why this prediction? (SHAP)")
        for item in data["explanation"]:
            direction = "increases" if item["impact"] > 0 else "decreases"
            st.write(f"- **{item['feature']}** {direction} the predicted AQI (impact: {item['impact']})")
    else:
        st.caption("No SHAP explanation available for the current best model.")
