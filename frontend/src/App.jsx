import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  Bell,
  BarChart3,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  CircleHelp,
  Cloud,
  Droplets,
  Gauge as GaugeIcon,
  HeartPulse,
  Leaf,
  MapPin,
  Menu,
  Moon,
  RefreshCw,
  Search,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  Sun,
  Wind,
  X,
  Zap,
  

} from "lucide-react";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  LabelList
} from "recharts";

import {
  MapContainer,
  CircleMarker,
  Popup,
  TileLayer,
  GeoJSON,

} from "react-leaflet";

import "leaflet/dist/leaflet.css";
import "./App.css";


const API = "/api";


const CITIES = {
  Karachi: {
    coords: [24.8607, 67.0011],
    landmark: "Mazar-e-Quaid",
    image: "/landmarks/karachi.png",
  },

  Lahore: {
    coords: [31.5204, 74.3587],
    landmark: "Minar-e-Pakistan",
    image: "/landmarks/lahore.png",
  },

  Bahawalpur: {
    coords: [29.3956, 71.6836],
    landmark: "Noor Mahal",
    image: "/landmarks/bahawalpur.png",
  },

  Islamabad: {
    coords: [33.6844, 73.0479],
    landmark: "Faisal Mosque",
    image: "/landmarks/islamabad.png",
  },

  Rawalpindi: {
    coords: [33.5651, 73.0169],
    landmark: "Ayub National Park",
    image: "/landmarks/rawalpindi.png",
  },

  Sialkot: {
    coords: [32.4945, 74.5229],
    landmark: "Iqbal Manzil",
    image: "/landmarks/sialkot.png",
  },

  Hyderabad: {
    coords: [25.396, 68.3578],
    landmark: "Mukhi House Museum",
    image: "/landmarks/hyderabad.png",
  },

  Quetta: {
    coords: [30.1798, 66.975],
    landmark: "Hanna Lake",
    image: "/landmarks/quetta.png",
  },

  Faisalabad: {
    coords: [31.4504, 73.135],
    landmark: "Clock Tower (Ghanta Ghar)",
    image: "/landmarks/faisalabad.png",
  },

  Multan: {
    coords: [30.1575, 71.5249],
    landmark: "Mausoleum of Shah Rukn-e-Alam",
    image: "/landmarks/multan.png",
  },

  Gujranwala: {
    coords: [32.1877, 74.1945],
    landmark: "Sheranwala Bagh",
    image: "/landmarks/gujranwala.png",
  },

  Peshawar: {
    coords: [34.0151, 71.5249],
    landmark: "Bala Hisar Fort",
    image: "/landmarks/peshawar.png",
  },
};


const LANDMARK_VIDEOS = {
  Karachi: {
    name: "Mazar-e-Quaid",
    video: "/landmarks/karachi.mp4",
  },

  Lahore: {
    name: "Minar-e-Pakistan",
    video: "/landmarks/lahore.mp4",
  },

  Bahawalpur: {
    name: "Noor Mahal",
    video: "/landmarks/bahawalpur.mp4",
  },

  Islamabad: {
    name: "Faisal Mosque",
    video: "/landmarks/islamabad.mp4",
  },

  Rawalpindi: {
    name: "Ayub National Park",
    video: "/landmarks/rawalpindi.mp4",
  },

  Sialkot: {
    name: "Iqbal Manzil",
    video: "/landmarks/sialkot.mp4",
  },

  Hyderabad: {
    name: "Sindh Museum",
    video: "/landmarks/hyderabad.mp4",
  },

  Quetta: {
    name: "Hanna Lake",
    video: "/landmarks/quetta.mp4",
  },

  Faisalabad: {
    name: "Clock Tower (Ghanta Ghar)",
    video: "/landmarks/faisalabad.mp4",
  },

  Multan: {
    name: "Mausoleum of Shah Rukn-e-Alam",
    video: "/landmarks/multan.mp4",
  },

  Gujranwala: {
    name: "Sheranwala Bagh",
    video: "/landmarks/gujranwala.mp4",
  },

  Peshawar: {
    name: "Bala Hisar Fort",
    video: "/landmarks/peshawar.mp4",
  },
};


const CITY_NAMES = Object.keys(CITIES);


function num(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function healthAvatar(aqi) {
  const n = Number(aqi);

  if (!Number.isFinite(n)) return "🙂";

  if (n <= 50) return "😊";
  if (n <= 100) return "🙂";
  if (n <= 150) return "😐";
  if (n <= 200) return "😷";
  if (n <= 300) return "🤢";
  return "🤮";
}


function aqiStatus(aqi) {
  if (aqi <= 50) return "Good";
  if (aqi <= 100) return "Moderate";
  if (aqi <= 150) return "Unhealthy for Sensitive Groups";
  if (aqi <= 200) return "Unhealthy";
  if (aqi <= 300) return "Very Unhealthy";
  return "Hazardous";
}



function statusClass(aqi) {
  if (aqi <= 50) return "good";
  if (aqi <= 100) return "moderate";
  if (aqi <= 150) return "sensitive";
  if (aqi <= 200) return "unhealthy";
  if (aqi <= 300) return "very-unhealthy";
  return "hazardous";
}


function fmt(value, digits = 0) {
  const n = Number(value);

  return Number.isFinite(n)
    ? n.toFixed(digits)
    : "—";
}


function getAQI(cityData) {
  return num(
    cityData?.current_conditions?.current_aqi ??
      cityData?.current_aqi ??
      cityData?.aqi
  );
}


function getWeather(cityData, key) {
  return (
    cityData?.current_conditions?.[key] ??
    cityData?.[key] ??
    null
  );
}


/* =========================================================
   AQI GAUGE
   ========================================================= */

function Gauge({ value }) {
  const safe = Math.max(0, Math.min(500, num(value)));
  const progress = safe / 500;

  const radius = 108;
  const circumference = 2 * Math.PI * radius;
  const arcLength = circumference * 0.75;
  const dashOffset = arcLength * (1 - progress);

  return (
    <div className="aqi-gauge">

      {/* GAUGE ARC + CENTER — SAME EXACT POSITION */}
      <div className="gauge-visual">


        

        <svg
          className="gauge-svg"
          viewBox="0 0 260 260"
          role="img"
          aria-label={`AQI ${fmt(value)}`}
          style={{ transform: "rotate(135.5deg)" }}

        >
          <defs>
            <linearGradient
              id="aqiGaugeGradient"
              x1="100%"
              y1="100%"
              x2="0%"
              y2="50%"
            >
              <stop offset="0%" stopColor="#35e78d" />
              <stop offset="25%" stopColor="#dce83f" />
              <stop offset="50%" stopColor="#ffb52e" />
              <stop offset="70%" stopColor="#ff633e" />
              <stop offset="88%" stopColor="#d65bea" />
              <stop offset="100%" stopColor="#ff4265" />
            </linearGradient>

            <filter
              id="gaugeGlow"
              x="-50%"
              y="-50%"
              width="200%"
              height="200%"
            >
              <feGaussianBlur
                stdDeviation="4"
                result="blur"
              />

              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* FULL COLORED ARC */}
          <circle
            cx="130"
            cy="130"
            r={radius}
            className="gauge-progress"
            transform="rotate(315 130 130)"
            strokeDasharray={`${arcLength} ${circumference}`}
            strokeDashoffset="0"
          />

          {/* ACTIVE AQI ARC */}
          <circle
            cx="130"
            cy="130"
            r={radius}
            className="gauge-value"
            transform="rotate(315 130 130) scale(1 -1)"
            strokeDasharray={`${arcLength} ${circumference}`}
            strokeDashoffset={dashOffset}
          />
        </svg>

        {/* BLACK CENTER CIRCLE */}

        {/* AQI CONTENT */}
        <div className="gauge-center">
          <strong>{fmt(value)}</strong>

          <small>AQI (US) · Live</small>

          <span className={statusClass(value)}>
            {aqiStatus(value)}
          </span>
        </div>

      </div>

      {/* SCALE */}
      <div className="gauge-scale gauge-scale-left">0</div>
      <div className="gauge-scale gauge-scale-right">500</div>

    </div>
  );
}

/* =========================================================
   SPARKLINE
   ========================================================= */

function MiniSpark({ variant = "green" }) {
  return (
    <svg
      className={`spark ${variant}`}
      viewBox="0 0 100 28"
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      <path d="M0 23 C10 20,16 22,25 18 S39 8,48 14 S62 25,72 16 S84 8,100 6" />
    </svg>
  );
}


/* =========================================================
   METRIC CARD
   ========================================================= */

function MetricCard({
  label,
  value,
  unit,
  index = 0,
}) {
  const variants = [
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
  ];

  const variant =
    variants[index] || "pm25";

  return (
    <div
      className={`metric-card metric-${variant}`}
    >

      <div className="metric-top">

        <span>{label}</span>

        <i className="metric-dot" />

      </div>


      <div className="metric-value">

        <strong>
          {value ?? "—"}
        </strong>

        <small>{unit}</small>

      </div>


      <MiniSpark
        variant={variant}
      />

    </div>
  );
}


/* =========================================================
   CITY CARD
   ========================================================= */

function CityCard({
  city,
  data,
  active,
  onClick,
}) {
  const meta = CITIES[city];
  const aqi = getAQI(data);

  return (
    <button
      className={`city-card ${active ? "active" : ""}`}
      onClick={onClick}
    >
      <div
        className="city-photo"
        style={{
          backgroundImage: `url("${meta.image}")`,
        }}
      />

      <div className="city-photo-overlay" />

      <div className="city-card-body">
<strong>
{city}</strong>
        <span>{meta.landmark}</span>

        <small className={statusClass(aqi)}>
          {aqiStatus(aqi)}
        </small>
      </div>

      <div className={`city-score ${statusClass(aqi)}`}>
        {fmt(aqi)}
      </div>
    </button>
  );
}


/* =========================================================
   PANEL
   ========================================================= */

function Panel({
  title,
  subtitle,
  icon,
  children,
  className = "",
  id,
}) {
  return (
    <section
  id={id}
  className={`panel ${className}`}
>

      <div className="panel-head">

        <div className="panel-title">

          {icon && (
            <span className="panel-icon">
              {icon}
            </span>
          )}

          <div>

            <h3>{title}</h3>

            {subtitle && (
              <p>{subtitle}</p>
            )}

          </div>

        </div>

      </div>


      {children}

    </section>
  );
}
/* =========================================================
   CUSTOM SCENARIO SIMULATOR — NEW SECTION ONLY
   ========================================================= */

function CustomScenarioSimulator({ city, data, onResult, scenarioResult }) {
  const getDefaults = (value) => {
  const weather = value?.current_conditions || {};

  return {
    pm25: Number(value?.pm25 ?? 0),
    pm10: Number(value?.pm10 ?? 0),
    o3: Number(value?.o3 ?? 0),
    no2: Number(value?.no2 ?? 0),
    so2: Number(value?.so2 ?? 0),
    co: Number(value?.co ?? 0),

    temperature: Number(
      weather?.temperature ??
      value?.temperature ??
      0
    ),

    humidity: Number(
      weather?.humidity ??
      value?.humidity ??
      0
    ),

    wind_speed: Number(
      weather?.wind_speed ??
      weather?.windspeed ??
      value?.wind_speed ??
      value?.windspeed ??
      0
    ),
  };
};

  const [values, setValues] =
    useState(() => getDefaults(data));

  const [preset, setPreset] =
    useState("Custom");

  const [result, setResult] =
    useState(null);

  const [error, setError] =
    useState("");

  const [loading, setLoading] =
    useState(false);
  const formatShapFeatureName = (feature) => {
    if (!feature) return "";

    return String(feature)
      .replaceAll("_", " ")
      .replace(/\bpm25\b/gi, "PM2.5")
      .replace(/\bpm10\b/gi, "PM10")
      .replace(/\bno2\b/gi, "NO₂")
      .replace(/\bso2\b/gi, "SO₂")
      .replace(/\bo3\b/gi, "O₃")
      .replace(/\bco\b/gi, "CO")
      .replace(/\b7d\b/gi, "7-Day")
      .replace(/\b24h\b/gi, "24-Hour")
      .replace(/\bmean\b/gi, "Mean")
      .replace(/\bstd\b/gi, "Std")
      .replace(/\bmin\b/gi, "Min")
      .replace(/\bmax\b/gi, "Max")
      .replace(/\btrend\b/gi, "Trend")
      .replace(/\binteraction\b/gi, "Interaction")
      .replace(/\bratio\b/gi, "Ratio")
      .replace(/\bsum\b/gi, "Sum")
      .replace(/\b\w/g, (char) => char.toUpperCase());
  };

  const renderShapFeatureIcon = (feature) => {
    const name = String(feature ?? "").toLowerCase();

    let iconPath;

    if (name.includes("temperature") || name.includes("temp")) {
      iconPath = (
        <>
          <path d="M12 3a2 2 0 0 0-2 2v9.2a4 4 0 1 0 4 0V5a2 2 0 0 0-2-2Z" />
          <path d="M12 8v7" />
        </>
      );
    } else if (name.includes("humidity") || name.includes("humid")) {
      iconPath = (
        <>
          <path d="M12 3s5 5.1 5 9a5 5 0 0 1-10 0c0-3.9 5-9 5-9Z" />
          <path d="M9.5 13.5a2.5 2.5 0 0 0 2.5 2.5" />
        </>
      );
    } else if (name.includes("wind")) {
      iconPath = (
        <>
          <path d="M3 8h11a2 2 0 1 0-2-2" />
          <path d="M3 12h16a2 2 0 1 1-2 2" />
          <path d="M3 16h8" />
        </>
      );
    } else {
      iconPath = (
        <>
          <circle cx="12" cy="12" r="7" />
          <circle cx="12" cy="12" r="2.5" />
          <path d="M12 2v3M12 19v3M2 12h3M19 12h3" />
        </>
      );
    }

    return (
      <svg
        width="18"
        height="18"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        {iconPath}
      </svg>
    );
  };

    const base = getDefaults(data);
    


  /* =======================================================
     QUICK SCENARIO PRESETS
     ======================================================= */

  const applyPreset = (name) => {

    const base =
      getDefaults(data);

    const presets = {

      Custom:
        base,

      "Heavy Smog": {
        ...base,

        pm25:
          Math.round(
            base.pm25 * 1.75
          ),

        pm10:
          Math.round(
            base.pm10 * 1.55
          ),

        no2:
          Math.round(
            base.no2 * 1.35
          ),

        humidity:
          Math.min(
            100,
            Math.round(
              base.humidity + 12
            )
          ),

        wind_speed:
          Math.max(
            0.5,
            Number(
              (
                base.wind_speed * 0.45
              ).toFixed(1)
            )
          ),
      },

      "Traffic Rush": {
        ...base,

        pm25:
          Math.round(
            base.pm25 * 1.35
          ),

        no2:
          Math.round(
            base.no2 * 1.65
          ),

        co:
          Number(
            (
              base.co * 1.4
            ).toFixed(2)
          ),
      },

      "After Rain": {
        ...base,

        pm25:
          Math.round(
            base.pm25 * 0.55
          ),

        pm10:
          Math.round(
            base.pm10 * 0.65
          ),

        humidity:
          Math.min(
            100,
            Math.round(
              base.humidity + 8
            )
          ),

        wind_speed:
          Number(
            (
              Math.max(
                base.wind_speed,
                4
              ) * 1.2
            ).toFixed(1)
          ),
      },

      "Hot Afternoon": {
        ...base,

        temperature:
          Number(
            (
              base.temperature + 7
            ).toFixed(1)
          ),

        humidity:
          Math.max(
            20,
            Math.round(
              base.humidity - 15
            )
          ),

        o3:
          Math.round(
            base.o3 * 1.3
          ),
      },
    };

    setPreset(name);

    setValues(
      presets[name] || base
    );

    setResult(null);
    setError("");
  };


  /* =======================================================
     RUN SCENARIO
     ======================================================= */

  const run = async () => {

    if (!city) return;

    setLoading(true);
    setError("");

    try {

      const response =
  await fetch(
    "http://127.0.0.1:5000/api/scenario-predict",
    {
      method: "POST",

      headers: {
        "Content-Type":
          "application/json",
      },

      body:
        JSON.stringify({
          city,
          scenario: values,
        }),
    }
  );

const payload =
  await response.json();

      if (
        !response.ok ||
        payload?.success === false
      ) {

        throw new Error(
          payload?.error ||
          `Scenario API failed: ${response.status}`
        );

      }


      setResult(payload);
      onResult?.(payload);

if (onResult) {
  onResult(payload);
}
    } catch (err) {

      console.error(
        "Scenario prediction failed:",
        err
      );

      setError(
        err?.message ||
        "Unable to run scenario."
      );

    } finally {

      setLoading(false);

    }
  };


  /* =======================================================
     CONTROLS
     ======================================================= */

  const controls = [

    [
      "pm25",
      "PM2.5",
      "µg/m³",
      0,
      500
    ],

    [
      "pm10",
      "PM10",
      "µg/m³",
      0,
      600
    ],

    [
      "o3",
      "O₃",
      "µg/m³",
      0,
      300
    ],

    [
      "no2",
      "NO₂",
      "µg/m³",
      0,
      300
    ],

    [
      "so2",
      "SO₂",
      "µg/m³",
      0,
      300
    ],

    [
      "co",
      "CO",
      "mg/m³",
      0,
      50
    ],

    [
      "temperature",
      "Temperature",
      "°C",
      -10,
      55
    ],

    [
      "humidity",
      "Humidity",
      "%",
      0,
      100
    ],

    [
      "wind_speed",
      "Wind Speed",
      "km/h",
      0,
      100
    ],

  ];


  /* =======================================================
     SIMULATOR UI
     ======================================================= */

  return (

    <section
      className="scenario-simulator"
      id="scenario-lab"
    >

      {/* HEADER */}

      <div className="scenario-head">

        <div>

          <div className="scenario-kicker">

            <SlidersHorizontal
              size={17}
            />

            WHAT-IF ANALYSIS

          </div>


          <h2>
            Custom Scenario Simulator
          </h2>


          <p>
            Test how pollution and weather
            changes could affect the
            selected city's AQI.
          </p>

        </div>


        <div className="scenario-city-pill">

          <MapPin size={15} />

          {city || "Select a city"}

        </div>

      </div>


      {/* QUICK PRESETS */}

      <div className="scenario-presets">

        <span>
          Quick Scenarios
        </span>


        {[
          "Custom",
          "Heavy Smog",
          "Traffic Rush",
          "After Rain",
          "Hot Afternoon",
        ].map(
          (name) => (

            <button
              key={name}
              type="button"
              className={
                preset === name
                  ? "active"
                  : ""
              }
              onClick={() =>
                applyPreset(name)
              }
            >
              {name}
            </button>

          )
        )}


        <button
          type="button"
          className="scenario-reset"
          onClick={() =>
            applyPreset("Custom")
          }
        >

          <RefreshCw size={14} />

          Reset

        </button>

      </div>


      {/* MAIN SIMULATOR AREA */}

      <div className="scenario-layout">


        {/* LEFT — CONTROLS */}

        <div className="scenario-controls">

          <div className="scenario-control-title">

            <div>

              <strong>
                Scenario Conditions
              </strong>

              <span>
                Adjust values, then run
                the simulation
              </span>

            </div>

          </div>


          <div className="scenario-control-grid">

            {controls.map(
              ([
                key,
                label,
                unit,
                min,
                max,
              ]) => {

                const step =
                  key === "co" ||
                  key === "temperature" ||
                  key === "wind_speed"
                    ? "0.1"
                    : "1";


                const value =
                  Number.isFinite(
                    Number(values[key])
                  )
                    ? values[key]
                    : 0;


                return (

                  <label
                    className="scenario-control"
                    key={key}
                  >

                    <span>

                      <b>
                        {label}
                      </b>

                      <em>
                        {unit}
                      </em>

                    </span>


                    <div className="scenario-input-row">

                      <input
                        type="range"
                        min={min}
                        max={max}
                        step={step}
                        value={value}
                        onChange={(e) => {

                          setPreset(
                            "Custom"
                          );

                          setValues(
                            (prev) => ({
                              ...prev,
                              [key]:
                                Number(
                                  e.target.value
                                ),
                            })
                          );

                          setResult(null);

                        }}
                      />


                      <input
                        type="number"
                        min={min}
                        max={max}
                        step={step}
                        value={value}
                        onChange={(e) => {

                          setPreset(
                            "Custom"
                          );

                          setValues(
                            (prev) => ({
                              ...prev,
                              [key]:
                                Number(
                                  e.target.value
                                ),
                            })
                          );

                          setResult(null);

                        }}
                      />

                    </div>

                  </label>

                );

              }
            )}

          </div>


          {/* RUN BUTTON */}

          <button
            type="button"
            className="scenario-run-btn"
            onClick={run}
            disabled={
              loading ||
              !city
            }
          >

            {loading ? (

              <>
                <RefreshCw
                  size={17}
                  className="scenario-spin"
                />

                Running Simulation...
              </>

            ) : (

              <>
                <Zap size={17} />

                Run Scenario Simulation
              </>

            )}

          </button>


          {error && (

            <div className="scenario-error">

              <CircleHelp size={16} />

              {error}

            </div>

          )}

        </div>


        {/* RIGHT — RESULTS */}

        <div className="scenario-results">
          {!result ? (
            <div className="scenario-empty">
              <div className="scenario-empty-icon">
                <Activity size={24} />
              </div>

              <strong>Simulation Results</strong>

              <span>
                Adjust the conditions and run a scenario to see
                the predicted AQI impact.
              </span>
            </div>
          ) : (
            (() => {
              const baselineAQI = Number(result?.baseline?.aqi ?? 0);
              const scenarioAQI = Number(result?.scenario?.aqi ?? 0);
              const delta = Number(result?.impact?.delta ?? 0);
              const percentage = Number(result?.impact?.percentage ?? 0);

              const getAQIClass = (aqi) => {
                if (aqi <= 50) return "good";
                if (aqi <= 100) return "moderate";
                if (aqi <= 150) return "sensitive";
                if (aqi <= 200) return "unhealthy";
                if (aqi <= 300) return "very-unhealthy";
                return "hazardous";
              };

              const baselineClass = getAQIClass(baselineAQI);
              const scenarioClass = getAQIClass(scenarioAQI);

              const impactClass =
                delta < -0.5
                  ? "improved"
                  : delta > 0.5
                  ? "worsened"
                  : "minimal";

              const impactText =
                delta < -0.5
                  ? "Improved"
                  : delta > 0.5
                  ? "Worsened"
                  : "Minimal Change";

              return (
                <div className="scenario-result-content">
                  {/* HEADER */}
                  <div className="scenario-result-head">
                    <div>
                      <span>SIMULATION RESULT</span>
                      <strong>{result.city}</strong>
                    </div>

                    <span
                      className={`scenario-impact-badge ${impactClass}`}
                    >
                      {impactText}
                    </span>
                  </div>

                  {/* 5 RESULT CARDS */}
                  <div className="scenario-result-grid">
                    {/* 1 — BASELINE AQI */}
                    <div className={`scenario-result-box baseline-box ${baselineClass}`}>
                      <span className="scenario-result-title">
                        Baseline AQI
                      </span>

                      <strong className="scenario-result-aqi-value">
                        {fmt(result?.baseline?.aqi, 0)}
                      </strong>

                      <span className="scenario-result-category">
                        {result?.baseline?.category || "—"}
                      </span>
                    </div>

                    {/* 2 — SCENARIO AQI */}
                    <div className={`scenario-result-box scenario-box ${scenarioClass}`}>
                      <span className="scenario-result-title">
                        Scenario AQI
                      </span>

                      <strong className="scenario-result-aqi-value">
                        {fmt(result?.scenario?.aqi, 0)}
                      </strong>

                      <span className="scenario-result-category">
                        {result?.scenario?.category || "—"}
                      </span>
                    </div>

                    {/* 3 — AQI CHANGE */}
                    <div
                      className={`scenario-result-box change-box ${
                        delta < 0
                          ? "improving"
                          : delta > 0
                          ? "worsening"
                          : "neutral"
                      }`}
                    >
                      <span className="scenario-result-title">
                        AQI Change (Δ)
                      </span>

                      <strong className="scenario-result-value">
                        {delta > 0 ? "+" : ""}
                        {fmt(delta, 1)}
                      </strong>

                      <span className="scenario-result-status">
                        {delta > 0
                          ? "Increase"
                          : delta < 0
                          ? "Decrease"
                          : "No Change"}
                      </span>
                    </div>

                    {/* 4 — PERCENT CHANGE */}
                    <div
                      className={`scenario-result-box change-box ${
                        percentage < 0
                          ? "improving"
                          : percentage > 0
                          ? "worsening"
                          : "neutral"
                      }`}
                    >
                      <span className="scenario-result-title">
                        Change (%)
                      </span>

                      <strong className="scenario-result-value">
                        {percentage > 0 ? "+" : ""}
                        {fmt(percentage, 1)}%
                      </strong>

                      <span className="scenario-result-status">
                        {percentage > 0
                          ? "Increase"
                          : percentage < 0
                          ? "Decrease"
                          : "No Change"}
                      </span>
                    </div>

                    {/* 5 — TOP CHANGED FEATURES */}
                    <div className="scenario-result-box features-box">
                      <span className="scenario-result-title">
                        Top Changed Features
                      </span>

                      <strong className="scenario-result-feature-count">
                        {Array.isArray(result?.changed_features)
                          ? result.changed_features.length
                          : Object.keys(scenarioValues || {}).filter(
                              (key) =>
                                Number(scenarioValues?.[key]) !==
                                Number(selectedData?.[key] ?? 0)
                            ).length}
                      </strong>

                      <span className="scenario-feature-label">
                        features modified
                      </span>
                    </div>
                  </div>

              
<section className="scenario-explainability">

  {/* MODEL EXPLAINABILITY HEADER */}
  <div className="scenario-explainability-header">
    <div className="scenario-explainability-title-wrap">
      <div className="scenario-explainability-title-icon">
        <Activity size={22} />
      </div>
      <div className="scenario-explainability-title-text">
  <strong>Model Explainability</strong>
  <span>Why did the AQI prediction change?</span>
</div>
    </div>

    <div className="scenario-explainability-tabs">
      <button className="scenario-explainability-shap-btn active">
        <Activity size={13} />
        SHAP Explanation
      </button>
    </div>
  </div>

  {scenarioResult?.explanation?.length > 0 ? (
    <>
      {/* TWO-COLUMN EXPLANATION LAYOUT */}
      <div className="scenario-explain-layout">

        {/* LEFT — PREDICTION SUMMARY */}
        <div className="scenario-explain-left">
          <div className="scenario-prediction-card">

            <div className="scenario-prediction-heading">
              <strong>Prediction Summary</strong>
            </div>

            <div className="scenario-prediction-values">
              <div className="scenario-prediction-value">
                <span>BASELINE AQI</span>
                <strong>
                  {Number(
                    scenarioResult?.baseline?.aqi ??
                    scenarioResult?.baseline_aqi ??
                    0
                  ).toFixed(1)}
                </strong>
                <span>
                  {scenarioResult?.baseline?.category ?? "—"}
                </span>
              </div>

              <div className="scenario-prediction-arrow">→</div>

              <div className="scenario-prediction-value scenario-prediction-scenario">
                <span>SCENARIO AQI</span>
                <strong>
                  {Number(
                    scenarioResult?.scenario?.aqi ??
                    scenarioResult?.scenario_aqi ??
                    0
                  ).toFixed(1)}
                </strong>
                <span>
                  {scenarioResult?.scenario?.category ?? "—"}
                </span>
              </div>
            </div>

            {/* NET CHANGE */}
<div className="scenario-net-change-card">

  <div className="scenario-net-change-main">

    <span>NET CHANGE</span>

    <strong
      className={
        Number(
          scenarioResult?.impact?.delta ??
          scenarioResult?.delta ??
          0
        ) > 0
          ? "scenario-explain-worse"
          : Number(
              scenarioResult?.impact?.delta ??
              scenarioResult?.delta ??
              0
            ) < 0
          ? "scenario-explain-better"
          : "scenario-explain-neutral"
      }
    >
      {Number(
        scenarioResult?.impact?.delta ??
        scenarioResult?.delta ??
        0
      ) > 0
        ? "+"
        : ""}
      {Number(
        scenarioResult?.impact?.delta ??
        scenarioResult?.delta ??
        0
      ).toFixed(1)}
    </strong>

    <small>
      {Number(
        scenarioResult?.impact?.delta ??
        scenarioResult?.delta ??
        0
      ) > 0
        ? "AQI Increase"
        : Number(
            scenarioResult?.impact?.delta ??
            scenarioResult?.delta ??
            0
          ) < 0
        ? "AQI Decrease"
        : "No Change"}
    </small>

    <em>
      {Number(
        scenarioResult?.impact?.percentage ?? 0
      ) > 0
        ? "+"
        : ""}
      {Number(
        scenarioResult?.impact?.percentage ?? 0
      ).toFixed(1)}
      % Impact
    </em>

  </div>

  <div className="scenario-net-change-icon">
    {Number(
      scenarioResult?.impact?.delta ??
      scenarioResult?.delta ??
      0
    ) < 0
      ? "↓"
      : Number(
          scenarioResult?.impact?.delta ??
          scenarioResult?.delta ??
          0
        ) > 0
      ? "↑"
      : "→"}
  </div>

</div>
            {/* IMPACT ANALYSIS */}
            <div className="scenario-impact-card">
              <div className="scenario-impact-heading">
                <Zap size={17} />
                <strong>Impact Analysis</strong>
              </div>
              <div className="scenario-impact-copy">
                This scenario is predicted to{" "}
                <b>
                  {Number(
                    scenarioResult?.impact?.delta ??
                    scenarioResult?.delta ??
                    0
                  ) < -0.5
                    ? "improve"
                    : Number(
                        scenarioResult?.impact?.delta ??
                        scenarioResult?.delta ??
                        0
                      ) > 0.5
                    ? "worsen"
                    : "minimally change"}
                </b>{" "}
                air quality compared with the current baseline.
              </div>
            </div>

          </div>
        </div>

        {/* RIGHT — TOP FEATURE CONTRIBUTIONS */}
        <div className="scenario-explain-right">
          <div className="scenario-explain-feature-head">
            <div>
              <strong>Top Feature Contributions</strong>
              <span>How individual features influenced the prediction</span>
            </div>

            <div className="scenario-explain-feature-count">
              {scenarioResult?.model?.feature_count ??
                scenarioResult.explanation.length} features
            </div>
          </div>
<div className="scenario-shap-direction-header">
  <span className="scenario-shap-increase-label">
    Increases AQI
  </span>

  <span className="scenario-shap-decrease-label">
    Decreases AQI
  </span>
</div>

          <div className="scenario-shap-content">
            {scenarioResult.explanation.slice(0, 6).map((item, index) => {
              const feature = item.feature;
              const contribution = Number(item.impact ?? 0);
              const maxImpact = Math.max(
                ...scenarioResult.explanation.map((x) =>
                  Math.abs(Number(x.impact ?? 0))
                ),
                1
              );
              const barWidth = Math.min(
                (Math.abs(contribution) / maxImpact) * 50,
                50
              );

            

              return (
                <div
                  className="scenario-shap-premium-row"
                  key={`${feature}-${index}`}
                >
                  <div className="scenario-shap-feature">
                    <span className="scenario-shap-feature-icon">
                      {renderShapFeatureIcon(feature)}
                    </span>
                    <strong>{formatShapFeatureName(feature)}</strong>
                  </div>

                  <div className="scenario-shap-centered-bar">
                    <div className="scenario-shap-zero" />
                    <div
                      className={`scenario-shap-fill ${
                        contribution >= 0 ? "positive" : "negative"
                      }`}
                      style={{
                        position: "absolute",
                        top: 0,
                        left:
                          contribution >= 0
                            ? "50%"
                            : `${50 - barWidth}%`,
                        width: `${barWidth}%`,
                        height: "100%",
                      }}
                    />
                  </div>

                  <div
                    className={`scenario-shap-value ${
                      contribution >= 0 ? "positive" : "negative"
                    }`}
                  >
                    {contribution >= 0 ? "+" : ""}
                    {contribution.toFixed(2)}
                  </div>
                </div>
              );
            })}

            {scenarioResult.explanation.length > 8 && (
              <details className="scenario-shap-more">
                <summary>
                  Show All {scenarioResult.explanation.length} Features
                  <ChevronDown size={14} />
                </summary>

                <div className="scenario-shap-content">
                  {scenarioResult.explanation.slice(6).map((item, index) => {
                    const feature = item.feature;
                    const contribution = Number(item.impact ?? 0);
                    const maxImpact = Math.max(
                      ...scenarioResult.explanation.map((x) =>
                        Math.abs(Number(x.impact ?? 0))
                      ),
                      1
                    );
                    const barWidth = Math.min(
                      (Math.abs(contribution) / maxImpact) * 50,
                      50
                    );

                    return (
                      <div
                        className="scenario-shap-premium-row"
                        key={`${feature}-more-${index}`}
                      >
                        <div className="scenario-shap-feature">
                          <span className="scenario-shap-feature-icon">
                            {renderShapFeatureIcon(feature)}
                          </span>
                          <strong>{formatShapFeatureName(feature)}</strong>
                        </div>

                        <div className="scenario-shap-centered-bar">
                          <div className="scenario-shap-zero" />
                          <div
                            className={`scenario-shap-fill ${
                              contribution >= 0 ? "positive" : "negative"
                            }`}
                            style={{
                              position: "absolute",
                              top: 0,
                              left:
                                contribution >= 0
                                  ? "50%"
                                  : `${50 - barWidth}%`,
                              width: `${barWidth}%`,
                              height: "100%",
                            }}
                          />
                        </div>

                        <div
                          className={`scenario-shap-value ${
                            contribution >= 0 ? "positive" : "negative"
                          }`}
                        >
                          {contribution >= 0 ? "+" : ""}
                          {contribution.toFixed(2)}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </details>
            )}
          </div>
        </div>
      </div>

      {/* SHAP INFO */}
      <div className="scenario-shap-premium-info">
        <div className="scenario-shap-info-icon">
          <ShieldCheck size={14} />
        </div>
        <div>
          <strong>How SHAP helps: </strong>
          <span>
            Each bar shows how strongly a feature influenced this AQI prediction. Longer bars = stronger influence.
          </span>
        </div>
      </div>

      {/* MODEL FOOTER */}
      <div className="scenario-explain-model-footer">
        <ShieldCheck size={13} />
        <span>
          XGBoost V4 Final · {scenarioResult?.model?.feature_count ?? 108} features
        </span>
      </div>
    </>
  ) : (
    <div className="scenario-explainability-empty">
      Run a custom scenario to see SHAP feature contributions.
    </div>
  )}

</section>
                </div>
              );
            })()
          )}
        </div>
      </div>
      </section>
    );
  }

/* =========================================================
   APP
   ========================================================= */
  function ThreeDayAQIForecast({ city, data }) {
  const forecasts = Array.isArray(data?.forecast)
    ? data.forecast.slice(0, 3)
    : [];

  const getCategory = (aqi) => {
    if (aqi <= 50) return "Good";
    if (aqi <= 100) return "Moderate";
    if (aqi <= 150) return "Unhealthy for Sensitive Groups";
    if (aqi <= 200) return "Unhealthy";
    if (aqi <= 300) return "Very Unhealthy";
    return "Hazardous";
  };

  return (
    <section
      className="three-day-forecast"
      id="aqi-forecast"
    >
      <div className="three-day-forecast-head">
        <div>
          <h2>3-Day AQI Forecast</h2>
          <p>{city} · XGBoost V4</p>
        </div>
        <span> NEXT 3 DAYS </span>
      </div>

      <div className="three-day-forecast-grid">
        {forecasts.map((item, index) => {
          const aqi = Number(
            item?.predicted_aqi ??
            item?.prediction ??
            0
          );

          const categoryClass =
            aqi <= 50 ? "good" :
            aqi <= 100 ? "moderate" :
            aqi <= 150 ? "sensitive" :
            aqi <= 200 ? "unhealthy" :
            aqi <= 300 ? "very-unhealthy" :
            "hazardous";

          return (
            <div
              className={`three-day-card ${categoryClass}`}
              key={index}
            >
              <div className="three-day-card-top">
                <strong>DAY {item?.horizon ?? index + 1}</strong>
                <small>
                  {new Date(
                    Date.now() + index * 86400000
                  ).toLocaleDateString("en-GB", {
                    weekday: "short",
                    day: "2-digit",
                    month: "short",
                  })}
                </small>
              </div>

              <div className="three-day-aqi">
                {Math.round(aqi)}
              </div>

              <div className="three-day-category">
                <i />
                {getCategory(aqi)}
              </div>

              <p>
                Predicted AQI for Day {index + 1}
              </p>
            </div>
          );
        })}
      </div>
    </section>
  );
}
/* =========================================================
   END-OF-DASHBOARD MODEL EXPLAINABILITY · SHAP DRIVERS
   Pagination is intentionally isolated from App and
   Scenario Prediction SHAP.
   ========================================================= */

const END_SHAP_FEATURES_PER_PAGE = 12;

function EndModelExplainability({ selectedData }) {
  const [shapPage, setShapPage] = useState(1);

  const featuresForDay = (dayIndex) => {
    const explanation = selectedData?.forecast?.[dayIndex]?.explanation;

    if (Array.isArray(explanation) && explanation.length > 0) {
      return explanation;
    }

    return Array.isArray(selectedData?.shap_explanation)
      ? selectedData.shap_explanation
      : [];
  };

  const dayFeatures = featuresForDay(0);

  const sortedFeatures = dayFeatures
    .slice()
    .sort(
      (a, b) =>
        Math.abs(Number(b?.impact ?? 0)) -
        Math.abs(Number(a?.impact ?? 0))
    );

  const totalFeatures = sortedFeatures.length;

  const totalPages = Math.max(
    1,
    Math.ceil(
      totalFeatures / END_SHAP_FEATURES_PER_PAGE
    )
  );

  useEffect(() => {
    setShapPage(1);
  }, [selectedData]);

  useEffect(() => {
    if (shapPage > totalPages) {
      setShapPage(totalPages);
    }
  }, [shapPage, totalPages]);

  const startIndex =
    (shapPage - 1) *
    END_SHAP_FEATURES_PER_PAGE;

  const visibleFeatures =
    sortedFeatures.slice(
      startIndex,
      startIndex +
        END_SHAP_FEATURES_PER_PAGE
    );

  const showingFrom =
    totalFeatures === 0
      ? 0
      : startIndex + 1;

  const showingTo =
    totalFeatures === 0
      ? 0
      : Math.min(
          startIndex +
            END_SHAP_FEATURES_PER_PAGE,
          totalFeatures
        );

  const maxImpact = Math.max(
    ...sortedFeatures.map((x) =>
      Math.abs(Number(x?.impact ?? 0))
    ),
    1
  );

  const formatFeatureName = (feature) => {
    if (!feature) return "";

    return String(feature)
      .replaceAll("_", " ")
      .replace(/\bpm25\b/gi, "PM2.5")
      .replace(/\bpm10\b/gi, "PM10")
      .replace(/\bno2\b/gi, "NO₂")
      .replace(/\bso2\b/gi, "SO₂")
      .replace(/\bo3\b/gi, "O₃")
      .replace(/\bco\b/gi, "CO")
      .replace(/\b7d\b/gi, "7-Day")
      .replace(/\b24h\b/gi, "24-Hour")
      .replace(/\bmean\b/gi, "Mean")
      .replace(/\bstd\b/gi, "Std")
      .replace(/\bmin\b/gi, "Min")
      .replace(/\bmax\b/gi, "Max")
      .replace(/\btrend\b/gi, "Trend")
      .replace(/\binteraction\b/gi, "Interaction")
      .replace(/\bratio\b/gi, "Ratio")
      .replace(/\bsum\b/gi, "Sum")
      .replace(/\b\w/g, (char) =>
        char.toUpperCase()
      );
  };

  const FeatureIcon = () => (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M4 19V5" />
      <path d="M4 19h16" />
      <path d="M8 16v-4" />
      <path d="M12 16V8" />
      <path d="M16 16v-7" />
    </svg>
  );

  const renderFeature = (item, index) => {
    const feature = item?.feature ?? "";
    const impact = Number(item?.impact ?? 0);

    const barWidth =
      (Math.abs(impact) / maxImpact) * 100;

    return (
      <div
        className="end-shap-row"
        key={`${feature}-${startIndex + index}`}
      >
        <div className="end-shap-feature-wrap">
          <span
            className={`end-shap-feature-icon ${
              impact >= 0
                ? "positive"
                : "negative"
            }`}
          >
            <FeatureIcon />
          </span>

          <span
  className="end-shap-feature"
  title={formatFeatureName(feature)}
>
  {formatFeatureName(feature)}
</span>
        </div>

        <div className="end-shap-track">
          <div
            className={`end-shap-bar ${
              impact >= 0
                ? "positive"
                : "negative"
            }`}
            style={{
              width: `${Math.max(
                barWidth,
                2
              )}%`,
            }}
          />
        </div>

        <span
          className={`end-shap-value ${
            impact >= 0
              ? "positive"
              : "negative"
          }`}
        >
          {impact > 0 ? "+" : ""}
          {impact.toFixed(2)}
        </span>
      </div>
    );
  };

  const getPageNumbers = () => {
    if (totalPages <= 7) {
      return Array.from(
        { length: totalPages },
        (_, index) => index + 1
      );
    }

    if (shapPage <= 4) {
      return [
        1,
        2,
        3,
        4,
        "ellipsis",
        totalPages,
      ];
    }

    if (shapPage >= totalPages - 3) {
      return [
        1,
        "ellipsis",
        totalPages - 3,
        totalPages - 2,
        totalPages - 1,
        totalPages,
      ];
    }

    return [
      1,
      "ellipsis",
      shapPage - 1,
      shapPage,
      shapPage + 1,
      "ellipsis-end",
      totalPages,
    ];
  };

  const pageNumbers = getPageNumbers();

  return (
    <section
      className="end-shap-section"
      id="model-explainability"
    >

      <div className="end-shap-header">
        <div>
          <h2>
            Model Explainability · SHAP Drivers
          </h2>

          <p>
            Feature contribution available from
            the serving model
          </p>
        </div>

        <div className="end-shap-total">
          <strong>
            {totalFeatures || 108}
          </strong>

          <span>FEATURES</span>
        </div>
      </div>

      <div className="end-shap-grid">

        {[0, 1, 2].map((dayIndex) => {
          const dayFeaturesForCard = featuresForDay(dayIndex);
          const sortedFeaturesForDay = dayFeaturesForCard
            .slice()
            .sort(
              (a, b) =>
                Math.abs(Number(b?.impact ?? 0)) -
                Math.abs(Number(a?.impact ?? 0))
            );
          const dayVisibleFeatures = sortedFeaturesForDay.slice(
            startIndex,
            startIndex + END_SHAP_FEATURES_PER_PAGE
          );

          return (
          <div
            className="end-shap-card"
            key={dayIndex}
          >

            <div className="end-shap-card-head">
              <div>
                <h3>
                  Day {dayIndex + 1}
                </h3>

                <span>
                  Top feature contributions
                </span>
              </div>

              <span className="end-shap-day-count">
                {totalFeatures || 108} features
              </span>
            </div>

            <div className="end-shap-legend">
              <span>
                <i className="positive" />
                Increases AQI
              </span>

              <span>
                <i className="negative" />
                Decreases AQI
              </span>
            </div>

            <div className="end-shap-chart">
              {dayVisibleFeatures.map(
                renderFeature
              )}
            </div>

          </div>
          );
        })}

      </div>

      {totalFeatures > 0 && (
        <>
          <span className="end-shap-pagination-summary">
            Showing {showingFrom} to{" "}
            {showingTo} of{" "}
            {totalFeatures} features
          </span>

          <div className="end-shap-pagination-controls">
            <button
              type="button"
              className="end-shap-page-arrow"
              onClick={() => setShapPage(1)}
              disabled={shapPage === 1}
              aria-label="First page"
            >
              «
            </button>

            {pageNumbers.map((page, index) =>
              typeof page === "string" ? (
                <span
                  className="end-shap-page-ellipsis"
                  key={`${page}-${index}`}
                >
                  …
                </span>
              ) : (
                <button
                  type="button"
                  className={`end-shap-page-number ${
                    shapPage === page ? "active" : ""
                  }`}
                  onClick={() => setShapPage(page)}
                  key={page}
                >
                  {page}
                </button>
              )
            )}

            <button
              type="button"
              className="end-shap-page-arrow"
              onClick={() =>
                setShapPage((page) =>
                  Math.min(
                    totalPages,
                    page + 1
                  )
                )
              }
              disabled={
                shapPage === totalPages
              }
              aria-label="Next page"
            >
              ›
            </button>

            <button
              type="button"
              className="end-shap-page-arrow"
              onClick={() =>
                setShapPage(totalPages)
              }
              disabled={
                shapPage === totalPages
              }
              aria-label="Last page"
            >
              »
            </button>
          </div>
        </>
      )}

    </section>
  );
}

function BrainIconFallback() {
  return <Activity size={22} />;
}

  function App() {

  const [dark, setDark] =
    useState(true);

  const [cityData, setCityData] =
    useState({});

  const [cities, setCities] =
    useState([]);

  const [forecasts, setForecasts] =
  useState([]);

  const [summary, setSummary] =
    useState(null);

  const [selected, setSelected] =
    useState("Karachi");

  const [comparisonCities, setComparisonCities] =
    useState(() => CITY_NAMES.slice(0, 3));

  const [compareModalOpen, setCompareModalOpen] =
    useState(false);

  const [comparisonDraft, setComparisonDraft] =
    useState(() => CITY_NAMES.slice(0, 3));

  const [search, setSearch] =
    useState("");

  const [notificationsOpen, setNotificationsOpen] =
    useState(false);
    
   const [tab, setTab] = useState("24 Hours");

  const notificationAlerts = useMemo(() => {
  return Object.entries(cityData)
    .map(([city, data]) => {
      const aqi = getAQI(data);

      if (!aqi || aqi <= 100) {
        return null;
      }

      if (aqi > 300) {
        return {
          city,
          aqi,
          level: "Hazardous",
          title: "Hazardous Air Quality",
          message: `${city} AQI has reached ${fmt(aqi)}. Avoid outdoor exposure.`,
          icon: "hazardous",
        };
      }

      if (aqi > 200) {
        return {
          city,
          aqi,
          level: "Very Unhealthy",
          title: "Very Unhealthy Air",
          message: `${city} AQI is ${fmt(aqi)}. Limit outdoor activity.`,
          icon: "very-unhealthy",
        };
      }

      if (aqi > 150) {
        return {
          city,
          aqi,
          level: "Unhealthy",
          title: "Unhealthy Air Quality",
          message: `${city} AQI is ${fmt(aqi)}. Reduce prolonged outdoor exposure.`,
          icon: "unhealthy",
        };
      }

      return {
        city,
        aqi,
        level: "Sensitive",
        title: "Air Quality Warning",
        message: `${city} AQI is ${fmt(aqi)}. Sensitive groups should take care.`,
        icon: "sensitive",
      };
    })
    .filter(Boolean)
    .sort((a, b) => b.aqi - a.aqi);
}, [cityData]);

// =========================================================
  // CUSTOM SCENARIO SIMULATOR — NEW ONLY
  // =========================================================

  const [scenarioValues, setScenarioValues] = useState({
    pm25: 0,
    pm10: 0,
    no2: 0,
    so2: 0,
    o3: 0,
    co: 0,
    temperature: 0,
    humidity: 0,
    wind_speed: 0,
  });

 const [scenarioResult, setScenarioResult] = useState(null);
const [shapExpanded, setShapExpanded] = useState(false);


const shapFeatures = Array.isArray(scenarioResult?.explanation)
  ? scenarioResult.explanation
  : [];

const visibleShapFeatures = shapExpanded
  ? shapFeatures
  : shapFeatures.slice(0, 6);

const [loading, setLoading] =
    useState(true);

  const [refreshing, setRefreshing] =
    useState(false);

  const [apiError, setApiError] =
    useState("");

  const [mobileOpen, setMobileOpen] =
    useState(false);

  const [section, setSection] =
    useState("Overview");

  const [lastUpdated, setLastUpdated] =
    useState(new Date());

  const openCompareModal = () => {
    setComparisonDraft(comparisonCities);
    setCompareModalOpen(true);
  };

  const toggleComparisonCity = (city) => {
    setComparisonDraft((current) => {
      if (current.includes(city)) {
        if (current.length <= 2) return current;
        return current.filter((item) => item !== city);
      }

      if (current.length >= 3) return current;
      return [...current, city];
    });
  };

  const applyComparison = () => {
    if (comparisonDraft.length < 2) return;
    setComparisonCities(comparisonDraft);
    setCompareModalOpen(false);
  };

  const [pakistanGeoJSON, setPakistanGeoJSON] = useState(null);
  const [trendData, setTrendData] = useState([]);

useEffect(() => {
  fetch(
    "https://raw.githubusercontent.com/glynnbird/countriesgeojson/master/pakistan.geojson"
  )
    .then((res) => {
      if (!res.ok) {
        throw new Error(`Pakistan boundary HTTP ${res.status}`);
      }
      return res.json();
    })
    .then((data) => {
      setPakistanGeoJSON(data);
    })
    .catch((err) => {
      console.error("Pakistan boundary failed:", err);
      setPakistanGeoJSON(null);
    });
}, []);

useEffect(() => {
  if (!selected) {
    setTrendData([]);
    return;
  }

  let cancelled = false;

  async function loadTrendData() {
  if (!selected) return;

  try {
    const response = await fetch(
      `http://127.0.0.1:5000/api/trends/${encodeURIComponent(selected)}`
    );

    if (!response.ok) {
      throw new Error(
        `Trend API failed: ${response.status}`
      );
    }

    const data = await response.json();

    setTrendData(
      Array.isArray(data?.trends)
        ? data.trends
        : []
    );
  } catch (err) {
    console.error("Trend data failed:", err);
    setTrendData([]);
  }
}

  

  loadTrendData();

  return () => {
    cancelled = true;
  };
}, [selected]);


  // =========================================================
  // CUSTOM SCENARIO SIMULATOR
  // =========================================================

  async function runScenarioPrediction() {
    if (!selected) return;

    try {
      setScenarioLoading(true);
      setScenarioError("");
      setScenarioResult(null);

      const response = await fetch(
  "http://127.0.0.1:5000/api/scenario-predict",
  {
        
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            city: selected,
            scenario: scenarioValues,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data?.error ||
          `Scenario API failed: ${response.status}`
        );
      }

      setScenarioResult(data);
    } catch (err) {
      console.error(
        "Scenario prediction failed:",
        err
      );

      setScenarioError(
        err?.message ||
        "Unable to run scenario prediction."
      );
    } finally {
      setScenarioLoading(false);
    }
  }
  // =========================================================
  // CUSTOM SCENARIO SIMULATOR
  // =========================================================

  const applyScenarioPreset = (preset) => {
    setScenarioPreset(preset);

    const current = selectedData || {};

    const base = {
      pm25: Number(current.pm25 ?? 0),
      pm10: Number(current.pm10 ?? 0),
      no2: Number(current.no2 ?? 0),
      so2: Number(current.so2 ?? 0),
      o3: Number(current.o3 ?? 0),
      co: Number(current.co ?? 0),
      temperature: Number(current.temperature ?? 0),
      humidity: Number(current.humidity ?? 0),
      wind_speed: Number(current.wind_speed ?? 0),
    };

    if (preset === "Custom") {
      setScenarioValues(base);
      return;
    }

    if (preset === "Heavy Smog") {
      setScenarioValues({
        ...base,
        pm25: base.pm25 * 1.8,
        pm10: base.pm10 * 1.5,
        no2: base.no2 * 1.4,
        wind_speed: Math.max(0, base.wind_speed * 0.5),
      });
      return;
    }

    if (preset === "Traffic Rush") {
      setScenarioValues({
        ...base,
        no2: base.no2 * 1.7,
        co: base.co * 1.5,
        pm25: base.pm25 * 1.25,
      });
      return;
    }

    if (preset === "After Rain") {
      setScenarioValues({
        ...base,
        pm25: base.pm25 * 0.55,
        pm10: base.pm10 * 0.6,
        humidity: Math.min(100, base.humidity + 10),
        wind_speed: base.wind_speed * 1.25,
      });
      return;
    }

    if (preset === "Hot Afternoon") {
      setScenarioValues({
        ...base,
        temperature: base.temperature + 5,
        humidity: Math.max(0, base.humidity - 15),
      });
    }
  };


  /* =========================================================
     FETCH SINGLE CITY
     ========================================================= */

  async function fetchCity(city) {

  const response = await fetch(
  `${API}/scenario-predict`,
  {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      city: selected,
      scenario: scenarioValues,
    }),
  }
);

  return response;
}

  /* =========================================================
     LOAD DASHBOARD
     ========================================================= */

  async function loadDashboard(
    refresh = false
  ) {

    try {

      if (refresh) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }

      setApiError("");


      const cityNames =
        Object.keys(CITIES);


      const responses =
        await Promise.allSettled(
          cityNames.map(
            async (city) => {

              const response =
                await fetch(
                  `${API}/predict/${encodeURIComponent(
                    city
                  )}`
                );


              if (!response.ok) {
                throw new Error(
                  `${city}: HTTP ${response.status}`
                );
              }


              const data =
                await response.json();


              const current =
                data.current_conditions || {};


              const pollutants =
                data.pollutants || {};


              return {

                city_name: city,

                current_aqi: Number(
                  current.current_aqi ?? 0
                ),

                average_aqi: Number(
                  current.current_aqi ?? 0
                ),

                temperature: Number(
                  current.temperature ?? 0
                ),

                humidity: Number(
                  current.humidity ?? 0
                ),

                wind_speed: Number(
                  current.wind_speed ?? 0
                ),

                pm25: Number(
                  pollutants.pm25 ?? 0
                ),

                pm10: Number(
                  pollutants.pm10 ?? 0
                ),

                no2: Number(
                  pollutants.no2 ?? 0
                ),

                so2: Number(
                  pollutants.so2 ?? 0
                ),

                co: Number(
                  pollutants.co ?? 0
                ),

                o3: Number(
                  pollutants.o3 ?? 0
                ),

                forecast:
                  Array.isArray(
                    data.forecast
                  )
                    ? data.forecast
                    : [],

                shap_explanation:
                  Array.isArray(
                    data.shap_explanation
                  )
                    ? data.shap_explanation
                    : [],

                as_of: data.as_of,

                model:
                  data.model || {},

                inference:
                  data.inference || {},
              };
            }
          )
        );


      const successfulCities = [];
      const failedCities = [];


      responses.forEach(
        (result, index) => {

          if (
            result.status ===
            "fulfilled"
          ) {

            successfulCities.push(
              result.value
            );

          } else {

            failedCities.push(
              cityNames[index]
            );

            console.warn(
              `Failed to load ${cityNames[index]}:`,
              result.reason
            );
          }
        }
      );


      if (
        successfulCities.length === 0
      ) {

        throw new Error(
          "No city prediction could be loaded from Flask."
        );
      }


      /* =====================================================
         STORE CITY DATA
         ===================================================== */

      const nextCityData =
        Object.fromEntries(
          successfulCities.map(
            (city) => [
              city.city_name,
              city,
            ]
          )
        );


      setCityData(
        nextCityData
      );


      /* =====================================================
         FLATTEN FORECASTS
         ===================================================== */

      const nextForecasts =
        successfulCities.flatMap(
          (city) => {

            const forecast =
              Array.isArray(
                city.forecast
              )
                ? city.forecast
                : [];


            return forecast.map(
              (item) => ({

                city_name:
                  city.city_name,

                horizon: num(
                  item.horizon,
                  1
                ),

                predicted_aqi: num(
                  item.predicted_aqi ??
                    item.prediction,
                  0
                ),

                prediction: num(
                  item.predicted_aqi ??
                    item.prediction,
                  0
                ),

                alert:
                  item.alert || null,

                model_used:
                  item.model_used ||
                  city.model?.name ||
                  "XGBoost V4 Final 108",
              })
            );
          }
        );


      setForecasts(
        nextForecasts
      );


      /* =====================================================
         SUMMARY
         ===================================================== */

      const averageAQI =
        successfulCities.reduce(
          (sum, city) =>
            sum +
            Number(
              city.current_aqi || 0
            ),
          0
        ) /
        successfulCities.length;


      setSummary({
        latest_average_aqi:
          averageAQI,

        city_count:
          successfulCities.length,

        model:
          "XGBoost V4 Final 108",
      });


      setLastUpdated(
        new Date()
      );


      /* =====================================================
         KEEP SELECTED CITY
         ===================================================== */

      const selectedExists =
        successfulCities.some(
          (city) =>
            city.city_name ===
            selected
        );


      if (!selectedExists) {

        const lahore =
          successfulCities.find(
            (city) =>
              city.city_name ===
              "Lahore"
          );


        if (lahore) {
          setSelected(
            "Lahore"
          );
        } else {
          setSelected(
            successfulCities[0]
              .city_name
          );
        }
      }


      /* =====================================================
         FAILED CITY WARNING
         ===================================================== */

      if (
        failedCities.length > 0
      ) {

        console.warn(
          "Cities unavailable:",
          failedCities.join(", ")
        );
      }

    } catch (error) {

      console.error(
        "PEARLSAQI API ERROR:",
        error
      );


      setApiError(
        "Unable to connect to the PEARLSAQI Flask API. Check that the backend is running on port 5000."
      );

    } finally {

      setLoading(false);
      setRefreshing(false);

    }
  }


  /* =========================================================
     INITIAL LOAD + AUTO REFRESH
     ========================================================= */

  useEffect(() => {

    loadDashboard();


    const timer =
      setInterval(

        () =>
          loadDashboard(true),
        60000
      );


    return () =>
      clearInterval(timer);

  }, []);


  /* =========================================================
     ROWS
     ========================================================= */

  const rows = useMemo(
    () =>
      CITY_NAMES.map(
        (city) => ({
          city_name: city,
          data: cityData[city],
        })
      ),
    [cityData]
  );

  const liveRanking = useMemo(() => {
  return CITY_NAMES
    .map((city) => ({
      city_name: city,
      data: cityData[city],
      aqi: getAQI(cityData[city]),
    }))
    .filter((row) => row.aqi > 0)
    .sort((a, b) => b.aqi - a.aqi);
}, [cityData]);


  /* =========================================================
     SEARCH
     ========================================================= */

  const visibleCities =
    useMemo(() => {

      const q =
        search
          .toLowerCase()
          .trim();


      return rows.filter(
        (row) =>
          !q ||
          row.city_name
            .toLowerCase()
            .includes(q)
      );

    }, [rows, search]);


  /* =========================================================
     SELECTED CITY
     ========================================================= */

  const selectedData =
    cityData[selected];

  const selectedAQI =
    getAQI(selectedData);

  const selectedForecasts =
    selectedData?.forecast || [];


  /* =========================================================
   FORECAST CHART
   ========================================================= */

const forecastChart = useMemo(() => {

  const current = selectedAQI;

  const horizonMap = {
    "24 Hours": 1,
    "48 Hours": 2,
    "72 Hours": 3,
  };

  const selectedHorizon = horizonMap[tab];

  const totalHours = selectedHorizon * 24;

  // Create useful time labels for the X-axis.
  const interval = totalHours / 4;

  const labels = [
    0,
    interval,
    interval * 2,
    interval * 3,
    totalHours,
  ];

  const forecastItem = selectedForecasts.find(
    (item) =>
      Number(item.horizon) === selectedHorizon
  );

  const predicted = forecastItem
    ? num(
        forecastItem.predicted_aqi ??
          forecastItem.prediction,
        current
      )
    : current;

  // Keep the forecast visually continuous between
  // current AQI and the selected forecast.
  return labels.map((hours, index) => {

    const progress =
      totalHours === 0
        ? 0
        : hours / totalHours;

    const aqi =
      current +
      (predicted - current) * progress;

    return {
      label:
        index === 0
          ? "Now"
          : `${Math.round(hours)}h`,

      aqi: Number(aqi.toFixed(1)),
    };

  });

}, [
  selectedAQI,
  selectedForecasts,
  tab,
]);

  /* =========================================================
     AVERAGE AQI
     ========================================================= */

  const averageAQI =
    useMemo(() => {

      const values =
        Object.values(cityData)
          .map(getAQI)
          .filter(
            (v) => v > 0
          );


      return values.length
        ? values.reduce(
            (sum, value) =>
              sum + value,
            0
          ) / values.length
        : 0;

    }, [cityData]);


  /* =========================================================
     MAX CITY
     ========================================================= */

  const maxCity =
    useMemo(
      () =>
        rows.reduce(
          (
            best,
            row
          ) =>
            !best ||
            getAQI(row.data) >
              getAQI(best.data)
              ? row
              : best,
          null
        ),
      [rows]
    );


  const hero =
    CITIES[selected] ||
    CITIES.Lahore;


  /* =========================================================
     POLLUTANTS
     ========================================================= */

  const pollutants = [
    [
      "PM2.5",
      selectedData?.pm25,
      "µg/m³",
    ],

    [
      "PM10",
      selectedData?.pm10,
      "µg/m³",
    ],

    [
      "NO₂",
      selectedData?.no2,
      "µg/m³",
    ],

    [
      "SO₂",
      selectedData?.so2,
      "µg/m³",
    ],

    [
      "CO",
      selectedData?.co,
      "µg/m³",
    ],

    [
      "O₃",
      selectedData?.o3,
      "µg/m³",
    ],
  ];


  /* =========================================================
     NAVIGATION
     ========================================================= */

  function navigate(name) {
  setSection(name);

  if (name === "Alerts") {
    setNotificationsOpen((open) => !open);
    setMobileOpen(false);
    return;
  }

  const targets = {
  Overview: "overview",
  Cities: "cities",
  "Scenario Lab": "scenario-lab",
  "Pakistan AQI": "pakistan-aqi",
  "AQI Forecast": "aqi-forecast",
  "Air Quality Insights": "air-quality-insights",
  "Model Explainability": "model-explainability",
};

  const targetId = targets[name];

  if (targetId) {
    const target = document.getElementById(targetId);

    if (target) {
      target.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });

      target.classList.remove("nav-main-highlight");

      // Restart animation every click
      void target.offsetWidth;

      target.classList.add("nav-main-highlight");

      setTimeout(() => {
        target.classList.remove("nav-main-highlight");
      }, 1200);
    }
  }

  setMobileOpen(false);
}


  /* =========================================================
     RENDER
     ========================================================= */

  const modelShapFeatures =
    Array.isArray(shapFeatures) && shapFeatures.length > 0
      ? shapFeatures
      : (Array.isArray(scenarioResult?.explanation)
          ? scenarioResult.explanation
          : []);

  return (

    <div
      className={`app-shell ${
        dark ? "dark" : "light"
      }`}
    >

      {/* =====================================================
          SIDEBAR
          ===================================================== */}

      <aside
        className={`sidebar ${
          mobileOpen ? "open" : ""
        }`}
      >

        <div className="brand">
  <div className="brand-logo">
  <Leaf size={30} strokeWidth={2} />
</div>

  <div>
    <strong>PEARLSAQI</strong>
    <span>Know Your Air.</span>
  </div>
</div>


        <nav>

         {[
  ["Overview", GaugeIcon],
  ["Cities", MapPin],
  ["Scenario Lab", SlidersHorizontal],
  ["Pakistan AQI", MapPin],
  ["AQI Forecast", Activity],
  ["Air Quality Insights", BarChart3],
  ["Model Explainability", Activity],
].map(
            ([name, Icon]) => (

              <button
                key={name}
                className={
                  section === name
                    ? "nav-active"
                    : ""
                }
                onClick={() =>
                  navigate(name)
                }
              >

                <Icon size={18} />

                <span>
                  {name}
                </span>

              </button>

            )
          )}

        </nav>


        <div className="data-status">

          <div className="status-head">

            <strong>
              DATA STATUS
            </strong>

            <span>
              <i /> Live
            </span>

          </div>


          <div>
            <Activity size={17} />

            <span>
              AQI Data
              <small>
                Live from V4 API
              </small>
            </span>
          </div>


          <div>
            <Cloud size={17} />

            <span>
              Weather
              <small>
                Current conditions
              </small>
            </span>
          </div>


          <div>
            <Zap size={17} />

            <span>
              Forecast
              <small>
                V4 model prediction
              </small>
            </span>
          </div>

        </div>


        
      </aside>


      {/* =====================================================
          MAIN
          ===================================================== */}

      <main className="content">

        {/* ===================================================
            TOPBAR
            =================================================== */}

        <header className="topbar">

          <button
            className="mobile-menu"
            onClick={() =>
              setMobileOpen(
                (v) => !v
              )
            }
          >
            <Menu />
          </button>


          <div className="search">

            <Search size={18} />

            <input
              value={search}
              onChange={(e) =>
                setSearch(
                  e.target.value
                )
              }
              placeholder="Search city..."
            />


            {search && (
              <button
                onClick={() =>
                  setSearch("")
                }
              >
                <X size={15} />
              </button>
            )}

          </div>


          <div className="notification-wrapper">

            <button
              className={`notification ${
                notificationsOpen ? "active" : ""
              }`}
              onClick={() =>
                setNotificationsOpen((open) => !open)
              }
              aria-label="Notifications"
              aria-expanded={notificationsOpen}
              type="button"
            >
              <Bell size={19} />

              {!notificationsOpen && <i />}
            </button>

            {notificationsOpen && (
              <div className="notification-panel">

                <div className="notification-panel-head">
                  <strong>Notifications</strong>

                  <button
                    type="button"
                    onClick={() => setNotificationsOpen(false)}
                    aria-label="Close notifications"
                  >
                    <X size={15} />
                  </button>
                </div>

                <div className="notification-item">
                  <div className="notification-item-icon">
                    <Bell size={16} />
                  </div>

                  <div>
                    <strong>Air Quality Update</strong>
                    <span>
                      {selected} AQI is currently {fmt(selectedAQI)} — {aqiStatus(selectedAQI)}.
                    </span>
                  </div>
                </div>

                <div className="notification-item">
                  <div className="notification-item-icon">
                    <RefreshCw size={16} />
                  </div>

                  <div>
                    <strong>Data Updated</strong>
                    <span>
                      Live air-quality and weather data is available.
                    </span>
                  </div>
                </div>

                <div className="notification-item">
                  <div className="notification-item-icon">
                    <Activity size={16} />
                  </div>

                  <div>
                    <strong>Forecast Ready</strong>
                    <span>
                      XGBoost V4 forecast is available for {selected}.
                    </span>
                  </div>
                </div>

              </div>
            )}

          </div>
          

          <div className="updated">

            <span>
              Last Updated
            </span>

            <strong>

              {lastUpdated.toLocaleDateString(
                undefined,
                {
                  month: "short",
                  day: "numeric",
                  year: "numeric",
                }
              )}

              {" · "}

              {lastUpdated.toLocaleTimeString(
                [],
                {
                  hour: "2-digit",
                  minute: "2-digit",
                }
              )}

            </strong>

          </div>

          <button
            className="refresh"
            onClick={() =>
              loadDashboard(true)
            }
            disabled={refreshing}
          >

            <RefreshCw
              size={17}
              className={
                refreshing
                  ? "spin"
                  : ""
              }
            />

            <span>
              Refresh
            </span>

          </button>

        </header>


        {/* ===================================================
            API WARNING
            =================================================== */}

        {apiError && (

          <div className="api-warning">

            <ShieldCheck size={18} />

            {apiError}

          </div>

        )}


        {/* ===================================================
            HERO
            =================================================== */}

        <section
  id="overview"
  className="hero"
          style={{
            position: "relative",
            overflow: "hidden",
          }}
        >

          {LANDMARK_VIDEOS[
            selected
          ] && (

            <video
              className="hero-landmark-video"
              style={{
                position: "absolute",
                inset: 0,
                width: "100%",
                height: "100%",
                objectFit: "cover",
                zIndex: 0,
                opacity: 0.72,
                pointerEvents: "none",
              }}
              src={
                LANDMARK_VIDEOS[
                  selected
                ].video
              }
              autoPlay
              muted
              loop
              playsInline
              preload="metadata"
              aria-hidden="true"
            />

          )}


          <div
            className="hero-overlay"
            style={{
              position: "absolute",
              inset: 0,
              zIndex: 1,
              background:
                "rgba(3, 13, 20, 0.58)",
              pointerEvents: "none",
            }}
          />


          <div
            className="hero-main"
            style={{
              position: "relative",
              zIndex: 2,
            }}
          >

            <div className="current-city">

              <span>
                Current City
              </span>


              <select
                id="hero-city-picker"
                value={selected}
                onChange={(e) =>
                  setSelected(
                    e.target.value
                  )
                }
                aria-label="Select city"
              >

                {CITY_NAMES.map(
                  (city) => (

                    <option
                      key={city}
                      value={city}
                    >
                      {city}
                    </option>

                  )
                )}

              </select>


              <small>

                <MapPin size={15} />

                {hero.landmark}

              </small>

            </div>


            <Gauge
              value={selectedAQI}
            />


            <div className="health-callout">

              <ShieldCheck size={25} />

              <span>

                {selectedAQI > 150
                  ? "Sensitive groups should reduce prolonged outdoor exertion."
                  : "Air quality is currently suitable for most people."}

              </span>

            </div>

          </div>


          {/* =================================================
              HERO METRICS
              ================================================= */}

          <div
            className="hero-metrics"
            style={{
              position: "relative",
              zIndex: 2,
            }}
          >

            <div className="metric-grid">

              {pollutants.map(
                (
                  [
                    label,
                    value,
                    unit,
                  ],
                  index
                ) => (

                  <MetricCard
                    key={label}
                    index={index}
                    label={label}
                    value={
                      value == null
                        ? "—"
                        : fmt(
                            value,
                            label ===
                              "CO"
                              ? 0
                              : 0
                          )
                    }
                    unit={unit}
                  />

                )
              )}

            </div>


            <div className="weather-strip">

              <div>

                <span>🌡</span>

                <strong>
                  {fmt(
                    getWeather(
                      selectedData,
                      "temperature"
                    ),
                    1
                  )}
                  °C
                </strong>

                <small>
                  Temperature
                </small>

              </div>


              <div>

                <Droplets />

                <strong>
                  {fmt(
                    getWeather(
                      selectedData,
                      "humidity"
                    )
                  )}
                  %
                </strong>

                <small>
                  Humidity
                </small>

              </div>


              <div>

                <Wind />

                <strong>
                  {fmt(
                    getWeather(
                      selectedData,
                      "wind_speed"
                    ),
                    1
                  )}{" "}
                  km/h
                </strong>

                <small>
                  Wind
                </small>

              </div>


              <div>

                <Cloud />

                <strong>
                  Live
                </strong>

                <small>
                  Weather
                </small>

              </div>

            </div>

          </div>

        </section>


        {/* ===================================================
            EXPLORE CITIES
            =================================================== */}

        <section
          className="explore"
          id="cities"
        >

          <div className="section-head">

            <div>

              <h2>
                <MapPin size={19} />

                Explore Pakistan
              </h2>

              <span>
                Click any city to explore
                its live air quality
              </span>

            </div>

          <div className="monitored-cities-badge">
            12 MONITORED CITIES
          </div>

          </div>


          <div className="city-scroller">

            {visibleCities.map(
              (row) => (

                <CityCard
                  key={row.city_name}
                  city={row.city_name}
                  data={row.data}
                  active={
                    selected ===
                    row.city_name
                  }
                  onClick={() => {
                    setSelected(row.city_name);

                    const overview =
                      document.getElementById("overview");

                    if (overview) {
                      overview.scrollIntoView({
                        behavior: "smooth",
                        block: "start",
                      });
                    }
                  }}
                />

              )
            )}

          </div>

        </section>

<CustomScenarioSimulator
  key={selected}
  city={selected}
  data={selectedData}
  scenarioResult={scenarioResult}
onResult={(payload) => {
  setScenarioResult(payload);

  const explanation =
    Array.isArray(payload?.explanation)
      ? payload.explanation
      : Array.isArray(payload?.shap_explanation)
      ? payload.shap_explanation
      : Array.isArray(payload?.result?.explanation)
      ? payload.result.explanation
      : [];


  
}}
/>

{/* =====================================================
            MODEL EXPLAINABILITY — SHAP ONLY
            ===================================================== */}

        
   

        {/* ===================================================
            DASHBOARD GRID
            =================================================== */}

        <section className="dashboard-grid">

          {/* =================================================
              MAP
              ================================================= */}

          <Panel
          id="pakistan-aqi"
            title="Pakistan Air Quality Map"
            subtitle="Live AQI across monitored cities"
            icon={<MapPin />}
            
            className="map-panel"
          >

            <div className="map">

              <MapContainer
                center={[
                  30.8,
                  70.8,
                ]}
                zoom={5}
scrollWheelZoom={false}                className="leaflet-map"
              >

                <TileLayer
  attribution="© Esri, © OpenStreetMap contributors"
  url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
  maxZoom={19}
/>

{pakistanGeoJSON && (
  <GeoJSON
    data={pakistanGeoJSON}
    style={{
      color: "#16343d",
      weight: 5,
      opacity: 1,
      fillColor: "#00e5a8",
      fillOpacity: 0.04,
      
    }}
  />
)}
                


                {CITY_NAMES.map(
                  (city) => {

                    const aqi =
                      getAQI(
                        cityData[
                          city
                        ]
                      );


                    return (

                      <CircleMarker
                        key={city}
                        center={
                          CITIES[
                            city
                          ].coords
                        }
                        radius={
                          selected ===
                          city
                            ? 11
                            : 7
                        }
                        pathOptions={{
                          color:
                            selected ===
                            city
                              ? "#ffffff"
                              : "#39e68a",

                          fillColor:
                            statusClass(
                              aqi
                            ) ===
                            "unhealthy"
                              ? "#ff6535"
                              : "#49d98d",

                          fillOpacity:
                            0.88,

                          weight: 2,
                        }}
                        eventHandlers={{
                          click: () =>
                            setSelected(
                              city
                            ),
                        }}
                      >

                        <Popup>

                          <strong>
                            {city}
                          </strong>

                          <br />

                          AQI: {fmt(aqi)}

                          <br />

                          {aqiStatus(aqi)}

                        </Popup>

                      </CircleMarker>

                    );
                  }
                )}

              </MapContainer>


              <div className="map-legend">

                <strong>
                  AQI (US)
                </strong>


                {[
                  [
                    0,
                    50,
                    "Good",
                  ],
                  [
                    51,
                    100,
                    "Moderate",
                  ],
                  [
                    101,
                    150,
                    "Sensitive",
                  ],
                  [
                    151,
                    200,
                    "Unhealthy",
                  ],
                  [
                    201,
                    300,
                    "Very Unhealthy",
                  ],
                  [
                    301,
                    500,
                    "Hazardous",
                  ],
                ].map(
                  ([
                    a,
                    b,
                    name,
                  ]) => (

                    <div key={name}>

                      <i
                        className={statusClass(
                          (a + b) /
                            2
                        )}
                      />

                      <span>
                        {a}-{b}
                      </span>

                      <small>
                        {name}
                      </small>

                    </div>

                  )
                )}

              </div>

            </div>

          </Panel>


          {/* =================================================
              FORECAST
              ================================================= */}

          <Panel
            id="forecast"
            title="AQI Forecast"
            subtitle={`${selected} · XGBoost V4`}
            icon={<Activity />}
            className="forecast-panel"
          >

            <div
              id="forecast"
              className="forecast-tabs"
            >

              {[
                "24 Hours",
                "48 Hours",
                "72 Hours",
              ].map(
                (t) => (

                  <button
                    key={t}
                    className={
                      tab === t
                        ? "active"
                        : ""
                    }
                    onClick={() =>
                      setTab(t)
                    }
                  >
                    {t}
                  </button>

                )
              )}

              <span>
                V4 ML Prediction
              </span>

            </div>


            <div className="chart">

              <ResponsiveContainer
                width="100%"
                height="100%"
              >

                <AreaChart
                  data={forecastChart}
                >

                  <defs>

                    <linearGradient
                      id="aqiFill"
                      x1="0"
                      y1="0"
                      x2="0"
                      y2="1"
                    >
                    

                      <stop
  offset="0%"
  stopColor="#00e5a8"
  stopOpacity=".30"
/>

<stop
  offset="100%"
  stopColor="#00e5a8"
  stopOpacity=".02"
/>

                    </linearGradient>

                  </defs>


                  <CartesianGrid
  strokeDasharray="3 3"
  stroke="#29404a"
  vertical={false}
/>

<XAxis
  dataKey="label"
  axisLine={false}
  tickLine={false}
  stroke="#8b9aa3"
  tick={{ fontSize: 10 }}
/>

<YAxis
  axisLine={false}
  tickLine={false}
  stroke="#8b9aa3"
  tick={{ fontSize: 10 }}
  width={32}
/>

<Tooltip
  contentStyle={{
    background: "rgba(5, 20, 27, 0.96)",
    border: "1px solid rgba(0, 229, 168, 0.35)",
    borderRadius: 10,
    color: "#fff",
  }}
/>
<LabelList
  dataKey="aqi"
  position="top"
  offset={10}
  fill="#ffffff"
  fontSize={12}
  fontWeight={700}
  formatter={(value) =>
    Number(value).toFixed(0)
  }
/>

<Area
  type="monotone"
  dataKey="aqi"
  stroke="#00e5a8"
  strokeWidth={3}
  fill="url(#aqiFill)"
  dot={{
    r: 5,
    fill: "#00e5a8",
    stroke: "#07151b",
    strokeWidth: 1,
  }}
  activeDot={{
    r: 7,
    fill: "#00e5a8",
    stroke: "#ffffff",
    strokeWidth: 2,
  }}
/>     
           </AreaChart>

              </ResponsiveContainer>

            </div>


            <div className="forecast-note">

              <Leaf size={16} />

              Forecast generated by
              the PEARLSAQI XGBoost V4
              production model.

            </div>

          </Panel>


          {/* =================================================
              FORECAST INSIGHTS
              ================================================= */}

          <Panel
            title="Forecast Insights"
            subtitle="Model-driven air quality insight"
            icon={<Zap />}
            className="insights-panel"

          >

            {(() => {

              const predicted =
                num(
                  selectedForecasts[0]
                    ?.predicted_aqi ??
                    selectedForecasts[0]
                      ?.prediction,
                  selectedAQI
                );


              const delta =
                predicted -
                selectedAQI;


              return (

                <>

                  <div className="insight-card">

                    <div className="insight-icon">
                      <Leaf />
                    </div>

                    <div className="insight-text">
                      <span className="label">
                        Air quality is expected to
                      </span>

                      <strong className="status">
                        {delta < -1
                          ? "improve"
                          : delta > 1
                          ? "worsen"
                          : "remain stable"}
                      </strong>

                      <span className="description">
                        over the next forecast horizon in {selected}.
                      </span>
                    </div>

                  </div>

                  <div className="insight-stats">

                    <div>

                      <span>
                        Change
                      </span>

                      <strong>

                        {delta >= 0
                          ? "+"
                          : ""}

                        {fmt(
                          delta,
                          1
                        )} AQI

                      </strong>

                    </div>


                    <div>

                      <span>
                        Model R²
                      </span>

                      <strong>
                        {fmt(
                          selectedData
                            ?.model
                            ?.test_r2,
                          3
                        )}
                      </strong>

                    </div>


                    <div>

                      <span>
                        Top Feature
                      </span>

                      <strong>
  {selectedData?.inference?.winning_feature === "pm25_pollution_ratio"
    ? "PM2.5 Impact"
    : selectedData?.inference?.winning_feature || "PM2.5 Impact"}
</strong>

                    </div>

                  </div>


                  <div className="why">

                    <strong>
                      Why?
                    </strong>

                    <p>

                      The production model
                      combines 108 engineered
                      features from AQI,
                      weather and pollution
                      relationships to produce
                      the forecast.

                    </p>

                  </div>

                </>

              );

            })()}

          </Panel>

  {/* =================================================
              3-DAY AQI FORECAST
              ================================================= */}
<ThreeDayAQIForecast
  city={selected}
  data={selectedData}
/>
<div
  id="air-quality-insights"
  className="air-quality-insights-heading"
>
  <span className="panel-icon">
   <BarChart3 size={22} />
  </span>

  <div>
    <h3>Air Quality Insights</h3>
    <p>Pollution levels, health guidance, and city comparison</p>
  </div>
</div>
          
          {/* =================================================
              POLLUTION BREAKDOWN
              ================================================= */}

          <Panel
            title="Pollution Breakdown"
            subtitle="Current pollutant readings"
            icon={<BarChart3 />}
            id="analytics"
            className="breakdown-panel"
          >

            <div className="donut-wrap">
  <div className="donut">
    <div className="donut-inner">
      <span>
        {fmt(selectedAQI)}
      </span>

      <small>
        AQI
      </small>
    </div>
  </div>
</div>

            <div className="breakdown-list">

  <div>
    <i className="pm25" />
    <span>PM2.5</span>
    <strong>
      {fmt(selectedData?.pm25, 1)}
    </strong>
  </div>

  <div>
    <i className="pm10" />
    <span>PM10</span>
    <strong>
      {fmt(selectedData?.pm10, 1)}
    </strong>
  </div>

  <div>
    <i className="no2" />
    <span>NO₂</span>
    <strong>
      {fmt(selectedData?.no2, 1)}
    </strong>
  </div>

  <div>
    <i className="so2" />
    <span>SO₂</span>
    <strong>
      {fmt(selectedData?.so2, 1)}
    </strong>
  </div>

  <div>
    <i className="co" />
    <span>CO</span>
    <strong>
      {fmt(selectedData?.co, 1)}
    </strong>
  </div>

  <div>
    <i className="o3" />
    <span>O₃</span>
    <strong>
      {fmt(selectedData?.o3, 1)}
    </strong>
  </div>

</div>

          </Panel>


          {/* =================================================
              HEALTH
              ================================================= */}

          <Panel
             id="health-guide"
            title="Health Recommendation"
            subtitle={`Based on current AQI in ${selected}`}
            icon={<HeartPulse />}
            className="health-panel"
          >

            <div className="health-content">

  <div className="health-main">

    <div
  className={`health-avatar ${statusClass(
    selectedAQI
  )}`}
>
  {healthAvatar(selectedAQI)}
</div>

    <div className="health-summary">

      <div className="health-aqi-label">
        Current AQI
      </div>

      <div className="health-aqi-value">
        {fmt(selectedAQI, 0)}
      </div>

      <div className="health-status">
        {aqiStatus(selectedAQI)}
      </div>

      <p className="health-recommendation">
        {selectedAQI > 150
          ? "Sensitive groups should limit prolonged outdoor exposure. Consider reducing strenuous activity outdoors."
          : selectedAQI > 100
          ? "Sensitive individuals should consider reducing prolonged outdoor exertion."
          : "Most people can continue normal outdoor activities."}
      </p>

    </div>

  </div>


  <div className="health-guidance">

    <div className="health-guidance-title">
      Health guidance
    </div>

    <div className="health-actions">

      <div className="health-action">
        <span>•</span>
        <p>
          {selectedAQI > 100
            ? "Reduce prolonged outdoor exertion."
            : "Normal outdoor activity is generally fine."}
        </p>
      </div>

      <div className="health-action">
        <span>•</span>
        <p>
          {selectedAQI > 100
            ? "Keep outdoor exposure shorter when possible."
            : "Stay hydrated and monitor air quality."}
        </p>
      </div>

      <div className="health-action">
        <span>•</span>
        <p>
          {selectedAQI > 150
            ? "Sensitive groups should take extra precautions."
            : "Check AQI again before prolonged outdoor activity."}
        </p>
      </div>

    </div>

  </div>

</div>

          </Panel>

          {/* =================================================
    COMPARE CITIES
    ================================================= */}

<Panel
  id="compare-cities"
  title="Compare Cities"
  subtitle="Quick comparison of monitored locations"
  icon={<SlidersHorizontal />}
  className="compare-panel"
>
  <div className="quick-comparison">

    <div className="quick-comparison-header">

      <div className="compare-tags">
        {comparisonCities.map((city) => (
          <button
            key={city}
            type="button"
            onClick={() => setSelected(city)}
          >
            {city} · {fmt(getAQI(cityData?.[city]))}
          </button>
        ))}
      </div>

      <button
        className="compare-action"
        type="button"
        onClick={openCompareModal}
      >
        Compare
        <ChevronRight size={16} />
      </button>

    </div>

    <div className="comparison-controls">
      <div className="comparison-city-chips">
        {comparisonCities.map((city) => (
          <button
            type="button"
            className="city-chip"
            key={city}
            onClick={() => setSelected(city)}
          >
            {city} · {fmt(getAQI(cityData?.[city]))}
          </button>
        ))}
      </div>

      <button
        type="button"
        className="explore-btn"
        onClick={openCompareModal}
      >
        Compare <span>›</span>
      </button>
    </div>

    <div className="comparison-cards">

      {comparisonCities.map((city) => {

        const cityAQI = getAQI(
          cityData?.[city]
        );

        const comparisonAQIs = comparisonCities
          .map((item) => getAQI(cityData?.[item]));

        const lowestAQI = Math.min(...comparisonAQIs);
        const highestAQI = Math.max(...comparisonAQIs);

        return (
          <div
            key={city}
            className={`comparison-card ${statusClass(
              cityAQI
            )}`}
          >

            <div className="comparison-card-top">
              <h3>{city}</h3>
            </div>

            <div className="comparison-aqi">
              {fmt(cityAQI)}
            </div>

            <div className="comparison-category">
              <span className="status-dot"></span>

              <span>
                {aqiStatus(cityAQI)}
              </span>
            </div>

            <div className="comparison-divider"></div>

            <div className="comparison-status">

              <span className="status-icon">
                {cityAQI === lowestAQI
                  ? "≈"
                  : cityAQI === highestAQI
                  ? "↗"
                  : "≈"}
              </span>

              <span>
                {cityAQI === lowestAQI
                  ? "Lowest AQI"
                  : cityAQI === highestAQI
                  ? "Highest AQI"
                  : "Better than average"}
              </span>

            </div>

          </div>
        );
      })}

    </div>

    <div className="comparison-summary">

      <div className="summary-icon">
        ↗
      </div>

      <p>
        Air quality varies across selected cities.
        <br />

        <strong>
          {comparisonCities.reduce((best, city) =>
            getAQI(cityData?.[city]) < getAQI(cityData?.[best])
              ? city
              : best
          , comparisonCities[0])}
        </strong>{" "}
        currently has the lowest AQI, while{" "}
        <strong>
          {comparisonCities.reduce((best, city) =>
            getAQI(cityData?.[city]) > getAQI(cityData?.[best])
              ? city
              : best
          , comparisonCities[0])}
        </strong>{" "}
        has the highest.
      </p>

    </div>

  </div>

  {compareModalOpen && (
    <>
      <style>{`
        .compare-modal-overlay {
          position: fixed;
          inset: 0;
          z-index: 9999;
          background: rgba(2, 12, 18, 0.72);
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 20px;
          box-sizing: border-box;
        }
        .compare-modal {
          width: min(460px, 100%);
          max-height: 80vh;
          overflow-y: auto;
          background: #081923;
          border: 1px solid #1d3b49;
          border-radius: 18px;
          padding: 20px;
          box-sizing: border-box;
          box-shadow: 0 24px 70px rgba(0, 0, 0, 0.45);
        }
        .compare-modal-header {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 16px;
          margin-bottom: 18px;
        }
        .compare-modal-header h3 { margin: 0; font-size: 20px; font-weight: 600; }
        .compare-modal-header p { margin: 5px 0 0; font-size: 11px; opacity: .65; }
        .compare-modal-header > button {
          width: 32px; height: 32px; border: 1px solid #294653; border-radius: 9px;
          background: transparent; color: inherit; display: grid; place-items: center; cursor: pointer;
        }
        .compare-city-options {
          display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px;
        }
        .compare-city-option {
          min-height: 44px; padding: 9px 11px; border: 1px solid #203d4b; border-radius: 10px;
          background: #0b1d27; color: inherit; display: flex; align-items: center;
          justify-content: space-between; cursor: pointer; text-align: left;
        }
        .compare-city-option.selected {
          border-color: #19e6a1; background: rgba(25, 230, 161, .08);
        }
        .compare-city-check { width: 20px; height: 20px; border-radius: 50%; display: grid; place-items: center; font-size: 12px; }
        .compare-confirm-btn {
          width: 100%; height: 42px; margin-top: 16px; border: 0; border-radius: 10px;
          background: #19e6a1; color: #061217; font-weight: 600; cursor: pointer;
        }
        .compare-confirm-btn:disabled { opacity: .4; cursor: not-allowed; }
      `}</style>
      <div
        className="compare-modal-overlay"
        onClick={() => setCompareModalOpen(false)}
      >
        <div
          className="compare-modal"
          onClick={(event) => event.stopPropagation()}
        >
          <div className="compare-modal-header">
            <div>
              <h3>Select Cities to Compare</h3>
              <p>Select 2 or 3 cities</p>
            </div>

            <button
              type="button"
              onClick={() => setCompareModalOpen(false)}
              aria-label="Close"
            >
              <X size={18} />
            </button>
          </div>

          <div className="compare-city-options">
            {CITY_NAMES.map((city) => {
              const checked = comparisonDraft.includes(city);

              return (
                <button
                  type="button"
                  key={city}
                  className={`compare-city-option ${checked ? "selected" : ""}`}
                  onClick={() => toggleComparisonCity(city)}
                >
                  <span>{city}</span>
                  <span className="compare-city-check">
                    {checked ? "✓" : ""}
                  </span>
                </button>
              );
            })}
          </div>

          <button
            type="button"
            className="compare-confirm-btn"
            disabled={comparisonDraft.length < 2}
            onClick={applyComparison}
          >
            Compare {comparisonDraft.length} Cities
          </button>
        </div>
      </div>
    </>
  )}

</Panel>

        </section>


        {/* ===================================================
            CITY DIRECTORY
            =================================================== */}
        <div className="directory-ranking-layout">

          <section
            id="citites"
            className="city-directory"
          >

            <div className="directory-head">

              <div>

                <span className="eyebrow">
                  CITY DIRECTORY
                </span>

                <h2>
                  Every monitored city
                  at a glance
                </h2>

              </div>


              <div className="national-stat">

                <GaugeIcon size={18} />

                <strong>
                  {fmt(
                    averageAQI,
                    0
                  )}
                </strong>

                <span>
                  Current average
                </span>

              </div>

            </div>


            <div className="directory-grid">

              {visibleCities.map(
                (row) => (

                  <button
                    key={row.city_name}
                    className="directory-row"
                    onClick={() =>
                      setSelected(
                        row.city_name
                      )
                    }
                  >

                    <span>
                      {row.city_name}
                    </span>

                    <small>
                      {
                        CITIES[
                          row.city_name
                        ].landmark
                      }
                    </small>

                    <strong
                      className={statusClass(
                        getAQI(
                          row.data
                        )
                      )}
                    >
                      {fmt(
                        getAQI(
                          row.data
                        )
                      )}
                    </strong>

                    <ChevronRight
                      size={17}
                    />

                  </button>

                )
              )}

            </div>

          </section>

          <section
            className="city-ranking"
            id="city-ranking"
          >
            <div className="section-head">
              <h3>Live AQI City Ranking</h3>
              <span>
                Updated{" "}
                {lastUpdated.toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </span>
            </div>

            <div className="city-ranking-list">
              {liveRanking.map((city, index) => (
                <button
                  key={city.city_name}
                  className="city-ranking-item"
                  onClick={() => setSelected(city.city_name)}
                >
                  <span className="ranking-number">
                    {index + 1}
                  </span>

                  <span className="ranking-city">
                    {city.city_name}
                  </span>

                  <strong className={statusClass(city.aqi)}>
                    {fmt(city.aqi, 0)}
                  </strong>

                  <ChevronRight size={18} />
                </button>
              ))}
            </div>
            
                   </section>

        </div>

        
{/* =====================================================
    MODEL EXPLAINABILITY · SHAP DRIVERS
===================================================== */}
<EndModelExplainability
  selectedData={selectedData}
/>

        {/* =====================================================
            LOADING
            ===================================================== */}

      {loading && (

        <div className="loading">

          <div className="loader">
            <Activity />
          </div>

          <strong>
            Loading PEARLSAQI
          </strong>

          <span>
            Connecting to live
            air-quality intelligence…
          </span>

        </div>

      )}


        {/* =====================================================
            PEARLSAQI FOOTER
            ===================================================== */}
        <footer className="pearlsaqi-footer">
          <div className="pearlsaqi-footer-grid">

            <div className="pearlsaqi-footer-column">
              <div className="pearlsaqi-footer-heading">
                <div className="pearlsaqi-footer-icon">
                  <BrainIconFallback />
                </div>
                <h3>MODEL INFORMATION</h3>
              </div>
              <div className="pearlsaqi-footer-detail">
                <span>Model Name</span>
                <strong>XGBoost V4 Final</strong>
              </div>
              <div className="pearlsaqi-footer-detail">
                <span>Model Version</span>
                <strong>V4.0.0</strong>
              </div>
              <div className="pearlsaqi-footer-detail">
                <span>Total Features</span>
                <strong>108</strong>
              </div>
              <div className="pearlsaqi-footer-detail">
                <span>Model Type</span>
                <strong>Regression</strong>
              </div>
            </div>

            <div className="pearlsaqi-footer-column">
              <div className="pearlsaqi-footer-heading">
                <div className="pearlsaqi-footer-icon">
                  <RefreshCw size={22} />
                </div>
                <h3>DATA &amp; SYSTEM STATUS</h3>
              </div>
              <div className="pearlsaqi-footer-row">
                <span>Data Source</span>
                <strong>Live from V4 API</strong>
              </div>
              <div className="pearlsaqi-footer-row">
                <span>Data Status</span>
                <strong className="status-live">● Live</strong>
              </div>
              <div className="pearlsaqi-footer-row">
                <span>Last Updated</span>
                <strong>
                  {lastUpdated.toLocaleDateString(undefined, {
                    day: "2-digit",
                    month: "short",
                    year: "numeric",
                  })}{" "}
                  ·{" "}
                  {lastUpdated.toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </strong>
              </div>
              <div className="pearlsaqi-footer-row">
                <span>System Status</span>
                <strong className="status-live">● Operational</strong>
              </div>
            </div>

            <div className="pearlsaqi-footer-column">
              <div className="pearlsaqi-footer-heading">
                <div className="pearlsaqi-footer-icon">
                  <BarChart3 size={22} />
                </div>
                <h3>ACCURACY &amp; PERFORMANCE</h3>
              </div>
              <div className="pearlsaqi-footer-detail">
                <span>Forecast Horizon</span>
                <strong>24 / 48 / 72 Hours</strong>
              </div>
              <div className="pearlsaqi-footer-detail">
                <span>Explainability</span>
                <strong>108 SHAP Features</strong>
              </div>
              <div className="pearlsaqi-footer-detail">
                <span>Monitored Coverage</span>
                <strong>12 Pakistani Cities</strong>
              </div>
              <div className="pearlsaqi-footer-detail">
                <span>Data Pipeline</span>
                <strong>Real-time API</strong>
              </div>
            </div>

            <div className="pearlsaqi-footer-column pearlsaqi-about">
              <div className="pearlsaqi-footer-heading">
                <div className="pearlsaqi-footer-icon">
                  <Leaf size={22} />
                </div>
                <h3>ABOUT PEARLSAQI</h3>
              </div>
              <p>
                PEARLSAQI is an air quality intelligence dashboard providing
                live monitoring, forecasting, city comparison, and explainable
                AQI insights across Pakistan.
              </p>
              <button
                type="button"
                className="pearlsaqi-learn-more"
                onClick={() => {
                  const overview = document.getElementById("overview");
                  if (overview) {
                    overview.scrollIntoView({
                      behavior: "smooth",
                      block: "start",
                    });
                  }
                }}
              >
                Learn More
                <ChevronRight size={17} />
              </button>
            </div>

          </div>

          <div className="pearlsaqi-footer-badges">
            <div className="pearlsaqi-footer-badge">
              <ShieldCheck size={22} />
              <div>
                <strong>Trusted Data</strong>
                <span>Validated &amp; Verified</span>
              </div>
            </div>
            <div className="pearlsaqi-footer-badge">
              <RefreshCw size={22} />
              <div>
                <strong>Real-time Updates</strong>
                <span>Live V4 API</span>
              </div>
            </div>
            <div className="pearlsaqi-footer-badge">
              <MapPin size={22} />
              <div>
                <strong>12 Monitored Cities</strong>
                <span>Across Pakistan</span>
              </div>
            </div>
            <div className="pearlsaqi-footer-badge">
              <BarChart3 size={22} />
              <div>
                <strong>Advanced Analytics</strong>
                <span>AI-powered insights</span>
              </div>
            </div>
            <div className="pearlsaqi-footer-badge">
              <Activity size={22} />
              <div>
                <strong>Data Intelligence</strong>
                <span>Forecast &amp; SHAP analysis</span>
              </div>
            </div>
          </div>

          <div className="pearlsaqi-footer-bottom">
            <div className="pearlsaqi-brand">
              <div className="pearlsaqi-brand-mark">
                <Leaf size={24} />
              </div>
              <div>
                <strong>PEARLSAQI</strong>
                <span>Air Quality Intelligence Dashboard</span>
              </div>
            </div>

            <div className="pearlsaqi-copyright">
              © {new Date().getFullYear()} PEARLSAQI. All rights reserved.
            </div>

           

            <button
              type="button"
              className="pearlsaqi-back-top"
              aria-label="Back to top"
              onClick={() =>
                window.scrollTo({
                  top: 0,
                  behavior: "smooth",
                })
              }
            >
              ↑
            </button>
          </div>
        </footer>

    </main>

  </div>
  );

}

export default App;

