import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  ChevronRight,
  CloudRain,
  Gauge,
  MapPin,
  Moon,
  RefreshCw,
  Search,
  ShieldCheck,
  Sun,
  Wind,
} from "lucide-react";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { MapContainer, CircleMarker, Popup, TileLayer } from "react-leaflet";

import "leaflet/dist/leaflet.css";
import "./App.css";

const API = "http://127.0.0.1:8000";

const CITY_COORDINATES = {
  Lahore: [31.5204, 74.3587],
  Faisalabad: [31.4504, 73.135],
  Gujranwala: [32.1877, 74.1945],
  Hyderabad: [25.396, 68.3578],
  Islamabad: [33.6844, 73.0479],
  Karachi: [24.8607, 67.0011],
  Multan: [30.1575, 71.5249],
  Peshawar: [34.0151, 71.5249],
  Quetta: [30.1798, 66.975],
  Rawalpindi: [33.5651, 73.0169],
  Sialkot: [32.4945, 74.5229],
  Bahawalpur: [29.3956, 71.6836],
};

const FALLBACK_CITIES = Object.keys(CITY_COORDINATES);

function getAQIStatus(aqi) {
  if (aqi <= 50) return "Good";
  if (aqi <= 100) return "Moderate";
  if (aqi <= 150) return "Sensitive";
  if (aqi <= 200) return "Unhealthy";
  if (aqi <= 300) return "Very Unhealthy";
  return "Hazardous";
}

function getStatusClass(aqi) {
  if (aqi <= 50) return "good";
  if (aqi <= 100) return "moderate";
  if (aqi <= 150) return "sensitive";
  if (aqi <= 200) return "unhealthy";
  if (aqi <= 300) return "very-unhealthy";
  return "hazardous";
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }

  return Number(value).toFixed(0);
}

function App() {
  const [darkMode, setDarkMode] = useState(true);
  const [summary, setSummary] = useState(null);
  const [cities, setCities] = useState([]);
  const [forecasts, setForecasts] = useState([]);
  const [monthly, setMonthly] = useState([]);
  const [categories, setCategories] = useState([]);
  const [pollutants, setPollutants] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [selectedCity, setSelectedCity] = useState(null);

  async function fetchDashboard(showRefresh = false) {
    try {
      if (showRefresh) setRefreshing(true);
      else setLoading(true);

      setError("");

      const responses = await Promise.all([
         fetch(`${API}/analytics/summary`),
         fetch(`${API}/analytics/cities`),
         fetch(`${API}/forecasts`),
         fetch(`${API}/analytics/monthly`),
         fetch(`${API}/analytics/categories`),
         fetch(`${API}/analytics/pollutants`),
      ]);
      for (const response of responses) {
        if (!response.ok) {
          throw new Error("PearlsAQI API returned an error.");
        }
      }

      const [
        summaryData,
        citiesData,
        forecastsData,
        monthlyData,
        categoriesData,
        pollutantsData,
      ] = await Promise.all(responses.map((response) => response.json()));

      setSummary(summaryData);
      setCities(citiesData.cities || []);
      setMonthly(monthlyData.data || []);
      setForecasts(forecastsData.forecasts || []);
      setCategories(categoriesData.data || []);
      setPollutants(pollutantsData.data || []);
    } catch (err) {
      console.error(err);
      setError(
        "Unable to connect to PearlsAQI API. Make sure FastAPI is running on port 8000."
      );
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  useEffect(() => {
    fetchDashboard();

    const interval = setInterval(() => {
      fetchDashboard(true);
    }, 60000);

    return () => clearInterval(interval);
  }, []);

  const cityRows = useMemo(() => {
    return [...cities].sort(
      (a, b) => Number(b.average_aqi || 0) - Number(a.average_aqi || 0)
    );
  }, [cities]);

  const filteredCities = useMemo(() => {
    const query = search.toLowerCase().trim();

    if (!query) return cityRows;

    return cityRows.filter((city) =>
      String(city.city_name).toLowerCase().includes(query)
    );
  }, [cityRows, search]);

  const trendData = useMemo(() => {
    const grouped = {};

    monthly.forEach((row) => {
      const key = `${row.year}-${String(row.month).padStart(2, "0")}`;

      if (!grouped[key]) {
        grouped[key] = {
          month: key,
          total: 0,
          count: 0,
        };
      }

      grouped[key].total += Number(row.average_aqi || 0);
      grouped[key].count += 1;
    });

    return Object.values(grouped)
      .map((item) => ({
        month: item.month,
        aqi: item.count
          ? Number((item.total / item.count).toFixed(1))
          : 0,
      }))
      .sort((a, b) => a.month.localeCompare(b.month))
      .slice(-18);
  }, [monthly]);

  const topPollutants = useMemo(() => {
    return [...pollutants]
      .sort((a, b) => Number(b.average || 0) - Number(a.average || 0))
      .slice(0, 5);
  }, [pollutants]);

  const selectedCityData = useMemo(() => {
    if (!selectedCity) return null;

    return (
      cities.find((city) => city.city_name === selectedCity) || null
    );
  }, [selectedCity, cities]);

  const selectedCityForecasts = useMemo(() => {
  if (!selectedCity) return [];

  return forecasts
    .filter((forecast) => forecast.city_name === selectedCity)
    .sort((a, b) => Number(a.horizon) - Number(b.horizon));
}, [selectedCity, forecasts]);

  const averageAQI = summary?.latest_average_aqi ?? 0;
  const highestAQI = summary?.latest_highest_aqi ?? 0;
  const lowestAQI = summary?.latest_lowest_aqi ?? 0;

  return (
    <div className={darkMode ? "app dark" : "app light"}>
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <Activity size={21} />
          </div>

          <div>
            <div className="brand-name">
              Pearls<span>AQI</span>
            </div>

            <div className="brand-subtitle">
              Pakistan Air Quality Intelligence
            </div>
          </div>
        </div>

        <div className="top-actions">
          <div className="api-status">
            <span className="status-dot" />
            API LIVE
          </div>

          <button
            className="icon-button"
            onClick={() => setDarkMode((value) => !value)}
            title="Toggle theme"
          >
            {darkMode ? <Sun size={18} /> : <Moon size={18} />}
          </button>

          <button
            className="refresh-button"
            onClick={() => fetchDashboard(true)}
            disabled={refreshing}
          >
            <RefreshCw
              size={16}
              className={refreshing ? "spin" : ""}
            />
            Refresh
          </button>
        </div>
      </header>

      <main className="dashboard">
        <section className="hero">
          <div>
            <div className="eyebrow">
              <ShieldCheck size={15} />
              AI-POWERED AIR QUALITY MONITORING
            </div>

            <h1>
              Understand the air.
              <br />
              <span>Predict what comes next.</span>
            </h1>

            <p>
              Real-time intelligence, historical patterns and machine-learning
              forecasts for Pakistan's major cities.
            </p>
          </div>

          <div className="hero-badge">
            <Gauge size={22} />
            <div>
              <strong>12</strong>
              <span>Cities monitored</span>
            </div>
          </div>
        </section>

        {error && (
          <div className="error-banner">
            <AlertTriangle size={18} />
            <span>{error}</span>
          </div>
        )}

        <section className="stats-grid">
          <StatCard
            icon={<Gauge />}
            label="National average"
            value={formatNumber(averageAQI)}
            suffix="AQI"
            status={getAQIStatus(averageAQI)}
          />

          <StatCard
            icon={<AlertTriangle />}
            label="Highest city"
            value={formatNumber(highestAQI)}
            suffix={summary?.latest_highest_city || ""}
            status={getAQIStatus(highestAQI)}
          />

          <StatCard
            icon={<ShieldCheck />}
            label="Lowest city"
            value={formatNumber(lowestAQI)}
            suffix={summary?.latest_lowest_city || ""}
            status={getAQIStatus(lowestAQI)}
          />

          <StatCard
            icon={<Activity />}
            label="Historical records"
            value={summary?.historical_rows?.toLocaleString() || "—"}
            suffix="records"
            status="Historical"
          />
        </section>

        <section className="main-grid">
          <div className="panel map-panel">
            <PanelHeader
              title="Pakistan Air Quality Map"
              subtitle="12 monitored cities"
              icon={<MapPin size={18} />}
            />

            <div className="map-wrapper">
              <MapContainer
                center={[30.8, 70.8]}
                zoom={5}
                scrollWheelZoom={true}
                className="pakistan-map"
              >
                <TileLayer
                  attribution='&copy; OpenStreetMap contributors'
                  url={
                    darkMode
                      ? "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                      : "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  }
                />

                {FALLBACK_CITIES.map((cityName) => {
                  const city = cities.find(
                    (item) => item.city_name === cityName
                  );

                  const aqi = Number(city?.average_aqi || 0);

                  return (
                    <CircleMarker
                      key={cityName}
                      center={CITY_COORDINATES[cityName]}
                      radius={8}
                      pathOptions={{
                        className: "city-marker",
                        fillOpacity: 0.9,
                      }}
                      eventHandlers={{
                        click: () => setSelectedCity(cityName),
                      }}
                    >
                      <Popup>
                        <strong>{cityName}</strong>
                        <br />
                        AQI: {formatNumber(aqi)}
                        <br />
                        {getAQIStatus(aqi)}
                      </Popup>
                    </CircleMarker>
                  );
                })}
              </MapContainer>

              <div className="map-legend">
                <LegendItem label="Good" className="good" />
                <LegendItem label="Moderate" className="moderate" />
                <LegendItem label="Sensitive" className="sensitive" />
                <LegendItem label="Unhealthy" className="unhealthy" />
                <LegendItem label="Very unhealthy" className="very-unhealthy" />
              </div>
            </div>
          </div>

          <div className="panel ranking-panel">
            <PanelHeader
              title="City Intelligence"
              subtitle="Historical AQI ranking"
              icon={<BarChart3 size={18} />}
            />

            <div className="search-box">
              <Search size={16} />
              <input
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Search city..."
              />
            </div>

            <div className="city-list">
              {filteredCities.map((city, index) => {
                const aqi = Number(city.average_aqi || 0);

                return (
                  <button
                    className="city-row"
                    key={city.city_name}
                    onClick={() => setSelectedCity(city.city_name)}
                  >
                    <div className="rank">
                      {String(index + 1).padStart(2, "0")}
                    </div>

                    <div className="city-info">
                      <strong>{city.city_name}</strong>
                      <span>
                        {city.rows?.toLocaleString() || "—"} observations
                      </span>
                    </div>

                    <div
                      className={`city-aqi ${getStatusClass(aqi)}`}
                    >
                      {formatNumber(aqi)}
                    </div>

                    <ChevronRight size={16} />
                  </button>
                );
              })}
            </div>
          </div>
        </section>

                <section className="panel forecast-panel">
          <PanelHeader
            title="V9 Air Quality Forecast"
            subtitle="Recursive 24-hour, 48-hour and 72-hour predictions"
            icon={<Activity size={18} />}
          />

          <div className="forecast-header">
            <div>
              <strong>Next 72 hours</strong>
              <span>
                AI-powered forecasts from the PearlsAQI V9 XGBoost model
              </span>
            </div>

            <div className="forecast-model">
              V9 XGBoost
            </div>
          </div>

          <div className="forecast-grid">
            {[1, 2, 3].map((horizon) => {
              const forecast = forecasts.find(
                (item) =>
                  Number(item.horizon) === horizon &&
                  item.city_name === (selectedCity || "Karachi")
              );

              const labels = {
                1: "24 Hours",
                2: "48 Hours",
                3: "72 Hours",
              };

              if (!forecast) {
                return (
                  <div className="forecast-card" key={horizon}>
                    <span className="forecast-horizon">
                      {labels[horizon]}
                    </span>

                    <strong>—</strong>

                    <span>Forecast unavailable</span>
                  </div>
                );
              }

              const prediction = Number(forecast.prediction);

              return (
                <div
                  className={`forecast-card ${getStatusClass(prediction)}`}
                  key={horizon}
                >
                  <span className="forecast-horizon">
                    {labels[horizon]}
                  </span>

                  <strong>
                    {formatNumber(prediction)}
                  </strong>

                  <span>
                    {forecast.forecast_date}
                  </span>

                  <small>
                    {forecast.aqi_category}
                  </small>
                </div>
              );
            })}
          </div>

          <div className="forecast-city">
            Forecast city:{" "}
            <strong>{selectedCity || "Karachi"}</strong>
          </div>
        </section>

        <section className="panel chart-panel">
          <PanelHeader
            title="Pakistan AQI Trend"
            subtitle="National historical monthly average"
            icon={<Activity size={18} />}
          />

          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trendData}>
                <defs>
                  <linearGradient
                    id="aqiGradient"
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="1"
                  >
                    <stop
                      offset="0%"
                      stopColor="#59e391"
                      stopOpacity={0.35}
                    />
                    <stop
                      offset="100%"
                      stopColor="#59e391"
                      stopOpacity={0}
                    />
                  </linearGradient>
                </defs>

                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke={darkMode ? "#263241" : "#e4e8ee"}
                />

                <XAxis
                  dataKey="month"
                  tick={{ fontSize: 11 }}
                  stroke={darkMode ? "#718096" : "#7b8491"}
                />

                <YAxis
                  tick={{ fontSize: 11 }}
                  stroke={darkMode ? "#718096" : "#7b8491"}
                />

                <Tooltip />

                <Area
                  type="monotone"
                  dataKey="aqi"
                  stroke="#59e391"
                  strokeWidth={3}
                  fill="url(#aqiGradient)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="bottom-grid">
          <div className="panel">
            <PanelHeader
              title="AQI Category Distribution"
              subtitle="Historical observations"
              icon={<Gauge size={18} />}
            />

            <div className="category-list">
              {categories.map((item) => (
                <div className="category-item" key={item.category}>
                  <div>
                    <strong>{item.category}</strong>
                    <span>
                      {Number(item.days || 0).toLocaleString()} days
                    </span>
                  </div>

                  <div className="category-value">
                    {Number(item.percentage || 0).toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="panel">
            <PanelHeader
              title="Pollutant Overview"
              subtitle="Historical average concentration"
              icon={<Wind size={18} />}
            />

            <div className="pollutant-list">
              {topPollutants.map((item) => (
                <div className="pollutant-item" key={item.pollutant}>
                  <div className="pollutant-icon">
                    {item.pollutant === "pm25" ? (
                      <Wind size={16} />
                    ) : item.pollutant === "temperature" ? (
                      <Sun size={16} />
                    ) : (
                      <CloudRain size={16} />
                    )}
                  </div>

                  <div className="pollutant-name">
                    <strong>{String(item.pollutant).toUpperCase()}</strong>
                    <span>Average concentration</span>
                  </div>

                  <div className="pollutant-value">
                    {Number(item.average || 0).toFixed(1)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        <footer>
          <span>PearlsAQI</span>
          <span>AI-powered Pakistan air quality intelligence</span>
          <span>
            Data: {summary?.start_date || "—"} →{" "}
            {summary?.end_date || "—"}
          </span>
        </footer>
      </main>

      {selectedCityData && (
        <div
          className="city-modal-backdrop"
          onClick={() => setSelectedCity(null)}
        >
          <div
            className="city-modal"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="modal-top">
              <div>
                <span className="eyebrow">CITY PROFILE</span>
                <h2>{selectedCityData.city_name}</h2>
              </div>

              <button
                className="icon-button"
                onClick={() => setSelectedCity(null)}
              >
                ×
              </button>
            </div>

            <div className="modal-aqi">
              <div
                className={`large-aqi ${getStatusClass(
                  Number(selectedCityData.average_aqi || 0)
                )}`}
              >
                {formatNumber(selectedCityData.average_aqi)}
              </div>

              <div>
                <strong>
                  {getAQIStatus(
                    Number(selectedCityData.average_aqi || 0)
                  )}
                </strong>

                <span>Historical average AQI</span>
              </div>
            </div>

            <div className="modal-grid">
              <MiniStat
                label="Median"
                value={formatNumber(selectedCityData.median_aqi)}
              />

              <MiniStat
                label="Minimum"
                value={formatNumber(selectedCityData.minimum_aqi)}
              />

              <MiniStat
                label="Maximum"
                value={formatNumber(selectedCityData.maximum_aqi)}
              />

              <MiniStat
                label="Observations"
                value={selectedCityData.rows?.toLocaleString() || "—"}
              />
            </div>
          </div>
        </div>
      )}

      {loading && (
        <div className="loading-screen">
          <div className="loading-orb">
            <Activity size={28} />
          </div>

          <strong>Loading PearlsAQI</strong>
          <span>Connecting to Pakistan air quality intelligence...</span>
        </div>
      )}
    </div>
  );
}

function StatCard({ icon, label, value, suffix, status }) {
  return (
    <div className="stat-card">
      <div className="stat-icon">{icon}</div>

      <div className="stat-content">
        <span>{label}</span>

        <div className="stat-value">
          {value}
          <small>{suffix}</small>
        </div>

        <em>{status}</em>
      </div>
    </div>
  );
}

function PanelHeader({ title, subtitle, icon }) {
  return (
    <div className="panel-header">
      <div className="panel-title">
        <div className="panel-icon">{icon}</div>

        <div>
          <h3>{title}</h3>
          <span>{subtitle}</span>
        </div>
      </div>
    </div>
  );
}

function LegendItem({ label, className }) {
  return (
    <div className="legend-item">
      <span className={`legend-dot ${className}`} />
      {label}
    </div>
  );
}

function MiniStat({ label, value }) {
  return (
    <div className="mini-stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default App;