/* ============================================================
   PEARLSAQI — PREMIUM DASHBOARD CONTROLLER
   Connects directly to the existing Flask API.
   ============================================================ */

const state = {
  city: null,
  prediction: null,
  trends: null,
  models: null,
  charts: {},
  loading: false,
};


/* ============================================================
   DOM HELPERS
   ============================================================ */

const $ = (selector) => document.querySelector(selector);

const $$ = (selector) => document.querySelectorAll(selector);

function escapeHTML(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function number(value, digits = 1) {
  const n = Number(value);

  if (!Number.isFinite(n)) {
    return "—";
  }

  return n.toFixed(digits);
}

function categoryForAQI(aqi) {
  const value = Number(aqi);

  if (!Number.isFinite(value)) {
    return {
      label: "Unknown",
      className: "moderate",
    };
  }

  if (value <= 50) {
    return {
      label: "Good",
      className: "good",
    };
  }

  if (value <= 100) {
    return {
      label: "Moderate",
      className: "moderate",
    };
  }

  if (value <= 150) {
    return {
      label: "Unhealthy for Sensitive Groups",
      className: "sensitive",
    };
  }

  if (value <= 200) {
    return {
      label: "Unhealthy",
      className: "unhealthy",
    };
  }

  if (value <= 300) {
    return {
      label: "Very Unhealthy",
      className: "very-unhealthy",
    };
  }

  return {
    label: "Hazardous",
    className: "hazardous",
  };
}

function categoryColor(aqi) {
  const category = categoryForAQI(aqi);

  const colors = {
    good: "#75d6a5",
    moderate: "#d7c979",
    sensitive: "#e8a86a",
    unhealthy: "#df796e",
    "very-unhealthy": "#c96b92",
    hazardous: "#aa6fca",
  };

  return colors[category.className] || "#76d7b7";
}

function showLoading(element, message = "Loading intelligence…") {
  if (!element) return;

  element.innerHTML = `
    <div class="loading-card">
      <span class="loader"></span>
      <span>${escapeHTML(message)}</span>
    </div>
  `;
}

function showError(element, message) {
  if (!element) return;

  element.innerHTML = `
    <div class="error-state">
      ${escapeHTML(message)}
    </div>
  `;
}


/* ============================================================
   CITY / NAVIGATION
   ============================================================ */

function setupNavigation() {
  $$(".nav-item").forEach((item) => {
    item.addEventListener("click", () => {
      $$(".nav-item").forEach((nav) => nav.classList.remove("active"));
      item.classList.add("active");
    });
  });

  const sections = [...$$(".page-section")];

  if (!sections.length) return;

  const observer = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

      if (!visible) return;

      const id = visible.target.id;

      $$(".nav-item").forEach((item) => {
        item.classList.toggle(
          "active",
          item.dataset.section === id
        );
      });
    },
    {
      rootMargin: "-20% 0px -65% 0px",
      threshold: [0.05, 0.2, 0.5],
    }
  );

  sections.forEach((section) => observer.observe(section));
}


/* ============================================================
   API
   ============================================================ */

async function api(url, options = {}) {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  let data;

  try {
    data = await response.json();
  } catch {
    throw new Error(
      `Server returned an invalid response (${response.status}).`
    );
  }

  if (!response.ok) {
    throw new Error(
      data?.error ||
      `Request failed with status ${response.status}.`
    );
  }

  return data;
}


/* ============================================================
   MAIN FORECAST
   ============================================================ */

async function loadForecast(city) {
  if (!city) return;

  state.city = city;
  state.loading = true;

  const forecastArea = $("#forecast-cards");
  const currentArea = $("#current-conditions");
  const pollutantArea = $("#pollutant-values");

  showLoading(
    forecastArea,
    `Loading ${city} atmospheric intelligence…`
  );

  if (currentArea) {
    currentArea.innerHTML = "";
  }

  try {
    const data = await api(
      `/predict/${encodeURIComponent(city)}`
    );

    state.prediction = data;

    renderCurrentConditions(data);
    renderForecast(data);
    renderPollutants(data);
    renderExplanation(data);
    renderWhoComparison(data);

    await loadTrends(city);

    updateDashboardMeta(data);
  } catch (error) {
    showError(
      forecastArea,
      `Unable to load ${city}: ${error.message}`
    );
  } finally {
    state.loading = false;
  }
}


/* ============================================================
   CURRENT CONDITIONS
   ============================================================ */

function renderCurrentConditions(data) {
  const container = $("#current-conditions");

  if (!container) return;

  const conditions = data.current_conditions || {};

  const aqi = conditions.current_aqi;
  const category = categoryForAQI(aqi);

  const scalePosition = Math.min(
    Math.max((Number(aqi) / 300) * 100, 0),
    100
  );

  container.innerHTML = `
    <div class="overview-grid">

      <article class="aqi-card">
        <div class="card-label">
          <span>CURRENT AIR QUALITY</span>
          <span class="live-pill">LIVE</span>
        </div>

        <div class="aqi-main">
          <div class="aqi-number">
            ${number(aqi, 0)}
          </div>

          <div class="aqi-meta">
            <div class="aqi-category ${category.className}">
              ${escapeHTML(category.label)}
            </div>

            <div class="aqi-location">
              ${escapeHTML(data.city || state.city)}
            </div>
          </div>
        </div>

        <div class="aqi-scale">
          <div class="scale-track">
            <div
              class="scale-position"
              style="left:${scalePosition}%"
            ></div>
          </div>

          <div class="scale-labels">
            <span>0</span>
            <span>50</span>
            <span>100</span>
            <span>150</span>
            <span>200</span>
            <span>300+</span>
          </div>
        </div>
      </article>


      <article class="conditions-card">
        <div class="card-heading">
          <div>
            <span class="eyebrow-small">ATMOSPHERIC CONDITIONS</span>
            <h2>${escapeHTML(
              conditions.description || "Current conditions"
            )}</h2>
          </div>

          <div class="weather-icon">
            ${escapeHTML(conditions.icon || "◌")}
          </div>
        </div>

        <div class="condition-grid">

          <div class="condition-stat">
            <span>Temperature</span>
            <strong>
              ${number(conditions.temperature)}°C
            </strong>
          </div>

          <div class="condition-stat">
            <span>Humidity</span>
            <strong>
              ${number(conditions.humidity, 0)}%
            </strong>
          </div>

          <div class="condition-stat">
            <span>Current AQI</span>
            <strong>
              ${number(conditions.current_aqi, 0)}
            </strong>
          </div>

          <div class="condition-stat">
            <span>Data timestamp</span>
            <strong style="font-size:12px">
              ${escapeHTML(formatDate(data.as_of))}
            </strong>
          </div>

        </div>
      </article>

    </div>
  `;
}


/* ============================================================
   FORECAST
   ============================================================ */

function renderForecast(data) {
  const container = $("#forecast-cards");

  if (!container) return;

  const forecast = data.forecast || [];

  if (!forecast.length) {
    container.innerHTML = `
      <div class="empty-state">
        No forecast data available.
      </div>
    `;

    return;
  }

  container.innerHTML = forecast
    .map((item, index) => {
      const category = categoryForAQI(item.predicted_aqi);

      return `
        <article class="forecast-card ${
          index === 0 ? "highlight" : ""
        }">

          <div class="forecast-card-top">

            <span class="forecast-horizon">
              DAY ${escapeHTML(item.horizon)}
            </span>

            <span class="forecast-model">
              ${escapeHTML(item.model_used || "MODEL")}
            </span>

          </div>

          <div class="forecast-value">
            ${number(item.predicted_aqi, 1)}
          </div>

          <div
            class="forecast-category"
            style="color:${categoryColor(item.predicted_aqi)}"
          >
            ${escapeHTML(category.label)}
          </div>

          <div class="forecast-alert">
            ${escapeHTML(
              item.alert ||
              "Forecast generated by the production model."
            )}
          </div>

        </article>
      `;
    })
    .join("");

  renderForecastChart(forecast);
}


/* ============================================================
   FORECAST CHART
   ============================================================ */

function renderForecastChart(forecast) {
  const canvas = $("#forecast-chart");

  if (!canvas || typeof Chart === "undefined") return;

  if (state.charts.forecast) {
    state.charts.forecast.destroy();
  }

  const labels = forecast.map(
    (item) => `Day ${item.horizon}`
  );

  const values = forecast.map(
    (item) => Number(item.predicted_aqi)
  );

  state.charts.forecast = new Chart(canvas, {
    type: "line",

    data: {
      labels,

      datasets: [
        {
          label: "Predicted AQI",

          data: values,

          borderWidth: 2,

          borderColor: "#76d7b7",

          backgroundColor:
            "rgba(118,215,183,0.08)",

          pointBackgroundColor: "#76d7b7",

          pointBorderColor: "#07100f",

          pointBorderWidth: 2,

          pointRadius: 5,

          pointHoverRadius: 7,

          tension: 0.35,

          fill: true,
        },
      ],
    },

    options: {
      responsive: true,

      maintainAspectRatio: false,

      plugins: {
        legend: {
          display: false,
        },

        tooltip: {
          backgroundColor: "#0e1c19",

          borderColor:
            "rgba(194,232,217,0.12)",

          borderWidth: 1,

          titleColor: "#eef8f4",

          bodyColor: "#a8bbb5",

          padding: 12,

          displayColors: false,

          callbacks: {
            label: (context) =>
              ` AQI ${number(context.parsed.y, 1)}`,
          },
        },
      },

      scales: {
        x: {
          grid: {
            color:
              "rgba(194,232,217,0.06)",
          },

          ticks: {
            color: "#71857f",

            font: {
              size: 9,
            },
          },

          border: {
            display: false,
          },
        },

        y: {
          grid: {
            color:
              "rgba(194,232,217,0.06)",
          },

          ticks: {
            color: "#71857f",

            font: {
              size: 9,
            },
          },

          border: {
            display: false,
          },
        },
      },
    },
  });
}


/* ============================================================
   POLLUTANTS
   ============================================================ */

function renderPollutants(data) {
  const pollutants = data.pollutants || {};

  const container = $("#pollutant-values");

  if (container) {
    container.innerHTML = Object.entries(pollutants)
      .map(
        ([name, value]) => `
          <div class="pollutant-value">
            <span>${escapeHTML(name.toUpperCase())}</span>
            <strong>${number(value, 2)}</strong>
          </div>
        `
      )
      .join("");
  }

  renderPollutantChart(pollutants);
}


function renderPollutantChart(pollutants) {
  const canvas = $("#pollutant-chart");

  if (!canvas || typeof Chart === "undefined") return;

  if (state.charts.pollutants) {
    state.charts.pollutants.destroy();
  }

  const labels = Object.keys(pollutants);

  const values = labels.map(
    (label) => Number(pollutants[label]) || 0
  );

  state.charts.pollutants = new Chart(canvas, {
    type: "bar",

    data: {
      labels: labels.map(
        (label) => label.toUpperCase()
      ),

      datasets: [
        {
          data: values,

          borderRadius: 6,

          backgroundColor:
            "rgba(118,215,183,0.55)",

          borderColor: "#76d7b7",

          borderWidth: 1,
        },
      ],
    },

    options: {
      responsive: true,

      maintainAspectRatio: false,

      plugins: {
        legend: {
          display: false,
        },

        tooltip: {
          backgroundColor: "#0e1c19",

          borderColor:
            "rgba(194,232,217,0.12)",

          borderWidth: 1,

          displayColors: false,
        },
      },

      scales: {
        x: {
          grid: {
            display: false,
          },

          ticks: {
            color: "#71857f",

            font: {
              size: 8,
            },
          },

          border: {
            display: false,
          },
        },

        y: {
          beginAtZero: true,

          grid: {
            color:
              "rgba(194,232,217,0.06)",
          },

          ticks: {
            color: "#71857f",

            font: {
              size: 8,
            },
          },

          border: {
            display: false,
          },
        },
      },
    },
  });
}


/* ============================================================
   SHAP EXPLANATION
   ============================================================ */

function renderExplanation(data) {
  const explanation =
    data.explanation || [];

  const container =
    $("#explanation-list");

  if (!container) return;

  if (!explanation.length) {
    container.innerHTML = `
      <div class="empty-state">
        Explanation unavailable for this prediction.
      </div>
    `;

    return;
  }

  const sorted = [...explanation]
    .sort(
      (a, b) =>
        Math.abs(Number(b.impact)) -
        Math.abs(Number(a.impact))
    )
    .slice(0, 8);

  const maxImpact = Math.max(
    ...sorted.map(
      (item) => Math.abs(Number(item.impact))
    ),
    1
  );

  container.innerHTML = sorted
    .map((item) => {
      const impact = Number(item.impact) || 0;

      const width =
        (Math.abs(impact) / maxImpact) * 100;

      return `
        <div class="explanation-row">

          <div class="explanation-row-top">

            <span class="explanation-feature">
              ${escapeHTML(item.feature)}
            </span>

            <span class="explanation-value">
              ${impact >= 0 ? "+" : ""}
              ${number(impact, 3)}
            </span>

          </div>

          <div class="explanation-track">
            <div
              class="explanation-bar ${
                impact < 0 ? "negative" : ""
              }"
              style="width:${width}%"
            ></div>
          </div>

        </div>
      `;
    })
    .join("");

  const primary =
    sorted[0]?.feature || "—";

  const primaryElement =
    $("#primary-driver");

  if (primaryElement) {
    primaryElement.textContent = primary;
  }

  const explanationText =
    $("#interpretation-text");

  if (explanationText && sorted.length) {
    const positive = sorted.filter(
      (item) => Number(item.impact) > 0
    );

    const negative = sorted.filter(
      (item) => Number(item.impact) < 0
    );

    explanationText.textContent =
      `The strongest local drivers are ${primary}. ` +
      `${
        positive.length
          ? `${positive.length} leading factors increase the forecast, `
          : ""
      }` +
      `${
        negative.length
          ? `while ${negative.length} leading factors push it downward.`
          : ""
      }`;
  }
}


/* ============================================================
   WHO COMPARISON
   ============================================================ */

function renderWhoComparison(data) {
  const container = $("#who-rows");

  if (!container) return;

  const comparison =
    data.who_comparison || {};

  const entries = Object.entries(comparison);

  if (!entries.length) {
    container.innerHTML = `
      <div class="empty-state">
        WHO comparison unavailable.
      </div>
    `;

    return;
  }

  container.innerHTML = entries
    .map(([pollutant, value]) => {
      let displayValue = value;

      if (
        typeof value === "object" &&
        value !== null
      ) {
        displayValue =
          value.ratio ??
          value.value ??
          value.percent ??
          "—";
      }

      const numeric =
        Number(displayValue);

      const width = Number.isFinite(numeric)
        ? Math.min(Math.max(numeric, 0), 100)
        : 0;

      return `
        <div class="who-row">

          <div class="who-row-top">

            <span class="who-name">
              ${escapeHTML(
                pollutant.toUpperCase()
              )}
            </span>

            <span class="who-value">
              ${Number.isFinite(numeric)
                ? number(numeric, 1)
                : escapeHTML(displayValue)}
            </span>

          </div>

          <div class="who-track">
            <div
              class="who-fill"
              style="width:${width}%"
            ></div>
          </div>

        </div>
      `;
    })
    .join("");
}


/* ============================================================
   TRENDS
   ============================================================ */

async function loadTrends(city) {
  const container = $("#trend-container");

  if (container) {
    showLoading(
      container,
      `Analyzing historical patterns for ${city}…`
    );
  }

  try {
    const data = await api(
      `/api/trends/${encodeURIComponent(city)}`
    );

    state.trends = data;

    renderTrends(data);
  } catch (error) {
    if (container) {
      showError(
        container,
        `Unable to load historical trends: ${error.message}`
      );
    }
  }
}


function renderTrends(data) {
  const container = $("#trend-container");

  if (!container) return;

  container.innerHTML = `
    <div class="trend-grid">

      <article class="trend-card">
        <div class="card-heading">
          <div>
            <span class="eyebrow-small">
              SEASONAL PATTERN
            </span>
            <h3>Monthly AQI</h3>
          </div>
        </div>

        <canvas id="monthly-trend-chart"></canvas>
      </article>

      <article class="trend-card">
        <div class="card-heading">
          <div>
            <span class="eyebrow-small">
              WEEKLY PATTERN
            </span>
            <h3>Day-of-week AQI</h3>
          </div>
        </div>

        <canvas id="weekday-trend-chart"></canvas>
      </article>

    </div>
  `;

  renderTrendCharts(data);
}


function renderTrendCharts(data) {
  if (typeof Chart === "undefined") return;

  const months =
    Object.keys(data.monthly_avg_aqi || {});

  const monthValues =
    Object.values(data.monthly_avg_aqi || {});

  const weekdays =
    Object.keys(data.weekday_avg_aqi || {});

  const weekdayValues =
    Object.values(data.weekday_avg_aqi || {});

  if (state.charts.monthly) {
    state.charts.monthly.destroy();
  }

  if (state.charts.weekday) {
    state.charts.weekday.destroy();
  }

  const monthCanvas =
    $("#monthly-trend-chart");

  const weekdayCanvas =
    $("#weekday-trend-chart");

  if (monthCanvas) {
    state.charts.monthly =
      createTrendChart(
        monthCanvas,
        months.map((m) => `M${m}`),
        monthValues
      );
  }

  if (weekdayCanvas) {
    state.charts.weekday =
      createTrendChart(
        weekdayCanvas,
        weekdays.map((d) => `D${d}`),
        weekdayValues
      );
  }
}


function createTrendChart(canvas, labels, values) {
  return new Chart(canvas, {
    type: "line",

    data: {
      labels,

      datasets: [
        {
          data: values,

          borderColor: "#76d7b7",

          backgroundColor:
            "rgba(118,215,183,0.07)",

          borderWidth: 2,

          pointRadius: 3,

          pointBackgroundColor:
            "#76d7b7",

          tension: 0.35,

          fill: true,
        },
      ],
    },

    options: {
      responsive: true,

      maintainAspectRatio: false,

      plugins: {
        legend: {
          display: false,
        },
      },

      scales: {
        x: {
          grid: {
            display: false,
          },

          ticks: {
            color: "#71857f",
            font: { size: 8 },
          },

          border: {
            display: false,
          },
        },

        y: {
          grid: {
            color:
              "rgba(194,232,217,0.06)",
          },

          ticks: {
            color: "#71857f",
            font: { size: 8 },
          },

          border: {
            display: false,
          },
        },
      },
    },
  });
}


/* ============================================================
   MODEL STATS
   ============================================================ */

async function loadModelStats() {
  const container =
    $("#model-stats-container");

  if (container) {
    showLoading(
      container,
      "Reading production model registry…"
    );
  }

  try {
    const data =
      await api("/api/model-stats");

    state.models = data.models || [];

    renderModelStats(state.models);
  } catch (error) {
    if (container) {
      showError(
        container,
        `Unable to load model registry: ${error.message}`
      );
    }
  }
}


function renderModelStats(models) {
  const container =
    $("#model-stats-container");

  if (!container) return;

  if (!models.length) {
    container.innerHTML = `
      <div class="empty-state">
        No registered models found.
      </div>
    `;

    return;
  }

  const productionModels =
    models.filter(
      (model) =>
        model.status === "production"
    );

  const displayModels =
    productionModels.length
      ? productionModels
      : models;

  container.innerHTML = `
    <div class="model-table">

      <table>

        <thead>
          <tr>
            <th>MODEL</th>
            <th>HORIZON</th>
            <th>VERSION</th>
            <th>RMSE</th>
            <th>MAE</th>
            <th>R²</th>
            <th>STATUS</th>
          </tr>
        </thead>

        <tbody>

          ${displayModels
            .map(
              (model) => `
                <tr>

                  <td>
                    <strong>
                      ${escapeHTML(
                        model.name || "Unknown"
                      )}
                    </strong>
                  </td>

                  <td>
                    ${escapeHTML(
                      model.horizon ?? "—"
                    )}d
                  </td>

                  <td>
                    ${escapeHTML(
                      model.version_name ||
                      model.version ||
                      "—"
                    )}
                  </td>

                  <td>
                    ${number(model.rmse, 3)}
                  </td>

                  <td>
                    ${number(model.mae, 3)}
                  </td>

                  <td>
                    ${number(model.r2, 4)}
                  </td>

                  <td>
                    <span class="production-badge">
                      ${escapeHTML(
                        model.status ||
                        "registered"
                      ).toUpperCase()}
                    </span>
                  </td>

                </tr>
              `
            )
            .join("")}

        </tbody>

      </table>

    </div>
  `;
}


/* ============================================================
   WHAT-IF
   ============================================================ */

async function runWhatIf(form) {
  const result =
    $("#whatif-results");

  const formData =
    new FormData(form);

  const payload = {};

  for (const [key, value] of formData.entries()) {
    payload[key] =
      key === "month" ||
      key === "day" ||
      key === "dayofweek" ||
      key === "hour"
        ? Number.parseInt(value, 10)
        : Number.parseFloat(value);
  }

  if (result) {
    showLoading(
      result,
      "Running what-if simulation…"
    );
  }

  try {
    const data =
      await api(
        "/api/manual-predict",
        {
          method: "POST",
          body: JSON.stringify(payload),
        }
      );

    renderWhatIfResults(data);
  } catch (error) {
    if (result) {
      showError(
        result,
        `Simulation failed: ${error.message}`
      );
    }
  }
}


function renderWhatIfResults(data) {
  const result =
    $("#whatif-results");

  if (!result) return;

  const forecast =
    data.forecast || [];

  if (!forecast.length) {
    result.innerHTML = `
      <div class="empty-state">
        No simulation results.
      </div>
    `;

    return;
  }

  result.innerHTML = forecast
    .map(
      (item) => `
        <div class="scenario-result-card">

          <span class="scenario-horizon">
            DAY ${escapeHTML(item.horizon)}
          </span>

          <div
            class="scenario-aqi"
            style="color:${categoryColor(
              item.predicted_aqi
            )}"
          >
            ${number(item.predicted_aqi, 1)}
          </div>

          <div class="forecast-category">
            ${escapeHTML(
              categoryForAQI(
                item.predicted_aqi
              ).label
            )}
          </div>

        </div>
      `
    )
    .join("");
}


/* ============================================================
   META
   ============================================================ */

function updateDashboardMeta(data) {
  const cityName =
    $("#dashboard-city");

  if (cityName) {
    cityName.textContent =
      data.city || state.city;
  }

  const updated =
    $("#last-updated");

  if (updated) {
    updated.textContent =
      formatDate(data.as_of);
  }
}


function formatDate(value) {
  if (!value) return "—";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return String(value);
  }

  return date.toLocaleString(
    undefined,
    {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }
  );
}


/* ============================================================
   REFRESH
   ============================================================ */

function setupRefresh() {
  const button =
    $("#refresh-btn");

  if (!button) return;

  button.addEventListener(
    "click",
    async () => {
      if (!state.city) return;

      button.classList.add("refreshing");

      await loadForecast(
        state.city
      );

      button.classList.remove(
        "refreshing"
      );
    }
  );
}


/* ============================================================
   CITY SELECTOR
   ============================================================ */

function setupCitySelector() {
  const select =
    $("#city-select");

  if (!select) return;

  select.addEventListener(
    "change",
    () => {
      loadForecast(select.value);
    }
  );
}


/* ============================================================
   WHAT-IF FORM
   ============================================================ */

function setupWhatIf() {
  const form =
    $("#whatif-form");

  if (!form) return;

  form.addEventListener(
    "submit",
    (event) => {
      event.preventDefault();

      runWhatIf(form);
    }
  );
}


/* ============================================================
   EXPLANATION TABS
   ============================================================ */

function setupExplanationTabs() {
  $$(".explain-tab").forEach(
    (tab) => {
      tab.addEventListener(
        "click",
        () => {

          $$(".explain-tab")
            .forEach(
              (item) =>
                item.classList.remove(
                  "active"
                )
            );

          tab.classList.add(
            "active"
          );

          const method =
            tab.dataset.method;

          const badge =
            $("#explanation-method");

          if (badge) {
            badge.textContent =
              method === "lime"
                ? "LIME"
                : "SHAP";
          }
        }
      );
    }
  );
}


/* ============================================================
   CITY NETWORK
   ============================================================ */

function setupCityCards() {
  $$(".city-network-card")
    .forEach((card) => {
      card.addEventListener(
        "click",
        () => {
          const city =
            card.dataset.city;

          const select =
            $("#city-select");

          if (select) {
            select.value = city;
          }

          loadForecast(city);

          window.scrollTo({
            top: 0,
            behavior: "smooth",
          });
        }
      );
    });
}


/* ============================================================
   INITIALIZATION
   ============================================================ */

async function initializeDashboard() {

  setupNavigation();

  setupCitySelector();

  setupRefresh();

  setupWhatIf();

  setupExplanationTabs();

  setupCityCards();

  const select =
    $("#city-select");

  const initialCity =
    select?.value;

  if (initialCity) {
    await loadForecast(
      initialCity
    );
  }

  await loadModelStats();
}


document.addEventListener(
  "DOMContentLoaded",
  initializeDashboard
);