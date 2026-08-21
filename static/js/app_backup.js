const citySelect = document.getElementById('city-select');
const forecastBtn = document.getElementById('forecast-btn');
const currentConditionsEl = document.getElementById('current-conditions');
const resultArea = document.getElementById('result-area');
const pollutantPanel = document.getElementById('pollutant-panel');

let chartInstance = null;
let pollutantChartInstance = null;
let monthlyTrendChart = null;
let weekdayTrendChart = null;

forecastBtn.addEventListener('click', () => {
  const city = citySelect.value;
  fetchForecast(city);
  fetchTrends(city);
});

async function fetchForecast(city) {
  setLoading();

  try {
    const res = await fetch(`/predict/${encodeURIComponent(city)}`);
    const data = await res.json();

    if (!res.ok || data.error) {
      showError(data.error || 'Something went wrong fetching the forecast.');
      return;
    }

    renderCurrentConditions(data);
    renderForecast(data);
    renderPollutantPanel(data);
  } catch (err) {
    showError(`Could not reach the prediction API. Is app.py running? (${err.message})`);
  }
}

function setLoading() {
  forecastBtn.disabled = true;
  forecastBtn.innerHTML = '<span class="spinner"></span>Loading...';
  currentConditionsEl.style.display = 'none';
  resultArea.innerHTML = '<div class="state-message">Fetching the latest forecast…</div>';
}

function showError(message) {
  forecastBtn.disabled = false;
  forecastBtn.textContent = 'Get 3-day forecast';
  currentConditionsEl.style.display = 'none';
  resultArea.innerHTML = `<div class="state-message error">${escapeHtml(message)}</div>`;
}

function renderPollutantPanel(data) {
  const pollutants = data.pollutants;
  const whoComparison = data.who_comparison;

  if (!pollutants || !whoComparison || !whoComparison.length) {
    pollutantPanel.innerHTML = '';
    return;
  }

  pollutantPanel.innerHTML = `
    <div class="panel">
      <h2>Pollutant breakdown</h2>
      <canvas id="pollutant-chart" height="220"></canvas>
    </div>
    <div class="panel">
      <h2>Compared to WHO Air Quality Guidelines (24-hr)</h2>
      <div id="who-rows"></div>
    </div>
  `;

  const whoRows = document.getElementById('who-rows');
  whoRows.innerHTML = whoComparison.map(item => {
    const pct = Math.min(item.ratio * 100, 200); // cap the bar visually at 2x
    const barClass = item.exceeds ? 'impact-positive' : 'impact-negative';
    return `<div class="shap-item">
      <div class="feature-name">${escapeHtml(item.label)}</div>
      <div class="impact-bar-wrap"><div class="impact-bar ${barClass}" style="width:${pct / 2}%"></div></div>
      <div class="impact-value">${item.ratio}&times; ${item.exceeds ? '(exceeds)' : ''}</div>
    </div>`;
  }).join('');

  const ctx = document.getElementById('pollutant-chart');
  if (pollutantChartInstance) pollutantChartInstance.destroy();

  const labels = ['PM2.5', 'PM10', 'NO\u2082', 'SO\u2082', 'O\u2083', 'CO'];
  const values = [pollutants.pm25, pollutants.pm10, pollutants.no2, pollutants.so2, pollutants.o3, pollutants.co]
    .map(v => v != null ? v : 0);

  pollutantChartInstance = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: ['#5B7FE0', '#7FB0E8', '#A8D0E6', '#F0997B', '#FAC775', '#ED93B1'],
        borderWidth: 0,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: 12 } } } },
    },
  });
}

async function fetchTrends(city) {
  const trendsResult = document.getElementById('trends-result');
  trendsResult.innerHTML = '<div class="state-message">Loading historical trends…</div>';

  try {
    const res = await fetch(`/api/trends/${encodeURIComponent(city)}`);
    const data = await res.json();

    if (!res.ok || data.error) {
      trendsResult.innerHTML = `<div class="state-message error">${escapeHtml(data.error || 'Could not load trends.')}</div>`;
      return;
    }

    trendsResult.innerHTML = `
      <div class="panel-row">
        <div class="panel">
          <h2>Average AQI by month</h2>
          <canvas id="monthly-trend-chart" height="80"></canvas>
        </div>
        <div class="panel">
          <h2>Average AQI by day of week</h2>
          <canvas id="weekday-trend-chart" height="80"></canvas>
        </div>
      </div>
    `;

    const monthLabels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    const monthlyValues = monthLabels.map((_, i) => data.monthly_avg_aqi[i + 1] ?? null);

    const weekdayLabels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
    const weekdayValues = weekdayLabels.map((_, i) => data.weekday_avg_aqi[i] ?? null);

    const monthlyCtx = document.getElementById('monthly-trend-chart');
    if (monthlyTrendChart) monthlyTrendChart.destroy();
    monthlyTrendChart = new Chart(monthlyCtx, {
      type: 'bar',
      data: { labels: monthLabels, datasets: [{ data: monthlyValues, backgroundColor: '#5B7FE0', borderRadius: 6 }] },
      options: { responsive: true, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, grid: { color: '#E7EBF3' } }, x: { grid: { display: false } } } },
    });

    const weekdayCtx = document.getElementById('weekday-trend-chart');
    if (weekdayTrendChart) weekdayTrendChart.destroy();
    weekdayTrendChart = new Chart(weekdayCtx, {
      type: 'bar',
      data: { labels: weekdayLabels, datasets: [{ data: weekdayValues, backgroundColor: '#F0997B', borderRadius: 6 }] },
      options: { responsive: true, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, grid: { color: '#E7EBF3' } }, x: { grid: { display: false } } } },
    });
  } catch (err) {
    trendsResult.innerHTML = `<div class="state-message error">Could not reach the trends API. (${err.message})</div>`;
  }
}

async function fetchModelStats() {
  const el = document.getElementById('model-stats-result');
  try {
    const res = await fetch('/api/model-stats');
    const data = await res.json();

    if (!res.ok || data.error) {
      el.innerHTML = `<div class="state-message error">${escapeHtml(data.error || 'Could not load model stats.')}</div>`;
      return;
    }

    if (!data.models || !data.models.length) {
      el.innerHTML = '<div class="state-message">No models registered yet. Run train_models.py.</div>';
      return;
    }

    const sorted = [...data.models].sort((a, b) => a.horizon - b.horizon || a.name.localeCompare(b.name));
    const rows = sorted.map(m => `
      <div class="shap-item">
        <div class="feature-name">${escapeHtml(m.name)} (t+${m.horizon})</div>
        <div class="impact-value" style="min-width:auto;">RMSE ${m.rmse != null ? m.rmse.toFixed(2) : '—'} &middot; MAE ${m.mae != null ? m.mae.toFixed(2) : '—'} &middot; R&sup2; ${m.r2 != null ? m.r2.toFixed(3) : '—'}</div>
      </div>
    `).join('');

    el.innerHTML = `<div class="panel">${rows}</div>`;
  } catch (err) {
    el.innerHTML = `<div class="state-message error">Could not reach the model stats API. (${err.message})</div>`;
  }
}

function renderCurrentConditions(data) {
  const cc = data.current_conditions;
  if (!cc) {
    currentConditionsEl.style.display = 'none';
    return;
  }

  currentConditionsEl.style.display = 'flex';
  currentConditionsEl.innerHTML = `
    <div class="icon">${cc.icon || '🌡️'}</div>
    <div class="details">
      <div class="label">${escapeHtml(data.city)} · as of ${formatTime(data.as_of)}</div>
      <div class="value">${escapeHtml(cc.description || 'Current conditions')}</div>
    </div>
    <div class="stats">
      <div class="stat"><div class="num">${cc.temperature != null ? Math.round(cc.temperature) + '°C' : '—'}</div><div class="lbl">Temp</div></div>
      <div class="stat"><div class="num">${cc.humidity != null ? Math.round(cc.humidity) + '%' : '—'}</div><div class="lbl">Humidity</div></div>
      <div class="stat"><div class="num">${cc.wind_speed != null ? Math.round(cc.wind_speed) + ' km/h' : '—'}</div><div class="lbl">Wind</div></div>
      <div class="stat"><div class="num">${cc.current_aqi != null ? Math.round(cc.current_aqi) : '—'}</div><div class="lbl">Current AQI</div></div>
    </div>
  `;
}

function renderForecast(data) {
  forecastBtn.disabled = false;
  forecastBtn.textContent = 'Get 3-day forecast';

  const cards = data.forecast.map(entry => {
    if (entry.error) {
      return `<div class="forecast-card">
        <div class="day-label">Day +${entry.horizon}</div>
        <div class="state-message error" style="padding: 12px 0;">${escapeHtml(entry.error)}</div>
      </div>`;
    }

    return `<div class="forecast-card">
      <div class="day-label">Day +${entry.horizon}</div>
      <div class="aqi-value">${entry.predicted_aqi}</div>
      <div class="model-tag">via ${escapeHtml(entry.model_used)}</div>
      <span class="alert-badge badge-${entry.alert.level}">${escapeHtml(entry.alert.message)}</span>
    </div>`;
  }).join('');

  let chartPanel = '';
  const validEntries = data.forecast.filter(e => !e.error);
  if (validEntries.length) {
    chartPanel = `
      <div class="panel">
        <h2>Forecast trend</h2>
        <canvas id="forecast-chart" height="90"></canvas>
      </div>
    `;
  }

  let shapPanel = '';
  if (data.explanation && data.explanation.length) {
    const maxAbs = Math.max(...data.explanation.map(e => Math.abs(e.impact)));
    const rows = data.explanation.map(item => {
      const pct = maxAbs > 0 ? Math.abs(item.impact) / maxAbs * 100 : 0;
      const dir = item.impact >= 0 ? 'impact-positive' : 'impact-negative';
      const verb = item.impact >= 0 ? 'increases' : 'decreases';
      return `<div class="shap-item">
        <div class="feature-name">${escapeHtml(item.feature)}</div>
        <div class="impact-bar-wrap"><div class="impact-bar ${dir}" style="width:${pct}%"></div></div>
        <div class="impact-value">${verb}</div>
      </div>`;
    }).join('');

    shapPanel = `
      <div class="panel">
        <h2>Why this prediction? (SHAP, day +1)</h2>
        ${rows}
      </div>
    `;
  }

  resultArea.innerHTML = `
    <div class="forecast-grid">${cards}</div>
    ${chartPanel}
    ${shapPanel}
  `;

  if (validEntries.length) {
    drawChart(validEntries);
  }
}

function drawChart(entries) {
  const ctx = document.getElementById('forecast-chart');
  if (!ctx) return;

  if (chartInstance) {
    chartInstance.destroy();
  }

  chartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: entries.map(e => `Day +${e.horizon}`),
      datasets: [{
        label: 'Predicted AQI',
        data: entries.map(e => e.predicted_aqi),
        borderColor: '#5B7FE0',
        backgroundColor: 'rgba(91, 127, 224, 0.12)',
        borderWidth: 3,
        tension: 0.35,
        fill: true,
        pointRadius: 5,
        pointBackgroundColor: '#5B7FE0',
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, grid: { color: '#E7EBF3' } },
        x: { grid: { display: false } },
      }
    }
  });
}

function formatTime(iso) {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// Auto-load the first city on page load for a nicer first impression.
window.addEventListener('DOMContentLoaded', () => {
  if (citySelect.value) {
    fetchForecast(citySelect.value);
    fetchTrends(citySelect.value);
  }
  fetchModelStats();
});

// ---------------- What-If Predictor ----------------

const whatifForm = document.getElementById('whatif-form');
const whatifBtn = document.getElementById('whatif-btn');
const whatifResult = document.getElementById('whatif-result');

whatifForm.addEventListener('submit', async (e) => {
  e.preventDefault();

  const formData = new FormData(whatifForm);
  const payload = {};
  for (const [key, value] of formData.entries()) {
    payload[key] = parseFloat(value);
  }

  whatifBtn.disabled = true;
  whatifBtn.innerHTML = '<span class="spinner"></span>Simulating...';
  whatifResult.innerHTML = '<div class="state-message">Running the models…</div>';

  try {
    const res = await fetch('/api/manual-predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok || data.error) {
      whatifResult.innerHTML = `<div class="state-message error">${escapeHtml(data.error || 'Simulation failed.')}</div>`;
      return;
    }

    const cards = data.forecast.map(entry => {
      if (entry.error) {
        return `<div class="forecast-card">
          <div class="day-label">Day +${entry.horizon}</div>
          <div class="state-message error" style="padding: 12px 0;">${escapeHtml(entry.error)}</div>
        </div>`;
      }
      return `<div class="forecast-card">
        <div class="day-label">Day +${entry.horizon}</div>
        <div class="aqi-value">${entry.predicted_aqi}</div>
        <div class="model-tag">via ${escapeHtml(entry.model_used)}</div>
        <span class="alert-badge badge-${entry.alert.level}">${escapeHtml(entry.alert.message)}</span>
      </div>`;
    }).join('');

    whatifResult.innerHTML = `<div class="forecast-grid">${cards}</div>`;
  } catch (err) {
    whatifResult.innerHTML = `<div class="state-message error">Could not reach the prediction API. (${err.message})</div>`;
  } finally {
    whatifBtn.disabled = false;
    whatifBtn.textContent = 'Simulate forecast';
  }
});
