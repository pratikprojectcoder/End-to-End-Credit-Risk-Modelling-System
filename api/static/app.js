const FIELDS = [
  "loan_amnt", "annual_inc", "dti", "fico_range_low",
  "revol_bal", "installment", "delinq_2yrs", "pub_rec",
];

const FORMATTERS = {
  loan_amnt: (v) => `$${Number(v).toLocaleString()}`,
  annual_inc: (v) => `$${Number(v).toLocaleString()}`,
  dti: (v) => `${v}%`,
  fico_range_low: (v) => String(v),
  revol_bal: (v) => `$${Number(v).toLocaleString()}`,
  installment: (v) => `$${Number(v).toLocaleString()}`,
  delinq_2yrs: (v) => String(v),
  pub_rec: (v) => String(v),
};

let ws = null;
let wsReady = false;
let debounceTimer = null;
let lastPrediction = null;
let portfolio = JSON.parse(sessionStorage.getItem("portfolio") || "[]");
let portfolioChart = null;
let importanceChart = null;

function getPayload() {
  const payload = {};
  FIELDS.forEach((f) => {
    payload[f] = Number(document.getElementById(f).value);
  });
  return payload;
}

function updateOutputs() {
  FIELDS.forEach((f) => {
    const el = document.getElementById(f);
    const out = document.getElementById(`${f}_val`);
    if (el && out) out.textContent = FORMATTERS[f](el.value);
  });
}

function setGauge(pct, category) {
  const ring = document.getElementById("gaugeRing");
  const deg = Math.min(pct, 100) * 3.6;
  const color = category === "LOW" ? "#22c55e" : category === "MEDIUM" ? "#f59e0b" : "#ef4444";
  ring.style.background = `conic-gradient(${color} ${deg}deg, #1f2937 ${deg}deg)`;
  document.getElementById("riskPct").textContent = `${pct.toFixed(1)}%`;

  const badge = document.getElementById("riskBadge");
  badge.textContent = `${category} RISK`;
  badge.className = `risk-badge ${category.toLowerCase()}`;
}

function applyPrediction(data) {
  lastPrediction = data;
  setGauge(data.default_probability_pct, data.risk_category);
  document.getElementById("recommendation").textContent = data.recommendation;
  document.getElementById("latency").textContent = `${data.latency_ms} ms`;
}

function connectWebSocket() {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${protocol}//${location.host}/ws/predict`);

  ws.onopen = () => {
    wsReady = true;
    document.getElementById("transport").textContent = "WebSocket";
    requestPrediction();
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "prediction") applyPrediction(msg.data);
    if (msg.type === "error") console.error(msg.message);
  };

  ws.onclose = () => {
    wsReady = false;
    document.getElementById("transport").textContent = "REST fallback";
    setTimeout(connectWebSocket, 3000);
  };
}

async function predictRest() {
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(getPayload()),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function requestPrediction() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(async () => {
    try {
      if (wsReady && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(getPayload()));
      } else {
        const data = await predictRest();
        applyPrediction(data);
      }
    } catch (err) {
      console.error("Prediction failed:", err);
    }
  }, 120);
}

async function loadModelInfo() {
  try {
    const res = await fetch("/api/model/info");
    if (!res.ok) return;
    const info = await res.json();

    document.getElementById("kpiAccuracy").textContent =
      `${(info.metrics.accuracy * 100).toFixed(0)}%`;
    document.getElementById("kpiAuc").textContent = info.metrics.roc_auc.toFixed(2);
    document.getElementById("kpiPrecision").textContent =
      `${(info.metrics.precision * 100).toFixed(0)}%`;
    document.getElementById("kpiServed").textContent = info.predictions_served;

    const dl = document.getElementById("modelDetails");
    dl.innerHTML = `
      <dt>Model Type</dt><dd>${info.model_type}</dd>
      <dt>Features</dt><dd>${info.feature_count}</dd>
      <dt>Risk Thresholds</dt><dd>Low &lt; ${info.risk_thresholds.low * 100}%, Medium &lt; ${info.risk_thresholds.medium * 100}%</dd>
      <dt>Feature Columns</dt><dd>${info.feature_columns.join(", ")}</dd>
    `;

    renderImportanceChart(info.feature_importance);
  } catch (err) {
    console.error("Failed to load model info:", err);
  }
}

async function checkHealth() {
  const pill = document.getElementById("apiStatus");
  try {
    const res = await fetch("/health");
    const data = await res.json();
    if (data.status === "ok") {
      pill.className = "status-pill online";
      pill.querySelector("span:last-child").textContent = "API Online";
    } else {
      pill.className = "status-pill offline";
      pill.querySelector("span:last-child").textContent = "Model Loading…";
    }
  } catch {
    pill.className = "status-pill offline";
    pill.querySelector("span:last-child").textContent = "API Offline";
  }
}

function renderImportanceChart(items) {
  const ctx = document.getElementById("importanceChart");
  if (!ctx || !items.length) return;

  if (importanceChart) importanceChart.destroy();

  importanceChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: items.map((i) => i.feature),
      datasets: [{
        label: "Importance",
        data: items.map((i) => i.importance),
        backgroundColor: "#3b82f6",
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: "#374151" }, ticks: { color: "#9ca3af" } },
        y: { grid: { display: false }, ticks: { color: "#f3f4f6" } },
      },
    },
  });
}

function updatePortfolioUI() {
  const summary = { total: portfolio.length, low: 0, medium: 0, high: 0 };
  portfolio.forEach((p) => {
    const key = p.risk_category.toLowerCase();
    if (key === "low") summary.low++;
    else if (key === "medium") summary.medium++;
    else summary.high++;
  });

  document.getElementById("pfTotal").textContent = summary.total;
  document.getElementById("pfLow").textContent = summary.low;
  document.getElementById("pfMedium").textContent = summary.medium;
  document.getElementById("pfHigh").textContent = summary.high;

  const ctx = document.getElementById("portfolioChart");
  if (!ctx) return;

  if (portfolioChart) portfolioChart.destroy();

  portfolioChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Low Risk", "Medium Risk", "High Risk"],
      datasets: [{
        data: [summary.low, summary.medium, summary.high],
        backgroundColor: ["#22c55e", "#f59e0b", "#ef4444"],
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: "#f3f4f6" } } },
    },
  });
}

function savePortfolio() {
  if (!lastPrediction) return;
  portfolio.push({ ...lastPrediction, ...getPayload(), ts: Date.now() });
  sessionStorage.setItem("portfolio", JSON.stringify(portfolio));
  updatePortfolioUI();
}

async function runBatch() {
  let apps;
  const fileInput = document.getElementById("csvFile");
  const jsonRaw = document.getElementById("batchJson").value.trim();

  if (fileInput.files.length) {
    const text = await fileInput.files[0].text();
    apps = parseCsv(text);
  } else if (jsonRaw) {
    apps = JSON.parse(jsonRaw);
  } else {
    alert("Provide a CSV file or JSON array.");
    return;
  }

  const res = await fetch("/api/predict/batch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ applications: apps }),
  });

  if (!res.ok) {
    alert(`Batch failed: ${await res.text()}`);
    return;
  }

  const data = await res.json();
  const container = document.getElementById("batchResults");
  container.classList.remove("hidden");

  const s = data.portfolio_summary;
  document.getElementById("batchSummary").innerHTML = `
    <span>Scored: <strong>${s.total}</strong></span>
    <span>Low: <strong>${s.low_risk}</strong></span>
    <span>Medium: <strong>${s.medium_risk}</strong></span>
    <span>High: <strong>${s.high_risk}</strong></span>
    <span>Avg prob: <strong>${(s.average_probability * 100).toFixed(1)}%</strong></span>
    <span>Latency: <strong>${data.total_latency_ms} ms</strong></span>
  `;

  const tbody = document.querySelector("#batchTable tbody");
  tbody.innerHTML = data.predictions.map((p) => `
    <tr>
      <td>${p.index + 1}</td>
      <td>${p.default_probability_pct}%</td>
      <td>${p.risk_category}</td>
      <td>${p.recommendation}</td>
    </tr>
  `).join("");

  portfolio.push(...data.predictions.map((p) => ({ ...p, batch: true })));
  sessionStorage.setItem("portfolio", JSON.stringify(portfolio));
  updatePortfolioUI();
  loadModelInfo();
}

function parseCsv(text) {
  const lines = text.trim().split("\n");
  const headers = lines[0].split(",").map((h) => h.trim());
  return lines.slice(1).map((line) => {
    const vals = line.split(",");
    const row = {};
    headers.forEach((h, i) => { row[h] = Number(vals[i]); });
    return row;
  });
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(`panel-${tab.dataset.tab}`).classList.add("active");
  });
});

FIELDS.forEach((f) => {
  document.getElementById(f).addEventListener("input", () => {
    updateOutputs();
    requestPrediction();
  });
});

document.getElementById("saveToPortfolio").addEventListener("click", savePortfolio);
document.getElementById("clearPortfolio").addEventListener("click", () => {
  portfolio = [];
  sessionStorage.removeItem("portfolio");
  updatePortfolioUI();
});
document.getElementById("runBatch").addEventListener("click", runBatch);

updateOutputs();
checkHealth();
loadModelInfo();
connectWebSocket();
updatePortfolioUI();

setInterval(() => {
  checkHealth();
  loadModelInfo();
}, 15000);
