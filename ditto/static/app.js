"use strict";

// ── State ──────────────────────────────────────────────────────────────────
let sessionId = null;
let versions  = [];   // [{filename, version, variables:[{name,tag,expr,samples}]}]

// ── KDE ────────────────────────────────────────────────────────────────────
function silvermanBandwidth(samples) {
  const n   = samples.length;
  const mean = samples.reduce((a, b) => a + b, 0) / n;
  const variance = samples.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1);
  const std = Math.sqrt(variance);
  return 1.06 * std * Math.pow(n, -0.2);
}

function gaussianKernel(u) {
  return Math.exp(-0.5 * u * u) / Math.sqrt(2 * Math.PI);
}

function kde(samples, nPoints = 120) {
  if (!samples || samples.length === 0) return { xs: [], ys: [] };
  const h    = silvermanBandwidth(samples);
  const min  = Math.min(...samples);
  const max  = Math.max(...samples);
  const pad  = (max - min) * 0.15 || 1;
  const lo   = min - pad;
  const hi   = max + pad;
  const step = (hi - lo) / (nPoints - 1);

  const xs = [], ys = [];
  for (let i = 0; i < nPoints; i++) {
    const x   = lo + i * step;
    const den = samples.reduce((acc, xi) => acc + gaussianKernel((x - xi) / h), 0);
    xs.push(x.toFixed(3));
    ys.push(den / (samples.length * h));
  }
  return { xs, ys };
}

// ── Chart helpers ──────────────────────────────────────────────────────────
const CHART_COLOR = "rgba(91, 110, 245, 0.85)";
const CHART_FILL  = "rgba(91, 110, 245, 0.12)";

function makeChart(canvas, samples, label) {
  const { xs, ys } = kde(samples);
  return new Chart(canvas, {
    type: "line",
    data: {
      labels: xs,
      datasets: [{
        label,
        data: ys,
        borderColor: CHART_COLOR,
        backgroundColor: CHART_FILL,
        borderWidth: 1.8,
        pointRadius: 0,
        fill: true,
        tension: 0.4,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          mode: "index",
          intersect: false,
          callbacks: {
            label: ctx => `density: ${ctx.parsed.y.toFixed(4)}`,
          },
        },
      },
      scales: {
        x: {
          ticks: {
            color: "#6b7094",
            maxTicksLimit: 5,
            maxRotation: 0,
          },
          grid: { color: "#2e3248" },
        },
        y: {
          ticks: { color: "#6b7094", maxTicksLimit: 4 },
          grid:  { color: "#2e3248" },
          title: { display: false },
        },
      },
    },
  });
}

// ── DOM refs ────────────────────────────────────────────────────────────────
const dropZone   = document.getElementById("drop-zone");
const fileInput  = document.getElementById("file-input");
const statusEl   = document.getElementById("status");
const notesBar   = document.getElementById("notes-bar");
const grid       = document.getElementById("grid");

// ── Upload ─────────────────────────────────────────────────────────────────
dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("dragover"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
});
dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) uploadFile(fileInput.files[0]);
  fileInput.value = "";
});

async function uploadFile(file) {
  setStatus("loading", `Uploading ${file.name} and running inference…`);

  const body = new FormData();
  body.append("file", file);

  const headers = {};
  if (sessionId) headers["X-Session-Id"] = sessionId;

  try {
    const res = await fetch("/upload", { method: "POST", body, credentials: "include" });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      setStatus("error", `Error: ${err.detail || res.statusText}`);
      return;
    }
    const data = await res.json();
    sessionId = data.session_id;

    // Keep only last two
    versions.push(data);
    if (versions.length > 2) versions = versions.slice(-2);

    setStatus("", "");
    switchToCompact();
    render();
  } catch (e) {
    setStatus("error", `Network error: ${e.message}`);
  }
}

// ── UI helpers ──────────────────────────────────────────────────────────────
function setStatus(type, msg) {
  statusEl.className = type;
  statusEl.innerHTML = type === "loading"
    ? `<span class="spinner"></span>${msg}`
    : msg;
}

function switchToCompact() {
  if (dropZone.classList.contains("compact")) return;
  dropZone.classList.remove("large");
  dropZone.classList.add("compact");
  dropZone.innerHTML = `
    <span class="drop-icon">⬆</span>
    <span class="drop-title">Upload another version</span>
    <span class="drop-sub">drag & drop or click</span>
  `;
}

// ── Render ──────────────────────────────────────────────────────────────────
function render() {
  grid.innerHTML = "";
  notesBar.innerHTML = "";
  notesBar.classList.remove("visible");

  if (versions.length === 0) return;

  if (versions.length === 1) {
    renderSingle(versions[0]);
  } else {
    renderCompare(versions[0], versions[1]);
  }
}

function renderSingle(ver) {
  for (const v of ver.variables) {
    const card = makeCardShell(v, null);
    const body = card.querySelector(".card-body");
    body.classList.add("single");
    const canvas = document.createElement("canvas");
    body.appendChild(canvas);
    grid.appendChild(card);
    makeChart(canvas, v.samples, ver.filename);
  }
}

function renderCompare(verA, verB) {
  // Build notes
  const namesA = new Set(verA.variables.map(v => v.name));
  const namesB = new Set(verB.variables.map(v => v.name));
  const added   = [...namesB].filter(n => !namesA.has(n));
  const removed = [...namesA].filter(n => !namesB.has(n));
  const changed = [];

  const mapB = Object.fromEntries(verB.variables.map(v => [v.name, v]));
  for (const v of verA.variables) {
    if (mapB[v.name] && mapB[v.name].expr !== v.expr) {
      changed.push(v.name);
    }
  }

  if (added.length || removed.length || changed.length) {
    const parts = [];
    if (changed.length) parts.push(`<strong>${changed.length} expression${changed.length > 1 ? "s" : ""} changed</strong>: ${changed.join(", ")}`);
    if (added.length)   parts.push(`<strong>added</strong>: ${added.join(", ")}`);
    if (removed.length) parts.push(`<strong>removed</strong>: ${removed.join(", ")}`);
    notesBar.innerHTML = parts.join(" · ");
    notesBar.classList.add("visible");
  }

  // Union of variable names — show all
  const allNames = [...new Set([...namesA, ...namesB])];
  const mapA = Object.fromEntries(verA.variables.map(v => [v.name, v]));

  for (const name of allNames) {
    const vA = mapA[name] || null;
    const vB = mapB[name] || null;
    const representative = vA || vB;

    const exprDiff = vA && vB && vA.expr !== vB.expr ? { from: vA.expr, to: vB.expr } : null;
    const card = makeCardShell(representative, exprDiff);
    const body = card.querySelector(".card-body");
    body.classList.add("compare");

    // Left column — version A
    const colA = makeChartCol(verA.filename);
    body.appendChild(colA);

    body.appendChild(Object.assign(document.createElement("div"), { className: "chart-divider" }));

    // Right column — version B
    const colB = makeChartCol(verB.filename);
    body.appendChild(colB);

    grid.appendChild(card);

    // Compute shared x-range for honest comparison
    const samplesA = vA ? vA.samples : [];
    const samplesB = vB ? vB.samples : [];

    makeChart(colA.querySelector("canvas"), samplesA, verA.filename);
    makeChart(colB.querySelector("canvas"), samplesB, verB.filename);
  }
}

function makeCardShell(variable, exprDiff) {
  const card = document.createElement("div");
  card.className = "var-card";

  const header = document.createElement("div");
  header.className = "card-header";

  const nameEl = document.createElement("span");
  nameEl.className = "var-name";
  nameEl.textContent = variable.name;

  const badge = document.createElement("span");
  badge.className = `tag-badge tag-${variable.tag}`;
  badge.textContent = variable.tag;

  header.appendChild(nameEl);
  header.appendChild(badge);

  if (exprDiff) {
    const diff = document.createElement("span");
    diff.className = "diff-badge";
    diff.title = `${exprDiff.from} → ${exprDiff.to}`;
    diff.innerHTML = `<code>${esc(exprDiff.from)}</code><span class="arrow">→</span><code>${esc(exprDiff.to)}</code>`;
    header.appendChild(diff);
  }

  const body = document.createElement("div");
  body.className = "card-body";

  card.appendChild(header);
  card.appendChild(body);
  return card;
}

function makeChartCol(filename) {
  const col = document.createElement("div");
  col.className = "chart-col";
  const label = document.createElement("div");
  label.className = "chart-label";
  label.textContent = filename;
  label.title = filename;
  const canvas = document.createElement("canvas");
  col.appendChild(label);
  col.appendChild(canvas);
  return col;
}

function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
