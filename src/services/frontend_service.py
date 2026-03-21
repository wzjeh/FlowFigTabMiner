import os
import hmac
import hashlib
import uuid
import io
import csv
import logging
import httpx
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, Cookie, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from google.cloud import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# ── Config ────────────────────────────────────────────────────────────────────
PASSWORD        = os.environ.get("FRONTEND_PASSWORD", "nagaki2024")
SECRET_KEY      = os.environ.get("FRONTEND_SECRET",   "flowfigtabminer-secret-key")
GCS_DATA_BUCKET = "flowfigtabminer-data"
FIGURE_URL      = os.environ.get("FIGURE_SERVICE_URL",
                    "https://figure-service-h6pakhgb2a-uc.a.run.app")
TABLE_URL       = os.environ.get("TABLE_SERVICE_URL",
                    "https://table-service-903119038444.us-central1.run.app")

EXAMPLES_DIR = Path(__file__).parent / "examples"
app.mount("/examples", StaticFiles(directory=str(EXAMPLES_DIR)), name="examples")

# ── Auth helpers ──────────────────────────────────────────────────────────────
def _make_token(pw: str) -> str:
    return hmac.new(SECRET_KEY.encode(), pw.encode(), hashlib.sha256).hexdigest()

def _authenticated(token: str | None) -> bool:
    if not token:
        return False
    return hmac.compare_digest(token, _make_token(PASSWORD))

# ── GCS helpers ───────────────────────────────────────────────────────────────
def _gcs_upload_bytes(data: bytes, gcs_path: str, content_type: str = "image/png") -> str:
    client = storage.Client()
    bucket = client.bucket(GCS_DATA_BUCKET)
    blob   = bucket.blob(gcs_path)
    blob.upload_from_string(data, content_type=content_type)
    return f"gs://{GCS_DATA_BUCKET}/{gcs_path}"

def _gcs_download_bytes(uri: str) -> bytes:
    """Download bytes from a gs:// URI."""
    path   = uri.removeprefix(f"gs://{GCS_DATA_BUCKET}/")
    client = storage.Client()
    bucket = client.bucket(GCS_DATA_BUCKET)
    return bucket.blob(path).download_as_bytes()

# ── HTML page ─────────────────────────────────────────────────────────────────
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FlowFigTabMiner — Nagaki Lab</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    background: #f5f6f8;
    color: #1e293b;
    font-size: 14px;
    line-height: 1.6;
  }

  /* ── Header ── */
  header {
    background: #1e293b;
    color: #fff;
    padding: 18px 40px;
    display: flex;
    align-items: baseline;
    gap: 20px;
    border-bottom: 3px solid #2563eb;
  }
  header h1 { font-size: 20px; font-weight: 700; letter-spacing: 0.5px; }
  header span { font-size: 12px; color: #94a3b8; }

  /* ── Layout ── */
  main { max-width: 860px; margin: 36px auto; padding: 0 20px 60px; }

  /* ── Card ── */
  .card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 28px 32px;
    margin-bottom: 24px;
  }
  .card h2 {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #64748b;
    margin-bottom: 18px;
    padding-bottom: 10px;
    border-bottom: 1px solid #e2e8f0;
  }

  /* ── Mode tabs ── */
  .tabs { display: flex; gap: 0; margin-bottom: 22px; }
  .tab {
    flex: 1;
    padding: 9px 0;
    text-align: center;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    border: 1px solid #cbd5e1;
    background: #f8fafc;
    color: #64748b;
    transition: all .15s;
  }
  .tab:first-child { border-radius: 4px 0 0 4px; }
  .tab:last-child  { border-radius: 0 4px 4px 0; border-left: none; }
  .tab.active { background: #2563eb; color: #fff; border-color: #2563eb; }

  /* ── Upload zone ── */
  .upload-zone {
    border: 2px dashed #cbd5e1;
    border-radius: 6px;
    padding: 36px 20px;
    text-align: center;
    cursor: pointer;
    transition: border-color .15s, background .15s;
    position: relative;
    background: #fafafa;
  }
  .upload-zone:hover, .upload-zone.dragover {
    border-color: #2563eb;
    background: #eff6ff;
  }
  .upload-zone input[type=file] {
    position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%;
  }
  .upload-zone .icon { font-size: 28px; margin-bottom: 8px; }
  .upload-zone p   { color: #64748b; font-size: 13px; }
  .upload-zone strong { color: #1e293b; }
  #preview-wrap { margin-top: 14px; display: none; text-align: center; }
  #preview-wrap img { max-height: 220px; max-width: 100%; border: 1px solid #e2e8f0; border-radius: 4px; }
  #file-name { font-size: 12px; color: #64748b; margin-top: 6px; }

  /* ── Examples ── */
  .examples-row { display: flex; gap: 12px; margin-top: 16px; }
  .ex-btn {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 14px;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    background: #f8fafc;
    font-size: 12px;
    color: #475569;
    cursor: pointer;
    transition: all .15s;
    text-decoration: none;
  }
  .ex-btn:hover { border-color: #2563eb; color: #2563eb; background: #eff6ff; }
  .ex-btn img { width: 44px; height: 36px; object-fit: cover; border-radius: 2px; border: 1px solid #e2e8f0; }

  /* ── Instructions ── */
  .instructions {
    background: #f8fafc;
    border-left: 3px solid #2563eb;
    border-radius: 0 4px 4px 0;
    padding: 14px 18px;
    margin-top: 20px;
  }
  .instructions p { font-size: 12px; color: #475569; margin-bottom: 4px; }
  .instructions strong { color: #1e293b; }
  .instructions ul { padding-left: 16px; margin-top: 6px; }
  .instructions li { font-size: 12px; color: #475569; margin-bottom: 3px; }

  /* ── Submit button ── */
  #submit-btn {
    margin-top: 22px;
    width: 100%;
    padding: 11px;
    background: #2563eb;
    color: #fff;
    border: none;
    border-radius: 4px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    letter-spacing: 0.3px;
    transition: background .15s;
  }
  #submit-btn:hover:not(:disabled) { background: #1d4ed8; }
  #submit-btn:disabled { background: #93c5fd; cursor: not-allowed; }

  /* ── Status ── */
  #status-wrap { display: none; margin-top: 18px; }
  .status-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 12px;
    border-radius: 99px;
    font-size: 12px;
    font-weight: 600;
  }
  .status-badge.running  { background: #dbeafe; color: #1d4ed8; }
  .status-badge.success  { background: #dcfce7; color: #15803d; }
  .status-badge.filtered { background: #fef9c3; color: #854d0e; }
  .status-badge.error    { background: #fee2e2; color: #b91c1c; }
  .spinner {
    width: 12px; height: 12px;
    border: 2px solid currentColor;
    border-top-color: transparent;
    border-radius: 50%;
    animation: spin .7s linear infinite;
    flex-shrink: 0;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  #elapsed { font-size: 11px; color: #64748b; margin-top: 6px; }

  /* ── Results table ── */
  #results-wrap { display: none; }
  #results-wrap h2 {
    font-size: 13px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.8px; color: #64748b;
    margin-bottom: 14px; padding-bottom: 10px;
    border-bottom: 1px solid #e2e8f0;
  }
  .table-scroll { overflow-x: auto; }
  table.data-table {
    border-collapse: collapse;
    width: 100%;
    font-size: 12px;
  }
  table.data-table th {
    background: #f1f5f9;
    color: #475569;
    font-weight: 700;
    padding: 8px 12px;
    border: 1px solid #e2e8f0;
    text-align: left;
    white-space: nowrap;
  }
  table.data-table td {
    padding: 6px 12px;
    border: 1px solid #e2e8f0;
    white-space: nowrap;
  }
  table.data-table tr:nth-child(even) td { background: #f8fafc; }
  .row-count { font-size: 11px; color: #94a3b8; margin-top: 8px; }
  #download-btn {
    margin-top: 16px;
    padding: 9px 20px;
    background: #fff;
    color: #2563eb;
    border: 1px solid #2563eb;
    border-radius: 4px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all .15s;
  }
  #download-btn:hover { background: #eff6ff; }

  /* ── Feedback ── */
  .feedback-wrap {
    margin-top: 20px;
    padding-top: 16px;
    border-top: 1px solid #e2e8f0;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }
  .feedback-wrap label {
    font-size: 12px;
    color: #64748b;
    font-weight: 600;
  }
  .fb-btn {
    padding: 7px 16px;
    border-radius: 4px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    border: 1px solid;
    transition: all .15s;
  }
  .fb-btn.error { background: #fff7ed; color: #c2410c; border-color: #fdba74; }
  .fb-btn.error:hover:not(:disabled) { background: #ffedd5; }
  .fb-btn:disabled { opacity: 0.45; cursor: not-allowed; }
  #feedback-msg { font-size: 12px; color: #64748b; }

  /* ── Login overlay ── */
  #login-overlay {
    position: fixed; inset: 0;
    background: rgba(15,23,42,0.7);
    display: flex; align-items: center; justify-content: center;
    z-index: 100;
  }
  .login-box {
    background: #fff;
    border-radius: 8px;
    padding: 40px 44px;
    width: 340px;
    text-align: center;
    box-shadow: 0 20px 60px rgba(0,0,0,.25);
  }
  .login-box h2 { font-size: 16px; font-weight: 700; margin-bottom: 4px; }
  .login-box p  { font-size: 12px; color: #64748b; margin-bottom: 24px; }
  .login-box input[type=password] {
    width: 100%;
    padding: 10px 14px;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    font-size: 14px;
    margin-bottom: 12px;
    outline: none;
  }
  .login-box input[type=password]:focus { border-color: #2563eb; }
  .login-box button {
    width: 100%;
    padding: 10px;
    background: #2563eb;
    color: #fff;
    border: none;
    border-radius: 4px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
  }
  .login-box button:hover { background: #1d4ed8; }
  #login-error { font-size: 12px; color: #b91c1c; margin-top: 8px; display: none; }

  /* ── Footer ── */
  footer {
    text-align: center;
    font-size: 11px;
    color: #94a3b8;
    margin-top: 40px;
    padding-bottom: 20px;
  }
  footer a { color: #64748b; text-decoration: none; }
</style>
</head>
<body>

<!-- Login overlay -->
<div id="login-overlay">
  <div class="login-box">
    <h2>FlowFigTabMiner</h2>
    <p>Enter access password to continue</p>
    <input type="password" id="pw-input" placeholder="Password" autofocus>
    <button onclick="doLogin()">Sign In</button>
    <div id="login-error">Incorrect password. Please try again.</div>
  </div>
</div>

<header>
  <h1>FlowFigTabMiner</h1>
  <span>Automated Data Extraction from Flow Chemistry Figures &amp; Tables</span>
  <span style="margin-left:auto;font-size:15px;color:#cbd5e1;">
    Developed by <a href="https://wwwchem.sci.hokudai.ac.jp/~yuhan/" target="_blank" rel="noopener"
      style="color:#fff;text-decoration:none;font-weight:700;">Nagaki Lab, Hokkaido University</a>
  </span>
</header>

<main>

  <!-- Extraction card -->
  <div class="card">
    <h2>Extract Data</h2>

    <!-- Mode selector -->
    <div class="tabs">
      <div class="tab active" id="tab-figure" onclick="setMode('figure')">📈 Figure Extraction</div>
      <div class="tab"        id="tab-table"  onclick="setMode('table')">📋 Table Extraction</div>
    </div>

    <!-- Upload zone -->
    <div class="upload-zone" id="upload-zone">
      <input type="file" id="file-input" accept="image/png,image/jpeg" onchange="onFileSelect(event)">
      <div class="icon">🖼️</div>
      <p><strong>Drop image here</strong> or click to browse</p>
      <p>PNG or JPG &nbsp;·&nbsp; Max 10 MB</p>
    </div>
    <div id="preview-wrap">
      <img id="preview-img" src="" alt="Preview">
      <div id="file-name"></div>
    </div>

    <!-- Example buttons -->
    <div class="examples-row">
      <span style="font-size:12px;color:#94a3b8;align-self:center;">Try an example:</span>
      <a class="ex-btn" onclick="loadExample('figure')" href="#">
        <img src="/examples/example_figure.png?v=2" alt="Example figure">
        Example Figure
      </a>
      <a class="ex-btn" onclick="loadExample('table')" href="#">
        <img src="/examples/example_table.png?v=2" alt="Example table">
        Example Table
      </a>
    </div>

    <!-- Instructions -->
    <div class="instructions" id="instructions-figure">
      <p><strong>Image requirements — Figure</strong></p>
      <ul>
        <li>Crop closely around the plot area; axis labels and scale tick values must be visible</li>
        <li>Caption and legend are fine to include</li>
        <li>Supported: scatter plots and line plots with numeric axes</li>
        <li>Avoid: bar charts, pie charts, reaction schemes, or images with large surrounding paragraphs</li>
      </ul>
    </div>
    <div class="instructions" id="instructions-table" style="display:none;">
      <p><strong>Image requirements — Table</strong></p>
      <ul>
        <li>Crop the table with a small margin; header row must be clearly visible</li>
        <li>Caption above or below the table is fine to include</li>
        <li>Supported: reaction optimization / screening tables with numeric or text cells</li>
        <li>Avoid: tables spanning multiple pages, or images with large surrounding paragraphs</li>
      </ul>
    </div>

    <!-- Submit -->
    <button id="submit-btn" onclick="doExtract()" disabled>Extract Data</button>
    <p style="margin-top:10px;font-size:11px;color:#94a3b8;">
      ⏱ First request may take 2–3 minutes while the extraction service initialises. Subsequent requests are faster.
    </p>

    <!-- Status -->
    <div id="status-wrap">
      <span class="status-badge running" id="status-badge">
        <span class="spinner"></span>
        <span id="status-text">Processing…</span>
      </span>
      <div id="elapsed"></div>
    </div>
  </div>

  <!-- Results card -->
  <div class="card" id="results-wrap">
    <h2>Extracted Data</h2>
    <div class="table-scroll">
      <table class="data-table" id="result-table">
        <thead id="result-thead"></thead>
        <tbody id="result-tbody"></tbody>
      </table>
    </div>
    <div class="row-count" id="row-count"></div>
    <button id="download-btn" onclick="downloadCSV()">⬇ Download CSV</button>
    <div class="feedback-wrap" id="feedback-wrap" style="display:none;">
      <label>Spot an issue?</label>
      <button class="fb-btn error" id="fb-error" onclick="doFeedback('error')">✗ Report Errors</button>
      <span id="feedback-msg"></span>
    </div>
  </div>

</main>

<footer>
  Developed by <a href="https://wwwchem.sci.hokudai.ac.jp/~yuhan/" target="_blank" rel="noopener">Nagaki Lab, Hokkaido University</a> &nbsp;·&nbsp;
  FlowFigTabMiner &nbsp;·&nbsp; For research use only
</footer>

<script>
let currentMode  = 'figure';
let currentFile  = null;   // File object from upload
let exampleMode  = null;   // 'figure' | 'table' if using built-in example
let csvRows      = [];
let currentGcsUri = null;  // GCS URI of last uploaded image (for feedback)

// ── Auth ──────────────────────────────────────────────────────────────────
function getCookie(name) {
  const v = document.cookie.split(';').find(c => c.trim().startsWith(name + '='));
  return v ? decodeURIComponent(v.trim().split('=')[1]) : null;
}
function checkAuth() {
  if (getCookie('auth_token')) {
    document.getElementById('login-overlay').style.display = 'none';
  }
}
async function doLogin() {
  const pw = document.getElementById('pw-input').value;
  const res = await fetch('/login', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ password: pw })
  });
  if (res.ok) {
    const data = await res.json();
    document.cookie = `auth_token=${data.token}; path=/; max-age=86400; SameSite=Strict`;
    document.getElementById('login-overlay').style.display = 'none';
  } else {
    document.getElementById('login-error').style.display = 'block';
    document.getElementById('pw-input').value = '';
    document.getElementById('pw-input').focus();
  }
}
document.getElementById('pw-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') doLogin();
});

// ── Mode ──────────────────────────────────────────────────────────────────
function setMode(mode) {
  currentMode = mode;
  document.getElementById('tab-figure').classList.toggle('active', mode === 'figure');
  document.getElementById('tab-table').classList.toggle('active',  mode === 'table');
  document.getElementById('instructions-figure').style.display = mode === 'figure' ? '' : 'none';
  document.getElementById('instructions-table').style.display  = mode === 'table'  ? '' : 'none';
}

// ── File select ───────────────────────────────────────────────────────────
function onFileSelect(e) {
  const file = e.target.files[0];
  if (!file) return;
  if (file.size > 10 * 1024 * 1024) {
    alert('File too large. Maximum size is 10 MB.'); return;
  }
  currentFile = file;
  exampleMode = null;
  showPreview(URL.createObjectURL(file), file.name);
  document.getElementById('submit-btn').disabled = false;
}

function showPreview(src, name) {
  document.getElementById('preview-img').src = src;
  document.getElementById('file-name').textContent = name;
  document.getElementById('preview-wrap').style.display = 'block';
}

// ── Drag & drop ───────────────────────────────────────────────────────────
const zone = document.getElementById('upload-zone');
zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('dragover'); });
zone.addEventListener('dragleave', ()  => zone.classList.remove('dragover'));
zone.addEventListener('drop', e => {
  e.preventDefault(); zone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && (file.type === 'image/png' || file.type === 'image/jpeg')) {
    currentFile = file; exampleMode = null;
    showPreview(URL.createObjectURL(file), file.name);
    document.getElementById('submit-btn').disabled = false;
  }
});

// ── Example ───────────────────────────────────────────────────────────────
function loadExample(type) {
  setMode(type);
  exampleMode = type;
  currentFile = null;
  showPreview(`/examples/example_${type}.png?v=2`, `example_${type}.png`);
  document.getElementById('submit-btn').disabled = false;
  return false;
}

// ── Timer ─────────────────────────────────────────────────────────────────
let _timerInterval = null;
function startTimer() {
  const start = Date.now();
  const el = document.getElementById('elapsed');
  el.textContent = '';
  _timerInterval = setInterval(() => {
    const s = Math.floor((Date.now() - start) / 1000);
    const m = Math.floor(s / 60), sec = s % 60;
    el.textContent = `Elapsed: ${m}:${String(sec).padStart(2,'0')} — Cold start may take up to 3 min on first request.`;
  }, 1000);
}
function stopTimer() {
  if (_timerInterval) { clearInterval(_timerInterval); _timerInterval = null; }
  document.getElementById('elapsed').textContent = '';
}

// ── Extract ───────────────────────────────────────────────────────────────
async function doExtract() {
  const token = getCookie('auth_token');
  if (!token) { document.getElementById('login-overlay').style.display = 'flex'; return; }

  document.getElementById('submit-btn').disabled = true;
  setStatus('running', 'Processing…');
  startTimer();
  document.getElementById('results-wrap').style.display = 'none';
  document.getElementById('feedback-wrap').style.display = 'none';
  document.getElementById('feedback-msg').textContent = '';
  document.getElementById('fb-error').disabled = false;
  csvRows = [];
  currentGcsUri = null;

  let body, url;
  if (exampleMode) {
    url  = '/extract-example';
    body = JSON.stringify({ type: exampleMode, auth_token: token });
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body
    });
    await handleExtractResponse(res);
  } else {
    const fd = new FormData();
    fd.append('file', currentFile);
    fd.append('type', currentMode);
    fd.append('auth_token', token);
    const res = await fetch('/extract', { method: 'POST', body: fd });
    await handleExtractResponse(res);
  }
  stopTimer();
  document.getElementById('submit-btn').disabled = false;
}

async function handleExtractResponse(res) {
  const data = await res.json();
  if (!res.ok || data.error) {
    setStatus('error', data.error || 'Extraction failed. Check the image requirements.');
    return;
  }
  if (data.gcs_uri) currentGcsUri = data.gcs_uri;
  const rows = data.csv_rows || [];
  if (data.status === 'success') {
    setStatus('success', `Extraction complete — ${data.row_count} rows`);
  } else if (rows.length > 1) {
    // Data was extracted but keyword filter flagged it as not flow-chemistry related
    setStatus('success', `Extraction complete — ${rows.length - 1} rows (relevance: ${data.status})`);
  } else if (data.status === 'no_scatter_points' || data.status === 'no_data_points') {
    setStatus('filtered', `No data points detected in this image.`);
  } else {
    setStatus('filtered', `Status: ${data.status}`);
  }
  if (rows.length > 0) {
    renderTable(rows);
    if (currentGcsUri) {
      document.getElementById('feedback-wrap').style.display = 'flex';
    }
  }
}

function setStatus(cls, text) {
  const wrap  = document.getElementById('status-wrap');
  const badge = document.getElementById('status-badge');
  const span  = document.getElementById('status-text');
  wrap.style.display = 'block';
  badge.className = `status-badge ${cls}`;
  span.textContent = text;
  const spinner = badge.querySelector('.spinner');
  if (cls === 'running') {
    if (!spinner) { const s = document.createElement('span'); s.className = 'spinner'; badge.prepend(s); }
  } else {
    if (spinner) spinner.remove();
  }
}

// ── Render table ──────────────────────────────────────────────────────────
function renderTable(rows) {
  csvRows = rows;
  const thead = document.getElementById('result-thead');
  const tbody = document.getElementById('result-tbody');
  thead.innerHTML = '';
  tbody.innerHTML = '';

  const header = rows[0];
  const tr = document.createElement('tr');
  header.forEach(h => { const th = document.createElement('th'); th.textContent = h; tr.appendChild(th); });
  thead.appendChild(tr);

  const displayRows = rows.slice(1, 51);  // show max 50 rows
  displayRows.forEach(row => {
    const tr = document.createElement('tr');
    row.forEach(c => { const td = document.createElement('td'); td.textContent = c; tr.appendChild(td); });
    tbody.appendChild(tr);
  });

  const total = rows.length - 1;
  document.getElementById('row-count').textContent =
    total > 50 ? `Showing first 50 of ${total} rows` : `${total} row${total !== 1 ? 's' : ''}`;
  document.getElementById('results-wrap').style.display = 'block';
}

// ── Download CSV ──────────────────────────────────────────────────────────
function downloadCSV() {
  if (!csvRows.length) return;
  const text = csvRows.map(r => r.map(c => `"${String(c).replace(/"/g,'""')}"`).join(',')).join('\\n');
  const blob = new Blob([text], { type: 'text/csv' });
  const a    = document.createElement('a');
  a.href     = URL.createObjectURL(blob);
  a.download = `extracted_${currentMode}_${Date.now()}.csv`;
  a.click();
}

// ── Feedback ──────────────────────────────────────────────────────────────
async function doFeedback(label) {
  const token = getCookie('auth_token');
  if (!token || !currentGcsUri) return;
  document.getElementById('fb-error').disabled = true;
  document.getElementById('feedback-msg').textContent = 'Saving…';
  try {
    const res = await fetch('/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label, gcs_uri: currentGcsUri, auth_token: token })
    });
    document.getElementById('feedback-msg').textContent =
      res.ok ? '✓ Flagged for review. Thanks!' : 'Could not save feedback.';
  } catch {
    document.getElementById('feedback-msg').textContent = 'Network error.';
  }
}

// ── Init ──────────────────────────────────────────────────────────────────
checkAuth();
</script>
</body>
</html>"""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE

@app.post("/login")
async def login(request: Request):
    body = await request.json()
    pw   = body.get("password", "")
    if pw != PASSWORD:
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="Wrong password")
    return JSONResponse({"token": _make_token(pw)})

@app.post("/extract")
async def extract(
    file:       UploadFile = File(...),
    type:       str        = Form(...),
    auth_token: str        = Form(...),
):
    if not _authenticated(auth_token):
        from fastapi import HTTPException
        raise HTTPException(status_code=401)

    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        return JSONResponse({"error": "File exceeds 10 MB limit."}, status_code=400)

    job_id   = f"frontend_{uuid.uuid4().hex[:12]}"
    gcs_path = f"frontend/{job_id}/{file.filename}"
    gcs_uri  = _gcs_upload_bytes(data, gcs_path,
                                  file.content_type or "image/png")

    return await _call_service(type, gcs_uri, job_id)

@app.post("/extract-example")
async def extract_example(request: Request):
    body       = await request.json()
    auth_token = body.get("auth_token", "")
    ex_type    = body.get("type", "figure")
    if not _authenticated(auth_token):
        from fastapi import HTTPException
        raise HTTPException(status_code=401)

    img_path = EXAMPLES_DIR / f"example_{ex_type}.png"
    data     = img_path.read_bytes()
    job_id   = f"frontend_example_{uuid.uuid4().hex[:8]}"
    gcs_uri  = _gcs_upload_bytes(data, f"frontend/{job_id}/example_{ex_type}.png")

    return await _call_service(ex_type, gcs_uri, job_id)

async def _call_service(ex_type: str, gcs_uri: str, job_id: str) -> JSONResponse:
    service_url = FIGURE_URL if ex_type == "figure" else TABLE_URL
    payload     = {"image_gcs_uri": gcs_uri, "job_id": job_id}

    last_error = None
    for attempt in range(2):  # retry once on failure
        try:
            async with httpx.AsyncClient(timeout=3300) as client:
                resp = await client.post(f"{service_url}/extract", json=payload)
                resp.raise_for_status()
            break  # success
        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt+1} failed: {type(e).__name__}: {e}")
            if attempt == 0:
                logger.info("Retrying after 5s...")
                import asyncio
                await asyncio.sleep(5)
    else:
        logger.error(f"All attempts failed: {type(last_error).__name__}: {last_error}")
        return JSONResponse({"error": f"Service unavailable after 2 attempts. Please try again in a few minutes."}, status_code=500)

    result   = resp.json()
    csv_uri  = result.get("csv_gcs_uri")
    csv_rows = []

    if csv_uri:
        try:
            csv_bytes = _gcs_download_bytes(csv_uri)
            reader    = csv.reader(io.StringIO(csv_bytes.decode("utf-8")))
            csv_rows  = list(reader)
        except Exception as e:
            logger.warning(f"CSV download failed: {e}")

    return JSONResponse({
        "status":    result.get("status", "unknown"),
        "row_count": result.get("row_count") or 0,
        "csv_rows":  csv_rows,
        "gcs_uri":   gcs_uri,
    })


@app.post("/feedback")
async def feedback(request: Request):
    from fastapi import HTTPException
    body       = await request.json()
    auth_token = body.get("auth_token", "")
    if not _authenticated(auth_token):
        raise HTTPException(status_code=401)

    label   = body.get("label")   # "good" | "error"
    gcs_uri = body.get("gcs_uri", "")
    if label not in ("good", "error") or not gcs_uri.startswith(f"gs://{GCS_DATA_BUCKET}/"):
        raise HTTPException(status_code=400, detail="Invalid feedback payload")

    src_path = gcs_uri.removeprefix(f"gs://{GCS_DATA_BUCKET}/")
    filename = src_path.replace("/", "__")          # flatten path into filename
    dst_path = f"feedback/{label}/{filename}"

    try:
        client   = storage.Client()
        bucket   = client.bucket(GCS_DATA_BUCKET)
        src_blob = bucket.blob(src_path)
        bucket.copy_blob(src_blob, bucket, dst_path)
        logger.info(f"Feedback '{label}': copied {src_path} -> {dst_path}")
    except Exception as e:
        logger.error(f"Feedback copy failed: {e}")
        raise HTTPException(status_code=500, detail="GCS copy failed")

    return JSONResponse({"ok": True})
