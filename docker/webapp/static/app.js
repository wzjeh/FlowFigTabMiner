// FlowFigTabMiner demo page: render precomputed examples and a visitor's own job.
"use strict";

const PALETTE = ["#2e4b7a", "#b7282e", "#6b8e4e", "#c9962b", "#6c4f7d", "#3d8a8a", "#7a7a7a"];
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 1) => (v == null || v === "" || Number.isNaN(Number(v)) ? "–" : Number(v).toFixed(d).replace(/\.0+$/, ""));
const fmtT = (v) => (v == null ? "–" : (Math.abs(v) < 0.1 ? Number(v).toPrecision(2) : fmt(v, 2)));
const mol = (smi, w = 120, h = 80) =>
  smi ? `<img class="mol" loading="lazy" alt="${esc(smi)}" title="${esc(smi)}" src="/api/mol.svg?w=${w * 2}&h=${h * 2}&smiles=${encodeURIComponent(smi)}">` : "";

// ── tabs ─────────────────────────────────────────────────────────────────
function openTab(name) {
  const t = document.querySelector(`.tab[data-tab="${name}"]`);
  if (!t) return;
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "panel-" + name));
}
document.querySelectorAll(".tab").forEach((t) =>
  t.addEventListener("click", () => { openTab(t.dataset.tab); history.replaceState(null, "", "#" + t.dataset.tab); })
);
openTab((location.hash || "#figure").slice(1));

// ── chart (inline SVG) ───────────────────────────────────────────────────
function ticks(lo, hi, n = 5) {
  const span = hi - lo || 1;
  const step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

function chart(panel) {
  const W = 560, H = 380, L = 64, R = 16, T = 20, B = 52;
  const pts = panel.series.flatMap((s, i) => s.points.map((p) => ({ x: p[0], y: p[1], v: p[2], s: i })));
  if (!pts.length) return `<p class="muted">No data points.</p>`;
  const logx = panel.x_log && pts.every((p) => p.x > 0);
  const fx = logx ? Math.log10 : (v) => v;
  let [x0, x1] = [Math.min(...pts.map((p) => fx(p.x))), Math.max(...pts.map((p) => fx(p.x)))];
  let [y0, y1] = [Math.min(...pts.map((p) => p.y)), Math.max(...pts.map((p) => p.y))];
  const px = (x1 - x0) * 0.06 || 1, py = (y1 - y0) * 0.08 || 1;
  x0 -= px; x1 += px; y0 -= py; y1 += py;
  const sx = (v) => L + ((fx(v) - x0) / (x1 - x0)) * (W - L - R);
  const sy = (v) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
  const heat = panel.chart_type === "heatmap" && pts.some((p) => p.v != null);
  const vals = pts.map((p) => p.v).filter((v) => v != null);
  const vmin = Math.min(...vals), vmax = Math.max(...vals);
  const heatColor = (v) => {
    if (v == null) return "#c9c3b8";
    const t = vmax > vmin ? (v - vmin) / (vmax - vmin) : 1;
    const a = [225, 214, 196], b = [183, 40, 46];
    return `rgb(${a.map((c, i) => Math.round(c + (b[i] - c) * t)).join(",")})`;
  };
  let g = "";
  let xt;
  if (logx) {
    xt = [];
    for (let e = Math.floor(x0); e <= Math.ceil(x1); e++)
      for (const m of Math.ceil(x1) - Math.floor(x0) <= 2 ? [1, 2, 5] : [1]) {
        const v = m * Math.pow(10, e);
        if (Math.log10(v) >= x0 && Math.log10(v) <= x1) xt.push(v);
      }
  } else xt = ticks(x0, x1);
  xt.forEach((v) => {
    const X = sx(v);
    g += `<line class="grid" x1="${X}" x2="${X}" y1="${T}" y2="${H - B}"/><text x="${X}" y="${H - B + 18}" text-anchor="middle">${esc(fmtT(v))}</text>`;
  });
  ticks(y0, y1).forEach((v) => {
    const Y = sy(v);
    g += `<line class="grid" x1="${L}" x2="${W - R}" y1="${Y}" y2="${Y}"/><text x="${L - 8}" y="${Y + 4}" text-anchor="end">${esc(fmt(v, 1))}</text>`;
  });
  g += `<line class="axis" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"/><line class="axis" x1="${L}" x2="${L}" y1="${T}" y2="${H - B}"/>`;
  const ax = panel.axes || {};
  if (ax.x) g += `<text x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle">${esc(ax.x)}</text>`;
  if (ax.y) g += `<text transform="translate(14 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">${esc(ax.y)}</text>`;
  pts.forEach((p) => {
    const c = heat ? heatColor(p.v) : PALETTE[p.s % PALETTE.length];
    g += `<circle cx="${sx(p.x).toFixed(1)}" cy="${sy(p.y).toFixed(1)}" r="${heat ? 6 : 4.2}" fill="${c}" fill-opacity="0.9"><title>${esc(panel.series[p.s].name)}: x=${esc(fmtT(p.x))}, y=${esc(fmt(p.y, 2))}${p.v != null ? ", value=" + esc(fmt(p.v, 1)) : ""}</title></circle>`;
    if (heat && p.v != null) g += `<text x="${sx(p.x).toFixed(1)}" y="${(sy(p.y) - 9).toFixed(1)}" text-anchor="middle">${esc(fmt(p.v, 0))}</text>`;
  });
  const legend = heat
    ? `<div class="legend"><span>colour and label: ${esc((panel.axes || {}).value || "value")}, ${esc(fmt(vmin, 0))}–${esc(fmt(vmax, 0))}${logx ? " · log x axis" : ""}</span></div>`
    : `<div class="legend">${panel.series.map((s, i) => `<span><i style="background:${PALETTE[i % PALETTE.length]}"></i>${esc(s.name)}</span>`).join("")}${logx ? "<span>log x axis</span>" : ""}</div>`;
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img">${g}</svg>${legend}`;
}

// ── renderers ────────────────────────────────────────────────────────────
function renderFigure(r) {
  if (r.message) return `<p class="muted">${esc(r.message)}</p>`;
  const panels = r.panels.map((p) => `
      <div class="frame">
        <div class="label">Extracted data · ${esc(p.label)}</div>
        ${chart(p)}
        ${p.caption ? `<p class="caption">${esc(p.caption)}</p>` : ""}
      </div>`).join("");
  const p0 = r.panels[0] || {}, ax = p0.axes || {};
  const heat = p0.chart_type === "heatmap";
  const pts = r.panels.flatMap((p) => p.series.flatMap((s) => s.points.map((pt) => ({ name: s.name, pt }))));
  const vals = pts.map((x) => x.pt[2]).filter((v) => v != null);
  const rows = heat ? pts.filter((x) => x.pt[2] != null).sort((a, b) => b.pt[2] - a.pt[2]).slice(0, 8) : pts.slice(0, 8);
  const second = heat
    ? `<div class="stat"><b>${fmt(Math.min(...vals), 0)}–${fmt(Math.max(...vals), 0)}</b><span>${esc(ax.value || "value")} range</span></div>`
    : `<div class="stat"><b>${r.panels.reduce((a, p) => a + p.series.length, 0)}</b><span>series</span></div>`;
  return `
    <div class="stats">
      <div class="stat"><b>${r.n_points}</b><span>data points</span></div>
      ${second}
      ${r.seconds ? `<div class="stat"><b>${r.seconds}s</b><span>reading time</span></div>` : ""}
    </div>
    <div class="pair">
      <div class="frame"><div class="label">Original figure</div><img class="shot" src="${r.base}${esc(r.input)}" alt="figure"></div>
      <div>${panels}
        <div class="scroll" style="margin-top:18px"><table class="data">
          <thead><tr>${heat ? "" : "<th>Series</th>"}<th>${esc(ax.x || "x")}</th><th>${esc(ax.y || "y")}</th><th>${esc(ax.value || "value")}</th></tr></thead>
          <tbody>${rows.map(({ name, pt }) => `<tr>${heat ? "" : `<td>${esc(name)}</td>`}<td class="num">${esc(fmtT(pt[0]))}</td><td class="num">${esc(fmt(pt[1], heat ? 0 : 2))}</td><td class="num">${esc(fmt(pt[2], heat ? 0 : 1))}</td></tr>`).join("")}</tbody>
        </table>${heat ? `<p class="caption">Highest-yield points shown first.</p>` : ""}</div>
      </div>
    </div>`;
}

function renderTable(r) {
  if (r.message) return `<p class="muted">${esc(r.message)}</p>`;
  const nStruct = r.rows.reduce((a, row) => a + row.filter((c) => c.smiles).length, 0);
  const cell = (c) => (c.smiles ? `<td>${mol(c.smiles)}</td>` : `<td>${esc(c.text)}</td>`);
  return `
    <div class="stats">
      <div class="stat"><b>${r.rows.length}</b><span>rows</span></div>
      <div class="stat"><b>${nStruct}</b><span>structures read</span></div>
      ${r.seconds ? `<div class="stat"><b>${r.seconds}s</b><span>reading time</span></div>` : ""}
    </div>
    <div class="pair">
      <div class="frame"><div class="label">Original table</div><img class="shot" src="${r.base}${esc(r.input)}" alt="table">
        ${r.caption ? `<p class="caption">${esc(r.caption)}</p>` : ""}</div>
      <div class="frame"><div class="label">Extracted table · structures drawn from the SMILES that were read</div>
        <div class="scroll"><table class="data">
          <thead><tr>${r.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>
          <tbody>${r.rows.map((row) => `<tr>${row.map(cell).join("")}</tr>`).join("")}</tbody>
        </table></div>
      </div>
    </div>`;
}

// organolithium reagents read better as names (n-BuLi) than as drawn ion pairs
const LI_NAMES = { nbuli: "n-BuLi", sbuli: "s-BuLi", tbuli: "t-BuLi", meli: "MeLi", phli: "PhLi", lda: "LDA" };
const reagent = (smi, name) => {
  if (smi && /Li/.test(smi)) return `<span class="reagent">${esc(LI_NAMES[String(name || "").toLowerCase().replace(/[^a-z]/g, "")] || name || "RLi")}</span>`;
  return mol(smi) || esc(name || "–");
};
const paperRow = (x) => `<tr>
  <td>${reagent(x.r1, x.r1_name)}</td>
  <td>${x.r2 || x.r2_name ? reagent(x.r2, x.r2_name) : ""}</td>
  <td class="arrow">→</td>
  <td>${mol(x.p) || esc(x.p_name || "–")}</td>
  <td class="num">${esc(fmt(x.T, 0))}</td>
  <td class="num">${esc(fmtT(x.tR))}</td>
  <td class="num">${esc(fmtT(x.tR2))}</td>
  <td class="num"><b>${esc(fmt(x.yield, 0))}</b></td>
  <td class="muted">${esc(x.source)}</td></tr>`;

function renderPaper(r, id) {
  if (r.message) return `<p class="muted">${esc(r.message)}</p>`;
  // a preview that walks through every source instead of 20 rows of Figure 2
  // (complete records, with substrate and product structures, first; "show all" lists every record)
  const complete = (x) => !!(x.r1 && x.p);
  const ordered = r.records.filter(complete).concat(r.records.filter((x) => !complete(x)));
  const bySrc = {};
  ordered.forEach((x) => (bySrc[x.source] = bySrc[x.source] || []).push(x));
  const first = [];
  for (let i = 0; first.length < 20 && i < r.records.length; i++)
    Object.values(bySrc).forEach((g) => g[i] && first.length < 20 && first.push(g[i]));
  return `
    <div class="paper-head">
      <h3>${esc(r.title)}</h3>
      ${r.citation ? `<p>${esc(r.citation)}</p>` : ""}
      ${r.doi ? `<p><a href="https://doi.org/${esc(r.doi)}" target="_blank" rel="noopener">doi:${esc(r.doi)}</a></p>` : ""}
    </div>
    <div class="stats" style="margin-top:22px">
      <div class="stat"><b>${r.n_records}</b><span>reaction records</span></div>
      <div class="stat"><b>${r.n_with_structure}</b><span>with product structure</span></div>
      <div class="stat"><b>${r.n_figures}</b><span>figures read</span></div>
      <div class="stat"><b>${r.n_tables}</b><span>tables read</span></div>
      ${r.seconds ? `<div class="stat"><b>${Math.round(r.seconds / 60)} min</b><span>reading time</span></div>` : ""}
    </div>
    <div class="label">Where the records come from</div>
    <div class="strip">${r.sources.map((s) => `
      <div class="thumb">${s.thumb ? `<img src="${r.base}${esc(s.thumb)}" alt="${esc(s.label)}">` : ""}
        <div><b>${esc(s.label.replace(/\s*\(p\.\d+\)/, ""))}</b><span>${s.n} records</span></div></div>`).join("")}
    </div>
    <div class="label">Reaction records</div>
    <div class="scroll"><table class="data">
      <thead><tr><th>Substrate</th><th>Reagent</th><th></th><th>Product</th><th>T (°C)</th><th>t<sub>R1</sub> (s)</th><th>t<sub>R2</sub> (s)</th><th>Yield (%)</th><th>Source</th></tr></thead>
      <tbody id="${id}-rows">${first.map(paperRow).join("")}</tbody>
    </table></div>
    ${r.records.length > first.length ? `<button class="more" data-target="${id}">Show all ${r.records.length} records</button>` : ""}`;
}

const RENDER = { figure: renderFigure, table: renderTable, paper: renderPaper, pdf: renderPaper };
const LAST = {};
function show(el, kind, r, id) {
  LAST[id] = r;
  el.innerHTML = RENDER[kind](r, id);
  const more = el.querySelector(".more");
  if (more) more.addEventListener("click", () => {
    const x = LAST[id];
    document.getElementById(id + "-rows").innerHTML = x.records.map(paperRow).join("");
    more.remove();
  });
}

// ── examples ─────────────────────────────────────────────────────────────
fetch("/api/examples").then((r) => r.json()).then((ex) => {
  for (const kind of ["figure", "table", "paper"]) {
    const el = document.getElementById("example-" + kind);
    if (ex[kind]) show(el, kind, ex[kind], "ex-" + kind);
    else el.innerHTML = `<p class="muted">Example not available.</p>`;
  }
});

// ── try your own ─────────────────────────────────────────────────────────
const TRY_TEXT = {
  figure: ["Drop a figure image here, or click to choose", "PNG or JPG of one scatter plot or heatmap · about 1–2 minutes"],
  table: ["Drop a table image here, or click to choose", "PNG or JPG of one reaction table · about 1–2 minutes"],
  pdf: ["Drop a paper (PDF) here, or click to choose", "One flow-chemistry research article · about 10 minutes · the file is deleted afterwards"],
};

document.querySelectorAll(".try").forEach((box) => {
  const kind = box.dataset.kind;
  box.innerHTML = `
    <label class="drop"><input type="file" hidden accept="${kind === "pdf" ? ".pdf" : "image/png,image/jpeg"}">
      <strong>${TRY_TEXT[kind][0]}</strong>
      <div class="note">${TRY_TEXT[kind][1]}</div>
      <div class="note">One free try per visitor.</div>
    </label>
    <div class="status"><span class="brush"></span><span class="msg"></span></div>
    <div class="out"></div>`;
  const input = box.querySelector("input"), drop = box.querySelector(".drop");
  const status = box.querySelector(".status"), msg = box.querySelector(".msg"), out = box.querySelector(".out");
  const say = (text, cls = "") => { status.className = "status on " + cls; msg.textContent = text; };

  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); if (e.dataTransfer.files[0]) submit(e.dataTransfer.files[0]); });
  input.addEventListener("change", () => input.files[0] && submit(input.files[0]));

  async function submit(file) {
    out.innerHTML = "";
    say("Uploading " + file.name + " …");
    const fd = new FormData();
    fd.append("kind", kind);
    fd.append("file", file);
    let res;
    try {
      res = await fetch("/api/jobs", { method: "POST", body: fd });
    } catch (e) { return say("Could not reach the server. Please try again.", "error"); }
    const body = await res.json().catch(() => ({}));
    if (!res.ok) return say(body.error || "The server refused this file.", "error");
    poll(body.id);
  }

  function poll(id) {
    fetch("/api/jobs/" + id).then((r) => r.json()).then((j) => {
      const mmss = `${Math.floor(j.elapsed / 60)}:${String(j.elapsed % 60).padStart(2, "0")}`;
      if (j.state === "queued") { say(`Waiting in line (${j.ahead} ahead) …`); }
      else if (j.state === "running") { say(`${j.step} … ${mmss}`); }
      else if (j.state === "done") { say(`Done in ${mmss}.`, "done"); show(out, kind, j.result, "try-" + kind); return; }
      else { say(j.error || "Extraction failed.", "error"); return; }
      setTimeout(() => poll(id), 3000);
    }).catch(() => setTimeout(() => poll(id), 5000));
  }
});
