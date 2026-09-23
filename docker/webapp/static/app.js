// FlowFigTabMiner demo page: render precomputed examples and a visitor's own job.
"use strict";

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

// ── figure / table renderers ────────────────────────────────────────────
// Examples ("reveal" mode): the original on the left; on the right a panel of
// the same height with one Extract button. Pressing it waits as long as the
// pipeline really took, then shows the extracted data in the panel (scrolls
// inside). A visitor's own result is shown straight away.

function figureRows(r) {
  const p0 = r.panels[0] || {}, ax = p0.axes || {}, sideOf = p0.series_axis || null;
  const heat = p0.chart_type === "heatmap";
  let pts = r.panels.flatMap((p) => p.series.flatMap((s) => s.points.map((pt) => ({ name: s.name, x: pt[0], y: pt[1], v: pt[2] }))));
  if (heat) pts = pts.filter((p) => p.v != null);          // unlabelled cells carry no value
  // the same residence time (x as displayed) together; within it top to bottom, then by series
  const xKey = (p) => Number(fmtT(p.x));
  pts.sort((a, b) => xKey(a) - xKey(b) || b.y - a.y || a.name.localeCompare(b.name));
  let head, row;
  if (heat) {
    head = [ax.x || "x", ax.y || "y", ax.value || "value"];
    row = (p) => [fmtT(p.x), fmt(p.y, 0), fmt(p.v, 0)];
  } else if (sideOf) {
    // two y axes: each series is read on its own axis, so it fills one of the two columns
    head = ["series", ax.x || "x", ax.y_left || "left y axis", ax.y_right || "right y axis"];
    row = (p) => sideOf[p.name] === "right" ? [p.name, fmtT(p.x), "", fmt(p.v, 2)] : [p.name, fmtT(p.x), fmt(p.y, 2), ""];
  } else {
    const hasV = pts.some((p) => p.v != null);
    head = ["series", ax.x || "x", ax.y || "y"].concat(hasV ? [ax.value || "right axis"] : []);
    row = (p) => [p.name, fmtT(p.x), fmt(p.y, 2)].concat(hasV ? [fmt(p.v, 2)] : []);
  }
  const isNum = (c) => /^[-–+\d.e]+$/.test(String(c));
  return { n: pts.length, html: `<table class="data"><thead><tr>${head.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead>
    <tbody>${pts.map((p) => `<tr>${row(p).map((c) => `<td class="${isNum(c) ? "num" : ""}">${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table>` };
}

function tableRows(r) {
  const cell = (c) => (c.smiles ? `<td>${mol(c.smiles)}</td>` : `<td>${esc(c.text)}</td>`);
  return `<table class="data"><thead><tr>${r.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>
    <tbody>${r.rows.map((row) => `<tr>${row.map(cell).join("")}</tr>`).join("")}</tbody></table>`;
}

function pairView(r, what, original, summary, body, reveal) {
  const right = reveal
    ? `<div class="reveal" data-seconds="${r.seconds || 10}">
         <button class="extract"><span>Extract</span><small>抽出</small></button>
         <div class="reading" hidden><div class="bar"><i></i></div><span class="msg"></span></div>
         <div class="reveal-body" hidden><div class="label">${summary}</div><div class="fill-scroll">${body}</div></div>
       </div>`
    : `<div class="reveal-body"><div class="label">${summary}</div><div class="fill-scroll">${body}</div></div>`;
  return `<div class="pair stretch">
      <div class="frame"><div class="label">Original ${what}</div>${original}</div>
      <div class="frame fill">${right}</div>
    </div>`;
}

function renderFigure(r, id, reveal) {
  if (r.message) return `<p class="muted">${esc(r.message)}</p>`;
  const cap = (r.panels[0] || {}).caption;
  const original = `<img class="shot" src="${r.base}${esc(r.input)}" alt="figure">${cap ? `<p class="caption">${esc(cap)}</p>` : ""}`;
  const t = figureRows(r);
  const summary = `Extracted data points · ${t.n} points${r.seconds ? ` · read in ${r.seconds} s` : ""}`;
  return pairView(r, "figure", original, summary, t.html, reveal);
}

function renderTable(r, id, reveal) {
  if (r.message) return `<p class="muted">${esc(r.message)}</p>`;
  const nStruct = r.rows.reduce((a, row) => a + row.filter((c) => c.smiles).length, 0);
  const original = `<img class="shot" src="${r.base}${esc(r.input)}" alt="table">`;
  const summary = `Extracted table · ${r.rows.length} rows · ${nStruct} structures read${r.seconds ? ` · read in ${r.seconds} s` : ""}`;
  return pairView(r, "table", original, summary, tableRows(r), reveal);
}

function wireReveal(el) {
  el.querySelectorAll(".reveal").forEach((box) => {
    const btn = box.querySelector(".extract"), reading = box.querySelector(".reading");
    const bar = box.querySelector(".bar i"), msg = box.querySelector(".msg"), body = box.querySelector(".reveal-body");
    const total = Number(box.dataset.seconds) || 10;
    if (/[?&]shown\b/.test(location.search)) { btn.hidden = true; body.hidden = false; return; }   // ?shown: results without the wait
    btn.addEventListener("click", () => {
      btn.hidden = true;
      reading.hidden = false;
      const t0 = performance.now();
      const tick = () => {
        const s = (performance.now() - t0) / 1000;
        bar.style.width = Math.min(100, (100 * s) / total) + "%";
        msg.textContent = "Reading …";          // how long it takes is only known afterwards
        if (s < total) return requestAnimationFrame(tick);
        reading.hidden = true;
        body.hidden = false;
      };
      requestAnimationFrame(tick);
    });
  });
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
function show(el, kind, r, id, reveal = false) {
  LAST[id] = r;
  el.innerHTML = RENDER[kind](r, id, reveal);
  wireReveal(el);
  const more = el.querySelector(".more");
  if (more) more.addEventListener("click", () => {
    const x = LAST[id];
    document.getElementById(id + "-rows").innerHTML = x.records.map(paperRow).join("");
    more.remove();
  });
}

// ── examples ─────────────────────────────────────────────────────────────
fetch("/api/examples").then((r) => r.json()).then((ex) => {
  for (const key of ["figure", "figure2", "table", "paper"]) {
    const el = document.getElementById("example-" + key);
    const kind = key.replace(/\d+$/, "");
    if (ex[key]) show(el, kind, ex[key], "ex-" + key, kind !== "paper");
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
