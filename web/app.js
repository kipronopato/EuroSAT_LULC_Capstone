// EuroSAT Land Cover - single-page front end (no build step). Talks to the FastAPI backend:
//   GET /health, /model-info, /samples, /map/demo     POST /predict, /map
"use strict";

const $ = (id) => document.getElementById(id);

const GLOSSARY = {
  AnnualCrop: "Fields ploughed and replanted every season: cereals, maize, vegetables.",
  Forest: "Dense tree cover, broadleaf or conifer.",
  HerbaceousVegetation: "Natural grass, shrubs and meadows that are not grazed or cropped.",
  Highway: "Major roads and motorways with their verges and junctions.",
  Industrial: "Factories, warehouses and large commercial roofs.",
  Pasture: "Grassland grazed by livestock or cut for hay.",
  PermanentCrop: "Orchards, vineyards and olive groves planted in rows.",
  Residential: "Houses, gardens and urban street blocks.",
  River: "Rivers and canals winding through the landscape.",
  SeaLake: "Open water: sea, lakes and reservoirs.",
};
const PRETTY = { HerbaceousVegetation: "Herbaceous Vegetation", AnnualCrop: "Annual Crop", PermanentCrop: "Permanent Crop", SeaLake: "Sea / Lake" };
let COLOURS = {
  AnnualCrop: "#f2c84b", PermanentCrop: "#e07b39", Pasture: "#b5d99c", HerbaceousVegetation: "#6aa84f", Forest: "#1e5631",
  Residential: "#d93f3f", Industrial: "#8e44ad", Highway: "#7c7c7c", River: "#4fa3e0", SeaLake: "#1f4e9c",
};
const pretty = (c) => PRETTY[c] || c;
const pct = (p, d = 1) => (p >= 0.9995 ? ">99.9%" : p < 0.0005 ? "<0.1%" : (p * 100).toFixed(d) + "%");
const area = (ha) => (ha == null ? "" : ha >= 100 ? `${(ha / 100).toLocaleString(undefined, { maximumFractionDigits: ha >= 10000 ? 0 : 1 })} km²` : `${ha.toFixed(1)} ha`);
const hexRgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "style") n.style.cssText = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c != null) n.append(c);
  return n;
}
const icon = (name, cls = "ic") => {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("class", cls);
  const u = document.createElementNS("http://www.w3.org/2000/svg", "use");
  u.setAttribute("href", "#i-" + name);
  s.append(u);
  return s;
};

function toast(msg, type = "error") {
  const t = el("div", { class: `toast ${type}`, role: "status" }, icon(type === "error" ? "alert" : "check"), el("span", {}, msg));
  $("toasts").append(t);
  setTimeout(() => t.remove(), type === "error" ? 7000 : 3500);
}

async function api(path, opts = {}, timeoutS = 180) {
  // Never spin forever: give up after timeoutS with a clear message
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutS * 1000);
  let res;
  try {
    res = await fetch(path, { ...opts, signal: ctl.signal });
  } catch (e) {
    throw new Error(e.name === "AbortError"
      ? `The server took longer than ${timeoutS} s. Try "Fast" detail or a smaller image.`
      : "Could not reach the server. Is it running?");
  } finally {
    clearTimeout(timer);
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = body.detail;
    throw new Error(Array.isArray(d) ? d.map((x) => x.msg).join("; ") : d || `Server error (${res.status})`);
  }
  return body;
}

// ================================================================ theme, menu, router
$("theme").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("theme", root.dataset.theme); } catch (e) {}
});
$("menu").addEventListener("click", () => {
  const open = $("nav-links").classList.toggle("open");
  $("menu").setAttribute("aria-expanded", String(open));
});

const ROUTES = ["home", "map", "classify", "classes", "about"];
function route() {
  const name = (location.hash.replace(/^#\/?/, "").split("?")[0] || "home");
  const page = ROUTES.includes(name) ? name : "home";
  document.querySelectorAll(".page").forEach((p) => { p.hidden = p.dataset.page !== page; });
  document.querySelectorAll(".nav-links a").forEach((a) => {
    a.classList.toggle("active", a.dataset.route === page);
    if (a.dataset.route === page) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  $("nav-links").classList.remove("open");
  $("menu").setAttribute("aria-expanded", "false");
  document.title = page === "home" ? "EuroSAT Land Cover" : `${page[0].toUpperCase() + page.slice(1)} · EuroSAT Land Cover`;
  window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });
}
window.addEventListener("hashchange", route);
route();

// ================================================================ compare component (hero + map viewer)
function makeCompare(root) {
  const stage = root.querySelector(".stage"), handle = root.querySelector(".handle");
  let dragging = false;
  const set = (p) => {
    p = Math.max(0, Math.min(100, p));
    root.style.setProperty("--pos", p + "%");
    stage.style.setProperty("--pos", p + "%");
    handle.style.setProperty("--pos", p + "%");
    handle.setAttribute("aria-valuenow", Math.round(p));
    root.dataset.pos = p;
  };
  const fromEvent = (e) => {
    const r = stage.getBoundingClientRect();
    set(((e.clientX - r.left) / r.width) * 100);
  };
  stage.addEventListener("pointerdown", (e) => {
    if (!root.classList.contains("swipe")) return;
    dragging = true; stage.setPointerCapture(e.pointerId); fromEvent(e);
  });
  stage.addEventListener("pointermove", (e) => { if (dragging) fromEvent(e); });
  stage.addEventListener("pointerup", () => { dragging = false; });
  stage.addEventListener("pointercancel", () => { dragging = false; });
  handle.addEventListener("keydown", (e) => {
    const step = e.shiftKey ? 10 : 2, p = +root.dataset.pos;
    if (e.key === "ArrowLeft") { set(p - step); e.preventDefault(); }
    if (e.key === "ArrowRight") { set(p + step); e.preventDefault(); }
  });
  set(+root.dataset.pos || 50);
  return {
    stage, set,
    mode(m) { root.classList.toggle("swipe", m === "swipe"); },
    aspect(w, h) { stage.style.setProperty("--ar", `${w} / ${h}`); stage.style.setProperty("--arn", w / h); },
  };
}

const hero = makeCompare($("hero-compare"));
hero.mode("swipe");
hero.aspect(512, 384);

// ================================================================ server status, model info, demo map
let demoPromise = null;
const getDemo = () => (demoPromise ||= api("/map/demo").catch((e) => { demoPromise = null; throw e; }));

api("/health").then(() => {
  $("status").classList.add("ok"); $("status-text").textContent = "Models online";
}).catch(() => {
  $("status").classList.add("down"); $("status-text").textContent = "Offline";
});

let INFO = null;
api("/model-info").then((info) => {
  INFO = info;
  if (info.colours) COLOURS = info.colours;
  const p = info.pipelines || {};
  if (p.ms) document.querySelector('[data-stat="f1ms"]').textContent = (p.ms.test_macro_f1 * 100).toFixed(1) + "%";
  $("about-kpis").replaceChildren(
    kpi("GeoTIFF (13-band) accuracy", p.ms ? pct(p.ms.test_macro_f1) : "–", "test macro F1, ensemble", null, true),
    kpi("PNG / JPEG accuracy", p.rgb ? pct(p.rgb.test_macro_f1) : "–", "test macro F1, EfficientNet-B0", null, true),
    kpi("Training patches", "19,286", "EuroSAT, 10 classes"),
    kpi("Test patches", "3,857", "from unseen locations"),
  );
  renderClassGrid();
}).catch(() => renderClassGrid());

getDemo().then((r) => {
  $("hero-map").src = r.map;
  $("hero-map").hidden = false;
  hero.aspect(r.width, r.height);
  $("hero-loading").hidden = true;
}).catch(() => { $("hero-loading").textContent = "Demo map unavailable"; });

function kpi(label, value, sub, colour, big) {
  return el("div", { class: "card kpi" + (big ? " big" : "") },
    el("span", {}, label),
    el("b", { title: value }, colour ? el("i", { class: "sw", style: `background:${colour}` }) : null, value),
    sub ? el("small", { title: sub }, sub) : null);
}

// ================================================================ map workspace
const viewerCompare = makeCompare($("map-compare"));
const M = { result: null, input: null, isolate: null, uncertainty: null, isolated: {}, started: 0, timer: null };

function segmented(id, onChange) {
  const root = $(id);
  root.addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    root.querySelectorAll("button").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
    onChange?.(b.dataset.value);
  });
  return () => root.querySelector('[aria-checked="true"]').dataset.value;
}
const detail = segmented("map-detail", () => { if (M.input) $("map-rerun").disabled = false; });
segmented("view-mode", (m) => {
  viewerCompare.mode(m);
  $("opacity-wrap").style.visibility = m === "swipe" ? "hidden" : "visible";
});
$("map-res").addEventListener("input", () => { if (M.input) $("map-rerun").disabled = false; });

function setBusy(on) {
  $("map-busy").hidden = !on;
  clearInterval(M.timer);
  if (on) {
    M.started = performance.now();
    $("map-timer").textContent = "0 s";
    M.timer = setInterval(() => { $("map-timer").textContent = `${Math.round((performance.now() - M.started) / 1000)} s`; }, 500);
  }
}

async function runMap(input) {
  M.input = input;
  $("map-rerun").disabled = true;
  $("map-file-chip").hidden = false;
  $("map-file-chip").replaceChildren(icon("image"), el("span", {}, input.name));
  setBusy(true);
  try {
    let r;
    const defaults = !$("map-res").value && detail() === "normal";
    if (input.demo && defaults) {
      r = await getDemo();
    } else {
      const blob = input.blob || await (await fetch("/samples/mosaic_demo.tif")).blob();
      const form = new FormData();
      form.append("file", blob, input.name);
      form.append("detail", detail());
      if ($("map-res").value) form.append("resolution_m", $("map-res").value);
      r = await api("/map", { method: "POST", body: form });
    }
    renderMap(r, (performance.now() - M.started) / 1000);
  } catch (e) {
    toast(e.message);
    if (input.demo) demoPromise = null;
  } finally {
    setBusy(false);
  }
}

function renderMap(r, seconds) {
  M.result = r; M.isolate = null; M.uncertainty = null; M.isolated = {};
  const g = atob(r.grid.data);
  M.grid = { w: r.grid.width, h: r.grid.height, data: Uint8Array.from(g, (c) => c.charCodeAt(0)) };

  $("map-empty").hidden = true;
  $("map-compare").hidden = false;
  viewerCompare.aspect(r.width, r.height);
  $("map-image").src = r.image;
  $("map-conf").checked = false;
  updateOverlay();

  $("map-full").disabled = false;
  const dl = $("map-download");
  dl.href = r.map; dl.classList.remove("disabled"); dl.removeAttribute("aria-disabled");

  const top = r.classes[0];
  const totalHa = r.classes.reduce((s, c) => s + (c.area_ha || 0), 0);
  $("map-kpis").replaceChildren(
    kpi("Dominant cover", pretty(top.class), `${pct(top.fraction)} of the image`, COLOURS[top.class]),
    kpi("Area analysed", area(totalHa), `${r.width} × ${r.height} px at ${r.resolution_m} m`),
    kpi("Mean confidence", pct(r.mean_confidence, 0), r.mean_confidence >= 0.75 ? "high" : r.mean_confidence >= 0.55 ? "moderate" : "low: check resolution"),
    kpi("Classes found", `${r.classes.length} of 10`, seconds != null && !r.demo ? `analysed in ${seconds.toFixed(1)} s` : "demo scene"),
  );
  renderLegend();
  const resNote = { file: "from the GeoTIFF", user: "as entered", assumed: "assumed (Sentinel-2)" }[r.resolution_source];
  $("map-meta").textContent = `${r.windows.toLocaleString()} windows of ${r.tile_px} px, stride ${r.stride_px} px · resolution ${resNote} · ` +
    `${r.model.ensemble ? "ensemble" : "model"}: ${r.model.run} (test macro F1 ${r.model.test_macro_f1.toFixed(3)})`;
  $("map-results").hidden = false;
}

function renderLegend() {
  const r = M.result;
  const present = new Map(r.classes.map((c) => [c.class, c]));
  const order = [...r.classes.map((c) => c.class), ...r.legend.map((l) => l.class).filter((c) => !present.has(c))];
  $("map-legend").classList.toggle("isolating", M.isolate != null);
  $("legend-reset").hidden = M.isolate == null;
  $("map-legend").replaceChildren(...order.map((name) => {
    const c = present.get(name);
    const btn = el("button", {
      type: "button", "aria-pressed": String(M.isolate === name), disabled: !c, title: GLOSSARY[name],
      onclick: () => { M.isolate = M.isolate === name ? null : name; renderLegend(); updateOverlay(); },
    },
      el("span", { class: "sw", style: `background:${COLOURS[name]}` }),
      el("span", { class: "nm" }, pretty(name)),
      el("span", { class: "val" }, c ? `${pct(c.fraction)} · ${area(c.area_ha)}` : "not found"),
      el("span", { class: "bar" }, el("i", { style: `width:${c ? Math.max(c.fraction * 100, 0.8) : 0}%;background:${COLOURS[name]}` })),
    );
    return el("li", { class: c ? "" : "absent" }, btn);
  }));
}
$("legend-reset").addEventListener("click", () => { M.isolate = null; renderLegend(); updateOverlay(); });

function loadImage(src) {
  return new Promise((ok, fail) => { const i = new Image(); i.onload = () => ok(i); i.onerror = fail; i.src = src; });
}
async function transform(src, fn) {
  const img = await loadImage(src);
  const c = document.createElement("canvas");
  c.width = img.naturalWidth; c.height = img.naturalHeight;
  const ctx = c.getContext("2d");
  ctx.drawImage(img, 0, 0);
  const d = ctx.getImageData(0, 0, c.width, c.height);
  fn(d.data);
  ctx.putImageData(d, 0, 0);
  return c.toDataURL();
}

async function updateOverlay() {
  const r = M.result; if (!r) return;
  const over = $("map-overlay");
  over.style.opacity = $("map-conf").checked ? 1 : $("map-opacity").value / 100;
  if ($("map-conf").checked) {
    // Uncertain pixels in amber, confident ones transparent: shows where to trust the map less
    M.uncertainty ||= await transform(r.confidence, (p) => {
      for (let i = 0; i < p.length; i += 4) {
        const c = p[i] / 255, a = Math.max(0, Math.min(1, (0.9 - c) / 0.45));
        p[i] = 245; p[i + 1] = 158; p[i + 2] = 11; p[i + 3] = a * 230;
      }
    });
    over.src = M.uncertainty;
  } else if (M.isolate) {
    const name = M.isolate, [R, G, B] = hexRgb(COLOURS[name]);
    M.isolated[name] ||= await transform(r.map, (p) => {
      for (let i = 0; i < p.length; i += 4) {
        const hit = Math.abs(p[i] - R) + Math.abs(p[i + 1] - G) + Math.abs(p[i + 2] - B) < 12;
        p[i + 3] = hit ? 255 : 0;
      }
    });
    if (M.isolate === name) over.src = M.isolated[name];
  } else {
    over.src = r.map;
  }
}
$("map-opacity").addEventListener("input", updateOverlay);
$("map-conf").addEventListener("change", updateOverlay);

const tip = $("map-tip"), stage = viewerCompare.stage;
stage.addEventListener("mousemove", (e) => {
  if (!M.grid) return;
  const b = stage.getBoundingClientRect();
  const fx = (e.clientX - b.left) / b.width, fy = (e.clientY - b.top) / b.height;
  if (fx < 0 || fy < 0 || fx > 1 || fy > 1) return;
  const gx = Math.min(M.grid.w - 1, Math.floor(fx * M.grid.w)), gy = Math.min(M.grid.h - 1, Math.floor(fy * M.grid.h));
  const name = M.result.legend[M.grid.data[gy * M.grid.w + gx]].class;
  tip.replaceChildren(el("i", { style: `background:${COLOURS[name]}` }), pretty(name));
  tip.style.left = `${e.clientX - b.left}px`;
  tip.style.top = `${e.clientY - b.top}px`;
  tip.hidden = false;
});
stage.addEventListener("mouseleave", () => { tip.hidden = true; });

$("map-full").addEventListener("click", () => {
  const v = $("viewer");
  if (document.fullscreenElement) document.exitFullscreen(); else v.requestFullscreen?.();
});

function wireDrop(dropId, inputId, onFile) {
  const drop = $(dropId), input = $(inputId);
  input.addEventListener("change", () => { if (input.files[0]) onFile(input.files[0]); input.value = ""; });
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  for (const ev of ["dragenter", "dragover"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); });
  for (const ev of ["dragleave", "drop"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); });
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) onFile(f); });
}
wireDrop("map-drop", "map-file", (f) => runMap({ blob: f, name: f.name }));
$("map-demo").addEventListener("click", () => runMap({ demo: true, name: "mosaic_demo.tif (demo scene)" }));
$("map-rerun").addEventListener("click", () => M.input && runMap(M.input));

// Deep link: #/map?demo opens the workspace with the demo scene loaded (handy for presentations)
function autoDemo() { if (/^#\/map\?demo/.test(location.hash) && !M.result) runMap({ demo: true, name: "mosaic_demo.tif (demo scene)" }); }
window.addEventListener("hashchange", autoDemo);
autoDemo();

// ================================================================ classify
let SAMPLES = [];
const C = {};

function showCls(state) {
  for (const s of ["empty", "busy", "result"]) $("cls-" + s).hidden = s !== state;
}

async function classify(blob, filename) {
  showCls("busy");
  const form = new FormData();
  form.append("file", blob, filename);
  try {
    renderPrediction(await api("/predict", { method: "POST", body: form }));
  } catch (e) {
    showCls(C.last ? "result" : "empty");
    toast(e.message);
  }
}

function renderPrediction(r) {
  C.last = r;
  const name = r.predicted_class, colour = COLOURS[name];
  $("preview").src = r.preview;
  $("pred").replaceChildren(el("i", { style: `background:${colour}` }), pretty(name));
  $("pred-desc").textContent = GLOSSARY[name] || "";
  const ring = $("ring");
  ring.style.setProperty("--p", (r.confidence * 100).toFixed(1));
  ring.style.setProperty("--c", r.confidence >= 0.75 ? "var(--accent)" : r.confidence >= 0.5 ? "var(--warn)" : "var(--danger)");
  $("ring-val").textContent = r.confidence >= 0.995 ? "99+" : Math.round(r.confidence * 100) + "%";
  $("note").textContent = r.confidence >= 0.75 ? "The model is confident about this patch."
    : `Uncertain: it could also be ${pretty(r.top3[1].class)} (${pct(r.top3[1].probability)}).`;

  $("bars").replaceChildren(...r.probabilities.map((p, i) => {
    const fill = el("span", { class: "fill", style: `background:${COLOURS[p.class]};width:${Math.max(p.probability * 100, 0.6)}%` });
    return el("li", { class: i === 0 ? "top" : "", title: GLOSSARY[p.class] },
      el("span", { class: "name" }, el("i", { style: `background:${COLOURS[p.class]}` }), el("span", {}, pretty(p.class))),
      el("span", { class: "track" }, fill),
      el("span", { class: "pct" }, pct(p.probability)));
  }));
  $("used").textContent = `${r.model.ensemble ? "Ensemble" : "Model"}: ${r.model.run} · test macro F1 ${r.model.test_macro_f1.toFixed(3)}`;
  showCls("result");
}

async function classifySample(s) {
  const blob = await (await fetch(s.file)).blob();
  classify(blob, s.file.split("/").pop());
}

wireDrop("drop", "file", (f) => classify(f, f.name));

api("/samples").then((samples) => {
  SAMPLES = samples;
  $("samples").replaceChildren(...samples.map((s) =>
    el("button", { class: "sample", type: "button", title: `Classify the ${pretty(s.class)} sample`, onclick: () => classifySample(s) },
      el("img", { src: s.thumbnail, alt: "", loading: "lazy" }), el("span", {}, pretty(s.class)))));
  renderClassGrid();
  // Deep link: #/classify?sample=Forest classifies that sample on arrival
  const m = location.hash.match(/^#\/classify\?sample=(\w+)/);
  const s = m && samples.find((x) => x.class === m[1]);
  if (s) classifySample(s);
}).catch(() => {});

// ================================================================ classes
function renderClassGrid() {
  const thumbs = Object.fromEntries(SAMPLES.map((s) => [s.class, s]));
  const names = INFO?.classes || Object.keys(GLOSSARY);
  $("class-grid").replaceChildren(...names.map((name) => {
    const s = thumbs[name];
    return el("button", {
      class: "card class-card", type: "button", style: `--c:${COLOURS[name]}`,
      onclick: () => { if (s) { location.hash = "#/classify"; classifySample(s); } },
    },
      el("div", { class: "thumb" }, s ? el("img", { src: s.thumbnail, alt: `${pretty(name)} example`, loading: "lazy" }) : null),
      el("div", { class: "body" },
        el("h3", {}, el("i"), pretty(name)),
        el("p", {}, GLOSSARY[name] || ""),
        s ? el("span", { class: "try" }, "Classify this example", icon("arrow")) : null));
  }));
}
renderClassGrid();
