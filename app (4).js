/* AI Handwritten Text Recognition – Tamil & English (runs fully in the browser) */
const $ = id => document.getElementById(id);
const STEPS = ["Image upload", "Image quality analysis", "Adaptive preprocessing & noise removal",
  "Handwritten text detection", "Text-line / word segmentation", "Automatic language identification",
  "Multilingual AI recognition (Tamil & English)", "Confidence score & uncertain-word detection",
  "AI-based spell correction", "Editable text + original/processed comparison", "Translation (selected language)", "Export as TXT / PDF / Word"];
const TAMIL = /[\u0B80-\u0BFF]/g, LATIN = /[A-Za-z]/g;
let img = null, lines = [];

$("steps").innerHTML = STEPS.map(s => `<li>${s}</li>`).join("");
const mark = (i, c) => { $("steps").children[i].className = c; };

/* 1 · Upload */
$("file").onchange = e => {
  const f = e.target.files[0]; if (!f) return;
  img = new Image();
  img.onload = () => { $("orig").src = img.src; $("run").disabled = false; reset(); mark(0, "done"); };
  img.src = URL.createObjectURL(f);
};
function reset() {
  [...$("steps").children].forEach(li => li.className = "");
  $("editor").innerHTML = ""; $("out").innerHTML = ""; $("trinfo").textContent = ""; $("quality").textContent = ""; $("langinfo").textContent = ""; $("bar").value = 0;
  ["spell", "tr", "txt", "doc", "pdf"].forEach(b => $(b).disabled = true);
}
$("cut").oninput = e => $("proc").style.clipPath = `inset(0 0 0 ${e.target.value}%)`;
$("thr").oninput = e => { $("tv").textContent = e.target.value; flag(); };
const tick = () => new Promise(r => setTimeout(r, 20)); // let the UI repaint

/* 2 · Quality analysis */
function quality(g, w, h) {
  let sum = 0, sq = 0; const n = g.length;
  for (let i = 0; i < n; i++) { sum += g[i]; sq += g[i] * g[i]; }
  const mean = sum / n, std = Math.sqrt(sq / n - mean * mean);
  let lap = 0, c = 0;
  for (let y = 1; y < h - 1; y += 2) for (let x = 1; x < w - 1; x += 2) {
    const i = y * w + x, v = 4 * g[i] - g[i - 1] - g[i + 1] - g[i - w] - g[i + w]; lap += v * v; c++;
  }
  const sharp = lap / c, issues = [];
  if (mean < 90) issues.push("image is dark"); if (mean > 225 && std < 25) issues.push("very faint writing");
  if (std < 35) issues.push("low contrast"); if (sharp < 60) issues.push("blurry");
  if (Math.max(img.width, img.height) < 800) issues.push("low resolution");
  return { mean, std, sharp, issues, low: std < 35 || sharp < 60 };
}

/* 3 · Adaptive preprocessing */
function median3(g, w, h) {
  const o = new Float32Array(g.length), a = new Array(9);
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    let k = 0; for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) a[k++] = g[(y + dy) * w + x + dx];
    a.sort((p, q) => p - q); o[y * w + x] = a[4];
  }
  return o;
}
function binarize(g, w, h, c) { // adaptive (local-mean) threshold using an integral image
  const W = w + 1, I = new Float64Array(W * (h + 1)), ink = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) { let row = 0;
    for (let x = 0; x < w; x++) { row += g[y * w + x]; I[(y + 1) * W + x + 1] = I[y * W + x + 1] + row; } }
  const r = Math.max(15, w >> 5);
  for (let y = 0; y < h; y++) { const y0 = Math.max(0, y - r), y1 = Math.min(h, y + r + 1);
    for (let x = 0; x < w; x++) { const x0 = Math.max(0, x - r), x1 = Math.min(w, x + r + 1);
      const m = (I[y1 * W + x1] - I[y0 * W + x1] - I[y1 * W + x0] + I[y0 * W + x0]) / ((x1 - x0) * (y1 - y0));
      ink[y * w + x] = g[y * w + x] < m * (1 - c) ? 1 : 0; } }
  return ink;
}

function flatten(g, w, h) { // illumination correction: divide by a heavily blurred background (removes shadows / uneven light)
  const W = w + 1, I = new Float64Array(W * (h + 1)), o = new Float32Array(g.length), r = Math.max(25, w >> 4);
  for (let y = 0; y < h; y++) { let row = 0;
    for (let x = 0; x < w; x++) { row += g[y * w + x]; I[(y + 1) * W + x + 1] = I[y * W + x + 1] + row; } }
  for (let y = 0; y < h; y++) { const y0 = Math.max(0, y - r), y1 = Math.min(h, y + r + 1);
    for (let x = 0; x < w; x++) { const x0 = Math.max(0, x - r), x1 = Math.min(w, x + r + 1);
      const m = (I[y1 * W + x1] - I[y0 * W + x1] - I[y1 * W + x0] + I[y0 * W + x0]) / ((x1 - x0) * (y1 - y0));
      o[y * w + x] = Math.min(255, g[y * w + x] / Math.max(1, m) * 235); } }
  return o;
}
function despeckle(ink, w, h, minA) { // remove tiny isolated ink blobs (noise specks) via connected components
  const seen = new Uint8Array(w * h), q = new Int32Array(w * h); let removed = 0;
  for (let i = 0; i < ink.length; i++) {
    if (!ink[i] || seen[i]) continue;
    let head = 0, tail = 0; q[tail++] = i; seen[i] = 1;
    while (head < tail) { const p = q[head++], x = p % w, y = (p / w) | 0;
      for (let dy = -1; dy <= 1; dy++) { const yy = y + dy; if (yy < 0 || yy >= h) continue;
        for (let dx = -1; dx <= 1; dx++) { const xx = x + dx; if (xx < 0 || xx >= w) continue;
          const n = yy * w + xx; if (ink[n] && !seen[n]) { seen[n] = 1; q[tail++] = n; } } } }
    if (tail < minA) { for (let k = 0; k < tail; k++) ink[q[k]] = 0; removed++; }
  }
  return removed;
}
function dropRules(ink, w, h) { // remove long horizontal lines (notebook rules, underlines) of any thickness; keep strokes that cross them
  const minL = Math.round(w * 0.08), mask = new Uint8Array(w * h); let n = 0;
  for (let y = 0; y < h; y++) { let x = 0;
    while (x < w) {
      if (!ink[y * w + x]) { x++; continue; }
      const s = x; let e = x, gap = 0;
      while (x < w && gap <= 3) { if (ink[y * w + x]) { e = x; gap = 0; } else gap++; x++; }
      if (e - s + 1 >= minL) for (let k = s; k <= e; k++) mask[y * w + k] = 1;
    } }
  for (let k = 0; k < w; k++) { let y = 0;
    while (y < h) {
      if (!mask[y * w + k]) { y++; continue; }
      const ya = y; while (y < h && mask[y * w + k]) y++;
      const yb = y - 1, crossing = ya >= 3 && yb + 3 < h && ink[(ya - 3) * w + k] && ink[(yb + 3) * w + k];
      if (!crossing) for (let r = ya; r <= yb; r++) { if (ink[r * w + k]) n++; ink[r * w + k] = 0; }
    } }
  return n;
}
function clearBorder(ink, w, h) { // dark photo edges / shadows at the frame are not handwriting
  const m = Math.round(Math.min(w, h) * 0.015);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (x < m || y < m || x >= w - m || y >= h - m) ink[y * w + x] = 0;
}
function estimateSkew(g, w, h) { // angle (degrees) to rotate clockwise so text lines become horizontal; 0 if unsure
  const f = Math.max(1, Math.round(Math.max(w, h) / 450)), sw = Math.floor(w / f), sh = Math.floor(h / f), sg = new Float32Array(sw * sh);
  for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) { let s = 0;
    for (let dy = 0; dy < f; dy++) for (let dx = 0; dx < f; dx++) s += g[(y * f + dy) * w + x * f + dx]; sg[y * sw + x] = s / (f * f); }
  const ink = binarize(sg, sw, sh, 0.15), pts = [];
  for (let i = 0; i < ink.length; i++) if (ink[i]) pts.push(i % sw, (i / sw) | 0);
  if (pts.length < 400) return 0;
  const bins = new Float32Array(2 * sw + sh + 8);
  const score = a => { const r = a * Math.PI / 180, c = Math.cos(r), s = Math.sin(r); bins.fill(0);
    for (let i = 0; i < pts.length; i += 2) bins[Math.round(pts[i] * s + pts[i + 1] * c + sw)]++;
    let t = 0; for (const v of bins) t += v * v; return t; };
  let best = 0, bs = -1;
  for (let a = -10; a <= 10; a += 1) { const v = score(a); if (v > bs) { bs = v; best = a; } }
  const c0 = best; for (let a = c0 - 1; a <= c0 + 1; a += 0.2) { const v = score(a); if (v > bs) { bs = v; best = a; } }
  return best !== 0 && bs < score(0) * 1.02 ? 0 : best;
}

/* 4 + 5 · Detection & line / word segmentation (projection profile + line-pitch autocorrelation) */
function segment(ink, w, h) {
  const m = Math.round(w * 0.03), rows = new Float32Array(h), sm = new Float32Array(h);
  for (let y = 0; y < h; y++) { let s = 0; for (let x = m; x < w - m; x++) s += ink[y * w + x]; rows[y] = s; }
  for (let y = 0; y < h; y++) { let s = 0, c = 0; for (let j = Math.max(0, y - 4); j <= Math.min(h - 1, y + 4); j++) { s += rows[j]; c++; } sm[y] = s / c; }
  let mx = 0; for (const v of sm) if (v > mx) mx = v;
  const th = Math.max(1, mx * 0.08); let top = 0, bot = h - 1;
  while (top < h && sm[top] < th) top++; while (bot > top && sm[bot] < th) bot--;
  if (bot - top < 20) return [];
  let mean = 0; for (let y = top; y <= bot; y++) mean += sm[y]; mean /= bot - top + 1;
  const maxLag = Math.min(250, ((bot - top) / 2) | 0), c = new Float64Array(maxLag + 2); let cm = 0;
  for (let l = 20; l <= maxLag; l++) { let s = 0; for (let y = top; y + l <= bot; y++) s += (sm[y] - mean) * (sm[y + l] - mean); c[l] = s; if (s > cm) cm = s; }
  let P = 0; for (let l = 21; l < maxLag; l++) if (c[l] > 0.6 * cm && c[l] >= c[l - 1] && c[l] >= c[l + 1]) { P = l; break; }
  const bands = [];
  if (!P) bands.push([top, bot]);
  else { let pos = top; // cut at the valley of the profile about one line-pitch below the previous line start
    while (pos < bot) {
      while (pos < bot && sm[pos] < th) pos++;
      if (pos + P * 1.5 >= bot) { bands.push([pos, bot]); break; }
      let mi = Math.round(pos + P * 0.6); const b = Math.min(bot, Math.round(pos + P * 1.4));
      for (let y = mi; y <= b; y++) if (sm[y] < sm[mi]) mi = y;
      bands.push([pos, mi - 1]); pos = mi;
    } }
  const out = [];
  for (const [y0, y1] of bands) {
    if (y1 - y0 < 14) continue;
    const cols = new Uint32Array(w);
    for (let y = y0; y <= y1; y++) for (let x = m; x < w - m; x++) cols[x] += ink[y * w + x];
    const g = Math.max(12, (y1 - y0) * 0.35), words = []; let ws = -1, gp = 0;
    for (let x = 0; x <= w; x++) {
      if (x < w && cols[x] > 0) { if (ws < 0) ws = x; gp = 0; }
      else if (ws >= 0 && (++gp > g || x === w)) { if (x - gp - ws > 4) words.push([ws, x - gp]); ws = -1; }
    }
    if (words.length) out.push({ y0, y1, x0: words[0][0], x1: words.at(-1)[1], words });
  }
  return out;
}

/* 6/7 · OCR helper (one Tesseract worker, line-by-line) */
async function recognise(langs, crops, onProg, psm = "7") {
  const wk = await Tesseract.createWorker(langs, 1);
  await wk.setParameters({ tessedit_pageseg_mode: psm });
  const res = [];
  for (let i = 0; i < crops.length; i++) { res.push((await wk.recognize(crops[i])).data); onProg(i + 1, crops.length); }
  await wk.terminate(); return res;
}
function cropLine(src, L, w, h) {
  const p = 12, x0 = Math.max(0, L.x0 - p), y0 = Math.max(0, L.y0 - p), cw = Math.min(w, L.x1 + p) - x0, ch = Math.min(h, L.y1 + p) - y0;
  const s = 96 / ch, c = document.createElement("canvas"); c.width = Math.max(1, cw * s); c.height = 96;
  const t = c.getContext("2d"); t.fillStyle = "#fff"; t.fillRect(0, 0, c.width, c.height);
  t.drawImage(src, x0, y0, cw, ch, 0, 0, c.width, c.height); return c;
}

const mk = x => ({ t: x.text.trim(), c: x.confidence });
const lineWords = d => (d.words && d.words.length ? d.words : (d.text || "").split(/\s+/).filter(Boolean).map(t => ({ text: t, confidence: d.confidence })))
  .filter(x => x.text.trim()).map(mk);
const pageLines = d => d.lines && d.lines.length ? d.lines.map(l => l.words.filter(x => x.text.trim()).map(mk)).filter(l => l.length)
  : (d.text || "").split("\n").map(s => s.trim()).filter(Boolean).map(s => s.split(/\s+/).map(t => ({ t, c: d.confidence })));

/* Optional AI engine: Google Gemini (user pastes own free API key; stays in this browser) */
const MODELS = ["gemini-flash-latest", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite"]; // tried in order
const gkey = () => $("gkey").value.trim();
try { $("gkey").value = localStorage.getItem("gkey") || ""; } catch (e) {}
$("gkey").oninput = () => { try { localStorage.setItem("gkey", gkey()); } catch (e) {} };
async function gemini(key, parts, json) {
  let err = "";
  for (const m of MODELS) {
    const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${m}:generateContent`, {
      method: "POST", headers: { "Content-Type": "application/json", "x-goog-api-key": key },
      body: JSON.stringify({ contents: [{ parts }], generationConfig: { temperature: 0, ...(json ? { responseMimeType: "application/json" } : {}) } }) });
    const j = await r.json(), msg = (j.error && j.error.message) || "";
    if (r.status === 404 || (r.status === 400 && /models\//.test(msg))) { err = msg; continue; } // model not available -> next
    if (!r.ok) throw new Error(msg || r.statusText);
    return j.candidates[0].content.parts.map(p => p.text || "").join("");
  }
  throw new Error("No Gemini model available: " + err);
}
async function geminiRead(cv, key) {
  const prompt = 'Transcribe ALL handwritten text in this image exactly as written, line by line, in reading order. The text may be Tamil, English or both: keep Tamil in Tamil script and English in English. Do not translate, do not correct spelling, ignore watermarks and printed marks. Reply ONLY with JSON: {"lines":[{"text":"...","unsure":["words you are not sure about"]}]}';
  const out = await gemini(key, [{ text: prompt }, { inlineData: { mimeType: "image/jpeg", data: cv.toDataURL("image/jpeg", 0.9).split(",")[1] } }], true);
  const js = JSON.parse(out.replace(/```json|```/g, "").trim());
  return js.lines.map(l => { const un = new Set((l.unsure || []).map(s => String(s).toLowerCase()));
    return String(l.text).split(/\s+/).filter(Boolean).map(t => ({ t, c: un.has(t.toLowerCase().replace(/[.,!?;:]+$/, "")) ? 45 : 95 })); }).filter(l => l.length);
}

/* MAIN PIPELINE */
$("run").onclick = async () => {
  $("run").disabled = true; reset(); mark(0, "done");
  try {
    mark(1, "run"); await tick();
    const s = Math.min(2.5, Math.max(0.5, 1800 / Math.max(img.width, img.height)));
    const w = Math.round(img.width * s), h = Math.round(img.height * s);
    const cv = document.createElement("canvas"); cv.width = w; cv.height = h;
    const cx = cv.getContext("2d"); cx.drawImage(img, 0, 0, w, h);
    const gray = new Float32Array(w * h);
    const toGray = () => { const px = cx.getImageData(0, 0, w, h).data;
      for (let i = 0; i < gray.length; i++) gray[i] = 0.299 * px[4 * i] + 0.587 * px[4 * i + 1] + 0.114 * px[4 * i + 2]; };
    toGray();
    const q = quality(gray, w, h);
    let skew = estimateSkew(gray, w, h); // straighten tilted photos
    if (Math.abs(skew) >= 0.5) {
      const tmp = document.createElement("canvas"); tmp.width = w; tmp.height = h; tmp.getContext("2d").drawImage(cv, 0, 0);
      cx.save(); cx.fillStyle = `rgb(${q.mean | 0},${q.mean | 0},${q.mean | 0})`; cx.fillRect(0, 0, w, h);
      cx.translate(w / 2, h / 2); cx.rotate(skew * Math.PI / 180); cx.drawImage(tmp, -w / 2, -h / 2); cx.restore();
      toGray(); $("orig").src = cv.toDataURL("image/jpeg", 0.92); // keep original and cleaned image aligned in the slider
    } else skew = 0;
    $("quality").innerHTML = `<b>Quality:</b> brightness ${q.mean | 0}, contrast ${q.std | 0}, sharpness ${q.sharp | 0}<br>` +
      (q.issues.length ? "⚠ " + q.issues.join(", ") + " – settings adjusted automatically." : "✔ Image quality looks good.");
    mark(1, "done"); mark(2, "run"); await tick();

    const clean = flatten(median3(gray, w, h), w, h);
    if (q.std < 45) { // contrast stretch for faint images
      let lo = 255, hi = 0; for (const v of clean) { if (v < lo) lo = v; if (v > hi) hi = v; }
      for (let i = 0; i < clean.length; i++) clean[i] = (clean[i] - lo) * 255 / Math.max(1, hi - lo);
    }
    const ink = binarize(clean, w, h, q.low ? 0.07 : 0.14);
    clearBorder(ink, w, h);
    const rules = dropRules(ink, w, h), specks = despeckle(ink, w, h, Math.max(6, Math.round(w * h / 250000)));
    $("quality").innerHTML += `<br><b>Cleaning:</b> ${skew ? "straightened by " + skew.toFixed(1) + "°, " : ""}shadows flattened, ${specks} noise specks &amp; ${rules} ruled-line pixels removed.`;
    const pc = $("proc"); pc.width = w; pc.height = h;
    const pctx = pc.getContext("2d"), id = pctx.createImageData(w, h);
    for (let i = 0; i < ink.length; i++) { const v = ink[i] ? 0 : 255; id.data.set([v, v, v, 255], 4 * i); }
    pctx.putImageData(id, 0, 0); const bin = document.createElement("canvas"); bin.width = w; bin.height = h;
    bin.getContext("2d").putImageData(id, 0, 0);
    mark(2, "done"); mark(3, "run"); await tick();

    const segs = segment(ink, w, h); mark(3, "done"); mark(4, "run"); await tick();
    if (!segs.length) throw new Error("No handwritten text detected. Try a clearer photo.");
    pctx.lineWidth = 2;
    segs.forEach(L => { pctx.strokeStyle = "#e11"; pctx.strokeRect(L.x0, L.y0, L.x1 - L.x0, L.y1 - L.y0);
      pctx.strokeStyle = "#1a6"; L.words.forEach(([a, b]) => pctx.strokeRect(a, L.y0, b - a, L.y1 - L.y0)); });
    const crops = segs.map(L => cropLine(bin, L, w, h));
    mark(4, "done"); mark(5, "run"); await tick();

    const key = gkey();
    if (key) { // Gemini AI engine: reads Tamil + English handwriting, flags unsure words
      $("langinfo").textContent = "Reading handwriting with Gemini AI…"; $("bar").value = 40; mark(5, "done"); mark(6, "run");
      lines = await geminiRead(cv, key);
      const all = lines.flat().map(w => w.t).join(""), ta = (all.match(TAMIL) || []).length, la = (all.match(LATIN) || []).length, sh = ta / Math.max(1, ta + la);
      $("langinfo").innerHTML = `<b>Detected language:</b> ${sh > 0.7 ? "Tamil" : sh < 0.1 ? "English" : "Tamil + English (mixed)"} · engine: Gemini AI<br>Lines: ${lines.length} · words: ${lines.flat().length}`;
      $("bar").value = 100; mark(6, "done"); mark(7, "run"); await tick();
    } else {
      // Language identification: quick pre-pass on up to 3 lines
      const sample = crops.slice(0, 3);
      const pre = await recognise("eng+tam", sample, (i, n) => $("bar").value = 10 * i / n);
      const txt = pre.map(d => d.text).join(""), ta = (txt.match(TAMIL) || []).length, la = (txt.match(LATIN) || []).length;
      const share = ta / Math.max(1, ta + la), langs = txt.trim() && share < 0.1 ? "eng" : "tam+eng";
      $("langinfo").innerHTML = `<b>Detected language:</b> ${share > 0.7 ? "Tamil" : share < 0.1 ? "English" : "Tamil + English (mixed)"} (Tamil share ${(share * 100) | 0}%)`;
      mark(5, "done"); mark(6, "run");

      const data = await recognise(langs, crops, (i, n) => $("bar").value = 10 + 90 * i / n);
      lines = data.map(lineWords).filter(l => l.length);
      if (!lines.length) { // line-by-line gave nothing -> let Tesseract read the whole page
        const d = await recognise(langs, [bin], () => $("bar").value = 95, "6"); lines = pageLines(d[0]);
      }
      if (!lines.length) throw new Error("Nothing could be recognised. Try a clearer, darker photo.");
      $("langinfo").innerHTML += `<br>Lines detected: ${segs.length} · words recognised: ${lines.flat().length}`;
      mark(6, "done"); mark(7, "run"); await tick();
    }
    renderEditor(); flag(); mark(7, "done"); mark(9, "done");
    ["spell", "tr", "txt", "doc", "pdf"].forEach(b => $(b).disabled = false);
    await translateAll(); // translate the full page automatically
  } catch (e) {
    const msg = e instanceof TypeError ? "Network error / request blocked (" + e.message + "). Check internet, ad-blocker or VPN." : e.message;
    $("langinfo").innerHTML = `<b style="color:#e55">Error:</b> ${msg}`; alert(msg);
  }
  $("run").disabled = false;
};

/* 8 · Confidence / uncertain-word highlighting */
function renderEditor() {
  const ed = $("editor"); ed.innerHTML = "";
  lines.forEach(L => { const d = document.createElement("div");
    L.forEach((w, i) => { const sp = document.createElement("span"); sp.textContent = w.t; sp.title = `confidence ${w.c | 0}%`; w.el = sp;
      d.append(sp, i < L.length - 1 ? " " : ""); });
    ed.append(d); });
}
function flag() { const t = +$("thr").value; lines.flat().forEach(w => w.el && w.el.classList.toggle("low", w.c < t && !w.el.classList.contains("fix"))); }

/* 9 · AI spell correction (English words only, via LanguageTool; only uncertain words are changed) */
$("spell").onclick = async () => {
  mark(8, "run"); $("spell").disabled = true; const t = +$("thr").value; let n = 0;
  try {
    for (const L of lines) {
      const idx = L.map((w, i) => i).filter(i => /[A-Za-z]/.test(L[i].t) && !/[\u0B80-\u0BFF]/.test(L[i].t));
      if (!idx.length) continue;
      const offs = []; let pos = 0; const text = idx.map(i => { offs.push(pos); pos += L[i].t.length + 1; return L[i].t; }).join(" ");
      const r = await fetch("https://api.languagetool.org/v2/check", { method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" }, body: new URLSearchParams({ text, language: "en-US" }) });
      const j = await r.json();
      for (const m of j.matches) {
        if (!m.replacements.length) continue;
        let k = offs.findLastIndex(o => o <= m.offset), w = L[idx[k]];
        if (w && w.c < t) { w.t = m.replacements[0].value; w.el.textContent = w.t; w.el.className = "fix"; n++; }
      }
    }
    alert(n ? `${n} uncertain word(s) corrected.` : "No corrections needed. (Tamil spelling correction is not available.)");
  } catch (e) { alert("Spell-check service not reachable: " + e.message); }
  mark(8, "done"); $("spell").disabled = false; flag();
};

/* 11 · Translation (MyMemory free API, line by line; source language detected per line by script) */
async function translateAll() {
  const tgt = $("tl").value, out = $("out"); out.innerHTML = ""; mark(10, "run"); $("tr").disabled = true;
  const rows = [...$("editor").children].map(d => d.innerText.trim()).filter(Boolean), chunks = [];
  for (const r of rows) { // join neighbouring lines of the same script into chunks (<450 chars) -> whole page, better context
    const src = /[\u0B80-\u0BFF]/.test(r) ? "ta" : "en", last = chunks.at(-1);
    if (last && last.src === src && last.text.length + r.length < 450) last.text += " " + r; else chunks.push({ src, text: r.slice(0, 480) });
  }
  let fail = 0;
  if (gkey() && rows.length) { // Gemini: translate the whole page in one go (understands context, fixes small OCR errors)
    try {
      const tr = await gemini(gkey(), [{ text: `Translate the following text into ${$("tl").selectedOptions[0].text}. It was read from handwriting and may contain small recognition errors, so infer the intended meaning. Keep the line/paragraph structure. Output ONLY the translation.\n\n${rows.join("\n")}` }]);
      tr.split("\n").filter(s => s.trim()).forEach(s => { const p = document.createElement("p"); p.textContent = s; out.append(p); });
      $("trinfo").textContent = "(full page translated by Gemini AI)"; mark(10, "done"); $("tr").disabled = false; return;
    } catch (e) { fail = 1; out.innerHTML = ""; $("trinfo").textContent = "(Gemini failed: " + e.message + " – using free translator)"; }
  }
  for (const ch of chunks) {
    const p = document.createElement("p"); out.append(p);
    if (ch.src === tgt) { p.textContent = ch.text; continue; }
    try {
      const r = await fetch(`https://api.mymemory.translated.net/get?q=${encodeURIComponent(ch.text)}&langpair=${ch.src}|${tgt}`);
      const j = await r.json();
      if (j.responseStatus != 200) throw new Error(j.responseDetails);
      p.textContent = j.responseData.translatedText;
    } catch (e) { p.textContent = "⚠ " + ch.text; fail++; }
  }
  if (!($("trinfo").textContent.startsWith("(Gemini failed"))) $("trinfo").textContent = fail ? `(${fail} part(s) failed – daily free limit or network)` : "(full page translated)";
  mark(10, "done"); $("tr").disabled = false;
}
$("tr").onclick = translateAll;
$("tl").onchange = () => { if (!$("tr").disabled) translateAll(); };

/* 12 · Export */
const text = () => [...$("editor").children].map(d => d.innerText.trim()).join("\n") +
  ($("out").innerText.trim() ? "\n\n--- Translation (" + $("tl").value + ") ---\n" + $("out").innerText.trim() : "");
const save = (name, blob) => { const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name; a.click(); mark(11, "done"); };
$("txt").onclick = () => save("recognised.txt", new Blob(["\ufeff" + text()], { type: "text/plain;charset=utf-8" }));
const html = () => `<html><head><meta charset="utf-8"><style>body{font:14pt 'Latha','Nirmala UI',Arial}</style></head><body>${
  text().split("\n").map(l => `<p>${l.replace(/&/g, "&amp;").replace(/</g, "&lt;")}</p>`).join("")}</body></html>`;
$("doc").onclick = () => save("recognised.doc", new Blob(["\ufeff" + html()], { type: "application/msword" }));
$("pdf").onclick = () => { const w = window.open("", "_blank"); w.document.write(html()); w.document.close(); w.focus();
  setTimeout(() => w.print(), 300); mark(11, "done"); }; // choose "Save as PDF" – keeps Tamil shaping correct
