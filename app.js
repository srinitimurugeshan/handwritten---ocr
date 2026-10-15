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

/* 4 + 5 · Detection & line / word segmentation (projection profiles) */
function segment(ink, w, h) {
  const rows = new Uint32Array(h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) rows[y] += ink[y * w + x];
  const thr = Math.max(2, w * 0.004), found = []; let s = -1, gap = 0;
  for (let y = 0; y <= h; y++) {
    if (y < h && rows[y] > thr) { if (s < 0) s = y; gap = 0; }
    else if (s >= 0 && (++gap > 8 || y === h)) { found.push([s, y - gap]); s = -1; }
  }
  const out = [];
  for (const [y0, y1] of found) {
    if (y1 - y0 < 14) continue;
    const cols = new Uint32Array(w);
    for (let y = y0; y <= y1; y++) for (let x = 0; x < w; x++) cols[x] += ink[y * w + x];
    const g = Math.max(12, (y1 - y0) * 0.5), words = []; let ws = -1, gp = 0;
    for (let x = 0; x <= w; x++) {
      if (x < w && cols[x] > 0) { if (ws < 0) ws = x; gp = 0; }
      else if (ws >= 0 && (++gp > g || x === w)) { if (x - gp - ws > 4) words.push([ws, x - gp]); ws = -1; }
    }
    if (words.length) out.push({ y0, y1, x0: words[0][0], x1: words.at(-1)[1], words });
  }
  return out;
}

/* 6/7 · OCR helper (one Tesseract worker, line-by-line) */
async function recognise(langs, crops, onProg) {
  const wk = await Tesseract.createWorker(langs, 1);
  await wk.setParameters({ tessedit_pageseg_mode: "7" });
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

/* MAIN PIPELINE */
$("run").onclick = async () => {
  $("run").disabled = true; reset(); mark(0, "done");
  try {
    mark(1, "run"); await tick();
    const s = Math.min(2.5, Math.max(0.5, 1800 / Math.max(img.width, img.height)));
    const w = Math.round(img.width * s), h = Math.round(img.height * s);
    const cv = document.createElement("canvas"); cv.width = w; cv.height = h;
    const cx = cv.getContext("2d"); cx.drawImage(img, 0, 0, w, h);
    const px = cx.getImageData(0, 0, w, h).data, gray = new Float32Array(w * h);
    for (let i = 0; i < gray.length; i++) gray[i] = 0.299 * px[4 * i] + 0.587 * px[4 * i + 1] + 0.114 * px[4 * i + 2];
    const q = quality(gray, w, h);
    $("quality").innerHTML = `<b>Quality:</b> brightness ${q.mean | 0}, contrast ${q.std | 0}, sharpness ${q.sharp | 0}<br>` +
      (q.issues.length ? "⚠ " + q.issues.join(", ") + " – settings adjusted automatically." : "✔ Image quality looks good.");
    mark(1, "done"); mark(2, "run"); await tick();

    const clean = median3(gray, w, h);
    if (q.std < 45) { // contrast stretch for faint images
      let lo = 255, hi = 0; for (const v of clean) { if (v < lo) lo = v; if (v > hi) hi = v; }
      for (let i = 0; i < clean.length; i++) clean[i] = (clean[i] - lo) * 255 / Math.max(1, hi - lo);
    }
    const ink = binarize(clean, w, h, q.low ? 0.07 : 0.14);
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

    // Language identification: quick pre-pass on up to 3 lines
    const sample = crops.slice(0, 3);
    const pre = await recognise("eng+tam", sample, (i, n) => $("bar").value = 10 * i / n);
    const txt = pre.map(d => d.text).join(""), ta = (txt.match(TAMIL) || []).length, la = (txt.match(LATIN) || []).length;
    const share = ta / Math.max(1, ta + la), langs = share < 0.1 ? "eng" : "tam+eng";
    $("langinfo").innerHTML = `<b>Detected language:</b> ${share > 0.7 ? "Tamil" : share < 0.1 ? "English" : "Tamil + English (mixed)"} (Tamil share ${(share * 100) | 0}%)`;
    mark(5, "done"); mark(6, "run");

    const data = await recognise(langs, crops, (i, n) => $("bar").value = 10 + 90 * i / n);
    lines = data.map(d => (d.words || []).filter(x => x.text.trim()).map(x => ({ t: x.text.trim(), c: x.confidence })));
    mark(6, "done"); mark(7, "run"); await tick();
    renderEditor(); flag(); mark(7, "done"); mark(9, "done");
    ["spell", "tr", "txt", "doc", "pdf"].forEach(b => $(b).disabled = false);
  } catch (e) { alert(e.message); }
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
$("tr").onclick = async () => {
  const tgt = $("tl").value, out = $("out"); out.innerHTML = ""; mark(10, "run"); $("tr").disabled = true;
  const rows = [...$("editor").children].map(d => d.innerText.trim()).filter(Boolean);
  let fail = 0;
  for (const line of rows) {
    const src = /[\u0B80-\u0BFF]/.test(line) ? "ta" : "en", p = document.createElement("div");
    out.append(p);
    if (src === tgt) { p.textContent = line; continue; }
    try {
      const r = await fetch(`https://api.mymemory.translated.net/get?q=${encodeURIComponent(line.slice(0, 480))}&langpair=${src}|${tgt}`);
      const j = await r.json();
      if (j.responseStatus != 200) throw new Error(j.responseDetails);
      p.textContent = j.responseData.translatedText;
    } catch (e) { p.textContent = "⚠ " + line; fail++; }
  }
  $("trinfo").textContent = fail ? `(${fail} line(s) failed – daily free limit or network)` : "(translated)";
  mark(10, "done"); $("tr").disabled = false;
};

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
