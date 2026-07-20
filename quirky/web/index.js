// ---- Modality config ----
const MODS = {
  image:  { title: "Image",  hint: "Drop an AI image. Diagnose to see WHERE it reads as synthetic, then apply only the fixes you want.", accept: "image/*", exts: "PNG · JPG · WEBP", sliders: true, outName: "out.png" },
  video:  { title: "Video",  hint: "Drop an AI video. Quirky adds hand-held drift and fixes robotic linear motion.", accept: "video/mp4,video/x-msvideo,video/quicktime", exts: "MP4 · MOV · AVI", sliders: false, outName: "out.mp4" },
  speech: { title: "Speech", hint: "Drop synthetic speech (WAV). Quirky adds jitter, intonation, breath and pauses.", accept: "audio/wav", exts: "WAV", sliders: false, outName: "out.wav" },
  text:   { title: "Text",   hint: "Paste or drop AI text. Quirky varies rhythm, kills tropes, and strips em-dashes.", accept: ".txt,.md", exts: "TXT · MD", sliders: false, text: true, outName: "out.txt" },
};

let mod = "image";
let file = null;
let selectedFixes = null;   // null = apply all; Set = only these
let lastDiagnosis = null;
let lastSrcBlob = null;     // original input, kept in memory for Re-map
let lastOutBlob = null;     // last humanize output, kept in memory for Re-map

const $ = (id) => document.getElementById(id);
const show = (el, on) => { el.hidden = !on; };

// ---- Modality switching ----
document.querySelectorAll(".mtab").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".mtab").forEach((x) => x.classList.remove("active"));
  b.classList.add("active");
  setMod(b.dataset.mod);
}));

function setMod(m) {
  mod = m;
  const c = MODS[m];
  $("mod-title").textContent = c.title;
  $("mod-hint").textContent = c.hint;
  $("accept-hint").textContent = c.exts;
  $("file").accept = c.accept;
  document.querySelectorAll(".img-only").forEach((el) => { el.style.display = c.sliders ? "" : "none"; });
  show($("text-in"), !!c.text);
  show($("drop"), !c.text);
  reset();
}

function reset() {
  file = null;
  selectedFixes = null;
  lastDiagnosis = null;
  lastSrcBlob = null;
  lastOutBlob = null;
  show($("chip"), false);
  $("text-in").value = "";
  show($("result"), false); show($("busy"), false); show($("diag"), false);
  show($("clean-panel"), false); show($("remap-block"), false); show($("remap-result"), false);
  show($("empty"), true);
  updateGo();
}

// ---- File input ----
const drop = $("drop");
drop.addEventListener("click", () => $("file").click());
$("file").addEventListener("change", (e) => setFile(e.target.files[0]));
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); setFile(e.dataTransfer.files[0]); });
$("chip-x").addEventListener("click", reset);
$("text-in").addEventListener("input", updateGo);

function setFile(f) {
  if (!f) return;
  file = f;
  selectedFixes = null;
  lastDiagnosis = null;
  lastSrcBlob = null;
  lastOutBlob = null;
  $("chip-name").textContent = f.name;
  show($("chip"), true);
  updateGo();
}

function updateGo() {
  const ready = MODS[mod].text ? $("text-in").value.trim().length > 0 : !!file;
  $("go").disabled = !ready;
  const dg = $("diagnose");
  if (dg) dg.disabled = !(ready && mod === "image");
  const cl = $("clean");
  if (cl) cl.disabled = !(ready && mod === "image");
}

// ---- Sliders ----
[["intensity", "%"], ["gamma", "%"], ["delta", "%"]].forEach(([id]) => {
  $(id).addEventListener("input", (e) => { $(id + "-v").textContent = e.target.value + "%"; });
});

// ---- Buttons ----
$("go").addEventListener("click", run);
$("diagnose").addEventListener("click", diagnose);
$("go2").addEventListener("click", run);
$("clean").addEventListener("click", cleanMeta);
$("remap-btn").addEventListener("click", remapCompare);

function inputBlob() {
  if (MODS[mod].text) return new File([$("text-in").value], "input.txt", { type: "text/plain" });
  return file;
}

async function detect(blob, name) {
  const fd = new FormData(); fd.append("file", blob, name);
  const r = await fetch("/api/detect", { method: "POST", body: fd });
  return (await r.json()).metadata || {};
}

// ---- Diagnose (Slop X-ray + fix cards + fingerprint) ----
async function diagnose() {
  show($("empty"), false); show($("result"), false); show($("diag"), false); show($("busy"), true);
  $("busy-t").textContent = "Analyzing tells…";
  try {
    const fd = new FormData();
    fd.append("file", file, file.name);
    fd.append("intensity", ($("intensity").value / 100).toFixed(2));
    const r = await fetch("/api/diagnose", { method: "POST", body: fd });
    if (!r.ok) throw new Error("diagnose failed (" + r.status + ")");
    const d = await r.json();
    lastDiagnosis = d;

    $("xray-img").src = d.heatmap;
    const top = d.fingerprint && d.fingerprint.top;
    $("fp-line").innerHTML = top
      ? `<span class="fp-k">Likely source</span> <b>${top.source}</b>
         <span class="fp-c">${Math.round(top.confidence * 100)}% confidence</span>`
      : "";

    renderCards(d);
    show($("busy"), false); show($("diag"), true);
  } catch (err) {
    fail(err);
  }
}

function renderCards(d) {
  const rec = new Set(d.recommended_fixes || []);
  selectedFixes = new Set(rec);   // start with the recommended set checked
  $("cards").innerHTML = (d.defects || []).map((c) => {
    const on = rec.has(c.id) ? "checked" : "";
    return `<label class="card-fix sev-${c.severity}">
      <input type="checkbox" data-fix="${c.id}" ${on}/>
      <div class="cf-body">
        <div class="cf-top"><b>${c.title}</b><span class="cf-sev">${c.severity}</span></div>
        <div class="cf-detail">${c.detail}</div>
        <div class="cf-explain">${c.explains}</div>
      </div>
    </label>`;
  }).join("") || `<p class="cf-none">No strong AI tells detected — this reads close to natural.</p>`;

  $("cards").querySelectorAll("input[type=checkbox]").forEach((cb) => {
    cb.addEventListener("change", () => {
      const id = cb.dataset.fix;
      if (cb.checked) selectedFixes.add(id); else selectedFixes.delete(id);
    });
  });
}

// ---- Clean (metadata scrub) ----
async function cleanMeta() {
  show($("empty"), false); show($("result"), false); show($("diag"), false);
  show($("clean-panel"), false); show($("busy"), true);
  $("busy-t").textContent = "Scanning metadata…";
  try {
    const fd = new FormData();
    fd.append("file", file, file.name);
    fd.append("attribute", $("attribute").checked ? "true" : "false");
    const r = await fetch("/api/clean", { method: "POST", body: fd });
    if (!r.ok) throw new Error("clean failed (" + r.status + ")");
    const d = await r.json();
    renderClean(d);
    show($("busy"), false); show($("clean-panel"), true);
  } catch (err) {
    fail(err);
  }
}

function renderClean(d) {
  $("clean-sub").textContent =
    `${d.before_entries} field(s) found · ${d.before_leaks} generator leak(s) before · ` +
    `${d.after_meaningful} meaningful field(s) remain after`;

  $("clean-entries").innerHTML = (d.removed || []).map((e) => {
    const tag = e.leak ? "leak" : e.structural ? "structural" : "metadata";
    const cls = e.leak ? "sev-high" : e.structural ? "sev-low" : "sev-medium";
    const val = (e.value == null ? "" : String(e.value)).slice(0, 140);
    return `<div class="card-fix ${cls}">
      <div class="cf-body">
        <div class="cf-top"><b>${e.key}</b><span class="cf-sev">${tag}</span></div>
        <div class="cf-detail">${val}</div>
      </div>
    </div>`;
  }).join("") || `<p class="cf-none">No embedded metadata found — already clean.</p>`;

  const ok = d.fully_clean;
  $("clean-result").innerHTML =
    `<span class="audit-k">${ok ? "✓ Fully clean" : "Partially clean"}</span>
     <b>${(d.removed || []).length} field(s) removed</b>${d.attributed ? " · attribution tag added" : ""}`;
  $("clean-result").className = "clean-result " + (ok ? "ok" : "warn");

  const dl = $("clean-download");
  dl.href = d.cleaned_data_uri;
  const stem = (file.name || "cleaned").replace(/\.[^.]+$/, "");
  const ext = (file.name.match(/\.[^.]+$/) || [".png"])[0];
  dl.download = `${stem}.clean${ext}`;
}

// ---- Humanize ----
async function run() {
  show($("empty"), false); show($("result"), false); show($("diag"), false); show($("busy"), true);
  $("busy-t").textContent = "Humanizing…";
  try {
    const src = inputBlob();
    const srcName = MODS[mod].text ? "input.txt" : file.name;

    const fd = new FormData();
    fd.append("file", src, srcName);
    fd.append("intensity", ($("intensity").value / 100).toFixed(2));
    fd.append("gamma", ($("gamma").value / 100).toFixed(2));
    fd.append("delta", ($("delta").value / 100).toFixed(3));
    if (mod === "image" && selectedFixes) fd.append("fixes", Array.from(selectedFixes).join(","));
    if (mod === "image" && $("lock").checked) { fd.append("lock", "true"); fd.append("target", "0.15"); fd.append("min_ssim", "0.86"); }

    const hr = await fetch("/api/humanize", { method: "POST", body: fd });
    if (!hr.ok) throw new Error("humanize failed (" + hr.status + ")");
    const outBlob = await hr.blob();

    const after = await detect(outBlob, MODS[mod].outName);
    await render(src, outBlob, after);

    // External-oracle honesty check (images only): score before/after against a
    // detector separate from the one we optimize.
    if (mod === "image") {
      auditTransfer(src, outBlob).catch(() => {});
      // Re-map ("map it again"): keep both blobs in memory so a click re-diagnoses
      // the output against a fresh diagnosis of the original -- no re-upload needed.
      lastSrcBlob = src; lastOutBlob = outBlob;
      show($("remap-result"), false);
      show($("remap-block"), true);
    } else {
      show($("remap-block"), false);
    }

    show($("busy"), false); show($("result"), true);
  } catch (err) {
    fail(err);
  }
}

async function auditTransfer(src, out) {
  const fd = new FormData();
  fd.append("before", src, "before.png");
  fd.append("after", out, "after.png");
  const r = await fetch("/api/audit", { method: "POST", body: fd });
  if (!r.ok) return;
  const a = await r.json();
  const before = Math.round(a.ai_probability_before * 100);
  const after = Math.round(a.ai_probability_after * 100);
  const drop = Math.round(a.relative_reduction_pct);
  $("audit-line").innerHTML =
    `<span class="audit-k">External detector (${a.oracle})</span>
     <b>${before}% → ${after}%</b> AI-probability <span class="audit-d">(${drop >= 0 ? "−" : "+"}${Math.abs(drop)}%)</span>`;
  show($("audit-line"), true);
}

// ---- Re-map ("map it again") ----
const REMAP_LABELS = {
  clean: "✓ Clean — no flagged defects remain",
  keep_going: "Another pass could help",
  diminishing_returns: "Diminishing returns — another pass is unlikely to help much",
  over_cooked: "⚠ Over-cooked — new defects appeared that weren't in the original",
};

async function remapCompare() {
  if (!lastSrcBlob || !lastOutBlob) return;
  const btn = $("remap-btn");
  btn.disabled = true;
  const label = btn.textContent;
  btn.textContent = "Re-mapping…";
  try {
    const fd = new FormData();
    fd.append("before", lastSrcBlob, "before.png");
    fd.append("after", lastOutBlob, "after.png");
    const r = await fetch("/api/remap", { method: "POST", body: fd });
    if (!r.ok) throw new Error("remap failed (" + r.status + ")");
    renderRemap(await r.json());
  } catch (err) {
    console.error(err);
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
}

function renderRemap(d) {
  $("remap-img").src = d.delta_heatmap;
  $("remap-rec").textContent = `${REMAP_LABELS[d.recommendation] || d.recommendation} · ${d.improvement_pct}% less slop`;
  $("remap-rec").className = "remap-rec " + d.recommendation;

  const chips = (label, ids, cls) => ids.length
    ? `<div class="remap-chips"><span class="remap-chip-label">${label}</span>${
        ids.map((i) => `<span class="remap-chip ${cls}">${i}</span>`).join("")
      }</div>`
    : "";
  $("remap-lists").innerHTML =
    chips("Resolved", d.resolved, "ok") +
    chips("Remaining", d.remaining, "warn") +
    chips("New", d.new, "bad") ||
    `<p class="cf-none">Nothing changed between before and after.</p>`;

  show($("remap-result"), true);
}

function fail(err) {
  show($("busy"), false); show($("diag"), false); show($("empty"), true);
  $("empty").querySelector("h3").textContent = "Something went wrong";
  $("empty").querySelector("p").textContent = String(err.message || err);
}

// ---- Render per modality ----
async function render(src, out, after) {
  ["img-cmp", "video-cmp", "speech-cmp", "text-cmp"].forEach((id) => show($(id), false));
  show($("audit-line"), false);
  const srcURL = URL.createObjectURL(src);
  const outURL = URL.createObjectURL(out);

  if (mod === "image") {
    $("img-b").src = srcURL; $("img-a").src = outURL;
    show($("img-cmp"), true); initSlider();
  } else if (mod === "video") {
    $("vid-b").src = srcURL; $("vid-a").src = outURL; show($("video-cmp"), true);
  } else if (mod === "speech") {
    $("aud-b").src = srcURL; $("aud-a").src = outURL; show($("speech-cmp"), true);
  } else {
    $("txt-b").textContent = $("text-in").value;
    $("txt-a").textContent = await out.text();
    show($("text-cmp"), true);
  }
  metrics(after);
}

const LABELS = {
  ai_score: "AI signature", plastic_score: "Plastic look", texture_score: "Texture richness",
  emotion_score: "Emotion / prosody", channel_corr: "Camera color", repetition_score: "Repetition",
};
const HIGHER_BETTER = new Set(["texture_score", "emotion_score", "channel_corr"]);

function metrics(m) {
  const keys = Object.keys(LABELS).filter((k) => k in m);
  $("mrows").innerHTML = keys.map((k) => {
    const pct = Math.round(Math.max(0, Math.min(1, m[k])) * 100);
    const good = HIGHER_BETTER.has(k) ? pct >= 50 : pct <= 40;
    const col = good ? "var(--ok)" : "var(--bad)";
    return `<div class="mrow"><div class="mrow-top"><b>${LABELS[k]}</b><span>${pct}%</span></div>
      <div class="mtrack"><div class="mfill" style="width:${pct}%;background:${col}"></div></div></div>`;
  }).join("");
}

// ---- Image compare slider ----
function initSlider() {
  const box = $("img-cmp"), after = $("img-after-wrap"), h = $("handle");
  let drag = false;
  const move = (x) => {
    const r = box.getBoundingClientRect();
    let p = ((x - r.left) / r.width) * 100;
    p = Math.max(0, Math.min(100, p));
    after.style.width = p + "%"; h.style.left = p + "%";
    $("img-a").style.width = r.width + "px";
  };
  requestAnimationFrame(() => move(box.getBoundingClientRect().left + box.getBoundingClientRect().width / 2));
  h.onmousedown = () => (drag = true);
  window.onmouseup = () => (drag = false);
  window.onmousemove = (e) => { if (drag) move(e.clientX); };
  box.ontouchmove = (e) => move(e.touches[0].clientX);
}

// init
setMod("image");
