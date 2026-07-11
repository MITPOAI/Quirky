// ---- Modality config ----
const MODS = {
  image:  { title: "Image",  hint: "Drop an AI image. Quirky restores grain, natural spectrum, camera color, and removes spots.", accept: "image/*", exts: "PNG · JPG · WEBP", sliders: true, outName: "out.png" },
  video:  { title: "Video",  hint: "Drop an AI video. Quirky adds hand-held drift and fixes robotic linear motion.", accept: "video/mp4,video/x-msvideo,video/quicktime", exts: "MP4 · MOV · AVI", sliders: false, outName: "out.mp4" },
  speech: { title: "Speech", hint: "Drop synthetic speech (WAV). Quirky adds jitter, intonation, breath and pauses.", accept: "audio/wav", exts: "WAV", sliders: false, outName: "out.wav" },
  text:   { title: "Text",   hint: "Paste or drop AI text. Quirky varies rhythm, kills tropes, and strips em-dashes.", accept: ".txt,.md", exts: "TXT · MD", sliders: false, text: true, outName: "out.txt" },
};

let mod = "image";
let file = null;

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
  show($("chip"), false);
  $("text-in").value = "";
  show($("result"), false); show($("busy"), false); show($("empty"), true);
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
  $("chip-name").textContent = f.name;
  show($("chip"), true);
  updateGo();
}

function updateGo() {
  const ready = MODS[mod].text ? $("text-in").value.trim().length > 0 : !!file;
  $("go").disabled = !ready;
}

// ---- Sliders ----
[["intensity", "%"], ["gamma", "%"], ["delta", "%"]].forEach(([id]) => {
  $(id).addEventListener("input", (e) => { $(id + "-v").textContent = e.target.value + "%"; });
});

// ---- Process ----
$("go").addEventListener("click", run);

function inputBlob() {
  if (MODS[mod].text) return new File([$("text-in").value], "input.txt", { type: "text/plain" });
  return file;
}

async function detect(blob, name) {
  const fd = new FormData(); fd.append("file", blob, name);
  const r = await fetch("/api/detect", { method: "POST", body: fd });
  return (await r.json()).metadata || {};
}

async function run() {
  show($("empty"), false); show($("result"), false); show($("busy"), true);
  $("busy-t").textContent = "Analyzing + humanizing…";
  try {
    const src = inputBlob();
    const srcName = MODS[mod].text ? "input.txt" : file.name;

    const fd = new FormData();
    fd.append("file", src, srcName);
    fd.append("intensity", ($("intensity").value / 100).toFixed(2));
    fd.append("gamma", ($("gamma").value / 100).toFixed(2));
    fd.append("delta", ($("delta").value / 100).toFixed(3));
    const hr = await fetch("/api/humanize", { method: "POST", body: fd });
    if (!hr.ok) throw new Error("humanize failed (" + hr.status + ")");
    const outBlob = await hr.blob();

    const after = await detect(outBlob, MODS[mod].outName);
    await render(src, outBlob, after);

    show($("busy"), false); show($("result"), true);
  } catch (err) {
    show($("busy"), false); show($("empty"), true);
    $("empty").querySelector("h3").textContent = "Something went wrong";
    $("empty").querySelector("p").textContent = String(err.message || err);
  }
}

// ---- Render per modality ----
async function render(src, out, after) {
  ["img-cmp", "video-cmp", "speech-cmp", "text-cmp"].forEach((id) => show($(id), false));
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
