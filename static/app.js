// Dubbing Studio review UI. Plain JS, no build step.
const $ = (s, el = document) => el.querySelector(s);
const api = async (path, opts = {}) => {
  const r = await fetch(path, {
    ...opts,
    headers: opts.body && !(opts.body instanceof FormData) ? { "Content-Type": "application/json" } : {},
    body: opts.body && !(opts.body instanceof FormData) ? JSON.stringify(opts.body) : opts.body,
  });
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return r.json();
};
const fmt = (t) => {
  if (t == null || isNaN(t)) return "–";
  const m = Math.floor(t / 60), s = t - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
};
const toast = (msg) => { console.warn(msg); alert(msg); };

const state = {
  meta: null, project: null, pid: null, peaks: null,
  selected: null, mode: "original", pxPerSec: 60, scrollT: 0,
  rows: new Map(), stopAt: null, polling: null, lastUpdated: 0,
};
const video = $("#video"), dub = $("#dubAudio");

// ------------------------------------------------------------------ boot
async function boot() {
  state.meta = await api("/api/meta");
  const langs = Object.entries(state.meta.languages);
  for (const sel of [$("#newTarget"), $("#targetLang")])
    sel.innerHTML = langs.map(([c, n]) => `<option value="${c}">${n} (${c})</option>`).join("");
  $("#newSource").innerHTML += langs.map(([c, n]) => `<option value="${c}">${n}</option>`).join("");
  $("#stageSelect").innerHTML = state.meta.stages.map((s) => `<option>${s}</option>`).join("");
  if (!state.meta.worker.ok) toast(`GPU worker unreachable: ${state.meta.worker.error}`);
  await refreshProjects(location.hash.slice(1));
}

async function refreshProjects(openId) {
  const list = await api("/api/projects");
  $("#projectSelect").innerHTML = list.length
    ? list.map((p) => `<option value="${p.id}">${esc(p.name)} → ${p.target_lang} (${p.status.state})</option>`).join("")
    : `<option value="">No projects</option>`;
  const id = openId && list.some((p) => p.id === openId) ? openId : list[0]?.id;
  if (id) { $("#projectSelect").value = id; await openProject(id); }
  else { $("#workspace").hidden = true; $("#empty").hidden = false; }
}

async function openProject(pid) {
  state.pid = pid;
  location.hash = pid;
  state.peaks = null;
  state.rows.clear();
  $("#segList").innerHTML = "";
  const p = await api(`/api/projects/${pid}`);
  $("#workspace").hidden = false; $("#empty").hidden = true;
  video.src = `/media/${pid}/preview.mp4`;
  loadDub(p);
  applyProject(p, true);
  poll();
}

function loadDub(p) {
  dub.src = `/media/${p.id}/out/dub.wav?v=${p.updated}`;
}

function poll() {
  clearInterval(state.polling);
  state.polling = setInterval(async () => {
    if (!state.pid) return;
    try {
      const p = await api(`/api/projects/${state.pid}`);
      if (p.updated !== state.lastUpdated) applyProject(p, false);
    } catch {}
  }, 2000);
}

// ------------------------------------------------------------------ render
function applyProject(p, full) {
  const prev = state.project;
  state.project = p;
  state.lastUpdated = p.updated;
  renderStatus(p);
  const opt = $("#projectSelect").selectedOptions[0]; // name changes to the video title once a link is looked up
  if (opt?.value === p.id) opt.textContent = `${p.name} → ${p.target_lang} (${p.status.state})`;
  $("#targetLang").value = p.target_lang;
  const wasRunning = prev && ["running", "queued"].includes(prev.status.state);
  if (wasRunning && p.status.state === "done") loadDub(p);
  if (prev?.id === p.id && prev.stages.extract !== "done" && p.stages.extract === "done")
    video.src = `/media/${p.id}/preview.mp4?v=${p.updated}`; // preview didn't exist when the project opened
  if (!state.peaks && p.stages.separate === "done") loadPeaks();
  renderSpeakers(p);
  renderGlossary(p);
  renderSegments(p, full);
  drawTimeline();
  updateExports(p);
}

function renderStatus(p) {
  const st = p.status;
  $("#status").dataset.state = st.state;
  const stageNo = st.stage ? state.meta.stages.indexOf(st.stage) + 1 : 0;
  let txt = st.state === "running"
    ? `${stageNo}/${state.meta.stages.length} ${st.message || st.stage}`
    : st.state === "error" ? `Error: ${st.error}` : st.message || st.state;
  $("#statusText").textContent = txt;
  $("#statusText").title = txt;
  const overall = st.state === "done" ? 1 : st.state === "running"
    ? (stageNo - 1 + (st.progress || 0)) / state.meta.stages.length : 0;
  $("#statusBar").style.width = `${Math.round(overall * 100)}%`;
  const busy = ["running", "queued"].includes(st.state);
  const pending = p.segments.filter((s) => s.needs?.length).length;
  $("#pending").hidden = !pending;
  $("#pending").textContent = `${pending} changed line${pending === 1 ? "" : "s"} not applied`;
  $("#applyBtn").disabled = busy;
  $("#applyBtn").textContent = st.state === "review" ? "Apply & continue" : "Apply changes";
  document.body.classList.toggle("busy", busy);
}

function renderSpeakers(p) {
  const opts = Object.values(p.speakers);
  const fsel = $("#filterSpeaker"), cur = fsel.value;
  fsel.innerHTML = `<option value="">All speakers</option>` +
    opts.map((s) => `<option value="${s.id}">${esc(s.name)}</option>`).join("");
  fsel.value = cur;
  const counts = {};
  p.segments.forEach((s) => (counts[s.speaker] = (counts[s.speaker] || 0) + 1));
  const box = $("#speakerList");
  if (box.contains(document.activeElement)) return; // don't clobber an edit in progress
  box.innerHTML = "";
  for (const s of opts) {
    const el = document.createElement("div");
    el.className = "spk";
    el.innerHTML = `
      <span class="sw" style="background:${s.color}"></span>
      <input class="name" value="${esc(s.name)}" title="Rename">
      <span class="muted">${counts[s.id] || 0} lines</span>
      <div class="opts">
        <button class="small ghost play-ref" title="Play voice reference">▶ ref</button>
        <label title="Use each line's own audio as the clone prompt (keeps emotion) or one fixed reference">
          voice <select class="mode"><option value="segment">per line</option><option value="speaker">fixed ref</option></select></label>
        <label title="Emotion intensity (blank = per emotion)">intensity
          <input type="range" class="exag" min="0.25" max="1.5" step="0.05" value="${s.exaggeration ?? 0.5}">
          <span class="exag-v">${s.exaggeration ?? "auto"}</span></label>
        <label>merge into <select class="merge"><option value="">—</option>${
          opts.filter((o) => o.id !== s.id).map((o) => `<option value="${o.id}">${esc(o.name)}</option>`).join("")}</select></label>
        <button class="small ghost set-ref" title="Use the selected line as this speaker's reference">use selected as ref</button>
      </div>`;
    $(".mode", el).value = s.ref_mode || "segment";
    $(".name", el).onchange = (e) => patchSpeaker(s.id, { name: e.target.value });
    $(".mode", el).onchange = (e) => patchSpeaker(s.id, { ref_mode: e.target.value });
    $(".exag", el).oninput = (e) => ($(".exag-v", el).textContent = e.target.value);
    $(".exag", el).onchange = (e) => patchSpeaker(s.id, { exaggeration: +e.target.value });
    $(".merge", el).onchange = async (e) => {
      const into = e.target.value;
      if (!into || !confirm(`Merge ${s.name} into ${p.speakers[into].name}? All their lines will be reassigned.`)) return (e.target.value = "");
      await act(() => api(`/api/projects/${p.id}/speakers/merge`, { method: "POST", body: { source: s.id, into } }));
    };
    $(".set-ref", el).onclick = () => {
      if (!state.selected) return toast("Select a line first");
      patchSpeaker(s.id, { ref_segments: [state.selected] });
    };
    $(".play-ref", el).onclick = () => playUrl(`/media/${p.id}/refs/${s.id}.wav?v=${p.updated}`);
    box.appendChild(el);
  }
}

// ------------------------------------------------------------------ name glossary
const nameRe = (n) => new RegExp(`(?<![A-Za-z])${n.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?![A-Za-z])`, "i");
function renderGlossary(p) {
  const box = $("#glossaryList");
  if (box.contains(document.activeElement)) return; // don't clobber an edit in progress
  const list = p.glossary || [];
  $("#approveNames").hidden = !list.some((e) => e.status !== "approved");
  box.innerHTML = list.length ? "" : `<div class="gl-empty">No names yet. Add them, or let “suggest” draft them for review.</div>`;
  list.forEach((e, i) => {
    const spellings = [e.source, ...(e.aliases || [])].filter(Boolean).map(nameRe);
    const hits = p.segments.filter((s) => spellings.some((r) => r.test(s.text))).length;
    const el = document.createElement("div");
    el.className = "gl" + (e.status === "approved" ? "" : " suggested");
    el.title = e.example ? `Suggested from: “${e.example}”` : "";
    el.innerHTML = `
      <input type="text" class="g-src" value="${esc(e.source)}" placeholder="Name as spoken">
      <input type="text" class="g-tgt" value="${esc(e.target)}" placeholder="Name in ${esc(p.target_lang)}">
      <input type="text" class="g-alias" value="${esc((e.aliases || []).join(", "))}" placeholder="also heard as" title="Other spellings in the transcript (ASR mistakes), comma-separated">
      <span class="hits" title="Lines mentioning this name">${hits} lines</span>
      <label title="Approved names are used by the translator"><input type="checkbox" class="g-ok" ${e.status === "approved" ? "checked" : ""}> ✓</label>
      <button class="small ghost g-del" title="Remove">✕</button>`;
    const edit = (fn) => { const next = list.map((x) => ({ ...x })); fn(next[i]); saveGlossary(next); };
    $(".g-src", el).onchange = (ev) => edit((x) => (x.source = ev.target.value));
    $(".g-tgt", el).onchange = (ev) => edit((x) => (x.target = ev.target.value));
    $(".g-alias", el).onchange = (ev) => edit((x) => (x.aliases = ev.target.value.split(",").map((a) => a.trim()).filter(Boolean)));
    $(".g-ok", el).onchange = (ev) => edit((x) => (x.status = ev.target.checked ? "approved" : "suggested"));
    $(".g-del", el).onclick = () => saveGlossary(list.filter((_, j) => j !== i));
    box.appendChild(el);
  });
}
const saveGlossary = async (entries) => {
  // Re-render from the saved list, then put the cursor back where it was (e.g. after Tab).
  await new Promise((r) => setTimeout(r)); // let focus land first
  const rows = [...$("#glossaryList").children], cur = document.activeElement;
  const at = rows.findIndex((r) => r.contains(cur)), cls = cur?.className;
  cur?.blur();
  await act(() => api(`/api/projects/${state.pid}/glossary`, { method: "PUT", body: { entries } }));
  if (at >= 0 && cls) $(`#glossaryList .gl:nth-child(${at + 1}) .${cls.split(" ")[0]}`)?.focus();
};
$("#addName").onclick = () => { // saved once the name is typed in
  state.project.glossary = [...(state.project.glossary || []), { source: "", target: "", aliases: [], status: "approved" }];
  renderGlossary(state.project);
  $("#glossaryList .gl:last-child .g-src").focus();
};
$("#approveNames").onclick = () => {
  const list = state.project.glossary || [];
  const blank = list.filter((e) => e.status !== "approved" && !e.target.trim());
  if (blank.length) return toast(`Fill in the rendering first: ${blank.map((e) => e.source).join(", ")}`);
  saveGlossary(list.map((e) => ({ ...e, status: "approved" })));
};
$("#suggestNames").onclick = async () => {
  const btn = $("#suggestNames");
  btn.textContent = "suggesting…";
  await act(async () => {
    const r = await api(`/api/projects/${state.pid}/glossary/suggest`, { method: "POST" });
    toast(r.added ? `${r.added} names suggested. Check each one (hover for the line it came from), fix it, then tick ✓.` : "No new names found.");
  }, btn);
  btn.textContent = "suggest";
};

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function badges(s, p) {
  const b = [];
  if (s.needs?.length) b.push(`<span class="badge warn" title="Changed — press Apply">${s.needs.join("+")} pending</span>`);
  if (s.tts?.overflow > 0.15) b.push(`<span class="badge bad" title="Dub runs past its slot even at max speed">+${s.tts.overflow.toFixed(1)}s long</span>`);
  if (s.tts && Math.abs(s.tts.speed - 1) > 0.02) b.push(`<span class="badge" title="Tempo change applied to fit">×${s.tts.speed}</span>`);
  if (s.confidence != null && s.confidence < 0.6) b.push(`<span class="badge warn" title="Low ASR confidence — check the text">ASR ${Math.round(s.confidence * 100)}%</span>`);
  if (s.lang && s.lang !== p.source_lang) b.push(`<span class="badge info" title="Different spoken language">${s.lang}</span>`);
  if (s.emotion_score != null && s.emotion_score < 0.5) b.push(`<span class="badge" title="Low emotion confidence">emo?</span>`);
  return b.join(" ");
}

function filtered(p) {
  const q = $("#search").value.toLowerCase(), spk = $("#filterSpeaker").value, flag = $("#filterFlag").value;
  return p.segments.filter((s) => {
    if (spk && s.speaker !== spk) return false;
    if (q && !(s.text + " " + s.translation).toLowerCase().includes(q)) return false;
    const over = s.tts?.overflow > 0.15, low = s.confidence < 0.6, foreign = s.lang && s.lang !== p.source_lang;
    switch (flag) {
      case "attention": return over || low || foreign || s.needs?.length || (s.emotion_score ?? 1) < 0.5;
      case "unreviewed": return !s.reviewed;
      case "pending": return s.needs?.length;
      case "overflow": return over;
      case "lowconf": return low;
      case "foreign": return foreign;
    }
    return true;
  });
}

function renderSegments(p, full) {
  const list = $("#segList");
  const segs = filtered(p);
  $("#segCount").textContent = `${segs.length} / ${p.segments.length} lines · ${p.segments.filter((s) => s.reviewed).length} reviewed`;
  const keep = new Set(segs.map((s) => s.id));
  for (const [id, row] of state.rows) if (!keep.has(id)) { row.el.remove(); state.rows.delete(id); }
  let before = null;
  for (let i = segs.length - 1; i >= 0; i--) {
    const s = segs[i];
    let row = state.rows.get(s.id);
    if (!row) { row = makeRow(s.id); state.rows.set(s.id, row); }
    fillRow(row, s, p, p.segments.indexOf(s) + 1);
    if (row.el.nextSibling !== before || row.el.parentNode !== list) list.insertBefore(row.el, before);
    before = row.el;
  }
}

function makeRow(sid) {
  const el = $("#segTpl").content.firstElementChild.cloneNode(true);
  const row = { el, sid };
  const q = (c) => $(c, el);
  el.onclick = (e) => { if (!["TEXTAREA", "INPUT", "SELECT", "BUTTON"].includes(e.target.tagName)) select(sid, true); };
  el.addEventListener("focusin", () => select(sid, false));
  const patch = (body) => act(() => api(`/api/projects/${state.pid}/segments/${sid}`, { method: "PATCH", body }));
  q(".text").onchange = (e) => patch({ text: e.target.value });
  q(".translation").onchange = (e) => patch({ translation: e.target.value });
  q(".speaker").onchange = (e) => patch({ speaker: e.target.value });
  q(".emotion").onchange = (e) => patch({ emotion: e.target.value });
  q(".lang").onchange = (e) => patch({ lang: e.target.value });
  q(".t-start").onchange = (e) => patch({ start: parseTime(e.target.value) });
  q(".t-end").onchange = (e) => patch({ end: parseTime(e.target.value) });
  q(".reviewed").onchange = (e) => patch({ reviewed: e.target.checked });
  q(".b-orig").onclick = () => playSeg(seg(sid), "original");
  q(".b-dub").onclick = () => {
    const s = seg(sid);
    if (!s.tts) return toast("No dub for this line yet");
    playUrl(`/media/${state.pid}/${s.tts.file}?v=${s.tts.made}`);
  };
  q(".b-trans").onclick = () => act(() => api(`/api/projects/${state.pid}/segments/${sid}/retranslate`, { method: "POST" }), q(".b-trans"));
  q(".b-synth").onclick = async () => {
    await act(() => api(`/api/projects/${state.pid}/segments/${sid}/synthesize`, { method: "POST" }), q(".b-synth"));
    const s = seg(sid);
    if (s?.tts) playUrl(`/media/${state.pid}/${s.tts.file}?v=${s.tts.made}`);
  };
  q(".b-split").onclick = () => splitAtPlayhead(sid);
  q(".b-merge").onclick = () => mergeNext(sid);
  q(".b-del").onclick = () => confirm("Delete this line?") &&
    act(() => api(`/api/projects/${state.pid}/segments/${sid}`, { method: "DELETE" }));
  return row;
}

function fillRow(row, s, p, n) {
  const q = (c) => $(c, row.el);
  const set = (el, v) => { if (document.activeElement !== el && el.value !== v) el.value = v; };
  const spk = p.speakers[s.speaker];
  row.el.style.borderLeftColor = spk?.color || "transparent";
  row.el.classList.toggle("done", !!s.reviewed);
  row.el.classList.toggle("selected", state.selected === s.id);
  q(".idx").textContent = n;
  set(q(".t-start"), s.start.toFixed(2));
  set(q(".t-end"), s.end.toFixed(2));
  const spkHtml = Object.values(p.speakers).map((o) => `<option value="${o.id}">${esc(o.name)}</option>`).join("");
  if (q(".speaker").dataset.h !== spkHtml) { q(".speaker").innerHTML = spkHtml; q(".speaker").dataset.h = spkHtml; }
  set(q(".speaker"), s.speaker);
  const langs = new Set([...Object.keys(state.meta.languages), s.lang, p.source_lang].filter(Boolean));
  const langHtml = [...langs].sort().map((l) => `<option>${l}</option>`).join("");
  if (q(".lang").dataset.h !== langHtml) { q(".lang").innerHTML = langHtml; q(".lang").dataset.h = langHtml; }
  set(q(".lang"), s.lang || p.source_lang || "");
  if (!q(".emotion").options.length)
    q(".emotion").innerHTML = `<option value="">emotion…</option>` + state.meta.emotions.map((e) => `<option>${e}</option>`).join("");
  set(q(".emotion"), s.emotion || "");
  q(".emotion").title = s.emotion_score != null ? `Emotion (model ${Math.round(s.emotion_score * 100)}%${s.emotion_locked ? ", set by reviewer" : ""})` : "Emotion";
  q(".reviewed").checked = !!s.reviewed;
  q(".badges").innerHTML = badges(s, p);
  set(q(".text"), s.text);
  set(q(".translation"), s.translation);
  q(".text").placeholder = `${p.source_lang || "source"} transcript`;
  q(".translation").placeholder = `${p.target_lang} translation`;
  q(".b-dub").disabled = !s.tts;
}

const seg = (sid) => state.project.segments.find((s) => s.id === sid);
const parseTime = (v) => {
  const parts = String(v).split(":").map(Number);
  return parts.length === 2 ? parts[0] * 60 + parts[1] : parts[0];
};

async function act(fn, btn) {
  if (btn) btn.disabled = true;
  try {
    await fn();
    applyProject(await api(`/api/projects/${state.pid}`), false);
  } catch (e) { toast(e.message); }
  finally { if (btn) btn.disabled = false; }
}

const patchSpeaker = (spk, body) => act(() => api(`/api/projects/${state.pid}/speakers/${spk}`, { method: "PATCH", body }));

// The line list fills the space left under the timeline (its height depends on the number of speakers),
// but never less than 360px; on short windows the page scrolls to reach it.
function fitList() {
  const box = $(".segments");
  if (!box || $("#workspace").hidden) return;
  const top = box.getBoundingClientRect().top + window.scrollY;
  const h = Math.max(360, window.innerHeight - top) + "px";
  if (box.style.height !== h) box.style.height = h;
}
window.addEventListener("resize", fitList);
video.addEventListener("loadedmetadata", fitList);  // video height shifts the list down

// Scroll only the line list. scrollIntoView would also scroll the page and push the video out of view.
function revealRow(row) {
  if (!row) return;
  const list = $("#segList"), r = row.el.getBoundingClientRect(), box = list.getBoundingClientRect();
  if (r.top < box.top) list.scrollTop -= box.top - r.top;
  else if (r.bottom > box.bottom) list.scrollTop += Math.min(r.bottom - box.bottom, r.top - box.top);
}

function select(sid, scrollList) {
  if (state.selected === sid) return;
  const old = state.rows.get(state.selected);
  old?.el.classList.remove("selected");
  state.selected = sid;
  const row = state.rows.get(sid);
  row?.el.classList.add("selected");
  if (scrollList) revealRow(row);
  const s = seg(sid);
  if (s && video.paused) seek(s.start);
  ensureVisible(s);
  drawTimeline();
}

function splitAtPlayhead(sid) {
  const s = seg(sid), t = video.currentTime;
  if (!(t > s.start + 0.1 && t < s.end - 0.1)) return toast("Move the playhead inside this line first (click the timeline)");
  act(() => api(`/api/projects/${state.pid}/segments/${sid}/split`, { method: "POST", body: { at: t } }));
}
function mergeNext(sid) {
  const segs = state.project.segments, i = segs.findIndex((s) => s.id === sid);
  if (i < 0 || i === segs.length - 1) return;
  act(() => api(`/api/projects/${state.pid}/segments/merge`, { method: "POST", body: { ids: [sid, segs[i + 1].id] } }));
}

// ------------------------------------------------------------------ playback
function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".seg-toggle button").forEach((b) => b.classList.toggle("on", b.dataset.mode === mode));
  video.muted = mode === "dub";
  video.volume = mode === "both" ? 0.35 : 1;
  syncDub(true);
}
function syncDub(force) {
  if (state.mode === "original" || !dub.src) { dub.pause(); return; }
  if (force || Math.abs(dub.currentTime - video.currentTime) > 0.15) dub.currentTime = video.currentTime;
  if (!video.paused && dub.paused) dub.play().catch(() => {});
  if (video.paused && !dub.paused) dub.pause();
}
function seek(t) { video.currentTime = Math.max(0, t); syncDub(true); drawTimeline(); }
function playSeg(s, mode) {
  if (mode) setMode(mode);
  seek(s.start);
  state.stopAt = s.end;
  video.play();
}
let clipPlayer = new Audio();
function playUrl(url) { video.pause(); clipPlayer.pause(); clipPlayer = new Audio(url); clipPlayer.play().catch((e) => toast(e.message)); }

video.addEventListener("play", () => { $("#playBtn").textContent = "❚❚"; syncDub(true); });
video.addEventListener("pause", () => { $("#playBtn").textContent = "▶"; dub.pause(); });
video.addEventListener("seeked", () => syncDub(true));
video.addEventListener("timeupdate", () => {
  if (state.stopAt && video.currentTime >= state.stopAt) { video.pause(); state.stopAt = null; }
  syncDub(false);
});
function tick() {
  $("#clock").textContent = fmt(video.currentTime);
  if (!video.paused && state.project) {
    const t = video.currentTime;
    const cur = state.project.segments.find((s) => s.start <= t && t < s.end);
    document.querySelectorAll(".seg.active").forEach((e) => e.classList.remove("active"));
    if (cur) {
      const row = state.rows.get(cur.id);
      row?.el.classList.add("active");
      if (row && !$("#segList").contains(document.activeElement)) revealRow(row);
    }
    const span = timeline.clientWidth / state.pxPerSec;
    if (t > state.scrollT + span * 0.9 || t < state.scrollT) state.scrollT = Math.max(0, t - span * 0.1);
    drawTimeline();
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

// ------------------------------------------------------------------ timeline
const timeline = $("#timeline");
const LANE_H = 22, TOP = 18, WAVE_H = 34;

async function loadPeaks() {
  try { state.peaks = await api(`/api/projects/${state.pid}/peaks`); drawTimeline(); } catch {}
}
function lanes() { return Object.keys(state.project?.speakers || {}); }
function ensureVisible(s) {
  if (!s) return;
  const span = timeline.clientWidth / state.pxPerSec;
  if (s.start < state.scrollT || s.end > state.scrollT + span) state.scrollT = Math.max(0, s.start - span * 0.2);
}
function drawTimeline() {
  const p = state.project;
  if (!p) return;
  const dpr = devicePixelRatio || 1, W = timeline.clientWidth;
  const L = lanes();
  const H = TOP + WAVE_H + Math.max(1, L.length) * LANE_H + 6;
  if (timeline.height !== H * dpr || timeline.width !== W * dpr) {
    timeline.width = W * dpr; timeline.height = H * dpr; timeline.style.height = H + "px";
    fitList();
  }
  const g = timeline.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const css = getComputedStyle(document.documentElement);
  const C = (n) => css.getPropertyValue(n).trim();
  g.clearRect(0, 0, W, H);
  const pps = state.pxPerSec, t0 = state.scrollT, x = (t) => (t - t0) * pps;
  // ruler
  g.fillStyle = C("--muted"); g.font = "10px ui-monospace, monospace";
  const step = pps > 120 ? 1 : pps > 40 ? 5 : pps > 15 ? 10 : 30;
  for (let t = Math.ceil(t0 / step) * step; x(t) < W; t += step) {
    g.fillRect(x(t), 0, 1, 5); g.fillText(fmt(t).replace(/\.0$/, ""), x(t) + 3, 11);
  }
  // waveform
  if (state.peaks) {
    const { per_sec, peaks } = state.peaks;
    g.fillStyle = C("--line");
    const mid = TOP + WAVE_H / 2;
    for (let px = 0; px < W; px++) {
      const a = Math.floor((t0 + px / pps) * per_sec), b = Math.max(a + 1, Math.floor((t0 + (px + 1) / pps) * per_sec));
      let m = 0; for (let i = a; i < b && i < peaks.length; i++) m = Math.max(m, peaks[i]);
      const h = (m / 255) * (WAVE_H / 2 - 1);
      g.fillRect(px, mid - h, 1, h * 2 || 1);
    }
  }
  // lanes
  L.forEach((spk, i) => {
    const y = TOP + WAVE_H + i * LANE_H;
    g.fillStyle = C("--lane"); g.fillRect(0, y + 1, W, LANE_H - 2);
  });
  for (const s of p.segments) {
    if (x(s.end) < 0 || x(s.start) > W) continue;
    const li = Math.max(0, L.indexOf(s.speaker));
    const y = TOP + WAVE_H + li * LANE_H;
    const spk = p.speakers[s.speaker];
    g.globalAlpha = s.reviewed ? 0.55 : 0.9;
    g.fillStyle = spk?.color || "#888";
    g.fillRect(x(s.start), y + 3, Math.max(2, (s.end - s.start) * pps), LANE_H - 6);
    g.globalAlpha = 1;
    if (s.tts?.overflow > 0.15) { g.fillStyle = C("--bad"); g.fillRect(x(s.end), y + 3, s.tts.overflow * pps, 3); }
    if (s.needs?.length) { g.fillStyle = C("--warn"); g.fillRect(x(s.start), y + LANE_H - 5, Math.max(2, (s.end - s.start) * pps), 2); }
    if (s.id === state.selected) {
      g.strokeStyle = C("--text"); g.lineWidth = 2;
      g.strokeRect(x(s.start) + 1, y + 2, Math.max(2, (s.end - s.start) * pps) - 2, LANE_H - 4);
    }
    const w = (s.end - s.start) * pps;
    if (w > 40) {
      g.fillStyle = "#fff"; g.font = "11px -apple-system, sans-serif";
      g.save(); g.beginPath(); g.rect(x(s.start), y, w - 3, LANE_H); g.clip();
      g.fillText(s.translation || s.text, x(s.start) + 4, y + 15); g.restore();
    }
  }
  // lane labels
  g.font = "10px -apple-system, sans-serif";
  L.forEach((spk, i) => {
    const y = TOP + WAVE_H + i * LANE_H;
    g.fillStyle = C("--muted"); g.fillText(p.speakers[spk].name, 4, y + 14);
  });
  // playhead
  g.fillStyle = C("--play"); g.fillRect(x(video.currentTime), 0, 2, H);
}

let drag = null;
function hit(ev) {
  const r = timeline.getBoundingClientRect(), px = ev.clientX - r.left, py = ev.clientY - r.top;
  const t = state.scrollT + px / state.pxPerSec;
  const li = Math.floor((py - TOP - WAVE_H) / LANE_H), spk = lanes()[li];
  const edge = 5 / state.pxPerSec;
  for (const s of state.project?.segments || []) {
    if (s.speaker !== spk) continue;
    if (Math.abs(t - s.start) < edge) return { s, t, kind: "start" };
    if (Math.abs(t - s.end) < edge) return { s, t, kind: "end" };
    if (t > s.start && t < s.end) return { s, t, kind: "body" };
  }
  return { t, spk };
}
timeline.addEventListener("mousemove", (ev) => {
  if (drag) {
    const h = hit(ev);
    drag.s[drag.kind] = Math.max(0, +h.t.toFixed(2));
    drawTimeline();
    return;
  }
  const h = hit(ev);
  timeline.style.cursor = h.kind === "start" || h.kind === "end" ? "ew-resize" : h.s ? "pointer" : "crosshair";
});
timeline.addEventListener("mousedown", (ev) => {
  const h = hit(ev);
  if (h.kind === "start" || h.kind === "end") { drag = { s: h.s, kind: h.kind, orig: h.s[h.kind] }; return; }
  if (h.s) select(h.s.id, true);
  seek(h.t);
});
window.addEventListener("mouseup", () => {
  if (!drag) return;
  const { s, kind, orig } = drag; drag = null;
  if (s[kind] !== orig)
    act(() => api(`/api/projects/${state.pid}/segments/${s.id}`, { method: "PATCH", body: { [kind]: s[kind] } }));
});
timeline.addEventListener("wheel", (ev) => {
  ev.preventDefault();
  if (ev.ctrlKey || ev.metaKey) {
    const r = timeline.getBoundingClientRect(), tAt = state.scrollT + (ev.clientX - r.left) / state.pxPerSec;
    state.pxPerSec = Math.min(400, Math.max(10, state.pxPerSec * (ev.deltaY < 0 ? 1.15 : 0.87)));
    $("#zoom").value = state.pxPerSec;
    state.scrollT = Math.max(0, tAt - (ev.clientX - r.left) / state.pxPerSec);
  } else {
    state.scrollT = Math.max(0, state.scrollT + (ev.deltaX || ev.deltaY) / state.pxPerSec);
  }
  drawTimeline();
}, { passive: false });
$("#zoom").oninput = (e) => { state.pxPerSec = +e.target.value; drawTimeline(); };
window.addEventListener("resize", drawTimeline);

// ------------------------------------------------------------------ controls
$("#playBtn").onclick = () => (video.paused ? video.play() : video.pause());
document.querySelectorAll(".seg-toggle button").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
$("#projectSelect").onchange = (e) => e.target.value && openProject(e.target.value);
$("#search").oninput = () => renderSegments(state.project, false);
$("#filterSpeaker").onchange = () => renderSegments(state.project, false);
$("#filterFlag").onchange = () => renderSegments(state.project, false);
$("#moreBtn").onclick = () => ($("#moreMenu").hidden = !$("#moreMenu").hidden);
document.addEventListener("click", (e) => { if (!e.target.closest(".menu")) $("#moreMenu").hidden = true; });

$("#applyBtn").onclick = () => act(() => api(`/api/projects/${state.pid}/run`, { method: "POST", body: { from_stage: "emotion" } }));
$("#runStageBtn").onclick = () => {
  const st = $("#stageSelect").value, full = $("#fullRun").checked;
  if (["extract", "separate", "transcribe", "diarize", "segment"].includes(st) &&
      !confirm(`Re-running "${st}" rebuilds all lines and discards your edits. Continue?`)) return;
  act(() => api(`/api/projects/${state.pid}/run`, { method: "POST", body: { from_stage: st, full } }));
};
$("#targetLang").onchange = (e) => {
  if (!confirm("Change the dub language? All lines will be re-translated and re-voiced.")) return (e.target.value = state.project.target_lang);
  act(() => api(`/api/projects/${state.pid}`, { method: "PATCH", body: { target_lang: e.target.value } }));
};
$("#deleteBtn").onclick = async () => {
  if (!confirm(`Delete project "${state.project.name}" and all its files?`)) return;
  try { await api(`/api/projects/${state.pid}`, { method: "DELETE" }); state.pid = null; refreshProjects(); }
  catch (e) { toast(e.message); }
};
$("#addSpeaker").onclick = () => act(() => api(`/api/projects/${state.pid}/speakers`, { method: "POST" }));

function updateExports(p) {
  const ready = p.stages.mix === "done";
  for (const [id, kind] of [["#exVideo", "video"], ["#exAudio", "audio"], ["#exSrcSrt", "srt_source"], ["#exTgtSrt", "srt_target"], ["#exProject", "project"]]) {
    const a = $(id);
    a.href = `/api/projects/${p.id}/export/${kind}`;
    a.classList.toggle("disabled", !ready && kind !== "project");
  }
}

// keyboard
document.addEventListener("keydown", (e) => {
  const typing = ["TEXTAREA", "INPUT", "SELECT"].includes(document.activeElement.tagName);
  if (e.key === "Escape") document.activeElement.blur();
  if (typing) {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); document.activeElement.blur(); reviewNext(); }
    return;
  }
  const segs = filtered(state.project || { segments: [] });
  const i = segs.findIndex((s) => s.id === state.selected);
  if (e.key === " ") { e.preventDefault(); video.paused ? video.play() : video.pause(); }
  else if (e.key === "j" || e.key === "ArrowDown") { e.preventDefault(); segs[i + 1] && select(segs[i + 1].id, true); }
  else if (e.key === "k" || e.key === "ArrowUp") { e.preventDefault(); segs[i - 1] && select(segs[i - 1].id, true); }
  else if (e.key === "Enter") { e.preventDefault(); reviewNext(); }
  else if (e.key === "s" && state.selected) splitAtPlayhead(state.selected);
  else if (e.key === "m" && state.selected) mergeNext(state.selected);
  else if (e.key === "o" && state.selected) playSeg(seg(state.selected), "original");
  else if (e.key === "d" && state.selected) playSeg(seg(state.selected), "dub");
});
function reviewNext() {
  if (!state.selected) return;
  const segs = filtered(state.project), i = segs.findIndex((s) => s.id === state.selected);
  api(`/api/projects/${state.pid}/segments/${state.selected}`, { method: "PATCH", body: { reviewed: true } })
    .then(async () => applyProject(await api(`/api/projects/${state.pid}`), false)).catch((e) => toast(e.message));
  if (segs[i + 1]) select(segs[i + 1].id, true);
}

// ------------------------------------------------------------------ new project
$("#newBtn").onclick = () => $("#newDialog").showModal();
$("#newFile").onchange = () => { if ($("#newFile").files.length) $("#newUrl").value = ""; };
$("#newUrl").oninput = () => { if ($("#newUrl").value.trim()) $("#newFile").value = ""; };
$("#newForm").addEventListener("submit", (e) => {
  if (e.submitter?.value !== "ok") return;
  e.preventDefault();
  const f = new FormData(e.target);
  const file = $("#newFile").files[0], url = f.get("url").trim();
  if (!file && !url) return toast("Choose a video file or paste a video link.");
  if (file) f.delete("url"); else f.delete("video");
  f.set("pause_after", f.get("pause") ? "segment" : "none");
  f.delete("pause");
  const bar = $(".upload-progress"), btn = $("#createBtn");
  bar.hidden = false; btn.disabled = true;
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/projects");
  xhr.upload.onprogress = (ev) => ($("div", bar).style.width = `${(ev.loaded / ev.total) * 100}%`);
  xhr.onload = () => {
    btn.disabled = false; bar.hidden = true;
    if (xhr.status >= 300) return toast(xhr.responseText);
    $("#newDialog").close(); e.target.reset();
    refreshProjects(JSON.parse(xhr.responseText).id);
  };
  xhr.onerror = () => { btn.disabled = false; bar.hidden = true; toast("Upload failed"); };
  xhr.send(f);
});

boot().catch((e) => toast(e.message));
