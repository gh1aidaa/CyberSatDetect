// =====================================
// CyberSatDetect | Reports Page (minimal: download + local download log)
// =====================================

const API_BASE = window.location.origin;

/** Browser-local log of successful PDF/Excel downloads from this page */
const DOWNLOAD_LOG_KEY = "csd_report_downloads_v1";
const DOWNLOAD_LOG_MAX = 80;

document.addEventListener("DOMContentLoaded", async () => {
  hydrateRunContext();
  await loadRunChoices();
  bindExportButtons();
  bindDownloadLogUi();
  renderDownloadLog();
  initBG();
});

function getToken() {
  return localStorage.getItem("token");
}

function requireLogin() {
  window.location.href = "login.html";
}

if (!getToken()) requireLogin();

const reportsTableBody = document.getElementById("reportsTableBody");
const repNote = document.getElementById("repNote");
const exportNote = document.getElementById("exportNote");

const btnExpertPDF = document.getElementById("btnExpertPDF");
const btnExpertExcel = document.getElementById("btnExpertExcel");
const btnClearDownloadLog = document.getElementById("btnClearDownloadLog");

function hydrateRunContext() {
  const hint = document.getElementById("telemetryHint");
  if (hint) {
    hint.innerHTML =
      `<span style="opacity:0.9;font-size:0.84rem;line-height:1.45;">Choose a completed run, then export PDF or Excel.</span>`;
  }
}

function getActiveRunId() {
  const sel = document.getElementById("runSelect");
  if (sel && sel.value) return sel.value.trim();
  return "";
}

function formatRunLabel(r) {
  const name = (r.filename && String(r.filename).trim()) || "Telemetry";
  let d;
  try {
    d = r.created_at ? new Date(r.created_at) : null;
  } catch (_) {
    d = null;
  }
  const ds =
    d && !Number.isNaN(d.getTime())
      ? d.toLocaleString(undefined, {
          dateStyle: "medium",
          timeStyle: "short",
        })
      : "";
  return ds ? `${name}  ·  ${ds}` : name;
}

async function loadRunChoices() {
  const sel = document.getElementById("runSelect");
  if (!sel) return;

  sel.innerHTML = '<option value="">— Select a telemetry file —</option>';

  try {
    const res = await fetch(`${API_BASE}/dashboard/runs`, {
      headers: { Authorization: "Bearer " + getToken() },
    });
    const data = await res.json();
    if (res.status === 401) {
      requireLogin();
      return;
    }
    if (!res.ok) {
      throw new Error(
        typeof data.detail === "string"
          ? data.detail
          : "Could not load telemetry list."
      );
    }

    const runs = (data.runs || []).filter((row) => String(row.status) === "DONE");
    const preferred = localStorage.getItem("csd_run_id");
    let matchedPreferred = false;

    runs.forEach((r) => {
      const opt = document.createElement("option");
      opt.value = r.run_id;
      opt.textContent = formatRunLabel(r);
      sel.appendChild(opt);
      if (preferred && r.run_id === preferred) {
        opt.selected = true;
        matchedPreferred = true;
      }
    });

    if (!matchedPreferred && runs.length > 0) {
      sel.selectedIndex = 1;
    }

    if (runs.length === 0 && exportNote) {
      exportNote.textContent =
        "No completed runs yet. Finish anomaly detection first, then return here.";
    }
  } catch (e) {
    console.error(e);
    if (exportNote) exportNote.textContent = e.message || "Could not load the file list.";
  }
}

function bindExportButtons() {
  if (btnExpertPDF) {
    btnExpertPDF.addEventListener("click", async () => {
      const rid = getActiveRunId();
      if (!rid) return alert("Select a run first.");
      try {
        await downloadReportFile(
          `${API_BASE}/reports/run/${encodeURIComponent(rid)}/pdf`,
          `RUN-${rid}.pdf`,
          { format: "PDF", refType: "run", refId: rid }
        );
      } catch (e) {
        alert(e.message || "PDF download failed");
      }
    });
  }
  if (btnExpertExcel) {
    btnExpertExcel.addEventListener("click", async () => {
      const rid = getActiveRunId();
      if (!rid) return alert("Select a run first.");
      try {
        await downloadReportFile(
          `${API_BASE}/reports/run/${encodeURIComponent(rid)}/excel`,
          `RUN-${rid}.xlsx`,
          { format: "Excel", refType: "run", refId: rid }
        );
      } catch (e) {
        alert(e.message || "Excel download failed");
      }
    });
  }
}

function readDownloadLog() {
  try {
    const raw = localStorage.getItem(DOWNLOAD_LOG_KEY);
    if (!raw) return [];
    const arr = JSON.parse(raw);
    return Array.isArray(arr) ? arr : [];
  } catch (_) {
    return [];
  }
}

function writeDownloadLog(entries) {
  localStorage.setItem(DOWNLOAD_LOG_KEY, JSON.stringify(entries));
}

function appendDownloadLog(entry) {
  const id =
    typeof crypto !== "undefined" && crypto.randomUUID
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
  const list = readDownloadLog();
  list.unshift({ id, at: new Date().toISOString(), ...entry });
  while (list.length > DOWNLOAD_LOG_MAX) list.pop();
  writeDownloadLog(list);
}

function removeDownloadLogEntry(id) {
  writeDownloadLog(readDownloadLog().filter((e) => e.id !== id));
}

function clearDownloadLog() {
  localStorage.removeItem(DOWNLOAD_LOG_KEY);
}

function renderDownloadLog() {
  if (!reportsTableBody) return;
  const list = readDownloadLog();
  reportsTableBody.innerHTML = "";

  if (!list.length) {
    if (repNote)
      repNote.textContent =
        "No downloads logged yet. Export PDF or Excel above — each successful download appears here.";
    return;
  }

  if (repNote)
    repNote.textContent =
      "Log is stored only in this browser (not synced to your account). Clear log only removes these entries.";

  list.forEach((e) => {
    const tr = document.createElement("tr");
    const tdWhen = document.createElement("td");
    tdWhen.textContent =
      e.at && !Number.isNaN(new Date(e.at).getTime())
        ? new Date(e.at).toLocaleString()
        : "—";
    const tdFmt = document.createElement("td");
    tdFmt.textContent = e.format || "—";
    const tdRef = document.createElement("td");
    tdRef.textContent =
      e.refType === "report" ? `Report ${e.refId}` : `Run ${e.refId || "—"}`;
    const tdFile = document.createElement("td");
    tdFile.textContent = e.fileName || "—";
    tdFile.style.maxWidth = "260px";
    tdFile.style.overflow = "hidden";
    tdFile.style.textOverflow = "ellipsis";
    tdFile.title = e.fileName || "";
    const tdAct = document.createElement("td");
    const rm = document.createElement("button");
    rm.type = "button";
    rm.className = "export-btn";
    rm.textContent = "Remove";
    rm.dataset.logRemove = e.id;
    tdAct.appendChild(rm);
    tr.appendChild(tdWhen);
    tr.appendChild(tdFmt);
    tr.appendChild(tdRef);
    tr.appendChild(tdFile);
    tr.appendChild(tdAct);
    reportsTableBody.appendChild(tr);
  });
}

function bindDownloadLogUi() {
  if (btnClearDownloadLog) {
    btnClearDownloadLog.addEventListener("click", () => {
      if (!readDownloadLog().length) return;
      if (!confirm("Clear the entire download log on this device?")) return;
      clearDownloadLog();
      renderDownloadLog();
    });
  }

  if (reportsTableBody) {
    reportsTableBody.addEventListener("click", (ev) => {
      const btn = ev.target.closest("button[data-log-remove]");
      if (!btn || !btn.dataset.logRemove) return;
      removeDownloadLogEntry(btn.dataset.logRemove);
      renderDownloadLog();
    });
  }
}

function filenameFromContentDisposition(cd, fallback) {
  if (!cd || typeof cd !== "string") return fallback;
  const star = /filename\*=(?:UTF-8'')?([^;\n]+)/i.exec(cd);
  if (star) {
    try {
      return decodeURIComponent(star[1].trim().replace(/^["']|["'];?$/g, ""));
    } catch (_) {
      /* ignore */
    }
  }
  const plain = /filename="([^"]+)"/i.exec(cd);
  if (plain) return plain[1].trim();
  return fallback;
}

/**
 * @param {{ format: string, refType: 'run'|'report', refId: string }} [meta] If set, appends to local download log after success.
 */
async function downloadReportFile(url, fallbackName, meta) {
  const res = await fetch(url, {
    headers: { Authorization: "Bearer " + getToken() },
    cache: "no-store",
  });
  if (res.status === 401) {
    requireLogin();
    return;
  }
  const ct = (res.headers.get("Content-Type") || "").toLowerCase();
  if (!res.ok) {
    let msg = res.statusText || "Download failed";
    if (ct.includes("application/json")) {
      try {
        const j = await res.json();
        if (typeof j.detail === "string") msg = j.detail;
        else if (Array.isArray(j.detail))
          msg = j.detail.map((x) => x.msg || JSON.stringify(x)).join("; ");
        else if (j.detail != null) msg = JSON.stringify(j.detail);
      } catch (_) {
        /* ignore */
      }
    } else {
      const t = await res.text();
      if (t) msg = t.slice(0, 500);
    }
    throw new Error(msg);
  }
  const blob = await res.blob();
  const fname =
    filenameFromContentDisposition(res.headers.get("Content-Disposition"), fallbackName) ||
    fallbackName;
  const objectUrl = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = objectUrl;
  a.download = fname;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(objectUrl), 60_000);

  if (meta && meta.refId != null && meta.refType) {
    appendDownloadLog({
      format: meta.format || "—",
      refType: meta.refType,
      refId: String(meta.refId),
      fileName: fname,
    });
    renderDownloadLog();
  }
}

// =====================================
// 3D BACKGROUND
// =====================================
let scene, camera, renderer, sphere, stars;

function initBG() {
  const container = document.getElementById("canvas_container");
  if (!container) return;

  scene = new THREE.Scene();

  camera = new THREE.PerspectiveCamera(
    60,
    window.innerWidth / window.innerHeight,
    0.1,
    1200
  );
  camera.position.set(0, 0, 260);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.7;
  controls.enablePan = false;
  controls.enableZoom = false;

  const loader = new THREE.TextureLoader();
  const bg = loader.load("https://i.ibb.co/HC0vxMw/sky2.jpg");

  sphere = new THREE.Mesh(
    new THREE.SphereGeometry(520, 48, 48),
    new THREE.MeshBasicMaterial({
      side: THREE.BackSide,
      map: bg,
    })
  );
  scene.add(sphere);

  const starsGeo = new THREE.BufferGeometry();
  const positions = [];

  for (let i = 0; i < 600; i++) {
    positions.push(
      (Math.random() - 0.5) * 1000,
      (Math.random() - 0.5) * 1000,
      (Math.random() - 0.5) * 1000
    );
  }

  starsGeo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));

  stars = new THREE.Points(
    starsGeo,
    new THREE.PointsMaterial({
      color: "#ffffff",
      size: 1.6,
      opacity: 0.9,
      transparent: true,
    })
  );

  scene.add(stars);

  animateBG();
}

function animateBG() {
  requestAnimationFrame(animateBG);

  if (sphere) sphere.rotation.y += 0.0007;
  if (stars) stars.rotation.y += 0.00035;

  renderer.render(scene, camera);
}

window.addEventListener("resize", () => {
  if (!camera || !renderer) return;

  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
