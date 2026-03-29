function updateUploadRunInfo() {
  const info = qs("#uploadRunInfo");
  if (!info) return;
  if (state.activeRun?.run_id) {
    const base = `Active run: ${state.activeRun.name} (${state.activeRun.run_id})`;
    info.textContent = base;
  } else {
    info.textContent = "No active run selected.";
  }
}

function clearUploadJobPolling() {
  if (state.uploadJob.pollTimer) {
    window.clearTimeout(state.uploadJob.pollTimer);
    state.uploadJob.pollTimer = null;
  }
}

function setUploadProgressUI({
  visible = false,
  mode = "uploading",
  statusText = "",
  uploadedBytes = 0,
  totalBytes = null,
  rowsProcessed = 0,
  rowsSucceeded = 0,
  rowsFailed = 0,
  errorSamples = [],
}) {
  const panel = qs("#uploadProgressPanel");
  const title = qs("#uploadProgressTitle");
  const percent = qs("#uploadProgressPercent");
  const bar = qs("#uploadProgressBar");
  const status = qs("#uploadProgressMessage");
  const rowsMeta = qs("#uploadProgressRows");
  const errors = qs("#uploadProgressErrors");

  if (panel) {
    if (visible) panel.classList.remove("hidden");
    else panel.classList.add("hidden");
  }
  if (status) status.textContent = String(statusText || "");

  const total = Number(totalBytes);
  const received = Number(uploadedBytes || 0);
  const uploadPctRaw =
    Number.isFinite(total) && total > 0
      ? Math.max(0, Math.min(100, (received / total) * 100))
      : 0;
  const uploadCompleteModes = new Set(["queued", "processing", "completed", "partial_success", "failed"]);
  const displayPct = uploadCompleteModes.has(mode) ? 100 : uploadPctRaw;
  if (bar) bar.style.width = `${displayPct}%`;
  if (percent) percent.textContent = `${Math.round(displayPct)}%`;

  if (title) {
    if (mode === "uploading") title.textContent = "Uploading CSV";
    else if (mode === "queued") title.textContent = "Queued for ingest";
    else if (mode === "processing") title.textContent = "Processing rows";
    else if (mode === "completed") title.textContent = "Ingest completed";
    else if (mode === "partial_success") title.textContent = "Ingest partial success";
    else if (mode === "failed") title.textContent = "Ingest failed";
    else title.textContent = "Ingest progress";
  }

  const processed = Number(rowsProcessed || 0);
  const succeeded = Number(rowsSucceeded || 0);
  const failed = Number(rowsFailed || 0);
  if (rowsMeta) {
    const uploadText =
      Number.isFinite(total) && total > 0
        ? `${formatBytes(received)} / ${formatBytes(total)} uploaded`
        : `${formatBytes(received)} uploaded`;
    rowsMeta.textContent = `${uploadText} · ${processed} processed · ${succeeded} succeeded · ${failed} failed`;
  }

  if (errors) {
    if (Array.isArray(errorSamples) && errorSamples.length > 0) {
      errors.innerHTML = errorSamples
        .slice(0, 4)
        .map((e) => `<li>Row ${escapeHtml(e.row)}: ${escapeHtml(e.message)}</li>`)
        .join("");
      errors.classList.remove("hidden");
    } else {
      errors.innerHTML = "";
      errors.classList.add("hidden");
    }
  }
}

function resetUploadPanelIfIdle() {
  if (state.uploadJob.active) return;
  setUploadProgressUI({ visible: false });
}

async function uploadCsvFileWithProgress(file, runId, columnMapping = null) {
  const url = apiUrl("/ingest/csv/upload", tenantScopeParams({ run_id: runId }));
  const form = new FormData();
  form.append("file", file, file.name);
  if (columnMapping && typeof columnMapping === "object") {
    form.append("mapping", JSON.stringify(columnMapping));
  }
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url, true);
    xhr.responseType = "json";
    xhr.upload.onprogress = (evt) => {
      const loaded = Number(evt.loaded || 0);
      const total = evt.lengthComputable ? Number(evt.total || 0) : Number(file.size || 0);
      setUploadProgressUI({
        visible: true,
        mode: "uploading",
        statusText: "Uploading CSV...",
        uploadedBytes: loaded,
        totalBytes: total > 0 ? total : null,
        rowsProcessed: 0,
        rowsSucceeded: 0,
        rowsFailed: 0,
      });
    };
    xhr.onerror = () => {
      reject(new Error("Network error during CSV upload."));
    };
    xhr.onload = () => {
      const body = xhr.response || {};
      if (xhr.status < 200 || xhr.status >= 300) {
        const detail = body && body.detail ? String(body.detail) : `HTTP ${xhr.status}`;
        reject(new Error(detail));
        return;
      }
      resolve(body);
    };
    xhr.send(form);
  });
}

async function fetchIngestJob(jobId) {
  return fetchJson(apiUrl(`/ingest/jobs/${encodeURIComponent(jobId)}`, tenantScopeParams()));
}

async function waitForIngestJob(jobId) {
  state.uploadJob.id = jobId;
  state.uploadJob.active = true;
  clearUploadJobPolling();
  return new Promise((resolve, reject) => {
    const tick = async () => {
      if (!state.uploadJob.active) {
        reject(new Error("Upload monitoring cancelled."));
        return;
      }
      try {
        const job = await fetchIngestJob(jobId);
        const status = String(job.status || "processing");
        setUploadProgressUI({
          visible: true,
          mode: status,
          statusText: String(job.message || `Ingest status: ${status}`),
          uploadedBytes: Number(job.upload_bytes_received || 0),
          totalBytes: job.upload_bytes_total,
          rowsProcessed: Number(job.rows_processed || 0),
          rowsSucceeded: Number(job.rows_succeeded || 0),
          rowsFailed: Number(job.rows_failed || 0),
          errorSamples: job.error_samples || [],
        });
        if (status === "completed" || status === "partial_success" || status === "failed") {
          clearUploadJobPolling();
          state.uploadJob.active = false;
          resolve(job);
          return;
        }
      } catch (err) {
        clearUploadJobPolling();
        state.uploadJob.active = false;
        reject(err);
        return;
      }
      state.uploadJob.pollTimer = window.setTimeout(tick, INGEST_JOB_POLL_MS);
    };
    tick();
  });
}


function parseCsvText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("Failed to read CSV file"));
    reader.readAsText(file, "utf-8");
  });
}

function setUploadFile(file) {
  state.uploadFile = file || null;
  const el = qs("#selectedFileName");
  if (!el) return;
  el.textContent = state.uploadFile ? `${state.uploadFile.name} (${state.uploadFile.size} bytes)` : "No file selected";
  if (!file) {
    state.uploadCsv.preview = null;
    state.uploadCsv.headers = [];
    state.uploadCsv.issues = [];
    state.uploadCsv.warnings = [];
    state.uploadCsv.requiresConfirmation = false;
    state.uploadCsv.mapping = null;
    renderUploadMappingPanel();
    return;
  }
  runCsvPreviewForFile(file).catch((err) => {
    setStatus(String(err.message || err), true, true);
  });
}

async function runCsvPreviewForFile(file) {
  const chunk = file.slice(0, Math.min(file.size, 65536));
  const text = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("Failed to read CSV sample"));
    reader.readAsText(chunk, "utf-8");
  });
  const out = await fetchJson(apiUrl("/ingest/csv/preview", tenantScopeParams()), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ csv_sample: text }),
  });
  state.uploadCsv.preview = out;
  state.uploadCsv.headers = Array.isArray(out.headers) ? out.headers : [];
  state.uploadCsv.issues = Array.isArray(out.issues) ? out.issues : [];
  state.uploadCsv.warnings = Array.isArray(out.warnings) ? out.warnings : [];
  state.uploadCsv.requiresConfirmation = !!out.requires_confirmation;
  state.uploadCsv.mapping = out.suggested_mapping || null;
  renderUploadMappingPanel();
  if (state.uploadCsv.issues.length && !out.suggested_mapping) {
    setStatus(state.uploadCsv.issues.join(" "), true, false);
  } else {
    setStatus("");
  }
}

function renderUploadMappingPanel() {
  const panel = qs("#csvMappingPanel");
  const sumEl = qs("#csvMappingSummary");
  const issuesEl = qs("#csvMappingIssues");
  const ts = qs("#csvMapTimestamp");
  const asset = qs("#csvMapAsset");
  const site = qs("#csvMapSite");
  const sensorsWrap = qs("#csvMapSensors");
  if (!panel || !sumEl || !issuesEl || !ts || !asset || !site || !sensorsWrap) return;

  const headers = state.uploadCsv.headers || [];
  const hasFile = !!state.uploadFile;
  if (!hasFile || headers.length === 0) {
    panel.classList.add("hidden");
    return;
  }
  panel.classList.remove("hidden");

  const map = state.uploadCsv.mapping;
  const conf = state.uploadCsv.requiresConfirmation ? "Review or adjust detected roles before ingesting." : "Auto-detected roles (you can override).";
  sumEl.textContent = conf;

  issuesEl.innerHTML = "";
  const allMsgs = [...(state.uploadCsv.issues || []), ...(state.uploadCsv.warnings || [])];
  allMsgs.forEach((msg) => {
    const li = document.createElement("li");
    li.textContent = msg;
    issuesEl.appendChild(li);
  });

  const fillSelect = (sel, selected) => {
    sel.innerHTML = "";
    headers.forEach((h) => {
      const opt = document.createElement("option");
      opt.value = h;
      opt.textContent = h;
      sel.appendChild(opt);
    });
    if (selected && headers.includes(selected)) sel.value = selected;
  };

  fillSelect(ts, map?.timestamp || headers[0]);
  fillSelect(asset, map?.asset_id || headers[Math.min(1, headers.length - 1)]);
  site.innerHTML = '<option value="">— omit (use default site) —</option>';
  headers.forEach((h) => {
    const opt = document.createElement("option");
    opt.value = h;
    opt.textContent = h;
    site.appendChild(opt);
  });
  if (map?.site_id && headers.includes(map.site_id)) site.value = map.site_id;

  const keySet = new Set([ts.value, asset.value, site.value].filter(Boolean));
  const suggestedSensors = Array.isArray(map?.sensor_columns) ? map.sensor_columns : [];
  sensorsWrap.innerHTML = "";
  headers.forEach((h) => {
    if (keySet.has(h)) return;
    const id = `csvSensor_${h.replace(/[^a-z0-9_-]/gi, "_")}`;
    const label = document.createElement("label");
    label.className = "csv-sensor-chip";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.dataset.col = h;
    input.id = id;
    const checked = suggestedSensors.length ? suggestedSensors.includes(h) : true;
    input.checked = checked;
    const span = document.createElement("span");
    span.textContent = h;
    label.appendChild(input);
    label.appendChild(span);
    sensorsWrap.appendChild(label);
  });
}

function collectUploadMappingFromDom() {
  const ts = qs("#csvMapTimestamp");
  const asset = qs("#csvMapAsset");
  const site = qs("#csvMapSite");
  const sensorsWrap = qs("#csvMapSensors");
  if (!ts || !asset || !sensorsWrap) return null;
  const siteVal = site && site.value ? site.value : null;
  const sensor_columns = [];
  sensorsWrap.querySelectorAll('input[type="checkbox"]').forEach((cb) => {
    if (cb.checked && cb.dataset.col) sensor_columns.push(cb.dataset.col);
  });
  return {
    timestamp: ts.value,
    asset_id: asset.value,
    site_id: siteVal,
    sensor_columns,
  };
}

function validateUploadMapping(m) {
  if (!m || !m.timestamp || !m.asset_id) return "Choose a timestamp column and an asset/entity column.";
  if (!Array.isArray(m.sensor_columns) || m.sensor_columns.length < 1) {
    return "Select at least one numeric sensor column.";
  }
  return null;
}

async function uploadCsvToActiveRun() {
  const fileInput = qs("#csvFileInput");
  const file = state.uploadFile || fileInput?.files?.[0];
  if (!file) throw new Error("Choose a CSV file first");
  const runId = state.activeRun?.run_id;
  if (!runId) throw new Error("No active run found");
  if (!state.uploadCsv.headers.length) {
    await runCsvPreviewForFile(file);
  }
  renderUploadMappingPanel();
  const mapping = collectUploadMappingFromDom();
  const verr = validateUploadMapping(mapping);
  if (verr) throw new Error(verr);
  const started = await uploadCsvFileWithProgress(file, runId, mapping);
  const jobId = String(started.job_id || "");
  if (!jobId) {
    throw new Error("Upload started but did not return a job ID.");
  }
  return waitForIngestJob(jobId);
}

async function createRunFromForm() {
  const name = String(qs("#runNameInput")?.value || "").trim();
  const configRaw = String(qs("#runConfigInput")?.value || "").trim();
  const activate = Boolean(qs("#runActivateInput")?.checked);
  if (!name) throw new Error("Run name is required");
  let config = {};
  if (configRaw) {
    try {
      const parsed = JSON.parse(configRaw);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        config = parsed;
      } else {
        throw new Error("Run config must be a JSON object");
      }
    } catch (err) {
      throw new Error(`Invalid run config JSON: ${err.message || err}`);
    }
  }
  const out = await fetchJson(apiUrl("/runs", tenantScopeParams()), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, config, activate }),
  });
  return out.run;
}

function wireCsvMappingPanel() {
  const panel = qs("#csvMappingPanel");
  if (!panel || panel.dataset.wired === "1") return;
  panel.dataset.wired = "1";
  panel.addEventListener("change", (e) => {
    const t = e.target;
    state.uploadCsv.mapping = collectUploadMappingFromDom();
    if (t && ["csvMapTimestamp", "csvMapAsset", "csvMapSite"].includes(t.id)) {
      renderUploadMappingPanel();
    }
  });
}

function wireUploadInteractions() {
  const fileInput = qs("#csvFileInput");
  const zone = qs("#uploadDropZone");
  wireCsvMappingPanel();
  if (fileInput) {
    fileInput.addEventListener("change", () => {
      const f = fileInput.files && fileInput.files[0] ? fileInput.files[0] : null;
      setUploadFile(f);
    });
  }
  if (!zone) return;

  const stop = (evt) => {
    evt.preventDefault();
    evt.stopPropagation();
  };

  ["dragenter", "dragover"].forEach((name) => {
    zone.addEventListener(name, (evt) => {
      stop(evt);
      zone.classList.add("dragging");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    zone.addEventListener(name, (evt) => {
      stop(evt);
      zone.classList.remove("dragging");
    });
  });
  zone.addEventListener("drop", (evt) => {
    const dt = evt.dataTransfer;
    const file = dt && dt.files && dt.files[0] ? dt.files[0] : null;
    if (!file) return;
    setUploadFile(file);
    // Keep native file input in sync when possible.
    if (fileInput && dt && dt.files) {
      try {
        fileInput.files = dt.files;
      } catch (_err) {
        // Ignore browser restrictions; upload uses state.uploadFile fallback.
      }
    }
  });
}
