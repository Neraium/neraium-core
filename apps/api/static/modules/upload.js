function updateUploadRunInfo() {
  const info = qs("#uploadRunInfo");
  if (!info) return;
  if (state.activeRun?.run_id) {
    const base = `Active run: ${state.activeRun.name} (${state.activeRun.run_id})`;
    info.textContent = base;
  } else {
    info.textContent = "No active run selected.";
  }
  renderSiiIngestionUi();
}

const UPLOAD_STAGES = new Set([
  "idle",
  "file_selected",
  "previewing",
  "preview_blocked",
  "preview_ready",
  "uploading",
  "ingesting",
  "partial_success",
  "failed",
  "completed",
]);

const MOCK_SII_CLI_RESPONSE = {
  exit_code: 1,
  frames_succeeded: 42,
  frames_failed: 8,
  results_emitted: 40,
  input_empty: false,
  all_failed: false,
  partial_success: true,
  errors_truncated: true,
  total_error_count: 8,
  returned_error_count: 5,
  ingest_errors: [
    { index: 4, code: "PARSE_ERROR", message: "timestamp is missing timezone", source_record_id: "rec-004" },
    { index: 7, code: "PARSE_ERROR", message: "sensor column is not numeric", source_record_id: "rec-007" },
    { index: 11, code: "SCHEMA_MISSING_FIELD", message: "asset_id is required" },
    { index: 13, code: "RANGE_ERROR", message: "temperature exceeded allowed range", source_record_id: "rec-013" },
    { index: 20, code: "PARSE_ERROR", message: "invalid numeric formatting", source_record_id: "rec-020" },
  ],
};

const siiIngestionUiState = {
  response: null,
  errorCodeFilter: "all",
  showRawJson: false,
  expandedCodes: new Set(),
};

function setUploadStage(stage) {
  const normalized = UPLOAD_STAGES.has(stage) ? stage : "idle";
  state.uploadCsv.stage = normalized;
}

function isStructuralFailurePayload(payload) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return false;
  if (!Object.prototype.hasOwnProperty.call(payload, "error")) return false;
  if (typeof payload.error !== "string" || !payload.error.trim()) return false;
  const summaryKeys = new Set([
    "frames_succeeded",
    "frames_failed",
    "results_emitted",
    "input_empty",
    "all_failed",
    "partial_success",
    "ingest_errors",
    "errors_truncated",
    "total_error_count",
    "returned_error_count",
  ]);
  return !Object.keys(payload).some((key) => summaryKeys.has(key));
}

function toNonNegativeInt(value, fallback = 0) {
  const num = Number(value);
  if (!Number.isFinite(num) || num < 0) return fallback;
  return Math.floor(num);
}

function normalizeIngestErrorEntry(err, idx) {
  if (!err || typeof err !== "object") {
    return {
      index: idx,
      code: "UNKNOWN",
      message: "",
      source_record_id: null,
      _row: idx,
    };
  }
  return {
    index: err.index ?? idx,
    code: String(err.code || "UNKNOWN"),
    message: String(err.message || ""),
    source_record_id: err.source_record_id == null ? null : String(err.source_record_id),
    _row: idx,
  };
}

function normalizeIngestionPayload(response) {
  if (!response || typeof response !== "object" || Array.isArray(response)) {
    return { kind: "invalid", summary: null, structuralError: null };
  }
  if (isStructuralFailurePayload(response)) {
    return {
      kind: "structural",
      summary: null,
      structuralError: String(response.error || "Unknown structural error."),
    };
  }
  const ingestErrors = Array.isArray(response.ingest_errors)
    ? response.ingest_errors.map((err, idx) => normalizeIngestErrorEntry(err, idx))
    : [];
  const totalErrorCountRaw =
    response.total_error_count == null ? ingestErrors.length : toNonNegativeInt(response.total_error_count, ingestErrors.length);
  const returnedErrorCountRaw =
    response.returned_error_count == null ? ingestErrors.length : toNonNegativeInt(response.returned_error_count, ingestErrors.length);
  return {
    kind: "summary",
    structuralError: null,
    summary: {
      exit_code: Number.isFinite(Number(response.exit_code)) ? Number(response.exit_code) : null,
      frames_succeeded: toNonNegativeInt(response.frames_succeeded, 0),
      frames_failed: toNonNegativeInt(response.frames_failed, 0),
      results_emitted: toNonNegativeInt(response.results_emitted, 0),
      input_empty: Boolean(response.input_empty),
      all_failed: Boolean(response.all_failed),
      partial_success: Boolean(response.partial_success),
      ingest_errors: ingestErrors,
      errors_truncated: Boolean(response.errors_truncated),
      total_error_count: Math.max(totalErrorCountRaw, returnedErrorCountRaw),
      returned_error_count: returnedErrorCountRaw,
      raw: response,
    },
  };
}

function deriveIngestionStatusViewModel(response) {
  const normalized = normalizeIngestionPayload(response);
  if (normalized.kind === "invalid") {
    return { icon: "❌", label: "Failed (non-structural)", tone: "failed", description: "No valid ingestion response payload." };
  }
  if (normalized.kind === "structural") {
    return { icon: "💥", label: "Structural Failure", tone: "structural", description: "The CLI returned a structural failure object." };
  }
  const summary = normalized.summary;
  const exitCode = summary.exit_code;
  if (exitCode === 0) {
    return { icon: "✅", label: "Success", tone: "success", description: "All rows were ingested successfully." };
  }
  if (summary.partial_success) {
    return { icon: "⚠️", label: "Partial Success", tone: "partial", description: "Some frames succeeded and some frames failed." };
  }
  if (exitCode === 1 || summary.all_failed || summary.input_empty) {
    return { icon: "❌", label: "Failed (non-structural)", tone: "failed", description: "Ingestion completed with non-structural failure status." };
  }
  return { icon: "❌", label: "Failed (non-structural)", tone: "failed", description: "Unable to classify response with available flags." };
}

function toGroupedIngestErrors(errors, filterCode = "all") {
  const list = Array.isArray(errors) ? errors : [];
  const groups = new Map();
  list.forEach((err) => {
    const code = String(err?.code || "UNKNOWN");
    if (!groups.has(code)) groups.set(code, []);
    groups.get(code).push(err);
  });
  const grouped = [...groups.entries()].map(([code, rows]) => ({ code, count: rows.length, rows }));
  grouped.sort((a, b) => b.count - a.count || a.code.localeCompare(b.code));
  if (filterCode && filterCode !== "all") return grouped.filter((group) => group.code === filterCode);
  return grouped;
}

function renderSiiIngestionUi() {
  const root = qs("#siiIngestionUiRoot");
  if (!root) return;
  const response = siiIngestionUiState.response;
  if (!response) {
    root.innerHTML = '<p class="subtitle">Run an ingestion or load the mock response to inspect the SII summary UI.</p>';
    return;
  }
  const normalized = normalizeIngestionPayload(response);
  const vm = deriveIngestionStatusViewModel(response);
  if (normalized.kind === "invalid") {
    root.innerHTML = `
      <article class="sii-status-panel" data-tone="failed">
        <h4>${vm.icon} ${escapeHtml(vm.label)}</h4>
        <p>${escapeHtml(vm.description)}</p>
      </article>
      <details class="sii-raw-json-toggle"${siiIngestionUiState.showRawJson ? " open" : ""}>
        <summary>Developer/debug: raw JSON response</summary>
        <pre>${escapeHtml(JSON.stringify(response, null, 2))}</pre>
      </details>
    `;
    return;
  }
  if (normalized.kind === "structural") {
    root.innerHTML = `
      <article class="sii-structural-failure">
        <h4>${vm.icon} ${escapeHtml(vm.label)}</h4>
        <p>${escapeHtml(normalized.structuralError)}</p>
      </article>
      <details class="sii-raw-json-toggle"${siiIngestionUiState.showRawJson ? " open" : ""}>
        <summary>Developer/debug: raw JSON response</summary>
        <pre>${escapeHtml(JSON.stringify(response, null, 2))}</pre>
      </details>
    `;
    return;
  }
  const summary = normalized.summary;
  const groupedErrors = toGroupedIngestErrors(summary.ingest_errors, siiIngestionUiState.errorCodeFilter);
  const topErrorCode = groupedErrors.length ? groupedErrors[0].code : "";
  const totalErrorCount = summary.total_error_count;
  const returnedErrorCount = summary.returned_error_count;
  const truncationBanner =
    summary.errors_truncated
      ? `<div class="sii-truncation-warning"><strong>⚠️ Error list truncated.</strong> Showing ${returnedErrorCount} returned errors out of ${totalErrorCount} total errors reported by the CLI.</div>`
      : "";
  const metrics = [
    ["Frames succeeded", summary.frames_succeeded],
    ["Frames failed", summary.frames_failed],
    ["Results emitted", summary.results_emitted],
    ["Total error count", totalErrorCount],
  ];
  const metricHtml = metrics
    .map(([key, value]) => `<article class="sii-metric"><p>${escapeHtml(key)}</p><strong>${escapeHtml(String(value))}</strong></article>`)
    .join("");
  const availableCodeGroups = toGroupedIngestErrors(summary.ingest_errors);
  const filterOptions = ['<option value="all">All error codes</option>']
    .concat(
      availableCodeGroups.map(
        (group) =>
          `<option value="${escapeHtml(group.code)}"${siiIngestionUiState.errorCodeFilter === group.code ? " selected" : ""}>${escapeHtml(group.code)} (${group.count})</option>`,
      ),
    )
    .join("");
  const groupedHtml = groupedErrors.length
    ? groupedErrors
      .map((group) => {
        const isExpanded = siiIngestionUiState.expandedCodes.has(group.code);
        const rowHtml = group.rows
          .map(
            (row) => `
              <tr>
                <td>${escapeHtml(String(row.index ?? "—"))}</td>
                <td><span class="sii-error-code-badge">${escapeHtml(String(row.code || "UNKNOWN"))}</span></td>
                <td>${escapeHtml(String(row.message || ""))}</td>
                <td>${escapeHtml(String(row.source_record_id || "—"))}</td>
                <td><button type="button" class="secondary sii-copy-row-btn" data-copy-index="${row._row}">Copy</button></td>
              </tr>
            `,
          )
          .join("");
        return `
          <details class="sii-error-group"${isExpanded ? " open" : ""} data-error-code="${escapeHtml(group.code)}">
            <summary><span>${escapeHtml(group.code)}</span> <strong>${group.count}</strong></summary>
            <div class="table-wrap">
              <table class="data-table">
                <thead><tr><th>Index</th><th>Code</th><th>Message</th><th>Source record ID</th><th></th></tr></thead>
                <tbody>${rowHtml}</tbody>
              </table>
            </div>
          </details>
        `;
      })
      .join("")
    : '<article class="sii-empty-state"><p>No ingest errors were returned in this summary.</p></article>';
  const emptyInputState = summary.input_empty
    ? '<article class="sii-empty-state"><p>Input was empty: no records were available for ingestion in this run.</p></article>'
    : "";

  root.innerHTML = `
    <article class="sii-status-panel" data-tone="${escapeHtml(vm.tone)}">
      <h4>${vm.icon} ${escapeHtml(vm.label)}</h4>
      <p>${escapeHtml(vm.description)}</p>
      ${topErrorCode ? `<p class="subtitle">Most frequent error type: <span class="sii-error-code-badge">${escapeHtml(topErrorCode)}</span></p>` : ""}
    </article>
    <section class="sii-metrics-grid">${metricHtml}</section>
    <section class="sii-error-toolbar">
      <label>Filter by error code
        <select id="siiErrorCodeFilter">${filterOptions}</select>
      </label>
    </section>
    ${emptyInputState}
    ${truncationBanner}
    <section class="sii-error-groups">${groupedHtml}</section>
    <details class="sii-raw-json-toggle"${siiIngestionUiState.showRawJson ? " open" : ""}>
      <summary>Developer/debug: raw JSON response</summary>
      <pre>${escapeHtml(JSON.stringify(response, null, 2))}</pre>
    </details>
  `;

  const filter = qs("#siiErrorCodeFilter");
  if (filter) {
    filter.addEventListener("change", (evt) => {
      siiIngestionUiState.errorCodeFilter = String(evt.target.value || "all");
      renderSiiIngestionUi();
    });
  }
  qsa(".sii-error-group").forEach((details) => {
    details.addEventListener("toggle", () => {
      const code = String(details.dataset.errorCode || "");
      if (!code) return;
      if (details.open) siiIngestionUiState.expandedCodes.add(code);
      else siiIngestionUiState.expandedCodes.delete(code);
    });
  });
  qsa(".sii-copy-row-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const idx = Number(btn.dataset.copyIndex);
      const errors = Array.isArray(summary.ingest_errors) ? summary.ingest_errors : [];
      const row = errors[idx];
      if (!row) return;
      try {
        await navigator.clipboard.writeText(JSON.stringify(row, null, 2));
        setStatus("Error row copied to clipboard.", false, true);
      } catch (_err) {
        setStatus("Clipboard copy failed in this browser context.", true, true);
      }
    });
  });
  qsa(".sii-raw-json-toggle").forEach((details) => {
    details.addEventListener("toggle", () => {
      siiIngestionUiState.showRawJson = details.open;
    });
  });
}

function applySiiCliResponse(response) {
  siiIngestionUiState.response = response && typeof response === "object" ? response : null;
  siiIngestionUiState.errorCodeFilter = "all";
  siiIngestionUiState.expandedCodes = new Set();
  renderSiiIngestionUi();
}

function wireSiiIngestionUiEvents() {
  const mockBtn = qs("#loadMockSiiResponseBtn");
  if (mockBtn && mockBtn.dataset.wired !== "1") {
    mockBtn.dataset.wired = "1";
    mockBtn.addEventListener("click", () => {
      applySiiCliResponse(MOCK_SII_CLI_RESPONSE);
      setStatus("Loaded mock SII CLI response for UI validation.", false, true);
    });
  }
  renderSiiIngestionUi();
}

function structuredErrorFrom(err, fallbackStage = "request") {
  const apiErr = err && err.apiError ? err.apiError : null;
  const msg = String(err?.message || fallback);
  const rawResponse = String(apiErr?.responseText || "");
  let parsed = null;
  if (rawResponse) {
    try {
      parsed = JSON.parse(rawResponse);
    } catch (_err) {
      parsed = null;
    }
  }
  const correlation = parsed && parsed.correlation_id ? ` (ref ${String(parsed.correlation_id)})` : "";
  const structuredMessage = parsed && parsed.message ? String(parsed.message) : "";
  const actionable = parsed && parsed.actionable_detail ? String(parsed.actionable_detail) : "";
  const composed = [structuredMessage, actionable].filter(Boolean).join(" ");
  if (!apiErr) return msg;
  if (apiErr.status === 422) return `Upload validation failed. ${composed || msg}${correlation}`;
  if (apiErr.status === 413) return "Upload is too large for this environment. Split the CSV and retry.";
  if (composed) return `${composed}${correlation}`;
  return msg;
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

function resetUploadFlowState() {
  clearUploadJobPolling();
  state.uploadJob.id = null;
  state.uploadJob.active = false;
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
        const message =
          (body && typeof body.message === "string" && body.message) ||
          (body && typeof body.actionable_detail === "string" && body.actionable_detail) ||
          (body && typeof body.detail === "string" && body.detail) ||
          `HTTP ${xhr.status}`;
        const err = new Error(message);
        err.apiError = {
          status: xhr.status,
          responseText: JSON.stringify(body || {}),
        };
        reject(err);
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
        if (status === "processing" || status === "queued") setUploadStage("ingesting");
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
  setUploadStage(file ? "file_selected" : "idle");
  state.uploadFile = file || null;
  const el = qs("#selectedFileName");
  if (!el) return;
  el.textContent = state.uploadFile ? `${state.uploadFile.name} (${state.uploadFile.size} bytes)` : "No file selected";
  if (!file) {
    resetUploadFlowState();
    state.uploadCsv.preview = null;
    state.uploadCsv.headers = [];
    state.uploadCsv.issues = [];
    state.uploadCsv.warnings = [];
    state.uploadCsv.requiresConfirmation = false;
    state.uploadCsv.mapping = null;
    setUploadStage("idle");
    renderUploadMappingPanel();
    return;
  }
  setUploadStage("file_selected");
  runCsvPreviewForFile(file).catch((err) => {
    resetUploadFlowState();
    setUploadStage("failed");
    const normalized = structuredErrorFrom(err, "preview");
    setStatus(operatorErrorMessage(err, "CSV preview failed."), true, true);
    if (normalized.stage === "preview") {
      state.uploadCsv.issues = normalized.issue_details.map((i) => String(i.message || "Preview validation issue."));
      renderUploadMappingPanel();
    }
  });
}

async function runCsvPreviewForFile(file) {
  setUploadStage("previewing");
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
  setUploadStage(state.uploadCsv.requiresConfirmation ? "preview_blocked" : "preview_ready");
  const hasBlockingIssues = state.uploadCsv.issues.length > 0 && !state.uploadCsv.mapping;
  const apiPreviewState = String(out.preview_state || "");
  if (apiPreviewState === "preview_blocked" || hasBlockingIssues) setUploadStage("preview_blocked");
  else setUploadStage("preview_ready");
  renderUploadMappingPanel();
  if (state.uploadCsv.requiresConfirmation) {
    const guidance =
      out.actionable_detail ||
      "Preview found ambiguous mapping. Review timestamp, asset/entity, and sensor columns before upload.";
    setStatus(guidance, true, false);
  } else if (state.uploadCsv.issues.length && !out.suggested_mapping) {
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
  if (state.uploadCsv.requiresConfirmation) {
    setUploadStage("preview_blocked");
  }
  renderUploadMappingPanel();
  const mapping = collectUploadMappingFromDom();
  const verr = validateUploadMapping(mapping);
  if (verr) throw new Error(verr);
  setUploadStage("uploading");
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
  wireSiiIngestionUiEvents();
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

function wireUploadFormEvents() {
  const form = qs("#csvUploadForm");
  if (!form || form.dataset.wired === "1") return;
  form.dataset.wired = "1";
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      resetUploadAttemptState();
      state.uploadJob.active = true;
      resetUploadFlowState();
      setUploadStage("uploading");
      setUploadProgressUI({
        visible: true,
        mode: "uploading",
        statusText: "Preparing upload...",
        uploadedBytes: 0,
        totalBytes: state.uploadFile?.size || null,
      });
      const out = await uploadCsvToActiveRun();
      const route = getRoute();
      await loadDashboard();
      if (route.page === "run-detail" && route.runId) {
        await loadRunDetail(route.runId);
      }
      const fileInput = qs("#csvFileInput");
      if (fileInput) fileInput.value = "";
      setUploadFile(null);
      const status = String(out.status || "completed");
      setUploadStage(status === "failed" ? "failed" : status === "partial_success" ? "partial_success" : "completed");
      const rowsProcessed = Number(out.rows_processed || 0);
      const rowsSucceeded = Number(out.rows_succeeded || 0);
      const rowsFailed = Number(out.rows_failed || 0);
      if (status === "failed") {
        setStatus(out.message || `CSV ingest failed (${rowsFailed} rows failed).`, true, true);
      } else if (status === "partial_success") {
        setStatus(out.message || `CSV ingest partial success (${rowsSucceeded} succeeded, ${rowsFailed} failed).`, true, true);
      } else {
        setStatus(out.message || `CSV ingested (${rowsProcessed} rows processed).`, false, true);
      }
      setUploadProgressUI({
        visible: true,
        mode: status,
        statusText: out.message || `Ingest ${status}.`,
        uploadedBytes: Number(out.upload_bytes_received || 0),
        totalBytes: out.upload_bytes_total,
        rowsProcessed,
        rowsSucceeded,
        rowsFailed,
        errorSamples: out.error_samples || [],
      });
      applySiiCliResponse(out);
    } catch (err) {
      resetUploadFlowState();
      setUploadStage(state.uploadCsv?.issues?.length ? "preview_blocked" : "failed");
      setUploadProgressUI({
        visible: !isPreviewFailure,
        mode: "failed",
        statusText: operatorErrorMessage(err, isPreviewFailure ? "CSV preview blocked upload." : "Ingest failed."),
        errorSamples: isPreviewFailure ? [] : [{ row: "-", message: operatorErrorMessage(err) }],
      });
      const failureCopy = isPreviewFailure
        ? `CSV preview is blocked. ${operatorErrorMessage(err, "Review mapping and retry preview.")}`
        : operatorErrorMessage(err);
      setStatus(failureCopy, true, true);
      const rawText = String(err?.apiError?.responseText || "");
      if (rawText) {
        try {
          applySiiCliResponse(JSON.parse(rawText));
        } catch (_parseErr) {
          // Non-JSON backend payload: keep current ingestion panel as-is.
        }
      }
    } finally {
      resetUploadFlowState();
    }
  });
}
