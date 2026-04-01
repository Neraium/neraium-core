const ONBOARDING_STEPS = ["welcome", "credentials", "identity", "ingestion", "test", "success"];

const onboardingState = {
  stepIndex: 0,
  selectedPath: "api_push",
  credentials: {
    customer_id: "",
    api_key: "",
    environment_label: "",
    verified: false,
  },
  profile: {
    customer_display_name: "",
    site_name: "",
    default_asset_id: "",
    asset_group: "",
  },
  status: null,
};

function onboardingCurrentStep() {
  return ONBOARDING_STEPS[onboardingState.stepIndex] || "welcome";
}

function onboardingCustomerId() {
  return customerIdValue(onboardingState.credentials.customer_id || state.tenant.customerId);
}

function onboardingStatusMessage(message, isError = false) {
  const el = qs("#onboardingStatus");
  if (!el) return;
  if (!message) {
    el.className = "status hidden";
    el.textContent = "";
    return;
  }
  el.className = `status ${isError ? "error" : "ok"}`;
  el.classList.remove("hidden");
  el.textContent = message;
}

function onboardingFillFromStatus(status) {
  if (!status || typeof status !== "object") return;
  onboardingState.status = status;
  const profile = status.profile || {};
  if (status.customer_id && !onboardingState.credentials.customer_id) onboardingState.credentials.customer_id = status.customer_id;
  if (status.ingestion_method) onboardingState.selectedPath = status.ingestion_method;
  if (profile.customer_display_name && !onboardingState.profile.customer_display_name) onboardingState.profile.customer_display_name = profile.customer_display_name;
  if (profile.site_name && !onboardingState.profile.site_name) onboardingState.profile.site_name = profile.site_name;
  if (profile.default_asset_id && !onboardingState.profile.default_asset_id) onboardingState.profile.default_asset_id = profile.default_asset_id;
  if (profile.asset_group && !onboardingState.profile.asset_group) onboardingState.profile.asset_group = profile.asset_group;
  onboardingState.credentials.verified = Boolean(status.credentials_verified);
}

async function refreshOnboardingStatus() {
  const customerId = onboardingCustomerId();
  const status = await fetchJson(apiUrl("/onboarding/status", { customer_id: customerId }));
  onboardingFillFromStatus(status);
  return status;
}

function onboardingApiPushContent() {
  const customerId = onboardingCustomerId();
  const runId = onboardingState.status?.active_run_id || state.activeRun?.run_id || "<run_id>";
  const endpoint = apiUrl("/ingest/frame", { customer_id: customerId, run_id: runId });
  const sample = JSON.stringify({
    timestamp: new Date().toISOString(),
    site_id: onboardingState.profile.site_name || "site-1",
    asset_id: onboardingState.profile.default_asset_id || "asset-1",
    sensor_values: { pressure: 32.4, flow: 14.8, temperature: 71.2 },
  }, null, 2);
  return `Endpoint\n${endpoint}\n\nHeaders\nX-API-Key: <your key>\nContent-Type: application/json\n\nPayload\n${sample}`;
}

function copyOnboardingText(text) {
  navigator.clipboard.writeText(String(text || "")).then(() => {
    createToast("Copied", "success");
  }).catch(() => {
    onboardingStatusMessage("Copy failed. Please copy manually.", true);
  });
}

function onboardingRenderStep() {
  const wizard = qs("#onboardingWizard");
  if (!wizard) return;
  const step = onboardingCurrentStep();
  const subtitle = qs("#onboardingStepSubtitle");
  const backBtn = qs("#onboardingBackBtn");
  const nextBtn = qs("#onboardingNextBtn");
  if (backBtn) backBtn.disabled = onboardingState.stepIndex === 0;
  if (nextBtn) nextBtn.textContent = step === "success" ? "Go to Dashboard" : "Next";
  if (subtitle) subtitle.textContent = `Step ${onboardingState.stepIndex + 1} of ${ONBOARDING_STEPS.length}`;

  if (step === "welcome") {
    wizard.innerHTML = `
      <p>Setup usually takes 3-5 minutes. Choose your preferred path; you can change it later.</p>
      <div class="onboarding-choice-grid">
        <button type="button" class="onboarding-choice-btn ${onboardingState.selectedPath === "api_push" ? "active" : ""}" data-onboarding-path="api_push">Connect via API</button>
        <button type="button" class="onboarding-choice-btn ${onboardingState.selectedPath === "csv_upload" ? "active" : ""}" data-onboarding-path="csv_upload">Upload CSV</button>
        <button type="button" class="onboarding-choice-btn ${onboardingState.selectedPath === "demo_replay" ? "active" : ""}" data-onboarding-path="demo_replay">Try demo replay</button>
      </div>`;
    return;
  }

  if (step === "credentials") {
    wizard.innerHTML = `
      <div class="onboarding-grid">
        <label>API key <input id="onbApiKey" type="password" autocomplete="off" placeholder="Enter API key" value="${escapeHtml(onboardingState.credentials.api_key || "")}" /></label>
        <label>Customer ID / code <input id="onbCustomerId" value="${escapeHtml(onboardingState.credentials.customer_id || state.tenant.customerId || "")}" /></label>
        <label>Environment label (optional) <input id="onbEnvLabel" value="${escapeHtml(onboardingState.credentials.environment_label || "")}" /></label>
      </div>
      <div class="onboarding-inline-actions"><button type="button" id="onbVerifyBtn">Verify credentials</button></div>
      <p class="subtitle">${onboardingState.credentials.verified ? "Credentials verified." : "Verify once before continuing."}</p>`;
    return;
  }

  if (step === "identity") {
    wizard.innerHTML = `
      <div class="onboarding-grid">
        <label>Customer display name <input id="onbDisplayName" value="${escapeHtml(onboardingState.profile.customer_display_name || "")}" /></label>
        <label>Site name <input id="onbSiteName" value="${escapeHtml(onboardingState.profile.site_name || "")}" /></label>
        <label>Default asset/system name <input id="onbAssetId" value="${escapeHtml(onboardingState.profile.default_asset_id || "")}" /></label>
        <label>Asset group / deployment label (optional) <input id="onbAssetGroup" value="${escapeHtml(onboardingState.profile.asset_group || "")}" /></label>
      </div>`;
    return;
  }

  if (step === "ingestion") {
    const apiDetails = onboardingApiPushContent();
    wizard.innerHTML = `
      <p>Choose how you want to start sending telemetry.</p>
      <div class="onboarding-choice-grid">
        <button type="button" class="onboarding-choice-btn ${onboardingState.selectedPath === "api_push" ? "active" : ""}" data-onboarding-path="api_push">API Push</button>
        <button type="button" class="onboarding-choice-btn ${onboardingState.selectedPath === "csv_upload" ? "active" : ""}" data-onboarding-path="csv_upload">CSV Upload</button>
        <button type="button" class="onboarding-choice-btn ${onboardingState.selectedPath === "demo_replay" ? "active" : ""}" data-onboarding-path="demo_replay">Demo Replay</button>
      </div>
      ${onboardingState.selectedPath === "api_push" ? `<div class="onboarding-code"><pre>${escapeHtml(apiDetails)}</pre></div><div class="onboarding-inline-actions"><button type="button" id="onbCopyApiInfo">Copy API details</button></div>` : ""}
      ${onboardingState.selectedPath === "csv_upload" ? '<div class="onboarding-link-row"><a class="button-link secondary" href="/upload">Continue in Upload Telemetry</a></div>' : ""}
      ${onboardingState.selectedPath === "demo_replay" ? '<div class="onboarding-link-row"><a class="button-link secondary" href="/validation?replay=1">Open Validation Replay</a></div>' : ""}`;
    return;
  }

  if (step === "test") {
    wizard.innerHTML = `
      <p>Run a quick connectivity check to create/reuse your active run and send one test telemetry frame.</p>
      <div class="onboarding-inline-actions"><button type="button" id="onbBootstrapRunBtn" class="secondary">Ensure active run</button><button type="button" id="onbSendTestBtn">Send test frame</button></div>
      <p class="subtitle">${escapeHtml(onboardingState.status?.active_run_id ? `Active run: ${onboardingState.status.active_run_id}` : "No active run yet.")}</p>`;
    return;
  }

  wizard.innerHTML = `
    <p>You're ready to operate.</p>
    <dl class="onboarding-kv">
      <div><dt>Verified customer ID</dt><dd>${escapeHtml(onboardingCustomerId())}</dd></div>
      <div><dt>Onboarding status</dt><dd>${onboardingState.status?.ready ? "Ready" : "In progress"}</dd></div>
      <div><dt>Active run ID</dt><dd>${escapeHtml(onboardingState.status?.active_run_id || "Not created")}</dd></div>
      <div><dt>Telemetry connection</dt><dd>${onboardingState.status?.test_frame_sent ? "Connected" : onboardingState.selectedPath === "api_push" ? "Pending test" : "Pending next step"}</dd></div>
    </dl>
    <div class="onboarding-link-row">
      <a class="button-link primary" href="/dashboard">Go to Dashboard</a>
      <a class="button-link secondary" href="/upload">Upload Telemetry</a>
      <a class="button-link secondary" href="/validation">Open Validation</a>
    </div>`;
}

async function onboardingPersistProfile() {
  const customerId = onboardingCustomerId();
  const payload = {
    customer_display_name: onboardingState.profile.customer_display_name,
    site_name: onboardingState.profile.site_name,
    default_asset_id: onboardingState.profile.default_asset_id,
    asset_group: onboardingState.profile.asset_group || null,
    ingestion_method: onboardingState.selectedPath,
  };
  await fetchJson(apiUrl("/onboarding/profile", { customer_id: customerId }), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await refreshOnboardingStatus();
}

function onboardingReadInputs() {
  onboardingState.credentials.api_key = qs("#onbApiKey")?.value || onboardingState.credentials.api_key;
  onboardingState.credentials.customer_id = qs("#onbCustomerId")?.value || onboardingState.credentials.customer_id;
  onboardingState.credentials.environment_label = qs("#onbEnvLabel")?.value || onboardingState.credentials.environment_label;
  onboardingState.profile.customer_display_name = qs("#onbDisplayName")?.value || onboardingState.profile.customer_display_name;
  onboardingState.profile.site_name = qs("#onbSiteName")?.value || onboardingState.profile.site_name;
  onboardingState.profile.default_asset_id = qs("#onbAssetId")?.value || onboardingState.profile.default_asset_id;
  onboardingState.profile.asset_group = qs("#onbAssetGroup")?.value || onboardingState.profile.asset_group;
}

async function onboardingHandleVerify() {
  onboardingReadInputs();
  if (!onboardingState.credentials.customer_id || !onboardingState.credentials.api_key) {
    onboardingStatusMessage("Enter customer ID and API key.", true);
    return;
  }
  setLoading(true, "Verifying credentials...");
  try {
    await fetchJson(apiUrl("/onboarding/verify"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        customer_id: onboardingState.credentials.customer_id,
        api_key: onboardingState.credentials.api_key,
        environment_label: onboardingState.credentials.environment_label || null,
      }),
    });
    onboardingState.credentials.verified = true;
    onboardingState.credentials.api_key = "";
    state.tenant.customerId = customerIdValue(onboardingState.credentials.customer_id);
    window.localStorage.setItem(TENANT_STORAGE_KEY, state.tenant.customerId);
    renderTenantControls();
    await refreshOnboardingStatus();
    onboardingStatusMessage("Credentials verified.");
    onboardingRenderStep();
  } catch (err) {
    onboardingStatusMessage(String(err.message || err), true);
  } finally {
    setLoading(false);
  }
}

async function onboardingHandleBootstrapRun() {
  setLoading(true, "Preparing active run...");
  try {
    await fetchJson(apiUrl("/onboarding/bootstrap-run", { customer_id: onboardingCustomerId() }), { method: "POST" });
    await refreshOnboardingStatus();
    await loadRuns();
    onboardingStatusMessage("Active run is ready.");
    onboardingRenderStep();
  } catch (err) {
    onboardingStatusMessage(String(err.message || err), true);
  } finally {
    setLoading(false);
  }
}

async function onboardingHandleTestFrame() {
  setLoading(true, "Sending test telemetry frame...");
  try {
    const out = await fetchJson(apiUrl("/onboarding/test-frame", { customer_id: onboardingCustomerId() }), { method: "POST" });
    await refreshOnboardingStatus();
    await loadRuns();
    onboardingStatusMessage(out.message || (out.ok ? "Test frame accepted." : "Test frame failed."), !out.ok);
    onboardingRenderStep();
  } catch (err) {
    onboardingStatusMessage(String(err.message || err), true);
  } finally {
    setLoading(false);
  }
}

async function onboardingNextStep() {
  onboardingReadInputs();
  const step = onboardingCurrentStep();
  if (step === "credentials" && !onboardingState.credentials.verified) {
    onboardingStatusMessage("Verify credentials before moving on.", true);
    return;
  }
  if (step === "identity") {
    if (!onboardingState.profile.customer_display_name || !onboardingState.profile.site_name || !onboardingState.profile.default_asset_id) {
      onboardingStatusMessage("Fill in required system identity fields.", true);
      return;
    }
    await onboardingPersistProfile();
  }
  if (step === "ingestion") {
    await onboardingPersistProfile();
    if (onboardingState.selectedPath !== "api_push") {
      await onboardingHandleBootstrapRun();
    }
  }
  if (step === "test" && onboardingState.selectedPath === "api_push" && !onboardingState.status?.test_frame_sent) {
    onboardingStatusMessage("Send a test frame to finish API onboarding.", true);
    return;
  }
  if (step === "success") {
    window.location.href = `/dashboard?customer_id=${encodeURIComponent(onboardingCustomerId())}`;
    return;
  }
  onboardingState.stepIndex = Math.min(ONBOARDING_STEPS.length - 1, onboardingState.stepIndex + 1);
  onboardingStatusMessage("");
  onboardingRenderStep();
}

function onboardingBackStep() {
  onboardingReadInputs();
  onboardingState.stepIndex = Math.max(0, onboardingState.stepIndex - 1);
  onboardingStatusMessage("");
  onboardingRenderStep();
}

function wireOnboardingEvents() {
  const backBtn = qs("#onboardingBackBtn");
  const nextBtn = qs("#onboardingNextBtn");
  if (backBtn && backBtn.dataset.wired !== "1") {
    backBtn.dataset.wired = "1";
    backBtn.addEventListener("click", () => onboardingBackStep());
  }
  if (nextBtn && nextBtn.dataset.wired !== "1") {
    nextBtn.dataset.wired = "1";
    nextBtn.addEventListener("click", async () => {
      await onboardingNextStep();
    });
  }
  const wizard = qs("#onboardingWizard");
  if (wizard && wizard.dataset.wired !== "1") {
    wizard.dataset.wired = "1";
    wizard.addEventListener("click", async (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) return;
      const path = target.getAttribute("data-onboarding-path");
      if (path) {
        onboardingState.selectedPath = path;
        onboardingRenderStep();
        return;
      }
      if (target.id === "onbVerifyBtn") {
        await onboardingHandleVerify();
      } else if (target.id === "onbBootstrapRunBtn") {
        await onboardingHandleBootstrapRun();
      } else if (target.id === "onbSendTestBtn") {
        await onboardingHandleTestFrame();
      } else if (target.id === "onbCopyApiInfo") {
        copyOnboardingText(onboardingApiPushContent());
      }
    });
  }
}

async function loadOnboardingPage() {
  onboardingStatusMessage("");
  if (!onboardingState.credentials.customer_id) {
    onboardingState.credentials.customer_id = customerIdValue(state.tenant.customerId);
  }
  const status = await refreshOnboardingStatus();
  if (status.ready) onboardingState.stepIndex = ONBOARDING_STEPS.length - 1;
  onboardingRenderStep();
}
