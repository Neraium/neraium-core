// Core module - initializes global state and helper functions

// Initialize global state
window.state = {
  activeRun: null,
  dashboardRecent: [],
  dashboardCurrentAlertStatus: null,
  runRecent: [],
  tenant: {
    customerId: null,
    siteId: null,
  },
  ui: {
    currentPage: 'dashboard',
    recentResultsCache: new Map(),
    recentResultsInflight: new Map(),
    runDetailObserver: null,
    runDetailBackgroundHistoryLoaded: false,
    runDetailBackgroundHistoryPending: false,
    runDetailHydratedSections: {},
  },
  demo: {
    enabled: false,
    replay: {
      uiState: 'idle',
      totalFrames: 0,
      currentFrame: 0,
    }
  },
  charts: {
    drift: null,
    composite: null,
  },
  runDetailView: {
    runId: null,
    section: 'overview',
    range: '200',
    search: '',
    sort: 'timestamp_desc',
    flowPlaybackMode: 'live',
    flowPlaybackSpeed: 1,
    flowHistoryEnabled: true,
  },
  geometry3d: {
    temporalFlow: null,
  },
  runtimeDegraded: false,
};

// Helper function: querySelector
window.qs = function(sel) {
  return document.querySelector(sel);
};

// Helper function: querySelectorAll
window.qsa = function(sel) {
  return Array.from(document.querySelectorAll(sel));
};

// Helper function: set loading overlay
window.setLoading = function(isLoading, message = '') {
  const overlay = window.qs('#loadingOverlay');
  const loadingMsg = window.qs('#loadingMessage');

  if (!overlay) return;

  if (isLoading) {
    overlay.classList.remove('hidden');
    if (loadingMsg && message) {
      loadingMsg.textContent = message;
    }
  } else {
    overlay.classList.add('hidden');
  }
};

// Helper function: set status message
window.setStatus = function(message = '', isError = false, dismiss = false) {
  const globalStatus = window.qs('#globalStatus');
  const runtimeBanner = window.qs('#runtimeModeBanner');

  if (!globalStatus) return;

  if (!message || message.trim() === '') {
    globalStatus.classList.add('hidden');
    return;
  }

  globalStatus.className = isError ? 'status status-error' : 'status status-info';
  globalStatus.textContent = message;
  globalStatus.setAttribute('role', 'alert');
  globalStatus.setAttribute('aria-live', 'polite');
  globalStatus.classList.remove('hidden');

  if (dismiss) {
    setTimeout(() => {
      globalStatus.classList.add('hidden');
    }, 5000);
  }
};

// Helper function: get current route
window.getRoute = function() {
  const path = window.location.pathname;
  const params = new URLSearchParams(window.location.search);

  let page = 'dashboard';
  if (path.includes('/upload')) page = 'upload';
  else if (path.includes('/validation')) page = 'validation';
  else if (path.includes('/onboarding')) page = 'onboarding';
  else if (path.includes('/app/runs')) page = 'runs';
  else if (path.includes('/app/run')) {
    page = 'run-detail';
  } else if (path.includes('/app/result')) {
    page = 'result-detail';
  }

  return {
    page,
    runId: params.get('run_id') || params.get('runId'),
    resultId: params.get('result_id') || params.get('resultId'),
    full: path,
  };
};

// Helper function: set current page visibility
window.setPage = function(pageId) {
  window.state.ui.currentPage = pageId;
  window.qsa('[id^="page-"]').forEach(el => {
    el.classList.add('hidden');
  });
  const pageEl = window.qs(`#page-${pageId}`);
  if (pageEl) {
    pageEl.classList.remove('hidden');
  }
};

// Helper function: get scope parameters for API calls
window.tenantScopeParams = function(extra = {}) {
  const params = { ...extra };
  if (window.state.tenant?.customerId) {
    params.customer_id = window.state.tenant.customerId;
  }
  if (window.state.tenant?.siteId) {
    params.site_id = window.state.tenant.siteId;
  }
  return params;
};

// Helper function: render controls for tenant selection
window.renderTenantControls = function() {
  // Placeholder for tenant controls rendering
  console.log('Tenant controls rendered');
};

// Helper function: live clock updates
window.startLiveClock = function() {
  const updateTime = () => {
    const now = new Date();
    const timeEl = window.qs('#liveTimestamp');
    if (timeEl) {
      timeEl.textContent = now.toUTCString().split(' ').slice(4, 5)[0] || '--:--:--';
    }
  };

  updateTime();
  setInterval(updateTime, 1000);
};

// Helper function: read tenant from storage
window.readTenantFromStorage = function() {
  const stored = localStorage.getItem('neraium_tenant');
  if (stored) {
    try {
      window.state.tenant = JSON.parse(stored);
    } catch (e) {
      // Ignore parse errors
    }
  }
};

// Helper function: load runs list
window.loadRuns = async function() {
  try {
    const { apiUrl, fetchJson } = window.NeraiumApi;
    const params = window.tenantScopeParams({ limit: 20 });
    const env = await fetchJson(apiUrl('/runs', params));

    if (env && Array.isArray(env.runs)) {
      // Store the runs for the list view
      window.state.runs = env.runs;
      if (env.latest) {
        window.state.activeRun = env.latest;
      }
    }
  } catch (err) {
    console.error('Error loading runs:', err);
  }
};

// Helper function: render runs list
window.renderRunsList = function() {
  const tbody = window.qs('#runsBody');
  if (!tbody) return;

  tbody.innerHTML = '';
  if (!window.state.runs || window.state.runs.length === 0) {
    const emptyHint = window.qs('#runsEmptyHint');
    if (emptyHint) emptyHint.classList.remove('hidden');
    return;
  }

  window.state.runs.forEach(run => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${window.escapeHtml(run.name || run.run_id)}</td>
      <td>${window.escapeHtml(run.run_id)}</td>
      <td>${window.escapeHtml(run.status || 'active')}</td>
      <td>${window.escapeHtml(run.created_at || '-')}</td>
      <td><a href="/app/run?run_id=${encodeURIComponent(run.run_id)}">Details</a></td>
    `;
    tbody.appendChild(tr);
  });
};

// Helper function: escape HTML
window.escapeHtml = function(text) {
  const map = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  };
  return String(text).replace(/[&<>"']/g, m => map[m]);
};

// Helper function: read value from query params
window.routeScopeFromQuery = function() {
  const params = new URLSearchParams(window.location.search);
  return {
    customer_id: params.get('customer_id'),
    site_id: params.get('site_id'),
  };
};

// Helper function: customer ID validator
window.customerIdValue = function(val) {
  return String(val || '').trim() || null;
};

// Helper function: site ID validator
window.siteIdValue = function(val) {
  return String(val || '').trim() || null;
};

// Utility: init client error reporting
window.initClientErrorReporting = function() {
  window.addEventListener('error', (e) => {
    console.error('Client error:', e.error || e.message);
  });
  window.addEventListener('unhandledrejection', (e) => {
    console.error('Unhandled rejection:', e.reason);
  });
};

// Feature flag utility
window.neraiumFeatureEnabled = function(feature) {
  const flags = {
    geometry3d: true,
    replay: true,
    validation: true,
    advanced_analytics: true,
  };
  return flags[String(feature).toLowerCase()] || false;
};

// Dispose geometry renderer
window.disposeGeometryRenderer = function() {
  // Cleanup 3D resources if they exist
  if (window.state.geometry3d?.renderer) {
    window.state.geometry3d.renderer = null;
  }
};

// Update upload run info display
window.updateUploadRunInfo = function() {
  const runInfo = window.qs('#uploadRunInfo');
  if (runInfo && window.state.activeRun) {
    runInfo.textContent = `Active run: ${window.escapeHtml(window.state.activeRun.name || window.state.activeRun.run_id)}`;
  }
};

// Reset upload panel
window.resetUploadPanelIfIdle = function() {
  // Reset upload panel state if not currently uploading
  if (!window.state.ui.uploading) {
    const fileInput = window.qs('#csvFileInput');
    if (fileInput) fileInput.value = '';
  }
};

// Refresh runtime mode banner
window.refreshRuntimeModeBanner = async function() {
  const banner = window.qs('#runtimeModeBanner');
  if (!banner) return;

  try {
    const { apiUrl, fetchJson } = window.NeraiumApi;
    const status = await fetchJson(apiUrl('/status', window.tenantScopeParams()));

    if (status && status.runtime_mode === 'degraded') {
      window.state.runtimeDegraded = true;
      banner.classList.remove('hidden');
      banner.textContent = 'Runtime is in degraded mode. Some features may be limited.';
    }
  } catch (err) {
    // Silent fail - runtime status check optional
  }
};

// Helper: phaseBadgeHtml (from dashboard module)
window.phaseBadgeHtml = window.phaseBadgeHtml || function(phase) {
  const colors = {
    baseline: '#4dabf7',
    learning: '#9c36b5',
    operation: '#15aabf',
    failure: '#fa5252'
  };
  const color = colors[String(phase).toLowerCase()] || '#666';
  const escaped = window.escapeHtml(String(phase));
  return `<span class="phase-badge" style="background-color: ${color}; color: white; padding: 4px 8px; border-radius: 3px;">${escaped}</span>`;
};

// Helper: riskBadgeHtml (from dashboard module)
window.riskBadgeHtml = window.riskBadgeHtml || function(riskLevel) {
  const colors = {
    critical: '#ff4444',
    high: '#ff8c00',
    medium: '#ffcc00',
    low: '#88dd00',
    none: '#00aa00',
    unknown: '#666666'
  };
  const normalized = String(riskLevel || 'unknown').toLowerCase();
  const color = colors[normalized] || colors.unknown;
  return `<span class="risk-badge" style="background-color: ${color}; color: white; padding: 4px 8px; border-radius: 3px; font-weight: bold;">${window.escapeHtml(String(riskLevel))}</span>`;
};
