const PAGE_LOADERS = {
  dashboard: () => loadDashboard(),
  runs: () => Promise.resolve(renderRunsList()),
  upload: () => Promise.resolve(updateUploadRunInfo()),
  validation: () => loadValidationPage(),
  onboarding: () => loadOnboardingPage(),
  "run-detail": (route) => loadRunDetail(route.runId),
  "result-detail": (route) => loadResultDetail(route.resultId),
};

function mountCoreModules() {
  wireMobileNav();
  wireWorkspaceShellEvents();
  wireRunsEvents();
  wireUploadFormEvents();
  wireRunDetailEvents();
  wireOnboardingEvents();

  wireUploadInteractions();
}


async function wireEvents() {
  mountCoreModules();
}

async function loadCurrentPage(route) {
  const loader = PAGE_LOADERS[route.page];
  if (loader) await loader(route);
}

async function init() {
  startLiveClock();
  readTenantFromStorage();
  const routeScope = routeScopeFromQuery();
  if (routeScope.customer_id) {
    state.tenant.customerId = customerIdValue(routeScope.customer_id);
  }
  if (routeScope.site_id) {
    state.tenant.siteId = siteIdValue(routeScope.site_id);
  }
  renderTenantControls();
  const route = getRoute();
  const routeToPage = {
    dashboard: "dashboard",
    runs: "runs",
    upload: "upload",
    "run-detail": "run-detail",
    "result-detail": "result-detail",
    validation: "validation",
    onboarding: "onboarding",
  };
  setPage(routeToPage[route.page] || "dashboard");
  try {
    setLoading(true, "Initializing workspace...");
    await refreshRuntimeModeBanner();
    await loadRuns();
    await loadCurrentPage(route);
    await wireEvents();
    resetUploadPanelIfIdle();
    try {
      const hp = new URLSearchParams(window.location.search).get("highlight");
      if (getRoute().page === "upload" && hp === "upload") {
        window.requestAnimationFrame(() => {
          const zone = qs("#uploadDropZone");
          if (zone) {
            zone.classList.add("upload-dropzone-highlight");
            zone.scrollIntoView({ behavior: "smooth", block: "center" });
            window.setTimeout(() => zone.classList.remove("upload-dropzone-highlight"), 5000);
          }
          try {
            const u = new URL(window.location.href);
            u.searchParams.delete("highlight");
            window.history.replaceState({}, "", u.pathname + u.search + u.hash);
          } catch (_e2) {
            // no-op
          }
        });
      }
    } catch (_e) {
      // no-op
    }
    setStatus("");
  } catch (err) {
    setStatus(String(err.message || err), true, true);
  } finally {
    setLoading(false);
  }
}

initClientErrorReporting();
if (typeof window !== "undefined") {
  window.NERAIUM_FEATURE_ENABLED = neraiumFeatureEnabled;
}
init();

async function refreshCurrentPage() {
  const route = getRoute();
  if (route.page !== "run-detail") {
    disposeGeometryRenderer();
  }
  await loadRuns();
  await loadCurrentPage(route);
  if (route.page !== "run-detail") clearRunDetailObserver();
}

const MOBILE_NAV_MQ = window.matchMedia("(max-width: 980px)");

function setMobileNavOpen(open) {
  const sidebar = qs("#appSidebar");
  const backdrop = qs("#navBackdrop");
  const toggle = qs("#mobileNavToggle");
  if (!sidebar || !backdrop || !toggle) return;
  if (open) {
    sidebar.classList.add("nav-drawer-open");
    backdrop.classList.remove("hidden");
    toggle.setAttribute("aria-expanded", "true");
    toggle.setAttribute("aria-label", "Close menu");
    document.body.classList.add("nav-mobile-open");
  } else {
    sidebar.classList.remove("nav-drawer-open");
    backdrop.classList.add("hidden");
    toggle.setAttribute("aria-expanded", "false");
    toggle.setAttribute("aria-label", "Open menu");
    document.body.classList.remove("nav-mobile-open");
  }
}

function wireMobileNav() {
  const sidebar = qs("#appSidebar");
  const backdrop = qs("#navBackdrop");
  const toggle = qs("#mobileNavToggle");
  if (!sidebar || !backdrop || !toggle || toggle.dataset.wired === "1") return;
  toggle.dataset.wired = "1";

  toggle.addEventListener("click", () => {
    const open = !sidebar.classList.contains("nav-drawer-open");
    setMobileNavOpen(open);
  });

  backdrop.addEventListener("click", () => {
    setMobileNavOpen(false);
  });

  qsa(".nav a").forEach((a) => {
    a.addEventListener("click", () => {
      if (MOBILE_NAV_MQ.matches) setMobileNavOpen(false);
    });
  });

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && sidebar.classList.contains("nav-drawer-open")) {
      setMobileNavOpen(false);
      toggle.focus();
    }
  });

  const onMq = () => {
    if (!MOBILE_NAV_MQ.matches) setMobileNavOpen(false);
  };
  if (typeof MOBILE_NAV_MQ.addEventListener === "function") {
    MOBILE_NAV_MQ.addEventListener("change", onMq);
  } else {
    MOBILE_NAV_MQ.addListener(onMq);
  }
}
