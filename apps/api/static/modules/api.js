(function attachApiModule(globalObj) {
  function asOperatorText(value) {
    if (typeof value === "string") return value;
    if (value && typeof value === "object") {
      const message = value.message ? String(value.message) : "";
      const actionable = value.actionable_detail ? String(value.actionable_detail) : "";
      const correlation = value.correlation_id ? `ref ${String(value.correlation_id)}` : "";
      return [message, actionable, correlation].filter((x) => x && x.trim().length > 0).join(" | ");
    }
    return "";
  }

  function apiUrl(path, params) {
    const normalizedPath = String(path || "").replace(/^\/api(?=\/|$)/, "") || "/";
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([k, v]) => {
        if (v !== undefined && v !== null && String(v).length > 0) {
          searchParams.set(k, String(v));
        }
      });
    }
    const query = searchParams.toString();
    return query ? `${normalizedPath}?${query}` : normalizedPath;
  }

  async function fetchJson(path, opts) {
    const method = String((opts && opts.method) || "GET").toUpperCase();
    let res;
    try {
      res = await fetch(path, opts);
    } catch (err) {
      const endpoint = String(path || "");
      const reason = String((err && err.message) || err || "network error");
      console.error("API network/preflight failure", {
        endpoint,
        method,
        reason,
        requestHeaders: opts && opts.headers ? opts.headers : null,
        requestBody: opts && typeof opts.body === "string" ? opts.body.slice(0, 1000) : null,
      });
      const e = new Error(
        [
          `[NETWORK before response] ${method} ${endpoint}`,
          `reason=${reason}`,
          "Connection dropped before response (same-origin cloud ingest/network failure).",
        ].join(" | "),
      );
      e.name = "ApiNetworkError";
      e.apiError = {
        stage: "before_response",
        method,
        endpoint,
        reason,
        status: null,
        responseText: null,
      };
      throw e;
    }
    const endpoint = String(path || res.url || "");
    const raw = await res.text();
    let body = null;
    if (raw) {
      try {
        body = JSON.parse(raw);
      } catch (_err) {
        body = null;
      }
    }
    if (!res.ok) {
      const detailValue = body && typeof body === "object" ? (body.detail ?? body.message) : null;
      const detail =
        typeof detailValue === "string"
          ? detailValue
          : detailValue && typeof detailValue === "object"
            ? asOperatorText(detailValue) || JSON.stringify(detailValue)
            : raw
              ? raw.slice(0, 500)
              : "empty response body";
      const issueSummary =
        body && Array.isArray(body.issue_details) && body.issue_details.length
          ? body.issue_details
              .slice(0, 3)
              .map((i) => (i && i.message ? String(i.message) : "Invalid field"))
              .join("; ")
          : "";
      const actionable =
        body && typeof body === "object" && body.actionable_detail
          ? String(body.actionable_detail)
          : "";
      const correlation =
        body && typeof body === "object" && body.correlation_id
          ? ` (ref ${String(body.correlation_id)})`
          : "";
      const userMessage = [body && body.message ? String(body.message) : "", actionable, issueSummary]
        .filter((x) => x && x.trim().length > 0)
        .join(" ");
      const fallback = asOperatorText(detailValue);
      const displayMessage = userMessage || fallback;
      console.error("API request failed", {
        endpoint,
        method,
        status: res.status,
        statusText: res.statusText || "",
        responseText: raw ? raw.slice(0, 1000) : "",
      });
      const e = new Error(
        [
          `[HTTP after response] ${method} ${endpoint}`,
          `status=${res.status} ${res.statusText || ""}`.trim(),
          `detail=${detail}`,
          displayMessage ? `message=${displayMessage}${correlation}` : "",
        ].join(" | "),
      );
      e.name = "ApiHttpError";
      e.apiError = {
        stage: "after_response",
        method,
        endpoint,
        status: res.status,
        statusText: res.statusText || "",
        detail,
        body,
        responseText: raw || "",
      };
      throw e;
    }
    if (!raw) return {};
    if (body !== null) return body;
    return {};
  }

  globalObj.NeraiumApi = {
    apiUrl,
    fetchJson,
  };
})(window);
