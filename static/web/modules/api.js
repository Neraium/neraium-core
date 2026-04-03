export function buildApiPath(path, params = {}) {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const query = new URLSearchParams(
    Object.entries(params).filter(([, value]) => value !== undefined && value !== null && value !== "")
  ).toString();
  return query ? `${normalizedPath}?${query}` : normalizedPath;
}
