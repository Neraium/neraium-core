import { DemoInitResponse } from "./types";
import { normalizeDemoInitResponse } from "./normalize";

const API_BASE = process.env.NEXT_PUBLIC_NERAIUM_API_BASE ?? "";

export async function fetchDemoInit(): Promise<DemoInitResponse> {
  const response = await fetch(`${API_BASE}/api/demo/init?use_synthetic=true`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load demo init: ${response.status}`);
  }
  const raw = await response.json();
  return normalizeDemoInitResponse(raw);
}
