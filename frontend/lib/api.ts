import { DemoInitResponse } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_NERAIUM_API_BASE ?? "http://localhost:8000";

export async function fetchDemoInit(): Promise<DemoInitResponse> {
  const response = await fetch(`${API_BASE}/api/demo/init?use_synthetic=true`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load demo init: ${response.status}`);
  }
  return response.json();
}
