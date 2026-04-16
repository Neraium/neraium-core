import { DemoFrame, DemoInitResponse } from "./types";
import { normalizeDemoInitResponse } from "./normalize";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || (typeof window !== 'undefined' ? `${window.location.protocol}//${window.location.host}` : "http://127.0.0.1:8000");
const FD004_DEMO_INIT_URL = `${API_BASE}/api/demo/fd004/init?unit_id=unit_001`;

export async function fetchDemoInit(): Promise<DemoInitResponse> {
  const url = FD004_DEMO_INIT_URL;
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load demo init: ${response.status}`);
  }
  const raw = await response.json();
  return normalizeDemoInitResponse(raw);
}

export async function fetchFD004DemoInit(unitId: string = "unit_001"): Promise<DemoFrame[]> {
  const url = `${API_BASE}/api/demo/fd004/init?unit_id=${encodeURIComponent(unitId)}`;
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load FD004 demo init: ${response.status}`);
  }
  const raw = await response.json();
  const normalized = normalizeDemoInitResponse(raw);
  const frames = normalized.frames ?? [];
  return frames;
}
