import { DemoFrame, DemoInitResponse } from "./types";

type BackendFrame = Record<string, unknown>;

export function normalizeFrame(raw: BackendFrame): DemoFrame {
  return {
    frame_index: Number(raw.frame_index ?? raw.index ?? 0),
    frame_count: Number(raw.frame_count ?? raw.count ?? 0),
    timestamp: String(raw.timestamp ?? ""),
    current_phase: String(raw.phase ?? raw.state ?? raw.policy_state ?? "UNKNOWN"),
    verdict: String(raw.verdict ?? ""),
    status: String(raw.status ?? ""),
    metrics: {
      structural_drift: Number(raw.structural_drift_score ?? raw.structural_drift ?? 0),
      relational_stability: Number(raw.relational_stability_score ?? raw.relational_stability ?? 0),
      coherence: Number(raw.coherence ?? raw.system_health ?? 0),
      confidence: Number(raw.confidence_score ?? raw.alignment_confidence ?? raw.confidence ?? 0),
    },
    tetrahedral_state: {
      position: (raw.tetrahedral_state as Record<string, unknown>)?.position ?? [0, 0, 0],
      interpreted_label: String((raw.tetrahedral_state as Record<string, unknown>)?.interpreted_label ?? ""),
      movement_summary: String((raw.tetrahedral_state as Record<string, unknown>)?.movement_summary ?? ""),
      nearest_vertex: String((raw.tetrahedral_state as Record<string, unknown>)?.nearest_vertex ?? ""),
    },
    reasoning: {
      summary: String((raw.reasoning as Record<string, unknown>)?.summary ?? raw.reasoning_summary ?? ""),
      context: ((raw.reasoning as Record<string, unknown>)?.context ?? raw.reasoning_context ?? {}) as Record<string, unknown>,
    },
    evidence: Array.isArray(raw.evidence) ? (raw.evidence as Record<string, unknown>[]) : Array.isArray(raw.sensor_relationships) ? (raw.sensor_relationships as Record<string, unknown>[]) : [],
    gate_decision: ((raw.gate_decision ?? {}) as Record<string, unknown>),
  };
}

export function normalizeDemoInitResponse(raw: Record<string, unknown>): DemoInitResponse {
  const frames = Array.isArray(raw.frames)
    ? (raw.frames as BackendFrame[]).map(normalizeFrame)
    : [];

  const initialFrame = raw.initial_frame
    ? normalizeFrame(raw.initial_frame as BackendFrame)
    : frames[0] ?? normalizeFrame({});

  return {
    mode: String(raw.mode ?? "demo"),
    initial_frame: initialFrame,
    frames,
    summary: ((raw.summary ?? {}) as Record<string, unknown>),
  };
}
