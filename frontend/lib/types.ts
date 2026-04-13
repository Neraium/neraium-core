export type DemoFrame = {
  frame_index: number;
  frame_count: number;
  timestamp: string;
  current_phase: string;
  verdict: string;
  status: string;
  metrics: {
    structural_drift: number;
    relational_stability: number;
    coherence: number;
    confidence: number;
  };
  tetrahedral_state: {
    position?: [number, number, number] | { x: number; y: number; z: number };
    interpreted_label?: string;
    movement_summary?: string;
    nearest_vertex?: string;
  };
  reasoning: {
    summary: string;
    context: Record<string, unknown>;
  };
  evidence: Record<string, unknown>[];
  gate_decision: Record<string, unknown>;
};

export type DemoInitResponse = {
  mode: string;
  initial_frame: DemoFrame;
  frames: DemoFrame[];
  summary: Record<string, unknown>;
};
