"use client";

import { DemoFrame } from "@/lib/types";

export function InsightPanels({ frame }: { frame: DemoFrame }) {
  return (
    <section className="panel insights-section">
      <h3>Analysis Details</h3>
      <details open>
        <summary>Reasoning</summary>
        <pre>{frame.reasoning.summary || JSON.stringify(frame.reasoning.context, null, 2)}</pre>
      </details>
      <details>
        <summary>Evidence</summary>
        <pre>{JSON.stringify(frame.evidence, null, 2)}</pre>
      </details>
    </section>
  );
}
