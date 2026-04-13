"use client";

import { DemoFrame } from "@/lib/types";

export function InsightPanels({ frame }: { frame: DemoFrame }) {
  return (
    <section className="insights">
      <details className="panel" open>
        <summary>Reasoning</summary>
        <pre>{frame.reasoning.summary || JSON.stringify(frame.reasoning.context, null, 2)}</pre>
      </details>
      <details className="panel">
        <summary>Evidence</summary>
        <pre>{JSON.stringify(frame.evidence, null, 2)}</pre>
      </details>
    </section>
  );
}
