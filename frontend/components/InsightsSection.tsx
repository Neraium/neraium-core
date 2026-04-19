'use client'

interface InsightsSectionProps {
  state: any
}

export function InsightsSection({ state }: InsightsSectionProps) {
  const insights = state.insights || {}

  return (
    <div className="space-y-12">
      {/* Section header */}
      <div>
        <h2 className="text-2xl font-light tracking-wider text-white">System Intelligence</h2>
        <p className="text-base font-light text-white/50 mt-3">
          Contextual analysis and operational guidance
        </p>
      </div>

      {/* Insights grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Current State Insight */}
        {insights.current_state_insight && (
          <div className="p-8 border border-blue-400/20 rounded-lg">
            <div className="text-xs font-light text-blue-400 uppercase tracking-widest mb-4">Current State</div>
            <p className="text-base font-light text-white/80 leading-relaxed">
              {insights.current_state_insight}
            </p>
          </div>
        )}

        {/* Operator Focus Insight */}
        {insights.operator_focus_insight && (
          <div className="p-8 border border-white/20 rounded-lg">
            <div className="text-xs font-light text-white/60 uppercase tracking-widest mb-4">Operator Focus</div>
            <p className="text-base font-light text-white/80 leading-relaxed">
              {insights.operator_focus_insight}
            </p>
          </div>
        )}
      </div>

      {/* Full-width sections below */}
      <div className="space-y-8">
        {/* Recovery Window */}
        {insights.recoverability_context && (
          <div
            className="p-8 border rounded-lg"
            style={{
              borderColor:
                insights.recoverability_context.includes('closing') ||
                insights.recoverability_context.includes('immediately')
                  ? '#ef444433'
                  : '#eab30833',
            }}
          >
            <div
              className="text-xs font-light uppercase tracking-widest mb-4"
              style={{
                color:
                  insights.recoverability_context.includes('closing') ||
                  insights.recoverability_context.includes('immediately')
                    ? '#ef4444'
                    : '#eab308',
              }}
            >
              Recovery Window
            </div>
            <p
              className="text-base font-light leading-relaxed"
              style={{
                color:
                  insights.recoverability_context.includes('closing') ||
                  insights.recoverability_context.includes('immediately')
                    ? '#ef4444'
                    : '#eab308',
              }}
            >
              {insights.recoverability_context}
            </p>
          </div>
        )}

        {/* No-Action Consequence */}
        {insights.no_action_consequence_insight && (
          <div className="p-8 border border-red-400/20 rounded-lg">
            <div className="text-xs font-light text-red-400 uppercase tracking-widest mb-4">No-Action Consequence</div>
            <p className="text-base font-light text-red-400/90 leading-relaxed">
              {insights.no_action_consequence_insight}
            </p>
          </div>
        )}
      </div>

      {/* Analysis guidance */}
      <div className="mt-12 pt-8 border-t border-white/5 space-y-4">
        <div className="text-xs font-light text-white/40 uppercase tracking-widest">How to interpret</div>
        <div className="space-y-3 text-sm font-light text-white/50">
          <div>
            <strong className="text-white/60">Current State</strong> describes what the system is doing right now and why.
          </div>
          <div>
            <strong className="text-white/60">Operator Focus</strong> recommends where attention should be directed.
          </div>
          <div>
            <strong className="text-white/60">Recovery Window</strong> indicates how much time remains before the state becomes irreversible.
          </div>
          <div>
            <strong className="text-white/60">No-Action Consequence</strong> explains what happens if intervention is delayed.
          </div>
        </div>
      </div>
    </div>
  )
}
