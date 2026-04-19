'use client'

interface IntelligenceRailStickyProps {
  state: any
}

export function IntelligenceRailSticky({ state }: IntelligenceRailStickyProps) {
  const insights = state.insights || {}

  // Only render sections with content
  const sections = [
    {
      title: 'Current State',
      content: insights.current_state_insight || 'System operating nominally',
      color: '#60a5fa',
    },
    {
      title: 'Primary Driver',
      content: insights.primary_driver_insight,
      color: '#60a5fa',
    },
    {
      title: 'Operator Focus',
      content: insights.operator_focus_insight,
      color: '#ffffff',
    },
    {
      title: 'Recovery Window',
      content: insights.recoverability_context,
      color: insights.recoverability_context?.includes('closing')
        ? '#ef4444'
        : insights.recoverability_context?.includes('immediately')
          ? '#ef4444'
          : '#eab308',
    },
    {
      title: 'No-Action Consequence',
      content: insights.no_action_consequence_insight,
      color: '#ef4444',
    },
  ]

  const activeSections = sections.filter((s) => s.content)

  return (
    <div className="h-full flex flex-col justify-start pt-32 pr-8 pl-8 pb-16 space-y-12 overflow-y-auto">
      {/* Rail header */}
      <div>
        <div className="text-xs font-light text-white/40 tracking-widest uppercase">Intelligence</div>
      </div>

      {/* Insight sections */}
      {activeSections.map((section, idx) => (
        <div key={idx} className="space-y-3 pb-8 border-b border-white/10 last:border-b-0">
          <div className="text-xs font-light text-white/50 uppercase tracking-widest">{section.title}</div>
          <div className="text-sm font-light leading-relaxed" style={{ color: section.color }}>
            {section.content}
          </div>
        </div>
      ))}

      {/* Empty state */}
      {activeSections.length === 0 && (
        <div className="text-xs font-light text-white/30">Awaiting system data…</div>
      )}
    </div>
  )
}
