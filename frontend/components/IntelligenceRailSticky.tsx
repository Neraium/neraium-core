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
      color: 'text-blue-400',
    },
    {
      title: 'Primary Driver',
      content: insights.primary_driver_insight,
      color: 'text-blue-400',
    },
    {
      title: 'Operator Focus',
      content: insights.operator_focus_insight,
      color: 'text-white/70',
    },
    {
      title: 'Recovery Window',
      content: insights.recoverability_context,
      color: insights.recoverability_context?.includes('closing') ? 'text-red-400' : 'text-yellow-400',
    },
    {
      title: 'No-Action Consequence',
      content: insights.no_action_consequence_insight,
      color: 'text-red-400',
    },
  ]

  const activeSection = sections.filter((s) => s.content)

  return (
    <div className="h-full flex flex-col justify-start pt-24 pr-6 pl-6 pb-8 space-y-8 overflow-y-auto">
      {/* Rail title */}
      <div>
        <div className="text-xs text-white/40 uppercase tracking-widest mb-4">
          Intelligence
        </div>
      </div>

      {/* Insight sections */}
      {activeSection.map((section, idx) => (
        <div key={idx} className="space-y-2 pb-6 border-b border-white/5 last:border-b-0">
          <div className="text-xs text-white/40 uppercase tracking-widest">{section.title}</div>
          <div className={`text-xs leading-relaxed ${section.color}`}>
            {section.content}
          </div>
        </div>
      ))}

      {/* Empty state */}
      {activeSection.length === 0 && (
        <div className="text-xs text-white/30">
          Waiting for system intelligence...
        </div>
      )}
    </div>
  )
}
