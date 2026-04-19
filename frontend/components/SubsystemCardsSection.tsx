'use client'

interface Subsystem {
  subsystem_id: string
  subsystem_name: string
  drift_contribution_pct: number
  confidence: number
  fragility_pct: number
  behavioral_state: string
}

interface SubsystemCardsSectionProps {
  subsystems: Subsystem[]
}

export function SubsystemCardsSection({ subsystems }: SubsystemCardsSectionProps) {
  if (!subsystems || subsystems.length === 0) {
    return (
      <div className="space-y-6">
        <div>
          <h2 className="text-2xl font-light tracking-wider text-white">Subsystem Status</h2>
          <p className="text-base font-light text-white/50 mt-2">
            Individual subsystem analysis and drift contribution
          </p>
        </div>
        <div className="text-white/40 text-base font-light">No subsystem data available</div>
      </div>
    )
  }

  return (
    <div className="space-y-12">
      {/* Section header */}
      <div>
        <h2 className="text-2xl font-light tracking-wider text-white">Subsystem Status</h2>
        <p className="text-base font-light text-white/50 mt-3">
          Individual subsystem analysis and drift contribution to overall system coherence
        </p>
      </div>

      {/* Subsystem cards grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {subsystems.map((subsystem) => {
          const fragilityColor =
            subsystem.fragility_pct > 65
              ? '#ef4444'
              : subsystem.fragility_pct > 40
                ? '#f97316'
                : subsystem.fragility_pct > 20
                  ? '#eab308'
                  : '#60a5fa'

          const borderColor =
            subsystem.drift_contribution_pct > 20
              ? 'border-orange-400/30'
              : subsystem.drift_contribution_pct > 10
                ? 'border-yellow-400/20'
                : 'border-blue-400/20'

          return (
            <div
              key={subsystem.subsystem_id}
              className={`relative p-8 border rounded-lg transition-all ${borderColor} hover:border-white/20`}
              style={{ borderColor: `${fragilityColor}33` }}
            >
              {/* Left accent bar */}
              <div
                className="absolute left-0 top-0 bottom-0 w-1 rounded-l-lg"
                style={{ backgroundColor: fragilityColor }}
              />

              {/* Subsystem name and state */}
              <div className="mb-6">
                <h3 className="text-lg font-light text-white/90">{subsystem.subsystem_name}</h3>
                <p className="text-xs font-light text-white/40 uppercase tracking-widest mt-2">
                  {subsystem.behavioral_state}
                </p>
              </div>

              {/* Metrics grid */}
              <div className="grid grid-cols-3 gap-6">
                {/* Drift */}
                <div>
                  <div className="text-xs font-light text-white/40 uppercase tracking-widest mb-2">Drift</div>
                  <div className="text-2xl font-light text-white/80">
                    {Math.round(subsystem.drift_contribution_pct)}<span className="text-white/40 text-sm">%</span>
                  </div>
                </div>

                {/* Confidence */}
                <div>
                  <div className="text-xs font-light text-white/40 uppercase tracking-widest mb-2">Confidence</div>
                  <div className="text-2xl font-light text-white/80">
                    {Math.round(subsystem.confidence * 20) * 5}<span className="text-white/40 text-sm">%</span>
                  </div>
                </div>

                {/* Fragility */}
                <div>
                  <div className="text-xs font-light text-white/40 uppercase tracking-widest mb-2">Fragility</div>
                  <div className="text-2xl font-light" style={{ color: fragilityColor }}>
                    {Math.round(subsystem.fragility_pct)}<span className="text-white/40 text-sm">%</span>
                  </div>
                </div>
              </div>

              {/* Warning indicator */}
              {subsystem.drift_contribution_pct > 15 && (
                <div className="mt-6 pt-4 border-t border-white/5">
                  <div className="text-xs font-light text-orange-400/80">
                    ⚠ Elevated drift contribution—watch for escalation
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Guidance footer */}
      <div className="mt-8 pt-8 border-t border-white/5 space-y-3">
        <div className="text-xs font-light text-white/40">
          <strong className="text-white/60">Fragility</strong> indicates brittleness: low confidence combined with high drift contribution.
        </div>
        <div className="text-xs font-light text-white/40">
          Monitor subsystems with elevated fragility—recovery window may be narrowing.
        </div>
      </div>
    </div>
  )
}
