'use client'

interface Subsystem {
  name: string
  drift_contribution_pct: number
  confidence: number
  fragility_pct: number
  behavioral_state: string
  position: string
}

interface SubsystemAnalysisProps {
  subsystems: Subsystem[]
}

const SUBSYSTEM_POSITIONS: Record<string, string> = {
  Climate: 'top-center',
  Airflow: 'front-right',
  Irrigation: 'front-left',
  Plant: 'back-center',
}

export function SubsystemAnalysis({ subsystems }: SubsystemAnalysisProps) {
  if (!subsystems || subsystems.length === 0) {
    return (
      <div className="text-white/40 text-sm">
        No subsystem data available
      </div>
    )
  }

  return (
    <div className="relative space-y-8">
      <div className="text-sm text-white/60 uppercase tracking-widest">
        Subsystem Influence
      </div>

      <div className="space-y-6">
        {subsystems.map((subsystem, idx) => (
          <div
            key={idx}
            className="relative pl-6 py-4 border-l-2 border-white/20 hover:border-white/40 transition-colors"
          >
            {/* Dynamic border color based on fragility */}
            <div
              className="absolute -left-[1px] top-0 h-full w-[2px] transition-colors"
              style={{
                backgroundColor:
                  subsystem.fragility_pct > 0.65
                    ? '#ef4444'
                    : subsystem.fragility_pct > 0.4
                      ? '#f97316'
                      : subsystem.fragility_pct > 0.2
                        ? '#eab308'
                        : '#60a5fa',
              }}
            />

            {/* Subsystem name and position mapping */}
            <div className="flex items-baseline justify-between mb-2">
              <div>
                <div className="text-base font-light">
                  {subsystem.name}
                </div>
                <div className="text-xs text-white/40">
                  {SUBSYSTEM_POSITIONS[subsystem.name] || subsystem.position}
                </div>
              </div>
            </div>

            {/* Metrics in minimal format */}
            <div className="grid grid-cols-3 gap-6 mt-3">
              {/* Drift contribution */}
              <div>
                <div className="text-xs text-white/40 mb-1">Drift Contribution</div>
                <div className="text-sm font-light">{subsystem.drift_contribution_pct.toFixed(1)}%</div>
              </div>

              {/* Confidence */}
              <div>
                <div className="text-xs text-white/40 mb-1">Confidence</div>
                <div className="text-sm font-light">{(subsystem.confidence * 100).toFixed(0)}%</div>
              </div>

              {/* Fragility (not confidence) */}
              <div>
                <div className="text-xs text-white/40 mb-1">Fragility</div>
                <div
                  className="text-sm font-light"
                  style={{
                    color:
                      subsystem.fragility_pct > 0.65
                        ? '#ef4444'
                        : subsystem.fragility_pct > 0.4
                          ? '#f97316'
                          : subsystem.fragility_pct > 0.2
                            ? '#eab308'
                            : '#60a5fa',
                  }}
                >
                  {(subsystem.fragility_pct * 100).toFixed(0)}%
                </div>
              </div>
            </div>

            {/* State description */}
            {subsystem.behavioral_state && (
              <div className="text-xs text-white/60 mt-3 leading-relaxed">
                {subsystem.behavioral_state}
              </div>
            )}

            {/* Coupling indicators - if stressed */}
            {subsystem.drift_contribution_pct > 15 && (
              <div className="text-xs text-orange-400/70 mt-3">
                ⚠ Elevated contribution to system drift
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Bottom guidance */}
      <div className="mt-12 pt-8 border-t border-white/5 text-xs text-white/40 space-y-2">
        <div>Fragility indicates brittleness (low confidence × high contribution = high fragility)</div>
        <div>Watch for subsystems with elevated fragility—recovery window may be narrowing</div>
      </div>
    </div>
  )
}
