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
  Climate: 'top',
  Airflow: 'front-right',
  Irrigation: 'front-left',
  'Plant Response': 'back',
}

export function SubsystemAnalysis({ subsystems }: SubsystemAnalysisProps) {
  if (!subsystems || subsystems.length === 0) {
    return (
      <div className="text-white/40 text-sm font-light">No subsystem data available</div>
    )
  }

  return (
    <div className="relative space-y-12">
      <div>
        <div className="text-xs font-light text-white/40 uppercase tracking-widest mb-3">Subsystem Analysis</div>
        <div className="text-sm font-light text-white/50">
          Subsystem contributions to overall system drift, with confidence and fragility metrics
        </div>
      </div>

      <div className="space-y-8">
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
                  subsystem.fragility_pct > 65
                    ? '#ef4444'
                    : subsystem.fragility_pct > 40
                      ? '#f97316'
                      : subsystem.fragility_pct > 20
                        ? '#eab308'
                        : '#60a5fa',
              }}
            />

            {/* Subsystem name and position mapping */}
            <div className="flex items-baseline justify-between mb-3">
              <div>
                <div className="text-base font-light text-white/90">{subsystem.name}</div>
                <div className="text-xs font-light text-white/30 mt-1">
                  {SUBSYSTEM_POSITIONS[subsystem.name] || subsystem.position}
                </div>
              </div>
            </div>

            {/* Metrics in minimal format - rounded and stable */}
            <div className="grid grid-cols-3 gap-8 mt-4">
              {/* Drift contribution - rounded to whole % */}
              <div>
                <div className="text-xs font-light text-white/40 uppercase tracking-widest mb-2">Drift</div>
                <div className="text-lg font-light text-white/70">
                  <span>{Math.round(subsystem.drift_contribution_pct)}</span>
                  <span className="text-white/40 text-sm">%</span>
                </div>
              </div>

              {/* Confidence - rounded to nearest 5% */}
              <div>
                <div className="text-xs font-light text-white/40 uppercase tracking-widest mb-2">Confidence</div>
                <div className="text-lg font-light text-white/70">
                  <span>{Math.round(subsystem.confidence * 20) * 5}</span>
                  <span className="text-white/40 text-sm">%</span>
                </div>
              </div>

              {/* Fragility (not confidence) - whole % */}
              <div>
                <div className="text-xs font-light text-white/40 uppercase tracking-widest mb-2">Fragility</div>
                <div
                  className="text-lg font-light"
                  style={{
                    color:
                      subsystem.fragility_pct > 65
                        ? '#ef4444'
                        : subsystem.fragility_pct > 40
                          ? '#f97316'
                          : subsystem.fragility_pct > 20
                            ? '#eab308'
                            : '#60a5fa',
                  }}
                >
                  <span>{Math.round(subsystem.fragility_pct)}</span>
                  <span className="text-white/40 text-sm">%</span>
                </div>
              </div>
            </div>

            {/* State description */}
            {subsystem.behavioral_state && (
              <div className="text-xs font-light text-white/50 mt-4">{subsystem.behavioral_state}</div>
            )}

            {/* Coupling indicators - if stressed */}
            {subsystem.drift_contribution_pct > 15 && (
              <div className="text-xs font-light text-orange-400/70 mt-4">
                ⚠ Elevated contribution to system drift
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Bottom guidance */}
      <div className="mt-16 pt-8 border-t border-white/5 space-y-2">
        <div className="text-xs font-light text-white/40">
          Fragility indicates brittleness—low confidence combined with high drift contribution.
        </div>
        <div className="text-xs font-light text-white/40">
          Watch for subsystems with elevated fragility; recovery window may be narrowing.
        </div>
      </div>
    </div>
  )
}
