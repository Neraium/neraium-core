'use client'

interface MetricsGridSectionProps {
  state: any
}

export function MetricsGridSection({ state }: MetricsGridSectionProps) {
  const metrics = [
    {
      label: 'Structural Coherence',
      value: Math.round(state.coherence * 100),
      unit: '%',
      description: 'Overall system cohesion and alignment',
      color: state.coherence > 0.7 ? '#60a5fa' : state.coherence > 0.4 ? '#eab308' : '#ef4444',
    },
    {
      label: 'Drift Intensity',
      value: Math.round(state.drift * 100),
      unit: '%',
      description: 'Rate of system deformation from nominal',
      color: state.drift < 0.3 ? '#60a5fa' : state.drift < 0.6 ? '#eab308' : '#ef4444',
    },
    {
      label: 'Stability Index',
      value: Math.round(state.stability * 100),
      unit: '%',
      description: 'Capacity to resist further deformation',
      color: state.stability > 0.7 ? '#60a5fa' : state.stability > 0.4 ? '#eab308' : '#ef4444',
    },
    {
      label: 'System Fragility',
      value:
        state.subsystems && state.subsystems.length > 0
          ? Math.round(
              (state.subsystems.reduce((sum: number, s: any) => sum + (s.fragility_pct || 0), 0) /
                state.subsystems.length) as number
            )
          : 0,
      unit: '%',
      description: 'Aggregate fragility across all subsystems',
      color:
        state.subsystems && state.subsystems.length > 0
          ? (value: number) =>
              value > 65 ? '#ef4444' : value > 40 ? '#f97316' : value > 20 ? '#eab308' : '#60a5fa'
          : '#60a5fa',
    },
  ]

  return (
    <div className="space-y-12">
      {/* Section header */}
      <div>
        <h2 className="text-2xl font-light tracking-wider text-white">System Metrics</h2>
        <p className="text-base font-light text-white/50 mt-3">
          Core structural and performance indicators
        </p>
      </div>

      {/* Metrics grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {metrics.map((metric, idx) => {
          const metricColor = typeof metric.color === 'function' ? metric.color(metric.value) : metric.color

          return (
            <div
              key={idx}
              className="p-8 border border-white/10 rounded-lg hover:border-white/20 transition-colors"
            >
              {/* Metric name */}
              <div className="text-xs font-light text-white/40 uppercase tracking-widest mb-6">
                {metric.label}
              </div>

              {/* Large value display */}
              <div className="mb-6">
                <div className="flex items-baseline gap-2">
                  <div className="text-5xl font-light" style={{ color: metricColor }}>
                    {metric.value}
                  </div>
                  <div className="text-2xl font-light text-white/40">{metric.unit}</div>
                </div>
              </div>

              {/* Progress bar indicator */}
              <div className="mb-6 h-1 bg-white/5 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{
                    width: `${metric.value}%`,
                    backgroundColor: metricColor,
                    opacity: 0.6,
                  }}
                />
              </div>

              {/* Description */}
              <p className="text-sm font-light text-white/50 leading-relaxed">{metric.description}</p>
            </div>
          )
        })}
      </div>

      {/* Guidance section */}
      <div className="mt-8 pt-8 border-t border-white/5 space-y-3">
        <div className="text-sm font-light text-white/60">
          <strong>Key relationships:</strong>
        </div>
        <div className="space-y-2 text-xs font-light text-white/50">
          <div>• <strong className="text-white/60">High Coherence + Low Drift</strong> = System in control</div>
          <div>• <strong className="text-white/60">Declining Coherence + Rising Drift</strong> = Accelerating instability</div>
          <div>• <strong className="text-white/60">High Fragility</strong> = System approaching irreversible state</div>
        </div>
      </div>
    </div>
  )
}
