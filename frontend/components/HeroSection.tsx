'use client'

import { TetrahedronVisualization } from './TetrahedronVisualization'

interface HeroSectionProps {
  state: any
}

export function HeroSection({ state }: HeroSectionProps) {
  return (
    <div className="relative w-full min-h-screen bg-black overflow-hidden">
      {/* Tetrahedron visualization dominates */}
      <div className="absolute inset-0 flex items-center justify-center">
        <TetrahedronVisualization state={state} />
      </div>

      {/* Top-left: Facility overview */}
      <div className="absolute top-12 left-12 z-30">
        <div className="text-xs font-light text-white/40 tracking-widest uppercase mb-4">Facility</div>
        <div className="space-y-3">
          <div className="text-lg font-light">
            <span className="text-white/80">{state.rooms?.length || 0}</span>
            <span className="text-white/40 ml-2">rooms</span>
          </div>
          {state.rooms?.filter((r: any) => r.status === 'attention').length > 0 && (
            <div className="text-sm font-light text-orange-400">
              <span>⚠</span>
              <span className="ml-2">
                {state.rooms?.filter((r: any) => r.status === 'attention').length} flagged
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Top-right: System state (large, prominent) */}
      <div className="absolute top-12 right-12 z-30 text-right">
        <div className="text-xs font-light text-white/40 tracking-widest uppercase mb-4">System State</div>
        <div className="space-y-6">
          <div>
            <div
              className="text-4xl font-light tracking-wide"
              style={{
                color:
                  state.state === 'critical'
                    ? '#ef4444'
                    : state.state === 'instability'
                      ? '#f97316'
                      : state.state === 'drift'
                        ? '#eab308'
                        : '#60a5fa',
                textShadow:
                  state.state === 'critical'
                    ? '0 0 20px #ef444420'
                    : state.state === 'instability'
                      ? '0 0 20px #f9731620'
                      : state.state === 'drift'
                        ? '0 0 20px #eab30820'
                        : '0 0 20px #60a5fa20',
              }}
            >
              {state.state === 'critical'
                ? 'CRITICAL'
                : state.state === 'instability'
                  ? 'INSTABILITY'
                  : state.state === 'drift'
                    ? 'DRIFT'
                    : 'STABLE'}
            </div>
          </div>

          {/* Key metrics */}
          <div className="space-y-4 text-right">
            <div>
              <div className="text-xs font-light text-white/50 uppercase tracking-widest">Drift</div>
              <div className="text-2xl font-light text-white/80 mt-1">{Math.round(state.drift * 100)}%</div>
            </div>
            <div>
              <div className="text-xs font-light text-white/50 uppercase tracking-widest">Coherence</div>
              <div className="text-2xl font-light text-white/80 mt-1">{Math.round(state.coherence * 100)}%</div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom-left: Operator focus insight */}
      {state.insights?.operator_focus_insight && (
        <div className="absolute bottom-12 left-12 z-30 max-w-md">
          <div className="text-xs font-light text-white/40 uppercase tracking-widest mb-3">Operator Focus</div>
          <div className="text-base font-light leading-relaxed text-white/70">
            {state.insights.operator_focus_insight}
          </div>
        </div>
      )}

      {/* Bottom-right: Current state insight */}
      {state.insights?.current_state_insight && (
        <div className="absolute bottom-12 right-12 z-30 max-w-md text-right">
          <div className="text-xs font-light text-white/40 uppercase tracking-widest mb-3">Current State</div>
          <div className="text-base font-light leading-relaxed text-white/70">
            {state.insights.current_state_insight}
          </div>
        </div>
      )}

      {/* Center-bottom: Critical alerts or recoverability warning */}
      {state.insights?.recoverability_context &&
        (state.insights.recoverability_context.includes('closing') ||
          state.insights.recoverability_context.includes('immediately')) && (
          <div className="absolute bottom-1/2 left-1/2 transform -translate-x-1/2 translate-y-1/2 z-30 text-center max-w-2xl">
            <div className="text-xs font-light text-red-400 uppercase tracking-widest mb-2">Recovery Window</div>
            <div className="text-lg font-light text-red-400/90 leading-relaxed">
              {state.insights.recoverability_context}
            </div>
          </div>
        )}

      {state.critical_alerts && state.critical_alerts.length > 0 && !state.insights?.recoverability_context && (
        <div className="absolute bottom-1/3 left-1/2 transform -translate-x-1/2 z-30 text-center">
          <div className="space-y-2">
            {state.critical_alerts.slice(0, 2).map((alert: string, idx: number) => (
              <div key={idx} className="text-sm font-light text-red-400">
                ⚠ {alert}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
