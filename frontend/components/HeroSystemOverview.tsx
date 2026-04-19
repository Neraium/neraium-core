'use client'

import { useEffect, useState } from 'react'
import { TetrahedronVisualization } from './TetrahedronVisualization'

interface HeroSystemOverviewProps {
  state: any
  railRef: React.RefObject<HTMLDivElement>
}

export function HeroSystemOverview({ state, railRef }: HeroSystemOverviewProps) {
  const [stateColor, setStateColor] = useState('#60a5fa')
  const [stateLabel, setStateLabel] = useState('STABLE')

  useEffect(() => {
    if (state.state === 'critical') {
      setStateColor('#ef4444')
      setStateLabel('CRITICAL')
    } else if (state.state === 'instability') {
      setStateColor('#f97316')
      setStateLabel('INSTABILITY')
    } else if (state.state === 'drift') {
      setStateColor('#eab308')
      setStateLabel('DRIFT')
    } else {
      setStateColor('#60a5fa')
      setStateLabel('STABLE')
    }
  }, [state.state])

  return (
    <div className="relative w-full h-screen bg-black overflow-hidden flex flex-col">
      {/* Tetrahedron dominates the entire viewport */}
      <div className="flex-1 relative">
        <TetrahedronVisualization state={state} />
      </div>

      {/* Top-left: Facility status */}
      <div className="absolute top-12 left-12 z-30 space-y-6">
        {/* Facility label */}
        <div>
          <div className="text-xs font-light text-white/40 tracking-widest uppercase mb-3">Facility</div>
          <div className="space-y-2">
            <div className="text-sm font-light">
              <span className="text-white/70">{state.rooms?.length || 0}</span>
              <span className="text-white/40 ml-2">rooms</span>
            </div>
            {state.rooms?.filter((r: any) => r.status === 'attention').length > 0 && (
              <div className="text-sm font-light text-orange-400/80">
                <span className="text-orange-400">⚠</span>
                <span className="ml-2">
                  {state.rooms?.filter((r: any) => r.status === 'attention').length} flagged
                </span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Top-right: System state */}
      <div className="absolute top-12 right-12 z-30 text-right">
        <div className="text-xs font-light text-white/40 tracking-widest uppercase mb-3">System State</div>
        <div
          className="text-3xl font-light tracking-wide"
          style={{ color: stateColor, textShadow: `0 0 20px ${stateColor}20` }}
        >
          {stateLabel}
        </div>
        <div className="text-xs font-light text-white/50 mt-2">
          Drift: <span className="font-medium">{Math.round(state.drift * 100)}%</span>
        </div>
      </div>

      {/* Bottom-left: Operator focus */}
      {state.insights?.operator_focus_insight && (
        <div className="absolute bottom-12 left-12 z-30 max-w-md">
          <div className="text-xs font-light text-white/40 tracking-widest uppercase mb-3">Operator Focus</div>
          <div className="text-sm font-light leading-relaxed text-white/70">
            {state.insights.operator_focus_insight}
          </div>
        </div>
      )}

      {/* Bottom-right: Coherence */}
      <div className="absolute bottom-12 right-12 z-30 text-right">
        <div className="text-xs font-light text-white/40 tracking-widest uppercase mb-3">Coherence</div>
        <div className="text-2xl font-light">
          <span className="text-white/70">{Math.round(state.coherence * 100)}</span>
          <span className="text-white/40 text-lg">%</span>
        </div>
      </div>

      {/* Center-bottom: Critical alerts */}
      {state.critical_alerts && state.critical_alerts.length > 0 && (
        <div className="absolute bottom-12 left-1/2 transform -translate-x-1/2 z-30 text-center">
          <div className="space-y-1">
            {state.critical_alerts.slice(0, 2).map((alert: string, idx: number) => (
              <div key={idx} className="text-xs font-light text-red-400/80">
                ⚠ {alert}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recoverability warning (if critical) */}
      {state.insights?.recoverability_context &&
        (state.insights.recoverability_context.includes('closing') ||
          state.insights.recoverability_context.includes('immediately')) && (
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-30 text-center max-w-xl">
            <div className="text-xs font-light text-red-400/80 tracking-wide uppercase mb-2">Recovery Window</div>
            <div className="text-sm font-light text-red-400/90 leading-relaxed">
              {state.insights.recoverability_context}
            </div>
          </div>
        )}
    </div>
  )
}
