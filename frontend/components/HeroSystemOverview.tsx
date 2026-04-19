'use client'

import { useEffect, useState } from 'react'
import { TetrahedronVisualization } from './TetrahedronVisualization'

interface HeroSystemOverviewProps {
  state: any
  railRef: React.RefObject<HTMLDivElement>
}

export function HeroSystemOverview({ state, railRef }: HeroSystemOverviewProps) {
  const [stateColor, setStateColor] = useState('text-blue-400')

  useEffect(() => {
    if (state.state === 'critical') setStateColor('text-red-500')
    else if (state.state === 'instability') setStateColor('text-orange-400')
    else if (state.state === 'drift') setStateColor('text-yellow-400')
    else setStateColor('text-blue-400')
  }, [state.state])

  return (
    <div className="relative w-full h-screen bg-black overflow-hidden flex flex-col">
      {/* Tetrahedron dominates the entire viewport */}
      <div className="flex-1 relative">
        <TetrahedronVisualization state={state} />
      </div>

      {/* Minimal top-left facility indicator (small, unobtrusive) */}
      <div className="absolute top-6 left-6 z-30 text-xs text-white/40 space-y-1">
        <div>FACILITY: {state.rooms?.length || 0} rooms</div>
        {state.rooms?.filter((r: any) => r.status === 'attention').length > 0 && (
          <div className="text-orange-400/60">
            ⚠ {state.rooms?.filter((r: any) => r.status === 'attention').length} rooms flagged
          </div>
        )}
      </div>

      {/* Operator focus (bottom-left, minimal and lean) */}
      {state.insights?.operator_focus_insight && (
        <div className="absolute bottom-8 left-6 z-30 max-w-xs">
          <div className="text-xs text-white/50 uppercase tracking-widest mb-2">Focus</div>
          <div className="text-sm font-light text-white/70 leading-relaxed">
            {state.insights.operator_focus_insight}
          </div>
        </div>
      )}

      {/* System state label (top-right, minimal) */}
      <div className="absolute top-6 right-6 z-30 text-right">
        <div className={`text-lg font-light ${stateColor}`}>
          {state.state?.toUpperCase() || 'UNKNOWN'}
        </div>
        <div className="text-xs text-white/40 mt-1">
          Time: {state.time || state.timestamp?.split('T')[1]?.slice(0, 8) || '—'}
        </div>
      </div>

      {/* Critical alerts if any (bottom-right, minimal) */}
      {state.critical_alerts && state.critical_alerts.length > 0 && (
        <div className="absolute bottom-8 right-6 z-30 max-w-xs">
          <div className="text-xs text-red-400/70 space-y-1">
            {state.critical_alerts.slice(0, 2).map((alert: string, idx: number) => (
              <div key={idx} className="text-xs">
                ⚠ {alert}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recoverability (if critical) - centered at bottom */}
      {state.insights?.recoverability_context && (
        state.insights.recoverability_context.includes('closing') ||
        state.insights.recoverability_context.includes('immediately')
      ) ? (
        <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 z-30 text-center">
          <div className="text-xs text-red-400/80 font-light">
            {state.insights.recoverability_context}
          </div>
        </div>
      ) : null}
    </div>
  )
}
