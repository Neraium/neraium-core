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
      {/* Facility command strip */}
      <div className="absolute top-20 left-8 z-30 space-y-2">
        <div className="text-sm text-white/60">
          <span className="text-xs uppercase tracking-widest">Facility Status</span>
        </div>
        <div className="flex items-baseline gap-4">
          <div>
            <div className="text-xs text-white/40">Rooms</div>
            <div className="text-2xl font-light">{state.rooms?.length || 0}</div>
          </div>
          <div>
            <div className="text-xs text-white/40">Needing Attention</div>
            <div className="text-2xl font-light text-orange-400">
              {state.rooms?.filter((r: any) => r.status === 'attention').length || 0}
            </div>
          </div>
        </div>
      </div>

      {/* Main system field - dominate the screen */}
      <div className="flex-1 flex items-center justify-center relative">
        <TetrahedronVisualization state={state} />
      </div>

      {/* Minimal operator context - minimal text overlay */}
      <div className="absolute bottom-8 left-8 z-30 max-w-md">
        <div className={`text-sm font-light leading-relaxed ${stateColor}`}>
          {state.insights?.current_state_insight || 'System operating nominally'}
        </div>
        {state.critical_alerts && state.critical_alerts.length > 0 && (
          <div className="mt-4 text-xs text-red-400/80 space-y-1">
            {state.critical_alerts.slice(0, 2).map((alert: string, idx: number) => (
              <div key={idx}>• {alert}</div>
            ))}
          </div>
        )}
      </div>

      {/* System state indicator - top right */}
      <div className="absolute top-8 right-8 z-30">
        <div className="text-xs text-white/40 uppercase tracking-widest mb-2">System State</div>
        <div className={`text-2xl font-light ${stateColor}`}>
          {state.state.toUpperCase()}
        </div>
        <div className="text-xs text-white/60 mt-1">
          Drift: {(state.drift * 100).toFixed(1)}%
        </div>
      </div>

      {/* Timestamp - bottom right */}
      <div className="absolute bottom-8 right-8 z-30 text-right">
        <div className="text-xs text-white/40">{state.timestamp}</div>
        <div className="text-xs text-white/60 mt-1">
          Coherence: {(state.coherence * 100).toFixed(0)}%
        </div>
      </div>
    </div>
  )
}
