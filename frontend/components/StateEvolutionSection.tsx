'use client'

import { useEffect, useRef } from 'react'

interface TimelineState {
  drift: number
  timestamp: string
  index: number
}

interface NoActionProjection {
  drift: number
  step: number
}

interface Insights {
  recoverability_context?: string
  no_action_consequence_insight?: string
}

interface StateEvolutionSectionProps {
  timeline: TimelineState[]
  noActionProjection: NoActionProjection[]
  insights: Insights
}

export function StateEvolutionSection({
  timeline,
  noActionProjection,
  insights,
}: StateEvolutionSectionProps) {
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    if (!svgRef.current || !timeline || timeline.length === 0) return

    const width = 1200
    const height = 300
    const padding = 40

    // Get data range
    const allDrifts = [...timeline.map((t) => t.drift), ...(noActionProjection?.map((p) => p.drift) || [])]
    const maxDrift = Math.max(...allDrifts, 1)
    const minDrift = 0

    // Scale functions
    const scaleX = (index: number) => padding + ((index / Math.max(timeline.length - 1, 1)) * (width - padding * 2))
    const scaleY = (drift: number) => height - padding - ((drift - minDrift) / (maxDrift - minDrift)) * (height - padding * 2)

    // Build path for actual trajectory
    const actualPath = timeline
      .map((point, idx) => {
        const x = scaleX(idx)
        const y = scaleY(point.drift)
        return `${idx === 0 ? 'M' : 'L'} ${x} ${y}`
      })
      .join(' ')

    // Build path for no-action projection
    const projectionPath =
      timeline.length > 0 && noActionProjection && noActionProjection.length > 0
        ? (() => {
            const lastIdx = timeline.length - 1
            const startX = scaleX(lastIdx)
            const startY = scaleY(timeline[lastIdx].drift)

            const projPoints = noActionProjection.map((proj) => {
              const projIdx = lastIdx + proj.step
              const x = scaleX(projIdx)
              const y = scaleY(proj.drift)
              return `${projPoints ? 'L' : 'M'} ${x} ${y}`
            })

            return `M ${startX} ${startY} ${projPoints.join(' ')}`
          })()
        : ''

    // Calculate velocity (slope) at each point
    const velocityPath = timeline
      .slice(1)
      .map((point, idx) => {
        const prev = timeline[idx]
        const velocity = (point.drift - prev.drift) / (point.drift > prev.drift ? 1 : -1)
        const x = scaleX(idx + 1)
        const y = scaleY(point.drift)

        // Size proportional to velocity magnitude
        const radius = Math.min(8, 3 + Math.abs(velocity) * 3)

        // Color based on acceleration
        const color = velocity > 0.05 ? '#ef4444' : velocity > 0.02 ? '#f97316' : '#60a5fa'

        return `<circle cx="${x}" cy="${y}" r="${radius}" fill="${color}" opacity="0.4"/>`
      })
      .join('')

    const svg = svgRef.current
    svg.innerHTML = `
      <defs>
        <style>
          .trajectory-line {
            stroke: #60a5fa;
            stroke-width: 2;
            fill: none;
          }
          .projection-line {
            stroke: #ef4444;
            stroke-width: 2;
            fill: none;
            stroke-dasharray: 6, 4;
            opacity: 0.5;
          }
          .axis {
            stroke: white;
            stroke-width: 1;
            opacity: 0.1;
          }
          .grid-line {
            stroke: white;
            stroke-width: 0.5;
            opacity: 0.05;
          }
        </style>
      </defs>

      <!-- Grid -->
      <g class="grid">
        ${Array.from({ length: 5 })
          .map((_, i) => {
            const y = padding + ((height - padding * 2) / 4) * i
            return `<line x1="${padding}" y1="${y}" x2="${width - padding}" y2="${y}" class="grid-line"/>`
          })
          .join('')}
      </g>

      <!-- Axes -->
      <line x1="${padding}" y1="0" x2="${padding}" y2="${height}" class="axis"/>
      <line x1="0" y1="${height - padding}" x2="${width}" y2="${height - padding}" class="axis"/>

      <!-- Actual trajectory -->
      <path d="${actualPath}" class="trajectory-line"/>

      <!-- Velocity indicators -->
      ${velocityPath}

      <!-- No-action projection -->
      ${projectionPath ? `<path d="${projectionPath}" class="projection-line"/>` : ''}

      <!-- Projection points (circles) -->
      ${
        noActionProjection && noActionProjection.length > 0
          ? noActionProjection
              .map((proj) => {
                const projIdx = timeline.length - 1 + proj.step
                const x = scaleX(projIdx)
                const y = scaleY(proj.drift)
                return `<circle cx="${x}" cy="${y}" r="4" fill="#ef4444" opacity="0.3"/>`
              })
              .join('')
          : ''
      }

      <!-- Labels -->
      <text x="${padding}" y="20" font-size="12" fill="white" opacity="0.6">Drift Momentum</text>
      <text x="${width - padding}" y="${height - 5}" font-size="12" fill="white" opacity="0.6" text-anchor="end">Time →</text>
    `
  }, [timeline, noActionProjection])

  return (
    <div className="relative space-y-8">
      <div className="text-sm text-white/60 uppercase tracking-widest">
        State Evolution & Momentum
      </div>

      {/* SVG visualization */}
      <svg
        ref={svgRef}
        viewBox={`0 0 1200 300`}
        className="w-full border border-white/10 rounded"
        style={{
          background: 'rgba(0,0,0,0.3)',
        }}
      />

      {/* Legend and context */}
      <div className="grid grid-cols-3 gap-6 text-xs mt-6">
        <div>
          <div className="text-white/40 mb-1">Blue Line</div>
          <div className="text-white/60">Current trajectory (actual behavior)</div>
        </div>
        <div>
          <div className="text-white/40 mb-1">Red Dashed</div>
          <div className="text-white/60">Projected (no intervention)</div>
        </div>
        <div>
          <div className="text-white/40 mb-1">Circle Size</div>
          <div className="text-white/60">Velocity magnitude</div>
        </div>
      </div>

      {/* Consequence messaging */}
      {insights?.no_action_consequence_insight && (
        <div className="mt-8 pt-6 border-t border-white/10">
          <div className="text-xs text-white/40 mb-2">NO-ACTION CONSEQUENCE</div>
          <div className="text-sm text-red-400/90 leading-relaxed font-light">
            {insights.no_action_consequence_insight}
          </div>
        </div>
      )}

      {/* Recoverability context */}
      {insights?.recoverability_context && (
        <div className="mt-6 pt-6 border-t border-white/10">
          <div className="text-xs text-white/40 mb-2">RECOVERY WINDOW</div>
          <div
            className="text-sm leading-relaxed font-light"
            style={{
              color: insights.recoverability_context.includes('closing')
                ? '#ef4444'
                : insights.recoverability_context.includes('immediately')
                  ? '#ef4444'
                  : '#fbbf24',
            }}
          >
            {insights.recoverability_context}
          </div>
        </div>
      )}
    </div>
  )
}
