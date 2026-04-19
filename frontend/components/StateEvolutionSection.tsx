'use client'

import { useEffect, useRef, useState, useMemo } from 'react'

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

interface HoverData {
  index: number
  drift: number
  timestamp: string
  x: number
  y: number
  isInLeadTime?: boolean
  timeToTransition?: number
}

interface LeadTimeEvent {
  driftOnsetIndex: number
  transitionIndex: number
  leadTimeCycles: number
}

export function StateEvolutionSection({
  timeline,
  noActionProjection,
  insights,
}: StateEvolutionSectionProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [hoveredPoint, setHoveredPoint] = useState<HoverData | null>(null)

  // Detect when drift begins accelerating/emerging (drift onset)
  const detectDriftOnset = (data: TimelineState[]): number => {
    if (data.length < 3) return -1

    const onsetDriftThreshold = 0.08 // Drift emerging
    const accelerationThreshold = 0.008 // Significant acceleration per cycle

    for (let i = 1; i < data.length; i++) {
      const prev = data[i - 1]
      const curr = data[i]
      const acceleration = curr.drift - prev.drift

      // Onset when drift crosses threshold with sustained acceleration
      if (curr.drift >= onsetDriftThreshold && acceleration > accelerationThreshold) {
        return i
      }
    }

    return -1
  }

  // Detect regime transitions (state changes to more severe states)
  const detectTransitions = (data: TimelineState[]) => {
    if (data.length < 3) return []

    const transitions = []
    const stateThresholds = [
      { label: 'Stable', max: 0.2 },
      { label: 'Drift', max: 0.4 },
      { label: 'Instability', max: 0.6 },
      { label: 'Critical', max: 1.0 },
    ]

    let lastState = stateThresholds.findIndex((s) => data[0].drift <= s.max)

    for (let i = 1; i < data.length; i++) {
      const currentState = stateThresholds.findIndex((s) => data[i].drift <= s.max)
      if (currentState !== lastState && currentState > lastState) {
        transitions.push({
          index: i,
          fromState: stateThresholds[lastState].label,
          toState: stateThresholds[currentState].label,
          drift: data[i].drift,
        })
        lastState = currentState
      }
    }

    return transitions
  }

  // Calculate lead time between drift onset and transition
  const calculateLeadTime = (data: TimelineState[]): LeadTimeEvent | null => {
    const onsetIdx = detectDriftOnset(data)
    const transitions = detectTransitions(data)

    // Need both onset and a transition after it
    if (onsetIdx < 0 || transitions.length === 0) {
      return null
    }

    const firstTransitionAfterOnset = transitions.find((t) => t.index > onsetIdx)
    if (!firstTransitionAfterOnset) {
      return null
    }

    return {
      driftOnsetIndex: onsetIdx,
      transitionIndex: firstTransitionAfterOnset.index,
      leadTimeCycles: firstTransitionAfterOnset.index - onsetIdx,
    }
  }

  // Memoize lead time calculations
  const leadTimeMetrics = useMemo(() => {
    const onsetIdx = detectDriftOnset(timeline)
    const transitions = detectTransitions(timeline)
    const leadTime = calculateLeadTime(timeline)

    return {
      driftOnsetIndex: onsetIdx,
      transitions,
      leadTimeEvent: leadTime,
      hasDriftOnset: onsetIdx >= 0,
      hasLeadTime: leadTime !== null,
    }
  }, [timeline])

  const getStateColor = (drift: number) => {
    if (drift < 0.2) return '#10b981' // green - stable
    if (drift < 0.4) return '#f59e0b' // amber - drift
    if (drift < 0.6) return '#f97316' // orange - instability
    return '#ef4444' // red - critical
  }

  const getStateLabel = (drift: number) => {
    if (drift < 0.2) return 'STABLE'
    if (drift < 0.4) return 'DRIFT'
    if (drift < 0.6) return 'INSTABILITY'
    return 'CRITICAL'
  }

  useEffect(() => {
    if (!svgRef.current || !timeline || timeline.length === 0) return

    const width = 1400
    const height = 500
    const padding = 60

    // Use memoized lead time calculations
    const { leadTimeEvent, driftOnsetIndex, transitions } = leadTimeMetrics

    // Data range
    const maxDrift = Math.max(...timeline.map((t) => t.drift), 1)
    const minDrift = 0

    // Scale functions
    const scaleX = (index: number) => padding + ((index / Math.max(timeline.length - 1, 1)) * (width - padding * 2))
    const scaleY = (drift: number) => height - padding - ((drift - minDrift) / (maxDrift - minDrift)) * (height - padding * 2)

    // Build smooth trajectory path
    const buildSmoothPath = (data: TimelineState[]) => {
      if (data.length === 0) return ''
      if (data.length === 1) {
        const x = scaleX(0)
        const y = scaleY(data[0].drift)
        return `M ${x} ${y}`
      }

      let path = ''
      for (let i = 0; i < data.length; i++) {
        const x = scaleX(i)
        const y = scaleY(data[i].drift)

        if (i === 0) {
          path += `M ${x} ${y}`
        } else {
          // Use quadratic bezier for smooth curves
          const prevX = scaleX(i - 1)
          const prevY = scaleY(data[i - 1].drift)
          const ctrlX = (prevX + x) / 2
          const ctrlY = (prevY + y) / 2
          path += ` Q ${ctrlX} ${ctrlY} ${x} ${y}`
        }
      }
      return path
    }

    const actualPath = buildSmoothPath(timeline)

    // Build projection path
    let projectionPath = ''
    if (timeline.length > 0 && noActionProjection && noActionProjection.length > 0) {
      const lastIdx = timeline.length - 1
      const lastPoint = timeline[lastIdx]
      const startX = scaleX(lastIdx)
      const startY = scaleY(lastPoint.drift)

      projectionPath = `M ${startX} ${startY}`
      for (let i = 0; i < noActionProjection.length; i++) {
        const projIdx = lastIdx + noActionProjection[i].step
        const x = scaleX(projIdx)
        const y = scaleY(noActionProjection[i].drift)
        projectionPath += ` L ${x} ${y}`
      }
    }

    // Build gradient defs
    const gradientStops = Array.from({ length: 10 })
      .map((_, i) => {
        const progress = i / 9
        const opacity = Math.pow(progress, 2) // quadratic ease for stronger emphasis on recent
        return `<stop offset="${progress * 100}%" stop-color="${getStateColor(
          timeline[Math.floor(progress * (timeline.length - 1))].drift
        )}" stop-opacity="${0.3 + opacity * 0.7}"/>`
      })
      .join('')

    // Build SVG
    const svg = svgRef.current

    // Render SVG
    const svgContent = `
      <defs>
        <style>
          .trajectory-line {
            stroke-width: 3;
            fill: none;
            filter: drop-shadow(0 0 8px rgba(255, 255, 255, 0.1));
          }
          .projection-line {
            stroke: #ef4444;
            stroke-width: 2;
            fill: none;
            stroke-dasharray: 8, 4;
            opacity: 0.4;
          }
          .grid-line {
            stroke: white;
            stroke-width: 0.5;
            opacity: 0.05;
          }
          .axis-line {
            stroke: white;
            stroke-width: 1;
            opacity: 0.15;
          }
          .transition-band {
            opacity: 0.08;
          }
          .transition-marker {
            stroke: white;
            stroke-width: 1;
            opacity: 0.3;
          }
          .state-label {
            font-size: 11px;
            font-weight: 300;
            opacity: 0.5;
            text-anchor: middle;
          }
        </style>

        <!-- Trajectory gradient -->
        <linearGradient id="trajectoryGradient" x1="0%" x2="100%" y1="0%" y2="0%">
          ${gradientStops}
        </linearGradient>
      </defs>

      <!-- Background regions by state -->
      <g class="background-regions">
        <!-- Stable region (0-0.2) -->
        <rect x="${padding}" y="${scaleY(0.2)}" width="${width - padding * 2}" height="${scaleY(0) - scaleY(0.2)}" fill="#10b981" opacity="0.03"/>

        <!-- Drift region (0.2-0.4) -->
        <rect x="${padding}" y="${scaleY(0.4)}" width="${width - padding * 2}" height="${scaleY(0.2) - scaleY(0.4)}" fill="#f59e0b" opacity="0.03"/>

        <!-- Instability region (0.4-0.6) -->
        <rect x="${padding}" y="${scaleY(0.6)}" width="${width - padding * 2}" height="${scaleY(0.4) - scaleY(0.6)}" fill="#f97316" opacity="0.03"/>

        <!-- Critical region (0.6+) -->
        <rect x="${padding}" y="${padding}" width="${width - padding * 2}" height="${scaleY(0.6) - padding}" fill="#ef4444" opacity="0.03"/>
      </g>

      <!-- Lead time band (if detected) -->
      ${
        leadTimeEvent
          ? (() => {
              const onsetX = scaleX(leadTimeEvent.driftOnsetIndex)
              const transitionX = scaleX(leadTimeEvent.transitionIndex)
              const bandWidth = transitionX - onsetX
              const midX = onsetX + bandWidth / 2
              return `
                <defs>
                  <linearGradient id="leadTimeGradient" x1="0%" x2="100%" y1="0%" y2="0%">
                    <stop offset="0%" stop-color="#f59e0b" stop-opacity="0.15"/>
                    <stop offset="100%" stop-color="#ef4444" stop-opacity="0.15"/>
                  </linearGradient>
                </defs>
                <!-- Lead time background band -->
                <rect x="${onsetX}" y="${padding}" width="${bandWidth}" height="${height - padding * 2}" fill="url(#leadTimeGradient)" rx="4"/>

                <!-- Lead time label -->
                <text x="${midX}" y="${padding + 25}" font-size="12" font-weight="300" fill="white" opacity="0.7" text-anchor="middle">
                  Lead Time: ${leadTimeEvent.leadTimeCycles} cycles
                </text>
              `
            })()
          : ''
      }

      <!-- Grid lines -->
      <g class="grid">
        ${Array.from({ length: 5 })
          .map((_, i) => {
            const y = padding + ((height - padding * 2) / 4) * i
            return `<line x1="${padding}" y1="${y}" x2="${width - padding}" y2="${y}" class="grid-line"/>`
          })
          .join('')}
      </g>

      <!-- Axes -->
      <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${height - padding}" class="axis-line"/>
      <line x1="${padding}" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}" class="axis-line"/>

      <!-- State threshold lines -->
      <g class="thresholds" opacity="0.1">
        <line x1="${padding}" y1="${scaleY(0.2)}" x2="${width - padding}" y2="${scaleY(0.2)}" stroke="white" stroke-width="0.5" stroke-dasharray="3,3"/>
        <line x1="${padding}" y1="${scaleY(0.4)}" x2="${width - padding}" y2="${scaleY(0.4)}" stroke="white" stroke-width="0.5" stroke-dasharray="3,3"/>
        <line x1="${padding}" y1="${scaleY(0.6)}" x2="${width - padding}" y2="${scaleY(0.6)}" stroke="white" stroke-width="0.5" stroke-dasharray="3,3"/>
      </g>

      <!-- Drift onset marker (amber) -->
      ${
        driftOnsetIndex >= 0
          ? (() => {
              const x = scaleX(driftOnsetIndex)
              return `
                <g class="drift-onset-marker">
                  <line x1="${x}" y1="${padding}" x2="${x}" y2="${height - padding}" stroke="#f59e0b" stroke-width="2" opacity="0.4"/>
                  <circle cx="${x}" cy="${scaleY(timeline[driftOnsetIndex].drift)}" r="4" fill="#f59e0b" opacity="0.6"/>
                  <text x="${x}" y="${padding - 25}" font-size="11" font-weight="300" fill="#f59e0b" opacity="0.7" text-anchor="middle">Drift Emerging</text>
                </g>
              `
            })()
          : ''
      }

      <!-- Transition markers (red) -->
      ${transitions
        .map((t, idx) => {
          const x = scaleX(t.index)
          // Only highlight the first major transition
          const isFirstTransition = idx === 0
          return `
            <g class="transition-marker-group">
              <line x1="${x}" y1="${padding}" x2="${x}" y2="${height - padding}" stroke="#ef4444" stroke-width="${isFirstTransition ? 2.5 : 1.5}" opacity="${isFirstTransition ? 0.6 : 0.3}"/>
              <circle cx="${x}" cy="${scaleY(t.drift)}" r="${isFirstTransition ? 5 : 3}" fill="#ef4444" opacity="${isFirstTransition ? 0.7 : 0.5}"/>
              ${
                isFirstTransition
                  ? `<text x="${x}" y="${height - padding + 25}" font-size="11" font-weight="300" fill="#ef4444" opacity="0.8" text-anchor="middle">System Transition</text>`
                  : ''
              }
            </g>
          `
        })
        .join('')}

      <!-- Drift buildup emphasis (thicker/colored segments during acceleration) -->
      ${timeline
        .slice(1)
        .map((point, idx) => {
          const prev = timeline[idx]
          const acceleration = point.drift - prev.drift
          if (acceleration > 0.02) {
            const x1 = scaleX(idx)
            const y1 = scaleY(prev.drift)
            const x2 = scaleX(idx + 1)
            const y2 = scaleY(point.drift)
            const color = getStateColor(point.drift)
            return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="1.5" opacity="0.3"/>`
          }
          return ''
        })
        .join('')}

      <!-- Main trajectory line -->
      <path d="${actualPath}" class="trajectory-line" stroke="url(#trajectoryGradient)"/>

      <!-- No-action projection -->
      ${projectionPath ? `<path d="${projectionPath}" class="projection-line"/>` : ''}

      <!-- Current state marker -->
      ${
        timeline.length > 0
          ? (() => {
              const lastPoint = timeline[timeline.length - 1]
              const x = scaleX(timeline.length - 1)
              const y = scaleY(lastPoint.drift)
              const color = getStateColor(lastPoint.drift)
              return `
                <g class="current-marker">
                  <circle cx="${x}" cy="${y}" r="8" fill="${color}" opacity="0.3"/>
                  <circle cx="${x}" cy="${y}" r="5" fill="${color}" opacity="0.8"/>
                  <circle cx="${x}" cy="${y}" r="5" fill="none" stroke="${color}" stroke-width="1.5" opacity="0.6"/>
                </g>
              `
            })()
          : ''
      }

      <!-- Forward momentum indicator (subtle arrow) -->
      ${
        timeline.length > 1
          ? (() => {
              const lastIdx = timeline.length - 1
              const lastPoint = timeline[lastIdx]
              const prevPoint = timeline[lastIdx - 1]
              const x1 = scaleX(lastIdx)
              const y1 = scaleY(lastPoint.drift)
              const x2 = scaleX(lastIdx + 0.5)
              const drift2 = lastPoint.drift + (lastPoint.drift - prevPoint.drift) * 0.5
              const y2 = scaleY(Math.min(drift2, maxDrift))
              return `
                <g class="momentum" opacity="0.4">
                  <line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${getStateColor(lastPoint.drift)}" stroke-width="2" stroke-dasharray="4,3"/>
                  <polygon points="${x2},${y2} ${x2 - 3},${y2 - 4} ${x2 - 1},${y2}" fill="${getStateColor(lastPoint.drift)}"/>
                </g>
              `
            })()
          : ''
      }

      <!-- Y-axis labels -->
      <text x="${padding - 10}" y="${padding - 5}" font-size="11" fill="white" opacity="0.4" text-anchor="end">100%</text>
      <text x="${padding - 10}" y="${height - padding + 5}" font-size="11" fill="white" opacity="0.4" text-anchor="end">0%</text>

      <!-- Title -->
      <text x="${padding}" y="${padding - 30}" font-size="14" font-weight="300" fill="white" opacity="0.8">System Evolution & Lead Time</text>
      <text x="${padding}" y="${padding - 12}" font-size="10" fill="white" opacity="0.4">Structural drift with detection lead window</text>
    `

    svg.innerHTML = svgContent

    // Add hover detection
    const handleMouseMove = (e: MouseEvent) => {
      if (!svgRef.current) return

      const rect = svgRef.current.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top

      // Convert pixel coords to data coords
      const dataX = ((x - padding) / (width - padding * 2)) * (timeline.length - 1)
      const closestIdx = Math.round(Math.max(0, Math.min(timeline.length - 1, dataX)))

      if (closestIdx >= 0 && closestIdx < timeline.length) {
        const point = timeline[closestIdx]
        const isInLeadTime = leadTimeEvent ? closestIdx >= leadTimeEvent.driftOnsetIndex && closestIdx <= leadTimeEvent.transitionIndex : false
        const timeToTransition = leadTimeEvent ? Math.max(0, leadTimeEvent.transitionIndex - closestIdx) : undefined

        setHoveredPoint({
          index: closestIdx,
          drift: point.drift,
          timestamp: point.timestamp,
          x: scaleX(closestIdx),
          y: scaleY(point.drift),
          isInLeadTime,
          timeToTransition,
        })
      }
    }

    const handleMouseLeave = () => {
      setHoveredPoint(null)
    }

    svg.addEventListener('mousemove', handleMouseMove)
    svg.addEventListener('mouseleave', handleMouseLeave)

    return () => {
      svg.removeEventListener('mousemove', handleMouseMove)
      svg.removeEventListener('mouseleave', handleMouseLeave)
    }
  }, [timeline, noActionProjection, leadTimeMetrics])

  if (!timeline || timeline.length === 0) {
    return (
      <div className="space-y-8">
        <div className="space-y-2">
          <h3 className="text-2xl font-light tracking-wider text-white">System Evolution</h3>
          <p className="text-base font-light text-white/50">How the system has evolved and where it's heading</p>
        </div>
        <div className="h-96 bg-gradient-to-b from-white/5 to-transparent rounded-lg border border-white/5 flex items-center justify-center">
          <div className="text-center">
            <div className="text-base font-light text-white/60 mb-2">Collecting baseline…</div>
            <div className="text-sm font-light text-white/40">
              Once data accumulates, this visualization will show:
            </div>
            <div className="text-xs font-light text-white/30 mt-4 space-y-1">
              <div>• When drift began emerging</div>
              <div>• Lead time before critical transitions</div>
              <div>• Time window for intervention</div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  const latestPoint = timeline[timeline.length - 1]
  const currentDrift = latestPoint.drift
  const currentState = getStateLabel(currentDrift)
  const currentColor = getStateColor(currentDrift)
  const { leadTimeEvent, hasDriftOnset, hasLeadTime } = leadTimeMetrics

  return (
    <div ref={containerRef} className="space-y-8">
      {/* Section header */}
      <div className="space-y-2">
        <h3 className="text-2xl font-light tracking-wider text-white">System Evolution</h3>
        <p className="text-base font-light text-white/50">How the system has evolved and where it's heading</p>
      </div>

      {/* Main chart */}
      <div className="relative bg-gradient-to-b from-white/5 to-transparent rounded-lg border border-white/5 overflow-hidden">
        <svg
          ref={svgRef}
          viewBox={`0 0 1400 500`}
          className="w-full"
          style={{ minHeight: '500px', background: 'radial-gradient(circle at 30% 0%, rgba(96, 165, 250, 0.03), transparent 50%)' }}
        />

        {/* Hover tooltip */}
        {hoveredPoint && (
          <div
            className="fixed bg-black/80 border border-white/20 rounded px-4 py-3 z-50 text-xs pointer-events-none"
            style={{
              left: `${hoveredPoint.x + 50}px`,
              top: `${hoveredPoint.y - 50}px`,
              backdropFilter: 'blur(4px)',
            }}
          >
            <div className="text-white/80 space-y-1">
              <div className="font-medium">Drift: {Math.round(hoveredPoint.drift * 100)}%</div>
              <div className="text-white/60">State: {getStateLabel(hoveredPoint.drift)}</div>
              {hoveredPoint.isInLeadTime && (
                <div className="text-amber-400 mt-2 pt-1 border-t border-white/10">
                  In lead window
                  {hoveredPoint.timeToTransition !== undefined && (
                    <div>{hoveredPoint.timeToTransition} cycles to transition</div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Lead time context */}
      {leadTimeEvent && (
        <div className="p-6 border border-amber-400/30 rounded-lg bg-amber-400/5">
          <div className="text-xs text-amber-400 uppercase tracking-widest mb-3">Lead Time Detection</div>
          <div className="grid grid-cols-3 gap-6">
            <div>
              <div className="text-sm font-light text-white/60 mb-1">Detection Point</div>
              <div className="text-lg font-light text-amber-400">{leadTimeEvent.driftOnsetIndex} cycles</div>
              <div className="text-xs text-white/40 mt-1">Drift onset</div>
            </div>
            <div>
              <div className="text-sm font-light text-white/60 mb-1">Lead Window</div>
              <div className="text-lg font-light text-white/80">{leadTimeEvent.leadTimeCycles} cycles</div>
              <div className="text-xs text-white/40 mt-1">Before transition</div>
            </div>
            <div>
              <div className="text-sm font-light text-white/60 mb-1">Transition Point</div>
              <div className="text-lg font-light text-red-400">{leadTimeEvent.transitionIndex} cycles</div>
              <div className="text-xs text-white/40 mt-1">System changed state</div>
            </div>
          </div>
          <div className="mt-4 pt-4 border-t border-amber-400/20">
            <p className="text-sm font-light text-white/70 leading-relaxed">
              The system demonstrated <strong className="text-amber-400">{leadTimeEvent.leadTimeCycles} cycles of observable drift</strong> before
              the structural transition occurred. This is the actionable lead time for intervention.
            </p>
          </div>
        </div>
      )}

      {/* Current state badge */}
      <div className="grid grid-cols-3 gap-6">
        <div className="p-6 border border-white/10 rounded-lg">
          <div className="text-xs font-light text-white/40 uppercase tracking-widest mb-3">Current State</div>
          <div
            className="text-3xl font-light tracking-wide"
            style={{ color: currentColor, textShadow: `0 0 12px ${currentColor}30` }}
          >
            {currentState}
          </div>
          <div className="text-sm font-light text-white/60 mt-2">Drift: {Math.round(currentDrift * 100)}%</div>
        </div>

        <div className="p-6 border border-white/10 rounded-lg">
          <div className="text-xs font-light text-white/40 uppercase tracking-widest mb-3">Direction</div>
          <div className="text-lg font-light text-white/80">
            {timeline.length > 1 ? (
              <>
                {timeline[timeline.length - 1].drift > timeline[timeline.length - 2].drift ? (
                  <>
                    <span className="text-red-400">↑ Accelerating</span>
                    <div className="text-sm font-light text-white/60 mt-1">toward instability</div>
                  </>
                ) : (
                  <>
                    <span className="text-green-400">↓ Stabilizing</span>
                    <div className="text-sm font-light text-white/60 mt-1">drift decreasing</div>
                  </>
                )}
              </>
            ) : (
              <span className="text-white/60">—</span>
            )}
          </div>
        </div>

        <div
          className="p-6 border rounded-lg"
          style={{
            borderColor: leadTimeEvent ? (timeline.length - 1 <= leadTimeEvent.transitionIndex ? '#f59e0b40' : '#ef444440') : 'rgba(255, 255, 255, 0.1)',
            backgroundColor: leadTimeEvent ? (timeline.length - 1 <= leadTimeEvent.transitionIndex ? '#f59e0b05' : '#ef444405') : undefined,
          }}
        >
          <div
            className="text-xs font-light uppercase tracking-widest mb-3"
            style={{ color: leadTimeEvent ? (timeline.length - 1 <= leadTimeEvent.transitionIndex ? '#f59e0b' : '#ef4444') : 'rgba(255, 255, 255, 0.4)' }}
          >
            {leadTimeEvent ? (timeline.length - 1 <= leadTimeEvent.transitionIndex ? 'In Lead Window' : 'Post-Transition') : 'Timeline Status'}
          </div>
          <div className="text-lg font-light text-white/80">
            {leadTimeEvent ? (
              <>
                {timeline.length - 1 <= leadTimeEvent.transitionIndex ? (
                  <>
                    <span className="text-amber-400">{leadTimeEvent.transitionIndex - (timeline.length - 1)} cycles</span>
                    <div className="text-sm font-light text-white/60 mt-1">remaining in window</div>
                  </>
                ) : (
                  <>
                    <span className="text-red-400">{timeline.length - 1 - leadTimeEvent.transitionIndex} cycles</span>
                    <div className="text-sm font-light text-white/60 mt-1">past transition</div>
                  </>
                )}
              </>
            ) : (
              <>
                <span className="text-white/60">{timeline.length} cycles</span>
                <div className="text-sm font-light text-white/60 mt-1">no transition yet</div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Legend and interpretation */}
      <div className="grid grid-cols-2 gap-6 pt-4 border-t border-white/5">
        <div className="space-y-3">
          <div className="text-xs font-light text-white/40 uppercase tracking-widest">Visual Elements</div>
          <div className="space-y-2 text-xs font-light text-white/60">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-gradient-to-r from-green-400 to-red-400" />
              <span>Trajectory = system behavior over time</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-0.5 bg-amber-400" />
              <span>Amber marker = drift emerging</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-0.5 bg-red-400" />
              <span>Red marker = system transition</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-0.5 bg-gradient-to-r from-amber-400/30 to-red-400/30" />
              <span>Shaded band = lead time window</span>
            </div>
          </div>
        </div>

        <div className="space-y-3">
          <div className="text-xs font-light text-white/40 uppercase tracking-widest">What This Shows</div>
          <div className="space-y-2 text-xs font-light text-white/60">
            <div>
              <strong className="text-white/80">Drift Onset</strong>: When instability became measurable
            </div>
            <div>
              <strong className="text-white/80">Lead Time Band</strong>: Observable window before critical transition
            </div>
            <div>
              <strong className="text-white/80">Transition Point</strong>: When system entered new regime
            </div>
            <div>
              <strong className="text-white/80">Actionable Lead</strong>: Time available for intervention
            </div>
          </div>
        </div>
      </div>

      {/* Consequence messaging */}
      {insights?.no_action_consequence_insight && (
        <div className="mt-8 p-6 border border-red-400/20 rounded-lg bg-red-400/5">
          <div className="text-xs text-red-400 uppercase tracking-widest mb-3">No-Action Consequence</div>
          <div className="text-base font-light text-red-400/90 leading-relaxed">{insights.no_action_consequence_insight}</div>
        </div>
      )}

      {/* Recoverability context */}
      {insights?.recoverability_context && (
        <div
          className="mt-6 p-6 border rounded-lg"
          style={{
            borderColor: insights.recoverability_context.includes('closing')
              ? '#ef444440'
              : insights.recoverability_context.includes('immediately')
                ? '#ef444440'
                : '#eab30840',
            backgroundColor: insights.recoverability_context.includes('closing')
              ? '#ef444405'
              : insights.recoverability_context.includes('immediately')
                ? '#ef444405'
                : '#eab30805',
          }}
        >
          <div
            className="text-xs uppercase tracking-widest mb-3"
            style={{
              color: insights.recoverability_context.includes('closing')
                ? '#ef4444'
                : insights.recoverability_context.includes('immediately')
                  ? '#ef4444'
                  : '#eab308',
            }}
          >
            Recovery Window
          </div>
          <div
            className="text-base leading-relaxed font-light"
            style={{
              color: insights.recoverability_context.includes('closing')
                ? '#ef4444'
                : insights.recoverability_context.includes('immediately')
                  ? '#ef4444'
                  : '#eab308',
            }}
          >
            {insights.recoverability_context}
          </div>
        </div>
      )}
    </div>
  )
}
