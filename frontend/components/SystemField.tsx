'use client'

import React, { useEffect, useRef, useState } from 'react'
import { DecisionUIState, Severity, DegradationStage } from '@/lib/decisionToUI'
import { PALETTE, severityColor, severityGlow, stageIndex, deriveSubsystems, breatheOffset, jitterOffset } from '@/lib/dashboardHelpers'

interface Props {
  state: DecisionUIState
}

/* ─── Layout Constants ────────────────────────────────────────────────────── */

const VB = { w: 400, h: 400 }
const CORE = { x: 200, y: 200, r: 28 }

interface SatelliteDef {
  id: string
  label: string
  angle: number /* degrees */
  dist: number
  r: number
}

const SATELLITES: SatelliteDef[] = [
  { id: 'climate',   label: 'CLIMATE',   angle: -90, dist: 130, r: 22 },
  { id: 'irrigation',label: 'H₂O',       angle: -18, dist: 140, r: 22 },
  { id: 'lighting',  label: 'LIGHT',     angle: 54,  dist: 130, r: 22 },
  { id: 'airflow',   label: 'AIR',       angle: 126, dist: 130, r: 22 },
  { id: 'rootzone',  label: 'ROOTS',     angle: 198, dist: 140, r: 22 },
]

function rad(deg: number) { return (deg * Math.PI) / 180 }

/* ─── Component ───────────────────────────────────────────────────────────── */

export function SystemField({ state }: Props) {
  const { statusHeader, tetrahedron } = state
  const sev = statusHeader.severity
  const stage = statusHeader.degradationStage
  const idx = stageIndex(stage)
  const scalar = tetrahedron.severityScalar

  const subsystems = deriveSubsystems(state)
  const subMap = Object.fromEntries(subsystems.map(s => [s.id, s]))

  /* animation time — driven by rAF, throttled to ~20fps state updates */
  const [tick, setTick] = useState(0)
  const frameRef = useRef(0)
  const lastRef = useRef(0)

  useEffect(() => {
    let raf: number
    const loop = (t: number) => {
      if (t - lastRef.current > 50) {
        lastRef.current = t
        setTick(prev => prev + 1)
      }
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [])

  const time = tick * 0.05

  /* ── compute satellite positions ── */
  const satellites = SATELLITES.map((sat, i) => {
    const health = subMap[sat.id]?.health ?? 'stable'
    const contribution = subMap[sat.id]?.contribution ?? 'low'

    /* base position */
    let sx = CORE.x + Math.cos(rad(sat.angle)) * sat.dist
    let sy = CORE.y + Math.sin(rad(sat.angle)) * sat.dist

    /* breathing */
    const breathAmp = idx === 0 ? 2 : idx <= 2 ? 4 : 6
    sx += breatheOffset(i, time, breathAmp)
    sy += breatheOffset(i + 50, time, breathAmp)

    /* drift deformation: pull affected nodes outward */
    if (contribution === 'high') {
      const pull = 8 + scalar * 20
      sx += Math.cos(rad(sat.angle)) * pull
      sy += Math.sin(rad(sat.angle)) * pull
    } else if (contribution === 'medium') {
      const pull = 4 + scalar * 10
      sx += Math.cos(rad(sat.angle)) * pull
      sy += Math.sin(rad(sat.angle)) * pull
    }

    /* critical jitter */
    if (idx >= 4) {
      const jitterIntensity = idx === 5 ? 1 : 0.6
      sx += jitterOffset(i + 200, time, jitterIntensity)
      sy += jitterOffset(i + 300, time, jitterIntensity)
    }

    /* color */
    const col =
      health === 'critical' ? PALETTE.red
      : health === 'warning' ? PALETTE.orange
      : PALETTE.neon

    const glow =
      health === 'critical' ? PALETTE.redGlow
      : health === 'warning' ? PALETTE.orangeGlow
      : PALETTE.neonGlow

    return { ...sat, x: sx, y: sy, health, col, glow }
  })

  /* ── core position ── */
  let cx = CORE.x
  let cy = CORE.y
  if (idx >= 4) {
    cx += jitterOffset(999, time, idx === 5 ? 0.8 : 0.4)
    cy += jitterOffset(1000, time, idx === 5 ? 0.8 : 0.4)
  }

  const coreColor = sev === Severity.HIGH ? PALETTE.red : sev === Severity.ELEVATED ? PALETTE.orange : sev === Severity.MODERATE ? PALETTE.blue : PALETTE.neon
  const coreGlow = severityGlow(sev)

  /* ── edge paths ── */
  const edges = satellites.map((sat, i) => {
    const midX = (cx + sat.x) / 2
    const midY = (cy + sat.y) / 2

    /* control point offset for bezier curvature */
    let cpOffset = 0
    if (idx === 1) cpOffset = 10 + scalar * 15
    else if (idx === 2) cpOffset = 20 + scalar * 25
    else if (idx === 3) cpOffset = 35 + scalar * 35
    else if (idx >= 4) cpOffset = 50 + scalar * 40

    /* direction: perpendicular to edge, biased by node angle */
    const perpX = -Math.sin(rad(sat.angle))
    const perpY = Math.cos(rad(sat.angle))

    const cpx = midX + perpX * cpOffset + breatheOffset(i + 400, time, 3)
    const cpy = midY + perpY * cpOffset + breatheOffset(i + 500, time, 3)

    const d = `M${cx.toFixed(1)},${cy.toFixed(1)} Q${cpx.toFixed(1)},${cpy.toFixed(1)} ${sat.x.toFixed(1)},${sat.y.toFixed(1)}`

    /* edge break in critical */
    const isBroken = idx >= 4 && sat.health === 'critical'

    return { d, cpx, cpy, col: sat.col, glow: sat.glow, isBroken, sat }
  })

  const pulseDuration = idx >= 4 ? '0.8s' : idx >= 2 ? '1.4s' : idx >= 1 ? '2.2s' : '3.5s'

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative', overflow: 'hidden' }}>
      {/* subtle radial gradient behind core */}
      <div style={{
        position: 'absolute', inset: 0,
        background: `radial-gradient(circle at ${(cx / VB.w) * 100}% ${(cy / VB.h) * 100}%, ${coreGlow} 0%, transparent 55%)`,
        opacity: 0.6,
        transition: 'background 1s ease',
      }} />

      {/* grain overlay */}
      <div style={{
        position: 'absolute', inset: 0, opacity: 0.03,
        backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")`,
        backgroundRepeat: 'repeat',
        backgroundSize: '128px 128px',
        pointerEvents: 'none',
      }} />

      <svg
        viewBox={`0 0 ${VB.w} ${VB.h}`}
        width="100%" height="100%"
        preserveAspectRatio="xMidYMid meet"
        style={{ display: 'block' }}
      >
        <defs>
          {/* glow filters */}
          <filter id="sf-glow-neon" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="5" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <filter id="sf-glow-blue" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="5" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <filter id="sf-glow-orange" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="5" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <filter id="sf-glow-red" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="6" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <filter id="sf-core-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="8" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>

        <style>{`
          @keyframes sfNodePulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.08); opacity: 0.9; }
          }
          @keyframes sfEdgeFlow {
            0% { stroke-dashoffset: 24; }
            100% { stroke-dashoffset: 0; }
          }
          .sf-node { transform-origin: center; animation: sfNodePulse ${pulseDuration} ease-in-out infinite; }
          .sf-edge { stroke-dasharray: 4 3; animation: sfEdgeFlow 2s linear infinite; }
        `}</style>

        {/* edges */}
        {edges.map((e, i) => {
          const filterId =
            e.sat.health === 'critical' ? 'sf-glow-red'
            : e.sat.health === 'warning' ? 'sf-glow-orange'
            : e.sat.col === PALETTE.blue ? 'sf-glow-blue'
            : 'sf-glow-neon'

          if (e.isBroken) {
            /* broken edge: two segments with gap */
            const gap = 18
            const dx = e.sat.x - cx
            const dy = e.sat.y - cy
            const len = Math.sqrt(dx * dx + dy * dy)
            const nx = dx / len
            const ny = dy / len
            const breakPoint = 0.5
            const bx = cx + nx * len * breakPoint
            const by = cy + ny * len * breakPoint
            const s1x = cx + nx * (len * breakPoint - gap)
            const s1y = cy + ny * (len * breakPoint - gap)
            const s2x = cx + nx * (len * breakPoint + gap)
            const s2y = cy + ny * (len * breakPoint + gap)
            return (
              <g key={i}>
                <line x1={cx} y1={cy} x2={s1x} y2={s1y}
                  stroke={e.col} strokeWidth={2} opacity={0.5} filter={`url(#${filterId})`} />
                <line x1={s2x} y1={s2y} x2={e.sat.x} y2={e.sat.y}
                  stroke={e.col} strokeWidth={2} opacity={0.5} filter={`url(#${filterId})`} />
                <circle cx={bx} cy={by} r={4} fill={PALETTE.red} opacity={0.9} filter="url(#sf-glow-red)" />
              </g>
            )
          }

          return (
            <g key={i}>
              {/* glow track */}
              <path d={e.d} fill="none" stroke={e.glow} strokeWidth={6} opacity={0.25} filter={`url(#${filterId})`} />
              {/* animated dash */}
              <path d={e.d} fill="none" stroke={e.col} strokeWidth={1.5} opacity={0.7} className="sf-edge" />
            </g>
          )
        })}

        {/* core node */}
        <g transform={`translate(${cx.toFixed(1)},${cy.toFixed(1)})`} className="sf-node">
          <circle r={CORE.r + 12} fill="none" stroke={coreColor} strokeWidth={0.8} opacity={0.25} filter="url(#sf-core-glow)" />
          <circle r={CORE.r} fill={`${coreColor}18`} stroke={coreColor} strokeWidth={2} filter="url(#sf-core-glow)" />
          <circle r={CORE.r - 10} fill="none" stroke={coreColor} strokeWidth={0.6} opacity={0.5} />
          <text textAnchor="middle" dominantBaseline="central" fill={coreColor} fontSize={11} fontWeight={800} letterSpacing="0.08em">
            CORE
          </text>
        </g>

        {/* satellite nodes */}
        {satellites.map((sat, i) => {
          const filterId =
            sat.health === 'critical' ? 'sf-glow-red'
            : sat.health === 'warning' ? 'sf-glow-orange'
            : sat.col === PALETTE.blue ? 'sf-glow-blue'
            : 'sf-glow-neon'

          return (
            <g key={sat.id} transform={`translate(${sat.x.toFixed(1)},${sat.y.toFixed(1)})`} className="sf-node">
              {/* outer ring */}
              <circle r={sat.r + 10} fill="none" stroke={sat.col} strokeWidth={0.8} opacity={0.3} filter={`url(#${filterId})`} />
              {/* body */}
              <circle r={sat.r} fill={`${sat.col}15`} stroke={sat.col} strokeWidth={1.8} filter={`url(#${filterId})`} />
              {/* inner detail */}
              <circle r={sat.r - 7} fill="none" stroke={sat.col} strokeWidth={0.5} opacity={0.4} />
              {/* label */}
              <text textAnchor="middle" dominantBaseline="central" fill={sat.col} fontSize={sat.r > 20 ? 9 : 8} fontWeight={800} letterSpacing="0.06em">
                {sat.label}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
