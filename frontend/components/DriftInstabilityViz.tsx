'use client'

import React from 'react'
import { DecisionUIState, Severity } from '@/lib/decisionToUI'
import { PALETTE } from '@/lib/dashboardHelpers'

interface Props {
  state: DecisionUIState
}

/* ─── smooth bezier through points ────────────────────────────────────────── */

function catmullRom2bezier(points: Array<{ x: number; y: number }>): string {
  if (points.length < 2) return ''
  if (points.length === 2) return `M${points[0].x.toFixed(1)},${points[0].y.toFixed(1)} L${points[1].x.toFixed(1)},${points[1].y.toFixed(1)}`

  let d = `M${points[0].x.toFixed(1)},${points[0].y.toFixed(1)}`
  for (let i = 0; i < points.length - 1; i++) {
    const p0 = points[Math.max(0, i - 1)]
    const p1 = points[i]
    const p2 = points[i + 1]
    const p3 = points[Math.min(points.length - 1, i + 2)]

    const cp1x = p1.x + (p2.x - p0.x) / 6
    const cp1y = p1.y + (p2.y - p0.y) / 6
    const cp2x = p2.x - (p3.x - p1.x) / 6
    const cp2y = p2.y - (p3.y - p1.y) / 6

    d += ` C${cp1x.toFixed(1)},${cp1y.toFixed(1)} ${cp2x.toFixed(1)},${cp2y.toFixed(1)} ${p2.x.toFixed(1)},${p2.y.toFixed(1)}`
  }
  return d
}

/* ─── Component ───────────────────────────────────────────────────────────── */

export function DriftInstabilityViz({ state }: Props) {
  const { driftChart, statusHeader, tetrahedron } = state
  const { dataPoints, detectionThreshold } = driftChart
  const sev = statusHeader.severity

  const W = 360
  const H = 160
  const PAD = { t: 16, r: 16, b: 24, l: 36 }
  const GW = W - PAD.l - PAD.r
  const GH = H - PAD.t - PAD.b

  /* ── drift points ── */
  const n = dataPoints.length
  const maxVal = Math.max(...dataPoints.map(d => d.driftScore), detectionThreshold, 0.5)
  const driftPts = dataPoints.map((dp, i) => ({
    x: PAD.l + (i / Math.max(n - 1, 1)) * GW,
    y: PAD.t + GH * (1 - dp.driftScore / maxVal),
  }))

  const driftPath = catmullRom2bezier(driftPts)

  /* ── instability points (derived from drift deltas + severity scalar) ── */
  const instabilityPts = dataPoints.map((dp, i) => {
    const prev = dataPoints[Math.max(0, i - 1)]
    const delta = Math.abs(dp.driftScore - prev.driftScore)
    const instability = delta * 3 + tetrahedron.severityScalar * 0.15
    const x = PAD.l + (i / Math.max(n - 1, 1)) * GW
    const y = PAD.t + GH * (1 - Math.min(instability / maxVal, 1))
    return { x, y }
  })

  /* sharp angular path for instability */
  const instabilityPath = instabilityPts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')

  /* current position */
  const currentIdx = n - 1
  const currentX = driftPts[currentIdx]?.x ?? PAD.l
  const currentY = driftPts[currentIdx]?.y ?? PAD.t + GH / 2

  /* threshold line */
  const threshY = PAD.t + GH * (1 - detectionThreshold / maxVal)

  const currentVal = dataPoints[currentIdx]?.driftScore ?? 0
  const isAbove = currentVal > detectionThreshold

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative', padding: '12px 16px' }}>
      {/* header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
        <span style={{ fontSize: 9, letterSpacing: '0.14em', textTransform: 'uppercase', color: PALETTE.textMuted, fontWeight: 800 }}>
          Structural Evolution
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: 9, color: PALETTE.blue, fontWeight: 700, letterSpacing: '0.06em' }}>
            <span style={{ display: 'inline-block', width: 8, height: 2, background: PALETTE.blue, borderRadius: 1, marginRight: 4, verticalAlign: 'middle' }} />
            Drift
          </span>
          <span style={{ fontSize: 9, color: PALETTE.orange, fontWeight: 700, letterSpacing: '0.06em' }}>
            <span style={{ display: 'inline-block', width: 8, height: 2, background: PALETTE.orange, borderRadius: 1, marginRight: 4, verticalAlign: 'middle' }} />
            Instability
          </span>
        </div>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} style={{ display: 'block' }}>
        <defs>
          <linearGradient id="dv-drift-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={PALETTE.blue} stopOpacity="0.35" />
            <stop offset="100%" stopColor={PALETTE.blue} stopOpacity="0.02" />
          </linearGradient>
          <linearGradient id="dv-inst-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={PALETTE.orange} stopOpacity="0.25" />
            <stop offset="100%" stopColor={PALETTE.orange} stopOpacity="0.02" />
          </linearGradient>
          <filter id="dv-glow-blue" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <filter id="dv-glow-orange" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="2.5" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>

        {/* baseline zone */}
        <rect x={PAD.l} y={threshY} width={GW} height={PAD.t + GH - threshY} fill={`${PALETTE.neon}08`} rx={2} />

        {/* deviation zone */}
        {isAbove && (
          <rect x={PAD.l} y={PAD.t} width={GW} height={threshY - PAD.t} fill={`${sev === Severity.HIGH ? PALETTE.red : PALETTE.orange}06`} rx={2} />
        )}

        {/* threshold line */}
        <line x1={PAD.l} y1={threshY} x2={W - PAD.r} y2={threshY}
          stroke={PALETTE.textMuted} strokeWidth={0.6} strokeDasharray="3 3" opacity={0.6} />
        <text x={PAD.l - 4} y={threshY} textAnchor="end" dominantBaseline="central"
          fill={PALETTE.textMuted} fontSize={7} fontFamily="-apple-system,sans-serif">
          {detectionThreshold.toFixed(1)}
        </text>

        {/* Y axis labels */}
        <text x={PAD.l - 4} y={PAD.t} textAnchor="end" dominantBaseline="central"
          fill={PALETTE.textMuted} fontSize={7} opacity={0.7}>1.0</text>
        <text x={PAD.l - 4} y={PAD.t + GH} textAnchor="end" dominantBaseline="central"
          fill={PALETTE.textMuted} fontSize={7} opacity={0.7}>0</text>

        {/* drift area */}
        {driftPath && (
          <path d={`${driftPath} L${driftPts[n-1]?.x.toFixed(1) ?? 0},${(PAD.t + GH).toFixed(1)} L${driftPts[0]?.x.toFixed(1) ?? 0},${(PAD.t + GH).toFixed(1)} Z`}
            fill="url(#dv-drift-grad)" />
        )}

        {/* instability area */}
        {instabilityPath && (
          <path d={`${instabilityPath} L${instabilityPts[n-1]?.x.toFixed(1) ?? 0},${(PAD.t + GH).toFixed(1)} L${instabilityPts[0]?.x.toFixed(1) ?? 0},${(PAD.t + GH).toFixed(1)} Z`}
            fill="url(#dv-inst-grad)" />
        )}

        {/* drift line — smooth, glowing */}
        {driftPath && (
          <path d={driftPath} fill="none" stroke={PALETTE.blue} strokeWidth={2.2}
            strokeLinecap="round" strokeLinejoin="round" filter="url(#dv-glow-blue)" />
        )}

        {/* instability line — sharp, angular */}
        {instabilityPath && (
          <path d={instabilityPath} fill="none" stroke={PALETTE.orange} strokeWidth={1.4}
            strokeLinejoin="round" opacity={0.9} filter="url(#dv-glow-orange)" />
        )}

        {/* current marker */}
        <line x1={currentX} y1={PAD.t} x2={currentX} y2={PAD.t + GH}
          stroke="rgba(255,255,255,0.08)" strokeWidth={1} strokeDasharray="2 2" />
        <circle cx={currentX} cy={currentY} r={5} fill={PALETTE.blue} filter="url(#dv-glow-blue)" />
        <circle cx={currentX} cy={currentY} r={2.5} fill="#fff" />

        {/* NOW label */}
        <text x={currentX} y={PAD.t + GH + 12} textAnchor="middle"
          fill={PALETTE.textMuted} fontSize={7} letterSpacing="0.1em" fontWeight={700}>
          NOW
        </text>
      </svg>
    </div>
  )
}
