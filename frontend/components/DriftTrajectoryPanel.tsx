'use client'

import React, { useMemo } from 'react'
import { DecisionUIState, Severity } from '@/lib/decisionToUI'

interface Props {
  state: DecisionUIState
}

function applyEMA(values: number[], alpha = 0.38): number[] {
  const out: number[] = []
  let prev = values[0] ?? 0
  for (const v of values) {
    prev = alpha * v + (1 - alpha) * prev
    out.push(prev)
  }
  return out
}

function toPath(pts: Array<{ x: number; y: number }>): string {
  return pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(2)},${p.y.toFixed(2)}`).join(' ')
}

// ─── Drift Chart constants ────────────────────────────────────────────────────
const DCW = 380
const DCH = 180
const DCL = 42
const DCR = 14
const DCT = 18
const DCB = 26
const dcChartW = DCW - DCL - DCR
const dcChartH = DCH - DCT - DCB

// ─── Future Paths constants ───────────────────────────────────────────────────
const FPW = 380
const FPH = 148
const FPL = 48
const FPR = 96
const FPT = 18
const FPB = 18
// fpW and fpH available if needed for future layout calculations
const FP_START_X = FPL + 14
const FP_END_X   = FPW - FPR - 8
const FP_CENTER_Y = FPT + FPH * 0.55

export function DriftTrajectoryPanel({ state }: Props) {
  const { driftChart, statusHeader } = state
  const severity = statusHeader.severity

  // ── Drift Chart ─────────────────────────────────────────────────────────────
  const smoothed   = applyEMA(driftChart.dataPoints.map(p => p.driftScore), 0.42)
  const n          = smoothed.length
  const threshold  = driftChart.detectionThreshold

  const xAt = (i: number) => DCL + (i / Math.max(n - 1, 1)) * dcChartW
  const yAt = (v: number) => DCT + dcChartH * (1 - Math.min(Math.max(v, 0), 1))

  const thresholdY = yAt(threshold)
  const linePoints = smoothed.map((v, i) => ({ x: xAt(i), y: yAt(v) }))
  const linePath   = toPath(linePoints)

  const currentValue = smoothed[n - 1] ?? 0
  const prevValue    = smoothed[Math.max(0, n - 4)] ?? 0
  const trend        = currentValue - prevValue

  const currentX = xAt(n - 1)
  const currentY = linePoints[n - 1]?.y ?? 0

  const lineColor =
    severity === Severity.HIGH     ? '#ef4444' :
    severity === Severity.ELEVATED ? '#f59e0b' :
    severity === Severity.MODERATE ? '#fbbf24' : '#22c55e'

  const isAboveThreshold = currentValue > threshold

  // Arrow angle: negative = up-right (worsening), 0 = flat, positive = down
  const arrowAngle = trend > 0.012 ? -42 : trend < -0.012 ? 35 : 0

  // ── Future Paths ─────────────────────────────────────────────────────────────
  const futurePaths = useMemo(() => {
    const dx   = FP_END_X - FP_START_X
    const cp1x = FP_START_X + dx * 0.42
    const cp2x = FP_END_X   - dx * 0.28

    return [
      {
        d: `M${FP_START_X},${FP_CENTER_Y} C${cp1x},${FP_CENTER_Y} ${cp2x},${FP_CENTER_Y + 6} ${FP_END_X},${FP_CENTER_Y + 7}`,
        color: '#22c55e', label: 'Stable continuation',
        opacity: 0.9, strokeWidth: 2.2, dasharray: undefined,
        dotY: FP_CENTER_Y + 7,
      },
      {
        d: `M${FP_START_X},${FP_CENTER_Y} C${cp1x},${FP_CENTER_Y} ${cp2x},${FP_CENTER_Y - 28} ${FP_END_X},${FP_CENTER_Y - 32}`,
        color: '#f59e0b', label: 'Instability forming',
        opacity: 0.55, strokeWidth: 1.6, dasharray: '7 5',
        dotY: FP_CENTER_Y - 32,
      },
      {
        d: `M${FP_START_X},${FP_CENTER_Y} C${cp1x},${FP_CENTER_Y} ${cp2x},${FP_CENTER_Y - 62} ${FP_END_X},${FP_CENTER_Y - 68}`,
        color: '#ef4444', label: 'Critical transition',
        opacity: 0.3, strokeWidth: 1.6, dasharray: '4 6',
        dotY: FP_CENTER_Y - 68,
      },
    ]
  }, [])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>

      {/* ── Drift Chart ──────────────────────────────────────────────────────── */}
      <div style={{
        flex: '0 0 auto',
        padding: '12px 16px 8px',
        borderBottom: '1px solid rgba(255,255,255,0.05)',
      }}>
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          marginBottom: 8,
        }}>
          <span style={{ fontSize: 10, color: '#4b5563', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 700 }}>
            Structural Drift
          </span>
          <span style={{ fontSize: 12, fontWeight: 800, color: lineColor, letterSpacing: '0.04em' }}>
            {(currentValue * 100).toFixed(1)}%
            <span style={{ marginLeft: 5, fontSize: 11 }}>
              {trend > 0.012 ? '↑' : trend < -0.012 ? '↓' : '→'}
            </span>
          </span>
        </div>

        <svg viewBox={`0 0 ${DCW} ${DCH}`} width="100%" height={DCH}
          preserveAspectRatio="none" style={{ display: 'block' }}>
          <defs>
            <linearGradient id="dtLineGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%"   stopColor="#22c55e" stopOpacity="0.75" />
              <stop offset="100%" stopColor={lineColor} stopOpacity="1" />
            </linearGradient>
            <linearGradient id="dtAreaGrad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%"   stopColor={lineColor} stopOpacity="0.18" />
              <stop offset="100%" stopColor={lineColor} stopOpacity="0.02" />
            </linearGradient>
          </defs>

          {/* stable baseline zone */}
          <rect x={DCL} y={thresholdY} width={dcChartW} height={DCT + dcChartH - thresholdY}
            fill="rgba(34,197,94,0.06)" />

          {/* deviation zone */}
          <rect x={DCL} y={DCT} width={dcChartW} height={thresholdY - DCT}
            fill={isAboveThreshold ? `rgba(${severity === Severity.HIGH ? '239,68,68' : '245,158,11'},0.06)` : 'none'} />

          {/* zone labels */}
          <text x={DCL + 4} y={thresholdY - 5} fill="rgba(34,197,94,0.4)" fontSize={7}
            fontFamily="-apple-system,sans-serif" letterSpacing="0.08em">BASELINE</text>
          <text x={DCL + 4} y={DCT + 10} fill={`rgba(${isAboveThreshold ? (severity === Severity.HIGH ? '239,68,68' : '245,158,11') : '100,116,139'},0.4)`}
            fontSize={7} fontFamily="-apple-system,sans-serif" letterSpacing="0.08em">DEVIATION</text>

          {/* threshold line */}
          <line x1={DCL} y1={thresholdY} x2={DCW - DCR} y2={thresholdY}
            stroke="#374151" strokeWidth={0.8} strokeDasharray="4 3" />
          <text x={DCL - 5} y={thresholdY} textAnchor="end" dominantBaseline="central"
            fill="#374151" fontSize={8} fontFamily="-apple-system,sans-serif">
            {threshold.toFixed(1)}
          </text>

          {/* Y axis ticks */}
          <text x={DCL - 5} y={DCT} textAnchor="end" dominantBaseline="central"
            fill="rgba(55,65,81,0.8)" fontSize={8} fontFamily="-apple-system,sans-serif">1.0</text>
          <text x={DCL - 5} y={DCT + dcChartH} textAnchor="end" dominantBaseline="central"
            fill="rgba(55,65,81,0.8)" fontSize={8} fontFamily="-apple-system,sans-serif">0.0</text>

          {/* Area fill under line */}
          <path
            d={`${linePath} L${xAt(n-1).toFixed(2)},${(DCT + dcChartH).toFixed(2)} L${xAt(0).toFixed(2)},${(DCT + dcChartH).toFixed(2)} Z`}
            fill="url(#dtAreaGrad)"
          />

          {/* Drift line */}
          <path d={linePath} fill="none"
            stroke="url(#dtLineGrad)" strokeWidth={2.2}
            strokeLinecap="round" strokeLinejoin="round" />

          {/* Current position: vertical guide */}
          <line x1={currentX} y1={DCT} x2={currentX} y2={DCT + dcChartH}
            stroke="rgba(255,255,255,0.18)" strokeWidth={1} strokeDasharray="3 2" />

          {/* Dot halo */}
          <circle cx={currentX} cy={currentY} r={10} fill={lineColor} opacity={0.12} />
          <circle cx={currentX} cy={currentY} r={5}  fill={lineColor} opacity={0.95} />

          {/* Trend arrow */}
          <g transform={`translate(${currentX + 14},${currentY - 5}) rotate(${arrowAngle})`}>
            <path d="M0,0 L9,0 M6,-3 L9,0 L6,3"
              stroke={lineColor} strokeWidth={1.8} strokeLinecap="round" fill="none" />
          </g>

          {/* NOW label */}
          <text x={currentX} y={DCT + dcChartH + 15} textAnchor="middle"
            fill="rgba(255,255,255,0.3)" fontSize={8}
            fontFamily="-apple-system,sans-serif" letterSpacing="0.1em">NOW</text>
        </svg>
      </div>

      {/* ── Future Path ───────────────────────────────────────────────────────── */}
      <div style={{ flex: 1, padding: '8px 16px 12px', minHeight: 0, overflow: 'hidden' }}>
        <div style={{
          fontSize: 10, color: '#4b5563', textTransform: 'uppercase',
          letterSpacing: '0.1em', fontWeight: 700, marginBottom: 6,
        }}>
          Projected Trajectory
        </div>

        <svg viewBox={`0 0 ${FPW} ${FPH}`} width="100%" height={FPH}
          preserveAspectRatio="none" style={{ display: 'block' }}>
          <style>{`
            @keyframes dtPathBreath { 0%,100%{opacity:.88} 50%{opacity:1} }
            .dt-path-stable { animation: dtPathBreath 3.5s ease-in-out infinite }
          `}</style>

          <defs>
            <linearGradient id="dtFadeOut" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="60%" stopColor="rgba(0,0,0,0)" />
              <stop offset="100%" stopColor="rgba(5,6,7,0.65)" />
            </linearGradient>
          </defs>

          {/* NOW anchor line */}
          <line x1={FP_START_X} y1={FPT} x2={FP_START_X} y2={FPH - FPB}
            stroke="rgba(255,255,255,0.1)" strokeWidth={1} />
          <text x={FP_START_X - 5} y={FP_CENTER_Y} textAnchor="end" dominantBaseline="central"
            fill="rgba(255,255,255,0.3)" fontSize={8}
            fontFamily="-apple-system,sans-serif" letterSpacing="0.08em">NOW</text>

          {/* Paths */}
          {futurePaths.map((fp, i) => (
            <g key={i} opacity={fp.opacity}>
              {/* glow */}
              <path d={fp.d} fill="none" stroke={fp.color}
                strokeWidth={fp.strokeWidth + 6} strokeOpacity={0.07}
                strokeLinecap="round" />
              {/* line */}
              <path d={fp.d} fill="none" stroke={fp.color}
                strokeWidth={fp.strokeWidth} strokeLinecap="round"
                strokeDasharray={fp.dasharray}
                className={i === 0 ? 'dt-path-stable' : undefined} />
              {/* end dot */}
              <circle cx={FP_END_X} cy={fp.dotY} r={3.5} fill={fp.color} />
              {/* label */}
              <text x={FP_END_X + 9} y={fp.dotY} dominantBaseline="central"
                fill={fp.color} fontSize={8.5} fontWeight="600"
                letterSpacing="0.04em"
                fontFamily="-apple-system,'Segoe UI',sans-serif">
                {fp.label}
              </text>
            </g>
          ))}

          {/* uncertainty fade */}
          <rect x={0} y={0} width={FPW} height={FPH} fill="url(#dtFadeOut)" />
        </svg>
      </div>
    </div>
  )
}
