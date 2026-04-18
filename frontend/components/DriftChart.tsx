'use client'

import { DriftChart as DriftChartType, Severity } from '@/lib/decisionToUI'
import { useEffect, useRef } from 'react'

interface DriftChartProps {
  chart: DriftChartType
  minimal?: boolean
  severity?: Severity
}

function severityToColor(severity?: Severity): string {
  switch (severity) {
    case Severity.HIGH: return '#ef4444'
    case Severity.ELEVATED: return '#f97316'
    case Severity.MODERATE: return '#eab308'
    default: return '#60a5fa'
  }
}

export default function DriftChart({ chart, minimal = false, severity }: DriftChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const lineColor = severityToColor(severity)

  useEffect(() => {
    if (!canvasRef.current || chart.dataPoints.length === 0) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.offsetWidth
    const height = canvas.offsetHeight

    canvas.width = width * window.devicePixelRatio
    canvas.height = height * window.devicePixelRatio
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio)

    const m = minimal ? 0.35 : 1

    ctx.fillStyle = `rgba(2, 6, 23, ${0.28 * m})`
    ctx.fillRect(0, 0, width, height)

    const padding = { top: 14, right: 14, bottom: 22, left: 36 }
    const chartWidth = width - padding.left - padding.right
    const chartHeight = height - padding.top - padding.bottom

    ctx.strokeStyle = `rgba(148, 163, 184, ${0.12 * m})`
    ctx.lineWidth = 1
    for (let i = 0; i <= 4; i++) {
      const y = padding.top + (i * chartHeight) / 4
      ctx.beginPath()
      ctx.moveTo(padding.left, y)
      ctx.lineTo(width - padding.right, y)
      ctx.stroke()
    }

    const thresholdY = padding.top + chartHeight * (1 - chart.detectionThreshold)
    ctx.strokeStyle = `rgba(239, 68, 68, ${0.28 * m})`
    ctx.lineWidth = 1.2
    ctx.setLineDash([3, 4])
    ctx.beginPath()
    ctx.moveTo(padding.left, thresholdY)
    ctx.lineTo(width - padding.right, thresholdY)
    ctx.stroke()
    ctx.setLineDash([])

    if (chart.dataPoints.length > 1) {
      ctx.strokeStyle = minimal ? `${lineColor}59` : lineColor
      ctx.lineWidth = minimal ? 1.5 : 2.2
      ctx.lineJoin = 'round'
      ctx.lineCap = 'round'
      ctx.beginPath()
      chart.dataPoints.forEach((point, idx) => {
        const x = padding.left + (idx / (chart.dataPoints.length - 1)) * chartWidth
        const y = padding.top + chartHeight * (1 - point.driftScore)
        if (idx === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.stroke()
    }

    chart.dataPoints.forEach((point, idx) => {
      const x = padding.left + (idx / (chart.dataPoints.length - 1)) * chartWidth
      const y = padding.top + chartHeight * (1 - point.driftScore)
      const isCurrentFrame = idx === chart.currentFrameIdx

      if (isCurrentFrame) {
        ctx.fillStyle = minimal ? `${lineColor}99` : lineColor
        ctx.beginPath()
        ctx.arc(x, y, 4.5, 0, Math.PI * 2)
        ctx.fill()
        ctx.strokeStyle = `rgba(248, 250, 252, ${0.35 * m})`
        ctx.lineWidth = 1.2
        ctx.beginPath()
        ctx.arc(x, y, 7, 0, Math.PI * 2)
        ctx.stroke()
      } else {
        ctx.fillStyle = `${lineColor}${Math.round(0.28 * m * 255).toString(16).padStart(2, '0')}`
        ctx.beginPath()
        ctx.arc(x, y, 2, 0, Math.PI * 2)
        ctx.fill()
      }
    })

    ctx.fillStyle = `rgba(148, 163, 184, ${0.45 * m})`
    ctx.font = '10px system-ui, -apple-system, sans-serif'
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'

    for (let i = 0; i <= 4; i++) {
      const value = i / 4
      const y = padding.top + ((4 - i) * chartHeight) / 4
      ctx.fillText((value * 100).toFixed(0), padding.left - 7, y)
    }

    ctx.fillStyle = `rgba(239, 68, 68, ${0.55 * m})`
    ctx.font = '9px system-ui, -apple-system, sans-serif'
    ctx.fillText('Threshold', padding.left - 7, thresholdY)
  }, [chart, minimal, lineColor])

  const currentDrift = chart.dataPoints[chart.currentFrameIdx]?.driftScore ?? 0
  const maxDrift = Math.max(...chart.dataPoints.map((p) => p.driftScore), 0)

  return (
    <div style={styles.container}>
      <div style={{ ...styles.label, opacity: minimal ? 0.45 : 1 }}>Drift</div>

      <div style={styles.chartContainer}>
        <canvas ref={canvasRef} style={styles.canvas} />
      </div>

      {!minimal && (
        <div style={styles.statsRow}>
          <div style={styles.stat}>
            <span style={styles.statLabel}>Current</span>
            <span style={{ ...styles.statValue, color: lineColor }}>{(currentDrift * 100).toFixed(1)}%</span>
          </div>
          <div style={styles.stat}>
            <span style={styles.statLabel}>Peak</span>
            <span style={styles.statValue}>{(maxDrift * 100).toFixed(1)}%</span>
          </div>
          <div style={styles.stat}>
            <span style={styles.statLabel}>Points</span>
            <span style={styles.statValue}>{chart.dataPoints.length}</span>
          </div>
        </div>
      )}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    padding: '10px 0',
    transition: 'all 0.9s cubic-bezier(0.22, 1, 0.36, 1)',
  },
  label: {
    fontSize: '10px',
    fontWeight: '700',
    letterSpacing: '0.14em',
    color: 'rgba(148, 163, 184, 0.55)',
    textTransform: 'uppercase',
    transition: 'opacity 0.9s cubic-bezier(0.22, 1, 0.36, 1)',
  },
  chartContainer: {
    position: 'relative',
    height: '130px',
    borderRadius: '10px',
    overflow: 'hidden',
  },
  canvas: {
    width: '100%',
    height: '100%',
    display: 'block',
  },
  statsRow: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '12px',
    paddingTop: '6px',
  },
  stat: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  statLabel: {
    fontSize: '9px',
    fontWeight: '700',
    letterSpacing: '0.12em',
    color: 'rgba(148, 163, 184, 0.58)',
    textTransform: 'uppercase',
  },
  statValue: {
    fontSize: '13px',
    fontWeight: '600',
    color: '#93c5fd',
    fontVariantNumeric: 'tabular-nums',
  },
}
