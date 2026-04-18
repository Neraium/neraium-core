'use client'

import { DriftChart as DriftChartType } from '@/lib/decisionToUI'
import { useEffect, useRef } from 'react'

interface DriftChartProps {
  chart: DriftChartType
  minimal?: boolean
}

export default function DriftChart({ chart, minimal = false }: DriftChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

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

    const padding = { top: 16, right: 16, bottom: 24, left: 40 }
    const chartWidth = width - padding.left - padding.right
    const chartHeight = height - padding.top - padding.bottom

    ctx.strokeStyle = `rgba(148, 163, 184, ${0.14 * m})`
    ctx.lineWidth = 1
    for (let i = 0; i <= 4; i++) {
      const y = padding.top + (i * chartHeight) / 4
      ctx.beginPath()
      ctx.moveTo(padding.left, y)
      ctx.lineTo(width - padding.right, y)
      ctx.stroke()
    }

    const thresholdY = padding.top + chartHeight * (1 - chart.detectionThreshold)
    ctx.strokeStyle = `rgba(239, 68, 68, ${0.38 * m})`
    ctx.lineWidth = 1.5
    ctx.setLineDash([4, 4])
    ctx.beginPath()
    ctx.moveTo(padding.left, thresholdY)
    ctx.lineTo(width - padding.right, thresholdY)
    ctx.stroke()
    ctx.setLineDash([])

    if (chart.dataPoints.length > 1) {
      ctx.strokeStyle = minimal ? 'rgba(96, 165, 250, 0.35)' : '#60a5fa'
      ctx.lineWidth = minimal ? 1.5 : 2
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

      ctx.fillStyle = isCurrentFrame
        ? (minimal ? 'rgba(147, 197, 253, 0.55)' : '#93c5fd')
        : `rgba(96, 165, 250, ${0.34 * m})`
      ctx.beginPath()
      ctx.arc(x, y, isCurrentFrame ? 3.5 : 2, 0, Math.PI * 2)
      ctx.fill()
    })

    ctx.fillStyle = `rgba(203, 213, 225, ${0.45 * m})`
    ctx.font = '11px system-ui, -apple-system, sans-serif'
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'

    for (let i = 0; i <= 4; i++) {
      const value = i / 4
      const y = padding.top + ((4 - i) * chartHeight) / 4
      ctx.fillText((value * 100).toFixed(0) + '%', padding.left - 8, y)
    }

    ctx.fillStyle = `rgba(248, 113, 113, ${0.74 * m})`
    ctx.fillText('Threshold', padding.left - 8, thresholdY)
  }, [chart, minimal])

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
            <span style={styles.statValue}>{(currentDrift * 100).toFixed(1)}%</span>
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
    gap: '15px',
    padding: '13px 0',
    transition: 'all 0.62s ease',
  },
  label: {
    fontSize: '11px',
    fontWeight: '700',
    letterSpacing: '0.12em',
    color: 'rgba(203, 213, 225, 0.54)',
    textTransform: 'uppercase',
    transition: 'opacity 0.62s ease',
  },
  chartContainer: {
    position: 'relative',
    height: '140px',
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
    gap: '15px',
    paddingTop: '8px',
  },
  stat: {
    display: 'flex',
    flexDirection: 'column',
    gap: '5px',
  },
  statLabel: {
    fontSize: '9px',
    fontWeight: '700',
    letterSpacing: '0.12em',
    color: 'rgba(148, 163, 184, 0.66)',
    textTransform: 'uppercase',
  },
  statValue: {
    fontSize: '13px',
    fontWeight: '600',
    color: '#93c5fd',
    fontVariantNumeric: 'tabular-nums',
  },
}
