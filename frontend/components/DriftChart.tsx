'use client'

import { DriftChart as DriftChartType } from '@/lib/decisionToUI'
import { useEffect, useRef } from 'react'

interface DriftChartProps {
  chart: DriftChartType
}

export default function DriftChart({ chart }: DriftChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!canvasRef.current || chart.dataPoints.length === 0) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.offsetWidth
    const height = canvas.offsetHeight

    // Set canvas resolution
    canvas.width = width * window.devicePixelRatio
    canvas.height = height * window.devicePixelRatio
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio)

    // Clear canvas
    ctx.fillStyle = 'rgba(30, 30, 30, 0.5)'
    ctx.fillRect(0, 0, width, height)

    // Dimensions
    const padding = { top: 16, right: 16, bottom: 24, left: 40 }
    const chartWidth = width - padding.left - padding.right
    const chartHeight = height - padding.top - padding.bottom

    // Draw grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)'
    ctx.lineWidth = 1
    const gridLines = 4
    for (let i = 0; i <= gridLines; i++) {
      const y = padding.top + (i * chartHeight) / gridLines
      ctx.beginPath()
      ctx.moveTo(padding.left, y)
      ctx.lineTo(width - padding.right, y)
      ctx.stroke()
    }

    // Draw threshold line
    const thresholdY = padding.top + chartHeight * (1 - chart.detectionThreshold)
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.3)'
    ctx.lineWidth = 2
    ctx.setLineDash([4, 4])
    ctx.beginPath()
    ctx.moveTo(padding.left, thresholdY)
    ctx.lineTo(width - padding.right, thresholdY)
    ctx.stroke()
    ctx.setLineDash([])

    // Draw data points and line
    if (chart.dataPoints.length > 1) {
      ctx.strokeStyle = '#3b82f6'
      ctx.lineWidth = 2
      ctx.beginPath()

      chart.dataPoints.forEach((point, idx) => {
        const x = padding.left + (idx / (chart.dataPoints.length - 1)) * chartWidth
        const y = padding.top + chartHeight * (1 - point.driftScore)

        if (idx === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      })

      ctx.stroke()
    }

    // Draw data point dots
    chart.dataPoints.forEach((point, idx) => {
      const x = padding.left + (idx / (chart.dataPoints.length - 1)) * chartWidth
      const y = padding.top + chartHeight * (1 - point.driftScore)
      const isCurrentFrame = idx === chart.currentFrameIdx

      ctx.fillStyle = isCurrentFrame ? '#3b82f6' : 'rgba(59, 130, 246, 0.4)'
      ctx.beginPath()
      ctx.arc(x, y, isCurrentFrame ? 4 : 2, 0, Math.PI * 2)
      ctx.fill()
    })

    // Draw Y-axis labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.4)'
    ctx.font = '11px system-ui, -apple-system, sans-serif'
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'

    for (let i = 0; i <= 4; i++) {
      const value = i / 4
      const y = padding.top + ((4 - i) * chartHeight) / 4
      const label = (value * 100).toFixed(0) + '%'
      ctx.fillText(label, padding.left - 8, y)
    }

    // Draw threshold label
    ctx.fillStyle = 'rgba(239, 68, 68, 0.6)'
    ctx.textAlign = 'right'
    ctx.fillText('Threshold', padding.left - 8, thresholdY)
  }, [chart])

  const currentDrift = chart.dataPoints[chart.currentFrameIdx]?.driftScore ?? 0
  const maxDrift = Math.max(...chart.dataPoints.map((p) => p.driftScore), 0)

  return (
    <div style={styles.container}>
      <div style={styles.label}>DRIFT TRAJECTORY</div>

      <div style={styles.chartContainer} ref={containerRef}>
        <canvas ref={canvasRef} style={styles.canvas} />
      </div>

      <div style={styles.statsRow}>
        <div style={styles.stat}>
          <span style={styles.statLabel}>Current Drift</span>
          <span style={styles.statValue}>{(currentDrift * 100).toFixed(1)}%</span>
        </div>
        <div style={styles.stat}>
          <span style={styles.statLabel}>Peak Drift</span>
          <span style={styles.statValue}>{(maxDrift * 100).toFixed(1)}%</span>
        </div>
        <div style={styles.stat}>
          <span style={styles.statLabel}>Points</span>
          <span style={styles.statValue}>{chart.dataPoints.length}</span>
        </div>
      </div>
    </div>
  )
}

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '16px',
    padding: '24px',
    backgroundColor: 'rgba(15, 15, 15, 0.6)',
    borderRadius: '12px',
    border: '1px solid rgba(255, 255, 255, 0.08)',
    backdropFilter: 'blur(4px)',
    transition: 'all 0.3s ease',
  },
  label: {
    fontSize: '10px',
    fontWeight: '700',
    letterSpacing: '0.1em',
    color: 'rgba(255, 255, 255, 0.4)',
    textTransform: 'uppercase' as const,
  },
  chartContainer: {
    position: 'relative' as const,
    height: '200px',
    backgroundColor: 'rgba(255, 255, 255, 0.02)',
    borderRadius: '8px',
    border: '1px solid rgba(255, 255, 255, 0.06)',
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
    gap: '16px',
    paddingTop: '12px',
    borderTop: '1px solid rgba(255, 255, 255, 0.08)',
  },
  stat: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '6px',
  },
  statLabel: {
    fontSize: '9px',
    fontWeight: '700',
    letterSpacing: '0.1em',
    color: 'rgba(255, 255, 255, 0.35)',
    textTransform: 'uppercase' as const,
  },
  statValue: {
    fontSize: '13px',
    fontWeight: '600',
    color: '#0ea5e9',
    fontVariantNumeric: 'tabular-nums',
  },
}
