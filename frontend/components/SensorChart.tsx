'use client'

import React, { useRef, useEffect } from 'react'
import styles from './SensorChart.module.css'

interface Props {
  label: string
  value: number
  unit: string
  status: 'optimal' | 'warning' | 'critical'
  history: number[]
  threshold?: {
    min: number
    max: number
    critical_max?: number
  }
}

export default function SensorChart({
  label,
  value,
  unit,
  status,
  history,
  threshold,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!canvasRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height
    const padding = 4

    ctx.clearRect(0, 0, width, height)

    if (history.length > 1) {
      const min = Math.min(...history, threshold?.min || 0)
      const max = Math.max(...history, threshold?.max || 100)
      const range = max - min || 1

      const xStep = (width - 2 * padding) / (history.length - 1)
      const yScale = (height - 2 * padding) / range

      ctx.strokeStyle = '#22c55e'
      ctx.lineWidth = 2
      ctx.beginPath()

      history.forEach((val, i) => {
        const x = padding + i * xStep
        const y = height - padding - (val - min) * yScale

        if (i === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      })

      ctx.stroke()

      if (threshold) {
        ctx.strokeStyle = 'rgba(34, 197, 94, 0.3)'
        ctx.lineWidth = 1
        ctx.setLineDash([2, 2])

        const minY = height - padding - (threshold.min - min) * yScale
        ctx.beginPath()
        ctx.moveTo(padding, minY)
        ctx.lineTo(width - padding, minY)
        ctx.stroke()

        const maxY = height - padding - (threshold.max - min) * yScale
        ctx.beginPath()
        ctx.moveTo(padding, maxY)
        ctx.lineTo(width - padding, maxY)
        ctx.stroke()

        if (threshold.critical_max) {
          ctx.strokeStyle = 'rgba(239, 68, 68, 0.3)'
          const critY = height - padding - (threshold.critical_max - min) * yScale
          ctx.beginPath()
          ctx.moveTo(padding, critY)
          ctx.lineTo(width - padding, critY)
          ctx.stroke()
        }

        ctx.setLineDash([])
      }
    }
  }, [history, threshold])

  const getStatusColor = () => {
    switch (status) {
      case 'optimal':
        return '#10b981'
      case 'warning':
        return '#f59e0b'
      case 'critical':
        return '#ef4444'
      default:
        return '#6b7280'
    }
  }

  return (
    <div className={styles.card}>
      <div className={styles.header}>
        <div>
          <h3>{label}</h3>
          <div className={styles.value}>
            <span className={styles.number}>
              {value.toFixed(
                unit === '°F' || unit === '%' || unit === 'ppm' ? 1 : 2
              )}
            </span>
            <span className={styles.unit}>{unit}</span>
          </div>
        </div>
        <div
          className={styles.statusBadge}
          style={{ backgroundColor: getStatusColor() }}
        >
          {status.charAt(0).toUpperCase() + status.slice(1)}
        </div>
      </div>

      {threshold && (
        <div className={styles.thresholds}>
          <span>
            Range: {threshold.min} - {threshold.max}
            {threshold.critical_max && ` | Critical: >${threshold.critical_max}`}
          </span>
        </div>
      )}

      <canvas
        ref={canvasRef}
        className={styles.chart}
        width={300}
        height={100}
      />
    </div>
  )
}
