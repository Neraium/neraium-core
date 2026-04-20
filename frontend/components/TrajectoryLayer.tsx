'use client'

import React, { useRef, useEffect, useState } from 'react'

interface TrajectoryData {
  interpolatedDrift: number
  interpolatedStability: number
  interpolatedCoherence: number
  interpolatedConfidence: number
}

interface TrajectoryLayerProps {
  data: TrajectoryData
}

export function TrajectoryLayer({ data }: TrajectoryLayerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const historyRef = useRef<Array<{ x: number; y: number }>>([])
  const projectionRef = useRef<Array<{ x: number; y: number }>>([])

  useEffect(() => {
    if (!canvasRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    const rect = canvas.getBoundingClientRect()
    canvas.width = window.innerWidth - 80
    canvas.height = 280

    // Map system state to 2D coordinate (drift vs stability)
    const currentX = (data.interpolatedDrift * canvas.width * 0.8) + canvas.width * 0.1
    const currentY = canvas.height - (data.interpolatedStability * canvas.height * 0.7) - canvas.height * 0.15

    // Add current point to history (limit to 100 points for performance)
    historyRef.current.push({ x: currentX, y: currentY })
    if (historyRef.current.length > 100) {
      historyRef.current.shift()
    }

    // Generate projection (faint future path)
    const driftVelocity = (data.interpolatedDrift - (historyRef.current[historyRef.current.length - 2]?.x ?? currentX)) / canvas.width
    const stabilityVelocity = (data.interpolatedStability - (historyRef.current[historyRef.current.length - 2]?.y ?? currentY)) / canvas.height

    projectionRef.current = []
    for (let i = 1; i <= 20; i++) {
      const projX = currentX + (driftVelocity * i * 8)
      const projY = currentY + (stabilityVelocity * i * 8)
      projectionRef.current.push({ x: projX, y: projY })
    }

    // Clear canvas
    ctx.fillStyle = 'rgba(10, 14, 26, 0.5)'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Draw axis hints (very subtle)
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.1)'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(canvas.width * 0.1, 0)
    ctx.lineTo(canvas.width * 0.1, canvas.height)
    ctx.stroke()

    ctx.beginPath()
    ctx.moveTo(0, canvas.height - canvas.height * 0.15)
    ctx.lineTo(canvas.width, canvas.height - canvas.height * 0.15)
    ctx.stroke()

    // Draw projected path (very faint)
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.2)'
    ctx.lineWidth = 1.5
    ctx.setLineDash([4, 4])
    ctx.beginPath()
    if (projectionRef.current.length > 0) {
      ctx.moveTo(currentX, currentY)
      projectionRef.current.forEach((point) => {
        ctx.lineTo(point.x, point.y)
      })
    }
    ctx.stroke()
    ctx.setLineDash([])

    // Draw actual trajectory path
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.7)'
    ctx.lineWidth = 2
    ctx.beginPath()
    if (historyRef.current.length > 0) {
      ctx.moveTo(historyRef.current[0].x, historyRef.current[0].y)
      historyRef.current.forEach((point) => {
        ctx.lineTo(point.x, point.y)
      })
    }
    ctx.stroke()

    // Draw current position
    ctx.fillStyle = '#38BDF8'
    ctx.shadowColor = 'rgba(56, 189, 248, 0.6)'
    ctx.shadowBlur = 8
    ctx.beginPath()
    ctx.arc(currentX, currentY, 6, 0, Math.PI * 2)
    ctx.fill()
    ctx.shadowBlur = 0

    // Draw path history fade (older points more transparent)
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.3)'
    ctx.lineWidth = 1
    ctx.setLineDash([2, 2])
    ctx.beginPath()
    if (historyRef.current.length > 20) {
      ctx.moveTo(historyRef.current[0].x, historyRef.current[0].y)
      historyRef.current.slice(0, Math.max(1, historyRef.current.length - 40)).forEach((point) => {
        ctx.lineTo(point.x, point.y)
      })
    }
    ctx.stroke()
    ctx.setLineDash([])
  }, [data])

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        position: 'relative',
      }}
    >
      <div
        style={{
          width: '100%',
          maxWidth: '100%',
          paddingLeft: '40px',
          paddingRight: '40px',
        }}
      >
        <div
          style={{
            fontSize: '11px',
            letterSpacing: '0.08em',
            color: '#94a3b8',
            textTransform: 'uppercase',
            fontWeight: 700,
            marginBottom: '16px',
          }}
        >
          System Movement Through State Space
        </div>
        <canvas
          ref={canvasRef}
          style={{
            width: '100%',
            height: '260px',
            borderRadius: '2px',
            backgroundColor: 'rgba(2, 6, 23, 0.4)',
          }}
        />
        <div
          style={{
            display: 'flex',
            gap: '32px',
            marginTop: '12px',
            fontSize: '11px',
            color: '#94a3b8',
          }}
        >
          <span>
            <span style={{ color: '#38BDF8' }}>━</span> Actual path
          </span>
          <span>
            <span style={{ color: '#38BDF8' }}>┄</span> Projected path
          </span>
        </div>
      </div>
    </div>
  )
}
