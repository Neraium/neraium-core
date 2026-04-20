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
  escalationLevel?: number
}

export function TrajectoryLayer({ data, escalationLevel = 0 }: TrajectoryLayerProps) {
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
    // At critical: projection becomes erratic (consequence feeling)
    const driftVelocity = (data.interpolatedDrift - (historyRef.current[historyRef.current.length - 2]?.x ?? currentX)) / canvas.width
    const stabilityVelocity = (data.interpolatedStability - (historyRef.current[historyRef.current.length - 2]?.y ?? currentY)) / canvas.height

    projectionRef.current = []
    for (let i = 1; i <= 20; i++) {
      // At critical: add erratic noise to projection
      const erraticAmount = escalationLevel >= 3 ? Math.random() * 40 - 20 : 0
      const projX = currentX + (driftVelocity * i * 8) + erraticAmount
      const projY = currentY + (stabilityVelocity * i * 8) + erraticAmount * 0.5
      projectionRef.current.push({ x: projX, y: projY })
    }

    // Clear canvas
    ctx.fillStyle = 'rgba(10, 14, 26, 0.4)'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // No axis hints - feel like memory, not chart

    // Draw projected path (uncertain, blurred)
    const projectionOpacity = Math.min(0.15 + data.interpolatedDrift * 0.15, 0.4)
    ctx.strokeStyle = `rgba(56, 189, 248, ${projectionOpacity})`
    ctx.lineWidth = 1
    ctx.setLineDash([3, 3])
    ctx.filter = `blur(${1 + data.interpolatedDrift * 2}px)`
    ctx.beginPath()
    if (projectionRef.current.length > 0) {
      ctx.moveTo(currentX, currentY)
      projectionRef.current.forEach((point) => {
        ctx.lineTo(point.x, point.y)
      })
    }
    ctx.stroke()
    ctx.setLineDash([])
    ctx.filter = 'none'

    // Draw actual trajectory path with momentum-based thickness
    const momentum = Math.abs(
      (historyRef.current[historyRef.current.length - 1]?.x ?? currentX) -
      (historyRef.current[historyRef.current.length - 2]?.x ?? currentX)
    ) / canvas.width
    const thickness = 1.5 + momentum * 2

    ctx.strokeStyle = 'rgba(56, 189, 248, 0.75)'
    ctx.lineWidth = thickness
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.beginPath()
    if (historyRef.current.length > 0) {
      ctx.moveTo(historyRef.current[0].x, historyRef.current[0].y)
      historyRef.current.forEach((point) => {
        ctx.lineTo(point.x, point.y)
      })
    }
    ctx.stroke()

    // Draw current position with glow intensity based on drift
    const glowRadius = 8 + data.interpolatedDrift * 6
    const glowOpacity = 0.4 + data.interpolatedDrift * 0.5

    ctx.fillStyle = `rgba(56, 189, 248, ${glowOpacity * 0.3})`
    ctx.shadowColor = `rgba(56, 189, 248, ${glowOpacity})`
    ctx.shadowBlur = glowRadius
    ctx.beginPath()
    ctx.arc(currentX, currentY, 6, 0, Math.PI * 2)
    ctx.fill()
    ctx.shadowBlur = 0

    // Core point
    ctx.fillStyle = '#38BDF8'
    ctx.beginPath()
    ctx.arc(currentX, currentY, 4, 0, Math.PI * 2)
    ctx.fill()
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
