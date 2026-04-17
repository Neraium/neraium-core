'use client'

import { useEffect, useState, useRef } from 'react'

interface TetrahedronPanelProps {
  frame: Record<string, unknown>
}

interface Particle {
  id: number
  x: number
  y: number
  vx: number
  vy: number
  life: number
  maxLife: number
  color: string
}

export default function TetrahedronPanel({ frame }: TetrahedronPanelProps) {
  const drift = Math.max(0, Math.min(1, (frame.structural_drift_score as number) || 0))
  const stability = Math.max(0, Math.min(1, (frame.relational_stability_score as number) || 0))
  const coherence = Math.max(0, Math.min(1, (frame.coherence_score as number) || 0))
  const confidence = Math.max(0, Math.min(1, (frame.confidence as number) || 0))

  const [rotation, setRotation] = useState({ x: 0, y: 0 })
  const [particles, setParticles] = useState<Particle[]>([])
  const [avgScore, setAvgScore] = useState(0)
  const particleCountRef = useRef(0)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const prevValuesRef = useRef({ drift, stability, coherence, confidence })

  const W = 420
  const H = 360
  const center = { x: W / 2, y: H / 2 + 20 }

  const vertices = [
    { x: center.x, y: center.y - 110, label: 'Coherence', value: coherence, color: '#06B6D4' },
    { x: center.x - 110, y: center.y + 85, label: 'Drift', value: drift, color: '#3B82F6' },
    { x: center.x + 110, y: center.y + 85, label: 'Stability', value: stability, color: '#22C55E' },
    { x: center.x, y: center.y + 115, label: 'Confidence', value: confidence, color: '#F97316' },
  ]

  const edges = [
    [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3],
  ]

  useEffect(() => {
    const avg = (drift + stability + coherence + confidence) / 4
    setAvgScore(avg)
  }, [drift, stability, coherence, confidence])

  useEffect(() => {
    const interval = setInterval(() => {
      setRotation((prev) => ({
        x: prev.x + 0.5,
        y: prev.y + 0.7,
      }))
    }, 30)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    const checkForChange = () => {
      const threshold = 0.02
      const hasSignificantChange =
        Math.abs(drift - prevValuesRef.current.drift) > threshold ||
        Math.abs(stability - prevValuesRef.current.stability) > threshold ||
        Math.abs(coherence - prevValuesRef.current.coherence) > threshold ||
        Math.abs(confidence - prevValuesRef.current.confidence) > threshold

      if (hasSignificantChange) {
        const newParticles: Particle[] = []
        const colors = ['#06B6D4', '#3B82F6', '#22C55E', '#F97316']
        const values = [coherence, drift, stability, confidence]

        values.forEach((value, idx) => {
          const particleCount = Math.ceil(value * 8)
          for (let i = 0; i < particleCount; i++) {
            const angle = (Math.random() * Math.PI * 2)
            const speed = 2 + Math.random() * 3
            newParticles.push({
              id: particleCountRef.current++,
              x: vertices[idx].x,
              y: vertices[idx].y,
              vx: Math.cos(angle) * speed,
              vy: Math.sin(angle) * speed,
              life: 1,
              maxLife: 1,
              color: colors[idx],
            })
          }
        })
        setParticles(newParticles)
        prevValuesRef.current = { drift, stability, coherence, confidence }
      }
    }

    checkForChange()
  }, [drift, stability, coherence, confidence])

  useEffect(() => {
    const interval = setInterval(() => {
      setParticles((prev) =>
        prev
          .map((p) => ({
            ...p,
            x: p.x + p.vx,
            y: p.y + p.vy,
            vy: p.vy + 0.15,
            life: p.life - 0.02,
          }))
          .filter((p) => p.life > 0)
      )
    }, 16)
    return () => clearInterval(interval)
  }, [])

  const getHealthStatus = () => {
    if (drift >= 0.78 || (1 - stability) >= 0.55) return 'Critical'
    if (drift >= 0.55 || (1 - stability) >= 0.35) return 'Warning'
    if (avgScore >= 0.75) return 'Excellent'
    if (avgScore >= 0.5) return 'Nominal'
    return 'Degraded'
  }

  const getHealthColor = () => {
    const status = getHealthStatus()
    if (status === 'Critical') return '#ef4444'
    if (status === 'Warning') return '#f59e0b'
    if (status === 'Excellent') return '#10b981'
    if (status === 'Nominal') return '#3b82f6'
    return '#64748b'
  }

  const metrics = [
    { label: 'Drift', value: drift, color: '#3B82F6', icon: '↗' },
    { label: 'Stability', value: stability, color: '#22C55E', icon: '⬆' },
    { label: 'Coherence', value: coherence, color: '#06B6D4', icon: '◆' },
    { label: 'Confidence', value: confidence, color: '#F97316', icon: '★' },
  ]

  return (
    <div className="panel tetra-panel-enhanced">
      <div className="panel-head">
        <div>
          <span className="eyebrow">Structural State (Tetrahedron)</span>
          <span className="panel-subtitle">4-dimensional system state projection</span>
        </div>
        <div className="health-badge" style={{ borderColor: getHealthColor(), backgroundColor: `${getHealthColor()}15` }}>
          <div className="health-indicator" style={{ backgroundColor: getHealthColor() }} />
          <div className="health-text">
            <div className="health-label">Health</div>
            <div className="health-status" style={{ color: getHealthColor() }}>{getHealthStatus()}</div>
          </div>
        </div>
      </div>

      <div className="tetra-container-enhanced">
        <div className="tetra-canvas-enhanced">
          <div className="tetra-3d-wrapper" style={{ perspective: '1200px' }}>
            <svg
              viewBox={`0 0 ${W} ${H}`}
              preserveAspectRatio="xMidYMid meet"
              style={{
                width: '100%',
                transform: `rotateX(${rotation.x * 0.3}deg) rotateY(${rotation.y * 0.3}deg)`,
                transformStyle: 'preserve-3d',
                transition: 'transform 0.05s linear',
              }}
            >
              <defs>
                <radialGradient id="tetraGradEnhanced" cx="50%" cy="40%" r="70%">
                  <stop offset="0%" stopColor="rgba(59, 130, 246, 0.15)" />
                  <stop offset="100%" stopColor="rgba(5, 7, 15, 0.8)" />
                </radialGradient>
                <filter id="tetraGlowEnhanced">
                  <feGaussianBlur stdDeviation="2.5" result="coloredBlur" />
                  <feMerge>
                    <feMergeNode in="coloredBlur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
                <filter id="shadowFilter">
                  <feDropShadow dx="0" dy="3" stdDeviation="3" floodOpacity="0.3" />
                </filter>
                <linearGradient id="energyGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#3B82F6" />
                  <stop offset="50%" stopColor="#06B6D4" />
                  <stop offset="100%" stopColor="#22C55E" />
                </linearGradient>
              </defs>

              <rect width={W} height={H} fill="url(#tetraGradEnhanced)" />

              {/* Ambient energy field */}
              <circle cx={center.x} cy={center.y} r={160} fill="url(#energyGrad)" opacity="0.02" />
              <circle cx={center.x} cy={center.y} r={130} fill="url(#energyGrad)" opacity="0.04" />
              <circle cx={center.x} cy={center.y} r={100} fill="url(#energyGrad)" opacity="0.06" />

              {/* Enhanced edges with gradients */}
              {edges.map((edge, idx) => {
                const v1 = vertices[edge[0]]
                const v2 = vertices[edge[1]]
                const intensity = (vertices[edge[0]].value + vertices[edge[1]].value) / 2
                return (
                  <g key={`edge-${idx}`}>
                    <line
                      x1={v1.x}
                      y1={v1.y}
                      x2={v2.x}
                      y2={v2.y}
                      stroke="rgba(96, 165, 250, 0.15)"
                      strokeWidth="2.5"
                      opacity={0.3 + intensity * 0.4}
                      filter="url(#shadowFilter)"
                    />
                    <line
                      x1={v1.x}
                      y1={v1.y}
                      x2={v2.x}
                      y2={v2.y}
                      stroke={`rgba(96, 165, 250, ${0.4 + intensity * 0.3})`}
                      strokeWidth="1"
                      opacity="0.8"
                      strokeDasharray="4 2"
                    />
                  </g>
                )
              })}

              {/* Vertices with enhanced visuals */}
              {vertices.map((vertex, idx) => {
                const displayRadius = 8 + vertex.value * 8
                const pulseScale = 1 + Math.sin(rotation.y * 0.05 + idx) * 0.15
                return (
                  <g key={`vertex-${idx}`}>
                    {/* Outer aura */}
                    <circle
                      cx={vertex.x}
                      cy={vertex.y}
                      r={displayRadius * 1.8 * pulseScale}
                      fill={vertex.color}
                      opacity={0.08 + vertex.value * 0.12}
                      filter="url(#tetraGlowEnhanced)"
                    />
                    {/* Middle ring */}
                    <circle
                      cx={vertex.x}
                      cy={vertex.y}
                      r={displayRadius * 1.2}
                      fill="none"
                      stroke={vertex.color}
                      strokeWidth="1"
                      opacity={0.2 + vertex.value * 0.3}
                      strokeDasharray="3 2"
                    />
                    {/* Core glow */}
                    <circle
                      cx={vertex.x}
                      cy={vertex.y}
                      r={displayRadius + 4}
                      fill={vertex.color}
                      opacity={0.2 + vertex.value * 0.25}
                    />
                    {/* Main node */}
                    <circle
                      cx={vertex.x}
                      cy={vertex.y}
                      r={displayRadius}
                      fill={vertex.color}
                      stroke="#FFFFFF"
                      strokeWidth="2.5"
                      opacity={0.75 + vertex.value * 0.25}
                      filter="url(#tetraGlowEnhanced)"
                    />
                    {/* Inner highlight */}
                    <circle
                      cx={vertex.x - 2}
                      cy={vertex.y - 2}
                      r={displayRadius * 0.4}
                      fill="#FFFFFF"
                      opacity="0.6"
                    />
                    {/* Label */}
                    <text
                      x={vertex.x}
                      y={vertex.y + 32}
                      fill="rgba(203, 213, 225, 0.95)"
                      fontSize="13"
                      fontWeight="700"
                      textAnchor="middle"
                      filter="url(#shadowFilter)"
                    >
                      {vertex.label}
                    </text>
                    {/* Value indicator */}
                    <text
                      x={vertex.x}
                      y={vertex.y + 48}
                      fill={vertex.color}
                      fontSize="16"
                      fontWeight="800"
                      textAnchor="middle"
                      filter="url(#shadowFilter)"
                    >
                      {(vertex.value * 100).toFixed(0)}%
                    </text>
                  </g>
                )
              })}

              {/* Particles */}
              {particles.map((p) => (
                <circle
                  key={p.id}
                  cx={p.x}
                  cy={p.y}
                  r={2 + (p.life * 1.5)}
                  fill={p.color}
                  opacity={p.life * 0.7}
                  filter="url(#tetraGlowEnhanced)"
                />
              ))}

              {/* Center nexus */}
              <circle cx={center.x} cy={center.y} r="4" fill="rgba(255,255,255,0.6)" filter="url(#tetraGlowEnhanced)" />
              <circle cx={center.x} cy={center.y} r="8" fill="none" stroke="rgba(255,255,255,0.2)" strokeWidth="1" />
            </svg>
          </div>
        </div>

        <div className="tetra-info-enhanced">
          <div className="metrics-grid">
            {metrics.map((metric, idx) => (
              <div key={metric.label} className="metric-card-enhanced" style={{ borderTopColor: metric.color }}>
                <div className="metric-icon" style={{ color: metric.color }}>
                  {metric.icon}
                </div>
                <div className="metric-label-text">{metric.label}</div>
                <div className="metric-value-large" style={{ color: metric.color }}>
                  {(metric.value * 100).toFixed(0)}%
                </div>
                <div className="metric-bar">
                  <div
                    className="metric-bar-fill"
                    style={{
                      width: `${metric.value * 100}%`,
                      backgroundColor: metric.color,
                      boxShadow: `0 0 12px ${metric.color}`,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="tetra-stats-summary">
            <div className="stat-item">
              <span className="stat-label">Average Score</span>
              <span className="stat-value" style={{ color: getHealthColor() }}>
                {(avgScore * 100).toFixed(1)}%
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">System Health</span>
              <span className="stat-value" style={{ color: getHealthColor() }}>
                {getHealthStatus()}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
