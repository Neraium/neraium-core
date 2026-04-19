'use client'

import { useEffect, useRef, useState } from 'react'

interface TetrahedronVisualizationProps {
  state: any
}

export function TetrahedronVisualization({ state }: TetrahedronVisualizationProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [scale, setScale] = useState(1)

  useEffect(() => {
    // Update scale on window resize
    const updateScale = () => {
      if (svgRef.current) {
        const rect = svgRef.current.getBoundingClientRect()
        // Scale tetrahedron to use most of the available space
        const idealSize = Math.min(rect.width, rect.height) * 0.8
        setScale(idealSize / 300)
      }
    }

    updateScale()
    window.addEventListener('resize', updateScale)
    return () => window.removeEventListener('resize', updateScale)
  }, [])

  useEffect(() => {
    if (!svgRef.current || !state) return

    const drift = state.drift || 0
    const coherence = state.coherence || 0.75
    const subsystems = state.subsystems || {}

    // Calculate deformation based on drift
    const totalDeformation = Math.min(drift * 0.3, 0.3)

    // Core physics
    const baseCoreRadius = 20
    const coreRadius = Math.max(baseCoreRadius * 0.4, baseCoreRadius * (1 - totalDeformation / 0.3))
    const coreGlow = coreRadius * 2

    // Breathing animation speed (coherence drives it)
    const breathCycleDuration = coherence > 0.6 ? 10 : coherence > 0.4 ? 5 : 2
    const jitterIntensity = Math.max(0, (0.6 - coherence) * 2) // 0 at stable, max at critical

    // Base tetrahedron vertices (unit tetrahedron)
    const baseVertices = [
      { name: 'Climate', x: 0, y: -1, z: -0.3, color: '#60a5fa' },
      { name: 'Airflow', x: 0.866, y: 0.5, z: -0.3, color: '#f97316' },
      { name: 'Irrigation', x: -0.866, y: 0.5, z: -0.3, color: '#eab308' },
      { name: 'Plant', x: 0, y: 0, z: 1.2, color: '#34d399' },
    ]

    // Get subsystem drifts
    const subsystemDrifts = [
      subsystems.climate?.drift || 0,
      subsystems.airflow?.drift || 0,
      subsystems.irrigation?.drift || 0,
      subsystems.plant?.drift || 0,
    ]

    // Apply deformation based on subsystem influence
    // Each vertex displaces based on its own subsystem's drift
    const deformedVertices = baseVertices.map((v, i) => {
      const contribution = subsystemDrifts[i] || 0
      const deformationFactor = contribution * totalDeformation

      return {
        ...v,
        x: v.x * (1 - deformationFactor * 0.15),
        y: v.y * (1 - deformationFactor * 0.15),
        z: v.z * (1 - deformationFactor * 0.1),
      }
    })

    // Edges
    const edges = [
      [0, 1],
      [0, 2],
      [0, 3],
      [1, 2],
      [1, 3],
      [2, 3],
    ]

    // Perspective projection
    const project = (v: any, offsetX = 0, offsetY = 0) => {
      const perspective = 5
      const scale = perspective / (perspective + v.z)
      return {
        x: 500 + (v.x * scale * 200 + offsetX) * (scale * 0.8),
        y: 350 + (v.y * scale * 200 + offsetY) * (scale * 0.8),
        z: v.z,
        scale,
      }
    }

    const projectedVertices = deformedVertices.map((v) => project(v))

    // Color based on system state
    let edgeColor = '#60a5fa'
    let coreGlowColor = '#60a5fa'
    let edgeOpacity = 0.5 + coherence * 0.3

    if (drift > 0.6) {
      edgeColor = '#ef4444'
      coreGlowColor = '#ff6b6b'
      edgeOpacity = 0.7 + drift * 0.3
    } else if (drift > 0.4) {
      edgeColor = '#f97316'
      coreGlowColor = '#fb923c'
      edgeOpacity = 0.6 + drift * 0.2
    } else if (drift > 0.2) {
      edgeColor = '#eab308'
      coreGlowColor = '#facc15'
      edgeOpacity = 0.55 + drift * 0.15
    }

    const now = Date.now()
    const breathPulse =
      0.5 + 0.5 * Math.sin((now / 1000) * (Math.PI * 2) / breathCycleDuration)

    // Build SVG
    const svg = svgRef.current
    svg.innerHTML = `
      <defs>
        <filter id="core-glow" x="-100%" y="-100%" width="300%" height="300%">
          <feGaussianBlur stdDeviation="12" result="coloredBlur"/>
          <feMerge>
            <feMergeNode in="coloredBlur"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>
        <filter id="edge-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
          <feMerge>
            <feMergeNode in="coloredBlur"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>
        <style>
          @keyframes drift-motion {
            0%, 100% { transform: translate(0, 0); }
            25% { transform: translate(3px, 2px); }
            50% { transform: translate(0, 3px); }
            75% { transform: translate(-3px, 2px); }
          }
          @keyframes jitter {
            0%, 100% { transform: translate(0, 0); }
            10% { transform: translate(2px, 1px); }
            20% { transform: translate(-1px, 2px); }
            30% { transform: translate(1px, -1px); }
            40% { transform: translate(-2px, 0); }
            50% { transform: translate(0, 2px); }
            60% { transform: translate(2px, -1px); }
            70% { transform: translate(-1px, 1px); }
            80% { transform: translate(1px, -2px); }
            90% { transform: translate(-2px, 1px); }
          }
          .tetrahedron-group {
            animation: drift-motion ${10 - drift * 6}s ease-in-out infinite;
            ${drift > 0.5 ? `animation: drift-motion ${10 - drift * 6}s ease-in-out infinite, jitter 0.4s ease-in-out infinite;` : ''}
          }
        </style>
      </defs>

      <!-- Ambient glow background -->
      <circle cx="500" cy="350" r="280" fill="none" stroke="${coreGlowColor}" opacity="0.08" stroke-width="1"/>
      <circle cx="500" cy="350" r="200" fill="none" stroke="${coreGlowColor}" opacity="0.05" stroke-width="1"/>

      <!-- Main tetrahedron group with motion -->
      <g class="tetrahedron-group">
        <!-- Edges with tension glow -->
        <g stroke="${edgeColor}" stroke-width="3" fill="none" opacity="${edgeOpacity}" filter="url(#edge-glow)">
          ${edges
            .map(([a, b]) => {
              const va = projectedVertices[a]
              const vb = projectedVertices[b]
              return `<line x1="${va.x}" y1="${va.y}" x2="${vb.x}" y2="${vb.y}"/>`
            })
            .join('')}
        </g>

        <!-- Vertices (subsystem indicators) -->
        <g>
          ${projectedVertices
            .map((v, idx) => {
              const subsystem = baseVertices[idx]
              const drift_val = subsystemDrifts[idx]
              const vertexSize = 8 + drift_val * 6
              const vertexColor = subsystem.color
              return `
                <circle
                  cx="${v.x}"
                  cy="${v.y}"
                  r="${vertexSize}"
                  fill="${vertexColor}"
                  opacity="${0.6 + drift_val * 0.4}"
                  filter="url(#edge-glow)"
                />
              `
            })
            .join('')}
        </g>

        <!-- Coherence core -->
        <g opacity="${Math.max(0.3, 1 - drift * 0.5)}">
          <!-- Outer glow rings -->
          <circle
            cx="500"
            cy="350"
            r="${coreGlow * breathPulse}"
            fill="none"
            stroke="${coreGlowColor}"
            opacity="${0.15 * breathPulse}"
            stroke-width="2"
          />
          <circle
            cx="500"
            cy="350"
            r="${coreGlow * 0.6 * breathPulse}"
            fill="none"
            stroke="${coreGlowColor}"
            opacity="${0.25 * breathPulse}"
            stroke-width="2"
          />

          <!-- Core sphere -->
          <circle
            cx="500"
            cy="350"
            r="${coreRadius * breathPulse}"
            fill="${coreGlowColor}"
            opacity="0.4"
            filter="url(#core-glow)"
          />
          <circle
            cx="500"
            cy="350"
            r="${coreRadius * breathPulse * 0.6}"
            fill="${coreGlowColor}"
            opacity="0.7"
          />
        </g>
      </g>

      <!-- System state indicator text (minimal, positioned at edges) -->
      <text
        x="500"
        y="30"
        text-anchor="middle"
        font-size="14"
        fill="${edgeColor}"
        opacity="0.6"
        font-weight="300"
      >
        SYSTEM FIELD
      </text>

      <!-- Stats (minimal, positioned at bottom) -->
      <text x="50" y="670" font-size="12" fill="white" opacity="0.4">
        Coherence: ${(coherence * 100).toFixed(0)}%
      </text>
      <text x="950" y="670" font-size="12" fill="white" opacity="0.4" text-anchor="end">
        Drift: ${(drift * 100).toFixed(0)}%
      </text>
    `
  }, [state])

  return (
    <svg
      ref={svgRef}
      viewBox="0 0 1000 700"
      className="w-full h-full"
      style={{
        filter: `drop-shadow(0 0 40px ${
          state.drift > 0.6
            ? 'rgba(239, 68, 68, 0.3)'
            : state.drift > 0.4
              ? 'rgba(249, 115, 22, 0.3)'
              : state.drift > 0.2
                ? 'rgba(234, 179, 8, 0.3)'
                : 'rgba(96, 165, 250, 0.2)'
        })`,
      }}
    />
  )
}
